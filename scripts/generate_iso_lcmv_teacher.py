"""Generate an isolation-gated LCMV/ZF teacher for the URA16 HFSS dataset.

The previous quick-model teachers primarily optimized PSLL and only reported
isolation after the fact. This script flips that priority:

1. keep each sample's active element count fixed;
2. build candidate sparse masks from the original mask, an optional PSLL teacher,
   deterministic aperture-spread masks, and random local swaps;
3. project per-task weights with a zero-forcing/LCMV constraint at the target
   directions and, for K=6 or large-scan samples, broaden local nulls around
   interfering targets;
4. reject candidates below the requested point/local task isolation;
5. among the gated candidates, select the mask with the best PSLL.

The output directory contains a dataset_arrays.npz-compatible payload, so it can
be passed directly to run_pagan_lite_mvp.py via --teacher-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_BASE_TEACHER_DIR = DEFAULT_DATASET_DIR / "optimized_teachers" / "greedy_psll_v2_canonical"
KMAX = 6
NUM_ELEMENTS = 256
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--base-teacher-dir", type=Path, default=DEFAULT_BASE_TEACHER_DIR)
    parser.add_argument("--run-name", default="iso_lcmv_zf_psll_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k-values", default="1,2,4,6")
    parser.add_argument("--active-ratios", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument(
        "--active-ratio-override",
        type=float,
        default=0.0,
        help=(
            "If >0, optimize selected samples with this active ratio instead of "
            "their dataset ratio, and write matching num_active/ratio metadata "
            "to the output teacher. This is useful for same-target ratio "
            "escalation during HFSS-in-loop search."
        ),
    )
    parser.add_argument(
        "--control-ratios",
        default="1.0",
        help="Ratios left untouched as control groups unless --optimize-control-ratios is set.",
    )
    parser.add_argument("--optimize-control-ratios", action="store_true")
    parser.add_argument(
        "--sample-indices-file",
        type=Path,
        default=None,
        help="Optional newline/CSV list of sample indices to optimize; overrides split/k/ratio selection.",
    )
    parser.add_argument("--samples-per-cell", type=int, default=0, help="0 means all selected samples.")
    parser.add_argument("--target-isolation-db", type=float, default=25.0)
    parser.add_argument("--target-local-isolation-db", type=float, default=25.0)
    parser.add_argument("--mainlobe-drop-limit-db", type=float, default=0.5)
    parser.add_argument("--selection-theta-step", type=float, default=4.0)
    parser.add_argument("--selection-phi-step", type=float, default=8.0)
    parser.add_argument("--eval-theta-step", type=float, default=2.0)
    parser.add_argument("--eval-phi-step", type=float, default=5.0)
    parser.add_argument("--random-candidates", type=int, default=12)
    parser.add_argument("--local-swap-candidates", type=int, default=12)
    parser.add_argument("--local-swap-rounds", type=int, default=4)
    parser.add_argument("--max-random-swaps", type=int, default=6)
    parser.add_argument(
        "--structured-mask-mode",
        choices=("off", "basic", "advanced"),
        default="basic",
        help=(
            "Structured mask seed family. 'basic' preserves the previous "
            "deterministic/checker/edge seeds; 'advanced' adds quadrant/ring "
            "balanced, mirrored, and spread-greedy masks for K-large scenes."
        ),
    )
    parser.add_argument("--diagonal-loading", type=float, default=0.0)
    parser.add_argument("--condition-limit", type=float, default=1.0e10)
    parser.add_argument("--local-null-broadening", choices=["off", "auto", "all"], default="auto")
    parser.add_argument("--local-null-trigger-k", type=int, default=6)
    parser.add_argument("--local-null-trigger-theta-deg", type=float, default=50.0)
    parser.add_argument("--local-null-offsets-deg", default="2,5")
    parser.add_argument(
        "--local-null-diagonal-loading",
        type=float,
        default=1.0e-2,
        help="Regularization used only when local-null broadening is active.",
    )
    parser.add_argument(
        "--psll-refine-mode",
        choices=("off", "projected"),
        default="off",
        help=(
            "Secondary PSLL optimization after isolation-gated LCMV/ZF. "
            "'projected' performs top-sidelobe descent in the null space of "
            "the mainlobe/null constraints, then projects weights back."
        ),
    )
    parser.add_argument("--psll-refine-steps", type=int, default=6)
    parser.add_argument("--psll-refine-topk", type=int, default=16)
    parser.add_argument("--psll-refine-step-size", type=float, default=0.12)
    parser.add_argument("--psll-refine-min-improvement-db", type=float, default=0.03)
    parser.add_argument(
        "--psll-refine-max-weak-peak-drop-db",
        type=float,
        default=0.25,
        help="Reject a refine step if the weakest target peak drops more than this from the pre-refine candidate.",
    )
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def ratio_to_active_count(ratio: float) -> int:
    if not (0.0 < float(ratio) <= 1.0):
        raise ValueError(f"active ratio override must be in (0, 1], got {ratio}")
    return int(np.clip(round(float(ratio) * NUM_ELEMENTS), 1, NUM_ELEMENTS))


def load_sample_indices_file(path: Path) -> np.ndarray:
    indices: list[int] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            for part in text.replace(",", " ").split():
                try:
                    indices.append(int(part))
                except ValueError:
                    continue
    if not indices:
        raise ValueError(f"No sample indices found in {path}")
    # Preserve file order but remove duplicates.
    seen: set[int] = set()
    unique: list[int] = []
    for index in indices:
        if index not in seen:
            seen.add(index)
            unique.append(index)
    return np.asarray(unique, dtype=np.int64)


def unit_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    ).astype(np.float32)


def valid_targets_deg_for_sample(targets_deg: np.ndarray, task_valid: np.ndarray) -> np.ndarray:
    return np.nan_to_num(targets_deg[task_valid.astype(bool)], nan=0.0).astype(np.float32)


def make_grid(theta_step: float, phi_step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_values = np.arange(0.0, 90.0 + 0.1, theta_step, dtype=np.float32)
    phi_values = np.arange(0.0, 360.0, phi_step, dtype=np.float32)
    theta, phi = np.meshgrid(theta_values, phi_values, indexing="ij")
    theta_flat = theta.reshape(-1)
    phi_flat = phi.reshape(-1)
    return theta_flat, phi_flat, unit_vectors(theta_flat, phi_flat)


def target_dirs_for_sample(targets_deg: np.ndarray, task_valid: np.ndarray) -> np.ndarray:
    valid_targets = valid_targets_deg_for_sample(targets_deg, task_valid)
    return unit_vectors(valid_targets[:, 0], valid_targets[:, 1])


def local_null_broadening_enabled(
    *,
    mode: str,
    k: int,
    valid_targets_deg: np.ndarray,
    trigger_k: int,
    trigger_theta_deg: float,
) -> bool:
    if k <= 1 or mode == "off":
        return False
    if mode == "all":
        return True
    if mode != "auto":
        return False
    max_theta = float(np.nanmax(valid_targets_deg[:, 0])) if valid_targets_deg.size else float("-inf")
    return int(k) >= int(trigger_k) or max_theta >= float(trigger_theta_deg)


def make_local_null_dirs_by_target(valid_targets_deg: np.ndarray, offsets_deg: list[float]) -> list[np.ndarray]:
    """Build theta/phi local null samples around each target, excluding the target itself."""
    out: list[np.ndarray] = []
    for theta, phi in valid_targets_deg.astype(np.float64):
        points: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for offset in offsets_deg:
            delta = float(abs(offset))
            if delta <= 0.0:
                continue
            for dtheta in (-delta, 0.0, delta):
                for dphi in (-delta, 0.0, delta):
                    if dtheta == 0.0 and dphi == 0.0:
                        continue
                    theta_i = min(90.0, max(0.0, theta + dtheta))
                    phi_i = (phi + dphi) % 360.0
                    key = (round(theta_i, 6), round(phi_i, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    points.append((theta_i, phi_i))
        if points:
            point_arr = np.asarray(points, dtype=np.float32)
            out.append(unit_vectors(point_arr[:, 0], point_arr[:, 1]))
        else:
            out.append(np.zeros((0, 3), dtype=np.float32))
    return out


def side_mask_for_targets(grid_dirs: np.ndarray, target_dirs: np.ndarray, radius_deg: float = 8.0) -> np.ndarray:
    dots = np.clip(grid_dirs @ target_dirs.T, -1.0, 1.0)
    dists = np.rad2deg(np.arccos(dots))
    return dists.min(axis=1) > radius_deg


def steering_rx(positions: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    phase = 2.0 * np.pi * (dirs @ positions.T)
    return np.exp(-1j * phase).astype(np.complex64)


def deterministic_element_order(element_ixiy: np.ndarray) -> np.ndarray:
    coords = element_ixiy.astype(np.float64)
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    radius = np.linalg.norm(coords - center[None, :], axis=1)
    ix = coords[:, 0].astype(np.int64)
    iy = coords[:, 1].astype(np.int64)
    morton = np.zeros(coords.shape[0], dtype=np.int64)
    for bit in range(4):
        morton |= ((ix >> bit) & 1) << (2 * bit)
        morton |= ((iy >> bit) & 1) << (2 * bit + 1)
    score = -0.65 * radius + 0.35 * (morton / max(float(morton.max()), 1.0))
    return np.argsort(score, kind="stable")


def mask_from_scores(scores: np.ndarray, num_active: int) -> np.ndarray:
    mask = np.zeros(scores.shape[0], dtype=np.bool_)
    if num_active >= scores.shape[0]:
        mask[:] = True
        return mask
    chosen = np.argpartition(scores, -num_active)[-num_active:]
    mask[chosen] = True
    return mask


def normalize_mask(mask: np.ndarray, num_active: int, rng: np.random.Generator) -> np.ndarray:
    out = mask.astype(bool).copy()
    active = np.flatnonzero(out)
    if active.size > num_active:
        drop = rng.choice(active, size=active.size - num_active, replace=False)
        out[drop] = False
    elif active.size < num_active:
        inactive = np.flatnonzero(~out)
        add = rng.choice(inactive, size=num_active - active.size, replace=False)
        out[add] = True
    return out


def mutate_mask(mask: np.ndarray, rng: np.random.Generator, max_swaps: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    active = np.flatnonzero(out)
    inactive = np.flatnonzero(~out)
    if active.size == 0 or inactive.size == 0:
        return out
    swaps = int(rng.integers(1, max(2, min(max_swaps, active.size, inactive.size) + 1)))
    off = rng.choice(active, size=swaps, replace=False)
    on = rng.choice(inactive, size=swaps, replace=False)
    out[off] = False
    out[on] = True
    return out


def fill_mask_from_scores(mask: np.ndarray, scores: np.ndarray, num_active: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    active = np.flatnonzero(out)
    if active.size > num_active:
        drop_order = active[np.argsort(scores[active], kind="stable")]
        out[drop_order[: active.size - num_active]] = False
    elif active.size < num_active:
        inactive = np.flatnonzero(~out)
        add_order = inactive[np.argsort(scores[inactive], kind="stable")[::-1]]
        out[add_order[: num_active - active.size]] = True
    return out


def ring_quadrant_balanced_mask(
    *,
    element_ixiy: np.ndarray,
    scores: np.ndarray,
    num_active: int,
    ring_count: int = 4,
) -> np.ndarray:
    coords = element_ixiy.astype(np.float64)
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    shifted = coords - center[None, :]
    radius = np.linalg.norm(shifted, axis=1)
    max_radius = max(float(radius.max()), 1.0)
    rings = np.minimum((radius / max_radius * ring_count).astype(np.int64), ring_count - 1)
    quadrants = (shifted[:, 0] >= 0.0).astype(np.int64) * 2 + (shifted[:, 1] >= 0.0).astype(np.int64)
    cells = rings * 4 + quadrants

    mask = np.zeros(NUM_ELEMENTS, dtype=np.bool_)
    unique_cells, counts = np.unique(cells, return_counts=True)
    ideal = counts.astype(np.float64) / float(NUM_ELEMENTS) * float(num_active)
    quotas = np.floor(ideal).astype(np.int64)
    remainder = int(num_active - int(quotas.sum()))
    if remainder > 0:
        order = np.argsort(ideal - quotas, kind="stable")[::-1]
        quotas[order[:remainder]] += 1
    for cell, quota in zip(unique_cells, quotas):
        if quota <= 0:
            continue
        members = np.flatnonzero(cells == cell)
        chosen = members[np.argsort(scores[members], kind="stable")[::-1][: int(quota)]]
        mask[chosen] = True
    return fill_mask_from_scores(mask, scores, num_active)


def mirrored_pair_mask(element_ixiy: np.ndarray, scores: np.ndarray, num_active: int) -> np.ndarray:
    coords = element_ixiy.astype(np.int64)
    max_ix = int(coords[:, 0].max())
    max_iy = int(coords[:, 1].max())
    coord_to_index = {(int(ix), int(iy)): idx for idx, (ix, iy) in enumerate(coords)}
    pairs: list[tuple[float, tuple[int, ...]]] = []
    seen: set[int] = set()
    for idx, (ix, iy) in enumerate(coords):
        if idx in seen:
            continue
        mirror = coord_to_index.get((max_ix - int(ix), max_iy - int(iy)), idx)
        members = tuple(sorted({idx, int(mirror)}))
        seen.update(members)
        pair_score = float(np.mean(scores[list(members)]))
        pairs.append((pair_score, members))
    pairs.sort(key=lambda item: item[0], reverse=True)
    mask = np.zeros(NUM_ELEMENTS, dtype=np.bool_)
    for _, members in pairs:
        if int(mask.sum()) + len(members) > num_active:
            continue
        mask[list(members)] = True
        if int(mask.sum()) >= num_active:
            break
    return fill_mask_from_scores(mask, scores, num_active)


def spread_greedy_mask(
    *,
    element_ixiy: np.ndarray,
    scores: np.ndarray,
    num_active: int,
    spacing_weight: float,
) -> np.ndarray:
    coords = element_ixiy.astype(np.float64)
    score = scores.astype(np.float64).copy()
    score = (score - float(score.mean())) / max(float(score.std()), 1.0e-6)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    dist = dist / max(float(dist.max()), 1.0)
    chosen: list[int] = []
    available = np.ones(NUM_ELEMENTS, dtype=np.bool_)
    min_dist = np.ones(NUM_ELEMENTS, dtype=np.float64)
    for _ in range(num_active):
        composite = score + float(spacing_weight) * min_dist
        composite[~available] = -np.inf
        idx = int(np.argmax(composite))
        if not np.isfinite(composite[idx]):
            break
        chosen.append(idx)
        available[idx] = False
        min_dist = np.minimum(min_dist, dist[idx])
    mask = np.zeros(NUM_ELEMENTS, dtype=np.bool_)
    mask[np.asarray(chosen, dtype=np.int64)] = True
    return fill_mask_from_scores(mask, scores, num_active)


def target_phase_scores(
    *,
    positions: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_targets = valid_targets_deg_for_sample(targets_deg, task_valid)
    if valid_targets.shape[0] <= 1:
        flat = np.zeros(NUM_ELEMENTS, dtype=np.float64)
        return flat, flat
    dirs = unit_vectors(valid_targets[:, 0], valid_targets[:, 1]).astype(np.float64)
    phase = 2.0 * np.pi * (dirs @ positions.astype(np.float64).T)
    phasors = np.exp(1j * phase)
    coherence = np.abs(np.mean(phasors, axis=0)).astype(np.float64)
    decorrelated = 1.0 - coherence
    return coherence, decorrelated


def unique_masks(masks: Iterable[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in masks:
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(mask.astype(bool).copy())
    return out


def seed_masks_for_sample(
    *,
    arrays: dict[str, np.ndarray],
    base_teacher: dict[str, np.ndarray] | None,
    sample_index: int,
    num_active: int,
    rng: np.random.Generator,
    structured_mode: str,
) -> list[np.ndarray]:
    element_ixiy = arrays["element_ixiy"]
    coords = element_ixiy.astype(np.float64)
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    radius = np.linalg.norm(coords - center[None, :], axis=1)
    ix = coords[:, 0]
    iy = coords[:, 1]
    checker = ((ix + iy) % 2.0) - 0.01 * radius
    anti_checker = (1.0 - ((ix + iy) % 2.0)) - 0.01 * radius
    diagonal = -np.abs(ix - iy) - 0.02 * radius
    edge = radius
    center_score = -radius
    order = deterministic_element_order(element_ixiy)
    deterministic = np.zeros(NUM_ELEMENTS, dtype=np.bool_)
    deterministic[order[:num_active]] = True
    masks = [arrays["masks"][sample_index].astype(bool)]
    if base_teacher is not None:
        masks.insert(1, base_teacher["masks"][sample_index].astype(bool))
    if structured_mode != "off":
        masks.extend(
            [
                deterministic,
                mask_from_scores(checker, num_active),
                mask_from_scores(anti_checker, num_active),
                mask_from_scores(diagonal, num_active),
                mask_from_scores(edge, num_active),
                mask_from_scores(center_score, num_active),
            ]
        )
    if structured_mode == "advanced":
        coherence, decorrelated = target_phase_scores(
            positions=arrays["positions_lambda"].astype(np.float32),
            targets_deg=arrays["targets_deg"][sample_index],
            task_valid=arrays["task_valid"][sample_index].astype(bool),
        )
        ring_mid = -np.abs(radius - np.percentile(radius, 62.5))
        spread_score = 0.65 * decorrelated + 0.35 * (radius / max(float(radius.max()), 1.0))
        masks.extend(
            [
                ring_quadrant_balanced_mask(element_ixiy=element_ixiy, scores=edge, num_active=num_active),
                ring_quadrant_balanced_mask(element_ixiy=element_ixiy, scores=ring_mid, num_active=num_active),
                ring_quadrant_balanced_mask(element_ixiy=element_ixiy, scores=decorrelated, num_active=num_active),
                mirrored_pair_mask(element_ixiy, spread_score, num_active),
                mirrored_pair_mask(element_ixiy, coherence - 0.08 * radius, num_active),
                spread_greedy_mask(
                    element_ixiy=element_ixiy,
                    scores=spread_score,
                    num_active=num_active,
                    spacing_weight=0.70,
                ),
                spread_greedy_mask(
                    element_ixiy=element_ixiy,
                    scores=decorrelated - 0.03 * radius,
                    num_active=num_active,
                    spacing_weight=0.45,
                ),
            ]
        )
    masks = [normalize_mask(mask, num_active, rng) for mask in masks]
    return unique_masks(masks)


def lcmv_zf_weights(
    *,
    mask: np.ndarray,
    positions: np.ndarray,
    target_dirs: np.ndarray,
    valid_indices: np.ndarray,
    diagonal_loading: float,
    local_null_dirs_by_target: list[np.ndarray] | None = None,
    local_null_diagonal_loading: float | None = None,
) -> tuple[np.ndarray, float, bool, int]:
    active = np.flatnonzero(mask)
    weights = np.zeros((positions.shape[0], KMAX), dtype=np.complex64)
    if valid_indices.size == 0:
        return weights, float("nan"), False, 0
    has_local_nulls = bool(local_null_dirs_by_target) and any(dirs.size for dirs in local_null_dirs_by_target)
    if not has_local_nulls:
        s = steering_rx(positions[active], target_dirs)  # K, M
        gram = s @ s.conj().T
        if diagonal_loading > 0.0:
            gram = gram + float(diagonal_loading) * np.eye(gram.shape[0], dtype=np.complex64)
        try:
            cond = float(np.linalg.cond(gram))
        except np.linalg.LinAlgError:
            cond = float("inf")
        try:
            inv = np.linalg.pinv(gram, rcond=1.0e-8)
            w_active = s.conj().T @ inv  # M, K; S @ W = I
        except np.linalg.LinAlgError:
            return weights, cond, False, max(0, int(valid_indices.size) - 1)
        weights[active[:, None], valid_indices[None, :]] = w_active.astype(np.complex64)
        return weights, cond, bool(np.all(np.isfinite(w_active))), max(0, int(valid_indices.size) - 1)

    cond_max = float("-inf")
    constraint_count_max = 0
    loading = float(diagonal_loading if local_null_diagonal_loading is None else local_null_diagonal_loading)
    for task_pos, valid_index in enumerate(valid_indices):
        dirs: list[np.ndarray] = [target_dirs[task_pos]]
        rhs: list[complex] = [1.0 + 0.0j]
        for other_pos in range(target_dirs.shape[0]):
            if other_pos == task_pos:
                continue
            dirs.append(target_dirs[other_pos])
            rhs.append(0.0 + 0.0j)
            if local_null_dirs_by_target is not None:
                for local_dir in local_null_dirs_by_target[other_pos]:
                    dirs.append(local_dir)
                    rhs.append(0.0 + 0.0j)
        constraint_dirs = np.asarray(dirs, dtype=np.float32)
        rhs_vec = np.asarray(rhs, dtype=np.complex64)
        constraint_count_max = max(constraint_count_max, max(0, constraint_dirs.shape[0] - 1))
        s = steering_rx(positions[active], constraint_dirs)
        gram = s @ s.conj().T
        if loading > 0.0:
            gram = gram + loading * np.eye(gram.shape[0], dtype=np.complex64)
        try:
            cond = float(np.linalg.cond(gram))
        except np.linalg.LinAlgError:
            cond = float("inf")
        cond_max = max(cond_max, cond)
        try:
            inv = np.linalg.pinv(gram, rcond=1.0e-8)
            w_active = s.conj().T @ inv @ rhs_vec
        except np.linalg.LinAlgError:
            return weights, cond_max, False, constraint_count_max
        if not np.all(np.isfinite(w_active)):
            return weights, cond_max, False, constraint_count_max
        weights[active, int(valid_index)] = w_active.astype(np.complex64)
    return weights, cond_max, True, constraint_count_max


def task_constraint_system(
    *,
    target_dirs: np.ndarray,
    task_pos: int,
    local_null_dirs_by_target: list[np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    dirs: list[np.ndarray] = []
    rhs: list[complex] = []
    for other_pos in range(target_dirs.shape[0]):
        dirs.append(target_dirs[other_pos])
        rhs.append(1.0 + 0.0j if other_pos == task_pos else 0.0 + 0.0j)
        if other_pos != task_pos and local_null_dirs_by_target is not None:
            for local_dir in local_null_dirs_by_target[other_pos]:
                dirs.append(local_dir)
                rhs.append(0.0 + 0.0j)
    return np.asarray(dirs, dtype=np.float32), np.asarray(rhs, dtype=np.complex64)


def project_active_weights_to_constraints(
    *,
    w_active: np.ndarray,
    positions_active: np.ndarray,
    constraint_dirs: np.ndarray,
    rhs: np.ndarray,
    diagonal_loading: float,
) -> np.ndarray:
    s = steering_rx(positions_active, constraint_dirs)
    residual = rhs - s @ w_active
    gram = s @ s.conj().T
    loading = float(diagonal_loading)
    if loading > 0.0:
        gram = gram + loading * np.eye(gram.shape[0], dtype=np.complex64)
    correction = s.conj().T @ np.linalg.pinv(gram, rcond=1.0e-8) @ residual
    return (w_active + correction).astype(np.complex64)


def constraint_nullspace_projector(
    *,
    positions_active: np.ndarray,
    constraint_dirs: np.ndarray,
    diagonal_loading: float,
) -> np.ndarray:
    s = steering_rx(positions_active, constraint_dirs)
    gram = s @ s.conj().T
    loading = float(diagonal_loading)
    if loading > 0.0:
        gram = gram + loading * np.eye(gram.shape[0], dtype=np.complex64)
    return (
        np.eye(s.shape[1], dtype=np.complex64)
        - s.conj().T @ np.linalg.pinv(gram, rcond=1.0e-8) @ s
    ).astype(np.complex64)


def refine_psll_projected(
    *,
    weights: np.ndarray,
    mask: np.ndarray,
    positions: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    target_dirs: np.ndarray,
    valid_indices: np.ndarray,
    grid_dirs: np.ndarray,
    grid_steer: np.ndarray,
    local_null_dirs_by_target: list[np.ndarray] | None,
    cond: float,
    k_value: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], bool, dict[str, float]]:
    initial_metrics = evaluate_weights(
        weights=weights,
        targets_deg=targets_deg,
        task_valid=task_valid,
        positions=positions,
        grid_dirs=grid_dirs,
        grid_steer=grid_steer,
        local_null_dirs_by_target=local_null_dirs_by_target,
    )
    initial_gate = candidate_passes_gate(initial_metrics, cond, args, k_value)
    if (
        str(args.psll_refine_mode) == "off"
        or int(args.psll_refine_steps) <= 0
        or valid_indices.size == 0
        or not initial_gate
    ):
        return weights, initial_metrics, initial_gate, {
            "psll_refine_steps_used": 0,
            "psll_refine_delta_db": 0.0,
        }

    active = np.flatnonzero(mask)
    if active.size == 0:
        return weights, initial_metrics, initial_gate, {
            "psll_refine_steps_used": 0,
            "psll_refine_delta_db": 0.0,
        }
    side_mask = side_mask_for_targets(grid_dirs, target_dirs)
    side_indices = np.flatnonzero(side_mask)
    if side_indices.size == 0:
        return weights, initial_metrics, initial_gate, {
            "psll_refine_steps_used": 0,
            "psll_refine_delta_db": 0.0,
        }

    constraint_payload: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    positions_active = positions[active]
    loading = float(args.local_null_diagonal_loading if local_null_dirs_by_target else args.diagonal_loading)
    for task_pos, valid_index in enumerate(valid_indices):
        constraint_dirs, rhs = task_constraint_system(
            target_dirs=target_dirs,
            task_pos=task_pos,
            local_null_dirs_by_target=local_null_dirs_by_target,
        )
        try:
            projector = constraint_nullspace_projector(
                positions_active=positions_active,
                constraint_dirs=constraint_dirs,
                diagonal_loading=loading,
            )
        except np.linalg.LinAlgError:
            return weights, initial_metrics, initial_gate, {
                "psll_refine_steps_used": 0,
                "psll_refine_delta_db": 0.0,
            }
        constraint_payload.append((int(valid_index), constraint_dirs, rhs, projector))

    current = weights.astype(np.complex64).copy()
    best_weights = current.copy()
    best_metrics = dict(initial_metrics)
    best_gate = initial_gate
    accepted_steps = 0
    weak_peak_floor = float(initial_metrics["weak_peak_db"]) - float(args.psll_refine_max_weak_peak_drop_db)
    min_improvement = float(args.psll_refine_min_improvement_db)

    for step_index in range(int(args.psll_refine_steps)):
        combined = current.sum(axis=1)
        side_resp = grid_steer[side_indices] @ combined
        side_power = np.abs(side_resp) ** 2
        if side_power.size == 0 or not np.all(np.isfinite(side_power)):
            break
        top_count = min(max(1, int(args.psll_refine_topk)), side_power.size)
        local_top = np.argpartition(side_power, -top_count)[-top_count:]
        top_indices = side_indices[local_top]
        top_resp = side_resp[local_top]
        gradient_active = (grid_steer[top_indices][:, active].conj().T @ top_resp) / float(top_count)
        grad_norm = float(np.linalg.norm(gradient_active))
        if not np.isfinite(grad_norm) or grad_norm <= 0.0:
            break
        trial = current.copy()
        step_size = float(args.psll_refine_step_size) / math.sqrt(float(step_index + 1))
        task_scale = 1.0 / math.sqrt(float(max(valid_indices.size, 1)))
        for valid_index, constraint_dirs, rhs, projector in constraint_payload:
            w_active = trial[active, valid_index]
            projected_grad = projector @ gradient_active
            proj_norm = float(np.linalg.norm(projected_grad))
            if not np.isfinite(proj_norm) or proj_norm <= 0.0:
                continue
            w_norm = max(float(np.linalg.norm(w_active)), 1.0)
            delta = step_size * task_scale * w_norm * projected_grad / proj_norm
            w_next = w_active - delta.astype(np.complex64)
            try:
                w_next = project_active_weights_to_constraints(
                    w_active=w_next,
                    positions_active=positions_active,
                    constraint_dirs=constraint_dirs,
                    rhs=rhs,
                    diagonal_loading=loading,
                )
            except np.linalg.LinAlgError:
                continue
            trial[active, valid_index] = w_next.astype(np.complex64)
        trial[~mask.astype(bool), :] = 0.0 + 0.0j
        metrics = evaluate_weights(
            weights=trial,
            targets_deg=targets_deg,
            task_valid=task_valid,
            positions=positions,
            grid_dirs=grid_dirs,
            grid_steer=grid_steer,
            local_null_dirs_by_target=local_null_dirs_by_target,
        )
        gate = candidate_passes_gate(metrics, cond, args, k_value)
        weak_peak_ok = float(metrics["weak_peak_db"]) >= weak_peak_floor
        improved = (
            gate
            and weak_peak_ok
            and float(metrics["psll_to_weakest_peak_db"]) + min_improvement
            < float(best_metrics["psll_to_weakest_peak_db"])
        )
        if improved:
            current = trial
            best_weights = trial.copy()
            best_metrics = metrics
            best_gate = gate
            accepted_steps += 1
        else:
            # Keep a small exploratory continuation only if the gate survived.
            # Once the projected descent breaks feasibility, return to the best
            # feasible point found so far.
            if not gate or not weak_peak_ok:
                break
            current = trial

    return best_weights, best_metrics, best_gate, {
        "psll_refine_steps_used": accepted_steps,
        "psll_refine_delta_db": float(best_metrics["psll_to_weakest_peak_db"])
        - float(initial_metrics["psll_to_weakest_peak_db"]),
    }


def local_isolation_min_db(
    *,
    weights: np.ndarray,
    positions: np.ndarray,
    target_dirs: np.ndarray,
    valid_indices: np.ndarray,
    local_null_dirs_by_target: list[np.ndarray] | None,
) -> float:
    if valid_indices.size <= 1 or not local_null_dirs_by_target:
        return float("nan")
    values: list[float] = []
    target_steer = steering_rx(positions, target_dirs)
    for task_pos, valid_index in enumerate(valid_indices):
        task_weights = weights[:, int(valid_index)]
        desired = float(abs(target_steer[task_pos] @ task_weights))
        leakage = math.sqrt(EPS)
        for other_pos in range(target_dirs.shape[0]):
            if other_pos == task_pos:
                continue
            dirs = [target_dirs[other_pos]]
            dirs.extend(local_null_dirs_by_target[other_pos])
            probe_dirs = np.asarray(dirs, dtype=np.float32)
            probe_steer = steering_rx(positions, probe_dirs)
            leakage = max(leakage, float(np.abs(probe_steer @ task_weights).max()))
        values.append(20.0 * math.log10(max(desired, math.sqrt(EPS)) / max(leakage, math.sqrt(EPS))))
    return float(min(values)) if values else float("nan")


def evaluate_weights(
    *,
    weights: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    positions: np.ndarray,
    grid_dirs: np.ndarray,
    grid_steer: np.ndarray,
    local_null_dirs_by_target: list[np.ndarray] | None = None,
) -> dict[str, float]:
    valid = task_valid.astype(bool)
    target_dirs = target_dirs_for_sample(targets_deg, valid)
    valid_indices = np.flatnonzero(valid)
    combined = weights.sum(axis=1)
    target_steer = steering_rx(positions, target_dirs)
    target_resp = target_steer @ combined
    target_db = 10.0 * np.log10(np.maximum(np.abs(target_resp) ** 2, EPS))
    side_mask = side_mask_for_targets(grid_dirs, target_dirs)
    grid_resp = grid_steer @ combined
    grid_db = 10.0 * np.log10(np.maximum(np.abs(grid_resp) ** 2, EPS))
    side_max = float(grid_db[side_mask].max()) if np.any(side_mask) else float("nan")
    weak_peak = float(target_db.min()) if target_db.size else float("nan")
    strong_peak = float(target_db.max()) if target_db.size else float("nan")
    isolation = float("nan")
    if valid_indices.size > 1:
        response = target_steer @ weights[:, valid_indices]
        mag = np.maximum(np.abs(response), math.sqrt(EPS))
        values: list[float] = []
        for task_index in range(valid_indices.size):
            desired = float(mag[task_index, task_index])
            leakage = float(np.delete(mag[:, task_index], task_index).max())
            values.append(20.0 * math.log10(desired / max(leakage, math.sqrt(EPS))))
        isolation = float(min(values))
    local_isolation = local_isolation_min_db(
        weights=weights,
        positions=positions,
        target_dirs=target_dirs,
        valid_indices=valid_indices,
        local_null_dirs_by_target=local_null_dirs_by_target,
    )
    return {
        "weak_peak_db": weak_peak,
        "strong_peak_db": strong_peak,
        "target_spread_db": strong_peak - weak_peak,
        "worst_sidelobe_db": side_max,
        "psll_to_weakest_peak_db": side_max - weak_peak,
        "psll_to_strongest_peak_db": side_max - strong_peak,
        "isolation_min_db": isolation,
        "local_isolation_min_db": local_isolation,
        "energy_proxy": float(np.sum(np.abs(weights) ** 2)),
    }


def candidate_passes_gate(metrics: dict[str, float], cond: float, args: argparse.Namespace, k: int) -> bool:
    if not np.isfinite(cond) or cond > float(args.condition_limit):
        return False
    if k <= 1:
        return True
    iso = float(metrics["isolation_min_db"])
    if not (np.isfinite(iso) and iso >= float(args.target_isolation_db)):
        return False
    local_iso = float(metrics.get("local_isolation_min_db", float("nan")))
    if np.isfinite(local_iso) and local_iso < float(args.target_local_isolation_db):
        return False
    return True


def score_tuple(metrics: dict[str, float], gate_pass: bool) -> tuple[int, float, float, float]:
    # Gate pass first, then PSLL, then stronger weak peak, then lower energy.
    return (
        0 if gate_pass else 1,
        float(metrics["psll_to_weakest_peak_db"]),
        -float(metrics["weak_peak_db"]),
        float(metrics["energy_proxy"]),
    )


def optimize_one(
    *,
    arrays: dict[str, np.ndarray],
    base_teacher: dict[str, np.ndarray] | None,
    sample_index: int,
    positions: np.ndarray,
    grid_dirs: np.ndarray,
    grid_steer: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    num_active = int(arrays["num_active"][sample_index])
    k_value = int(arrays["k_values"][sample_index])
    task_valid = arrays["task_valid"][sample_index].astype(bool)
    valid_indices = np.flatnonzero(task_valid)
    valid_targets_deg = valid_targets_deg_for_sample(arrays["targets_deg"][sample_index], task_valid)
    target_dirs = target_dirs_for_sample(arrays["targets_deg"][sample_index], task_valid)
    local_null_broadened = local_null_broadening_enabled(
        mode=str(args.local_null_broadening),
        k=k_value,
        valid_targets_deg=valid_targets_deg,
        trigger_k=int(args.local_null_trigger_k),
        trigger_theta_deg=float(args.local_null_trigger_theta_deg),
    )
    local_null_dirs_by_target = (
        make_local_null_dirs_by_target(valid_targets_deg, parse_float_list(str(args.local_null_offsets_deg)))
        if local_null_broadened
        else None
    )
    local_null_offset_count = (
        int(sum(dirs.shape[0] for dirs in local_null_dirs_by_target))
        if local_null_dirs_by_target is not None
        else 0
    )
    original_weights = (
        arrays["task_weights_real_imag"][sample_index, :, :, 0]
        + 1j * arrays["task_weights_real_imag"][sample_index, :, :, 1]
    ).astype(np.complex64)
    original_metrics = evaluate_weights(
        weights=original_weights,
        targets_deg=arrays["targets_deg"][sample_index],
        task_valid=task_valid,
        positions=positions,
        grid_dirs=grid_dirs,
        grid_steer=grid_steer,
        local_null_dirs_by_target=local_null_dirs_by_target,
    )

    seeds = seed_masks_for_sample(
        arrays=arrays,
        base_teacher=base_teacher,
        sample_index=sample_index,
        num_active=num_active,
        rng=rng,
        structured_mode=str(args.structured_mask_mode),
    )
    candidates = list(seeds)
    for seed in seeds:
        for _ in range(max(0, int(args.random_candidates))):
            candidates.append(mutate_mask(seed, rng, int(args.max_random_swaps)))
    candidates = unique_masks(candidates)

    best: dict[str, Any] | None = None
    evaluated = 0
    gate_pass_count = 0

    def evaluate_mask(mask: np.ndarray, source: str) -> dict[str, Any]:
        nonlocal evaluated, gate_pass_count
        weights, cond, ok, constraint_count = lcmv_zf_weights(
            mask=mask,
            positions=positions,
            target_dirs=target_dirs,
            valid_indices=valid_indices,
            diagonal_loading=float(args.diagonal_loading),
            local_null_dirs_by_target=local_null_dirs_by_target,
            local_null_diagonal_loading=float(args.local_null_diagonal_loading),
        )
        evaluated += 1
        if not ok:
            metrics = {
                "weak_peak_db": float("nan"),
                "strong_peak_db": float("nan"),
                "target_spread_db": float("nan"),
                "worst_sidelobe_db": float("nan"),
                "psll_to_weakest_peak_db": float("inf"),
                "psll_to_strongest_peak_db": float("inf"),
                "isolation_min_db": float("nan"),
                "local_isolation_min_db": float("nan"),
                "energy_proxy": float("inf"),
            }
            gate_pass = False
        else:
            metrics = evaluate_weights(
                weights=weights,
                targets_deg=arrays["targets_deg"][sample_index],
                task_valid=task_valid,
                positions=positions,
                grid_dirs=grid_dirs,
                grid_steer=grid_steer,
                local_null_dirs_by_target=local_null_dirs_by_target,
            )
            gate_pass = candidate_passes_gate(metrics, cond, args, k_value)
            refine_info = {"psll_refine_steps_used": 0, "psll_refine_delta_db": 0.0}
            if gate_pass and str(args.psll_refine_mode) == "projected":
                weights, metrics, gate_pass, refine_info = refine_psll_projected(
                    weights=weights,
                    mask=mask,
                    positions=positions,
                    targets_deg=arrays["targets_deg"][sample_index],
                    task_valid=task_valid,
                    target_dirs=target_dirs,
                    valid_indices=valid_indices,
                    grid_dirs=grid_dirs,
                    grid_steer=grid_steer,
                    local_null_dirs_by_target=local_null_dirs_by_target,
                    cond=cond,
                    k_value=k_value,
                    args=args,
                )
        if not ok:
            refine_info = {"psll_refine_steps_used": 0, "psll_refine_delta_db": 0.0}
        if gate_pass:
            gate_pass_count += 1
        return {
            "mask": mask,
            "weights": weights,
            "metrics": metrics,
            "condition": cond,
            "null_constraint_count": constraint_count,
            "gate_pass": gate_pass,
            "source": source,
            **refine_info,
        }

    for i, candidate in enumerate(candidates):
        item = evaluate_mask(candidate, "seed_or_random")
        if best is None or score_tuple(item["metrics"], item["gate_pass"]) < score_tuple(best["metrics"], best["gate_pass"]):
            best = item

    assert best is not None
    current = best
    improvements = 0
    for round_index in range(max(0, int(args.local_swap_rounds))):
        local_items = [
            evaluate_mask(mutate_mask(current["mask"], rng, max(1, int(args.max_random_swaps) // 2)), "local_swap")
            for _ in range(max(0, int(args.local_swap_candidates)))
        ]
        local_best = min(local_items + [current], key=lambda item: score_tuple(item["metrics"], item["gate_pass"]))
        if score_tuple(local_best["metrics"], local_best["gate_pass"]) < score_tuple(current["metrics"], current["gate_pass"]):
            current = local_best
            improvements += 1
        else:
            break

    metrics = current["metrics"]
    mainlobe_kept = (
        float(metrics["weak_peak_db"]) - float(original_metrics["weak_peak_db"])
        >= -float(args.mainlobe_drop_limit_db)
    )
    return {
        "sample_index": sample_index,
        "mask": current["mask"].astype(np.int8),
        "weights": current["weights"].astype(np.complex64),
        "metrics": metrics,
        "original_metrics": original_metrics,
        "gate_pass": bool(current["gate_pass"]),
        "mainlobe_kept": bool(mainlobe_kept),
        "condition": float(current["condition"]),
        "null_constraint_count": int(current["null_constraint_count"]),
        "local_null_broadened": bool(local_null_broadened),
        "local_null_offset_count": int(local_null_offset_count),
        "candidate_count": evaluated,
        "gate_pass_count": gate_pass_count,
        "local_improvements": improvements,
        "source": str(current["source"]),
        "psll_refine_steps_used": int(current.get("psll_refine_steps_used", 0)),
        "psll_refine_delta_db": float(current.get("psll_refine_delta_db", 0.0)),
    }


def load_splits(dataset_dir: Path) -> dict[str, np.ndarray]:
    with (dataset_dir / "training_split_manifest.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key.replace("_id", ""): np.asarray(value, dtype=np.int64) for key, value in payload["splits"].items()}


def select_indices(
    arrays: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    *,
    split: str,
    k_values: list[int],
    active_ratios: list[float],
    samples_per_cell: int,
) -> np.ndarray:
    if split == "all":
        base = np.arange(arrays["k_values"].shape[0], dtype=np.int64)
    else:
        base = splits[split]
    selected: list[int] = []
    for k in k_values:
        for active_ratio in active_ratios:
            cell = base[
                (arrays["k_values"][base] == int(k))
                & np.isclose(arrays["active_ratios_requested"][base], float(active_ratio), atol=1.0e-6)
            ]
            cell = np.sort(cell)
            if samples_per_cell > 0:
                cell = cell[:samples_per_cell]
            selected.extend(int(index) for index in cell)
    return np.asarray(selected, dtype=np.int64)


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def safe_mean(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(arr.mean()) if arr.size else float("nan")


def safe_percentile(values: Iterable[float], q: float) -> float:
    arr = finite_values(values)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: list[tuple[str, str]] = [("all", "all")]
    for ratio in sorted({row["active_ratio"] for row in rows}):
        keys.append(("all", ratio))
    for k in sorted({int(row["k"]) for row in rows}):
        keys.append((str(k), "all"))
        for ratio in sorted({row["active_ratio"] for row in rows if int(row["k"]) == k}):
            keys.append((str(k), ratio))
    out: list[dict[str, Any]] = []
    for k_label, ratio_label in keys:
        group = rows
        if k_label != "all":
            group = [row for row in group if int(row["k"]) == int(k_label)]
        if ratio_label != "all":
            group = [row for row in group if row["active_ratio"] == ratio_label]
        if not group:
            continue
        out.append(
            {
                "k": k_label,
                "active_ratio": ratio_label,
                "n": len(group),
                "gate_pass_rate": safe_mean(1.0 if row["gate_pass"] else 0.0 for row in group),
                "mainlobe_kept_rate": safe_mean(1.0 if row["mainlobe_kept"] else 0.0 for row in group),
                "candidate_count_mean": safe_mean(float(row["candidate_count"]) for row in group),
                "gate_pass_count_mean": safe_mean(float(row["gate_pass_count"]) for row in group),
                "psll_refine_steps_used_mean": safe_mean(float(row.get("psll_refine_steps_used", 0.0)) for row in group),
                "psll_refine_delta_mean_db": safe_mean(float(row.get("psll_refine_delta_db", 0.0)) for row in group),
                "local_null_broadened_rate": safe_mean(float(row["local_null_broadened"]) for row in group),
                "null_constraint_count_mean": safe_mean(float(row["null_constraint_count"]) for row in group),
                "original_psll_mean_db": safe_mean(float(row["original_psll_to_weakest_peak_db"]) for row in group),
                "chosen_psll_mean_db": safe_mean(float(row["chosen_psll_to_weakest_peak_db"]) for row in group),
                "delta_psll_mean_db": safe_mean(float(row["delta_psll_to_weakest_peak_db"]) for row in group),
                "chosen_psll_p95_db": safe_percentile((float(row["chosen_psll_to_weakest_peak_db"]) for row in group), 95),
                "chosen_isolation_mean_db": safe_mean(float(row["chosen_isolation_min_db"]) for row in group),
                "chosen_isolation_p05_db": safe_percentile((float(row["chosen_isolation_min_db"]) for row in group), 5),
                "chosen_isolation_p50_db": safe_percentile((float(row["chosen_isolation_min_db"]) for row in group), 50),
                "chosen_local_isolation_mean_db": safe_mean(float(row["chosen_local_isolation_min_db"]) for row in group),
                "chosen_local_isolation_p05_db": safe_percentile((float(row["chosen_local_isolation_min_db"]) for row in group), 5),
                "chosen_local_isolation_p50_db": safe_percentile((float(row["chosen_local_isolation_min_db"]) for row in group), 50),
                "chosen_energy_mean": safe_mean(float(row["chosen_energy_proxy"]) for row in group),
                "condition_p95": safe_percentile((float(row["condition"]) for row in group), 95),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    out_dir = dataset_dir / "optimized_teachers" / args.run_name
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_npz(dataset_dir / "dataset_arrays.npz")
    base_teacher = None
    if args.base_teacher_dir and (args.base_teacher_dir / "dataset_arrays.npz").exists():
        base_teacher = load_npz(args.base_teacher_dir / "dataset_arrays.npz")
    splits = load_splits(dataset_dir)
    if args.sample_indices_file is not None:
        selected = load_sample_indices_file(args.sample_indices_file)
    else:
        selected = select_indices(
            arrays,
            splits,
            split=str(args.split),
            k_values=parse_int_list(args.k_values),
            active_ratios=parse_float_list(args.active_ratios),
            samples_per_cell=int(args.samples_per_cell),
        )
    control_filtered_count = 0
    if not bool(args.optimize_control_ratios):
        control_ratios = parse_float_list(str(args.control_ratios))
        if control_ratios:
            keep = np.ones(selected.shape[0], dtype=np.bool_)
            selected_ratios = arrays["active_ratios_requested"][selected]
            for ratio in control_ratios:
                keep &= ~np.isclose(selected_ratios, float(ratio), atol=1.0e-6)
            control_filtered_count = int(selected.size - int(keep.sum()))
            selected = selected[keep]
            if control_filtered_count:
                print(f"skipped {control_filtered_count} control-ratio samples: {args.control_ratios}")
    if selected.size == 0:
        raise RuntimeError("No samples selected after split/ratio/control filtering.")

    active_ratio_override = float(args.active_ratio_override)
    if active_ratio_override > 0.0:
        override_count = ratio_to_active_count(active_ratio_override)
        arrays = dict(arrays)
        arrays["num_active"] = arrays["num_active"].copy()
        arrays["active_ratios_requested"] = arrays["active_ratios_requested"].copy()
        arrays["num_active"][selected] = override_count
        arrays["active_ratios_requested"][selected] = active_ratio_override
        print(
            "active-ratio override applied to "
            f"{selected.size} samples: ratio={active_ratio_override:.4f}, num_active={override_count}"
        )

    positions = arrays["positions_lambda"].astype(np.float32)
    _, _, selection_dirs = make_grid(float(args.selection_theta_step), float(args.selection_phi_step))
    _, _, eval_dirs = make_grid(float(args.eval_theta_step), float(args.eval_phi_step))
    eval_phase = 2.0 * np.pi * (eval_dirs @ positions.T)
    eval_steer = np.exp(-1j * eval_phase).astype(np.complex64)

    masks = arrays["masks"].astype(np.int8).copy()
    weights = (
        arrays["task_weights_real_imag"][..., 0] + 1j * arrays["task_weights_real_imag"][..., 1]
    ).astype(np.complex64)

    rows: list[dict[str, Any]] = []
    start = time.time()
    rng = np.random.default_rng(int(args.seed))
    for n_done, sample_index in enumerate(selected, start=1):
        result = optimize_one(
            arrays=arrays,
            base_teacher=base_teacher,
            sample_index=int(sample_index),
            positions=positions,
            grid_dirs=eval_dirs,
            grid_steer=eval_steer,
            rng=rng,
            args=args,
        )
        masks[sample_index] = result["mask"]
        weights[sample_index] = result["weights"]
        metrics = result["metrics"]
        original = result["original_metrics"]
        row = {
            "sample_index": int(sample_index),
            "sample_id": str(arrays["sample_ids"][sample_index]),
            "split": str(args.split),
            "k": int(arrays["k_values"][sample_index]),
            "active_ratio": f"{float(arrays['active_ratios_requested'][sample_index]):.1f}",
            "num_active": int(arrays["num_active"][sample_index]),
            "channel_saving": 1.0 - float(arrays["num_active"][sample_index]) / float(NUM_ELEMENTS),
            "gate_pass": int(result["gate_pass"]),
            "mainlobe_kept": int(result["mainlobe_kept"]),
            "candidate_count": int(result["candidate_count"]),
            "gate_pass_count": int(result["gate_pass_count"]),
            "local_improvements": int(result["local_improvements"]),
            "condition": float(result["condition"]),
            "null_constraint_count": int(result["null_constraint_count"]),
            "local_null_broadened": int(result["local_null_broadened"]),
            "local_null_offset_count": int(result["local_null_offset_count"]),
            "psll_refine_steps_used": int(result["psll_refine_steps_used"]),
            "psll_refine_delta_db": float(result["psll_refine_delta_db"]),
            "source": str(result["source"]),
            "original_psll_to_weakest_peak_db": float(original["psll_to_weakest_peak_db"]),
            "chosen_psll_to_weakest_peak_db": float(metrics["psll_to_weakest_peak_db"]),
            "delta_psll_to_weakest_peak_db": float(metrics["psll_to_weakest_peak_db"])
            - float(original["psll_to_weakest_peak_db"]),
            "original_weak_peak_db": float(original["weak_peak_db"]),
            "chosen_weak_peak_db": float(metrics["weak_peak_db"]),
            "delta_weak_peak_db": float(metrics["weak_peak_db"]) - float(original["weak_peak_db"]),
            "original_isolation_min_db": float(original["isolation_min_db"]),
            "chosen_isolation_min_db": float(metrics["isolation_min_db"]),
            "delta_isolation_min_db": float(metrics["isolation_min_db"]) - float(original["isolation_min_db"]),
            "original_local_isolation_min_db": float(original["local_isolation_min_db"]),
            "chosen_local_isolation_min_db": float(metrics["local_isolation_min_db"]),
            "delta_local_isolation_min_db": float(metrics["local_isolation_min_db"])
            - float(original["local_isolation_min_db"]),
            "chosen_target_spread_db": float(metrics["target_spread_db"]),
            "chosen_energy_proxy": float(metrics["energy_proxy"]),
        }
        rows.append(row)
        if n_done == selected.size or n_done % 30 == 0:
            print(f"processed {n_done}/{selected.size} selected samples, elapsed {time.time() - start:.1f}s")

    compatible = dict(arrays)
    compatible["masks"] = masks
    compatible["task_weights_real_imag"] = np.stack([weights.real, weights.imag], axis=-1)
    np.savez_compressed(out_dir / "dataset_arrays.npz", **compatible)
    np.savez_compressed(
        out_dir / "iso_lcmv_teacher_arrays.npz",
        masks=masks,
        task_weights_real_imag=np.stack([weights.real, weights.imag], axis=-1),
        selected_indices=selected,
        sample_ids=arrays["sample_ids"],
        k_values=arrays["k_values"],
        active_ratios_requested=arrays["active_ratios_requested"],
        num_active=arrays["num_active"],
        targets_deg=arrays["targets_deg"],
        task_valid=arrays["task_valid"],
        positions_lambda=arrays["positions_lambda"],
        element_ixiy=arrays["element_ixiy"],
    )
    shutil.copy2(dataset_dir / "training_split_manifest.json", out_dir / "training_split_manifest.json")
    write_csv(out_dir / "iso_lcmv_metrics.csv", rows)
    summary_rows = summarize(rows)
    write_csv(out_dir / "iso_lcmv_summary_by_k_active.csv", summary_rows)
    run_summary = {
        "run_name": str(args.run_name),
        "dataset_dir": str(dataset_dir),
        "base_teacher_dir": str(args.base_teacher_dir),
        "out_dir": str(out_dir),
        "selected_count": int(selected.size),
        "control_filtered_count": int(control_filtered_count),
        "elapsed_s": time.time() - start,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "headline": [row for row in summary_rows if row["k"] == "all" and row["active_ratio"] == "all"],
        "outputs": {
            "dataset_arrays": str(out_dir / "dataset_arrays.npz"),
            "metrics": str(out_dir / "iso_lcmv_metrics.csv"),
            "summary": str(out_dir / "iso_lcmv_summary_by_k_active.csv"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    arrays_npz = np.load(path, allow_pickle=False)
    return {key: arrays_npz[key] for key in arrays_npz.files}


if __name__ == "__main__":
    main()
