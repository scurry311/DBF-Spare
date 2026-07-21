"""Generate adaptive sparse-array candidates without launching HFSS.

This stage expands one target scene across several active ratios, solves each
mask with regional leakage inequalities, performs a gated PSLL refinement, and
uses the full-wave residual critic only for conservative reranking.  The output
is an offline smoke-test pool; only rows marked ``hfss_eligible`` may be passed
to the existing HFSS materialization workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from generate_iso_lcmv_teacher import (
    evaluate_weights,
    make_grid,
    make_local_null_dirs_by_target,
    mutate_mask,
    normalize_mask,
    seed_masks_for_sample,
    side_mask_for_targets,
    steering_rx,
    target_dirs_for_sample,
    unique_masks,
    valid_targets_deg_for_sample,
)
from train_fullwave_residual_critic_v2 import (
    FullwaveResidualCritic,
    spatial_features,
    target_features,
)
from train_hfss_grid_mask import GridMaskNet
from train_hfss_surrogate import build_features
from project_active_return_weights import project_single_source_weights


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
RESIDUAL_DATASET_DIR = DATASET_DIR / "stage1_fullwave_residual_dataset_v2_20260714"
CRITIC_DIR = DATASET_DIR / "training_runs" / "fullwave_residual_critic_v2_20260714"
ATTENTION_CHECKPOINT = (
    DATASET_DIR
    / "training_runs"
    / "grid_mask_v2_canonical_cnn_k46_140ep"
    / "grid_mask_model.pt"
)
BASE_TEACHER_DIR = DATASET_DIR / "optimized_teachers" / "greedy_psll_v2_canonical"
DEFAULT_OUT_DIR = DATASET_DIR / "adaptive_candidate_pool_v2_smoke_20260714"
DEFAULT_ACTIVE_NETWORK = (
    DATASET_DIR
    / "full_s256p_matched_v2_20260714"
    / "port_class_matching_20260714"
    / "port_class_matched_s256.npz"
)
KMAX = 6
NUM_ELEMENTS = 256
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--residual-dataset-dir", type=Path, default=RESIDUAL_DATASET_DIR)
    parser.add_argument("--critic-dir", type=Path, default=CRITIC_DIR)
    parser.add_argument("--attention-checkpoint", type=Path, default=ATTENTION_CHECKPOINT)
    parser.add_argument("--base-teacher-dir", type=Path, default=BASE_TEACHER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ratios", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--candidates-per-ratio", type=int, default=24)
    parser.add_argument("--shortlist-per-ratio", type=int, default=2)
    parser.add_argument("--theta-step", type=float, default=4.0)
    parser.add_argument("--phi-step", type=float, default=8.0)
    parser.add_argument("--regional-offsets-deg", default="1,2,5")
    parser.add_argument("--nearest-isolation-db", type=float, default=25.0)
    parser.add_argument("--local-isolation-db", type=float, default=20.0)
    parser.add_argument(
        "--projection-margin-db",
        type=float,
        default=1.5,
        help="Extra isolation margin used by POCS; engineering gates remain 25/20 dB.",
    )
    parser.add_argument("--pocs-iterations", type=int, default=35)
    parser.add_argument("--channel-factor", type=float, default=6.0)
    parser.add_argument("--norm-factor", type=float, default=4.0)
    parser.add_argument("--psll-refine-top", type=int, default=6)
    parser.add_argument("--psll-refine-steps", type=int, default=6)
    parser.add_argument("--psll-step-size", type=float, default=0.08)
    parser.add_argument("--uncertainty-kappa", type=float, default=1.64)
    parser.add_argument("--boundary-distance", type=float, default=1.5)
    parser.add_argument("--active-network", type=Path, default=DEFAULT_ACTIVE_NETWORK)
    parser.add_argument("--active-return-min-db", type=float, default=10.0)
    parser.add_argument("--active-projection-iterations", type=int, default=50)
    parser.add_argument("--active-projection-step-size", type=float, default=0.04)
    parser.add_argument("--disable-active-return-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


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


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(mask.astype(np.uint8)).tobytes()).hexdigest()


def target_separation_deg(targets_deg: np.ndarray) -> float:
    if targets_deg.shape[0] < 2:
        return 180.0
    dirs = target_dirs_for_sample(targets_deg, np.ones(targets_deg.shape[0], dtype=bool))
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    upper = np.triu_indices(targets_deg.shape[0], 1)
    return float(np.rad2deg(np.arccos(dots[upper])).min())


def amplitude_stats(weights: np.ndarray) -> tuple[float, float, float, float]:
    amplitude = np.abs(weights)
    nonzero = amplitude[amplitude > 1.0e-9]
    energy = float(np.sum(amplitude**2))
    l2 = float(np.sqrt(energy))
    maximum = float(nonzero.max()) if nonzero.size else 0.0
    minimum = float(nonzero.min()) if nonzero.size else 0.0
    dynamic = 20.0 * math.log10(maximum / minimum) if minimum > 0.0 else 0.0
    return energy, l2, maximum, dynamic


def project_complex_disk(w: np.ndarray, row: np.ndarray, center: complex, radius: float) -> np.ndarray:
    response = complex(row @ w)
    delta = response - center
    magnitude = abs(delta)
    if magnitude <= radius:
        return w
    target = center + radius * delta / max(magnitude, math.sqrt(EPS))
    denom = float(np.vdot(row, row).real)
    return w + row.conj() * ((target - response) / max(denom, EPS))


def project_task_weights(
    w: np.ndarray,
    desired_row: np.ndarray,
    leakage_rows: list[tuple[np.ndarray, float]],
    *,
    channel_limit: float,
    norm_limit: float,
    iterations: int,
) -> np.ndarray:
    out = w.astype(np.complex64).copy()
    desired_denom = float(np.vdot(desired_row, desired_row).real)
    for _ in range(max(1, int(iterations))):
        response = complex(desired_row @ out)
        out += desired_row.conj() * ((1.0 - response) / max(desired_denom, EPS))
        for row, bound in leakage_rows:
            out = project_complex_disk(out, row, 0.0 + 0.0j, float(bound))
        amplitude = np.abs(out)
        over = amplitude > channel_limit
        if np.any(over):
            out[over] *= (channel_limit / amplitude[over]).astype(np.float32)
        norm = float(np.linalg.norm(out))
        if norm > norm_limit:
            out *= np.float32(norm_limit / norm)
    response = complex(desired_row @ out)
    out += desired_row.conj() * ((1.0 - response) / max(desired_denom, EPS))
    return out.astype(np.complex64)


def regional_inequality_weights(
    *,
    mask: np.ndarray,
    positions: np.ndarray,
    target_dirs: np.ndarray,
    valid_indices: np.ndarray,
    local_dirs: list[np.ndarray],
    nearest_isolation_db: float,
    local_isolation_db: float,
    projection_margin_db: float,
    channel_factor: float,
    norm_factor: float,
    pocs_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    active = np.flatnonzero(mask)
    positions_active = positions[active]
    target_rows = steering_rx(positions_active, target_dirs)
    nearest_bound = 10.0 ** (-(float(nearest_isolation_db) + float(projection_margin_db)) / 20.0)
    local_bound = 10.0 ** (-(float(local_isolation_db) + float(projection_margin_db)) / 20.0)
    channel_limit = float(channel_factor) / max(active.size, 1)
    weights = np.zeros((positions.shape[0], KMAX), dtype=np.complex64)
    worst_condition = 0.0
    max_desired_error = 0.0
    max_nearest_ratio = 0.0
    max_local_ratio = 0.0
    for task_pos, valid_index in enumerate(valid_indices):
        desired = target_rows[task_pos]
        initial = desired.conj() / max(float(np.vdot(desired, desired).real), EPS)
        leakage: list[tuple[np.ndarray, float]] = []
        condition_rows = [desired]
        for other_pos in range(target_dirs.shape[0]):
            if other_pos == task_pos:
                continue
            leakage.append((target_rows[other_pos], nearest_bound))
            condition_rows.append(target_rows[other_pos])
            for local_dir in local_dirs[other_pos]:
                local_row = steering_rx(positions_active, local_dir[None, :])[0]
                leakage.append((local_row, local_bound))
        matrix = np.asarray(condition_rows, dtype=np.complex64)
        try:
            worst_condition = max(worst_condition, float(np.linalg.cond(matrix @ matrix.conj().T)))
        except np.linalg.LinAlgError:
            worst_condition = float("inf")
        norm_limit = float(norm_factor) * max(float(np.linalg.norm(initial)), math.sqrt(EPS))
        solved = project_task_weights(
            initial,
            desired,
            leakage,
            channel_limit=channel_limit,
            norm_limit=norm_limit,
            iterations=pocs_iterations,
        )
        desired_value = abs(complex(desired @ solved))
        max_desired_error = max(max_desired_error, abs(desired_value - 1.0))
        for other_pos in range(target_dirs.shape[0]):
            if other_pos == task_pos:
                continue
            nearest_value = abs(complex(target_rows[other_pos] @ solved))
            max_nearest_ratio = max(max_nearest_ratio, nearest_value / max(desired_value, math.sqrt(EPS)))
            if local_dirs[other_pos].size:
                local_rows = steering_rx(positions_active, local_dirs[other_pos])
                local_value = float(np.max(np.abs(local_rows @ solved)))
                max_local_ratio = max(max_local_ratio, local_value / max(desired_value, math.sqrt(EPS)))
        weights[active, int(valid_index)] = solved
    constraint_ok = bool(
        max_desired_error <= 0.02
        and max_nearest_ratio <= nearest_bound * 1.03
        and max_local_ratio <= local_bound * 1.03
    )
    return weights, {
        "condition": worst_condition,
        "constraint_ok": constraint_ok,
        "desired_error_max": max_desired_error,
        "nearest_ratio_max": max_nearest_ratio,
        "local_ratio_max": max_local_ratio,
        "channel_limit": channel_limit,
        "null_constraint_count": (target_dirs.shape[0] - 1) * (1 + len(local_dirs[0])),
    }


def af_gate20(metrics: dict[str, float], solver: dict[str, Any]) -> bool:
    return bool(
        solver["constraint_ok"]
        and np.isfinite(solver["condition"])
        and solver["condition"] <= 1.0e8
        and metrics["psll_to_weakest_peak_db"] <= 0.0
        and metrics["isolation_min_db"] >= 25.0
        and metrics["local_isolation_min_db"] >= 20.0
        and metrics["target_spread_db"] <= 3.0
    )


def refine_psll(
    *,
    weights: np.ndarray,
    mask: np.ndarray,
    positions: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    target_dirs: np.ndarray,
    valid_indices: np.ndarray,
    local_dirs: list[np.ndarray],
    grid_dirs: np.ndarray,
    grid_steer: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    current = weights.copy()
    current_metrics = evaluate_weights(
        weights=current,
        targets_deg=targets_deg,
        task_valid=task_valid,
        positions=positions,
        grid_dirs=grid_dirs,
        grid_steer=grid_steer,
        local_null_dirs_by_target=local_dirs,
    )
    best = current.copy()
    best_metrics = dict(current_metrics)
    accepted = 0
    active = np.flatnonzero(mask)
    side_indices = np.flatnonzero(side_mask_for_targets(grid_dirs, target_dirs))
    for step in range(max(0, int(args.psll_refine_steps))):
        combined = current.sum(axis=1)
        response = grid_steer[side_indices] @ combined
        top_count = min(16, response.size)
        local_top = np.argpartition(np.abs(response) ** 2, -top_count)[-top_count:]
        top_rows = grid_steer[side_indices[local_top]][:, active]
        gradient = top_rows.conj().T @ response[local_top] / float(top_count)
        grad_norm = float(np.linalg.norm(gradient))
        if not np.isfinite(grad_norm) or grad_norm <= 0.0:
            break
        trial = current.copy()
        step_size = float(args.psll_step_size) / math.sqrt(step + 1.0)
        for task_pos, valid_index in enumerate(valid_indices):
            desired_row = steering_rx(positions[active], target_dirs[task_pos : task_pos + 1])[0]
            leakage: list[tuple[np.ndarray, float]] = []
            for other_pos in range(target_dirs.shape[0]):
                if other_pos == task_pos:
                    continue
                nearest_bound = 10.0 ** (
                    -(float(args.nearest_isolation_db) + float(args.projection_margin_db)) / 20.0
                )
                local_bound = 10.0 ** (
                    -(float(args.local_isolation_db) + float(args.projection_margin_db)) / 20.0
                )
                leakage.append(
                    (steering_rx(positions[active], target_dirs[other_pos : other_pos + 1])[0], nearest_bound)
                )
                for local_dir in local_dirs[other_pos]:
                    leakage.append((steering_rx(positions[active], local_dir[None, :])[0], local_bound))
            w_active = trial[active, int(valid_index)] - step_size * gradient / max(grad_norm, EPS) / math.sqrt(valid_indices.size)
            baseline_norm = 1.0 / math.sqrt(max(active.size, 1))
            trial[active, int(valid_index)] = project_task_weights(
                w_active,
                desired_row,
                leakage,
                channel_limit=float(args.channel_factor) / max(active.size, 1),
                norm_limit=float(args.norm_factor) * baseline_norm,
                iterations=max(8, int(args.pocs_iterations) // 2),
            )
        metrics = evaluate_weights(
            weights=trial,
            targets_deg=targets_deg,
            task_valid=task_valid,
            positions=positions,
            grid_dirs=grid_dirs,
            grid_steer=grid_steer,
            local_null_dirs_by_target=local_dirs,
        )
        kept = (
            metrics["isolation_min_db"] >= 25.0
            and metrics["local_isolation_min_db"] >= 20.0
            and metrics["target_spread_db"] <= 3.0
            and metrics["weak_peak_db"] >= best_metrics["weak_peak_db"] - 0.25
        )
        if kept and metrics["psll_to_weakest_peak_db"] < best_metrics["psll_to_weakest_peak_db"] - 0.02:
            best = trial.copy()
            best_metrics = metrics
            current = trial
            accepted += 1
        elif kept:
            current = trial
        else:
            break
    return best, best_metrics, {"psll_steps_accepted": accepted}


def load_attention_model(path: Path, base: dict[str, np.ndarray]) -> GridMaskNet | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    model = GridMaskNet(
        feature_dim=int(config["feature_dim"]),
        element_ixiy=base["element_ixiy"],
        positions_lambda=base["positions_lambda"],
        channels=int(model_config["channels"]),
        condition_dim=int(model_config["condition_dim"]),
        attention_layers=int(model_config["attention_layers"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def attention_masks(
    model: GridMaskNet | None,
    *,
    k_value: int,
    ratio: float,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    num_active: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if model is None:
        return []
    features = build_features(
        np.asarray([k_value], dtype=np.int8),
        np.asarray([ratio], dtype=np.float32),
        targets_deg[None, ...],
        task_valid[None, ...],
    )
    with torch.no_grad():
        logits = model(
            torch.from_numpy(features),
            torch.from_numpy(targets_deg[None, ...].astype(np.float32)),
            torch.from_numpy(task_valid[None, ...].astype(np.float32)),
        )[0].numpy()
    masks: list[np.ndarray] = []
    for noise_scale in (0.0, 0.03, 0.06, 0.10):
        scores = logits + rng.normal(0.0, noise_scale, logits.shape)
        selected = np.argpartition(scores, -num_active)[-num_active:]
        mask = np.zeros(NUM_ELEMENTS, dtype=bool)
        mask[selected] = True
        masks.append(mask)
    return unique_masks(masks)


def build_mask_pool(
    *,
    base: dict[str, np.ndarray],
    base_teacher: dict[str, np.ndarray] | None,
    residual: dict[str, np.ndarray],
    attention: GridMaskNet | None,
    sample_index: int,
    ratio: float,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, str]]:
    num_active = int(round(ratio * NUM_ELEMENTS))
    entries: list[tuple[np.ndarray, str]] = []
    learned = attention_masks(
        attention,
        k_value=int(base["k_values"][sample_index]),
        ratio=ratio,
        targets_deg=base["targets_deg"][sample_index],
        task_valid=base["task_valid"][sample_index],
        num_active=num_active,
        rng=rng,
    )
    entries.extend((mask, "attention_proposal") for mask in learned)
    structured = seed_masks_for_sample(
        arrays=base,
        base_teacher=base_teacher,
        sample_index=sample_index,
        num_active=num_active,
        rng=rng,
        structured_mode="advanced",
    )
    entries.extend((mask, "structured_seed") for mask in structured)
    hard_positive = np.flatnonzero(
        (residual["sample_index"] == sample_index) & (residual["gates"][:, 1] >= 0.5)
    )
    if hard_positive.size == 0:
        valid = base["task_valid"][sample_index].astype(bool)
        current_targets = base["targets_deg"][sample_index][valid]
        current_max_scan = float(current_targets[:, 0].max())
        current_min_sep = target_separation_deg(current_targets)
        positive_rows = np.flatnonzero(
            (residual["gates"][:, 1] >= 0.5)
            & (residual["k_values"] == int(base["k_values"][sample_index]))
        )
        distances: list[tuple[float, int]] = []
        for row_index in positive_rows.tolist():
            donor = int(residual["sample_index"][row_index])
            donor_valid = base["task_valid"][donor].astype(bool)
            donor_targets = base["targets_deg"][donor][donor_valid]
            distance = (
                abs(float(residual["active_ratios"][row_index]) - ratio)
                + abs(float(donor_targets[:, 0].max()) - current_max_scan) / 90.0
                + abs(target_separation_deg(donor_targets) - current_min_sep) / 90.0
            )
            distances.append((distance, row_index))
        distances.sort()
        hard_positive = np.asarray([row for _distance, row in distances[:4]], dtype=np.int64)
    for row_index in hard_positive[:4].tolist():
        mask = normalize_mask(residual["masks"][row_index].astype(bool), num_active, rng)
        entries.append((mask, "hard_positive_neighborhood"))
        entries.append((mutate_mask(mask, rng, 3), "hard_positive_local_swap"))
    seed_copy = list(entries)
    cursor = 0
    while len(entries) < max(count * 2, count + 8):
        parent = seed_copy[cursor % max(len(seed_copy), 1)][0] if seed_copy else base["masks"][sample_index].astype(bool)
        parent = normalize_mask(parent, num_active, rng)
        entries.append((mutate_mask(parent, rng, 2 + cursor % 5), "local_swap"))
        cursor += 1
    unique: list[tuple[np.ndarray, str]] = []
    seen: set[str] = set()
    for mask, source in entries:
        mask = normalize_mask(mask, num_active, rng)
        key = mask_hash(mask)
        if key in seen:
            continue
        seen.add(key)
        unique.append((mask, source))
    if len(unique) < count:
        raise RuntimeError(f"Could only generate {len(unique)} unique masks for sample={sample_index}, ratio={ratio}")
    # Preserve every proposal family before filling the remaining slots.
    selected: list[tuple[np.ndarray, str]] = []
    for source in ("attention_proposal", "structured_seed", "hard_positive_neighborhood", "hard_positive_local_swap", "local_swap"):
        source_rows = [item for item in unique if item[1] == source]
        selected.extend(source_rows[: min(4, len(source_rows))])
    selected_hash = {mask_hash(mask) for mask, _source in selected}
    selected.extend(item for item in unique if mask_hash(item[0]) not in selected_hash)
    return selected[:count]


def load_critic_ensemble(critic_dir: Path) -> list[tuple[FullwaveResidualCritic, dict[str, Any]]]:
    ensemble: list[tuple[FullwaveResidualCritic, dict[str, Any]]] = []
    for path in sorted(critic_dir.glob("seed_*/residual_critic_v2.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = FullwaveResidualCritic(
            scalar_dim=int(checkpoint["scalar_dim"]),
            residual_dim=int(checkpoint["residual_dim"]),
            gate_dim=int(checkpoint["gate_dim"]),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        ensemble.append((model, checkpoint))
    if not ensemble:
        raise FileNotFoundError(f"No critic checkpoints found under {critic_dir}")
    return ensemble


def critic_inference(
    records: list[dict[str, Any]],
    ensemble: list[tuple[FullwaveResidualCritic, dict[str, Any]]],
    residual_dataset: dict[str, np.ndarray],
    kappa: float,
) -> None:
    masks = np.asarray([row["mask"] for row in records], dtype=np.int8)
    weights_ri = np.asarray([row["weights_ri"] for row in records], dtype=np.float32)
    num_active = np.asarray([row["num_active"] for row in records], dtype=np.float32)
    targets = np.asarray([row["targets_deg"] for row in records], dtype=np.float32)
    valid = np.asarray([row["task_valid"] for row in records], dtype=np.int8)
    spatial_raw = spatial_features(masks, weights_ri, num_active)
    target_input = target_features(targets, valid)
    scalar_rows: list[np.ndarray] = []
    for row in records:
        metrics = row["metrics"]
        energy, l2, max_amp, dynamic = amplitude_stats(row["weights"])
        reference = row["reference_scalars"]
        af_peak = float(metrics["weak_peak_db"])
        scalar_rows.append(
            np.asarray(
                [
                    row["k"] / KMAX,
                    row["active_ratio"],
                    row["num_active"] / NUM_ELEMENTS,
                    metrics["psll_to_weakest_peak_db"],
                    metrics["isolation_min_db"],
                    metrics["local_isolation_min_db"],
                    af_peak,
                    metrics["target_spread_db"],
                    energy,
                    energy / max(10.0 ** (af_peak / 10.0), EPS),
                    l2,
                    max_amp,
                    dynamic,
                    math.log10(max(row["solver"]["condition"], 1.0)),
                    row["solver"]["null_constraint_count"] / 256.0,
                    row["min_target_separation_deg"] / 180.0,
                    row["max_scan_theta_deg"] / 90.0,
                    row["mean_scan_theta_deg"] / 90.0,
                    reference[0],
                    reference[1],
                ],
                dtype=np.float32,
            )
        )
    scalar_raw = np.nan_to_num(np.asarray(scalar_rows), nan=0.0, posinf=0.0, neginf=0.0)
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    for model, checkpoint in ensemble:
        spatial = spatial_raw.copy()
        spatial[:, 1:] = (spatial[:, 1:] - checkpoint["spatial_mean"]) / checkpoint["spatial_scale"]
        scalars = (scalar_raw - checkpoint["scalar_mean"]) / checkpoint["scalar_scale"]
        with torch.no_grad():
            mean_z, logvar_z, gate_logits, rank_score = model(
                torch.from_numpy(spatial),
                torch.from_numpy(target_input),
                torch.from_numpy(scalars.astype(np.float32)),
            )
        mean = mean_z.numpy() * checkpoint["residual_scale"] + checkpoint["residual_mean"]
        sigma = np.exp(0.5 * logvar_z.numpy()) * checkpoint["residual_scale"]
        temperature = np.asarray(checkpoint["temperatures"], dtype=np.float32)
        prob = 1.0 / (1.0 + np.exp(-np.clip(gate_logits.numpy() / temperature[None, :], -40, 40)))
        means.append(mean)
        variances.append(sigma**2)
        probabilities.append(prob)
        ranks.append(rank_score.numpy())
    mean_residual = np.mean(means, axis=0)
    second_moment = np.mean([var + mean**2 for var, mean in zip(variances, means)], axis=0)
    sigma_residual = np.sqrt(np.maximum(second_moment - mean_residual**2, 1.0e-8))
    gate_prob = np.mean(probabilities, axis=0)
    rank_mean = np.mean(ranks, axis=0)
    for index, row in enumerate(records):
        metrics = row["metrics"]
        af = np.asarray(
            [
                metrics["psll_to_weakest_peak_db"],
                metrics["isolation_min_db"],
                metrics["local_isolation_min_db"],
                metrics["weak_peak_db"],
                metrics["target_spread_db"],
            ],
            dtype=np.float32,
        )
        predicted = af + mean_residual[index]
        psll_ucb = predicted[0] + kappa * sigma_residual[index, 0]
        nearest_lcb = predicted[1] - kappa * sigma_residual[index, 1]
        local_lcb = predicted[2] - kappa * sigma_residual[index, 2]
        peak_lcb = predicted[3] - kappa * sigma_residual[index, 3]
        spread_ucb = predicted[4] + kappa * sigma_residual[index, 4]
        reference_peak = float(row["reference_scalars"][0])
        drop_ucb = reference_peak - peak_lcb
        boundary_distance = (
            max(psll_ucb, 0.0) / 3.0
            + max(25.0 - nearest_lcb, 0.0) / 5.0
            + max(20.0 - local_lcb, 0.0) / 5.0
            + max(drop_ucb - 0.5, 0.0)
            + max(spread_ucb - 3.0, 0.0) / 3.0
        )
        conservative_pass = bool(
            psll_ucb <= 0.0
            and nearest_lcb >= 25.0
            and local_lcb >= 20.0
            and drop_ucb <= 0.5
            and spread_ucb <= 3.0
            and gate_prob[index, 2] >= 0.5
        )
        utility = (
            2.0 * gate_prob[index, 1]
            + gate_prob[index, 2]
            + 0.15 * rank_mean[index]
            - boundary_distance
            - 0.05 * row["active_ratio"]
        )
        row.update(
            {
                "critic_gate15_probability": float(gate_prob[index, 0]),
                "critic_gate20_probability": float(gate_prob[index, 1]),
                "critic_mainlobe_probability": float(gate_prob[index, 2]),
                "critic_strict_probability": float(gate_prob[index, 3]),
                "critic_rank_score": float(rank_mean[index]),
                "pred_hfss_psll_ucb_db": float(psll_ucb),
                "pred_hfss_nearest_iso_lcb_db": float(nearest_lcb),
                "pred_hfss_local_iso_lcb_db": float(local_lcb),
                "pred_hfss_mainlobe_drop_ucb_db": float(drop_ucb),
                "pred_hfss_peak_spread_ucb_db": float(spread_ucb),
                "critic_boundary_distance": float(boundary_distance),
                "critic_conservative_pass": conservative_pass,
                "critic_utility": float(utility),
            }
        )


def select_scenes(residual: dict[str, np.ndarray], base: dict[str, np.ndarray], split: str, count: int) -> list[int]:
    split_index = {"train": 0, "val": 1, "test": 2}[split]
    rows = np.flatnonzero(residual["split_id"] == split_index)
    scenes = np.unique(residual["sample_index"][rows])
    risk: list[tuple[float, int]] = []
    for scene in scenes.tolist():
        valid = base["task_valid"][scene].astype(bool)
        targets = base["targets_deg"][scene][valid]
        min_sep = target_separation_deg(targets)
        max_theta = float(targets[:, 0].max())
        existing = rows[residual["sample_index"][rows] == scene]
        oracle_fail = float(not np.any(residual["gates"][existing, 1] >= 0.5))
        risk_score = 40.0 * oracle_fail + max_theta + 2.0 * max(0.0, 25.0 - min_sep)
        risk.append((risk_score, int(scene)))
    risk.sort(reverse=True)
    return [scene for _score, scene in risk[:count]]


def reference_scalar_map(residual: dict[str, np.ndarray]) -> dict[int, tuple[float, float]]:
    names = [str(name) for name in residual["scalar_names"]]
    peak_col = names.index("reference_hfss_peak_min_db")
    spread_col = names.index("reference_hfss_peak_spread_db")
    out: dict[int, tuple[float, float]] = {}
    for row, sample in enumerate(residual["sample_index"].tolist()):
        out.setdefault(
            int(sample),
            (
                float(residual["scalar_features"][row, peak_col]),
                float(residual["scalar_features"][row, spread_col]),
            ),
        )
    return out


def csv_record(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    solver = row["solver"]
    return {
        key: value
        for key, value in {
            "candidate_id": row["candidate_id"],
            "sample_index": row["sample_index"],
            "sample_id": row["sample_id"],
            "k": row["k"],
            "active_ratio": row["active_ratio"],
            "num_active": row["num_active"],
            "source": row["source"],
            "mask_hash": row["mask_hash"],
            "min_target_separation_deg": row["min_target_separation_deg"],
            "max_scan_theta_deg": row["max_scan_theta_deg"],
            "af_psll_db": metrics["psll_to_weakest_peak_db"],
            "af_nearest_iso_db": metrics["isolation_min_db"],
            "af_local_iso_db": metrics["local_isolation_min_db"],
            "af_peak_min_db": metrics["weak_peak_db"],
            "af_peak_spread_db": metrics["target_spread_db"],
            "energy_proxy": metrics["energy_proxy"],
            "condition": solver["condition"],
            "max_channel_amplitude": row["max_channel_amplitude"],
            "channel_limit": solver["channel_limit"],
            "constraint_ok": int(solver["constraint_ok"]),
            "af_gate20_pass": int(row["af_gate20_pass"]),
            "active_return_gate_pass": int(row.get("active_return_gate_pass", False)),
            "active_worst_return_loss_db": row.get("active_worst_return_loss_db", float("nan")),
            "active_total_return_loss_db": row.get("active_total_return_loss_db", float("nan")),
            "active_target_response_error_max": row.get("active_target_response_error_max", float("nan")),
            "psll_steps_accepted": row["psll_steps_accepted"],
            "critic_gate15_probability": row["critic_gate15_probability"],
            "critic_gate20_probability": row["critic_gate20_probability"],
            "critic_mainlobe_probability": row["critic_mainlobe_probability"],
            "critic_strict_probability": row["critic_strict_probability"],
            "pred_hfss_psll_ucb_db": row["pred_hfss_psll_ucb_db"],
            "pred_hfss_nearest_iso_lcb_db": row["pred_hfss_nearest_iso_lcb_db"],
            "pred_hfss_local_iso_lcb_db": row["pred_hfss_local_iso_lcb_db"],
            "pred_hfss_mainlobe_drop_ucb_db": row["pred_hfss_mainlobe_drop_ucb_db"],
            "pred_hfss_peak_spread_ucb_db": row["pred_hfss_peak_spread_ucb_db"],
            "critic_boundary_distance": row["critic_boundary_distance"],
            "critic_conservative_pass": int(row["critic_conservative_pass"]),
            "critic_utility": row["critic_utility"],
            "candidate_rank_within_scene_ratio": row.get("candidate_rank_within_scene_ratio", ""),
            "hfss_eligible": int(row.get("hfss_eligible", False)),
            "shortlist_rank": row.get("shortlist_rank", ""),
        }.items()
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(int(args.seed))
    base = load_npz(args.dataset_dir / "dataset_arrays.npz")
    residual = load_npz(args.residual_dataset_dir / "fullwave_residual_dataset_v2.npz")
    base_teacher_path = args.base_teacher_dir / "dataset_arrays.npz"
    base_teacher = load_npz(base_teacher_path) if base_teacher_path.exists() else None
    attention = load_attention_model(args.attention_checkpoint, base)
    ensemble = load_critic_ensemble(args.critic_dir)
    active_network: dict[str, np.ndarray] | None = None
    if not args.disable_active_return_gate:
        active_network = load_npz(args.active_network)
        expected_ports = [str(port) for port in base["port_names"]]
        network_ports = [str(port) for port in active_network["port_names"]]
        if network_ports != expected_ports:
            raise RuntimeError("Active-return network port order does not match the base dataset")
    ratios = parse_float_list(args.ratios)
    scenes = select_scenes(residual, base, args.split, int(args.max_scenes))
    references = reference_scalar_map(residual)
    theta_values, phi_values, grid_dirs = make_grid(float(args.theta_step), float(args.phi_step))
    del theta_values, phi_values
    positions = base["positions_lambda"].astype(np.float32)
    grid_steer = steering_rx(positions, grid_dirs)
    records: list[dict[str, Any]] = []
    for scene_pos, sample_index in enumerate(scenes, 1):
        task_valid = base["task_valid"][sample_index].astype(bool)
        valid_indices = np.flatnonzero(task_valid)
        targets_deg = base["targets_deg"][sample_index].astype(np.float32)
        valid_targets = valid_targets_deg_for_sample(targets_deg, task_valid)
        target_dirs = target_dirs_for_sample(targets_deg, task_valid)
        local_dirs = make_local_null_dirs_by_target(valid_targets, parse_float_list(args.regional_offsets_deg))
        min_sep = target_separation_deg(valid_targets)
        max_theta = float(valid_targets[:, 0].max())
        mean_theta = float(valid_targets[:, 0].mean())
        for ratio in ratios:
            pool = build_mask_pool(
                base=base,
                base_teacher=base_teacher,
                residual=residual,
                attention=attention,
                sample_index=sample_index,
                ratio=ratio,
                count=int(args.candidates_per_ratio),
                rng=rng,
            )
            ratio_records: list[dict[str, Any]] = []
            for candidate_pos, (mask, source) in enumerate(pool):
                weights, solver = regional_inequality_weights(
                    mask=mask,
                    positions=positions,
                    target_dirs=target_dirs,
                    valid_indices=valid_indices,
                    local_dirs=local_dirs,
                    nearest_isolation_db=float(args.nearest_isolation_db),
                    local_isolation_db=float(args.local_isolation_db),
                    projection_margin_db=float(args.projection_margin_db),
                    channel_factor=float(args.channel_factor),
                    norm_factor=float(args.norm_factor),
                    pocs_iterations=int(args.pocs_iterations),
                )
                metrics = evaluate_weights(
                    weights=weights,
                    targets_deg=targets_deg,
                    task_valid=task_valid,
                    positions=positions,
                    grid_dirs=grid_dirs,
                    grid_steer=grid_steer,
                    local_null_dirs_by_target=local_dirs,
                )
                energy, _l2, max_amp, _dynamic = amplitude_stats(weights)
                ratio_records.append(
                    {
                        "candidate_id": f"s{sample_index}_r{ratio:.1f}_c{candidate_pos:02d}",
                        "sample_index": sample_index,
                        "sample_id": str(base["sample_ids"][sample_index]),
                        "k": int(base["k_values"][sample_index]),
                        "active_ratio": float(ratio),
                        "num_active": int(mask.sum()),
                        "source": source,
                        "mask_hash": mask_hash(mask),
                        "mask": mask.astype(np.int8),
                        "weights": weights,
                        "weights_ri": np.stack([weights.real, weights.imag], axis=-1).astype(np.float32),
                        "targets_deg": targets_deg,
                        "task_valid": task_valid.astype(np.int8),
                        "metrics": metrics,
                        "solver": solver,
                        "min_target_separation_deg": min_sep,
                        "max_scan_theta_deg": max_theta,
                        "mean_scan_theta_deg": mean_theta,
                        "reference_scalars": references[sample_index],
                        "max_channel_amplitude": max_amp,
                        "energy_proxy": energy,
                        "psll_steps_accepted": 0,
                    }
                )
            refine_order = sorted(
                range(len(ratio_records)),
                key=lambda idx: (
                    not ratio_records[idx]["solver"]["constraint_ok"],
                    ratio_records[idx]["metrics"]["psll_to_weakest_peak_db"],
                ),
            )[: int(args.psll_refine_top)]
            for index in refine_order:
                row = ratio_records[index]
                weights, metrics, refine = refine_psll(
                    weights=row["weights"],
                    mask=row["mask"].astype(bool),
                    positions=positions,
                    targets_deg=targets_deg,
                    task_valid=task_valid,
                    target_dirs=target_dirs,
                    valid_indices=valid_indices,
                    local_dirs=local_dirs,
                    grid_dirs=grid_dirs,
                    grid_steer=grid_steer,
                    args=args,
                )
                row["weights"] = weights
                row["weights_ri"] = np.stack([weights.real, weights.imag], axis=-1).astype(np.float32)
                row["metrics"] = metrics
                row["max_channel_amplitude"] = amplitude_stats(weights)[2]
                row.update(refine)
            for row in ratio_records:
                if active_network is not None:
                    valid = row["task_valid"].astype(bool)
                    source_initial = np.conjugate(np.sum(row["weights"][:, valid], axis=1))
                    source_initial[~row["mask"].astype(bool)] = 0.0
                    source_target_rows = (
                        steering_rx(positions, target_dirs).conj()
                        @ np.asarray(active_network["antenna_incident_wave_map"], dtype=np.complex64)
                    )
                    source_projected, active_metrics = project_single_source_weights(
                        source_initial,
                        row["mask"].astype(bool),
                        np.asarray(active_network["s_parameters"], dtype=np.complex64),
                        source_target_rows,
                        iterations=int(args.active_projection_iterations),
                        step_size=float(args.active_projection_step_size),
                        return_loss_min_db=float(args.active_return_min_db),
                    )
                    row["hfss_source_weights"] = source_projected
                    row["hfss_source_weights_ri"] = np.stack(
                        [source_projected.real, source_projected.imag], axis=-1
                    ).astype(np.float32)
                    row["active_return_gate_pass"] = bool(active_metrics["engineering_10db_gate_pass"])
                    row["active_worst_return_loss_db"] = active_metrics["worst_active_return_loss_db"]
                    row["active_total_return_loss_db"] = active_metrics["total_return_loss_db"]
                    row["active_target_response_error_max"] = active_metrics["target_response_error_max"]
                else:
                    row["hfss_source_weights"] = np.conjugate(
                        np.sum(row["weights"][:, row["task_valid"].astype(bool)], axis=1)
                    ).astype(np.complex64)
                    row["hfss_source_weights_ri"] = np.stack(
                        [row["hfss_source_weights"].real, row["hfss_source_weights"].imag], axis=-1
                    ).astype(np.float32)
                    row["active_return_gate_pass"] = True
                    row["active_worst_return_loss_db"] = float("nan")
                    row["active_total_return_loss_db"] = float("nan")
                    row["active_target_response_error_max"] = float("nan")
                row["af_gate20_pass"] = af_gate20(row["metrics"], row["solver"])
            records.extend(ratio_records)
        print(f"scene {scene_pos}/{len(scenes)} complete: sample_index={sample_index}")

    critic_inference(records, ensemble, residual, float(args.uncertainty_kappa))
    shortlist: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    by_scene_ratio: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_scene_ratio[(row["sample_index"], row["active_ratio"])].append(row)
    for key, group in sorted(by_scene_ratio.items()):
        group.sort(key=lambda item: item["critic_utility"], reverse=True)
        for rank, item in enumerate(group, 1):
            item["candidate_rank_within_scene_ratio"] = rank
        eligible_group = [
            item
            for item in group
            if item["af_gate20_pass"]
            and item["active_return_gate_pass"]
            and (
                item["critic_conservative_pass"]
                or item["critic_boundary_distance"] <= float(args.boundary_distance)
            )
        ]
        selected = eligible_group[: int(args.shortlist_per_ratio)]
        for rank, row in enumerate(selected, 1):
            row["hfss_eligible"] = True
            row["shortlist_rank"] = rank
            shortlist.append(row)
    for sample_index in scenes:
        chosen: dict[str, Any] | None = None
        fallback: dict[str, Any] | None = None
        for ratio in sorted(ratios):
            group = by_scene_ratio[(sample_index, ratio)]
            best = max(group, key=lambda item: item["critic_utility"])
            if fallback is None or best["critic_utility"] > fallback["critic_utility"]:
                fallback = best
            if any(item.get("hfss_eligible", False) for item in group):
                chosen = max(
                    (item for item in group if item.get("hfss_eligible", False)),
                    key=lambda item: item["critic_utility"],
                )
                break
        selected = chosen or fallback
        assert selected is not None
        adaptive_rows.append(
            {
                "sample_index": sample_index,
                "candidate_id": selected["candidate_id"],
                "selected_ratio": selected["active_ratio"],
                "hfss_eligible": int(selected.get("hfss_eligible", False)),
                "selection_status": "minimum_feasible_ratio" if chosen is not None else "no_feasible_ratio_fallback",
                "critic_gate20_probability": selected["critic_gate20_probability"],
                "critic_boundary_distance": selected["critic_boundary_distance"],
                "af_psll_db": selected["metrics"]["psll_to_weakest_peak_db"],
                "af_nearest_iso_db": selected["metrics"]["isolation_min_db"],
                "af_local_iso_db": selected["metrics"]["local_isolation_min_db"],
            }
        )

    masks = np.asarray([row["mask"] for row in records], dtype=np.int8)
    weights_ri = np.asarray([row["weights_ri"] for row in records], dtype=np.float32)
    source_weights_ri = np.asarray([row["hfss_source_weights_ri"] for row in records], dtype=np.float32)
    np.savez_compressed(
        args.out_dir / "candidate_pool.npz",
        candidate_id=np.asarray([row["candidate_id"] for row in records]),
        sample_index=np.asarray([row["sample_index"] for row in records], dtype=np.int32),
        active_ratio=np.asarray([row["active_ratio"] for row in records], dtype=np.float32),
        source=np.asarray([row["source"] for row in records]),
        masks=masks,
        weights_real_imag=weights_ri,
        hfss_source_weights_real_imag=source_weights_ri,
        targets_deg=np.asarray([row["targets_deg"] for row in records], dtype=np.float32),
        task_valid=np.asarray([row["task_valid"] for row in records], dtype=np.int8),
        hfss_eligible=np.asarray([row.get("hfss_eligible", False) for row in records], dtype=np.int8),
        active_return_gate_pass=np.asarray([row["active_return_gate_pass"] for row in records], dtype=np.int8),
    )
    write_csv(args.out_dir / "candidate_pool_metrics.csv", [csv_record(row) for row in records])
    ranking_rows = sorted(
        (csv_record(row) for row in records),
        key=lambda row: (
            int(row["sample_index"]),
            float(row["active_ratio"]),
            int(row["candidate_rank_within_scene_ratio"]),
        ),
    )
    write_csv(args.out_dir / "candidate_ranking.csv", ranking_rows)
    shortlist_path = args.out_dir / "hfss_smoke_shortlist.csv"
    if shortlist:
        write_csv(shortlist_path, [csv_record(row) for row in shortlist])
    else:
        shortlist_path.write_text(
            "candidate_id,sample_index,active_ratio,hfss_eligible,reason\n",
            encoding="utf-8-sig",
        )
    write_csv(args.out_dir / "adaptive_ratio_selection.csv", adaptive_rows)
    source_counts = Counter(row["source"] for row in records)
    ratio_summary: list[dict[str, Any]] = []
    for ratio in ratios:
        group = [row for row in records if math.isclose(row["active_ratio"], ratio)]
        ratio_summary.append(
            {
                "active_ratio": ratio,
                "candidate_count": len(group),
                "scene_count": len({row["sample_index"] for row in group}),
                "af_gate20_rate": float(np.mean([row["af_gate20_pass"] for row in group])),
                "active_return_gate_rate": float(np.mean([row["active_return_gate_pass"] for row in group])),
                "critic_conservative_rate": float(np.mean([row["critic_conservative_pass"] for row in group])),
                "hfss_eligible_count": sum(bool(row.get("hfss_eligible", False)) for row in group),
                "af_psll_mean_db": float(np.mean([row["metrics"]["psll_to_weakest_peak_db"] for row in group])),
                "af_local_iso_mean_db": float(np.mean([row["metrics"]["local_isolation_min_db"] for row in group])),
            }
        )
    write_csv(args.out_dir / "ratio_smoke_summary.csv", ratio_summary)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "offline_candidate_oracle_smoke",
        "hfss_launched": False,
        "scene_count": len(scenes),
        "scenes": scenes,
        "ratios": ratios,
        "candidate_count": len(records),
        "candidates_per_scene_ratio": int(args.candidates_per_ratio),
        "critic_ensemble_size": len(ensemble),
        "attention_checkpoint_used": bool(attention is not None),
        "source_counts": dict(source_counts),
        "af_gate20_rate": float(np.mean([row["af_gate20_pass"] for row in records])),
        "active_return_gate_rate": float(np.mean([row["active_return_gate_pass"] for row in records])),
        "critic_conservative_rate": float(np.mean([row["critic_conservative_pass"] for row in records])),
        "hfss_eligible_count": len(shortlist),
        "adaptive_minimum_ratio_found_rate": float(
            np.mean([row["selection_status"] == "minimum_feasible_ratio" for row in adaptive_rows])
        ),
        "launch_guard": {
            "large_hfss_allowed": False,
            "reason": (
                "Full-S active-return gate is mandatory. No large HFSS batch is allowed until both per-port and "
                "total reflected-power return loss reach 10 dB."
            ),
        },
        "ratio_summary": ratio_summary,
        "elapsed_s": time.time() - started,
        "outputs": {
            "pool": str(args.out_dir / "candidate_pool.npz"),
            "metrics": str(args.out_dir / "candidate_pool_metrics.csv"),
            "shortlist": str(args.out_dir / "hfss_smoke_shortlist.csv"),
            "adaptive_selection": str(args.out_dir / "adaptive_ratio_selection.csv"),
        },
    }
    (args.out_dir / "candidate_pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
