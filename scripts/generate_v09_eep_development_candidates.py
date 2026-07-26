#!/usr/bin/env python3
"""Generate the independent v0.9 sparse EEP/S256 development candidate pool.

The package contains no ratio-1.0 or nominal-control candidate.  Nominal and
implementation-perturbed EEP metrics are kept separate so that downstream
models learn residual physical margins instead of opaque pass/fail labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from generate_dense_boundary_hard_negatives import quantized_perturbation
from generate_expanded_independent_residual_scenes import (
    angular_separation_deg,
    phase_migrate,
    target_hash,
)
from generate_gate15_boundary_scenes import (
    FastPatternEvaluator,
    GLOBAL_SHIFTS,
    metric_at,
    prospective_grid_targets,
)
from generate_iso_lcmv_teacher import (
    fill_mask_from_scores,
    mask_from_scores,
    mirrored_pair_mask,
    mutate_mask,
    normalize_mask,
    ring_quadrant_balanced_mask,
    spread_greedy_mask,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from optimize_trusted_eep_s256_joint_weights import active_return
from refine_trusted_dense_local_eep_joint import (
    DenseConfig,
    DenseExternalEEP,
    build_constraints,
    dense_local_indices,
    nearest_grid_index,
    project_combined_targets,
    project_dense_task,
    refine_one,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_dataset_20260724_run01"
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_EXCITATIONS = (
    ROOT
    / "hfss_outputs"
    / "trusted_dense_joint_hfss_smoke_20260724_run01"
    / "case_excitations.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_eep_development_candidates_20260726_run01"
DEFAULT_EXCLUDES = (
    ROOT / "baselines" / "2026-07-25-gate15-boundary" / "snapshots" / "critic_dataset.npz",
    ROOT / "hfss_outputs" / "prospective_gate15_scenes_20260725_run01" / "dataset_arrays.npz",
)
KMAX = 6
NUM_ELEMENTS = 256
EPS = 1.0e-12
MARGIN_NAMES = np.asarray(
    ["psll", "nearest_iso", "local20_iso", "mainlobe", "active_rl"]
)
METRIC_NAMES = np.asarray(
    [
        "psll_db",
        "weakest_target_gain_db",
        "target_spread_db",
        "nearest_iso_db",
        "local_iso_db",
        "pointing_error_deg",
    ]
)
FAST_CONFIG = DenseConfig(
    "v09_fast_dense",
    True,
    24,
    3,
    0.018,
    48.0,
    12.0,
    4.0,
    -20.0,
    1,
    48,
    2.0,
)
PERTURBATION_PROFILES = (
    (1.0, 0.05, 0),
    (2.0, 0.10, 0),
    (4.0, 0.20, 0),
    (6.0, 0.30, 0),
    (8.0, 0.40, 0),
    (10.0, 0.50, 1),
    (12.0, 0.60, 1),
    (15.0, 0.75, 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--excitations", type=Path, default=DEFAULT_EXCITATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scene-count", type=int, default=60)
    parser.add_argument(
        "--k-counts",
        default="",
        help="Optional explicit K=2,4,6 scene counts, for example 24,6,24.",
    )
    parser.add_argument("--masks-per-ratio", type=int, default=8)
    parser.add_argument("--ratios", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--phase-bits", type=int, default=6)
    parser.add_argument("--amplitude-bits", type=int, default=7)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument(
        "--donor-dataset",
        type=Path,
        action="append",
        default=[],
        help="Development-only package supplying historical strict-positive masks.",
    )
    parser.add_argument(
        "--scene-source",
        type=Path,
        default=None,
        help="Reuse target scenes from an existing development pool.",
    )
    parser.add_argument(
        "--rescue-dataset",
        type=Path,
        default=None,
        help="Generate local masks around the best candidates in this pool.",
    )
    parser.add_argument("--only-failed-scenes", action="store_true")
    parser.add_argument("--targeted-hard-fraction", type=float, default=0.0)
    parser.add_argument("--small-separation-deg", type=float, default=10.5)
    parser.add_argument("--large-scan-deg", type=float, default=50.0)
    parser.add_argument("--continuation-from-high-ratio", action="store_true")
    parser.add_argument("--sample-index-start", type=int, default=390000)
    parser.add_argument("--dataset-version", default="v0.9-eep-development-pool")
    parser.add_argument("--k6-cycles", type=int, default=24)
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser.parse_args()


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


def complex_to_ri(values: np.ndarray) -> np.ndarray:
    return np.stack((values.real, values.imag), axis=-1).astype(np.float32)


def load_package(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def excluded_target_hashes(packages: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for source_path in packages:
        path = source_path / "dataset_arrays.npz" if source_path.is_dir() else source_path
        if not path.exists():
            continue
        data = load_package(path)
        if "targets_deg" not in data:
            continue
        k_values = np.asarray(data.get("k_values", data["task_valid"].sum(axis=1)), dtype=int)
        for index, k_value in enumerate(k_values):
            hashes.add(target_hash(np.asarray(data["targets_deg"][index, :k_value], dtype=float)))
    return hashes


def generate_scenes(
    base: dict[str, np.ndarray],
    scene_count: int,
    used_hashes: set[str],
) -> list[dict[str, Any]]:
    if scene_count < 3:
        raise ValueError("scene-count must be at least 3")
    per_k = [scene_count // 3] * 3
    for index in range(scene_count % 3):
        per_k[index] += 1
    scenes: list[dict[str, Any]] = []
    for k_value, required in zip((2, 4, 6), per_k):
        parents = np.flatnonzero(np.asarray(base["k_values"], dtype=int) == k_value).tolist()
        if not parents:
            raise RuntimeError(f"No trusted K={k_value} parent")
        accepted = 0
        attempts = 0
        for pattern in range(1, 18):
            for dtheta, dphi in GLOBAL_SHIFTS:
                for parent in parents:
                    attempts += 1
                    old = np.asarray(base["targets_deg"][parent, :k_value], dtype=float)
                    targets = prospective_grid_targets(old, dtheta, dphi, pattern)
                    digest = target_hash(targets)
                    if digest in used_hashes or angular_separation_deg(targets) < 5.0:
                        continue
                    used_hashes.add(digest)
                    local_index = accepted
                    split_id = 0 if local_index < math.ceil(required * 0.60) else (
                        1 if local_index < math.ceil(required * 0.80) else 2
                    )
                    scenes.append(
                        {
                            "sample_index": 390000 + len(scenes),
                            "scene_id": f"v09_dev_k{k_value}_{accepted:02d}_{digest}",
                            "target_hash": digest,
                            "parent": int(parent),
                            "k_value": int(k_value),
                            "targets": targets,
                            "old_targets": old,
                            "split_id": split_id,
                            "min_separation": angular_separation_deg(targets),
                            "max_theta": float(np.max(targets[:, 0])),
                            "large_scan": int(np.max(targets[:, 0]) >= 50.0),
                        }
                    )
                    accepted += 1
                    if accepted >= required:
                        break
                if accepted >= required:
                    break
            if accepted >= required:
                break
        if accepted != required:
            raise RuntimeError(
                f"Only generated {accepted}/{required} independent K={k_value} scenes after {attempts} attempts"
            )
    return scenes


def compressed_gap_variants(
    old: np.ndarray,
    dtheta: int,
    dphi: int,
    pattern: int,
) -> list[np.ndarray]:
    """Create grid-aligned near-boundary task pairs without changing K."""
    base = prospective_grid_targets(old, dtheta, dphi, pattern)
    k_value = int(base.shape[0])
    anchor = int(pattern % k_value)
    moving = int((anchor + 1 + (pattern // max(k_value, 1)) % max(k_value - 1, 1)) % k_value)
    if moving == anchor:
        moving = (anchor + 1) % k_value
    variants: list[np.ndarray] = []
    gap_offsets = (16.0, 18.0, 20.0) if k_value == 2 else (12.0, 14.0, 16.0)
    for offset_index, delta in enumerate(gap_offsets):
        trial = base.copy()
        sign = -1.0 if (pattern + offset_index) % 2 else 1.0
        theta = float(trial[anchor, 0]) + sign * delta
        if theta < 1.0 or theta > 75.0:
            theta = float(trial[anchor, 0]) - sign * delta
        if theta < 1.0 or theta > 75.0:
            continue
        trial[moving] = (theta, float(trial[anchor, 1]))
        separation = angular_separation_deg(trial)
        if 5.0 <= separation <= max(gap_offsets) + 0.5:
            variants.append(trial)
    return variants


def generate_targeted_scenes(
    base: dict[str, np.ndarray],
    k_counts: dict[int, int],
    used_hashes: set[str],
    *,
    hard_fraction: float,
    small_separation_deg: float,
    large_scan_deg: float,
    sample_index_start: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate independent K-stratified scenes with explicit hard-scene quotas."""
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("targeted-hard-fraction must be within [0, 1]")
    scenes: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for k_value in (2, 4, 6):
        required = int(k_counts.get(k_value, 0))
        if required <= 0:
            continue
        parents = np.flatnonzero(np.asarray(base["k_values"], dtype=int) == k_value).tolist()
        if not parents:
            raise RuntimeError(f"No trusted K={k_value} parent")
        candidates: list[dict[str, Any]] = []
        local_hashes: set[str] = set()
        for pattern in range(1, 24):
            for dtheta, dphi in GLOBAL_SHIFTS:
                for parent in parents:
                    old = np.asarray(base["targets_deg"][parent, :k_value], dtype=float)
                    variants = [
                        prospective_grid_targets(old, dtheta, dphi, pattern),
                        *compressed_gap_variants(old, dtheta, dphi, pattern),
                    ]
                    for targets in variants:
                        separation = angular_separation_deg(targets)
                        if separation < 5.0:
                            continue
                        digest = target_hash(targets)
                        if digest in used_hashes or digest in local_hashes:
                            continue
                        local_hashes.add(digest)
                        max_theta = float(np.max(targets[:, 0]))
                        candidates.append(
                            {
                                "target_hash": digest,
                                "parent": int(parent),
                                "targets": targets,
                                "old_targets": old,
                                "min_separation": separation,
                                "max_theta": max_theta,
                                "small_gap": int(separation <= small_separation_deg),
                                "large_scan": int(max_theta >= large_scan_deg),
                            }
                        )
        if len(candidates) < required:
            raise RuntimeError(f"Only generated {len(candidates)}/{required} K={k_value} scenes")
        order = rng.permutation(len(candidates)).tolist()
        shuffled = [candidates[index] for index in order]
        selected: list[dict[str, Any]] = []
        selected_hashes: set[str] = set()

        def take(predicate: Any, count: int) -> None:
            for candidate in shuffled:
                if len([row for row in selected if predicate(row)]) >= count:
                    break
                if candidate["target_hash"] in selected_hashes or not predicate(candidate):
                    continue
                selected.append(candidate)
                selected_hashes.add(candidate["target_hash"])

        hard_target = int(math.ceil(required * hard_fraction))
        small_quota = int(math.ceil(hard_target / 2.0))
        large_quota = hard_target - small_quota
        take(lambda row: bool(row["small_gap"]), small_quota)
        take(lambda row: bool(row["large_scan"]), large_quota)
        take(lambda row: bool(row["small_gap"] or row["large_scan"]), hard_target)
        take(lambda _row: True, required)
        if len(selected) != required:
            raise RuntimeError(f"Could only select {len(selected)}/{required} K={k_value} scenes")
        for local_index, candidate in enumerate(selected):
            digest = str(candidate["target_hash"])
            used_hashes.add(digest)
            split_id = 0 if local_index < math.ceil(required * 0.60) else (
                1 if local_index < math.ceil(required * 0.80) else 2
            )
            scenes.append(
                {
                    "sample_index": int(sample_index_start + len(scenes)),
                    "scene_id": f"targeted_k{k_value}_{local_index:02d}_{digest}",
                    "target_hash": digest,
                    "parent": int(candidate["parent"]),
                    "k_value": int(k_value),
                    "targets": np.asarray(candidate["targets"], dtype=float),
                    "old_targets": np.asarray(candidate["old_targets"], dtype=float),
                    "split_id": split_id,
                    "min_separation": float(candidate["min_separation"]),
                    "max_theta": float(candidate["max_theta"]),
                    "small_gap": int(candidate["small_gap"]),
                    "large_scan": int(candidate["large_scan"]),
                }
            )
    return scenes


def load_source_scenes(
    source_dir: Path,
    base: dict[str, np.ndarray],
    *,
    only_failed: bool,
) -> list[dict[str, Any]]:
    data = load_package(source_dir / "dataset_arrays.npz")
    manifest_path = source_dir / "scene_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest = {
            int(row["sample_index"]): row for row in csv.DictReader(handle)
        }
    scenes: list[dict[str, Any]] = []
    sample_values = np.asarray(data["sample_index"], dtype=np.int64)
    strict = np.asarray(data["strict_gate20"], dtype=bool)
    for sample in np.unique(sample_values):
        members = np.flatnonzero(sample_values == sample)
        if only_failed and bool(np.any(strict[members])):
            continue
        first = int(members[0])
        k_value = int(data["k_values"][first])
        row = manifest[int(sample)]
        parent = int(row["parent_candidate_index"])
        scenes.append(
            {
                "sample_index": int(sample),
                "scene_id": str(row["scene_id"]),
                "target_hash": str(data["target_hashes"][first]),
                "parent": parent,
                "k_value": k_value,
                "targets": np.asarray(data["targets_deg"][first, :k_value], dtype=float),
                "old_targets": np.asarray(base["targets_deg"][parent, :k_value], dtype=float),
                "split_id": int(data["split_id"][first]),
                "min_separation": float(data["min_target_separation_deg"][first]),
                "max_theta": float(data["max_target_theta_deg"][first]),
                "small_gap": int(
                    data.get("small_target_gap", np.zeros_like(sample_values))[first]
                ),
                "large_scan": int(data["large_scan"][first]),
            }
        )
    return scenes


def load_rescue_pool(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    source = path / "dataset_arrays.npz" if path.is_dir() else path
    return load_package(source)


def best_rescue_masks(
    rescue: dict[str, np.ndarray] | None,
    *,
    sample_index: int,
    ratio: float,
    limit: int = 4,
) -> list[np.ndarray]:
    if rescue is None:
        return []
    members = np.flatnonzero(
        (np.asarray(rescue["sample_index"], dtype=np.int64) == int(sample_index))
        & np.isclose(
            np.asarray(rescue["active_ratios_requested"], dtype=float),
            float(ratio),
            atol=1.0e-5,
        )
    )
    if members.size == 0:
        return []
    if "strict_violation" in rescue:
        score = np.asarray(rescue["strict_violation"], dtype=float)[members]
    else:
        score = np.maximum(
            -np.asarray(rescue["actual_margins"], dtype=float)[members], 0.0
        ).sum(axis=1)
    order = members[np.argsort(score, kind="stable")[:limit]]
    masks = rescue["masks"] if "masks" in rescue else rescue["mask"]
    return [np.asarray(masks[index], dtype=bool) for index in order]


def matched_steering_tasks(
    migrated_parent: np.ndarray,
    targets: np.ndarray,
    effective: DenseExternalEEP,
    grid_dirs: np.ndarray,
) -> np.ndarray:
    tasks = np.zeros_like(migrated_parent, dtype=np.complex128)
    for task_index, (theta, phi) in enumerate(targets):
        center = nearest_grid_index(grid_dirs, float(theta), float(phi))
        row_theta = np.asarray(effective.etheta[:, center], dtype=np.complex128)
        row_phi = np.asarray(effective.ephi[:, center], dtype=np.complex128)
        parent = np.asarray(migrated_parent[:, task_index], dtype=np.complex128)
        field = np.asarray([parent @ row_theta, parent @ row_phi])
        if float(np.linalg.norm(field)) <= EPS:
            polarization = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
        else:
            polarization = field / np.linalg.norm(field)
        row = np.conjugate(polarization[0]) * row_theta + np.conjugate(polarization[1]) * row_phi
        matched = np.conjugate(row)
        matched *= max(float(np.linalg.norm(parent)), EPS) / max(float(np.linalg.norm(matched)), EPS)
        parent_scalar = parent @ row
        matched_scalar = matched @ row
        if abs(parent_scalar) > EPS and abs(matched_scalar) > EPS:
            matched *= np.exp(1j * (np.angle(parent_scalar) - np.angle(matched_scalar)))
        tasks[:, task_index] = matched
    return tasks.astype(np.complex64)


def structured_masks(
    parent_mask: np.ndarray,
    scores: np.ndarray,
    element_ixiy: np.ndarray,
    num_active: int,
    count: int,
    rng: np.random.Generator,
    donor_masks: list[np.ndarray] | None = None,
    continuation_masks: list[np.ndarray] | None = None,
    priority_masks: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    coords = element_ixiy.astype(np.float64)
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    radius = np.linalg.norm(coords - center[None, :], axis=1)
    radius /= max(float(radius.max()), 1.0)
    normalized_score = np.asarray(scores, dtype=np.float64)
    normalized_score = (normalized_score - normalized_score.min()) / max(
        float(np.ptp(normalized_score)), EPS
    )
    checker = ((coords[:, 0] + coords[:, 1]) % 2.0) - 0.05 * radius
    seeds: list[tuple[np.ndarray, str]] = []
    for priority_index, source in enumerate(priority_masks or []):
        normalized = fill_mask_from_scores(
            np.asarray(source, dtype=bool), normalized_score, num_active
        )
        for swap_count in (1, 2, 3):
            seeds.append(
                (
                    mutate_mask(normalized, rng, max_swaps=swap_count),
                    f"failed_best_swap_{priority_index:02d}_{swap_count}",
                )
            )
    seeds.extend([
        (mask_from_scores(scores, num_active), "power_top"),
        (normalize_mask(parent_mask, num_active, rng), "parent_normalized"),
        (
            ring_quadrant_balanced_mask(
                element_ixiy=element_ixiy, scores=scores, num_active=num_active
            ),
            "ring_quadrant",
        ),
        (mirrored_pair_mask(element_ixiy, scores, num_active), "mirrored_pair"),
        (
            spread_greedy_mask(
                element_ixiy=element_ixiy,
                scores=scores,
                num_active=num_active,
                spacing_weight=0.30,
            ),
            "spread_030",
        ),
        (
            spread_greedy_mask(
                element_ixiy=element_ixiy,
                scores=scores,
                num_active=num_active,
                spacing_weight=0.90,
            ),
            "spread_090",
        ),
        (
            spread_greedy_mask(
                element_ixiy=element_ixiy,
                scores=0.65 * normalized_score + 0.35 * radius,
                num_active=num_active,
                spacing_weight=1.50,
            ),
            "min_spacing_150",
        ),
        (mask_from_scores(0.70 * normalized_score + 0.30 * radius, num_active), "edge_density"),
        (mask_from_scores(0.75 * normalized_score - 0.25 * radius, num_active), "center_density"),
        (mask_from_scores(checker + 0.10 * normalized_score, num_active), "checker_balance"),
    ])
    for donor_index, donor in enumerate(donor_masks or []):
        normalized = fill_mask_from_scores(
            np.asarray(donor, dtype=bool), normalized_score, num_active
        )
        seeds.append((normalized, f"hard_positive_neighbor_{donor_index:02d}"))
        seeds.append(
            (
                mutate_mask(normalized, rng, max_swaps=max(2, num_active // 64)),
                f"hard_positive_swap_{donor_index:02d}",
            )
        )
    for continuation_index, source in enumerate(continuation_masks or []):
        pruned = fill_mask_from_scores(
            np.asarray(source, dtype=bool), normalized_score, num_active
        )
        seeds.append((pruned, f"ratio_continuation_{continuation_index:02d}"))
        seeds.append(
            (
                mutate_mask(pruned, rng, max_swaps=max(2, num_active // 48)),
                f"continuation_swap_{continuation_index:02d}",
            )
        )
    seen: set[bytes] = set()
    masks: list[np.ndarray] = []
    names: list[str] = []
    for mask, name in seeds:
        key = np.packbits(mask).tobytes()
        if key not in seen:
            seen.add(key)
            masks.append(mask)
            names.append(name)
    mutation_index = 0
    while len(masks) < count:
        source = masks[mutation_index % len(masks)]
        local_scale = 2 + mutation_index % max(3, min(10, num_active // 20))
        candidate = mutate_mask(source, rng, max_swaps=local_scale)
        candidate = normalize_mask(candidate, num_active, rng)
        key = np.packbits(candidate).tobytes()
        mutation_index += 1
        if key in seen:
            continue
        seen.add(key)
        masks.append(candidate)
        names.append(f"local_swap_{mutation_index:02d}")
    return masks[:count], names[:count]


def load_donor_pool(paths: list[Path]) -> dict[str, np.ndarray] | None:
    rows: dict[str, list[np.ndarray]] = {
        "masks": [],
        "k_values": [],
        "ratios": [],
        "min_separation": [],
        "max_theta": [],
    }
    for source_path in paths:
        path = source_path / "dataset_arrays.npz" if source_path.is_dir() else source_path
        if not path.exists():
            raise FileNotFoundError(f"Donor dataset not found: {path}")
        data = load_package(path)
        strict = np.asarray(data["strict_gate20"], dtype=bool)
        keep = np.flatnonzero(strict)
        if keep.size == 0:
            continue
        mask_values = data["masks"] if "masks" in data else data["mask"]
        rows["masks"].append(np.asarray(mask_values[keep], dtype=bool))
        rows["k_values"].append(np.asarray(data["k_values"][keep], dtype=np.int8))
        rows["ratios"].append(
            np.asarray(data["active_ratios_requested"][keep], dtype=np.float32)
        )
        rows["min_separation"].append(
            np.asarray(data["min_target_separation_deg"][keep], dtype=np.float32)
        )
        rows["max_theta"].append(
            np.asarray(data["max_target_theta_deg"][keep], dtype=np.float32)
        )
    if not rows["masks"]:
        return None
    return {key: np.concatenate(values, axis=0) for key, values in rows.items()}


def nearest_donor_masks(
    donor: dict[str, np.ndarray] | None,
    *,
    k_value: int,
    ratio: float,
    min_separation: float,
    max_theta: float,
    limit: int = 4,
) -> list[np.ndarray]:
    if donor is None:
        return []
    same_k = np.flatnonzero(np.asarray(donor["k_values"], dtype=int) == int(k_value))
    if same_k.size == 0:
        return []
    distance = (
        4.0 * np.abs(np.asarray(donor["ratios"])[same_k] - float(ratio))
        + np.abs(np.asarray(donor["min_separation"])[same_k] - float(min_separation)) / 30.0
        + np.abs(np.asarray(donor["max_theta"])[same_k] - float(max_theta)) / 60.0
    )
    order = same_k[np.argsort(distance, kind="stable")[:limit]]
    return [np.asarray(donor["masks"][index], dtype=bool) for index in order]


def full_active_metrics(tasks: np.ndarray, mask: np.ndarray, s_matrix: np.ndarray) -> dict[str, float | int]:
    combined = active_return(
        s_matrix, np.sum(tasks, axis=1), mask, relative_db=None, threshold_db=10.0
    )
    per_task = [
        active_return(
            s_matrix,
            tasks[:, task_index],
            mask,
            relative_db=-20.0,
            threshold_db=10.0,
        )
        for task_index in range(tasks.shape[1])
    ]
    task_worst = min(float(value["worst_active_rl_db"]) for value in per_task)
    task_total = min(float(value["total_rl_db"]) for value in per_task)
    return {
        "combined_worst_active_rl_db": float(combined["worst_active_rl_db"]),
        "combined_total_rl_db": float(combined["total_rl_db"]),
        "task_significant_worst_active_rl_db": task_worst,
        "task_significant_worst_total_rl_db": task_total,
        "active_rl_floor_db": min(
            float(combined["worst_active_rl_db"]),
            float(combined["total_rl_db"]),
            task_worst,
            task_total,
        ),
        "active_rl_gate": int(
            int(combined["gate_pass"]) == 1
            and all(int(value["gate_pass"]) == 1 for value in per_task)
        ),
    }


def project_combined_local_peak(
    tasks: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    grid_dirs: np.ndarray,
    operator: DenseExternalEEP,
    constraints: list[Any],
    combined_constraint: Any,
    config: DenseConfig,
    *,
    shape_margin_db: float,
    passes: int = 6,
) -> np.ndarray:
    """Keep combined-pattern local maxima on target points without relaxing pointing."""
    active = np.flatnonzero(mask)
    inner: set[int] = set()
    for theta, phi in targets:
        inner.update(
            dense_local_indices(grid_dirs, float(theta), float(phi), 1.5).tolist()
        )
    regions = []
    for theta, phi in targets:
        outer = set(
            dense_local_indices(grid_dirs, float(theta), float(phi), 5.0).tolist()
        )
        regions.append(np.asarray(sorted(outer - inner), dtype=np.int64))
    if not any(region.size for region in regions):
        return np.asarray(tasks, dtype=np.complex64)
    center_indices = [
        nearest_grid_index(grid_dirs, float(theta), float(phi)) for theta, phi in targets
    ]
    desired = (
        combined_constraint.equal_desired
        if config.equalize_combined_targets
        else combined_constraint.preserve_desired
    )
    out = np.asarray(tasks, dtype=np.complex128).copy()
    for _ in range(max(1, passes)):
        combined_active = np.sum(out[active], axis=1)
        before = combined_active.copy()
        for center, region in zip(center_indices, regions):
            if region.size == 0:
                continue
            center_theta, center_phi = operator.point_rows(center, active)
            center_field = np.asarray(
                [combined_active @ center_theta, combined_active @ center_phi]
            )
            center_norm = max(float(np.linalg.norm(center_field)), EPS)
            local_theta = np.asarray(operator.etheta[np.ix_(active, region)]).T
            local_phi = np.asarray(operator.ephi[np.ix_(active, region)]).T
            field_theta = local_theta @ combined_active
            field_phi = local_phi @ combined_active
            local_power = np.abs(field_theta) ** 2 + np.abs(field_phi) ** 2
            worst_local = int(np.argmax(local_power))
            field = np.asarray([field_theta[worst_local], field_phi[worst_local]])
            field_norm = max(float(np.linalg.norm(field)), EPS)
            bound = center_norm * 10.0 ** (-float(shape_margin_db) / 20.0)
            if field_norm <= bound:
                continue
            polarization = field / field_norm
            row = (
                np.conjugate(polarization[0]) * local_theta[worst_local]
                + np.conjugate(polarization[1]) * local_phi[worst_local]
            )
            response = complex(row @ combined_active)
            target = response * (bound / max(abs(response), EPS))
            combined_active += np.conjugate(row) * (
                (target - response) / max(float(np.vdot(row, row).real), EPS)
            )
        delta = combined_active - before
        out[active] += delta[:, None] / out.shape[1]
        for task_index, constraint in enumerate(constraints):
            out[:, task_index] = project_dense_task(
                out[:, task_index],
                constraint,
                passes=max(1, config.dense_passes),
                top_count=max(64, config.dense_top_count),
                margin_db=config.projection_margin_db,
            )
        project_combined_targets(out, combined_constraint, desired)
        out[~mask] = 0.0
    return out.astype(np.complex64)


def regional_lcmv_warm_start(
    constraints: list[Any],
    mask: np.ndarray,
    *,
    nearest_weight: float = 500.0,
    local_weight: float = 3.0,
) -> np.ndarray:
    """Build a minimum-leakage EEP warm start before nonlinear active-RL projection."""
    active = np.flatnonzero(mask)
    tasks = np.zeros((mask.size, len(constraints)), dtype=np.complex128)
    identity = np.eye(active.size, dtype=np.complex128)
    for task_index, constraint in enumerate(constraints):
        rows = np.asarray(constraint.leakage_rows, dtype=np.complex128)
        covariance = identity.copy()
        if rows.shape[0]:
            normalized = rows / np.sqrt(
                np.maximum(np.asarray(constraint.leakage_row_norm_sq)[:, None], EPS)
            )
            weights = np.where(
                np.asarray(constraint.leakage_kind, dtype=int) == 0,
                float(nearest_weight),
                float(local_weight),
            )
            covariance += normalized.conj().T @ (weights[:, None] * normalized)
        steering = np.conjugate(
            np.asarray(constraint.equality_row, dtype=np.complex128)
        )
        solution = np.linalg.solve(covariance + 1.0e-7 * identity, steering)
        response = complex(constraint.equality_row @ solution)
        if abs(response) <= EPS:
            continue
        solution *= complex(constraint.desired) / response
        tasks[active, task_index] = solution
    return tasks.astype(np.complex64)


def lcmv_active_aware_mask_score(
    steering_tasks: np.ndarray,
    targets: np.ndarray,
    grid_dirs: np.ndarray,
    operator: DenseExternalEEP,
    s_matrix: np.ndarray,
) -> np.ndarray:
    """Score ports from a full-aperture regional LCMV solution and S256 reflection."""
    full_mask = np.ones(steering_tasks.shape[0], dtype=bool)
    constraints, _combined, _stats = build_constraints(
        steering_tasks,
        full_mask,
        targets,
        grid_dirs,
        operator,
        local_radius_deg=5.0,
        nearest_isolation_db=25.0,
        local_isolation_db=20.0,
    )
    lcmv = regional_lcmv_warm_start(constraints, full_mask)
    sources = [np.sum(lcmv, axis=1), *[lcmv[:, index] for index in range(lcmv.shape[1])]]
    power = np.sum(np.abs(lcmv) ** 2, axis=1)
    reflection_penalty = np.zeros(lcmv.shape[0], dtype=np.float64)
    for source in sources:
        reflected = s_matrix @ source
        floor = 0.05 * max(float(np.max(np.abs(source))), EPS)
        reflection_penalty = np.maximum(
            reflection_penalty,
            np.abs(reflected) / np.maximum(np.abs(source), floor),
        )
    power /= max(float(np.max(power)), EPS)
    reflection_penalty = np.minimum(reflection_penalty, 10.0)
    return power / (1.0 + 3.0 * reflection_penalty)


def physical_margins(
    metrics: dict[str, float],
    reference: dict[str, float],
    active: dict[str, float | int],
) -> np.ndarray:
    mainlobe = min(
        float(metrics["weakest_target_gain_db"])
        - (float(reference["weakest_target_gain_db"]) - 0.5),
        3.0 - float(metrics["target_spread_db"]),
        1.5 - float(metrics["pointing_error_deg"]),
    )
    return np.asarray(
        [
            -float(metrics["psll_db"]),
            float(metrics["nearest_iso_db"]) - 25.0,
            float(metrics["local_iso_db"]) - 20.0,
            mainlobe,
            float(active["active_rl_floor_db"]) - 10.0,
        ],
        dtype=np.float32,
    )


def metric_vector(metrics: dict[str, float]) -> np.ndarray:
    return np.asarray([float(metrics[name]) for name in METRIC_NAMES], dtype=np.float32)


def choose_implementation(
    nominal_tasks: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    reference: dict[str, float],
    nominal_margins: np.ndarray,
    fast: FastPatternEvaluator,
    s_matrix: np.ndarray,
    role: str,
    phase_bits: int,
    amplitude_bits: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, float | int], np.ndarray, int]:
    score = np.sum(np.abs(nominal_tasks) ** 2, axis=1)
    trials = [
        quantized_perturbation(
            nominal_tasks,
            mask,
            score,
            phase_rms_deg=phase,
            gain_rms_db=gain,
            dropout_count=dropout,
            phase_bits=phase_bits,
            amplitude_bits=amplitude_bits,
            seed=seed + profile_index * 1009,
        )
        for profile_index, (phase, gain, dropout) in enumerate(PERTURBATION_PROFILES)
    ]
    pattern_batch = fast.evaluate(np.stack(trials), targets)
    metrics = [metric_at(pattern_batch, index) for index in range(len(trials))]
    active = [full_active_metrics(value, mask, s_matrix) for value in trials]
    margins = np.stack(
        [physical_margins(m, reference, a) for m, a in zip(metrics, active)]
    )
    strict = np.all(margins >= 0.0, axis=1)
    nominal_strict = bool(np.all(nominal_margins >= 0.0))
    min_margin = np.min(margins, axis=1)
    main_penalty = 4.0 * np.maximum(-margins[:, 3], 0.0)
    if role == "hard_positive":
        eligible = np.flatnonzero(strict)
        selected = int(eligible[np.argmax(min_margin[eligible])]) if eligible.size else int(np.argmax(min_margin))
    elif role == "hard_negative" and nominal_strict:
        eligible = np.flatnonzero(~strict)
        selected = int(eligible[np.argmin(np.abs(min_margin[eligible]) + main_penalty[eligible])]) if eligible.size else int(np.argmin(np.abs(min_margin)))
    elif role == "near_boundary":
        selected = int(np.argmin(np.abs(min_margin) + main_penalty))
    else:
        selected = min(4, len(trials) - 1)
    return trials[selected], metrics[selected], active[selected], margins[selected], selected


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite candidate pool: {args.out_dir}")
    ratios = [float(value.strip()) for value in args.ratios.split(",") if value.strip()]
    if ratios != sorted(ratios) or any(value <= 0.0 or value >= 1.0 for value in ratios):
        raise ValueError("ratios must be ordered sparse values strictly between 0 and 1")
    if int(args.masks_per_ratio) < 4:
        raise ValueError("masks-per-ratio must be at least 4")
    k_counts: dict[int, int] | None = None
    if str(args.k_counts).strip():
        values = [int(value.strip()) for value in str(args.k_counts).split(",")]
        if len(values) != 3 or any(value < 0 for value in values):
            raise ValueError("k-counts must contain three non-negative K=2,4,6 counts")
        k_counts = dict(zip((2, 4, 6), values))
        if sum(values) < 3:
            raise ValueError("k-counts must request at least three scenes")
    targeted_joint_search = bool(
        k_counts is not None
        or float(args.targeted_hard_fraction) > 0.0
        or args.scene_source is not None
        or args.rescue_dataset is not None
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    base = load_package(args.base_dir / "dataset_arrays.npz")
    operator = load_package(args.operator)
    excitations = load_package(args.excitations)
    antenna_map = np.asarray(excitations["antenna_wave_map"], dtype=np.complex64)
    s_matrix = np.asarray(excitations["matched_s"], dtype=np.complex128)
    expected_s = np.asarray(operator["s_matched"], dtype=np.complex128)
    if float(np.max(np.abs(s_matrix - expected_s))) > 1.0e-7:
        raise RuntimeError("S256 mismatch between operator and excitation package")
    theta = np.asarray(operator["theta_deg"], dtype=np.float64)
    phi = np.asarray(operator["phi_deg"], dtype=np.float64)
    grid_dirs = pattern_grid_dirs(theta, phi)
    effective = DenseExternalEEP(operator["etheta"], operator["ephi"], antenna_map)
    fast = FastPatternEvaluator(effective, theta, phi)
    positions = np.asarray(base["positions_lambda"], dtype=np.float64)
    element_ixiy = np.asarray(base["element_ixiy"], dtype=np.int64)
    internal = (
        np.asarray(base["task_weights_real_imag"][..., 0], dtype=np.float64)
        + 1j * np.asarray(base["task_weights_real_imag"][..., 1], dtype=np.float64)
    )

    exclusion_paths = [
        args.base_dir / "dataset_arrays.npz",
        *DEFAULT_EXCLUDES,
        *args.exclude,
        *args.donor_dataset,
    ]
    used_hashes = excluded_target_hashes(exclusion_paths)
    if args.scene_source is not None:
        scenes = load_source_scenes(
            args.scene_source,
            base,
            only_failed=bool(args.only_failed_scenes),
        )
    elif k_counts is not None or float(args.targeted_hard_fraction) > 0.0:
        if k_counts is None:
            per_k = [int(args.scene_count) // 3] * 3
            for index in range(int(args.scene_count) % 3):
                per_k[index] += 1
            k_counts = dict(zip((2, 4, 6), per_k))
        scenes = generate_targeted_scenes(
            base,
            k_counts,
            used_hashes,
            hard_fraction=float(args.targeted_hard_fraction),
            small_separation_deg=float(args.small_separation_deg),
            large_scan_deg=float(args.large_scan_deg),
            sample_index_start=int(args.sample_index_start),
            seed=int(args.seed),
        )
    else:
        scenes = generate_scenes(base, int(args.scene_count), used_hashes)
    if int(args.max_scenes) > 0:
        scenes = scenes[: int(args.max_scenes)]
    donor_pool = load_donor_pool(list(args.donor_dataset))
    rescue_pool = load_rescue_pool(args.rescue_dataset)

    records: list[dict[str, Any]] = []
    tasks_nominal: list[np.ndarray] = []
    tasks_actual: list[np.ndarray] = []
    masks_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    valid_all: list[np.ndarray] = []
    references: list[np.ndarray] = []
    nominal_metrics_all: list[np.ndarray] = []
    actual_metrics_all: list[np.ndarray] = []
    nominal_margins_all: list[np.ndarray] = []
    actual_margins_all: list[np.ndarray] = []
    scene_rows: list[dict[str, Any]] = []
    candidate_index = 0
    for scene_position, scene in enumerate(scenes):
        parent = int(scene["parent"])
        k_value = int(scene["k_value"])
        parent_mask = np.asarray(base["masks"][parent], dtype=bool)
        parent_command = np.conjugate(internal[parent, :, :k_value]).astype(np.complex64)
        parent_command[~parent_mask] = 0.0
        migrated = phase_migrate(
            parent_command,
            np.asarray(scene["old_targets"], dtype=float),
            np.asarray(scene["targets"], dtype=float),
            positions,
        )
        steering = matched_steering_tasks(
            migrated, np.asarray(scene["targets"], dtype=float), effective, grid_dirs
        )
        score = np.sum(np.abs(steering) ** 2, axis=1)
        mask_score_kind = "matched_steering_power"
        if targeted_joint_search and (k_value == 6 or int(scene["large_scan"]) == 1):
            active_aware = lcmv_active_aware_mask_score(
                steering,
                np.asarray(scene["targets"], dtype=float),
                grid_dirs,
                effective,
                s_matrix,
            )
            steering_score = score / max(float(np.max(score)), EPS)
            score = 0.25 * steering_score + 0.75 * (
                active_aware / max(float(np.max(active_aware)), EPS)
            )
            mask_score_kind = "regional_lcmv_s256_active_aware"
        scene_candidate_start = candidate_index
        continuation_masks: list[np.ndarray] = []
        ratio_iteration = list(reversed(ratios)) if args.continuation_from_high_ratio else ratios
        for ratio_position, ratio in enumerate(ratio_iteration):
            num_active = int(round(NUM_ELEMENTS * ratio))
            rng = np.random.default_rng(
                int(args.seed) + int(scene["sample_index"]) * 97 + ratio_position * 10007
            )
            donors = nearest_donor_masks(
                donor_pool,
                k_value=k_value,
                ratio=ratio,
                min_separation=float(scene["min_separation"]),
                max_theta=float(scene["max_theta"]),
            )
            rescue_masks = best_rescue_masks(
                rescue_pool,
                sample_index=int(scene["sample_index"]),
                ratio=ratio,
            )
            masks, mask_names = structured_masks(
                parent_mask,
                score,
                element_ixiy,
                num_active,
                int(args.masks_per_ratio),
                rng,
                donor_masks=donors,
                continuation_masks=continuation_masks,
                priority_masks=rescue_masks,
            )
            ratio_seed_candidates: list[tuple[np.ndarray, np.ndarray]] = []
            for mask_position, (mask, mask_name) in enumerate(zip(masks, mask_names)):
                warm = np.asarray(steering, dtype=np.complex128).copy()
                warm[~mask] = 0.0
                for task_index in range(k_value):
                    target_norm = max(float(np.linalg.norm(migrated[:, task_index])), EPS)
                    warm[:, task_index] *= target_norm / max(
                        float(np.linalg.norm(warm[:, task_index])), EPS
                    )
                reference = metric_at(fast.evaluate(warm[None].astype(np.complex64), scene["targets"]), 0)
                constraints, combined_constraint, _point_stats = build_constraints(
                    warm,
                    mask,
                    np.asarray(scene["targets"], dtype=float),
                    grid_dirs,
                    effective,
                    local_radius_deg=5.0,
                    nearest_isolation_db=25.0,
                    local_isolation_db=20.0,
                )
                optimizer_warm = warm
                warm_start_kind = "matched_steering"
                if targeted_joint_search and (
                    k_value == 6 or int(scene["large_scan"]) == 1
                ):
                    optimizer_warm = regional_lcmv_warm_start(constraints, mask)
                    warm_start_kind = "regional_eep_lcmv"
                optimizer_config = FAST_CONFIG
                if (k_value == 6 or int(scene["large_scan"]) == 1) and int(
                    args.k6_cycles
                ) != FAST_CONFIG.cycles:
                    optimizer_config = DenseConfig(
                        "targeted_k6_dense",
                        True,
                        int(args.k6_cycles),
                        max(4, FAST_CONFIG.active_steps),
                        min(FAST_CONFIG.step_size, 0.014),
                        max(FAST_CONFIG.combined_penalty, 56.0),
                        max(FAST_CONFIG.task_penalty, 16.0),
                        max(FAST_CONFIG.total_penalty, 5.0),
                        FAST_CONFIG.amplitude_floor_db,
                        max(2, FAST_CONFIG.dense_passes),
                        max(96, FAST_CONFIG.dense_top_count),
                        max(3.0, FAST_CONFIG.projection_margin_db),
                    )
                refined, optimizer_active = refine_one(
                    optimizer_warm,
                    optimizer_warm,
                    mask,
                    constraints,
                    combined_constraint,
                    s_matrix,
                    optimizer_config,
                    # A 2 dB design reserve is needed for quantization and
                    # implementation perturbations before applying the 10 dB gate.
                    rl_min_db=12.0,
                    task_relative_db=-20.0,
                )
                peak_trials: list[tuple[np.ndarray, float]] = [(refined, 0.0)]
                if float(scene["min_separation"]) <= 10.5:
                    for shape_margin in (0.05, 0.15, 0.30):
                        peak_trials.append(
                            (
                                project_combined_local_peak(
                                    refined,
                                    mask,
                                    np.asarray(scene["targets"], dtype=float),
                                    grid_dirs,
                                    effective,
                                    constraints,
                                    combined_constraint,
                                    optimizer_config,
                                    shape_margin_db=shape_margin,
                                ),
                                shape_margin,
                            )
                        )
                peak_metrics = [
                    metric_at(fast.evaluate(value[None], scene["targets"]), 0)
                    for value, _shape_margin in peak_trials
                ]
                peak_active = [
                    full_active_metrics(value, mask, s_matrix)
                    for value, _shape_margin in peak_trials
                ]
                peak_margins = [
                    physical_margins(metrics, metrics, active_values)
                    for metrics, active_values in zip(peak_metrics, peak_active)
                ]
                peak_order = sorted(
                    range(len(peak_trials)),
                    key=lambda index: (
                        int(np.all(peak_margins[index] >= 0.0)),
                        int(peak_margins[index][3] >= 0.0),
                        float(np.min(peak_margins[index])),
                        -float(np.maximum(-peak_margins[index], 0.0).sum()),
                    ),
                    reverse=True,
                )
                peak_index = int(peak_order[0])
                refined, peak_shape_margin_db = peak_trials[peak_index]
                nominal_metrics = peak_metrics[peak_index]
                nominal_active = peak_active[peak_index]
                # Residual mainlobe labels compare the implemented/HFSS result
                # with this candidate's nominal EEP result.  The warm-start
                # comparison is retained separately as an optimizer diagnostic.
                nominal_margins = physical_margins(
                    nominal_metrics, nominal_metrics, nominal_active
                )
                if mask_name in ("power_top", "parent_normalized") or mask_name.startswith(
                    (
                        "hard_positive_",
                        "ratio_continuation_",
                        "continuation_swap_",
                        "failed_best_swap_",
                    )
                ):
                    role = "hard_positive"
                elif mask_name in (
                    "ring_quadrant",
                    "spread_090",
                    "min_spacing_150",
                    "edge_density",
                    "center_density",
                    "checker_balance",
                ):
                    role = "near_boundary"
                elif mask_name == "mirrored_pair":
                    role = "hard_negative"
                else:
                    role = "intermediate"
                actual, actual_metrics, actual_active, actual_margins, profile_index = choose_implementation(
                    refined,
                    mask,
                    np.asarray(scene["targets"], dtype=float),
                    nominal_metrics,
                    nominal_margins,
                    fast,
                    s_matrix,
                    role,
                    int(args.phase_bits),
                    int(args.amplitude_bits),
                    int(args.seed) + candidate_index * 7919,
                )
                nominal_strict = int(np.all(nominal_margins >= 0.0))
                actual_strict = int(np.all(actual_margins >= 0.0))
                actual_gate15 = int(
                    actual_margins[0] >= 0.0
                    and actual_margins[1] >= 0.0
                    and actual_margins[2] >= -5.0
                )
                delta = actual - refined
                phase_error, gain_error, dropout = PERTURBATION_PROFILES[profile_index]
                padded_targets = np.full((KMAX, 2), np.nan, dtype=np.float32)
                padded_targets[:k_value] = np.asarray(scene["targets"], dtype=np.float32)
                task_valid = np.zeros(KMAX, dtype=np.int8)
                task_valid[:k_value] = 1
                padded_nominal = np.zeros((NUM_ELEMENTS, KMAX), dtype=np.complex64)
                padded_actual = np.zeros((NUM_ELEMENTS, KMAX), dtype=np.complex64)
                padded_nominal[:, :k_value] = refined
                padded_actual[:, :k_value] = actual
                near_boundary = int(float(np.min(np.abs(actual_margins))) <= 1.5)
                hard_negative = int(nominal_strict == 1 and actual_strict == 0)
                hard_positive = int(actual_strict == 1)
                ratio_seed_candidates.append((mask.copy(), actual_margins.copy()))
                record = {
                    "candidate_index": candidate_index,
                    "sample_index": int(scene["sample_index"]),
                    "scene_id": scene["scene_id"],
                    "target_hash": scene["target_hash"],
                    "split_id": int(scene["split_id"]),
                    "parent_candidate_index": parent,
                    "source_sample_index": int(base["source_sample_indices"][parent]),
                    "k_value": k_value,
                    "ratio_requested": ratio,
                    "ratio_actual": float(mask.mean()),
                    "num_active": int(mask.sum()),
                    "mask_family": mask_name,
                    "proposal_role": role,
                    "mask_score_kind": mask_score_kind,
                    "min_target_separation_deg": float(scene["min_separation"]),
                    "max_target_theta_deg": float(scene["max_theta"]),
                    "large_scan": int(scene["large_scan"]),
                    "small_target_gap": int(scene.get("small_gap", 0)),
                    "nominal_gate15": int(
                        nominal_margins[0] >= 0.0
                        and nominal_margins[1] >= 0.0
                        and nominal_margins[2] >= -5.0
                    ),
                    "nominal_strict_gate20": nominal_strict,
                    "actual_gate15": actual_gate15,
                    "actual_strict_gate20": actual_strict,
                    "actual_mainlobe_gate": int(actual_margins[3] >= 0.0),
                    "actual_active_rl_gate": int(actual_margins[4] >= 0.0),
                    "near_boundary": near_boundary,
                    "hard_negative": hard_negative,
                    "hard_positive": hard_positive,
                    "strict_violation": float(np.maximum(-actual_margins, 0.0).sum()),
                    "worst_actual_margin_db": float(np.min(actual_margins)),
                    "phase_error_rms_deg": phase_error,
                    "gain_error_rms_db": gain_error,
                    "dropout_count": dropout,
                    "phase_bits": int(args.phase_bits),
                    "amplitude_bits": int(args.amplitude_bits),
                    "perturbation_profile_index": profile_index,
                    "implementation_delta_norm": float(
                        np.linalg.norm(delta) / max(float(np.linalg.norm(refined)), EPS)
                    ),
                    "implementation_delta_max": float(
                        np.max(np.abs(delta)) / max(float(np.max(np.abs(refined))), EPS)
                    ),
                    "optimizer_best_cycle": int(optimizer_active["best_cycle"]),
                    "warm_start_kind": warm_start_kind,
                    "peak_shape_margin_db": float(peak_shape_margin_db),
                    "optimizer_mainlobe_margin_db": float(
                        physical_margins(nominal_metrics, reference, nominal_active)[3]
                    ),
                }
                for name, value in zip(MARGIN_NAMES, nominal_margins):
                    record[f"nominal_margin_{name}_db"] = float(value)
                for name, value in zip(MARGIN_NAMES, actual_margins):
                    record[f"actual_margin_{name}_db"] = float(value)
                    record[f"residual_margin_{name}_db"] = float(
                        value - nominal_margins[list(MARGIN_NAMES).index(name)]
                    )
                for prefix, active_values in (("nominal", nominal_active), ("actual", actual_active)):
                    for name, value in active_values.items():
                        record[f"{prefix}_{name}"] = value
                records.append(record)
                tasks_nominal.append(padded_nominal)
                tasks_actual.append(padded_actual)
                masks_all.append(mask.astype(np.int8))
                targets_all.append(padded_targets)
                valid_all.append(task_valid)
                references.append(metric_vector(reference))
                nominal_metrics_all.append(metric_vector(nominal_metrics))
                actual_metrics_all.append(metric_vector(actual_metrics))
                nominal_margins_all.append(nominal_margins)
                actual_margins_all.append(actual_margins)
                candidate_index += 1
            if args.continuation_from_high_ratio and ratio_seed_candidates:
                ratio_seed_candidates.sort(
                    key=lambda item: (
                        int(np.all(item[1] >= 0.0)),
                        float(np.min(item[1])),
                        -float(np.maximum(-item[1], 0.0).sum()),
                    ),
                    reverse=True,
                )
                continuation_masks = [item[0] for item in ratio_seed_candidates[:4]]
        scene_rows.append(
            {
                "sample_index": int(scene["sample_index"]),
                "scene_id": scene["scene_id"],
                "target_hash": scene["target_hash"],
                "split_id": int(scene["split_id"]),
                "k_value": k_value,
                "parent_candidate_index": parent,
                "candidate_start": scene_candidate_start,
                "candidate_count": candidate_index - scene_candidate_start,
                "min_target_separation_deg": float(scene["min_separation"]),
                "max_target_theta_deg": float(scene["max_theta"]),
                "large_scan": int(scene["large_scan"]),
                "small_target_gap": int(scene.get("small_gap", 0)),
            }
        )
        print(
            f"scene {scene_position + 1:03d}/{len(scenes):03d} "
            f"K={k_value} candidates={candidate_index} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    nominal = np.stack(tasks_nominal).astype(np.complex64)
    actual = np.stack(tasks_actual).astype(np.complex64)
    masks_array = np.stack(masks_all).astype(np.int8)
    targets_array = np.stack(targets_all).astype(np.float32)
    valid_array = np.stack(valid_all).astype(np.int8)
    combined_nominal = np.sum(nominal, axis=2)
    combined_actual = np.sum(actual, axis=2)
    nominal_internal = np.conjugate(nominal)
    actual_internal = np.conjugate(actual)
    combined_nominal_internal = np.conjugate(combined_nominal)
    combined_actual_internal = np.conjugate(combined_actual)
    sample_index = np.asarray([row["sample_index"] for row in records], dtype=np.int64)
    payload = {
        "candidate_index": np.arange(candidate_index, dtype=np.int64),
        "candidate_indices": np.arange(candidate_index, dtype=np.int64),
        "sample_index": sample_index,
        "sample_indices": sample_index,
        "sample_ids": np.asarray([row["scene_id"] for row in records]),
        "scene_ids": np.asarray([row["scene_id"] for row in records]),
        "target_hashes": np.asarray([row["target_hash"] for row in records]),
        "source_dataset": np.asarray([str(args.dataset_version)] * candidate_index),
        "source_sample_indices": np.asarray([row["source_sample_index"] for row in records], dtype=np.int64),
        "selection_roles": np.asarray([row["proposal_role"] for row in records]),
        "variant_kind": np.asarray([row["mask_family"] for row in records]),
        "split_id": np.asarray([row["split_id"] for row in records], dtype=np.int8),
        "k_values": np.asarray([row["k_value"] for row in records], dtype=np.int8),
        "active_ratios_requested": np.asarray([row["ratio_requested"] for row in records], dtype=np.float32),
        "active_ratios_actual": np.asarray([row["ratio_actual"] for row in records], dtype=np.float32),
        "num_active": np.asarray([row["num_active"] for row in records], dtype=np.int16),
        "targets_deg": targets_array,
        "task_valid": valid_array,
        "mask": masks_array,
        "masks": masks_array,
        "w_tasks_real_imag": complex_to_ri(nominal),
        "task_weights_real_imag": complex_to_ri(nominal_internal),
        "w_combined_real_imag": complex_to_ri(combined_nominal),
        "combined_weights_real_imag": complex_to_ri(combined_nominal_internal),
        "hfss_actual_task_weights_real_imag": complex_to_ri(actual_internal),
        "hfss_actual_combined_weights_real_imag": complex_to_ri(combined_actual_internal),
        "hfss_weights_real_imag": complex_to_ri(combined_actual_internal),
        "nominal_external_task_weights_real_imag": complex_to_ri(nominal),
        "actual_external_task_weights_real_imag": complex_to_ri(actual),
        "reference_metrics": np.stack(references).astype(np.float32),
        "nominal_metrics": np.stack(nominal_metrics_all).astype(np.float32),
        "actual_metrics": np.stack(actual_metrics_all).astype(np.float32),
        "metric_names": METRIC_NAMES,
        "nominal_margins": np.stack(nominal_margins_all).astype(np.float32),
        "actual_margins": np.stack(actual_margins_all).astype(np.float32),
        "margin_residuals": (
            np.stack(actual_margins_all) - np.stack(nominal_margins_all)
        ).astype(np.float32),
        "margin_names": MARGIN_NAMES,
        "gate15": np.asarray([row["actual_gate15"] for row in records], dtype=np.int8),
        "strict_gate20": np.asarray([row["actual_strict_gate20"] for row in records], dtype=np.int8),
        "mainlobe_gate": np.asarray([row["actual_mainlobe_gate"] for row in records], dtype=np.int8),
        "active_rl_gate": np.asarray([row["actual_active_rl_gate"] for row in records], dtype=np.int8),
        "near_boundary": np.asarray([row["near_boundary"] for row in records], dtype=np.int8),
        "hard_negative": np.asarray([row["hard_negative"] for row in records], dtype=np.int8),
        "hard_positive": np.asarray([row["hard_positive"] for row in records], dtype=np.int8),
        "strict_violation": np.asarray([row["strict_violation"] for row in records], dtype=np.float32),
        "min_target_separation_deg": np.asarray([row["min_target_separation_deg"] for row in records], dtype=np.float32),
        "max_target_theta_deg": np.asarray([row["max_target_theta_deg"] for row in records], dtype=np.float32),
        "large_scan": np.asarray([row["large_scan"] for row in records], dtype=np.int8),
        "small_target_gap": np.asarray(
            [row["small_target_gap"] for row in records], dtype=np.int8
        ),
        "implementation_delta_norm": np.asarray([row["implementation_delta_norm"] for row in records], dtype=np.float32),
        "implementation_delta_max": np.asarray([row["implementation_delta_max"] for row in records], dtype=np.float32),
        "phase_error_rms_deg": np.asarray([row["phase_error_rms_deg"] for row in records], dtype=np.float32),
        "gain_error_rms_db": np.asarray([row["gain_error_rms_db"] for row in records], dtype=np.float32),
        "dropout_count": np.asarray([row["dropout_count"] for row in records], dtype=np.int16),
        "peak_shape_margin_db": np.asarray(
            [row["peak_shape_margin_db"] for row in records], dtype=np.float32
        ),
        "phase_bits": np.asarray([row["phase_bits"] for row in records], dtype=np.int16),
        "amplitude_bits": np.asarray([row["amplitude_bits"] for row in records], dtype=np.int16),
        "port_names": np.asarray(operator["port_names"]),
        "element_ixiy": element_ixiy,
        "positions_lambda": positions,
    }
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **payload)
    write_csv(args.out_dir / "candidate_manifest.csv", records)
    write_csv(args.out_dir / "scene_manifest.csv", scene_rows)
    summary = {
        "dataset_version": str(args.dataset_version),
        "candidate_count": candidate_index,
        "independent_scene_count": len(scenes),
        "ratios": ratios,
        "masks_per_ratio": int(args.masks_per_ratio),
        "k_scene_counts": {
            str(k_value): sum(int(row["k_value"]) == k_value for row in scene_rows)
            for k_value in (2, 4, 6)
        },
        "targeted_hard_fraction_requested": float(args.targeted_hard_fraction),
        "small_gap_scene_count": sum(int(row["small_target_gap"]) for row in scene_rows),
        "large_scan_scene_count": sum(int(row["large_scan"]) for row in scene_rows),
        "continuation_from_high_ratio": bool(args.continuation_from_high_ratio),
        "donor_dataset_count": len(args.donor_dataset),
        "scene_source": str(args.scene_source.resolve()) if args.scene_source else None,
        "rescue_dataset": str(args.rescue_dataset.resolve()) if args.rescue_dataset else None,
        "only_failed_scenes": bool(args.only_failed_scenes),
        "targeted_joint_search": targeted_joint_search,
        "contains_ratio_1": False,
        "contains_nominal_control": False,
        "split_scene_counts": {
            name: sum(int(row["split_id"]) == split for row in scene_rows)
            for split, name in enumerate(("train", "val", "test"))
        },
        "gate15_count": int(np.sum(payload["gate15"])),
        "strict_gate20_count": int(np.sum(payload["strict_gate20"])),
        "mainlobe_gate_count": int(np.sum(payload["mainlobe_gate"])),
        "active_rl_gate_count": int(np.sum(payload["active_rl_gate"])),
        "near_boundary_count": int(np.sum(payload["near_boundary"])),
        "hard_negative_count": int(np.sum(payload["hard_negative"])),
        "hard_positive_count": int(np.sum(payload["hard_positive"])),
        "excluded_target_hash_count": len(used_hashes) - len(scenes),
        "v08_used_for_training": False,
        "label_scope": "EEP/S256 implementation residual; not HFSS full-wave",
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
