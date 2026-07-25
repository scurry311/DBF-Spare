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
    nearest_grid_index,
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
    parser.add_argument("--masks-per-ratio", type=int, default=8)
    parser.add_argument("--ratios", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--phase-bits", type=int, default=6)
    parser.add_argument("--amplitude-bits", type=int, default=7)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
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
    for path in packages:
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
) -> tuple[list[np.ndarray], list[str]]:
    seeds = [
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
    ]
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
        candidate = mutate_mask(source, rng, max_swaps=max(8, num_active // 12))
        candidate = normalize_mask(candidate, num_active, rng)
        key = np.packbits(candidate).tobytes()
        mutation_index += 1
        if key in seen:
            continue
        seen.add(key)
        masks.append(candidate)
        names.append(f"local_swap_{mutation_index:02d}")
    return masks[:count], names[:count]


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
        raise FileExistsError(f"Refusing to overwrite v0.9 candidate pool: {args.out_dir}")
    ratios = [float(value.strip()) for value in args.ratios.split(",") if value.strip()]
    if ratios != sorted(ratios) or any(value <= 0.0 or value >= 1.0 for value in ratios):
        raise ValueError("ratios must be ordered sparse values strictly between 0 and 1")
    if int(args.masks_per_ratio) < 4:
        raise ValueError("masks-per-ratio must be at least 4")
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

    exclusion_paths = [args.base_dir / "dataset_arrays.npz", *DEFAULT_EXCLUDES, *args.exclude]
    used_hashes = excluded_target_hashes(exclusion_paths)
    scenes = generate_scenes(base, int(args.scene_count), used_hashes)
    if int(args.max_scenes) > 0:
        scenes = scenes[: int(args.max_scenes)]

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
        scene_candidate_start = candidate_index
        for ratio_position, ratio in enumerate(ratios):
            num_active = int(round(NUM_ELEMENTS * ratio))
            rng = np.random.default_rng(
                int(args.seed) + int(scene["sample_index"]) * 97 + ratio_position * 10007
            )
            masks, mask_names = structured_masks(
                parent_mask,
                score,
                element_ixiy,
                num_active,
                int(args.masks_per_ratio),
                rng,
            )
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
                refined, optimizer_active = refine_one(
                    warm,
                    warm,
                    mask,
                    constraints,
                    combined_constraint,
                    s_matrix,
                    FAST_CONFIG,
                    # A 2 dB design reserve is needed for quantization and
                    # implementation perturbations before applying the 10 dB gate.
                    rl_min_db=12.0,
                    task_relative_db=-20.0,
                )
                nominal_metrics = metric_at(
                    fast.evaluate(refined[None], scene["targets"]), 0
                )
                nominal_active = full_active_metrics(refined, mask, s_matrix)
                # Residual mainlobe labels compare the implemented/HFSS result
                # with this candidate's nominal EEP result.  The warm-start
                # comparison is retained separately as an optimizer diagnostic.
                nominal_margins = physical_margins(
                    nominal_metrics, nominal_metrics, nominal_active
                )
                if mask_name in ("power_top", "parent_normalized"):
                    role = "hard_positive"
                elif mask_name in ("ring_quadrant", "spread_090"):
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
                    "min_target_separation_deg": float(scene["min_separation"]),
                    "max_target_theta_deg": float(scene["max_theta"]),
                    "large_scan": int(scene["large_scan"]),
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
        "source_dataset": np.asarray(["v09_independent_eep_s256"] * candidate_index),
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
        "implementation_delta_norm": np.asarray([row["implementation_delta_norm"] for row in records], dtype=np.float32),
        "implementation_delta_max": np.asarray([row["implementation_delta_max"] for row in records], dtype=np.float32),
        "phase_error_rms_deg": np.asarray([row["phase_error_rms_deg"] for row in records], dtype=np.float32),
        "gain_error_rms_db": np.asarray([row["gain_error_rms_db"] for row in records], dtype=np.float32),
        "dropout_count": np.asarray([row["dropout_count"] for row in records], dtype=np.int16),
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
        "dataset_version": "v0.9-eep-development-pool",
        "candidate_count": candidate_index,
        "independent_scene_count": len(scenes),
        "ratios": ratios,
        "masks_per_ratio": int(args.masks_per_ratio),
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
