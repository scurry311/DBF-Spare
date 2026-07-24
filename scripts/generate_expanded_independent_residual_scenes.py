#!/usr/bin/env python3
"""Expand trusted EEP scenes with paired implementation and lower-ratio variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from generate_dense_boundary_hard_negatives import (
    active_metrics,
    gate15,
    gate20,
    mainlobe_gate,
    paired_low_ratio,
    quantized_perturbation,
)
from hfss_task_fullwave_validate import pattern_grid_dirs, unit_vector
from optimize_trusted_eep_s256_joint_weights import pattern_metrics
from refine_trusted_dense_local_eep_joint import (
    CONFIGS,
    DenseExternalEEP,
    build_constraints,
    refine_one,
)
from validate_trusted_eep_hfss_residuals import series_network_map


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
DEFAULT_OUT = ROOT / "hfss_outputs" / "expanded_independent_scenes_20260724_run01"
KMAX = 6
EPS = 1.0e-12

# Grid-aligned changes avoid counting the 1 deg x 2 deg sampling error as pointing failure.
TARGET_VARIANTS = (
    (0.0, 2.0),
    (0.0, -2.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (1.0, 2.0),
    (1.0, -2.0),
    (-1.0, 2.0),
    (-1.0, -2.0),
    (0.0, 4.0),
    (0.0, -4.0),
)

INTERMEDIATE_PROFILES = (
    (4.0, 0.20, 0),
    (6.0, 0.30, 0),
    (8.0, 0.40, 0),
    (10.0, 0.50, 0),
    (12.0, 0.60, 1),
    (15.0, 0.75, 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--excitations", type=Path, default=DEFAULT_EXCITATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenes-per-parent", type=int, default=3)
    parser.add_argument("--mainlobe-failure-count", type=int, default=24)
    parser.add_argument("--low-ratio-pair-count", type=int, default=15)
    parser.add_argument("--low-ratio-prune-fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260724)
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


def angular_separation_deg(targets: np.ndarray) -> float:
    if targets.shape[0] < 2:
        return 180.0
    directions = np.stack([unit_vector(float(t), float(p)) for t, p in targets])
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    distance = np.rad2deg(np.arccos(cosine))
    distance[np.eye(distance.shape[0], dtype=bool)] = np.inf
    return float(np.min(distance))


def target_hash(targets: np.ndarray) -> str:
    canonical = np.round(np.asarray(targets, dtype=np.float64), decimals=4)
    return hashlib.sha256(canonical.tobytes()).hexdigest()[:16]


def shifted_targets(old_targets: np.ndarray, dtheta: float, dphi: float) -> np.ndarray:
    theta = np.clip(np.rint(old_targets[:, 0]) + dtheta, 1.0, 75.0)
    phi = (np.rint(old_targets[:, 1] / 2.0) * 2.0 + dphi) % 360.0
    return np.column_stack((theta, phi)).astype(np.float64)


def phase_migrate(
    tasks: np.ndarray,
    old_targets: np.ndarray,
    new_targets: np.ndarray,
    positions_lambda: np.ndarray,
) -> np.ndarray:
    migrated = np.asarray(tasks, dtype=np.complex128).copy()
    for task_index in range(migrated.shape[1]):
        delta_u = unit_vector(*new_targets[task_index]) - unit_vector(*old_targets[task_index])
        migrated[:, task_index] *= np.exp(
            -1j * 2.0 * np.pi * (positions_lambda @ delta_u)
        )
    return migrated.astype(np.complex64)


def strict_metrics(
    metrics: dict[str, float],
    reference: dict[str, float],
    active: dict[str, Any],
) -> bool:
    return bool(
        gate20(metrics)
        and mainlobe_gate(metrics, reference)
        and int(active["combined_active_gate"]) == 1
        and int(active["all_tasks_significant_gate"]) == 1
    )


def optimization_key(
    metrics: dict[str, float],
    reference: dict[str, float],
    active: dict[str, Any],
) -> tuple[int, int, int, int, float, float, float]:
    return (
        int(strict_metrics(metrics, reference, active)),
        int(gate20(metrics)),
        int(mainlobe_gate(metrics, reference)),
        int(active["combined_active_gate"] and active["all_tasks_significant_gate"]),
        -max(float(metrics["pointing_error_deg"]) - 1.5, 0.0),
        float(metrics["local_iso_db"]),
        -float(metrics["psll_db"]),
    )


def optimize_scene(
    warm_tasks: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    etheta: np.ndarray,
    ephi: np.ndarray,
    antenna_map: np.ndarray,
    s_matrix: np.ndarray,
    grid_dirs: np.ndarray,
    dense_operator: DenseExternalEEP,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any], dict[str, float]]:
    reference = pattern_metrics(
        warm_tasks, targets, theta, phi, etheta, ephi, antenna_map
    )
    constraints, combined_constraint, _stats = build_constraints(
        warm_tasks,
        mask,
        targets,
        grid_dirs,
        dense_operator,
        local_radius_deg=5.0,
        nearest_isolation_db=25.0,
        local_isolation_db=20.0,
    )
    trials: list[tuple[np.ndarray, dict[str, float], dict[str, Any]]] = []
    for config in CONFIGS:
        refined, active = refine_one(
            warm_tasks,
            warm_tasks,
            mask,
            constraints,
            combined_constraint,
            s_matrix,
            config,
            rl_min_db=10.0,
            task_relative_db=-20.0,
        )
        metrics = pattern_metrics(
            refined, targets, theta, phi, etheta, ephi, antenna_map
        )
        trials.append((refined, metrics, active))
    trials.sort(
        key=lambda item: optimization_key(item[1], reference, item[2]), reverse=True
    )
    refined, metrics, active = trials[0]
    return refined, metrics, active, reference


def residual_intensity(nominal: dict[str, float], actual: dict[str, float]) -> float:
    return float(
        abs(actual["psll_db"] - nominal["psll_db"])
        + 0.25 * abs(actual["nearest_iso_db"] - nominal["nearest_iso_db"])
        + 0.25 * abs(actual["local_iso_db"] - nominal["local_iso_db"])
        + abs(actual["weakest_target_gain_db"] - nominal["weakest_target_gain_db"])
        + 0.5 * abs(actual["pointing_error_deg"] - nominal["pointing_error_deg"])
    )


def select_intermediate_error(
    tasks: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    nominal: dict[str, float],
    score: np.ndarray,
    operator: dict[str, np.ndarray],
    antenna_map: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    trials: list[tuple[tuple[int, float, float], np.ndarray, dict[str, float], dict[str, Any]]] = []
    for profile_index, (phase_rms, gain_rms, dropout_count) in enumerate(
        INTERMEDIATE_PROFILES
    ):
        for repeat in range(4):
            perturb_seed = seed + 97 * profile_index + repeat
            actual = quantized_perturbation(
                tasks,
                mask,
                score,
                phase_rms_deg=phase_rms,
                gain_rms_db=gain_rms,
                dropout_count=dropout_count,
                phase_bits=6,
                amplitude_bits=7,
                seed=perturb_seed,
            )
            metrics = pattern_metrics(
                actual,
                targets,
                operator["theta_deg"],
                operator["phi_deg"],
                operator["etheta"],
                operator["ephi"],
                antenna_map,
            )
            intensity = residual_intensity(nominal, metrics)
            # Prefer neither identity-like nor catastrophic perturbations.
            in_band = int(1.5 <= intensity <= 8.0)
            key = (in_band, -abs(intensity - 4.0), -float(phase_rms))
            trials.append(
                (
                    key,
                    actual,
                    metrics,
                    {
                        "phase_error_rms_deg": phase_rms,
                        "gain_error_rms_db": gain_rms,
                        "dropout_count": dropout_count,
                        "perturbation_seed": perturb_seed,
                        "phase_ramp_deg": 0.0,
                        "residual_intensity": intensity,
                    },
                )
            )
    trials.sort(key=lambda item: item[0], reverse=True)
    _key, actual, metrics, descriptor = trials[0]
    return actual, metrics, descriptor


def select_mainlobe_failure(
    tasks: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    nominal: dict[str, float],
    positions_lambda: np.ndarray,
    operator: dict[str, np.ndarray],
    antenna_map: np.ndarray,
    prefer_phi: bool,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    trials: list[tuple[tuple[int, float, float], np.ndarray, dict[str, float], float]] = []
    for offset in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        shifted = np.asarray(targets, dtype=np.float64).copy()
        if prefer_phi:
            shifted[:, 1] = (shifted[:, 1] + offset) % 360.0
        else:
            direction = -1.0 if float(np.max(shifted[:, 0])) >= 70.0 else 1.0
            shifted[:, 0] = np.clip(shifted[:, 0] + direction * offset, 1.0, 75.0)
        actual = phase_migrate(tasks, targets, shifted, positions_lambda)
        actual[~mask] = 0.0
        metrics = pattern_metrics(
            actual,
            targets,
            operator["theta_deg"],
            operator["phi_deg"],
            operator["etheta"],
            operator["ephi"],
            antenna_map,
        )
        failed = int(not mainlobe_gate(metrics, nominal))
        intensity = residual_intensity(nominal, metrics)
        # The first boundary crossing is favored over a catastrophic miss.
        key = (failed, -offset if failed else offset, -abs(intensity - 4.0))
        trials.append((key, actual, metrics, offset))
    trials.sort(key=lambda item: item[0], reverse=True)
    _key, actual, metrics, offset = trials[0]
    if mainlobe_gate(metrics, nominal):
        raise RuntimeError("No targeted mainlobe failure found")
    phase_error = np.angle(actual / np.where(np.abs(tasks) > EPS, tasks, 1.0 + 0j))
    phase_error = phase_error[np.abs(tasks) > EPS]
    return actual, metrics, {
        "phase_error_rms_deg": float(np.rad2deg(np.sqrt(np.mean(phase_error**2)))),
        "gain_error_rms_db": 0.0,
        "dropout_count": 0,
        "perturbation_seed": -1,
        "phase_ramp_deg": float(offset),
        "residual_intensity": residual_intensity(nominal, metrics),
    }


def variant_record(
    *,
    scene: dict[str, Any],
    kind: str,
    command: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    nominal: dict[str, float],
    actual_metrics_value: dict[str, float],
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scene": scene,
        "variant_kind": kind,
        "command": np.asarray(command, dtype=np.complex64),
        "actual": np.asarray(actual, dtype=np.complex64),
        "mask": np.asarray(mask, dtype=bool),
        "nominal_metrics": nominal,
        "actual_metrics": actual_metrics_value,
        **descriptor,
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with np.load(args.base_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        base = {key: source[key] for key in source.files}
    with np.load(args.operator, allow_pickle=False) as source:
        operator = {key: source[key] for key in source.files}
    with np.load(args.excitations, allow_pickle=False) as source:
        antenna_map = np.asarray(source["antenna_wave_map"], dtype=np.complex64)
        s_matrix = np.asarray(source["matched_s"], dtype=np.complex128)
    matched_s, mapped_again, _series_z = series_network_map(
        np.asarray(operator["s_raw"], dtype=np.complex128), 1.0e10
    )
    if np.max(np.abs(matched_s - s_matrix)) > 1.0e-7:
        raise RuntimeError("Matched S in excitation package does not match the EEP operator")
    if np.max(np.abs(mapped_again - antenna_map)) > 1.0e-7:
        raise RuntimeError("Antenna-wave map mismatch")

    theta = np.asarray(operator["theta_deg"], dtype=np.float64)
    phi = np.asarray(operator["phi_deg"], dtype=np.float64)
    grid_dirs = pattern_grid_dirs(theta, phi)
    dense_operator = DenseExternalEEP(
        operator["etheta"], operator["ephi"], antenna_map
    )
    positions_lambda = np.asarray(base["positions_lambda"], dtype=np.float64)
    internal = (
        np.asarray(base["task_weights_real_imag"][..., 0], dtype=np.float64)
        + 1j * np.asarray(base["task_weights_real_imag"][..., 1], dtype=np.float64)
    )

    scenes: list[dict[str, Any]] = []
    used_hashes = {
        target_hash(np.asarray(base["targets_deg"][p, : int(base["k_values"][p])]))
        for p in range(int(base["candidate_indices"].size))
    }
    attempt_rows: list[dict[str, Any]] = []
    for parent in range(int(base["candidate_indices"].size)):
        k_value = int(base["k_values"][parent])
        old_targets = np.asarray(base["targets_deg"][parent, :k_value], dtype=np.float64)
        mask = np.asarray(base["masks"][parent], dtype=bool)
        original = np.conjugate(internal[parent, :, :k_value]).astype(np.complex64)
        original[~mask] = 0.0
        accepted = 0
        for dtheta, dphi in TARGET_VARIANTS:
            targets = shifted_targets(old_targets, dtheta, dphi)
            digest = target_hash(targets)
            if digest in used_hashes or angular_separation_deg(targets) < 5.0:
                continue
            warm = phase_migrate(original, old_targets, targets, positions_lambda)
            warm[~mask] = 0.0
            refined, metrics, active, reference = optimize_scene(
                warm,
                mask,
                targets,
                theta,
                phi,
                operator["etheta"],
                operator["ephi"],
                antenna_map,
                s_matrix,
                grid_dirs,
                dense_operator,
            )
            strict = strict_metrics(metrics, reference, active)
            attempt_rows.append(
                {
                    "parent_candidate_index": parent,
                    "k": k_value,
                    "dtheta_deg": dtheta,
                    "dphi_deg": dphi,
                    "target_hash": digest,
                    "accepted": int(strict),
                    "psll_db": metrics["psll_db"],
                    "nearest_iso_db": metrics["nearest_iso_db"],
                    "local_iso_db": metrics["local_iso_db"],
                    "target_spread_db": metrics["target_spread_db"],
                    "pointing_error_deg": metrics["pointing_error_deg"],
                    "combined_active_rl_db": active["combined_worst_active_rl_db"],
                    "task_active_rl_db": active["all_tasks_significant_worst_active_rl_db"],
                }
            )
            if not strict:
                continue
            scene_number = len(scenes)
            scene = {
                "scene_number": scene_number,
                "sample_index": 100000 + scene_number,
                "scene_id": f"expanded_{100000 + scene_number}_{digest}",
                "target_hash": digest,
                "parent": parent,
                "k": k_value,
                "mask": mask,
                "targets": targets,
                "tasks": refined,
                "nominal_metrics": metrics,
                "active_metrics": active,
                "min_target_separation_deg": angular_separation_deg(targets),
                "max_target_theta_deg": float(np.max(targets[:, 0])),
                "large_scan": int(np.max(targets[:, 0]) >= 45.0),
                "dtheta_deg": dtheta,
                "dphi_deg": dphi,
            }
            scenes.append(scene)
            used_hashes.add(digest)
            accepted += 1
            print(
                f"scene={scene_number + 1:02d} parent={parent:02d} K={k_value} "
                f"shift=({dtheta:+.0f},{dphi:+.0f}) psll={metrics['psll_db']:.2f} "
                f"local={metrics['local_iso_db']:.2f} rl={active['combined_worst_active_rl_db']:.2f}",
                flush=True,
            )
            if accepted >= int(args.scenes_per_parent):
                break
        if accepted < int(args.scenes_per_parent):
            raise RuntimeError(
                f"Parent {parent} produced only {accepted}/{args.scenes_per_parent} strict scenes"
            )

    expected_scenes = int(base["candidate_indices"].size) * int(args.scenes_per_parent)
    if len(scenes) < max(45, expected_scenes):
        raise RuntimeError(f"Only {len(scenes)} independent scenes were generated")

    records: list[dict[str, Any]] = []
    mainlobe_count = min(int(args.mainlobe_failure_count), len(scenes))
    for scene_position, scene in enumerate(scenes):
        tasks = scene["tasks"]
        mask = scene["mask"]
        nominal = scene["nominal_metrics"]
        records.append(
            variant_record(
                scene=scene,
                kind="nominal_control",
                command=tasks,
                actual=tasks,
                mask=mask,
                nominal=nominal,
                actual_metrics_value=nominal,
                descriptor={
                    "phase_error_rms_deg": 0.0,
                    "gain_error_rms_db": 0.0,
                    "dropout_count": 0,
                    "perturbation_seed": -1,
                    "phase_ramp_deg": 0.0,
                    "residual_intensity": 0.0,
                },
            )
        )
        if scene_position < mainlobe_count:
            actual, actual_metrics_value, descriptor = select_mainlobe_failure(
                tasks,
                mask,
                scene["targets"],
                nominal,
                positions_lambda,
                operator,
                antenna_map,
                prefer_phi=bool(scene_position % 2),
            )
            kind = "targeted_mainlobe_failure"
        else:
            score = np.sum(np.abs(tasks) ** 2, axis=1)
            actual, actual_metrics_value, descriptor = select_intermediate_error(
                tasks,
                mask,
                scene["targets"],
                nominal,
                score,
                operator,
                antenna_map,
                int(args.seed) + 1009 * scene_position,
            )
            kind = "intermediate_implementation_error"
        records.append(
            variant_record(
                scene=scene,
                kind=kind,
                command=tasks,
                actual=actual,
                mask=mask,
                nominal=nominal,
                actual_metrics_value=actual_metrics_value,
                descriptor=descriptor,
            )
        )

    low_pair_scenes = sorted(
        scenes,
        key=lambda item: (item["scene_number"] % int(args.scenes_per_parent), item["parent"]),
    )[: int(args.low_ratio_pair_count)]
    for scene in low_pair_scenes:
        selected: tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, Any], dict[str, float]] | None = None
        for fraction in (
            float(args.low_ratio_prune_fraction),
            0.03,
            0.02,
            0.01,
        ):
            low_warm, low_mask = paired_low_ratio(scene["tasks"], scene["mask"], fraction)
            low_tasks, low_metrics, low_active, low_reference = optimize_scene(
                low_warm,
                low_mask,
                scene["targets"],
                theta,
                phi,
                operator["etheta"],
                operator["ephi"],
                antenna_map,
                s_matrix,
                grid_dirs,
                dense_operator,
            )
            selected = (low_tasks, low_mask, low_metrics, low_active, low_reference)
            if strict_metrics(low_metrics, low_reference, low_active):
                break
        assert selected is not None
        low_tasks, low_mask, low_metrics, low_active, low_reference = selected
        records.append(
            variant_record(
                scene=scene,
                kind="paired_lower_ratio_reoptimized",
                command=low_tasks,
                actual=low_tasks,
                mask=low_mask,
                nominal=low_metrics,
                actual_metrics_value=low_metrics,
                descriptor={
                    "phase_error_rms_deg": 0.0,
                    "gain_error_rms_db": 0.0,
                    "dropout_count": int(np.sum(scene["mask"]) - np.sum(low_mask)),
                    "perturbation_seed": -1,
                    "phase_ramp_deg": 0.0,
                    "residual_intensity": 0.0,
                    "low_ratio_strict_gate": int(
                        strict_metrics(low_metrics, low_reference, low_active)
                    ),
                },
            )
        )

    n = len(records)
    masks = np.zeros((n, 256), dtype=np.int8)
    nominal_tasks = np.zeros((n, 256, KMAX), dtype=np.complex64)
    actual_tasks = np.zeros((n, 256, KMAX), dtype=np.complex64)
    targets_padded = np.zeros((n, KMAX, 2), dtype=np.float32)
    task_valid = np.zeros((n, KMAX), dtype=np.int8)
    manifest: list[dict[str, Any]] = []
    for candidate, record in enumerate(records):
        scene = record["scene"]
        k_value = int(scene["k"])
        mask = np.asarray(record["mask"], dtype=bool)
        nominal_tasks[candidate, :, :k_value] = record["command"]
        actual_tasks[candidate, :, :k_value] = record["actual"]
        targets_padded[candidate, :k_value] = scene["targets"]
        task_valid[candidate, :k_value] = 1
        masks[candidate] = mask.astype(np.int8)
        nominal = record["nominal_metrics"]
        actual = record["actual_metrics"]
        actual_rl = active_metrics(record["actual"], mask, s_matrix)
        manifest.append(
            {
                "candidate_index": candidate,
                "sample_index": scene["sample_index"],
                "scene_id": scene["scene_id"],
                "target_hash": scene["target_hash"],
                "parent_candidate_index": scene["parent"],
                "variant_kind": record["variant_kind"],
                "k": k_value,
                "parent_ratio": float(np.mean(scene["mask"])),
                "active_ratio": float(np.mean(mask)),
                "ratio_delta": float(np.mean(scene["mask"]) - np.mean(mask)),
                "active_count": int(np.sum(mask)),
                "phase_error_rms_deg": record["phase_error_rms_deg"],
                "gain_error_rms_db": record["gain_error_rms_db"],
                "dropout_count": record["dropout_count"],
                "phase_ramp_deg": record["phase_ramp_deg"],
                "perturbation_seed": record["perturbation_seed"],
                "residual_intensity": record["residual_intensity"],
                "nominal_eep_gate15": int(gate15(nominal)),
                "nominal_eep_gate20": int(gate20(nominal)),
                "actual_basis_gate15": int(gate15(actual)),
                "actual_basis_gate20": int(gate20(actual)),
                "actual_basis_mainlobe_gate": int(mainlobe_gate(actual, nominal)),
                "predicted_mainlobe_failure": int(not mainlobe_gate(actual, nominal)),
                "nominal_psll_db": nominal["psll_db"],
                "actual_basis_psll_db": actual["psll_db"],
                "nominal_nearest_iso_db": nominal["nearest_iso_db"],
                "actual_basis_nearest_iso_db": actual["nearest_iso_db"],
                "nominal_local_iso_db": nominal["local_iso_db"],
                "actual_basis_local_iso_db": actual["local_iso_db"],
                "nominal_weakest_target_gain_db": nominal["weakest_target_gain_db"],
                "actual_basis_weakest_target_gain_db": actual["weakest_target_gain_db"],
                "actual_basis_pointing_error_deg": actual["pointing_error_deg"],
                **actual_rl,
                "expected_hfss_case_count": 1 + k_value,
            }
        )

    nominal_internal = np.conjugate(nominal_tasks)
    actual_internal = np.conjugate(actual_tasks)
    nominal_combined = np.sum(nominal_internal, axis=2)
    actual_combined = np.sum(actual_internal, axis=2)
    scene_array = np.asarray([record["scene"]["sample_index"] for record in records])
    parent_array = np.asarray([record["scene"]["parent"] for record in records])
    k_array = np.asarray([record["scene"]["k"] for record in records])
    actual_ratio = masks.mean(axis=1).astype(np.float32)
    np.savez_compressed(
        args.out_dir / "dataset_arrays.npz",
        candidate_index=np.arange(n, dtype=np.int64),
        candidate_indices=np.arange(n, dtype=np.int64),
        sample_index=scene_array.astype(np.int64),
        sample_indices=scene_array.astype(np.int64),
        sample_ids=np.asarray([f"expand_c{i:03d}_s{s}" for i, s in enumerate(scene_array)]),
        scene_ids=np.asarray([record["scene"]["scene_id"] for record in records]),
        target_hashes=np.asarray([record["scene"]["target_hash"] for record in records]),
        source_dataset=np.full(n, "expanded_independent_scenes_run01"),
        source_sample_indices=np.asarray(base["source_sample_indices"][parent_array]),
        selection_roles=np.asarray([record["variant_kind"] for record in records]),
        variant_kind=np.asarray([record["variant_kind"] for record in records]),
        parent_candidate_index=parent_array.astype(np.int64),
        parent_ratio=np.asarray([np.mean(record["scene"]["mask"]) for record in records], dtype=np.float32),
        ratio_delta=np.asarray([np.mean(record["scene"]["mask"]) for record in records], dtype=np.float32) - actual_ratio,
        k_values=k_array.astype(np.int64),
        active_ratios_requested=actual_ratio,
        active_ratios_actual=actual_ratio,
        num_active=np.sum(masks, axis=1).astype(np.int64),
        targets_deg=targets_padded,
        task_valid=task_valid,
        mask=masks,
        masks=masks,
        w_tasks_real_imag=np.stack([nominal_internal.real, nominal_internal.imag], axis=-1).astype(np.float32),
        task_weights_real_imag=np.stack([nominal_internal.real, nominal_internal.imag], axis=-1).astype(np.float32),
        w_combined_real_imag=np.stack([nominal_combined.real, nominal_combined.imag], axis=-1).astype(np.float32),
        combined_weights_real_imag=np.stack([nominal_combined.real, nominal_combined.imag], axis=-1).astype(np.float32),
        hfss_actual_task_weights_real_imag=np.stack([actual_internal.real, actual_internal.imag], axis=-1).astype(np.float32),
        hfss_actual_combined_weights_real_imag=np.stack([actual_combined.real, actual_combined.imag], axis=-1).astype(np.float32),
        hfss_weights_real_imag=np.stack([actual_combined.real, actual_combined.imag], axis=-1).astype(np.float32),
        min_target_separation_deg=np.asarray([record["scene"]["min_target_separation_deg"] for record in records], dtype=np.float32),
        max_target_theta_deg=np.asarray([record["scene"]["max_target_theta_deg"] for record in records], dtype=np.float32),
        large_scan=np.asarray([record["scene"]["large_scan"] for record in records], dtype=np.int8),
        phase_error_rms_deg=np.asarray([record["phase_error_rms_deg"] for record in records], dtype=np.float32),
        gain_error_rms_db=np.asarray([record["gain_error_rms_db"] for record in records], dtype=np.float32),
        dropout_count=np.asarray([record["dropout_count"] for record in records], dtype=np.int16),
        phase_ramp_deg=np.asarray([record["phase_ramp_deg"] for record in records], dtype=np.float32),
        residual_intensity=np.asarray([record["residual_intensity"] for record in records], dtype=np.float32),
        phase_bits=np.full(n, 6, dtype=np.int16),
        amplitude_bits=np.full(n, 7, dtype=np.int16),
        perturbation_seed=np.asarray([record["perturbation_seed"] for record in records], dtype=np.int64),
        port_names=np.asarray(base["port_names"]),
        element_ixiy=np.asarray(base["element_ixiy"]),
        positions_lambda=positions_lambda,
    )
    write_csv(args.out_dir / "scene_generation_attempts.csv", attempt_rows)
    write_csv(args.out_dir / "candidate_manifest.csv", manifest)

    kind_counts = {
        kind: int(sum(row["variant_kind"] == kind for row in manifest))
        for kind in sorted({row["variant_kind"] for row in manifest})
    }
    mainlobe_failures = int(sum(int(row["predicted_mainlobe_failure"]) for row in manifest))
    intermediate_count = int(
        sum(1.5 <= float(row["residual_intensity"]) <= 8.0 for row in manifest)
    )
    expected_cases = int(sum(int(row["expected_hfss_case_count"]) for row in manifest))
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "independent_scene_count": len(scenes),
        "candidate_count": n,
        "expected_hfss_case_count": expected_cases,
        "variant_kind_counts": kind_counts,
        "predicted_mainlobe_failure_count": mainlobe_failures,
        "intermediate_residual_intensity_count": intermediate_count,
        "paired_lower_ratio_count": kind_counts.get("paired_lower_ratio_reoptimized", 0),
        "paired_lower_ratio_strict_count": int(
            sum(
                row["variant_kind"] == "paired_lower_ratio_reoptimized"
                and int(row["nominal_eep_gate20"]) == 1
                and int(row["robust_active_gate"]) == 1
                for row in manifest
            )
        ),
        "k_scene_counts": {
            str(k): int(sum(int(scene["k"]) == k for scene in scenes)) for k in (2, 4, 6)
        },
        "large_scan_scene_count": int(sum(int(scene["large_scan"]) for scene in scenes)),
        "ratio1_included": False,
        "command_actual_mismatch_enabled": True,
        "runtime_seconds": time.time() - started,
        "hfss_gate_pass": bool(
            len(scenes) >= 45
            and mainlobe_failures >= 20
            and kind_counts.get("paired_lower_ratio_reoptimized", 0) >= 15
            and intermediate_count >= 20
        ),
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
