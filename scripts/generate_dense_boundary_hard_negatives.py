#!/usr/bin/env python3
"""Generate paired boundary, lower-ratio, and implementation hard negatives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from optimize_trusted_eep_s256_joint_weights import active_return, pattern_metrics
from validate_trusted_eep_hfss_residuals import series_network_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_dataset_20260724_run01"
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_dense_boundary_dataset_20260724_run01"
KMAX = 6


PROFILES = (
    (1.0, 0.05, 0),
    (2.0, 0.10, 0),
    (3.0, 0.15, 0),
    (4.0, 0.20, 0),
    (5.0, 0.25, 0),
    (6.0, 0.30, 0),
    (3.0, 0.15, 1),
    (5.0, 0.25, 1),
    (8.0, 0.40, 0),
    (3.0, 0.15, 2),
    (6.0, 0.30, 1),
    (5.0, 0.25, 2),
    (10.0, 0.50, 0),
    (8.0, 0.40, 1),
    (6.0, 0.30, 2),
    (12.0, 0.60, 1),
    (10.0, 0.50, 2),
    (8.0, 0.40, 3),
    (12.0, 0.60, 3),
    (15.0, 0.75, 4),
    (18.0, 0.90, 4),
    (22.0, 1.10, 5),
    (28.0, 1.40, 6),
    (35.0, 1.75, 8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--phase-bits", type=int, default=6)
    parser.add_argument("--amplitude-bits", type=int, default=7)
    parser.add_argument("--trials-per-profile", type=int, default=8)
    parser.add_argument("--low-ratio-prune-fraction", type=float, default=0.03)
    parser.add_argument("--low-ratio-per-k", type=int, default=2)
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


def gate15(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["psll_db"] <= 0.0
        and metrics["nearest_iso_db"] >= 25.0
        and metrics["local_iso_db"] >= 15.0
    )


def gate20(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["psll_db"] <= 0.0
        and metrics["nearest_iso_db"] >= 25.0
        and metrics["local_iso_db"] >= 20.0
    )


def mainlobe_gate(metrics: dict[str, float], reference: dict[str, float]) -> bool:
    return bool(
        metrics["weakest_target_gain_db"]
        >= reference["weakest_target_gain_db"] - 0.5
        and metrics["target_spread_db"] <= 3.0
        and metrics["pointing_error_deg"] <= 1.5
    )


def violation(metrics: dict[str, float]) -> float:
    return float(
        max(metrics["psll_db"], 0.0)
        + 0.5 * max(25.0 - metrics["nearest_iso_db"], 0.0)
        + 0.35 * max(15.0 - metrics["local_iso_db"], 0.0)
    )


def quantized_perturbation(
    tasks: np.ndarray,
    mask: np.ndarray,
    score: np.ndarray,
    *,
    phase_rms_deg: float,
    gain_rms_db: float,
    dropout_count: int,
    phase_bits: int,
    amplitude_bits: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gain_error = rng.normal(size=tasks.shape[0]) * gain_rms_db
    phase_error = rng.normal(size=tasks.shape[0]) * phase_rms_deg
    factor = 10.0 ** (gain_error / 20.0) * np.exp(1j * np.deg2rad(phase_error))
    actual = np.asarray(tasks, dtype=np.complex128) * factor[:, None]
    amplitude = np.abs(actual)
    phase = np.angle(actual)
    phase_step = 2.0 * np.pi / float(2 ** max(int(phase_bits), 1))
    phase = np.round(phase / phase_step) * phase_step
    amplitude_levels = max(2 ** max(int(amplitude_bits), 1) - 1, 1)
    for task_index in range(actual.shape[1]):
        maximum = max(float(np.max(amplitude[:, task_index])), 1.0e-12)
        amplitude[:, task_index] = (
            np.round(amplitude[:, task_index] / maximum * amplitude_levels)
            / amplitude_levels
            * maximum
        )
    actual = amplitude * np.exp(1j * phase)
    actual[~mask] = 0.0
    active = np.flatnonzero(mask)
    drop_order = active[np.argsort(score[active])[::-1]]
    actual[drop_order[: int(dropout_count)]] = 0.0
    return actual.astype(np.complex64)


def paired_low_ratio(tasks: np.ndarray, mask: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    score = np.sum(np.abs(tasks) ** 2, axis=1)
    active = np.flatnonzero(mask)
    remove_count = max(1, int(round(active.size * float(fraction))))
    remove = active[np.argsort(score[active])[:remove_count]]
    paired_mask = mask.copy()
    paired_mask[remove] = False
    paired_tasks = np.asarray(tasks, dtype=np.complex64).copy()
    paired_tasks[~paired_mask] = 0.0
    return paired_tasks, paired_mask


def active_metrics(
    tasks: np.ndarray, mask: np.ndarray, s_matrix: np.ndarray
) -> dict[str, float | int]:
    combined = active_return(
        s_matrix, np.sum(tasks, axis=1), mask, relative_db=None, threshold_db=10.0
    )
    task = [
        active_return(
            s_matrix,
            tasks[:, task_index],
            mask,
            relative_db=-20.0,
            threshold_db=10.0,
        )
        for task_index in range(tasks.shape[1])
    ]
    return {
        "combined_worst_active_rl_db": float(combined["worst_active_rl_db"]),
        "combined_total_rl_db": float(combined["total_rl_db"]),
        "task_significant_worst_active_rl_db": min(
            float(item["worst_active_rl_db"]) for item in task
        ),
        "robust_active_gate": int(
            int(combined["gate_pass"]) == 1
            and all(int(item["gate_pass"]) == 1 for item in task)
        ),
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
    s_matrix, antenna_map, _series_z = series_network_map(
        np.asarray(operator["s_raw"], dtype=np.complex128), 1.0e10
    )
    task_internal = (
        base["task_weights_real_imag"][..., 0]
        + 1j * base["task_weights_real_imag"][..., 1]
    )

    candidate_records: list[dict[str, Any]] = []
    hard_negative_records: list[dict[str, Any]] = []
    low_ratio_pool: list[dict[str, Any]] = []
    for parent in range(int(base["candidate_indices"].size)):
        k_value = int(base["k_values"][parent])
        mask = np.asarray(base["masks"][parent], dtype=bool)
        command = np.conjugate(
            np.asarray(task_internal[parent, :, :k_value], dtype=np.complex128)
        ).astype(np.complex64)
        command[~mask] = 0.0
        targets = np.asarray(base["targets_deg"][parent, :k_value], dtype=np.float64)
        nominal_metrics = pattern_metrics(
            command,
            targets,
            operator["theta_deg"],
            operator["phi_deg"],
            operator["etheta"],
            operator["ephi"],
            antenna_map,
        )
        if not gate15(nominal_metrics):
            raise RuntimeError(f"Parent {parent} is not a nominal EEP gate15 positive")
        power_score = np.sum(np.abs(command) ** 2, axis=1)
        crossings: list[tuple[tuple[float, float], dict[str, Any]]] = []
        for profile_index, (phase_rms, gain_rms, dropout_count) in enumerate(PROFILES):
            severity = phase_rms + 4.0 * gain_rms + 2.0 * dropout_count
            for trial in range(int(args.trials_per_profile)):
                seed = int(args.seed) + 1009 * parent + 37 * profile_index + trial
                actual = quantized_perturbation(
                    command,
                    mask,
                    power_score,
                    phase_rms_deg=phase_rms,
                    gain_rms_db=gain_rms,
                    dropout_count=dropout_count,
                    phase_bits=int(args.phase_bits),
                    amplitude_bits=int(args.amplitude_bits),
                    seed=seed,
                )
                actual_metrics = pattern_metrics(
                    actual,
                    targets,
                    operator["theta_deg"],
                    operator["phi_deg"],
                    operator["etheta"],
                    operator["ephi"],
                    antenna_map,
                )
                if gate15(actual_metrics):
                    continue
                crossings.append(
                    (
                        (severity, violation(actual_metrics)),
                        {
                            "command": command,
                            "actual": actual,
                            "mask": mask,
                            "nominal_metrics": nominal_metrics,
                            "actual_metrics": actual_metrics,
                            "phase_error_rms_deg": phase_rms,
                            "gain_error_rms_db": gain_rms,
                            "dropout_count": dropout_count,
                            "perturbation_seed": seed,
                        },
                    )
                )
            if crossings:
                break
        if not crossings:
            raise RuntimeError(f"No implementation hard negative found for parent {parent}")
        _key, selected = min(crossings, key=lambda item: item[0])
        selected.update(
            {
                "parent": parent,
                "variant_kind": "implementation_boundary_hard_negative",
            }
        )
        hard_negative_records.append(selected)

        low_tasks, low_mask = paired_low_ratio(
            command, mask, float(args.low_ratio_prune_fraction)
        )
        low_metrics = pattern_metrics(
            low_tasks,
            targets,
            operator["theta_deg"],
            operator["phi_deg"],
            operator["etheta"],
            operator["ephi"],
            antenna_map,
        )
        low_ratio_pool.append(
            {
                "parent": parent,
                "variant_kind": "paired_lower_ratio",
                "command": low_tasks,
                "actual": low_tasks.copy(),
                "mask": low_mask,
                "nominal_metrics": low_metrics,
                "actual_metrics": low_metrics,
                "phase_error_rms_deg": 0.0,
                "gain_error_rms_db": 0.0,
                "dropout_count": 0,
                "perturbation_seed": -1,
            }
        )

    candidate_records.extend(hard_negative_records)
    for k_value in (2, 4, 6):
        members = [
            record
            for record in low_ratio_pool
            if int(base["k_values"][record["parent"]]) == k_value
        ]
        members.sort(
            key=lambda record: (
                int(gate15(record["nominal_metrics"])),
                -violation(record["nominal_metrics"]),
                -int(record["parent"]),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        passing = [record for record in members if gate15(record["nominal_metrics"])]
        failing = [record for record in members if not gate15(record["nominal_metrics"])]
        if passing:
            selected.append(passing[0])
        if failing and len(selected) < int(args.low_ratio_per_k):
            selected.append(failing[0])
        for record in members:
            if len(selected) >= int(args.low_ratio_per_k):
                break
            if record not in selected:
                selected.append(record)
        candidate_records.extend(selected[: int(args.low_ratio_per_k)])

    n = len(candidate_records)
    candidate_indices = np.arange(n, dtype=np.int64)
    masks = np.zeros((n, 256), dtype=np.int8)
    nominal_tasks = np.zeros((n, 256, KMAX), dtype=np.complex64)
    actual_tasks = np.zeros((n, 256, KMAX), dtype=np.complex64)
    manifest: list[dict[str, Any]] = []
    for candidate, record in enumerate(candidate_records):
        parent = int(record["parent"])
        k_value = int(base["k_values"][parent])
        mask = np.asarray(record["mask"], dtype=bool)
        nominal_tasks[candidate, :, :k_value] = record["command"]
        actual_tasks[candidate, :, :k_value] = record["actual"]
        masks[candidate] = mask.astype(np.int8)
        nominal = record["nominal_metrics"]
        actual = record["actual_metrics"]
        actual_rl = active_metrics(record["actual"], mask, s_matrix)
        manifest.append(
            {
                "candidate_index": candidate,
                "parent_candidate_index": parent,
                "sample_index": int(base["sample_indices"][parent]),
                "variant_kind": record["variant_kind"],
                "k": k_value,
                "parent_ratio": float(base["active_ratios_requested"][parent]),
                "active_ratio": float(np.mean(mask)),
                "ratio_delta": float(np.mean(base["masks"][parent]) - np.mean(mask)),
                "active_count": int(np.sum(mask)),
                "phase_error_rms_deg": float(record["phase_error_rms_deg"]),
                "gain_error_rms_db": float(record["gain_error_rms_db"]),
                "dropout_count": int(record["dropout_count"]),
                "phase_bits": int(args.phase_bits),
                "amplitude_bits": int(args.amplitude_bits),
                "perturbation_seed": int(record["perturbation_seed"]),
                "nominal_eep_gate15": int(gate15(nominal)),
                "nominal_eep_gate20": int(gate20(nominal)),
                "actual_basis_gate15": int(gate15(actual)),
                "actual_basis_gate20": int(gate20(actual)),
                "actual_basis_mainlobe_gate": int(mainlobe_gate(actual, nominal)),
                "predicted_hard_negative": int(gate15(nominal) and not gate15(actual)),
                "nominal_psll_db": float(nominal["psll_db"]),
                "actual_basis_psll_db": float(actual["psll_db"]),
                "nominal_nearest_iso_db": float(nominal["nearest_iso_db"]),
                "actual_basis_nearest_iso_db": float(actual["nearest_iso_db"]),
                "nominal_local_iso_db": float(nominal["local_iso_db"]),
                "actual_basis_local_iso_db": float(actual["local_iso_db"]),
                **actual_rl,
                "expected_hfss_case_count": 1 + k_value,
            }
        )

    parent = np.asarray([int(record["parent"]) for record in candidate_records], dtype=int)
    nominal_internal = np.conjugate(nominal_tasks)
    actual_internal = np.conjugate(actual_tasks)
    nominal_combined = np.sum(nominal_internal, axis=2)
    actual_combined = np.sum(actual_internal, axis=2)
    actual_ratio = masks.mean(axis=1).astype(np.float32)
    np.savez_compressed(
        args.out_dir / "dataset_arrays.npz",
        candidate_index=candidate_indices,
        candidate_indices=candidate_indices,
        sample_index=np.asarray(base["sample_indices"][parent]),
        sample_indices=np.asarray(base["sample_indices"][parent]),
        sample_ids=np.asarray(
            [f"boundary_c{index:03d}_p{p:02d}" for index, p in enumerate(parent)]
        ),
        scene_ids=np.asarray(base["scene_ids"][parent]),
        source_dataset=np.full(n, "dense_boundary_run01"),
        source_sample_indices=np.asarray(base["source_sample_indices"][parent]),
        selection_roles=np.asarray([record["variant_kind"] for record in candidate_records]),
        variant_kind=np.asarray([record["variant_kind"] for record in candidate_records]),
        parent_candidate_index=parent.astype(np.int64),
        parent_ratio=np.asarray(base["active_ratios_requested"][parent], dtype=np.float32),
        ratio_delta=(
            np.asarray(base["masks"][parent], dtype=np.float32).mean(axis=1) - actual_ratio
        ).astype(np.float32),
        k_values=np.asarray(base["k_values"][parent]),
        active_ratios_requested=actual_ratio,
        active_ratios_actual=actual_ratio,
        num_active=np.sum(masks, axis=1).astype(np.int64),
        targets_deg=np.asarray(base["targets_deg"][parent]),
        task_valid=np.asarray(base["task_valid"][parent]),
        mask=masks,
        masks=masks,
        w_tasks_real_imag=np.stack(
            [nominal_internal.real, nominal_internal.imag], axis=-1
        ).astype(np.float32),
        task_weights_real_imag=np.stack(
            [nominal_internal.real, nominal_internal.imag], axis=-1
        ).astype(np.float32),
        w_combined_real_imag=np.stack(
            [nominal_combined.real, nominal_combined.imag], axis=-1
        ).astype(np.float32),
        combined_weights_real_imag=np.stack(
            [nominal_combined.real, nominal_combined.imag], axis=-1
        ).astype(np.float32),
        hfss_actual_task_weights_real_imag=np.stack(
            [actual_internal.real, actual_internal.imag], axis=-1
        ).astype(np.float32),
        hfss_actual_combined_weights_real_imag=np.stack(
            [actual_combined.real, actual_combined.imag], axis=-1
        ).astype(np.float32),
        hfss_weights_real_imag=np.stack(
            [actual_combined.real, actual_combined.imag], axis=-1
        ).astype(np.float32),
        min_target_separation_deg=np.asarray(base["min_target_separation_deg"][parent]),
        max_target_theta_deg=np.asarray(base["max_target_theta_deg"][parent]),
        large_scan=np.asarray(base["large_scan"][parent]),
        phase_error_rms_deg=np.asarray(
            [record["phase_error_rms_deg"] for record in candidate_records], dtype=np.float32
        ),
        gain_error_rms_db=np.asarray(
            [record["gain_error_rms_db"] for record in candidate_records], dtype=np.float32
        ),
        dropout_count=np.asarray(
            [record["dropout_count"] for record in candidate_records], dtype=np.int16
        ),
        phase_bits=np.full(n, int(args.phase_bits), dtype=np.int16),
        amplitude_bits=np.full(n, int(args.amplitude_bits), dtype=np.int16),
        perturbation_seed=np.asarray(
            [record["perturbation_seed"] for record in candidate_records], dtype=np.int64
        ),
        port_names=np.asarray(base["port_names"]),
        element_ixiy=np.asarray(base["element_ixiy"]),
        positions_lambda=np.asarray(base["positions_lambda"]),
    )
    write_csv(args.out_dir / "candidate_manifest.csv", manifest)
    expected_cases = int(sum(int(row["expected_hfss_case_count"]) for row in manifest))
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": n,
        "independent_scene_count": int(np.unique(base["sample_indices"][parent]).size),
        "implementation_boundary_count": len(hard_negative_records),
        "paired_lower_ratio_count": n - len(hard_negative_records),
        "predicted_hard_negative_count": int(
            sum(int(row["predicted_hard_negative"]) for row in manifest)
        ),
        "expected_hfss_case_count": expected_cases,
        "k_counts": {
            str(k): int(sum(int(row["k"]) == k for row in manifest)) for k in (2, 4, 6)
        },
        "ratio1_included": False,
        "command_actual_mismatch_enabled": True,
        "runtime_seconds": time.time() - started,
        "hfss_gate_pass": bool(
            len(hard_negative_records) >= 10
            and 50 <= expected_cases <= 100
            and n - len(hard_negative_records) >= 6
        ),
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
