#!/usr/bin/env python3
"""Prospectively evaluate frozen joint-projected commands at physical 10.04 GHz."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import metric_at
from generate_v09_eep_development_candidates import (
    MARGIN_NAMES,
    METRIC_NAMES,
    full_active_metrics,
    metric_vector,
    physical_margins,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from run_v16_robust_drift_oracle import (
    apply_calibration,
    complex_to_ri,
    hardware_margin,
    load_nominal_operator,
    load_npz,
    ri_to_complex,
    scene_calibration_states,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTED = (
    ROOT
    / "hfss_outputs"
    / "v19_nominal_9p96_joint_projection_20260729_run01"
    / "projected_commands.npz"
)
DEFAULT_SOURCE = (
    ROOT
    / "hfss_outputs"
    / "v18_perturbed_operator_frequency_low_evaluation_20260729_run01"
    / "dataset_arrays.npz"
)
DEFAULT_POOL = (
    ROOT
    / "hfss_outputs"
    / "v16_robust_drift_oracle_20260727_run01"
    / "pool"
    / "candidate_pool.npz"
)
DEFAULT_NOMINAL = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_HIGH = (
    ROOT
    / "hfss_outputs"
    / "v19_perturbed_operator_frequency_high_20260729_run01"
    / "operator"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_PROTOCOL = ROOT / "configs" / "v19_nominal_9p96_joint_active_rl_projection.json"
DEFAULT_OUT = (
    ROOT / "hfss_outputs" / "v19_frequency_high_prospective_evaluation_20260729_run01"
)
ROBUST_MARGIN_NAMES = np.concatenate((MARGIN_NAMES, np.asarray(["hardware"])))
CORNER_NAME = "frequency_high_x0.20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected", type=Path, default=DEFAULT_PROJECTED)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--high-operator", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_state(
    command: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    reference: dict[str, float],
    effective: Any,
    fast: Any,
    s_matrix: np.ndarray,
    grid_dirs: np.ndarray,
    gates: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float | int], np.ndarray, dict[str, float]]:
    metrics = metric_at(fast.evaluate(actual, targets), 0)
    active = full_active_metrics(actual, mask, s_matrix)
    physical = physical_margins(metrics, reference, active)
    hardware_value, hardware = hardware_margin(
        command, actual, mask, targets, effective, grid_dirs, gates
    )
    margins = np.concatenate((physical, np.asarray([hardware_value], dtype=np.float32)))
    return metrics, active, margins, hardware


def append_values(
    row: dict[str, Any],
    prefix: str,
    metrics: dict[str, float],
    active: dict[str, float | int],
    margins: np.ndarray,
    hardware: dict[str, float],
) -> None:
    for name in METRIC_NAMES:
        row[f"{prefix}_{name}"] = float(metrics[str(name)])
    row[f"{prefix}_active_rl_floor_db"] = float(active["active_rl_floor_db"])
    for name, value in zip(ROBUST_MARGIN_NAMES, margins):
        row[f"{prefix}_{name}_margin_db"] = float(value)
    for name, value in hardware.items():
        row[f"{prefix}_{name}"] = float(value)
    row[f"{prefix}_strict_pass"] = int(float(np.min(margins)) >= 0.0)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite high-corner evaluation: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    projected = load_npz(args.projected)
    source = load_npz(args.source)
    pool = load_npz(args.pool)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    nominal_base, _nominal_effective, nominal_fast, _nominal_s = load_nominal_operator(
        args.nominal_operator
    )
    high_base, high_effective, high_fast, high_s = load_nominal_operator(args.high_operator)
    if not np.isclose(float(high_base["frequency_ghz"]), 10.04, atol=1.0e-6):
        raise RuntimeError("High operator is not 10.04 GHz")
    if not np.array_equal(nominal_base["element_ixiy"], high_base["element_ixiy"]):
        raise RuntimeError("Nominal and high operator port ordering differs")

    commands = ri_to_complex(projected["selected_task_weights_real_imag"])
    source_indices = np.asarray(projected["source_candidate_index"], dtype=np.int64)
    samples = np.asarray(projected["sample_index"], dtype=np.int64)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    grid_dirs = pattern_grid_dirs(high_base["theta_deg"], high_base["phi_deg"])
    gates = protocol["gates_unchanged"]
    count = len(samples)
    actual_high = np.zeros_like(commands, dtype=np.complex64)
    nominal_metrics_array = np.zeros((count, len(METRIC_NAMES)), dtype=np.float32)
    high_metrics_array = np.zeros_like(nominal_metrics_array)
    high_margins_array = np.zeros((count, len(ROBUST_MARGIN_NAMES)), dtype=np.float32)
    rows: list[dict[str, Any]] = []

    for candidate in range(count):
        source_index = int(source_indices[candidate])
        sample = int(samples[candidate])
        k_value = int(projected["k_values"][candidate])
        mask = np.asarray(projected["masks"][candidate], dtype=bool)
        targets = np.asarray(projected["targets_deg"][candidate, :k_value], dtype=float)
        command = np.asarray(commands[candidate, :, :k_value], dtype=np.complex64)
        original = ri_to_complex(
            source["nominal_external_task_weights_real_imag"][source_index, :, :k_value]
        )
        reference = metric_at(nominal_fast.evaluate(original, targets), 0)
        nominal_selected = metric_at(nominal_fast.evaluate(command, targets), 0)
        nominal_metrics_array[candidate] = metric_vector(nominal_selected)
        pool_scene = np.flatnonzero(pool_samples == sample)
        state = scene_calibration_states(
            pool,
            pool_scene,
            {CORNER_NAME: {"profile": "frequency_high", "level": 0.20}},
            high_base["element_ixiy"],
            int(protocol["seed"]),
        )[CORNER_NAME]
        high_actual = apply_calibration(command, mask, state)
        actual_high[candidate, :, :k_value] = high_actual
        identity_values = evaluate_state(
            command,
            command,
            mask,
            targets,
            reference,
            high_effective,
            high_fast,
            high_s,
            grid_dirs,
            gates,
        )
        source_values = evaluate_state(
            command,
            high_actual,
            mask,
            targets,
            reference,
            high_effective,
            high_fast,
            high_s,
            grid_dirs,
            gates,
        )
        high_metrics_array[candidate] = metric_vector(source_values[0])
        high_margins_array[candidate] = source_values[2]
        root_index = int(np.argmin(source_values[2]))
        row: dict[str, Any] = {
            "candidate_index": candidate,
            "source_candidate_index": source_index,
            "sample_index": sample,
            "k": k_value,
            "ratio": float(projected["active_ratios_requested"][candidate]),
            "variant": str(projected["selected_variants"][candidate]),
            "common_command_hash": str(projected["selected_command_hashes"][candidate]),
            "weights_frozen_before_high_operator_evaluation": 1,
            "thresholds_changed": 0,
            "high_source_root_cause": str(ROBUST_MARGIN_NAMES[root_index]),
        }
        append_values(row, "high_identity", *identity_values)
        append_values(row, "high_E2_source", *source_values)
        rows.append(row)

    write_csv(args.out_dir / "candidate_frequency_high_metrics.csv", rows)
    output = {
        key: (
            value[source_indices]
            if value.ndim >= 1 and value.shape[0] == len(source["candidate_indices"])
            else value
        )
        for key, value in source.items()
    }
    padded_combined = np.sum(commands, axis=2)
    padded_actual_combined = np.sum(actual_high, axis=2)
    command_hfss = np.conjugate(commands)
    actual_hfss = np.conjugate(actual_high)
    output.update(
        {
            "candidate_index": np.arange(count, dtype=np.int64),
            "candidate_indices": np.arange(count, dtype=np.int64),
            "sample_index": samples,
            "sample_indices": samples,
            "task_weights_real_imag": complex_to_ri(command_hfss),
            "w_tasks_real_imag": complex_to_ri(command_hfss),
            "combined_weights_real_imag": complex_to_ri(np.conjugate(padded_combined)),
            "w_combined_real_imag": complex_to_ri(np.conjugate(padded_combined)),
            "nominal_external_task_weights_real_imag": complex_to_ri(commands),
            "actual_external_task_weights_real_imag": complex_to_ri(actual_high),
            "hfss_actual_task_weights_real_imag": complex_to_ri(actual_hfss),
            "hfss_actual_combined_weights_real_imag": complex_to_ri(
                np.conjugate(padded_actual_combined)
            ),
            "hfss_weights_real_imag": complex_to_ri(np.conjugate(padded_actual_combined)),
            "nominal_metrics": nominal_metrics_array,
            "actual_metrics": high_metrics_array,
            "actual_margins": high_margins_array[:, : len(MARGIN_NAMES)],
            "gate15": (high_margins_array[:, 0] >= 0.0).astype(np.int8),
            "strict_gate20": (
                np.min(high_margins_array[:, :3], axis=1) >= 0.0
            ).astype(np.int8),
            "mainlobe_gate": (high_margins_array[:, 3] >= 0.0).astype(np.int8),
            "active_rl_gate": (high_margins_array[:, 4] >= 0.0).astype(np.int8),
            "hardware_gate": (high_margins_array[:, 5] >= 0.0).astype(np.int8),
            "physical_16x16_operator_corner_included": np.ones(count, dtype=np.int8),
            "physical_operator_frequency_ghz": np.full(count, 10.04, dtype=np.float32),
            "frozen_E2_corner": np.asarray([CORNER_NAME] * count),
            "selected_variants": np.asarray(projected["selected_variants"]),
            "selected_command_hashes": np.asarray(projected["selected_command_hashes"]),
        }
    )
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **output)

    identity_pass = sum(int(row["high_identity_strict_pass"]) for row in rows)
    source_pass = sum(int(row["high_E2_source_strict_pass"]) for row in rows)
    summary = {
        "protocol": "v19-frozen-joint-projection-on-physical-frequency-high-operator",
        "candidate_count": count,
        "independent_scene_count": len(set(samples.tolist())),
        "physical_frequency_ghz": 10.04,
        "common_masks_and_weights_frozen": True,
        "thresholds_changed": False,
        "high_identity_strict_pass_count": identity_pass,
        "high_identity_strict_pass_rate": identity_pass / count,
        "high_source_strict_pass_count": source_pass,
        "high_source_strict_pass_rate": source_pass / count,
        "high_source_pattern_pass_count": sum(
            int(
                min(
                    float(row[f"high_E2_source_{name}_margin_db"])
                    for name in MARGIN_NAMES[:4]
                )
                >= 0.0
            )
            for row in rows
        ),
        "high_source_active_rl_pass_count": sum(
            int(float(row["high_E2_source_active_rl_margin_db"]) >= 0.0) for row in rows
        ),
        "high_source_hardware_pass_count": sum(
            int(float(row["high_E2_source_hardware_margin_db"]) >= 0.0) for row in rows
        ),
        "failure_root_causes": dict(
            Counter(
                row["high_source_root_cause"]
                for row in rows
                if not int(row["high_E2_source_strict_pass"])
            )
        ),
        "minimum_high_source_active_rl_db": min(
            float(row["high_E2_source_active_rl_floor_db"]) for row in rows
        ),
        "critic_training_allowed": False,
        "direct_hfss_smoke_required": True,
    }
    (args.out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
