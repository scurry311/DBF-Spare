#!/usr/bin/env python3
"""Project one command jointly against nominal and physical 9.96 GHz operators."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import metric_at
from generate_v09_eep_development_candidates import (
    MARGIN_NAMES,
    METRIC_NAMES,
    full_active_metrics,
    physical_margins,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from run_v16_k6_multifrequency_rescue import Variant, optimize_common_command
from run_v16_robust_drift_oracle import (
    apply_calibration,
    complex_to_ri,
    hardware_margin,
    load_nominal_operator,
    load_npz,
    ri_to_complex,
    scene_calibration_states,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v19_nominal_9p96_joint_active_rl_projection.json"
DEFAULT_DATASET = (
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
DEFAULT_LOW = (
    ROOT
    / "hfss_outputs"
    / "v18_perturbed_operator_frequency_low_20260729_run01"
    / "operator"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_OUT = (
    ROOT / "hfss_outputs" / "v19_nominal_9p96_joint_projection_20260729_run01"
)
ROBUST_MARGIN_NAMES = np.concatenate((MARGIN_NAMES, np.asarray(["hardware"])))
SOURCE_CORNER = "frequency_low_x0.20"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--low-operator", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-variants", type=int, default=0)
    return parser.parse_args()


def identity_state() -> dict[str, Any]:
    return {
        "factor": np.ones(256, dtype=np.complex128),
        "soft_ports": np.asarray([], dtype=np.int64),
        "phase_rms_deg": 0.0,
        "gain_rms_db": 0.0,
        "group_phase_bias_rms_deg": 0.0,
        "compression": 0.0,
        "temperature_offset_c": 0.0,
        "phase_bits": 30,
        "amplitude_bits": 30,
        "drift_intensity": 0.0,
    }


def command_hash(command: np.ndarray) -> str:
    value = np.ascontiguousarray(complex_to_ri(command))
    return hashlib.sha256(value.tobytes()).hexdigest()


def append_corner_values(
    row: dict[str, Any],
    name: str,
    metrics: dict[str, float],
    active: dict[str, float | int],
    margins: np.ndarray,
) -> None:
    for metric_name in METRIC_NAMES:
        row[f"{name}_{metric_name}"] = float(metrics[str(metric_name)])
    row[f"{name}_active_rl_floor_db"] = float(active["active_rl_floor_db"])
    row[f"{name}_active_rl_gate"] = int(active["active_rl_gate"])
    for margin_name, value in zip(ROBUST_MARGIN_NAMES, margins):
        row[f"{name}_{margin_name}_margin_db"] = float(value)
    row[f"{name}_strict_pass"] = int(float(np.min(margins)) >= 0.0)


def evaluate_command(
    command: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    reference: dict[str, float],
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    grid_dirs: np.ndarray,
    gates: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    all_margins = np.zeros((len(corners), len(ROBUST_MARGIN_NAMES)), dtype=np.float32)
    row: dict[str, Any] = {}
    for corner_index, (name, corner) in enumerate(corners.items()):
        actual = apply_calibration(command, mask, states[name])
        metrics = metric_at(corner["fast"].evaluate(actual, targets), 0)
        active = full_active_metrics(actual, mask, corner["s"])
        physical = physical_margins(metrics, reference, active)
        hardware_value, hardware = hardware_margin(
            command,
            actual,
            mask,
            targets,
            corner["effective"],
            grid_dirs,
            gates,
        )
        margins = np.concatenate((physical, np.asarray([hardware_value], dtype=np.float32)))
        all_margins[corner_index] = margins
        append_corner_values(row, name, metrics, active, margins)
        for key, value in hardware.items():
            row[f"{name}_{key}"] = float(value)
    source_indices = [index for index, name in enumerate(corners) if name.endswith("source")]
    pattern_indices = np.arange(4, dtype=int)
    row.update(
        {
            "all_corner_strict_pass": int(np.min(all_margins) >= 0.0),
            "source_pair_strict_pass": int(np.min(all_margins[source_indices]) >= 0.0),
            "all_corner_pattern_pass": int(np.min(all_margins[:, pattern_indices]) >= 0.0),
            "all_corner_active_rl_pass": int(np.min(all_margins[:, 4]) >= 0.0),
            "robust_worst_margin_db": float(np.min(all_margins)),
            "robust_pattern_margin_db": float(np.min(all_margins[:, pattern_indices])),
            "robust_active_rl_margin_db": float(np.min(all_margins[:, 4])),
            "robust_hardware_margin_db": float(np.min(all_margins[:, 5])),
        }
    )
    return row, all_margins


def selection_key(row: dict[str, Any]) -> tuple[int, int, int, int, float, float, float]:
    return (
        int(row["all_corner_strict_pass"]),
        int(row["source_pair_strict_pass"]),
        int(row["all_corner_pattern_pass"]),
        int(row["all_corner_active_rl_pass"]),
        min(float(row["robust_worst_margin_db"]), 3.0),
        min(float(row["robust_active_rl_margin_db"]), 3.0),
        -float(row["relative_command_change"]),
    )


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite joint projection: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    variants = [Variant(**item) for item in protocol["design_variants"]]
    if args.max_variants > 0:
        variants = variants[: args.max_variants]

    data = load_npz(args.dataset)
    pool = load_npz(args.pool)
    nominal_base, nominal_effective, nominal_fast, nominal_s = load_nominal_operator(
        args.nominal_operator
    )
    low_base, low_effective, low_fast, low_s = load_nominal_operator(args.low_operator)
    if not np.isclose(float(nominal_base["frequency_ghz"]), 10.0, atol=1.0e-6):
        raise RuntimeError("Nominal operator frequency changed")
    if not np.isclose(float(low_base["frequency_ghz"]), 9.96, atol=1.0e-6):
        raise RuntimeError("Low operator is not 9.96 GHz")
    if not np.array_equal(nominal_base["element_ixiy"], low_base["element_ixiy"]):
        raise RuntimeError("Nominal and 9.96 GHz port ordering differs")
    if not (
        np.array_equal(nominal_base["theta_deg"], low_base["theta_deg"])
        and np.array_equal(nominal_base["phi_deg"], low_base["phi_deg"])
    ):
        raise RuntimeError("Nominal and 9.96 GHz EEP grids differ")

    corners = {
        "nominal_identity": {"effective": nominal_effective, "fast": nominal_fast, "s": nominal_s},
        "nominal_E2_source": {"effective": nominal_effective, "fast": nominal_fast, "s": nominal_s},
        "frequency_low_identity": {"effective": low_effective, "fast": low_fast, "s": low_s},
        "frequency_low_E2_source": {"effective": low_effective, "fast": low_fast, "s": low_s},
    }
    grid_dirs = pattern_grid_dirs(nominal_base["theta_deg"], nominal_base["phi_deg"])
    samples = np.asarray(data["sample_index"], dtype=np.int64)
    source_indices = np.arange(len(samples))
    if args.max_scenes > 0:
        source_indices = source_indices[: args.max_scenes]
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    gates = protocol["gates_unchanged"]
    optimizer = protocol["optimizer"]

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_tasks = np.zeros((len(source_indices), 256, 6), dtype=np.complex64)
    selected_actual_low = np.zeros_like(selected_tasks)
    selected_margins = np.zeros(
        (len(source_indices), len(corners), len(ROBUST_MARGIN_NAMES)), dtype=np.float32
    )
    source_reconstruction_error = 0.0
    started = time.time()

    for output_index, source_index in enumerate(source_indices):
        sample = int(samples[source_index])
        k_value = int(data["k_values"][source_index])
        mask = np.asarray(data["masks"][source_index], dtype=bool)
        targets = np.asarray(data["targets_deg"][source_index, :k_value], dtype=float)
        initial = ri_to_complex(
            data["nominal_external_task_weights_real_imag"][source_index, :, :k_value]
        )
        pool_scene = np.flatnonzero(pool_samples == sample)
        if pool_scene.size == 0:
            raise RuntimeError(f"Missing source scene {sample} in robust pool")
        source_state = scene_calibration_states(
            pool,
            pool_scene,
            {SOURCE_CORNER: {"profile": "frequency_low", "level": 0.20}},
            nominal_base["element_ixiy"],
            int(protocol["seed"]),
        )[SOURCE_CORNER]
        identity = identity_state()
        states = {
            "nominal_identity": identity,
            "nominal_E2_source": source_state,
            "frequency_low_identity": identity,
            "frequency_low_E2_source": source_state,
        }
        expected_actual = ri_to_complex(
            data["actual_external_task_weights_real_imag"][source_index, :, :k_value]
        )
        regenerated = apply_calibration(initial, mask, source_state)
        mismatch = float(np.max(np.abs(expected_actual - regenerated)))
        source_reconstruction_error = max(source_reconstruction_error, mismatch)
        if mismatch > 2.0e-6:
            raise RuntimeError(f"Frozen source state mismatch for sample {sample}: {mismatch}")

        reference = metric_at(nominal_fast.evaluate(initial, targets), 0)
        candidates: list[tuple[np.ndarray, str, dict[str, float]]] = [
            (initial, "baseline_frozen", {})
        ]
        for variant in variants:
            projected, diagnostics = optimize_common_command(
                initial,
                mask,
                targets,
                corners,
                states,
                grid_dirs,
                variant,
                int(optimizer["active_rl_nullspace_steps"]),
                float(optimizer["active_rl_nullspace_step_size"]),
                bool(optimizer["quantization_aware_selection"]),
            )
            candidates.append((projected, variant.name, diagnostics))

        evaluated: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
        initial_norm = max(float(np.linalg.norm(initial)), EPS)
        for candidate_index, (command, variant_name, diagnostics) in enumerate(candidates):
            values, margins = evaluate_command(
                command, mask, targets, reference, corners, states, grid_dirs, gates
            )
            values.update(
                {
                    "source_candidate_index": int(source_index),
                    "candidate_index": candidate_index,
                    "sample_index": sample,
                    "k": k_value,
                    "ratio": float(data["active_ratios_requested"][source_index]),
                    "active_count": int(np.sum(mask)),
                    "variant": variant_name,
                    "command_hash": command_hash(command),
                    "relative_command_change": float(np.linalg.norm(command - initial) / initial_norm),
                    "source_state_reconstruction_max_abs": mismatch,
                    **{f"optimizer_{key}": float(value) for key, value in diagnostics.items()},
                }
            )
            rows.append(values)
            evaluated.append((values, command, margins))
        best_values, best_command, best_margins = max(evaluated, key=lambda item: selection_key(item[0]))
        best_values["selected"] = 1
        selected_rows.append(dict(best_values))
        selected_tasks[output_index, :, :k_value] = best_command
        selected_actual_low[output_index, :, :k_value] = apply_calibration(
            best_command, mask, source_state
        )
        selected_margins[output_index] = best_margins
        print(
            f"joint projection {output_index + 1:02d}/{len(source_indices):02d} "
            f"sample={sample} K={k_value} ratio={float(data['active_ratios_requested'][source_index]):.1f} "
            f"variant={best_values['variant']} strict={best_values['all_corner_strict_pass']} "
            f"RLmargin={best_values['robust_active_rl_margin_db']:.3f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    selected_keys = {
        (int(row["source_candidate_index"]), int(row["candidate_index"])) for row in selected_rows
    }
    for row in rows:
        row["selected"] = int(
            (int(row["source_candidate_index"]), int(row["candidate_index"])) in selected_keys
        )
    write_csv(args.out_dir / "candidate_metrics.csv", rows)
    write_csv(args.out_dir / "selected_candidate_metrics.csv", selected_rows)
    np.savez_compressed(
        args.out_dir / "projected_commands.npz",
        source_candidate_index=source_indices,
        sample_index=samples[source_indices],
        k_values=np.asarray(data["k_values"])[source_indices],
        active_ratios_requested=np.asarray(data["active_ratios_requested"])[source_indices],
        masks=np.asarray(data["masks"])[source_indices],
        targets_deg=np.asarray(data["targets_deg"])[source_indices],
        selected_task_weights_real_imag=complex_to_ri(selected_tasks),
        selected_frequency_low_actual_weights_real_imag=complex_to_ri(selected_actual_low),
        robust_margins=selected_margins,
        corner_names=np.asarray(list(corners)),
        robust_margin_names=ROBUST_MARGIN_NAMES,
        selected_variants=np.asarray([str(row["variant"]) for row in selected_rows]),
        selected_command_hashes=np.asarray([str(row["command_hash"]) for row in selected_rows]),
    )

    baseline_rows = [row for row in rows if row["variant"] == "baseline_frozen"]
    summary = {
        "protocol": protocol["protocol"],
        "scene_count": len(source_indices),
        "variant_count_per_scene_including_baseline": len(variants) + 1,
        "same_mask_and_command_across_corners": True,
        "source_state_reconstruction_max_abs": source_reconstruction_error,
        "baseline_all_corner_strict_pass_count": sum(
            int(row["all_corner_strict_pass"]) for row in baseline_rows
        ),
        "selected_all_corner_strict_pass_count": sum(
            int(row["all_corner_strict_pass"]) for row in selected_rows
        ),
        "baseline_source_pair_strict_pass_count": sum(
            int(row["source_pair_strict_pass"]) for row in baseline_rows
        ),
        "selected_source_pair_strict_pass_count": sum(
            int(row["source_pair_strict_pass"]) for row in selected_rows
        ),
        "selected_pattern_pass_count": sum(
            int(row["all_corner_pattern_pass"]) for row in selected_rows
        ),
        "selected_active_rl_pass_count": sum(
            int(row["all_corner_active_rl_pass"]) for row in selected_rows
        ),
        "selected_minimum_active_rl_floor_db": min(
            10.0 + float(row["robust_active_rl_margin_db"]) for row in selected_rows
        ),
        "selected_k_pass": {
            str(k): sum(
                int(row["all_corner_strict_pass"]) for row in selected_rows if int(row["k"]) == k
            )
            for k in (2, 4, 6)
        },
        "critic_training_allowed": False,
        "symmetric_10p04_operator_smoke_allowed": True,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
