#!/usr/bin/env python3
"""Evaluate frozen candidates on the physical 9.96 GHz 16x16 operator."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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
from run_v16_robust_drift_oracle import (
    apply_calibration,
    load_nominal_operator,
    load_npz,
    scene_calibration_states,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT / "hfss_outputs" / "v17_frozen_16x16_perturbed_hfss_smoke_dataset_20260728_run02"
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
DEFAULT_PHYSICAL = (
    ROOT
    / "hfss_outputs"
    / "v18_perturbed_operator_frequency_low_20260729_run01"
    / "operator"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_OUT = (
    ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_evaluation_20260729_run01"
)
CORNER_NAME = "frequency_low_x0.20"
PROFILE = "frequency_low"
LEVEL = 0.20
SEED = 20260827


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--physical-operator", type=Path, default=DEFAULT_PHYSICAL)
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


def complex_to_ri(value: np.ndarray) -> np.ndarray:
    return np.stack((value.real, value.imag), axis=-1).astype(np.float32)


def append_metrics(row: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for name in METRIC_NAMES:
        row[f"{prefix}_{name}"] = float(metrics[str(name)])


def append_margins(row: dict[str, Any], prefix: str, margins: np.ndarray) -> None:
    for name, value in zip(MARGIN_NAMES, margins):
        row[f"{prefix}_{name}_margin_db"] = float(value)


def grouped(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    output: list[dict[str, Any]] = []
    for value, members in sorted(groups.items(), key=lambda item: item[0]):
        count = len(members)
        output.append(
            {
                "stratum": field,
                "value": value,
                "candidate_count": count,
                "operator_only_pass_count": sum(int(row["operator_only_strict_pass"]) for row in members),
                "operator_only_pass_rate": sum(int(row["operator_only_strict_pass"]) for row in members) / count,
                "physical_plus_source_pass_count": sum(
                    int(row["physical_plus_source_strict_pass"]) for row in members
                ),
                "physical_plus_source_pass_rate": sum(
                    int(row["physical_plus_source_strict_pass"]) for row in members
                )
                / count,
                "worst_physical_plus_source_psll_db": max(
                    float(row["physical_plus_source_psll_db"]) for row in members
                ),
                "minimum_physical_plus_source_nearest_iso_db": min(
                    float(row["physical_plus_source_nearest_iso_db"]) for row in members
                ),
                "minimum_physical_plus_source_local_iso_db": min(
                    float(row["physical_plus_source_local_iso_db"]) for row in members
                ),
                "minimum_physical_plus_source_active_rl_db": min(
                    float(row["physical_plus_source_active_rl_floor_db"]) for row in members
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite perturbed-operator evaluation: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_npz(args.dataset_dir / "dataset_arrays.npz")
    pool = load_npz(args.pool)
    nominal_base, _nominal_effective, nominal_fast, nominal_s = load_nominal_operator(
        args.nominal_operator
    )
    physical_base, _physical_effective, physical_fast, physical_s = load_nominal_operator(
        args.physical_operator
    )
    if not np.isclose(float(physical_base["frequency_ghz"]), 9.96, atol=1.0e-6):
        raise RuntimeError("Physical operator is not the frozen 9.96 GHz corner")
    if not np.array_equal(nominal_base["element_ixiy"], physical_base["element_ixiy"]):
        raise RuntimeError("Nominal and physical operator port ordering differs")

    nominal_external = (
        np.asarray(dataset["nominal_external_task_weights_real_imag"][..., 0], dtype=np.float32)
        + 1j
        * np.asarray(dataset["nominal_external_task_weights_real_imag"][..., 1], dtype=np.float32)
    )
    masks = np.asarray(dataset["masks"], dtype=bool)
    samples = np.asarray(dataset["sample_index"], dtype=np.int64)
    k_values = np.asarray(dataset["k_values"], dtype=np.int8)
    targets = np.asarray(dataset["targets_deg"], dtype=np.float32)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    corners = {CORNER_NAME: {"profile": PROFILE, "level": LEVEL}}

    actual_external = np.zeros_like(nominal_external, dtype=np.complex64)
    reference_metrics = np.zeros((len(samples), len(METRIC_NAMES)), dtype=np.float32)
    operator_metrics = np.zeros_like(reference_metrics)
    actual_metrics = np.zeros_like(reference_metrics)
    operator_margins = np.zeros((len(samples), len(MARGIN_NAMES)), dtype=np.float32)
    actual_margins = np.zeros_like(operator_margins)
    rows: list[dict[str, Any]] = []

    for candidate in range(len(samples)):
        sample = int(samples[candidate])
        k_value = int(k_values[candidate])
        mask = masks[candidate]
        target = np.asarray(targets[candidate, :k_value], dtype=float)
        command = np.asarray(nominal_external[candidate, :, :k_value], dtype=np.complex64)
        scene_members = np.flatnonzero(pool_samples == sample)
        if scene_members.size == 0:
            raise RuntimeError(f"Missing source scene {sample} in v16 pool")
        state = scene_calibration_states(
            pool,
            scene_members,
            corners,
            nominal_base["element_ixiy"],
            SEED,
        )[CORNER_NAME]
        actual = apply_calibration(command, mask, state)
        actual_external[candidate, :, :k_value] = actual

        reference = metric_at(nominal_fast.evaluate(command[None, ...], target), 0)
        operator_only = metric_at(physical_fast.evaluate(command[None, ...], target), 0)
        physical_plus_source = metric_at(physical_fast.evaluate(actual[None, ...], target), 0)
        operator_active = full_active_metrics(command, mask, physical_s)
        actual_active = full_active_metrics(actual, mask, physical_s)
        operator_margin = physical_margins(operator_only, reference, operator_active)
        actual_margin = physical_margins(physical_plus_source, reference, actual_active)

        reference_metrics[candidate] = metric_vector(reference)
        operator_metrics[candidate] = metric_vector(operator_only)
        actual_metrics[candidate] = metric_vector(physical_plus_source)
        operator_margins[candidate] = operator_margin
        actual_margins[candidate] = actual_margin
        root_index = int(np.argmin(actual_margin))
        row: dict[str, Any] = {
            "candidate_index": candidate,
            "sample_index": sample,
            "k": k_value,
            "ratio": float(dataset["active_ratios_requested"][candidate]),
            "active_count": int(dataset["num_active"][candidate]),
            "selection_role": str(dataset["selection_roles"][candidate]),
            "corner": CORNER_NAME,
            "physical_frequency_ghz": 9.96,
            "phase_rms_deg": float(state["phase_rms_deg"]),
            "gain_rms_db": float(state["gain_rms_db"]),
            "soft_active_port_count": int(np.sum(mask[np.asarray(state["soft_ports"], dtype=int)])),
            "operator_only_active_rl_floor_db": float(operator_active["active_rl_floor_db"]),
            "physical_plus_source_active_rl_floor_db": float(actual_active["active_rl_floor_db"]),
            "operator_only_strict_pass": int(np.min(operator_margin) >= 0.0),
            "physical_plus_source_strict_pass": int(np.min(actual_margin) >= 0.0),
            "physical_plus_source_root_cause": str(MARGIN_NAMES[root_index]),
            "physical_plus_source_worst_margin_db": float(np.min(actual_margin)),
        }
        append_metrics(row, "nominal", reference)
        append_metrics(row, "operator_only", operator_only)
        append_metrics(row, "physical_plus_source", physical_plus_source)
        append_margins(row, "operator_only", operator_margin)
        append_margins(row, "physical_plus_source", actual_margin)
        rows.append(row)

    write_csv(args.out_dir / "candidate_physical_corner_metrics.csv", rows)
    strata: list[dict[str, Any]] = []
    for field in ("k", "ratio", "selection_role"):
        strata.extend(grouped(rows, field))
    write_csv(args.out_dir / "stratified_physical_corner_metrics.csv", strata)

    output = dict(dataset)
    actual_combined = np.sum(actual_external, axis=2)
    output.update(
        {
            "reference_metrics": reference_metrics,
            "nominal_metrics": operator_metrics,
            "actual_metrics": actual_metrics,
            "nominal_margins": operator_margins,
            "actual_margins": actual_margins,
            "margin_residuals": actual_margins - operator_margins,
            "hfss_actual_task_weights_real_imag": complex_to_ri(np.conjugate(actual_external)),
            "hfss_actual_combined_weights_real_imag": complex_to_ri(
                np.conjugate(actual_combined)
            ),
            "hfss_weights_real_imag": complex_to_ri(np.conjugate(actual_combined)),
            "actual_external_task_weights_real_imag": complex_to_ri(actual_external),
            "frozen_E2_corner": np.asarray([CORNER_NAME] * len(samples)),
            "physical_16x16_operator_corner_included": np.ones(len(samples), dtype=np.int8),
            "physical_operator_frequency_ghz": np.full(len(samples), 9.96, dtype=np.float32),
            "gate15": (actual_margins[:, 0] >= 0.0).astype(np.int8),
            "strict_gate20": (np.min(actual_margins[:, :3], axis=1) >= 0.0).astype(np.int8),
            "mainlobe_gate": (actual_margins[:, 3] >= 0.0).astype(np.int8),
            "active_rl_gate": (actual_margins[:, 4] >= 0.0).astype(np.int8),
        }
    )
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **output)

    operator_pass = sum(int(row["operator_only_strict_pass"]) for row in rows)
    actual_pass = sum(int(row["physical_plus_source_strict_pass"]) for row in rows)
    summary = {
        "protocol": "v18-frozen-candidates-on-physical-frequency-low-operator",
        "candidate_count": len(rows),
        "independent_scene_count": len(set(samples.tolist())),
        "physical_frequency_ghz": 9.96,
        "common_mask_and_weights_frozen": True,
        "thresholds_changed": False,
        "operator_only_strict_pass_count": operator_pass,
        "operator_only_strict_pass_rate": operator_pass / len(rows),
        "physical_plus_source_strict_pass_count": actual_pass,
        "physical_plus_source_strict_pass_rate": actual_pass / len(rows),
        "physical_plus_source_k6_pass_count": sum(
            int(row["physical_plus_source_strict_pass"]) for row in rows if int(row["k"]) == 6
        ),
        "failure_root_causes": dict(
            Counter(
                row["physical_plus_source_root_cause"]
                for row in rows
                if not int(row["physical_plus_source_strict_pass"])
            )
        ),
        "worst_physical_plus_source_psll_db": max(
            float(row["physical_plus_source_psll_db"]) for row in rows
        ),
        "minimum_physical_plus_source_nearest_iso_db": min(
            float(row["physical_plus_source_nearest_iso_db"]) for row in rows
        ),
        "minimum_physical_plus_source_local_iso_db": min(
            float(row["physical_plus_source_local_iso_db"]) for row in rows
        ),
        "minimum_physical_plus_source_active_rl_db": min(
            float(row["physical_plus_source_active_rl_floor_db"]) for row in rows
        ),
        "direct_hfss_superposition_required": True,
        "critic_training_allowed": False,
    }
    (args.out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
