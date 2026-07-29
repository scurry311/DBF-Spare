#!/usr/bin/env python3
"""Rebuild three external S256/EEP operators and replay frozen v1.12 commands."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import FastPatternEvaluator, metric_at
from generate_v09_eep_development_candidates import (
    METRIC_NAMES,
    full_active_metrics,
    physical_margins,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from refine_trusted_dense_local_eep_joint import DenseExternalEEP
from run_v16_robust_drift_oracle import hardware_margin, load_npz, ri_to_complex
from run_v19_nominal_9p96_joint_projection import ROBUST_MARGIN_NAMES, append_corner_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v21_three_frequency_broadband_matching.json"
DEFAULT_V20_PROTOCOL = ROOT / "configs" / "v20_three_frequency_mask_weight_joint.json"
DEFAULT_FROZEN = ROOT / "hfss_outputs" / "v21_frozen_v112_replay_20260729_run03" / "frozen_v112_replay_candidates.npz"
DEFAULT_DESIGN = ROOT / "hfss_outputs" / "v21_three_frequency_broadband_match_20260729_run02_q50_geometry01" / "network_matrices.npz"
DEFAULT_NOMINAL = ROOT / "hfss_outputs" / "fixed_mesh_eep256_20260723_run05" / "grounded_patch_eep_operator_256port.npz"
DEFAULT_LOW = ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_20260729_run01" / "operator" / "grounded_patch_eep_operator_256port.npz"
DEFAULT_HIGH = ROOT / "hfss_outputs" / "v19_perturbed_operator_frequency_high_20260729_run01" / "operator" / "grounded_patch_eep_operator_256port.npz"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v21_broadband_s256_eep_replay_20260729_run01"
STATE_TO_FREQUENCY = (0, 1, 1, 2, 2)
KMAX = 6
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--v20-protocol", type=Path, default=DEFAULT_V20_PROTOCOL)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--design-variant", default="geometry3_sls")
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--low-operator", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--high-operator", type=Path, default=DEFAULT_HIGH)
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


def complex_nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(
        np.sum(np.abs(reference - estimate) ** 2)
        / max(float(np.sum(np.abs(reference) ** 2)), EPS)
    )


def network_efficiency(
    external_tasks: np.ndarray,
    raw_s: np.ndarray,
    external_s: np.ndarray,
    antenna_map: np.ndarray,
) -> float:
    values = [np.sum(external_tasks, axis=1)]
    values.extend(external_tasks[:, task] for task in range(external_tasks.shape[1]))
    minimum = float("inf")
    for external in values:
        reflected = external_s @ external
        antenna_incident = antenna_map @ external
        antenna_reflected = raw_s @ antenna_incident
        accepted_external = float(np.vdot(external, external).real - np.vdot(reflected, reflected).real)
        accepted_antenna = float(
            np.vdot(antenna_incident, antenna_incident).real
            - np.vdot(antenna_reflected, antenna_reflected).real
        )
        minimum = min(minimum, accepted_antenna / max(accepted_external, EPS))
    return minimum


def build_operator(
    raw: dict[str, np.ndarray],
    external_s: np.ndarray,
    antenna_map: np.ndarray,
    variant: str,
) -> tuple[DenseExternalEEP, dict[str, Any]]:
    effective = DenseExternalEEP(raw["etheta"], raw["ephi"], antenna_map)
    rng = np.random.default_rng(20260829 + int(round(float(raw["frequency_ghz"]) * 100)))
    validation_nmse: list[float] = []
    for _ in range(3):
        command = rng.normal(size=256) + 1j * rng.normal(size=256)
        command /= np.linalg.norm(command)
        raw_field = (antenna_map @ command) @ np.asarray(raw["etheta"], np.complex128)
        external_field = command @ np.asarray(effective.etheta, np.complex128)
        validation_nmse.append(complex_nmse(raw_field, external_field))
    metrics = {
        "variant": variant,
        "frequency_ghz": float(raw["frequency_ghz"]),
        "port_count": int(external_s.shape[0]),
        "grid_point_count": int(len(raw["theta_deg"])),
        "reciprocity_max_abs": float(np.max(np.abs(external_s - external_s.T))),
        "passivity_sigma_max": float(np.max(np.linalg.svd(external_s, compute_uv=False))),
        "passive_rl_min_db": float(
            np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(external_s)), 1.0e-15)))
        ),
        "eep_map_complex_nmse_max": max(validation_nmse),
    }
    return effective, metrics


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite S256/EEP replay: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    v20_protocol = json.loads(args.v20_protocol.read_text(encoding="utf-8"))
    gates = v20_protocol["gates"]
    replay_gate = protocol["replay_gate"]
    frozen = load_npz(args.frozen)
    design = load_npz(args.design)
    raw_operators = [load_npz(path) for path in (args.nominal_operator, args.low_operator, args.high_operator)]
    names = [str(value) for value in design["variant_names"]]
    if args.design_variant not in names:
        raise ValueError(f"Design variant {args.design_variant!r} not found in {names}")
    design_index = names.index(args.design_variant)
    variants = [names[0], args.design_variant]
    variant_indices = [0, design_index]
    tasks = ri_to_complex(frozen["tasks_real_imag"])
    state_tasks = ri_to_complex(frozen["state_tasks_real_imag"])
    masks = np.asarray(frozen["masks"], dtype=bool)
    targets_all = np.asarray(frozen["targets"], dtype=float)
    references = np.asarray(frozen["reference_metrics"], dtype=float)
    reference_names = [str(value) for value in frozen["reference_metric_names"]]
    k_values = np.asarray(frozen["k_values"], dtype=int)
    state_names = [str(value) for value in frozen["state_names"]]
    grid_dirs = pattern_grid_dirs(raw_operators[0]["theta_deg"], raw_operators[0]["phi_deg"])
    structural_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    operator_cache: dict[str, list[dict[str, Any]]] = {}

    for variant, variant_index in zip(variants, variant_indices):
        operators: list[dict[str, Any]] = []
        for frequency_index, raw in enumerate(raw_operators):
            external_s = np.asarray(design["s_external"][variant_index, frequency_index], np.complex128)
            antenna_map = np.asarray(design["antenna_map"][variant_index, frequency_index], np.complex128)
            effective, structural = build_operator(raw, external_s, antenna_map, variant)
            structural_rows.append(structural)
            operators.append(
                {
                    "raw": raw,
                    "s": external_s,
                    "map": antenna_map,
                    "effective": effective,
                    "fast": FastPatternEvaluator(effective, raw["theta_deg"], raw["phi_deg"]),
                }
            )
            if variant == args.design_variant:
                export_dir = args.out_dir / "operators" / f"{float(raw['frequency_ghz']):.2f}GHz"
                export_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    export_dir / "broadband_external_eep_operator_256port.npz",
                    port_names=raw["port_names"],
                    element_indices=raw["element_indices"],
                    element_ixiy=raw["element_ixiy"],
                    theta_deg=raw["theta_deg"],
                    phi_deg=raw["phi_deg"],
                    etheta=np.asarray(effective.etheta, np.complex64),
                    ephi=np.asarray(effective.ephi, np.complex64),
                    raw_antenna_s=np.asarray(raw["s_raw"], np.complex64),
                    external_s=np.asarray(external_s, np.complex64),
                    antenna_incident_wave_map=np.asarray(antenna_map, np.complex64),
                    frequency_ghz=raw["frequency_ghz"],
                    network_variant=np.asarray(variant),
                    network_topology=design["topology"],
                    component_q=design["component_q"],
                    network_parameters=design["parameters"][variant_index],
                    model_scope=np.asarray("three-frequency finite-Q circuit cascade on physical HFSS raw S256/EEP"),
                )
        operator_cache[variant] = operators

        for candidate in range(len(k_values)):
            k_value = int(k_values[candidate])
            command = np.asarray(tasks[candidate, :, :k_value], np.complex64)
            mask = masks[candidate]
            targets = targets_all[candidate, :k_value]
            reference = {
                name: float(references[candidate, index])
                for index, name in enumerate(reference_names)
            }
            all_margins = np.zeros((len(state_names), len(ROBUST_MARGIN_NAMES)), np.float32)
            row: dict[str, Any] = {
                "variant": variant,
                "freeze_index": candidate,
                "sample_index": int(frozen["sample_index"][candidate]),
                "k": k_value,
                "ratio": float(frozen["ratio"][candidate]),
            }
            minimum_efficiency = float("inf")
            for state_index, state_name in enumerate(state_names):
                frequency_index = STATE_TO_FREQUENCY[state_index]
                operator = operators[frequency_index]
                actual = np.asarray(state_tasks[candidate, state_index, :, :k_value], np.complex64)
                metrics = metric_at(operator["fast"].evaluate(actual, targets), 0)
                active = full_active_metrics(actual, mask, operator["s"])
                physical = physical_margins(metrics, reference, active)
                hardware_value, hardware = hardware_margin(
                    command,
                    actual,
                    mask,
                    targets,
                    operator["effective"],
                    grid_dirs,
                    gates,
                )
                margins = np.concatenate((physical, np.asarray([hardware_value], np.float32)))
                all_margins[state_index] = margins
                append_corner_values(row, state_name, metrics, active, margins)
                for key, value in hardware.items():
                    row[f"{state_name}_{key}"] = float(value)
                efficiency = network_efficiency(
                    actual,
                    np.asarray(operator["raw"]["s_raw"], np.complex128),
                    operator["s"],
                    operator["map"],
                )
                row[f"{state_name}_network_efficiency"] = efficiency
                minimum_efficiency = min(minimum_efficiency, efficiency)
            pattern_indices = np.arange(4, dtype=int)
            row.update(
                {
                    "all_corner_strict_pass": int(np.min(all_margins) >= 0.0),
                    "all_corner_pattern_pass": int(np.min(all_margins[:, pattern_indices]) >= 0.0),
                    "all_corner_active_rl_pass": int(np.min(all_margins[:, 4]) >= 0.0),
                    "robust_worst_margin_db": float(np.min(all_margins)),
                    "robust_pattern_margin_db": float(np.min(all_margins[:, pattern_indices])),
                    "robust_active_rl_margin_db": float(np.min(all_margins[:, 4])),
                    "robust_hardware_margin_db": float(np.min(all_margins[:, 5])),
                    "minimum_network_efficiency": minimum_efficiency,
                    "reserve11_pass": int(
                        np.min(all_margins[:, pattern_indices]) >= 0.0
                        and np.min(all_margins[:, 4]) >= 1.0
                        and np.min(all_margins[:, 5]) >= 0.0
                        and minimum_efficiency >= float(replay_gate["all_corner_network_efficiency_min"])
                    ),
                }
            )
            replay_rows.append(row)

    write_csv(args.out_dir / "operator_structural_validation.csv", structural_rows)
    write_csv(args.out_dir / "frozen_replay_candidate_metrics.csv", replay_rows)
    group_rows: list[dict[str, Any]] = []
    for variant in variants:
        members = [row for row in replay_rows if row["variant"] == variant]
        for k_value in (2, 4, 6):
            group = [row for row in members if int(row["k"]) == k_value]
            group_rows.append(
                {
                    "variant": variant,
                    "k": k_value,
                    "scene_count": len(group),
                    "strict_count": sum(int(row["all_corner_strict_pass"]) for row in group),
                    "pattern_pass_count": sum(int(row["all_corner_pattern_pass"]) for row in group),
                    "active_rl_pass_count": sum(int(row["all_corner_active_rl_pass"]) for row in group),
                    "reserve11_count": sum(int(row["reserve11_pass"]) for row in group),
                    "minimum_network_efficiency": min(float(row["minimum_network_efficiency"]) for row in group),
                }
            )
    write_csv(args.out_dir / "frozen_replay_by_variant_k.csv", group_rows)
    structural_by_variant = {
        variant: [row for row in structural_rows if row["variant"] == variant]
        for variant in variants
    }
    design_members = [row for row in replay_rows if row["variant"] == args.design_variant]
    k2 = sum(int(row["all_corner_strict_pass"]) for row in design_members if int(row["k"]) == 2)
    k4 = sum(int(row["all_corner_strict_pass"]) for row in design_members if int(row["k"]) == 4)
    reserve = sum(int(row["reserve11_pass"]) for row in design_members)
    passive = min(float(row["passive_rl_min_db"]) for row in structural_by_variant[args.design_variant])
    efficiency = min(float(row["minimum_network_efficiency"]) for row in design_members)
    structural_pass = all(
        float(row["reciprocity_max_abs"]) <= 1.0e-5
        and float(row["passivity_sigma_max"]) <= 1.0001
        and float(row["eep_map_complex_nmse_max"]) <= 1.0e-10
        for row in structural_by_variant[args.design_variant]
    )
    gate = bool(
        structural_pass
        and k2 >= int(replay_gate["k2_strict_count_min"])
        and k4 >= int(replay_gate["k4_strict_count_min"])
        and reserve >= int(replay_gate["reserve11_count_min"])
        and passive >= float(replay_gate["all_corner_passive_rl_min_db"])
        and efficiency >= float(replay_gate["all_corner_network_efficiency_min"])
    )
    baseline_members = [row for row in replay_rows if row["variant"] == variants[0]]
    decision = {
        "protocol": protocol["protocol"],
        "frozen_scene_count": len(k_values),
        "baseline_strict_count": sum(int(row["all_corner_strict_pass"]) for row in baseline_members),
        "baseline_k2_strict_count": sum(int(row["all_corner_strict_pass"]) for row in baseline_members if int(row["k"]) == 2),
        "baseline_k4_strict_count": sum(int(row["all_corner_strict_pass"]) for row in baseline_members if int(row["k"]) == 4),
        "design_variant": args.design_variant,
        "design_strict_count": sum(int(row["all_corner_strict_pass"]) for row in design_members),
        "design_k2_strict_count": k2,
        "design_k4_strict_count": k4,
        "design_reserve11_count": reserve,
        "design_minimum_passive_rl_db": passive,
        "design_minimum_network_efficiency": efficiency,
        "operator_structural_gate_pass": structural_pass,
        "replay_gate_pass": gate,
        "candidate_optimization_allowed": gate,
        "small_hfss_smoke_allowed": gate,
        "critic_training_allowed": False,
        "bulk_hfss_allowed": False,
        "thresholds_changed": False,
        "mask_or_weight_changes": False,
        "embedded_hfss_validated": False,
        "next_action": (
            "freeze_selected_candidates_for_small_embedded-network_hfss_smoke"
            if gate
            else "do_not_reopen_algorithm_or_hfss; matching network remains insufficient"
        ),
    }
    (args.out_dir / "stage_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
