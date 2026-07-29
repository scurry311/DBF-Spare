#!/usr/bin/env python3
"""Audit three-frequency per-port active return and derive mask-swap scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hfss_task_fullwave_validate import pattern_grid_dirs
from generate_v09_eep_development_candidates import matched_steering_tasks
from run_v16_robust_drift_oracle import (
    apply_calibration,
    load_nominal_operator,
    load_npz,
    ri_to_complex,
    scene_calibration_states,
)
from run_v19_nominal_9p96_joint_projection import identity_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v20_three_frequency_mask_weight_joint.json"
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
DEFAULT_LOW = (
    ROOT
    / "hfss_outputs"
    / "v18_perturbed_operator_frequency_low_20260729_run01"
    / "operator"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_HIGH = (
    ROOT
    / "hfss_outputs"
    / "v19_perturbed_operator_frequency_high_20260729_run01"
    / "operator"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "v20_three_frequency_active_rl_audit_20260729_run01"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--projected", type=Path, default=DEFAULT_PROJECTED)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--low-operator", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--high-operator", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
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


def port_class(ix: int, iy: int) -> str:
    x_edge = ix in (0, 15)
    y_edge = iy in (0, 15)
    if x_edge and y_edge:
        return "corner"
    if x_edge or y_edge:
        return "edge"
    return "interior"


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - float(np.min(values))) / max(float(np.ptp(values)), EPS)


def source_active_return(
    s_matrix: np.ndarray,
    excitation: np.ndarray,
    mask: np.ndarray,
    relative_db: float | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(excitation, dtype=np.complex128)
    value /= max(float(np.linalg.norm(value)), EPS)
    reflected = s_matrix @ value
    amplitude = np.abs(value)
    maximum = max(float(np.max(amplitude)), EPS)
    if relative_db is None:
        considered = mask & (amplitude >= maximum * 1.0e-8)
    else:
        considered = mask & (amplitude >= maximum * 10.0 ** (relative_db / 20.0))
    indices = np.flatnonzero(considered)
    if indices.size == 0:
        raise RuntimeError("No significant driven port in active-RL audit")
    gamma = np.zeros(256, dtype=np.float64)
    gamma[indices] = np.abs(reflected[indices] / value[indices])
    worst = int(indices[np.argmax(gamma[indices])])
    worst_rl = float(-20.0 * np.log10(max(float(gamma[worst]), 1.0e-30)))
    total_rl = float(
        -10.0 * np.log10(max(float(np.vdot(reflected, reflected).real), 1.0e-30))
    )
    metrics = {
        "evaluated_port_count": int(indices.size),
        "worst_port_index": worst,
        "worst_active_rl_db": worst_rl,
        "total_rl_db": total_rl,
        "gate10_pass": int(worst_rl >= 10.0 and total_rl >= 10.0),
        "reserve11_pass": int(worst_rl >= 11.0 and total_rl >= 11.0),
    }
    return metrics, gamma, value, reflected


def operator_bundle(path: Path, expected_frequency: float) -> tuple[dict[str, Any], dict[str, Any]]:
    base, effective, fast, s_matrix = load_nominal_operator(path)
    frequency = float(base["frequency_ghz"])
    if not np.isclose(frequency, expected_frequency, atol=1.0e-6):
        raise RuntimeError(f"Expected {expected_frequency} GHz operator, got {frequency}")
    return base, {"effective": effective, "fast": fast, "s": s_matrix}


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite audit: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    projected = load_npz(args.projected)
    source = load_npz(args.source)
    pool = load_npz(args.pool)
    nominal_base, nominal = operator_bundle(args.nominal_operator, 10.0)
    low_base, low = operator_bundle(args.low_operator, 9.96)
    high_base, high = operator_bundle(args.high_operator, 10.04)
    for other in (low_base, high_base):
        if not np.array_equal(nominal_base["element_ixiy"], other["element_ixiy"]):
            raise RuntimeError("Three-frequency operator port ordering differs")
        if not (
            np.array_equal(nominal_base["theta_deg"], other["theta_deg"])
            and np.array_equal(nominal_base["phi_deg"], other["phi_deg"])
        ):
            raise RuntimeError("Three-frequency EEP grids differ")

    commands = ri_to_complex(projected["selected_task_weights_real_imag"])
    samples = np.asarray(projected["sample_index"], dtype=np.int64)
    source_indices = np.asarray(projected["source_candidate_index"], dtype=np.int64)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    element_ixiy = np.asarray(nominal_base["element_ixiy"], dtype=np.int64)
    grid_dirs = pattern_grid_dirs(nominal_base["theta_deg"], nominal_base["phi_deg"])
    identity = identity_state()
    case_rows: list[dict[str, Any]] = []
    coupling_events: list[dict[str, Any]] = []
    candidate_port_rows: list[dict[str, Any]] = []
    repeated_port = defaultdict(lambda: {"cases": 0, "fail10": 0, "fail11": 0, "gamma": 0.0})
    repeated_pair = defaultdict(lambda: {"count": 0, "fraction_sum": 0.0, "fraction_max": 0.0})

    for candidate, sample in enumerate(samples):
        source_index = int(source_indices[candidate])
        k_value = int(projected["k_values"][candidate])
        ratio = float(projected["active_ratios_requested"][candidate])
        mask = np.asarray(projected["masks"][candidate], dtype=bool)
        targets = np.asarray(projected["targets_deg"][candidate, :k_value], dtype=float)
        command = np.asarray(commands[candidate, :, :k_value], dtype=np.complex64)
        pool_scene = np.flatnonzero(pool_samples == int(sample))
        low_state = scene_calibration_states(
            pool,
            pool_scene,
            {"frequency_low_x0.20": {"profile": "frequency_low", "level": 0.20}},
            element_ixiy,
            int(protocol["seed"]),
        )["frequency_low_x0.20"]
        high_state = scene_calibration_states(
            pool,
            pool_scene,
            {"frequency_high_x0.20": {"profile": "frequency_high", "level": 0.20}},
            element_ixiy,
            int(protocol["seed"]),
        )["frequency_high_x0.20"]
        states = {
            "nominal_identity": (10.0, nominal, identity),
            "frequency_low_identity": (9.96, low, identity),
            "frequency_low_E2_source": (9.96, low, low_state),
            "frequency_high_identity": (10.04, high, identity),
            "frequency_high_E2_source": (10.04, high, high_state),
        }
        utility = np.zeros(256, dtype=np.float64)
        for corner in (nominal, low, high):
            steering = matched_steering_tasks(command, targets, corner["effective"], grid_dirs)
            utility += np.sum(np.abs(steering) ** 2, axis=1)
        utility = normalize01(utility)
        passive_burden = np.zeros(256, dtype=np.float64)
        for corner in (nominal, low, high):
            s_matrix = np.asarray(corner["s"], dtype=np.complex128)
            passive_burden += np.abs(np.diag(s_matrix))
            passive_burden += np.sqrt(np.sum(np.abs(s_matrix) ** 2, axis=0))
        passive_burden = normalize01(passive_burden)
        gamma_by_frequency = {9.96: np.zeros(256), 10.0: np.zeros(256), 10.04: np.zeros(256)}
        gamma_observed = np.zeros(256, dtype=np.int64)

        for state_name, (frequency, corner, state) in states.items():
            actual = apply_calibration(command, mask, state)
            sources = [("combined", -1, np.sum(actual, axis=1), None)]
            sources.extend(
                ("task", task_index, actual[:, task_index], -20.0)
                for task_index in range(k_value)
            )
            for source_kind, task_index, excitation, relative_db in sources:
                metrics, gamma, normalized, _reflected = source_active_return(
                    corner["s"], excitation, mask, relative_db
                )
                worst = int(metrics["worst_port_index"])
                ix, iy = (int(value) for value in element_ixiy[worst])
                base_row = {
                    "candidate_index": candidate,
                    "sample_index": int(sample),
                    "k": k_value,
                    "ratio": ratio,
                    "state": state_name,
                    "frequency_ghz": frequency,
                    "source_kind": source_kind,
                    "task_index": task_index,
                    **metrics,
                    "worst_port_ix": ix,
                    "worst_port_iy": iy,
                    "worst_port_class": port_class(ix, iy),
                }
                case_rows.append(base_row)
                considered = gamma > 0.0
                gamma_by_frequency[frequency] = np.maximum(
                    gamma_by_frequency[frequency], gamma
                )
                gamma_observed += considered.astype(np.int64)
                for port in np.flatnonzero(considered):
                    record = repeated_port[int(port)]
                    record["cases"] += 1
                    record["fail10"] += int(gamma[port] > 10.0 ** (-10.0 / 20.0))
                    record["fail11"] += int(gamma[port] > 10.0 ** (-11.0 / 20.0))
                    record["gamma"] = max(float(record["gamma"]), float(gamma[port]))

                contributions = np.abs(np.asarray(corner["s"])[worst] * normalized)
                total = max(float(np.sum(contributions)), EPS)
                order = np.argsort(contributions, kind="stable")[::-1][:8]
                for rank, source_port in enumerate(order, start=1):
                    fraction = float(contributions[source_port] / total)
                    sx, sy = (int(value) for value in element_ixiy[source_port])
                    coupling_events.append(
                        {
                            **base_row,
                            "coupling_rank": rank,
                            "source_port_index": int(source_port),
                            "source_port_ix": sx,
                            "source_port_iy": sy,
                            "source_port_class": port_class(sx, sy),
                            "contribution_fraction": fraction,
                            "s_abs": float(abs(np.asarray(corner["s"])[worst, source_port])),
                        }
                    )
                    if rank <= 3:
                        key = (worst, int(source_port))
                        repeated_pair[key]["count"] += 1
                        repeated_pair[key]["fraction_sum"] += fraction
                        repeated_pair[key]["fraction_max"] = max(
                            float(repeated_pair[key]["fraction_max"]), fraction
                        )

        frequency_stack = np.stack([gamma_by_frequency[value] for value in (9.96, 10.0, 10.04)])
        active_stress = np.max(frequency_stack, axis=0) / (10.0 ** (-11.0 / 20.0))
        frequency_span = np.ptp(frequency_stack, axis=0)
        stress_normalized = normalize01(np.minimum(active_stress, 4.0))
        span_normalized = normalize01(frequency_span)
        keep_score = (
            0.55 * utility
            - 0.30 * stress_normalized
            - 0.10 * span_normalized
            - 0.05 * passive_burden
        )
        add_score = 0.70 * utility - 0.20 * passive_burden - 0.10 * span_normalized
        for port in range(256):
            ix, iy = (int(value) for value in element_ixiy[port])
            candidate_port_rows.append(
                {
                    "candidate_index": candidate,
                    "sample_index": int(sample),
                    "k": k_value,
                    "ratio": ratio,
                    "port_index": port,
                    "ix": ix,
                    "iy": iy,
                    "port_class": port_class(ix, iy),
                    "active": int(mask[port]),
                    "observed_source_count": int(gamma_observed[port]),
                    "worst_gamma_over_three_frequencies": float(np.max(frequency_stack[:, port])),
                    "worst_active_rl_db": float(
                        -20.0
                        * np.log10(max(float(np.max(frequency_stack[:, port])), 1.0e-30))
                    ),
                    "cross_frequency_gamma_span": float(frequency_span[port]),
                    "directional_utility": float(utility[port]),
                    "passive_coupling_burden": float(passive_burden[port]),
                    "keep_score": float(keep_score[port]),
                    "remove_priority": float(-keep_score[port]) if mask[port] else float("nan"),
                    "add_score": float(add_score[port]) if not mask[port] else float("nan"),
                }
            )

    port_rows: list[dict[str, Any]] = []
    for port in range(256):
        ix, iy = (int(value) for value in element_ixiy[port])
        record = repeated_port[port]
        cases = max(int(record["cases"]), 1)
        port_rows.append(
            {
                "port_index": port,
                "ix": ix,
                "iy": iy,
                "port_class": port_class(ix, iy),
                "observed_case_count": int(record["cases"]),
                "active_rl_fail10_count": int(record["fail10"]),
                "active_rl_fail11_count": int(record["fail11"]),
                "fail10_rate_when_observed": float(record["fail10"] / cases),
                "fail11_rate_when_observed": float(record["fail11"] / cases),
                "worst_gamma": float(record["gamma"]),
                "worst_active_rl_db": float(
                    -20.0 * np.log10(max(float(record["gamma"]), 1.0e-30))
                ),
            }
        )
    pair_rows: list[dict[str, Any]] = []
    for (destination, source_port), record in repeated_pair.items():
        dx, dy = (int(value) for value in element_ixiy[destination])
        sx, sy = (int(value) for value in element_ixiy[source_port])
        pair_rows.append(
            {
                "destination_port": destination,
                "destination_ix": dx,
                "destination_iy": dy,
                "destination_class": port_class(dx, dy),
                "source_port": source_port,
                "source_ix": sx,
                "source_iy": sy,
                "source_class": port_class(sx, sy),
                "top3_occurrence_count": int(record["count"]),
                "mean_contribution_fraction": float(record["fraction_sum"] / record["count"]),
                "max_contribution_fraction": float(record["fraction_max"]),
                "manhattan_distance": abs(dx - sx) + abs(dy - sy),
            }
        )
    pair_rows.sort(
        key=lambda row: (int(row["top3_occurrence_count"]), float(row["max_contribution_fraction"])),
        reverse=True,
    )
    port_rows.sort(
        key=lambda row: (int(row["active_rl_fail11_count"]), float(row["worst_gamma"])),
        reverse=True,
    )
    write_csv(args.out_dir / "active_rl_case_audit.csv", case_rows)
    write_csv(args.out_dir / "candidate_port_swap_scores.csv", candidate_port_rows)
    write_csv(args.out_dir / "repeated_high_risk_ports.csv", port_rows)
    write_csv(args.out_dir / "coupling_channel_events.csv", coupling_events)
    write_csv(args.out_dir / "repeated_coupling_channels.csv", pair_rows)

    state_summary: list[dict[str, Any]] = []
    for state_name in protocol["evaluated_states"]:
        for k_value in (0, 2, 4, 6):
            members = [
                row
                for row in case_rows
                if row["state"] == state_name
                and (k_value == 0 or int(row["k"]) == k_value)
            ]
            candidate_keys = {
                int(row["candidate_index"])
                for row in members
            }
            pass10 = 0
            pass11 = 0
            floors: list[float] = []
            for candidate in candidate_keys:
                cases = [row for row in members if int(row["candidate_index"]) == candidate]
                floor = min(
                    min(float(row["worst_active_rl_db"]), float(row["total_rl_db"]))
                    for row in cases
                )
                floors.append(floor)
                pass10 += int(floor >= 10.0)
                pass11 += int(floor >= 11.0)
            if floors:
                state_summary.append(
                    {
                        "state": state_name,
                        "k": "all" if k_value == 0 else k_value,
                        "candidate_count": len(floors),
                        "gate10_count": pass10,
                        "gate10_rate": pass10 / len(floors),
                        "reserve11_count": pass11,
                        "reserve11_rate": pass11 / len(floors),
                        "active_rl_floor_min_db": min(floors),
                        "active_rl_floor_mean_db": float(np.mean(floors)),
                        "active_rl_floor_max_db": max(floors),
                    }
                )
    write_csv(args.out_dir / "active_rl_state_k_summary.csv", state_summary)
    class_counts = Counter(
        str(row["worst_port_class"])
        for row in case_rows
        if float(row["worst_active_rl_db"]) < 11.0
    )
    summary = {
        "protocol": protocol["protocol"],
        "candidate_count": len(samples),
        "case_count": len(case_rows),
        "state_count": len(protocol["evaluated_states"]),
        "combined_and_significant_tasks_audited": True,
        "active_rl_engineering_gate_db": 10.0,
        "active_rl_design_reserve_db": 11.0,
        "worst_port_class_counts_below_11db": dict(class_counts),
        "highest_risk_ports": port_rows[:20],
        "highest_recurrence_coupling_channels": pair_rows[:20],
        "swap_score": {
            "keep_score": "0.55*directional_utility - 0.30*active_stress - 0.10*frequency_span - 0.05*passive_burden",
            "add_score": "0.70*directional_utility - 0.20*passive_burden - 0.10*frequency_span",
        },
    }
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# v20 Three-Frequency Active-RL Sensitivity Audit",
        "",
        f"- Frozen candidates: {len(samples)}.",
        f"- Combined/significant-task cases: {len(case_rows)}.",
        f"- Below-11-dB worst-port classes: {dict(class_counts)}.",
        "- The candidate-specific swap score combines target-direction utility, active-reflection stress, cross-frequency variation, and passive coupling burden.",
        "- No HFSS labels or critic training are enabled by this audit.",
    ]
    (args.out_dir / "AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
