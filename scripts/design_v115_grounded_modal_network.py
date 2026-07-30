#!/usr/bin/env python3
"""Synthesize and gate a finite-Q dual-reference-plane network on trusted S4."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from run_v114_small_cell_broadband_feed import load_stimuli, parse_touchstone


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v115_grounded_modal_network.json"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v115_grounded_modal_network_20260730_run01"
EPS = 1.0e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--maxiter", type=int, default=0)
    parser.add_argument("--popsize", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def s_to_z(matrix: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=complex)
    return z0 * (identity + matrix) @ np.linalg.inv(identity - matrix)


def z_to_s(matrix: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=complex)
    return (matrix - z0 * identity) @ np.linalg.inv(matrix + z0 * identity)


def y_to_s(matrix: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=complex)
    return (identity - z0 * matrix) @ np.linalg.inv(identity + z0 * matrix)


def frequency_scaled(value: float, ratio: float) -> float:
    return value * ratio if value >= 0.0 else value / ratio


def network_matrices(
    parameters: np.ndarray,
    frequency_ghz: float,
    z0: float,
    pairs: list[tuple[int, int]],
    q_series: float,
    q_ground: float,
    q_bridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio = frequency_ghz / 10.0
    series_x = frequency_scaled(float(parameters[0]), ratio)
    ground_b = frequency_scaled(float(parameters[1]), ratio)
    bridge_b = frequency_scaled(float(parameters[2]), ratio)
    series_z = np.diag(np.full(4, abs(series_x) / q_series + 1j * series_x))
    shunt_y = np.diag(np.full(4, abs(ground_b) / q_ground + 1j * ground_b))
    bridge_y = abs(bridge_b) / q_bridge + 1j * bridge_b
    for first, second in pairs:
        shunt_y[first, first] += bridge_y
        shunt_y[second, second] += bridge_y
        shunt_y[first, second] -= bridge_y
        shunt_y[second, first] -= bridge_y
    series_y = np.linalg.inv(series_z)
    network_y = np.block(
        [[series_y, -series_y], [-series_y, series_y + shunt_y]]
    )
    return series_z, shunt_y, y_to_s(network_y, z0)


def terminate_network(
    network_s: np.ndarray, antenna_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s11 = network_s[:4, :4]
    s12 = network_s[:4, 4:]
    s21 = network_s[4:, :4]
    s22 = network_s[4:, 4:]
    identity = np.eye(4, dtype=complex)
    antenna_incident_map = np.linalg.solve(identity - s22 @ antenna_s, s21)
    antenna_reflected_map = antenna_s @ antenna_incident_map
    external_s = s11 + s12 @ antenna_reflected_map
    return external_s, antenna_incident_map, antenna_reflected_map


def direct_terminated_model(
    antenna_z: np.ndarray,
    series_z: np.ndarray,
    shunt_y: np.ndarray,
    z0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    antenna_y = np.linalg.inv(antenna_z)
    node_z = np.linalg.inv(antenna_y + shunt_y)
    external_s = z_to_s(series_z + node_z, z0)
    return external_s, node_z, antenna_y


def component_description(parameters: np.ndarray) -> dict[str, Any]:
    omega = 2.0 * math.pi * 10.0e9
    series_x, ground_b, bridge_b = (float(value) for value in parameters)
    series = (
        {"kind": "series_inductor", "value_h": series_x / omega}
        if series_x >= 0.0
        else {"kind": "series_capacitor", "value_f": -1.0 / (omega * series_x)}
    )
    ground = (
        {"kind": "shunt_capacitor_to_ground", "value_f": ground_b / omega}
        if ground_b >= 0.0
        else {"kind": "grounded_inductive_stub", "equivalent_inductance_h": -1.0 / (omega * ground_b)}
    )
    bridge = (
        {"kind": "x_pair_bridge_capacitor", "value_f": bridge_b / omega}
        if bridge_b >= 0.0
        else {"kind": "x_pair_inductive_bridge", "equivalent_inductance_h": -1.0 / (omega * bridge_b)}
    )
    return {"series_per_port": series, "ground_branch_per_port": ground, "bridge_per_x_pair": bridge}


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.15 output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    shutil.copy2(args.protocol, args.out_dir / "protocol_snapshot.json")
    z0 = float(protocol["reference_impedance_ohm"])
    antenna_path = ROOT / protocol["trusted_antenna_s4"]
    frequencies, antenna_s = parse_touchstone(antenna_path, 4)
    target_frequencies = np.asarray(protocol["frequencies_ghz"], dtype=float)
    if not np.allclose(frequencies, target_frequencies, atol=1.0e-9):
        raise RuntimeError("Trusted S4 frequency grid differs from the preregistered grid")
    antenna_z = np.stack([s_to_z(matrix, z0) for matrix in antenna_s])

    stimulus_root = ROOT / protocol["trusted_stimulus_root"]
    stimulus_rows, vectors, considered = load_stimuli(stimulus_root)
    side_mask = np.asarray([int(row["side"]) == 2 for row in stimulus_rows])
    stimulus_rows = [row for row, keep in zip(stimulus_rows, side_mask) if keep]
    vectors = vectors[side_mask, :4]
    considered = considered[side_mask, :4]
    groups = [
        np.asarray(
            [index for index, row in enumerate(stimulus_rows) if abs(float(row["frequency_ghz"]) - frequency) <= 1.0e-9],
            dtype=int,
        )
        for frequency in frequencies
    ]
    pairs = [tuple(int(value) for value in pair) for pair in protocol["network"]["x_neighbor_pairs_zero_based"]]
    q_series = float(protocol["network"]["series_q"])
    q_ground = float(protocol["network"]["ground_capacitor_q"])
    q_bridge = float(protocol["network"]["bridge_inductor_q"])
    gates = protocol["gates"]

    def evaluate(parameters: np.ndarray, detailed: bool = False) -> dict[str, Any]:
        passive_values: list[float] = []
        active_values: list[float] = []
        total_values: list[float] = []
        insertion_values: list[float] = []
        transducer_values: list[float] = []
        matched_values: list[float] = []
        cascade_errors: list[float] = []
        frequency_rows: list[dict[str, Any]] = []
        source_output: list[dict[str, Any]] = []
        external_matrices: list[np.ndarray] = []
        network_matrices_all: list[np.ndarray] = []
        incident_maps: list[np.ndarray] = []
        reflected_maps: list[np.ndarray] = []
        for frequency_index, (frequency, s_load, z_load) in enumerate(zip(frequencies, antenna_s, antenna_z)):
            series_z, shunt_y, network_s = network_matrices(
                parameters, float(frequency), z0, pairs, q_series, q_ground, q_bridge
            )
            external_s, incident_map, reflected_map = terminate_network(network_s, s_load)
            direct_s, node_z, antenna_y = direct_terminated_model(z_load, series_z, shunt_y, z0)
            cascade_error = float(np.max(np.abs(external_s - direct_s)))
            passive_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(external_s)), EPS))
            indices = groups[frequency_index]
            sources = vectors[indices].T
            active_mask = considered[indices].T
            reflected = external_s @ sources
            gamma = np.where(
                active_mask,
                np.abs(reflected) / np.maximum(np.abs(sources), EPS),
                0.0,
            )
            active_rl = -20.0 * np.log10(np.maximum(np.max(gamma, axis=0), EPS))
            incident_power = np.sum(np.abs(sources) ** 2, axis=0)
            reflected_power = np.sum(np.abs(reflected) ** 2, axis=0)
            total_rl = -10.0 * np.log10(np.maximum(reflected_power / incident_power, EPS))
            antenna_incident = incident_map @ sources
            antenna_reflected = reflected_map @ sources
            external_accepted = incident_power - reflected_power
            antenna_accepted = np.sum(np.abs(antenna_incident) ** 2, axis=0) - np.sum(
                np.abs(antenna_reflected) ** 2, axis=0
            )
            insertion = antenna_accepted / np.maximum(external_accepted, EPS)
            transducer = antenna_accepted / np.maximum(incident_power, EPS)

            matched_s, matched_incident, matched_reflected = terminate_network(
                network_s, np.zeros((4, 4), dtype=complex)
            )
            canonical = np.eye(4, dtype=complex)
            matched_return = matched_s @ canonical
            matched_external_accepted = 1.0 - np.sum(np.abs(matched_return) ** 2, axis=0)
            matched_delivered = np.sum(np.abs(matched_incident @ canonical) ** 2, axis=0) - np.sum(
                np.abs(matched_reflected @ canonical) ** 2, axis=0
            )
            matched_efficiency = matched_delivered / np.maximum(matched_external_accepted, EPS)

            passive_values.extend(passive_rl.tolist())
            active_values.extend(active_rl.tolist())
            total_values.extend(total_rl.tolist())
            insertion_values.extend(insertion.tolist())
            transducer_values.extend(transducer.tolist())
            matched_values.extend(matched_efficiency.tolist())
            cascade_errors.append(cascade_error)
            frequency_rows.append(
                {
                    "frequency_ghz": float(frequency),
                    "passive_rl_min_db": float(np.min(passive_rl)),
                    "representative_active_rl_min_db": float(np.min(active_rl)),
                    "representative_total_rl_min_db": float(np.min(total_rl)),
                    "actual_load_insertion_efficiency_min": float(np.min(insertion)),
                    "actual_load_transducer_efficiency_min": float(np.min(transducer)),
                    "matched_load_network_efficiency_min": float(np.min(matched_efficiency)),
                    "reference_plane_cascade_error": cascade_error,
                }
            )
            if detailed:
                for local_index, global_index in enumerate(indices):
                    source_output.append(
                        {
                            **stimulus_rows[global_index],
                            "active_rl_db": float(active_rl[local_index]),
                            "total_rl_db": float(total_rl[local_index]),
                            "external_incident_power": float(incident_power[local_index]),
                            "external_accepted_power": float(external_accepted[local_index]),
                            "antenna_accepted_power": float(antenna_accepted[local_index]),
                            "network_dissipated_power": float(external_accepted[local_index] - antenna_accepted[local_index]),
                            "actual_load_insertion_efficiency": float(insertion[local_index]),
                            "actual_load_transducer_efficiency": float(transducer[local_index]),
                        }
                    )
            external_matrices.append(external_s)
            network_matrices_all.append(network_s)
            incident_maps.append(incident_map)
            reflected_maps.append(reflected_map)
        return {
            "passive_rl_min_db": float(min(passive_values)),
            "active_rl_min_db": float(min(active_values)),
            "total_rl_min_db": float(min(total_values)),
            "actual_load_insertion_efficiency_min": float(min(insertion_values)),
            "actual_load_transducer_efficiency_min": float(min(transducer_values)),
            "matched_load_network_efficiency_min": float(min(matched_values)),
            "reference_plane_cascade_error_max": float(max(cascade_errors)),
            "frequency_rows": frequency_rows,
            "source_rows": source_output,
            "external_s": np.stack(external_matrices),
            "network_s8": np.stack(network_matrices_all),
            "antenna_incident_map": np.stack(incident_maps),
            "antenna_reflected_map": np.stack(reflected_maps),
        }

    def objective(parameters: np.ndarray) -> float:
        result = evaluate(parameters)
        return float(
            60.0 * max(float(gates["minimum_representative_active_rl_db"]) - result["active_rl_min_db"], 0.0) ** 2
            + 25.0 * max(float(gates["minimum_representative_total_rl_db"]) - result["total_rl_min_db"], 0.0) ** 2
            + 10.0 * max(float(gates["minimum_passive_rl_db"]) - result["passive_rl_min_db"], 0.0) ** 2
            + 3000.0 * max(float(gates["minimum_actual_load_insertion_efficiency"]) - result["actual_load_insertion_efficiency_min"], 0.0) ** 2
            + 3000.0 * max(float(gates["minimum_matched_load_network_efficiency"]) - result["matched_load_network_efficiency_min"], 0.0) ** 2
            - 2.0 * min(result["active_rl_min_db"], 13.0)
            - min(result["total_rl_min_db"], 13.0)
            - 0.2 * min(result["passive_rl_min_db"], 15.0)
        )

    variant_results: list[dict[str, Any]] = []
    detailed_by_variant: dict[str, dict[str, Any]] = {}
    maxiter = int(args.maxiter or protocol["optimization"]["differential_evolution_maxiter"])
    popsize = int(args.popsize or protocol["optimization"]["differential_evolution_popsize"])
    for variant_index, (variant, bounds_map) in enumerate(protocol["network"]["variants"].items()):
        bounds = [
            bounds_map["series_reactance_ohm_at_10ghz"],
            bounds_map["ground_susceptance_siemens_at_10ghz"],
            bounds_map["bridge_susceptance_siemens_at_10ghz"],
        ]
        optimized = differential_evolution(
            objective,
            bounds=bounds,
            seed=int(protocol["seed"]) + variant_index,
            popsize=popsize,
            maxiter=maxiter,
            tol=float(protocol["optimization"]["tolerance"]),
            polish=True,
        )
        result = evaluate(np.asarray(optimized.x), detailed=True)
        tolerance = float(protocol["optimization"]["component_corner_fraction"])
        corner_metrics = [
            evaluate(np.asarray(optimized.x) * np.asarray(scales))
            for scales in itertools.product((1.0 - tolerance, 1.0 + tolerance), repeat=3)
        ]
        summary = {
            "variant": variant,
            "objective": float(optimized.fun),
            "series_reactance_ohm_at_10ghz": float(optimized.x[0]),
            "ground_susceptance_siemens_at_10ghz": float(optimized.x[1]),
            "bridge_susceptance_siemens_at_10ghz": float(optimized.x[2]),
            **{key: value for key, value in result.items() if not isinstance(value, (list, np.ndarray))},
            "component_corner_active_rl_min_db": min(item["active_rl_min_db"] for item in corner_metrics),
            "component_corner_total_rl_min_db": min(item["total_rl_min_db"] for item in corner_metrics),
            "component_corner_insertion_efficiency_min": min(item["actual_load_insertion_efficiency_min"] for item in corner_metrics),
        }
        summary["nominal_gate_pass"] = bool(
            summary["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
            and summary["active_rl_min_db"] >= float(gates["minimum_representative_active_rl_db"])
            and summary["total_rl_min_db"] >= float(gates["minimum_representative_total_rl_db"])
            and summary["actual_load_insertion_efficiency_min"] >= float(gates["minimum_actual_load_insertion_efficiency"])
            and summary["matched_load_network_efficiency_min"] >= float(gates["minimum_matched_load_network_efficiency"])
            and summary["reference_plane_cascade_error_max"] <= float(gates["maximum_reference_plane_cascade_error"])
        )
        summary["component_corner_gate_pass"] = bool(
            summary["component_corner_active_rl_min_db"] >= float(gates["minimum_component_corner_active_rl_db"])
            and summary["component_corner_insertion_efficiency_min"] >= float(gates["minimum_component_corner_insertion_efficiency"])
        )
        summary["components"] = component_description(np.asarray(optimized.x))
        variant_results.append(summary)
        detailed_by_variant[variant] = {"parameters": np.asarray(optimized.x), "result": result}

    priority = protocol["optimization"]["physical_selection_priority"]
    passing = [row for row in variant_results if row["nominal_gate_pass"] and row["component_corner_gate_pass"]]
    selected = min(passing, key=lambda row: priority.index(row["variant"])) if passing else None
    write_csv(
        args.out_dir / "variant_summary.csv",
        [{key: value for key, value in row.items() if key != "components"} for row in variant_results],
    )
    write_json(args.out_dir / "variant_components.json", {row["variant"]: row["components"] for row in variant_results})
    if selected is None:
        write_json(args.out_dir / "stage_decision.json", {"decision": "stop_no_circuit_candidate_passed", "allow_physical_2x2": False})
        raise RuntimeError("No v1.15 circuit candidate passed nominal and component-corner gates")

    selected_payload = detailed_by_variant[str(selected["variant"])]
    selected_result = selected_payload["result"]
    write_csv(args.out_dir / "selected_frequency_metrics.csv", selected_result["frequency_rows"])
    write_csv(args.out_dir / "selected_stimulus_metrics.csv", selected_result["source_rows"])
    write_json(args.out_dir / "selected_network.json", selected)
    np.savez_compressed(
        args.out_dir / "selected_dual_reference_plane_network.npz",
        frequencies_ghz=frequencies,
        parameters_at_10ghz=selected_payload["parameters"],
        network_s8=selected_result["network_s8"],
        external_s4=selected_result["external_s"],
        antenna_incident_map=selected_result["antenna_incident_map"],
        antenna_reflected_map=selected_result["antenna_reflected_map"],
        pre_network_port_names=np.asarray([f"PRE_{index}" for index in range(4)]),
        post_network_port_names=np.asarray([f"POST_{index}" for index in range(4)]),
        reference_impedance_ohm=np.asarray(z0),
    )
    decision = {
        "decision": "allow_one_physical_2x2_grounded_modal_feed",
        "selected_variant": selected["variant"],
        "circuit_nominal_gate_pass": True,
        "component_corner_gate_pass": True,
        "allow_physical_2x2": True,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_hfss_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(args.out_dir / "stage_decision.json", decision)
    print(json.dumps({**decision, "selected_metrics": selected}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
