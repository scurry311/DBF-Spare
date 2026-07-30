#!/usr/bin/env python3
"""Calibrate the distributed v1.15 S8 fixture and resynthesize the same topology."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from design_v115_grounded_modal_network import terminate_network
from run_v114_small_cell_broadband_feed import load_stimuli, parse_touchstone
from run_v115_physical_modal_feed_fixture import touchstone_port_names


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v115_grounded_modal_network.json"
DEFAULT_PHYSICAL = (
    ROOT
    / "hfss_outputs"
    / "v115_physical_modal_feed_fixture_20260730_run02"
    / "physical_s8_direct01"
    / "v115_physical_modal_feed_s8_direct01.s8p"
)
DEFAULT_CIRCUIT = ROOT / "hfss_outputs" / "v115_grounded_modal_network_20260730_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v115_physical_aware_resynthesis_20260730_run01"
EPS = 1.0e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--physical-s8", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--circuit-dir", type=Path, default=DEFAULT_CIRCUIT)
    parser.add_argument("--initial-physical-design", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def network_from_nodal_y(matrix: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=complex)
    return (identity - z0 * matrix) @ np.linalg.inv(identity + z0 * matrix)


def reduced_network(
    frequency_ghz: float,
    components: np.ndarray,
    lengths_mm: np.ndarray,
    propagation: np.ndarray,
    protocol: dict[str, Any],
) -> np.ndarray:
    series_l = float(components[0]) * 1.0e-9
    ground_c = float(components[1]) * 1.0e-12
    bridge_l = float(components[2]) * 1.0e-9
    zc, beta_10, alpha_10 = (float(value) for value in propagation)
    ratio = frequency_ghz / 10.0
    omega = 2.0 * math.pi * frequency_ghz * 1.0e9
    q_series = float(protocol["network"]["series_q"])
    q_ground = float(protocol["network"]["ground_capacitor_q"])
    q_bridge = float(protocol["network"]["bridge_inductor_q"])
    series_z = omega * series_l / q_series + 1j * omega * series_l
    ground_y = omega * ground_c / q_ground + 1j * omega * ground_c
    bridge_y = 1.0 / (q_bridge * omega * bridge_l) + 1.0 / (1j * omega * bridge_l)
    nodal = np.zeros((24, 24), dtype=complex)

    def add_branch(first: int, second: int, admittance: complex) -> None:
        nodal[first, first] += admittance
        nodal[second, second] += admittance
        nodal[first, second] -= admittance
        nodal[second, first] -= admittance

    def add_line(first: int, second: int, length_mm: float) -> None:
        gamma_l = (alpha_10 * math.sqrt(ratio) + 1j * beta_10 * ratio) * length_mm
        diagonal = (1.0 / zc) / np.tanh(gamma_l)
        transfer = -(1.0 / zc) / np.sinh(gamma_l)
        nodal[first, first] += diagonal
        nodal[second, second] += diagonal
        nodal[first, second] += transfer
        nodal[second, first] += transfer

    for channel in range(4):
        pre = channel
        post = 4 + channel
        first = 8 + 4 * channel
        second = first + 1
        cap_node = first + 2
        bridge_node = first + 3
        add_line(pre, first, float(lengths_mm[0]))
        add_branch(first, second, 1.0 / series_z)
        add_line(second, cap_node, float(lengths_mm[1]))
        add_line(cap_node, bridge_node, float(lengths_mm[2]))
        add_line(bridge_node, post, float(lengths_mm[3]))
        nodal[cap_node, cap_node] += ground_y
    for first_port, second_port in ((0, 2), (1, 3)):
        add_branch(8 + 4 * first_port + 3, 8 + 4 * second_port + 3, bridge_y)
    external = slice(0, 8)
    internal = slice(8, 24)
    reduced_y = nodal[external, external] - nodal[external, internal] @ np.linalg.solve(
        nodal[internal, internal], nodal[internal, external]
    )
    return network_from_nodal_y(reduced_y, float(protocol["reference_impedance_ohm"]))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite physical-aware resynthesis: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    selected = json.loads((args.circuit_dir / "selected_network.json").read_text(encoding="utf-8"))
    frequencies, physical_s8 = parse_touchstone(args.physical_s8, 8)
    exported_names = touchstone_port_names(args.physical_s8)
    desired_names = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    permutation = [exported_names.index(name) for name in desired_names]
    physical_s8 = physical_s8[:, permutation][:, :, permutation]
    initial_physical = (
        json.loads(args.initial_physical_design.read_text(encoding="utf-8"))
        if args.initial_physical_design
        else None
    )
    if initial_physical:
        values = initial_physical["physical_component_values"]
        components_initial = np.asarray(
            [values["series_inductor_nh"], values["ground_capacitor_pf"], values["bridge_inductor_nh"]]
        )
        fixture = {
            **protocol["physical_fixture"],
            **initial_physical["physical_fixture_override"],
        }
    else:
        components_initial = np.asarray(
            [
                selected["components"]["series_per_port"]["value_h"] * 1.0e9,
                selected["components"]["ground_branch_per_port"]["value_f"] * 1.0e12,
                selected["components"]["bridge_per_x_pair"]["equivalent_inductance_h"] * 1.0e9,
            ]
        )
        fixture = protocol["physical_fixture"]
    gap_center = float(fixture["series_gap_center_x_mm"])
    gap_half = float(fixture["series_gap_length_mm"]) / 2.0
    lengths_initial = np.asarray(
        [
            gap_center - gap_half - float(fixture["pre_reference_x_mm"]),
            float(fixture["ground_cap_x_mm"]) - (gap_center + gap_half),
            float(fixture["bridge_x_mm"]) - float(fixture["ground_cap_x_mm"]),
            float(fixture["post_reference_x_mm"]) - float(fixture["bridge_x_mm"]),
        ]
    )

    def fit_objective(parameters: np.ndarray) -> float:
        predicted = np.stack(
            [
                reduced_network(frequency, components_initial, lengths_initial, parameters, protocol)
                for frequency in frequencies
            ]
        )
        return float(np.mean(np.abs(predicted - physical_s8) ** 2))

    fit = differential_evolution(
        fit_objective,
        bounds=[(30.0, 100.0), (0.15, 0.60), (0.0, 0.10)],
        seed=int(protocol["seed"]) + 20,
        popsize=20,
        maxiter=500,
        tol=1.0e-10,
        polish=True,
    )
    propagation = np.asarray(fit.x)
    fitted_s8 = np.stack(
        [
            reduced_network(frequency, components_initial, lengths_initial, propagation, protocol)
            for frequency in frequencies
        ]
    )
    fit_max = float(np.max(np.abs(fitted_s8 - physical_s8)))
    fit_rms = float(np.sqrt(np.mean(np.abs(fitted_s8 - physical_s8) ** 2)))
    if fit_max > 0.05:
        write_json(args.out_dir / "stage_decision.json", {"decision": "stop_distributed_surrogate_fit_failed", "fit_max_abs_delta_s": fit_max})
        raise RuntimeError(f"Distributed fixture fit misses the 0.05 gate: {fit_max}")

    antenna_frequencies, antenna_s4 = parse_touchstone(ROOT / protocol["trusted_antenna_s4"], 4)
    rows, vectors, considered = load_stimuli(ROOT / protocol["trusted_stimulus_root"])
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    groups = [
        np.asarray([index for index, row in enumerate(rows) if abs(float(row["frequency_ghz"]) - frequency) <= 1.0e-9])
        for frequency in frequencies
    ]

    def evaluate(parameters: np.ndarray) -> dict[str, float]:
        components = parameters[:3]
        lengths = np.r_[0.2, parameters[3:]]
        minima = np.full(6, np.inf)
        for frequency_index, frequency in enumerate(frequencies):
            network_s8 = reduced_network(frequency, components, lengths, propagation, protocol)
            external_s, incident_map, reflected_map = terminate_network(network_s8, antenna_s4[frequency_index])
            minima[0] = min(
                minima[0],
                float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(external_s)), EPS)))),
            )
            indices = groups[frequency_index]
            sources = vectors[indices].T
            active = considered[indices].T
            reflected = external_s @ sources
            gamma = np.where(active, np.abs(reflected) / np.maximum(np.abs(sources), EPS), 0.0)
            minima[1] = min(minima[1], float(np.min(-20.0 * np.log10(np.maximum(np.max(gamma, axis=0), EPS)))))
            incident_power = np.sum(np.abs(sources) ** 2, axis=0)
            reflected_power = np.sum(np.abs(reflected) ** 2, axis=0)
            minima[2] = min(minima[2], float(np.min(-10.0 * np.log10(np.maximum(reflected_power / incident_power, EPS)))))
            antenna_incident = incident_map @ sources
            antenna_reflected = reflected_map @ sources
            external_accepted = incident_power - reflected_power
            antenna_accepted = np.sum(np.abs(antenna_incident) ** 2, axis=0) - np.sum(np.abs(antenna_reflected) ** 2, axis=0)
            minima[3] = min(minima[3], float(np.min(antenna_accepted / np.maximum(external_accepted, EPS))))
            minima[4] = min(minima[4], float(np.min(antenna_accepted / incident_power)))
            matched_s, matched_incident, matched_reflected = terminate_network(network_s8, np.zeros((4, 4), dtype=complex))
            matched_external_accepted = 1.0 - np.sum(np.abs(matched_s) ** 2, axis=0)
            matched_delivered = np.sum(np.abs(matched_incident) ** 2, axis=0) - np.sum(np.abs(matched_reflected) ** 2, axis=0)
            minima[5] = min(minima[5], float(np.min(matched_delivered / np.maximum(matched_external_accepted, EPS))))
        keys = (
            "passive_rl_min_db",
            "active_rl_min_db",
            "total_rl_min_db",
            "actual_load_insertion_efficiency_min",
            "actual_load_transducer_efficiency_min",
            "matched_load_network_efficiency_min",
        )
        return dict(zip(keys, minima.tolist()))

    gates = protocol["gates"]

    def objective(parameters: np.ndarray) -> float:
        metrics = evaluate(parameters)
        design_active_target = float(
            gates.get(
                "minimum_physical_aware_design_active_rl_db",
                gates["minimum_representative_active_rl_db"],
            )
        )
        return float(
            120.0 * max(design_active_target - metrics["active_rl_min_db"], 0.0) ** 2
            + 25.0 * max(float(gates["minimum_representative_total_rl_db"]) - metrics["total_rl_min_db"], 0.0) ** 2
            + 10.0 * max(float(gates["minimum_passive_rl_db"]) - metrics["passive_rl_min_db"], 0.0) ** 2
            + 4000.0 * max(float(gates["minimum_actual_load_insertion_efficiency"]) - metrics["actual_load_insertion_efficiency_min"], 0.0) ** 2
            + 4000.0 * max(float(gates["minimum_matched_load_network_efficiency"]) - metrics["matched_load_network_efficiency_min"], 0.0) ** 2
            - 2.0 * min(metrics["active_rl_min_db"], 13.0)
            - min(metrics["total_rl_min_db"], 13.0)
            - 0.2 * min(metrics["passive_rl_min_db"], 15.0)
        )

    optimized = differential_evolution(
        objective,
        bounds=[(0.05, 2.0), (0.01, 1.0), (0.2, 12.0), (0.1, 1.5), (0.1, 1.5), (0.1, 1.5)],
        seed=int(protocol["seed"]) + 30,
        popsize=14,
        maxiter=350,
        tol=1.0e-7,
        polish=True,
    )
    nominal = evaluate(np.asarray(optimized.x))
    corner_metrics = [
        evaluate(np.asarray(optimized.x) * np.r_[scales, (1.0, 1.0, 1.0)])
        for scales in itertools.product((0.95, 1.05), repeat=3)
    ]
    corner_active = min(item["active_rl_min_db"] for item in corner_metrics)
    corner_insertion = min(item["actual_load_insertion_efficiency_min"] for item in corner_metrics)
    gate_pass = bool(
        nominal["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
        and nominal["active_rl_min_db"] >= float(
            gates.get(
                "minimum_physical_aware_design_active_rl_db",
                gates["minimum_representative_active_rl_db"],
            )
        )
        and nominal["total_rl_min_db"] >= float(gates["minimum_representative_total_rl_db"])
        and nominal["actual_load_insertion_efficiency_min"] >= float(gates["minimum_actual_load_insertion_efficiency"])
        and nominal["matched_load_network_efficiency_min"] >= float(gates["minimum_matched_load_network_efficiency"])
        and corner_active >= float(gates["minimum_component_corner_active_rl_db"])
        and corner_insertion >= float(gates["minimum_component_corner_insertion_efficiency"])
    )
    components = np.asarray(optimized.x[:3])
    internal_lengths = np.asarray(optimized.x[3:])
    gap_center = -1.4
    gap_half = float(fixture["series_gap_length_mm"]) / 2.0
    pre_x = gap_center - gap_half - 0.2
    right_start = gap_center + gap_half
    cap_x = right_start + internal_lengths[0]
    bridge_x = cap_x + internal_lengths[1]
    post_x = bridge_x + internal_lengths[2]
    physical_design = {
        "variant": "physical_aware_grounded_lowpass_modal",
        "calibration_source": str(args.physical_s8.resolve()),
        "initial_physical_design": (
            str(args.initial_physical_design.resolve())
            if args.initial_physical_design
            else None
        ),
        "distributed_surrogate": {
            "characteristic_impedance_ohm": float(propagation[0]),
            "phase_constant_rad_per_mm_at_10ghz": float(propagation[1]),
            "attenuation_np_per_mm_at_10ghz": float(propagation[2]),
            "fit_rms_abs_delta_s": fit_rms,
            "fit_max_abs_delta_s": fit_max,
        },
        "physical_component_values": {
            "series_inductor_nh": float(components[0]),
            "ground_capacitor_pf": float(components[1]),
            "bridge_inductor_nh": float(components[2]),
        },
        "physical_fixture_override": {
            "pre_reference_x_mm": pre_x,
            "post_reference_x_mm": post_x,
            "series_gap_center_x_mm": gap_center,
            "ground_cap_x_mm": cap_x,
            "bridge_x_mm": bridge_x,
        },
        "predicted_metrics": nominal,
        "component_corner_active_rl_min_db": corner_active,
        "component_corner_insertion_efficiency_min": corner_insertion,
        "physical_aware_gate_pass": gate_pass,
    }
    write_json(args.out_dir / "physical_aware_selected.json", physical_design)
    write_json(
        args.out_dir / "stage_decision.json",
        {
            "decision": "allow_one_same_topology_physical_confirmation" if gate_pass else "stop_physical_aware_resynthesis_failed",
            "distributed_surrogate_fit_gate_pass": True,
            "physical_aware_circuit_gate_pass": gate_pass,
            "allow_one_physical_confirmation": gate_pass,
            "allow_4x4": False,
            "allow_16x16": False,
            "allow_hfss_training_labels": False,
            "allow_critic_training": False,
        },
    )
    print(json.dumps(physical_design, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
