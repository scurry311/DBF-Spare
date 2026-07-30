#!/usr/bin/env python3
"""Jointly synthesize route, launch, and sparse adjacent POST corrections."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

from design_v115_grounded_modal_network import terminate_network
from design_v119_multiport_post_decoupler import active_metrics, deembed_load, network_from_nodal_y, reordered_network
from run_v114_small_cell_broadband_feed import load_stimuli, parse_touchstone


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v120_joint_feed_fanout_sparse_graph.json"
EPS = 1.0e-15
ALL_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def branch_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.diag(values[:4].copy())
    for value, (first, second) in zip(values[4:], ALL_PAIRS):
        matrix[first, first] += value
        matrix[second, second] += value
        matrix[first, second] -= value
        matrix[second, first] -= value
    return matrix


def sparse_values_from_matrix(matrix: np.ndarray, graph: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    pair_values = np.asarray([-matrix[first, second] for first, second in graph], dtype=float)
    ground = np.diag(matrix).astype(float).copy()
    for value, (first, second) in zip(pair_values, graph):
        ground[first] -= value
        ground[second] -= value
    return ground, pair_values


def graph_matrix(ground: np.ndarray, pair_values: np.ndarray, graph: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.diag(ground.copy())
    for value, (first, second) in zip(pair_values, graph):
        matrix[first, first] += value
        matrix[second, second] += value
        matrix[first, second] -= value
        matrix[second, first] -= value
    return matrix


def finite_q_series(value: float, ratio: float, ql: float, qc: float) -> complex:
    if abs(value) <= 1.0e-12:
        return complex(1.0e-6, 0.0)
    reactance = value * ratio if value > 0 else value / ratio
    quality = ql if value > 0 else qc
    return abs(reactance) / quality + 1j * reactance


def finite_q_shunt(value: float, ratio: float, ql: float, qc: float) -> complex:
    if abs(value) <= 1.0e-12:
        return 0.0j
    if value > 0:
        susceptance = value * ratio
        return susceptance / qc + 1j * susceptance
    susceptance = value / ratio
    inverse_q = 1.0 / ql
    return (-susceptance) * inverse_q / (1.0 + inverse_q**2) + 1j * susceptance / (1.0 + inverse_q**2)


def unpack(parameters: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        parameters[0:4],
        parameters[4:8],
        parameters[8:11],
        parameters[11:15],
        parameters[15:18],
        parameters[18:22],
        parameters[22:25],
    )


def physicalize_parameters(parameters: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Project the continuous optimum onto a sparse, layout-ready value grid."""
    physical = np.asarray(parameters, dtype=float).copy()
    network = config["network"]
    route_step = float(network["route_quantization_mm"])
    series_step = float(network["series_reactance_quantization_ohm"])
    shunt_step = float(network["shunt_susceptance_quantization_s"])
    physical[:4] = np.round(physical[:4] / route_step) * route_step
    physical[4:11][np.abs(physical[4:11]) < float(network["series_prune_threshold_ohm"])] = 0.0
    physical[4:11] = np.round(physical[4:11] / series_step) * series_step
    physical[11:][np.abs(physical[11:]) < float(network["shunt_prune_threshold_s"])] = 0.0
    physical[11:] = np.round(physical[11:] / shunt_step) * shunt_step
    return physical


def sparse_pi_s8(
    frequency_ghz: float,
    series_ground: np.ndarray,
    series_pair: np.ndarray,
    input_ground: np.ndarray,
    input_pair: np.ndarray,
    output_ground: np.ndarray,
    output_pair: np.ndarray,
    graph: list[tuple[int, int]],
    config: dict[str, Any],
) -> np.ndarray:
    ratio = frequency_ghz / 10.0
    network = config["network"]
    series_ground_z = np.asarray(
        [finite_q_series(value, ratio, float(network["series_inductor_q"]), float(network["series_capacitor_q"])) for value in series_ground]
    )
    series_pair_z = np.asarray(
        [finite_q_series(value, ratio, float(network["series_inductor_q"]), float(network["series_capacitor_q"])) for value in series_pair]
    )
    input_ground_y = np.asarray(
        [finite_q_shunt(value, ratio, float(network["shunt_inductor_q"]), float(network["shunt_capacitor_q"])) for value in input_ground]
    )
    input_pair_y = np.asarray(
        [finite_q_shunt(value, ratio, float(network["shunt_inductor_q"]), float(network["shunt_capacitor_q"])) for value in input_pair]
    )
    output_ground_y = np.asarray(
        [finite_q_shunt(value, ratio, float(network["shunt_inductor_q"]), float(network["shunt_capacitor_q"])) for value in output_ground]
    )
    output_pair_y = np.asarray(
        [finite_q_shunt(value, ratio, float(network["shunt_inductor_q"]), float(network["shunt_capacitor_q"])) for value in output_pair]
    )
    series_z = graph_matrix(series_ground_z, series_pair_z, graph)
    series_y = np.linalg.inv(series_z)
    input_y = graph_matrix(input_ground_y, input_pair_y, graph)
    output_y = graph_matrix(output_ground_y, output_pair_y, graph)
    nodal = np.block([[input_y + series_y, -series_y], [-series_y, output_y + series_y]])
    return network_from_nodal_y(nodal, float(config["reference_impedance_ohm"]))


def projection_audit(name: str, matrix: np.ndarray, graph: list[tuple[int, int]]) -> dict[str, Any]:
    ground, pair = sparse_values_from_matrix(matrix, graph)
    projected = graph_matrix(ground, pair, graph)
    residual = matrix - projected
    unsupported = [
        {"pair": [first, second], "value": float(matrix[first, second])}
        for first, second in ALL_PAIRS
        if tuple(sorted((first, second))) not in {tuple(sorted(item)) for item in graph}
    ]
    return {
        "name": name,
        "full_matrix": matrix.tolist(),
        "projected_matrix": projected.tolist(),
        "retained_frobenius_fraction": float(np.linalg.norm(projected) / np.linalg.norm(matrix)),
        "residual_frobenius_fraction": float(np.linalg.norm(residual) / np.linalg.norm(matrix)),
        "maximum_abs_residual": float(np.max(np.abs(residual))),
        "unsupported_pairs": unsupported,
        "projected_ground_values": ground.tolist(),
        "projected_graph_pair_values": pair.tolist(),
    }


def physical_translation(parameters: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    route_delta, series_ground, series_pair, input_ground, input_pair, output_ground, output_pair = unpack(parameters)
    fanout = config["fanout"]
    omega = 2.0 * math.pi * 10.0e9
    beta = 2.0 * math.pi * 10.0 * math.sqrt(float(fanout["effective_relative_permittivity"])) / 299.792458
    open_zc = float(fanout["open_stub_impedance_ohm"])
    short_zc = float(fanout["short_stub_impedance_ohm"])
    minimum_stub = float(fanout["minimum_distributed_stub_length_mm"])

    def series_part(value: float) -> dict[str, Any]:
        if abs(value) <= EPS:
            return {"type": "omitted", "reactance_ohm": 0.0}
        if value >= 0:
            return {"type": "series_inductive_or_high_Z_line", "value_nh": value / omega * 1.0e9, "reactance_ohm": value}
        return {"type": "interdigital_series_capacitor", "value_pf": -1.0 / (omega * value) * 1.0e12, "reactance_ohm": value}

    absorbed_limit = float(config["network"]["maximum_absorbed_shunt_susceptance_s"])

    def shunt_part(value: float, role: str) -> dict[str, Any]:
        if abs(value) <= EPS:
            return {"type": "omitted", "susceptance_s": 0.0}
        if abs(value) <= absorbed_limit:
            absorbed_type = "absorbed_by_coupled_line_even_odd_offset" if role == "graph" else "absorbed_by_launch_transition"
            return {"type": absorbed_type, "susceptance_s": value}
        if value >= 0:
            electrical = math.atan(value * open_zc)
            length = electrical / beta
            if length < minimum_stub:
                return {"type": "absorbed_by_launch_pad", "value_pf": value / omega * 1.0e12, "susceptance_s": value}
            return {"type": "low_Z_open_stub", "characteristic_impedance_ohm": open_zc, "stub_length_mm": length, "susceptance_s": value}
        electrical = math.atan(1.0 / max(short_zc * abs(value), EPS))
        return {"type": "high_Z_short_stub", "characteristic_impedance_ohm": short_zc, "stub_length_mm": electrical / beta, "susceptance_s": value}

    return {
        "route_length_mm_by_port": (float(fanout["baseline_common_length_mm"]) + route_delta).tolist(),
        "route_delta_mm_by_port": route_delta.tolist(),
        "series_ground_realization": [series_part(float(value)) for value in series_ground],
        "series_graph_realization": [series_part(float(value)) for value in series_pair],
        "input_ground_realization": [shunt_part(float(value), "ground") for value in input_ground],
        "input_graph_realization": [shunt_part(float(value), "graph") for value in input_pair],
        "output_ground_realization": [shunt_part(float(value), "ground") for value in output_ground],
        "output_graph_realization": [shunt_part(float(value), "graph") for value in output_pair],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_json(args.config)
    out_dir = resolve(config["output_directory"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.20 output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = [tuple(int(value) for value in pair) for pair in config["manufacturable_graph_pairs"]]

    frequencies, integrated = reordered_network(resolve(config["integrated_v118_s4"]), [f"PRE_{index}" for index in range(4)], 4)
    antenna_f, antenna = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    feed_f, feed = reordered_network(resolve(config["validated_feed_s8"]), [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)], 8)
    if not np.allclose(frequencies, antenna_f) or not np.allclose(frequencies, feed_f):
        raise RuntimeError("Frequency grids differ")
    effective_load = np.stack([deembed_load(integrated[index], feed[index]) for index in range(3)])
    target = np.stack([terminate_network(feed[index], antenna[index])[0] for index in range(3)])
    rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    stimulus_rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    groups = [np.asarray([index for index, row in enumerate(stimulus_rows) if abs(float(row["frequency_ghz"]) - float(frequency)) <= 1.0e-9]) for frequency in frequencies]

    full = read_json(resolve(config["v119_full_matrix_summary"]))
    transform = np.asarray(full["modal_transform"], dtype=float)
    series_full = transform @ np.diag(np.asarray(full["optimized_series_modal_reactance_ohm"], dtype=float)) @ transform.T
    input_full = branch_matrix(np.asarray(full["optimized_input_shunt_susceptance_s"], dtype=float))
    output_full = branch_matrix(np.asarray(full["optimized_output_shunt_susceptance_s"], dtype=float))
    audits = [projection_audit("series_reactance_ohm", series_full, graph), projection_audit("input_susceptance_s", input_full, graph), projection_audit("output_susceptance_s", output_full, graph)]
    write_json(out_dir / "full_matrix_projection_audit.json", audits)
    initial_sections = [sparse_values_from_matrix(matrix, graph) for matrix in (series_full, input_full, output_full)]
    initial = np.r_[np.zeros(4), initial_sections[0][0], initial_sections[0][1], initial_sections[1][0], initial_sections[1][1], initial_sections[2][0], initial_sections[2][1]]

    def evaluate(parameters: np.ndarray, q_scale: float = 1.0) -> dict[str, float]:
        route_delta, series_ground, series_pair, input_ground, input_pair, output_ground, output_pair = unpack(parameters)
        local = json.loads(json.dumps(config))
        for key in ("series_inductor_q", "series_capacitor_q", "shunt_inductor_q", "shunt_capacitor_q"):
            local["network"][key] = float(local["network"][key]) * q_scale
        active_min = math.inf
        total_min = math.inf
        passive_min = math.inf
        delta_max = 0.0
        try:
            for index, frequency in enumerate(frequencies):
                beta = 2.0 * math.pi * float(frequency) * math.sqrt(float(config["fanout"]["effective_relative_permittivity"])) / 299.792458
                delay = np.diag(np.exp(-1j * beta * route_delta))
                routed_load = delay @ effective_load[index] @ delay
                network_s8 = sparse_pi_s8(float(frequency), series_ground, series_pair, input_ground, input_pair, output_ground, output_pair, graph, local)
                post = terminate_network(network_s8, routed_load)[0]
                external = terminate_network(feed[index], post)[0]
                selected = groups[index]
                active_rl, total_rl = active_metrics(external, vectors[selected].T, considered[selected].T)
                active_min = min(active_min, active_rl)
                total_min = min(total_min, total_rl)
                passive_min = min(passive_min, float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(external)), EPS)))))
                delta_max = max(delta_max, float(np.max(np.abs(external - target[index]))))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return {"active_rl_min_db": -100.0, "total_rl_min_db": -100.0, "passive_rl_min_db": -100.0, "corrected_vs_target_max_abs_delta_s": 10.0}
        return {"active_rl_min_db": active_min, "total_rl_min_db": total_min, "passive_rl_min_db": passive_min, "corrected_vs_target_max_abs_delta_s": delta_max}

    optimization = config["optimization"]
    fanout = config["fanout"]

    def objective(parameters: np.ndarray) -> float:
        metrics = evaluate(parameters)
        active_gap = max(float(optimization["design_active_rl_db"]) - metrics["active_rl_min_db"], 0.0)
        total_gap = max(float(optimization["design_total_rl_db"]) - metrics["total_rl_min_db"], 0.0)
        delta_gap = max(metrics["corrected_vs_target_max_abs_delta_s"] - float(optimization["design_max_abs_delta_s"]), 0.0)
        passive_gap = max(float(config["gates"]["minimum_passive_rl_db"]) - metrics["passive_rl_min_db"], 0.0)
        route_delta = parameters[:4]
        route_lengths = float(fanout["baseline_common_length_mm"]) + route_delta
        spread_gap = max(float(np.ptp(route_lengths)) - float(fanout["maximum_route_spread_mm"]), 0.0)
        series_values = parameters[4:11]
        minimum_capacitive_reactance = 1.0 / (
            2.0 * math.pi * 10.0e9 * float(config["network"]["maximum_series_capacitance_pf"]) * 1.0e-12
        )
        capacitance_gap = sum(
            max(minimum_capacitive_reactance - abs(float(value)), 0.0) ** 2
            for value in series_values
            if value < 0.0
        )
        delta_weight = float(optimization.get("delta_penalty_weight", 25000.0))
        capacitance_weight = float(optimization.get("series_capacitance_penalty_weight", 0.0))
        return float(130.0 * active_gap**2 + 35.0 * total_gap**2 + delta_weight * delta_gap**2 + 10.0 * passive_gap**2 + 10.0 * spread_gap**2 + capacitance_weight * capacitance_gap + 0.01 * np.sum(route_delta**2) - 0.25 * min(metrics["active_rl_min_db"], 14.0) - 0.06 * min(metrics["total_rl_min_db"], 14.0) + 3.0 * metrics["corrected_vs_target_max_abs_delta_s"])

    route_low = float(fanout["minimum_route_length_mm"]) - float(fanout["baseline_common_length_mm"])
    route_high = float(fanout["maximum_route_length_mm"]) - float(fanout["baseline_common_length_mm"])
    series_bound = float(config["network"]["series_reactance_bound_ohm"])
    graph_series_bound = float(config["network"]["graph_series_reactance_bound_ohm"])
    shunt_bound = float(config["network"]["shunt_susceptance_bound_s"])
    bounds = [(route_low, route_high)] * 4 + [(-series_bound, series_bound)] * 4 + [(-graph_series_bound, graph_series_bound)] * 3 + [(-shunt_bound, shunt_bound)] * 4 + [(-shunt_bound, shunt_bound)] * 3 + [(-shunt_bound, shunt_bound)] * 4 + [(-shunt_bound, shunt_bound)] * 3
    reused_summary = optimization.get("reuse_solution_summary")
    if reused_summary:
        reused = read_json(resolve(str(reused_summary)))
        selected_continuous = np.asarray(reused.get("continuous_optimized_parameters", reused["optimized_parameters"]), dtype=float)
        optimizer_success = bool(reused.get("optimizer_success", False))
        optimizer_message = f"Reused continuous solution from {reused_summary}"
        optimizer_objective = float(reused.get("objective", objective(selected_continuous)))
    else:
        result = differential_evolution(objective, bounds=bounds, x0=initial, seed=int(optimization["seed"]), popsize=int(optimization["population_size"]), maxiter=int(optimization["maximum_iterations"]), tol=1.0e-8, polish=bool(optimization["polish"]), updating="immediate", workers=1)
        selected_continuous = np.asarray(result.x)
        optimizer_success = bool(result.success)
        optimizer_message = str(result.message)
        optimizer_objective = float(result.fun)
    if bool(optimization.get("local_refinement", False)):
        local_result = minimize(
            objective,
            selected_continuous,
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(optimization.get("local_maximum_iterations", 120)), "xtol": 1.0e-5, "ftol": 1.0e-9},
        )
        selected_continuous = np.asarray(local_result.x)
        optimizer_success = bool(local_result.success)
        optimizer_message = f"{optimizer_message}; Powell: {local_result.message}"
        optimizer_objective = float(local_result.fun)
    selected = physicalize_parameters(selected_continuous, config)
    continuous_nominal = evaluate(selected_continuous)
    nominal = evaluate(selected)
    translation = physical_translation(selected, config)
    route_lengths = np.asarray(translation["route_length_mm_by_port"])
    all_stub_lengths = [
        float(item["stub_length_mm"])
        for key, values in translation.items()
        if key.endswith("_realization") and ("ground" in key or "graph" in key)
        for item in values
        if "stub_length_mm" in item
    ]
    series_parts = translation["series_ground_realization"] + translation["series_graph_realization"]
    maximum_capacitance = max((float(item.get("value_pf", 0.0)) for item in series_parts), default=0.0)
    maximum_inductance = max((float(item.get("value_nh", 0.0)) for item in series_parts), default=0.0)
    physical_geometry_gate = bool(
        np.min(route_lengths) >= float(fanout["minimum_route_length_mm"])
        and np.max(route_lengths) <= float(fanout["maximum_route_length_mm"])
        and np.ptp(route_lengths) <= float(fanout["maximum_route_spread_mm"])
        and (not all_stub_lengths or max(all_stub_lengths) <= float(fanout["maximum_stub_length_mm"]))
        and maximum_capacitance <= float(config["network"]["maximum_series_capacitance_pf"])
        and maximum_inductance <= float(config["network"]["maximum_series_inductance_nh"])
    )
    gates = config["gates"]
    nominal_gate = bool(nominal["active_rl_min_db"] >= float(gates["minimum_active_rl_db"]) and nominal["total_rl_min_db"] >= float(gates["minimum_total_rl_db"]) and nominal["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"]) and nominal["corrected_vs_target_max_abs_delta_s"] <= float(gates["maximum_corrected_vs_target_abs_delta_s"]))

    audit = config["tolerance_audit"]
    trials = []
    if nominal_gate and physical_geometry_gate:
        rng = np.random.default_rng(int(optimization["seed"]) + 1)
        for trial in range(int(audit["samples"])):
            varied = selected.copy()
            varied[:4] += rng.normal(0.0, float(audit["route_sigma_mm"]), size=4)
            varied[4:] *= rng.normal(1.0, float(audit["reactance_sigma_fraction"]), size=len(varied) - 4)
            q_scale = max(0.5, float(rng.normal(1.0, float(audit["q_sigma_fraction"]))))
            metrics = evaluate(varied, q_scale)
            passed = bool(metrics["active_rl_min_db"] >= float(gates["minimum_active_rl_db"]) and metrics["total_rl_min_db"] >= float(gates["minimum_total_rl_db"]) and metrics["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"]) and metrics["corrected_vs_target_max_abs_delta_s"] <= float(gates["maximum_corrected_vs_target_abs_delta_s"]))
            trials.append({"trial": trial, **metrics, "joint_gate_pass": int(passed)})
    tolerance_rate = float(np.mean([row["joint_gate_pass"] for row in trials])) if trials else 0.0
    joint_gate = bool(nominal_gate and physical_geometry_gate and tolerance_rate >= float(audit["minimum_joint_pass_rate"]))
    summary = {
        "protocol": config["protocol"],
        "evidence_level": "finite-Q sparse-graph circuit and route surrogate; not physical HFSS",
        "graph_pairs": [list(pair) for pair in graph],
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "objective": optimizer_objective,
        "continuous_nominal_metrics": continuous_nominal,
        "nominal_metrics": nominal,
        "physical_translation": translation,
        "physical_geometry_gate_pass": physical_geometry_gate,
        "nominal_joint_gate_pass": nominal_gate,
        "tolerance_sample_count": len(trials),
        "tolerance_joint_pass_rate": tolerance_rate,
        "joint_sparse_graph_gate_pass": joint_gate,
        "continuous_optimized_parameters": selected_continuous.tolist(),
        "optimized_parameters": selected.tolist(),
    }
    decision = {
        "allow_one_physical_front_gate": joint_gate,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "next_action": "build one sparse-graph feed/fanout physical front gate" if joint_gate else "stop physical HFSS and revise the radiator/feed geometry because the realizable graph cannot remove the correction burden",
    }
    write_json(out_dir / "config_snapshot.json", config)
    write_json(out_dir / "synthesis_summary.json", summary)
    write_json(out_dir / "stage_decision.json", decision)
    if trials:
        with (out_dir / "tolerance_trials.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
    print(json.dumps({"summary": summary, "decision": decision}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
