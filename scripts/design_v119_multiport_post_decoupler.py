#!/usr/bin/env python3
"""Synthesize a finite-Q four-port POST network with full coupling support."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from design_v115_grounded_modal_network import terminate_network
from run_v114_small_cell_broadband_feed import load_stimuli, parse_touchstone
from run_v115_physical_modal_feed_fixture import touchstone_port_names


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v119_multiport_post_decoupler.json"
EPS = 1.0e-15
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
GIVENS_PAIRS = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def reordered_network(path: Path, names: list[str], nports: int) -> tuple[np.ndarray, np.ndarray]:
    frequencies, network = parse_touchstone(path, nports)
    exported = touchstone_port_names(path)
    if not exported:
        return frequencies, network
    order = [exported.index(name) for name in names]
    return frequencies, network[:, order][:, :, order]


def network_from_nodal_y(y_matrix: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(y_matrix.shape[0], dtype=complex)
    return np.linalg.solve((identity + z0 * y_matrix).T, (identity - z0 * y_matrix).T).T


def write_touchstone(path: Path, frequencies: np.ndarray, matrices: np.ndarray, z0: float) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("! v1.19 finite-Q E96 modal POST target; circuit synthesis, not HFSS.\n")
        handle.write(f"# GHZ S RI R {z0:.12g}\n")
        for frequency, matrix in zip(frequencies, matrices):
            values = []
            for column in range(matrix.shape[0]):
                for row in range(matrix.shape[0]):
                    values.extend((matrix[row, column].real, matrix[row, column].imag))
            tokens = [f"{float(frequency):.12g}"] + [f"{value:.15e}" for value in values]
            handle.write(" ".join(tokens) + "\n")


def deembed_load(external_s: np.ndarray, feed_s8: np.ndarray) -> np.ndarray:
    s11 = feed_s8[:4, :4]
    s12 = feed_s8[:4, 4:]
    s21 = feed_s8[4:, :4]
    s22 = feed_s8[4:, 4:]
    y = np.linalg.solve(s12, external_s - s11) @ np.linalg.inv(s21)
    return np.linalg.solve(np.eye(4, dtype=complex) + y @ s22, y)


def branch_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=complex)
    for port in range(4):
        matrix[port, port] += values[port]
    for value, (first, second) in zip(values[4:], PAIRS):
        matrix[first, first] += value
        matrix[second, second] += value
        matrix[first, second] -= value
        matrix[second, first] -= value
    return matrix


def finite_q_series(x0: np.ndarray, ratio: float, ql: float, qc: float) -> np.ndarray:
    impedance = np.empty(4, dtype=complex)
    for index, value in enumerate(x0):
        reactance = value * ratio if value > 0 else value / ratio
        quality = ql if value > 0 else qc
        impedance[index] = abs(reactance) / quality + 1j * reactance
    return impedance


def finite_q_shunt(b0: np.ndarray, ratio: float, ql: float, qc: float) -> np.ndarray:
    admittance = np.empty(10, dtype=complex)
    for index, value in enumerate(b0):
        if value > 0:
            susceptance = value * ratio
            admittance[index] = susceptance / qc + 1j * susceptance
        elif value < 0:
            susceptance = value / ratio
            resistance_ratio = 1.0 / ql
            admittance[index] = (-susceptance) * resistance_ratio / (1.0 + resistance_ratio**2) + 1j * susceptance / (1.0 + resistance_ratio**2)
        else:
            admittance[index] = 0.0
    return admittance


def modal_pi_s8(
    frequency_ghz: float,
    x0: np.ndarray,
    input_b0: np.ndarray,
    output_b0: np.ndarray,
    transform: np.ndarray,
    z0: float,
    ql: float,
    qc: float,
) -> np.ndarray:
    ratio = frequency_ghz / 10.0
    modal_z = finite_q_series(x0, ratio, ql, qc)
    series_z = transform @ np.diag(modal_z) @ transform.T
    series_y = np.linalg.inv(series_z)
    input_y = branch_matrix(finite_q_shunt(input_b0, ratio, ql, qc))
    output_y = branch_matrix(finite_q_shunt(output_b0, ratio, ql, qc))
    nodal = np.block(
        [[input_y + series_y, -series_y], [-series_y, output_y + series_y]]
    )
    return network_from_nodal_y(nodal, z0)


def active_metrics(s4: np.ndarray, sources: np.ndarray, considered: np.ndarray) -> tuple[float, float]:
    reflected = s4 @ sources
    gamma = np.where(
        considered,
        np.abs(reflected) / np.maximum(np.abs(sources), EPS),
        0.0,
    )
    active_rl = -20.0 * np.log10(np.maximum(np.max(gamma, axis=0), EPS))
    incident_power = np.sum(np.abs(sources) ** 2, axis=0)
    reflected_power = np.sum(np.abs(reflected) ** 2, axis=0)
    total_rl = -10.0 * np.log10(np.maximum(reflected_power / incident_power, EPS))
    return float(np.min(active_rl)), float(np.min(total_rl))


def unpack(parameters: np.ndarray, input_template: np.ndarray, output_template: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = parameters[:4]
    input_b = np.zeros(10)
    output_b = np.zeros(10)
    input_indices = np.flatnonzero(input_template)
    output_indices = np.flatnonzero(output_template)
    split = 4 + len(input_indices)
    input_b[input_indices] = parameters[4:split]
    output_b[output_indices] = parameters[split:]
    return x0, input_b, output_b


def component_bounds(value: float) -> tuple[float, float]:
    scaled = sorted((0.35 * value, 1.80 * value))
    return float(scaled[0]), float(scaled[1])


def e96_nearest(value: float) -> float:
    if value <= 0:
        raise ValueError("E96 quantization requires a positive value")
    exponent = math.floor(math.log10(value))
    normalized = value / 10.0**exponent
    candidates = np.asarray(
        [
            100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130,
            133, 137, 140, 143, 147, 150, 154, 158, 162, 165, 169, 174,
            178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232,
            237, 243, 249, 255, 261, 267, 274, 280, 287, 294, 301, 309,
            316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412,
            422, 432, 442, 453, 464, 475, 487, 499, 511, 523, 536, 549,
            562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732,
            750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976,
            1000,
        ],
        dtype=float,
    ) / 100.0
    selected = float(candidates[np.argmin(np.abs(np.log(candidates / normalized)))])
    return selected * 10.0**exponent


def parameter_components(x0: np.ndarray, input_b: np.ndarray, output_b: np.ndarray) -> list[dict[str, Any]]:
    omega = 2.0 * math.pi * 10.0e9
    rows: list[dict[str, Any]] = []
    for index, reactance in enumerate(x0):
        if reactance > 0:
            rows.append({"section": "series_mode", "branch": f"m{index}", "type": "L", "value": reactance / omega, "unit": "H"})
        else:
            rows.append({"section": "series_mode", "branch": f"m{index}", "type": "C", "value": -1.0 / (omega * reactance), "unit": "F"})
    for section, values in (("input_shunt", input_b), ("output_shunt", output_b)):
        for index, susceptance in enumerate(values):
            if susceptance > 0:
                rows.append({"section": section, "branch": index, "type": "C", "value": susceptance / omega, "unit": "F"})
            elif susceptance < 0:
                rows.append({"section": section, "branch": index, "type": "L", "value": -1.0 / (omega * susceptance), "unit": "H"})
    return rows


def components_to_parameters(rows: list[dict[str, Any]], input_template: np.ndarray, output_template: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = 2.0 * math.pi * 10.0e9
    x0 = np.zeros(4)
    input_b = np.zeros(10)
    output_b = np.zeros(10)
    for row in rows:
        value = float(row["value"])
        if row["section"] == "series_mode":
            index = int(str(row["branch"])[1:])
            x0[index] = omega * value if row["type"] == "L" else -1.0 / (omega * value)
        else:
            index = int(row["branch"])
            susceptance = omega * value if row["type"] == "C" else -1.0 / (omega * value)
            (input_b if row["section"] == "input_shunt" else output_b)[index] = susceptance
    input_b[~input_template.astype(bool)] = 0.0
    output_b[~output_template.astype(bool)] = 0.0
    return x0, input_b, output_b


def givens_matrix(pair: tuple[int, int], angle: float) -> np.ndarray:
    matrix = np.eye(4)
    first, second = pair
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix[first, first] = cosine
    matrix[second, second] = cosine
    matrix[first, second] = -sine
    matrix[second, first] = sine
    return matrix


def decompose_transform(transform: np.ndarray) -> dict[str, Any]:
    best = None
    for signs_bits in range(16):
        signs = np.asarray([1.0 if signs_bits & (1 << index) else -1.0 for index in range(4)])
        target = transform @ np.diag(signs)
        if np.linalg.det(target) < 0:
            continue

        def residual(angles: np.ndarray) -> np.ndarray:
            product = np.eye(4)
            for pair, angle in zip(GIVENS_PAIRS, angles):
                product = product @ givens_matrix(pair, float(angle))
            return (product - target).ravel()

        fit = least_squares(residual, np.zeros(6), max_nfev=10000, ftol=1.0e-14, xtol=1.0e-14, gtol=1.0e-14)
        error = float(np.max(np.abs(residual(fit.x))))
        if best is None or error < best["max_abs_error"]:
            best = {
                "pairs": [f"{first}-{second}" for first, second in GIVENS_PAIRS],
                "angles_deg": np.degrees(fit.x).tolist(),
                "column_signs": signs.astype(int).tolist(),
                "max_abs_error": error,
            }
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_json(args.config)
    out_dir = resolve(config["output_directory"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.19 result: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    frequencies, integrated = reordered_network(
        resolve(config["integrated_v118_s4"]), [f"PRE_{index}" for index in range(4)], 4
    )
    antenna_f, antenna = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    feed_f, feed = reordered_network(
        resolve(config["validated_feed_s8"]),
        [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)],
        8,
    )
    if not np.allclose(frequencies, antenna_f) or not np.allclose(frequencies, feed_f):
        raise RuntimeError("S4/S8 frequency grids differ")
    effective_load = np.stack([deembed_load(integrated[index], feed[index]) for index in range(3)])
    target = np.stack([terminate_network(feed[index], antenna[index])[0] for index in range(3)])

    rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    stimulus_rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    groups = [
        np.asarray([index for index, row in enumerate(stimulus_rows) if abs(float(row["frequency_ghz"]) - float(frequency)) <= 1.0e-9])
        for frequency in frequencies
    ]

    network = config["network"]
    transform = np.asarray(network["modal_transform"], dtype=float)
    initial_x = np.asarray(network["initial_series_modal_reactance_ohm"], dtype=float)
    threshold = float(network["shunt_pruning_threshold_s"])
    input_template = np.asarray(network["initial_input_shunt_susceptance_s"], dtype=float)
    output_template = np.asarray(network["initial_output_shunt_susceptance_s"], dtype=float)
    input_template[np.abs(input_template) < threshold] = 0.0
    output_template[np.abs(output_template) < threshold] = 0.0
    initial = np.r_[initial_x, input_template[input_template != 0.0], output_template[output_template != 0.0]]
    ql = float(network["inductor_q"])
    qc = float(network["capacitor_q"])

    def evaluate(parameters: np.ndarray, q_inductor: float = ql, q_capacitor: float = qc) -> dict[str, float]:
        x0, input_b, output_b = unpack(parameters, input_template, output_template)
        active_min = math.inf
        total_min = math.inf
        passive_min = math.inf
        delta_max = 0.0
        for frequency_index, frequency in enumerate(frequencies):
            pi_s8 = modal_pi_s8(float(frequency), x0, input_b, output_b, transform, float(config["reference_impedance_ohm"]), q_inductor, q_capacitor)
            post_network = terminate_network(pi_s8, effective_load[frequency_index])[0]
            external = terminate_network(feed[frequency_index], post_network)[0]
            indices = groups[frequency_index]
            active_rl, total_rl = active_metrics(external, vectors[indices].T, considered[indices].T)
            active_min = min(active_min, active_rl)
            total_min = min(total_min, total_rl)
            passive_min = min(passive_min, float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(external)), EPS)))))
            delta_max = max(delta_max, float(np.max(np.abs(external - target[frequency_index]))))
        return {
            "active_rl_min_db": active_min,
            "total_rl_min_db": total_min,
            "passive_rl_min_db": passive_min,
            "integrated_vs_target_max_abs_delta_s": delta_max,
        }

    optimization = config["optimization"]
    gates = config["gates"]

    def objective(parameters: np.ndarray) -> float:
        metrics = evaluate(parameters)
        active_gap = max(float(optimization["design_active_rl_db"]) - metrics["active_rl_min_db"], 0.0)
        total_gap = max(float(optimization["design_total_rl_db"]) - metrics["total_rl_min_db"], 0.0)
        delta_gap = max(metrics["integrated_vs_target_max_abs_delta_s"] - float(optimization["design_max_abs_delta_s"]), 0.0)
        passive_gap = max(float(gates["minimum_passive_rl_db"]) - metrics["passive_rl_min_db"], 0.0)
        return float(120.0 * active_gap**2 + 30.0 * total_gap**2 + 30000.0 * delta_gap**2 + 10.0 * passive_gap**2 - 0.30 * min(metrics["active_rl_min_db"], 14.0) - 0.08 * min(metrics["total_rl_min_db"], 14.0) + 4.0 * metrics["integrated_vs_target_max_abs_delta_s"])

    bounds = [(-35.0, -4.0), (0.5, 9.0), (5.0, 50.0), (15.0, 90.0)]
    bounds.extend(component_bounds(float(value)) for value in input_template[input_template != 0.0])
    bounds.extend(component_bounds(float(value)) for value in output_template[output_template != 0.0])
    result = differential_evolution(
        objective,
        bounds=bounds,
        x0=initial,
        seed=int(optimization["seed"]),
        popsize=int(optimization["population_size"]),
        maxiter=int(optimization["maximum_iterations"]),
        tol=1.0e-8,
        polish=bool(optimization["polish"]),
        updating="immediate",
        workers=1,
    )
    optimized_x, optimized_input, optimized_output = unpack(np.asarray(result.x), input_template, output_template)
    nominal = evaluate(np.asarray(result.x))

    component_rows = parameter_components(optimized_x, optimized_input, optimized_output)
    quantized_rows = [{**row, "unquantized_value": row["value"], "value": e96_nearest(float(row["value"]))} for row in component_rows]
    quantized_x, quantized_input, quantized_output = components_to_parameters(quantized_rows, input_template, output_template)
    quantized_parameters = np.r_[quantized_x, quantized_input[input_template != 0.0], quantized_output[output_template != 0.0]]
    quantized_metrics = evaluate(quantized_parameters)
    quantized_s8 = np.stack(
        [
            modal_pi_s8(
                float(frequency),
                quantized_x,
                quantized_input,
                quantized_output,
                transform,
                float(config["reference_impedance_ohm"]),
                ql,
                qc,
            )
            for frequency in frequencies
        ]
    )
    matched_efficiency_min = math.inf
    for matrix in quantized_s8:
        matched_s, load_incident, load_reflected = terminate_network(
            matrix, np.zeros((4, 4), dtype=complex)
        )
        accepted = 1.0 - np.sum(np.abs(matched_s) ** 2, axis=0)
        delivered = np.sum(np.abs(load_incident) ** 2, axis=0) - np.sum(
            np.abs(load_reflected) ** 2, axis=0
        )
        matched_efficiency_min = min(
            matched_efficiency_min,
            float(np.min(delivered / np.maximum(accepted, EPS))),
        )

    audit = config["tolerance_audit"]
    rng = np.random.default_rng(int(optimization["seed"]) + 1)
    trials = []
    pass_count = 0
    base_components = quantized_rows
    for trial in range(int(audit["samples"])):
        varied_rows = []
        for row in base_components:
            scale = max(0.80, float(rng.normal(1.0, float(audit["component_sigma_fraction"]))))
            varied_rows.append({**row, "value": float(row["value"]) * scale})
        trial_x, trial_input, trial_output = components_to_parameters(varied_rows, input_template, output_template)
        parameters = np.r_[trial_x, trial_input[input_template != 0.0], trial_output[output_template != 0.0]]
        trial_ql = max(10.0, ql * float(rng.normal(1.0, float(audit["q_sigma_fraction"]))))
        trial_qc = max(20.0, qc * float(rng.normal(1.0, float(audit["q_sigma_fraction"]))))
        metrics = evaluate(parameters, trial_ql, trial_qc)
        passed = bool(
            metrics["active_rl_min_db"] >= float(gates["minimum_active_rl_db"])
            and metrics["total_rl_min_db"] >= float(gates["minimum_total_rl_db"])
            and metrics["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
            and metrics["integrated_vs_target_max_abs_delta_s"] <= float(gates["maximum_integrated_vs_target_abs_delta_s"])
        )
        pass_count += int(passed)
        trials.append({"trial": trial, **metrics, "joint_gate_pass": int(passed)})

    tolerance_pass_rate = pass_count / len(trials)
    nominal_gate = bool(
        nominal["active_rl_min_db"] >= float(gates["minimum_active_rl_db"])
        and nominal["total_rl_min_db"] >= float(gates["minimum_total_rl_db"])
        and nominal["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
        and nominal["integrated_vs_target_max_abs_delta_s"] <= float(gates["maximum_integrated_vs_target_abs_delta_s"])
    )
    tolerance_gate = bool(
        nominal_gate
        and quantized_metrics["active_rl_min_db"] >= float(gates["minimum_active_rl_db"])
        and quantized_metrics["integrated_vs_target_max_abs_delta_s"] <= float(gates["maximum_integrated_vs_target_abs_delta_s"])
        and tolerance_pass_rate >= float(audit["minimum_joint_pass_rate"])
    )
    decomposition = decompose_transform(transform)
    realization = config["compact_realization"]
    impractical_components = []
    for row in quantized_rows:
        scaled_value = float(row["value"]) * (1.0e9 if row["type"] == "L" else 1.0e12)
        lower = float(
            realization[
                "minimum_discrete_inductance_nh"
                if row["type"] == "L"
                else "minimum_discrete_capacitance_pf"
            ]
        )
        upper = float(
            realization[
                "maximum_discrete_inductance_nh"
                if row["type"] == "L"
                else "maximum_discrete_capacitance_pf"
            ]
        )
        if not lower <= scaled_value <= upper:
            impractical_components.append(
                {
                    "section": row["section"],
                    "branch": row["branch"],
                    "type": row["type"],
                    "value_nh_or_pf": scaled_value,
                    "allowed_min": lower,
                    "allowed_max": upper,
                }
            )
    modal_stage_count = 2 * len(decomposition["angles_deg"])
    estimated_modal_efficiency = float(
        realization["estimated_efficiency_per_givens_stage"]
    ) ** modal_stage_count
    compact_realization_gate = bool(
        not impractical_components
        and max(abs(float(value)) for value in decomposition["angles_deg"])
        <= float(realization["maximum_single_givens_angle_deg"])
        and estimated_modal_efficiency
        >= float(realization["minimum_estimated_modal_transform_efficiency"])
        and matched_efficiency_min
        >= float(realization["minimum_matched_load_network_efficiency"])
    )
    summary = {
        "protocol": config["protocol"],
        "evidence_level": "finite-Q circuit synthesis using deembedded integrated HFSS S4; not a physical HFSS network",
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "objective": float(result.fun),
        "stimulus_count": len(stimulus_rows),
        "reactive_component_count": len(component_rows),
        "input_shunt_branch_count": int(np.count_nonzero(optimized_input)),
        "output_shunt_branch_count": int(np.count_nonzero(optimized_output)),
        "nominal_metrics": nominal,
        "e96_quantized_metrics": quantized_metrics,
        "e96_matched_load_network_efficiency_min": matched_efficiency_min,
        "tolerance_joint_pass_rate": tolerance_pass_rate,
        "tolerance_sample_count": len(trials),
        "nominal_gate_pass": nominal_gate,
        "tolerance_gate_pass": tolerance_gate,
        "compact_realization_gate_pass": compact_realization_gate,
        "impractical_discrete_component_count": len(impractical_components),
        "impractical_discrete_components": impractical_components,
        "modal_transform_physical_stage_count": modal_stage_count,
        "estimated_modal_transform_efficiency": estimated_modal_efficiency,
        "optimized_series_modal_reactance_ohm": optimized_x.tolist(),
        "optimized_input_shunt_susceptance_s": optimized_input.tolist(),
        "optimized_output_shunt_susceptance_s": optimized_output.tolist(),
        "modal_transform": transform.tolist(),
        "givens_realization": decomposition,
    }
    decision = {
        "circuit_tolerance_gate_pass": tolerance_gate,
        "compact_realization_gate_pass": compact_realization_gate,
        "allow_one_physical_s8": bool(tolerance_gate and compact_realization_gate),
        "allow_integrated_2x2_repeat": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "next_action": (
            "build one physical four-channel modal POST S8"
            if tolerance_gate and compact_realization_gate
            else "replace the cascaded Givens realization with one directly optimized multiconductor distributed POST block"
            if tolerance_gate
            else "increase circuit manufacturing reserve before physical HFSS"
        ),
    }
    with (out_dir / "components.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quantized_rows[0]))
        writer.writeheader()
        writer.writerows(quantized_rows)
    with (out_dir / "tolerance_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    write_json(out_dir / "synthesis_summary.json", summary)
    write_json(out_dir / "stage_decision.json", decision)
    write_json(out_dir / "config_snapshot.json", config)
    write_touchstone(
        out_dir / "e96_modal_pi_target.s8p",
        frequencies,
        quantized_s8,
        float(config["reference_impedance_ohm"]),
    )
    print(json.dumps({"summary": summary, "decision": decision}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
