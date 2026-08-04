#!/usr/bin/env python3
"""Audit and synthesize a load-aware single-block local modal transformer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from design_v115_grounded_modal_network import s_to_z, terminate_network
from design_v119_multiport_post_decoupler import active_metrics, reordered_network
from run_v114_small_cell_broadband_feed import load_stimuli, parse_touchstone


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v123_load_aware_modal_transformer_preregistered.json"
EPS = 1.0e-15


def resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complex_matrix_payload(matrix: np.ndarray) -> dict[str, Any]:
    return {"real": np.real(matrix).tolist(), "imag": np.imag(matrix).tolist()}


def modal_transform(pairs: list[tuple[int, int]]) -> np.ndarray:
    transform = np.zeros((4, 4), dtype=float)
    for pair_index, (first, second) in enumerate(pairs):
        transform[first, 2 * pair_index] = 1.0 / math.sqrt(2.0)
        transform[second, 2 * pair_index] = 1.0 / math.sqrt(2.0)
        transform[first, 2 * pair_index + 1] = 1.0 / math.sqrt(2.0)
        transform[second, 2 * pair_index + 1] = -1.0 / math.sqrt(2.0)
    return transform


def repeated_launch_s8(s2: np.ndarray) -> np.ndarray:
    identity = np.eye(4, dtype=complex)
    return np.block(
        [
            [s2[0, 0] * identity, s2[0, 1] * identity],
            [s2[1, 0] * identity, s2[1, 1] * identity],
        ]
    )


def ideal_through_s8() -> np.ndarray:
    identity = np.eye(4, dtype=complex)
    zero = np.zeros((4, 4), dtype=complex)
    return np.block([[zero, identity], [identity, zero]])


def abcd_to_s(abcd: np.ndarray, z0: float) -> np.ndarray:
    a, b, c, d = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
    denominator = a + b / z0 + c * z0 + d
    return np.asarray(
        [
            [(a + b / z0 - c * z0 - d) / denominator, 2.0 * (a * d - b * c) / denominator],
            [2.0 / denominator, (-a + b / z0 - c * z0 + d) / denominator],
        ],
        dtype=complex,
    )


def finite_q_shunt(value: float, frequency_ratio: float, q_cap: float, q_ind: float) -> complex:
    if abs(value) <= 1.0e-14:
        return 0.0j
    if value > 0.0:
        susceptance = value * frequency_ratio
        return susceptance / q_cap + 1j * susceptance
    susceptance = value / frequency_ratio
    inverse_q = 1.0 / q_ind
    return (-susceptance) * inverse_q / (1.0 + inverse_q**2) + 1j * susceptance / (1.0 + inverse_q**2)


def modal_two_port(
    impedance_ohm: float,
    theta_rad: float,
    shunt_admittance: complex,
    reference_ohm: float,
    line_q: float | None,
    input_loading_fraction: float,
) -> np.ndarray:
    attenuation = 0.0 if line_q is None else theta_rad / (2.0 * line_q)
    propagation = attenuation + 1j * theta_rad
    line = np.asarray(
        [
            [np.cosh(propagation), impedance_ohm * np.sinh(propagation)],
            [np.sinh(propagation) / impedance_ohm, np.cosh(propagation)],
        ],
        dtype=complex,
    )
    input_shunt = np.asarray([[1.0, 0.0], [input_loading_fraction * shunt_admittance, 1.0]], dtype=complex)
    output_shunt = np.asarray([[1.0, 0.0], [(1.0 - input_loading_fraction) * shunt_admittance, 1.0]], dtype=complex)
    return abcd_to_s(input_shunt @ line @ output_shunt, reference_ohm)


def correction_s8_from_modes(mode_networks: list[np.ndarray], transform: np.ndarray) -> np.ndarray:
    modal_blocks = [np.diag([network[row, column] for network in mode_networks]) for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))]
    physical = [transform @ block @ transform.T for block in modal_blocks]
    return np.block([[physical[0], physical[1]], [physical[2], physical[3]]])


def ideal_correction_s8(parameters: np.ndarray, frequency_ghz: float, z0: float, transform: np.ndarray) -> np.ndarray:
    ratio = frequency_ghz / 10.0
    impedances = parameters[:4]
    theta = np.deg2rad(parameters[4:8]) * ratio
    loading = parameters[8:12]
    loading_fraction = parameters[12:16]
    modes = [
        modal_two_port(
            float(impedances[index]),
            float(theta[index]),
            1j * (float(loading[index]) * ratio if loading[index] >= 0.0 else float(loading[index]) / ratio),
            z0,
            None,
            float(loading_fraction[index]),
        )
        for index in range(4)
    ]
    return correction_s8_from_modes(modes, transform)


def physical_correction_s8(
    parameters: np.ndarray,
    frequency_ghz: float,
    z0: float,
    transform: np.ndarray,
    finite_q: dict[str, float],
    q_scale: float = 1.0,
) -> np.ndarray:
    even_z, odd_z, even_theta_deg, odd_ratio, ground_b, bridge_b, loading_fraction = (float(value) for value in parameters)
    frequency_ratio = frequency_ghz / 10.0
    theta_even = math.radians(even_theta_deg) * frequency_ratio
    theta_odd = theta_even * odd_ratio
    ground = finite_q_shunt(
        ground_b,
        frequency_ratio,
        float(finite_q["ground_capacitor_q"]) * q_scale,
        float(finite_q["ground_inductor_q"]) * q_scale,
    )
    bridge = finite_q_shunt(
        bridge_b,
        frequency_ratio,
        float(finite_q["bridge_capacitor_q"]) * q_scale,
        float(finite_q["bridge_inductor_q"]) * q_scale,
    )
    line_q = float(finite_q["coupled_line_unloaded_q"]) * q_scale
    pair_modes = [
        modal_two_port(even_z, theta_even, ground, z0, line_q, loading_fraction),
        modal_two_port(odd_z, theta_odd, ground + 2.0 * bridge, z0, line_q, loading_fraction),
    ]
    return correction_s8_from_modes(pair_modes + pair_modes, transform)


def matched_efficiency(network: np.ndarray) -> float:
    external, incident_map, reflected_map = terminate_network(network, np.zeros((4, 4), dtype=complex))
    accepted = 1.0 - np.sum(np.abs(external) ** 2, axis=0)
    delivered = np.sum(np.abs(incident_map) ** 2, axis=0) - np.sum(np.abs(reflected_map) ** 2, axis=0)
    return float(np.min(delivered / np.maximum(accepted, EPS)))


def cascade_matched_efficiency(launch: np.ndarray, correction: np.ndarray) -> float:
    correction_external, correction_incident, _ = terminate_network(correction, np.zeros((4, 4), dtype=complex))
    total_external, launch_incident, _ = terminate_network(launch, correction_external)
    load_incident = correction_incident @ launch_incident
    accepted = 1.0 - np.sum(np.abs(total_external) ** 2, axis=0)
    delivered = np.sum(np.abs(load_incident) ** 2, axis=0)
    return float(np.min(delivered / np.maximum(accepted, EPS)))


def evaluate_networks(
    correction_matrices: np.ndarray,
    launch_matrices: np.ndarray,
    antenna_matrices: np.ndarray,
    frequencies: np.ndarray,
    stimulus_rows: list[dict[str, str]],
    vectors: np.ndarray,
    considered: np.ndarray,
) -> dict[str, Any]:
    frequency_rows = []
    source_rows = []
    for frequency_index, frequency in enumerate(frequencies):
        correction = correction_matrices[frequency_index]
        launch = launch_matrices[frequency_index]
        antenna = antenna_matrices[frequency_index]
        correction_external, correction_incident, _ = terminate_network(correction, antenna)
        total_external, launch_incident, _ = terminate_network(launch, correction_external)
        selected = np.asarray([abs(float(row["frequency_ghz"]) - frequency) <= 1.0e-9 for row in stimulus_rows])
        sources = vectors[selected].T
        active = considered[selected].T
        active_rl, total_rl = active_metrics(total_external, sources, active)
        reflected = total_external @ sources
        gamma = np.where(active, np.abs(reflected) / np.maximum(np.abs(sources), EPS), -1.0)
        active_by_source = -20.0 * np.log10(np.maximum(np.max(gamma, axis=0), EPS))
        worst_port = np.argmax(gamma, axis=0)
        incident_power = np.sum(np.abs(sources) ** 2, axis=0)
        reflected_power = np.sum(np.abs(reflected) ** 2, axis=0)
        source_accepted = incident_power - reflected_power
        antenna_incident = correction_incident @ launch_incident @ sources
        antenna_reflected = antenna @ antenna_incident
        antenna_accepted = np.sum(np.abs(antenna_incident) ** 2, axis=0) - np.sum(np.abs(antenna_reflected) ** 2, axis=0)
        insertion = antenna_accepted / np.maximum(source_accepted, EPS)
        transducer = antenna_accepted / np.maximum(incident_power, EPS)
        passive_rl = float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(total_external)), EPS))))
        block_efficiency = matched_efficiency(correction)
        total_efficiency = cascade_matched_efficiency(launch, correction)
        frequency_rows.append(
            {
                "frequency_ghz": float(frequency),
                "passive_rl_min_db": passive_rl,
                "active_rl_min_db": active_rl,
                "total_rl_min_db": total_rl,
                "correction_block_efficiency_min": block_efficiency,
                "launch_plus_block_efficiency_min": total_efficiency,
                "actual_load_insertion_efficiency_min": float(np.min(insertion)),
                "actual_load_transducer_efficiency_min": float(np.min(transducer)),
                "source_count": int(np.sum(selected)),
            }
        )
        selected_rows = [row for row, keep in zip(stimulus_rows, selected) if keep]
        total_by_source = -10.0 * np.log10(np.maximum(reflected_power / np.maximum(incident_power, EPS), EPS))
        for index, metadata in enumerate(selected_rows):
            source_rows.append(
                {
                    **metadata,
                    "worst_port_index": int(worst_port[index]),
                    "active_rl_db": float(active_by_source[index]),
                    "total_rl_db": float(total_by_source[index]),
                    "actual_load_insertion_efficiency": float(insertion[index]),
                    "actual_load_transducer_efficiency": float(transducer[index]),
                }
            )
    summary = {
        "passive_rl_min_db": min(row["passive_rl_min_db"] for row in frequency_rows),
        "active_rl_min_db": min(row["active_rl_min_db"] for row in frequency_rows),
        "total_rl_min_db": min(row["total_rl_min_db"] for row in frequency_rows),
        "correction_block_efficiency_min": min(row["correction_block_efficiency_min"] for row in frequency_rows),
        "launch_plus_block_efficiency_min": min(row["launch_plus_block_efficiency_min"] for row in frequency_rows),
        "actual_load_insertion_efficiency_min": min(row["actual_load_insertion_efficiency_min"] for row in frequency_rows),
        "actual_load_transducer_efficiency_min": min(row["actual_load_transducer_efficiency_min"] for row in frequency_rows),
    }
    return {"summary": summary, "frequency_rows": frequency_rows, "source_rows": source_rows}


def objective_value(metrics: dict[str, float], gates: dict[str, float], ideal: bool) -> float:
    active_target = float(gates["minimum_ideal_active_rl_db"] if ideal else gates["minimum_nominal_active_rl_db"])
    total_target = float(gates["minimum_ideal_total_rl_db"] if ideal else gates["minimum_nominal_total_rl_db"])
    value = 180.0 * max(active_target - metrics["active_rl_min_db"], 0.0) ** 2
    value += 90.0 * max(total_target - metrics["total_rl_min_db"], 0.0) ** 2
    value += 20.0 * max(float(gates["minimum_nominal_passive_rl_db"]) - metrics["passive_rl_min_db"], 0.0) ** 2
    value += 12000.0 * max(float(gates["minimum_launch_plus_block_efficiency"]) - metrics["launch_plus_block_efficiency_min"], 0.0) ** 2
    value += 12000.0 * max(float(gates["minimum_actual_load_insertion_efficiency"]) - metrics["actual_load_insertion_efficiency_min"], 0.0) ** 2
    if not ideal:
        value += 12000.0 * max(float(gates["minimum_correction_block_efficiency"]) - metrics["correction_block_efficiency_min"], 0.0) ** 2
    value -= 3.0 * min(metrics["active_rl_min_db"], active_target + 2.0)
    value -= min(metrics["total_rl_min_db"], total_target + 2.0)
    return float(value)


def design_scenarios(parameters: np.ndarray) -> list[tuple[np.ndarray, float]]:
    scenarios = [(parameters.copy(), 1.0)]
    for z_even, z_odd, phase, loading, q_scale in (
        (1.03, 0.97, 1.00, 1.00, 1.0),
        (0.97, 1.03, 1.00, 1.00, 1.0),
        (1.00, 1.00, 1.02, 1.00, 1.0),
        (1.00, 1.00, 0.98, 1.00, 1.0),
        (1.00, 1.00, 1.00, 1.07, 1.0),
        (1.00, 1.00, 1.00, 0.93, 1.0),
        (1.00, 1.00, 1.00, 1.00, 0.9),
    ):
        item = parameters.copy()
        item[0] *= z_even
        item[1] *= z_odd
        item[2] *= phase
        item[4:6] *= loading
        scenarios.append((item, q_scale))
    return scenarios


def gate_summary(metrics: dict[str, float], gates: dict[str, float], ideal: bool) -> dict[str, bool]:
    checks = {
        "active_rl": metrics["active_rl_min_db"] >= float(gates["minimum_ideal_active_rl_db"] if ideal else gates["minimum_nominal_active_rl_db"]),
        "total_rl": metrics["total_rl_min_db"] >= float(gates["minimum_ideal_total_rl_db"] if ideal else gates["minimum_nominal_total_rl_db"]),
        "passive_rl": metrics["passive_rl_min_db"] >= float(gates["minimum_nominal_passive_rl_db"]),
        "launch_plus_block_efficiency": metrics["launch_plus_block_efficiency_min"] >= float(gates["minimum_launch_plus_block_efficiency"]),
        "actual_load_insertion_efficiency": metrics["actual_load_insertion_efficiency_min"] >= float(gates["minimum_actual_load_insertion_efficiency"]),
    }
    if not ideal:
        checks["correction_block_efficiency"] = metrics["correction_block_efficiency_min"] >= float(gates["minimum_correction_block_efficiency"])
    return checks


def component_description(parameters: np.ndarray) -> dict[str, Any]:
    omega = 2.0 * math.pi * 10.0e9

    def shunt(value: float) -> dict[str, float | str]:
        if value >= 0.0:
            return {"kind": "capacitive", "value_pf": value / omega * 1.0e12, "susceptance_s": value}
        return {"kind": "inductive", "value_nh": -1.0 / (omega * value) * 1.0e9, "susceptance_s": value}

    return {
        "even_mode_impedance_ohm": float(parameters[0]),
        "odd_mode_impedance_ohm": float(parameters[1]),
        "even_mode_electrical_length_deg_at_10ghz": float(parameters[2]),
        "odd_mode_electrical_length_deg_at_10ghz": float(parameters[2] * parameters[3]),
        "odd_to_even_electrical_length_ratio": float(parameters[3]),
        "symmetric_ground_branch": shunt(float(parameters[4])),
        "pair_bridge_branch": shunt(float(parameters[5])),
        "input_loading_fraction": float(parameters[6]),
        "output_loading_fraction": float(1.0 - parameters[6]),
    }


def finalize_existing(config: dict[str, Any], out: Path) -> dict[str, Any]:
    topology = read_json(out / "local_topology_support_gate.json")
    ideal = read_json(out / "ideal_local_modal_upper_bound.json")
    physical = read_json(out / "finite_q_physical_upper_bound.json")
    tolerance_rows = list(csv.DictReader((out / "tolerance_1000_metrics.csv").open(encoding="utf-8-sig")))
    gates = config["gates"]

    requested_tolerance = []
    full_chain_tolerance = []
    for row in tolerance_rows:
        requested = bool(
            float(row["active_rl_min_db"]) >= float(gates["minimum_tolerance_active_rl_db"])
            and float(row["total_rl_min_db"]) >= float(gates["minimum_tolerance_total_rl_db"])
            and float(row["passive_rl_min_db"]) >= float(gates["minimum_tolerance_passive_rl_db"])
            and float(row["correction_block_efficiency_min"]) >= float(gates["minimum_correction_block_efficiency"])
        )
        full_chain = bool(
            requested
            and float(row["launch_plus_block_efficiency_min"]) >= float(gates["minimum_launch_plus_block_efficiency"])
            and float(row["actual_load_insertion_efficiency_min"]) >= float(gates["minimum_actual_load_insertion_efficiency"])
        )
        requested_tolerance.append(requested)
        full_chain_tolerance.append(full_chain)

    requested_rate = float(np.mean(requested_tolerance))
    full_chain_rate = float(np.mean(full_chain_tolerance))
    ideal_capability = bool(
        ideal["active_rl_min_db"] >= float(gates["minimum_ideal_active_rl_db"])
        and ideal["total_rl_min_db"] >= float(gates["minimum_ideal_total_rl_db"])
        and ideal["passive_rl_min_db"] >= float(gates["minimum_nominal_passive_rl_db"])
    )
    parameters = np.asarray(physical["parameters"], dtype=float)
    physical_circuit = bool(
        physical["active_rl_min_db"] >= float(gates["minimum_nominal_active_rl_db"])
        and physical["total_rl_min_db"] >= float(gates["minimum_nominal_total_rl_db"])
        and physical["passive_rl_min_db"] >= float(gates["minimum_nominal_passive_rl_db"])
        and physical["correction_block_efficiency_min"] >= float(gates["minimum_correction_block_efficiency"])
        and parameters[0] - parameters[1] >= float(gates["minimum_even_minus_odd_impedance_ohm"])
    )
    full_chain_nominal = bool(
        physical["launch_plus_block_efficiency_min"] >= float(gates["minimum_launch_plus_block_efficiency"])
        and physical["actual_load_insertion_efficiency_min"] >= float(gates["minimum_actual_load_insertion_efficiency"])
    )
    tolerance_summary = read_json(out / "tolerance_summary.json")
    tolerance_summary.update(
        {
            "requested_circuit_joint_pass_count": int(sum(requested_tolerance)),
            "requested_circuit_joint_pass_rate": requested_rate,
            "requested_circuit_gate_pass": requested_rate >= float(gates["minimum_tolerance_joint_pass_rate"]),
            "full_chain_joint_pass_count": int(sum(full_chain_tolerance)),
            "full_chain_joint_pass_rate": full_chain_rate,
            "full_chain_gate_pass": full_chain_rate >= float(gates["minimum_tolerance_joint_pass_rate"]),
            "gate_semantics": "The 97% preregistered circuit gate applies to the new correction block. Combined-launch and actual-load efficiency are retained as the subsequent physical-chain gate.",
        }
    )
    write_json(out / "tolerance_summary.json", tolerance_summary)

    circuit_gate = bool(
        topology["gate_pass"]
        and ideal_capability
        and physical_circuit
        and tolerance_summary["requested_circuit_gate_pass"]
    )
    decision = {
        "stage": "v1.23_circuit_upper_bound_complete",
        "decision": "allow_one_10ghz_network_only_s8_smoke" if circuit_gate else "stop_circuit_upper_bound_failed",
        "topology_support_gate_pass": topology["gate_pass"],
        "ideal_local_modal_capability_gate_pass": ideal_capability,
        "finite_q_correction_block_gate_pass": physical_circuit,
        "requested_tolerance_gate_pass": tolerance_summary["requested_circuit_gate_pass"],
        "circuit_upper_bound_gate_pass": circuit_gate,
        "full_chain_nominal_efficiency_gate_pass": full_chain_nominal,
        "full_chain_tolerance_gate_pass": tolerance_summary["full_chain_gate_pass"],
        "allow_initial_10ghz_network_only_s8": circuit_gate,
        "allow_three_frequency_hfss": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "warning": "The correction-block upper bound passes, but full-chain efficiency is predicted to fail; only one 10 GHz physical smoke is authorized.",
    }
    write_json(out / "stage_decision.json", decision)
    return {"decision": decision, "tolerance": tolerance_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ideal-maxiter", type=int)
    parser.add_argument("--physical-maxiter", type=int)
    parser.add_argument("--tolerance-samples", type=int)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    out = args.out_dir.resolve() if args.out_dir else resolve(config["output_directory"])
    if args.finalize_existing:
        if not out.exists():
            raise FileNotFoundError(out)
        print(json.dumps(finalize_existing(config, out), indent=2, ensure_ascii=True))
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.23 output: {out}")
    out.mkdir(parents=True)
    shutil.copy2(args.config, out / "config_snapshot.json")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag_commit = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag_commit != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag_commit}")

    inputs = config["inputs"]
    frozen_paths = {
        "launch_s2": resolve(inputs["frozen_launch_s2"]),
        "antenna_s4": resolve(inputs["trusted_antenna_s4"]),
        "stimulus_csv": resolve(inputs["stimulus_csv"]),
        "stimulus_npz": resolve(inputs["stimulus_npz"]),
    }
    expected_hashes = {
        "launch_s2": inputs["frozen_launch_s2_sha256"],
        "antenna_s4": inputs["trusted_antenna_s4_sha256"],
        "stimulus_csv": inputs["stimulus_csv_sha256"],
        "stimulus_npz": inputs["stimulus_npz_sha256"],
    }
    manifest_rows = []
    for role, path in frozen_paths.items():
        observed = sha256(path)
        if observed != expected_hashes[role]:
            raise RuntimeError(f"Hash mismatch for {role}: {observed}")
        manifest_rows.append({"role": role, "path": str(path), "sha256": observed, "size_bytes": path.stat().st_size})
    current_paths = []
    for frequency in config["frequencies_ghz"]:
        key = str(float(frequency))
        path = resolve(inputs["current_modal_s8_by_frequency"][key])
        observed = sha256(path)
        if observed != inputs["current_modal_s8_sha256_by_frequency"][key]:
            raise RuntimeError(f"Hash mismatch for current S8 at {key} GHz")
        current_paths.append(path)
        manifest_rows.append({"role": f"current_modal_s8_{key}ghz", "path": str(path), "sha256": observed, "size_bytes": path.stat().st_size})
    write_csv(out / "frozen_input_manifest.csv", manifest_rows)

    target_frequencies = np.asarray(config["frequencies_ghz"], dtype=float)
    launch_f, launch_s2 = reordered_network(frozen_paths["launch_s2"], ["PRE_0", "POST_0"], 2)
    launch_matrices = np.asarray([repeated_launch_s8(launch_s2[int(np.argmin(np.abs(launch_f - value)))]) for value in target_frequencies])
    antenna_f, antenna_s4 = parse_touchstone(frozen_paths["antenna_s4"], 4)
    antenna_matrices = np.asarray([antenna_s4[int(np.argmin(np.abs(antenna_f - value)))] for value in target_frequencies])
    expected_ports = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    current_matrices = []
    for frequency, path in zip(target_frequencies, current_paths):
        local_f, local_s = reordered_network(path, expected_ports, 8)
        current_matrices.append(local_s[int(np.argmin(np.abs(local_f - frequency)))])
    current_matrices = np.asarray(current_matrices)

    all_rows, all_vectors, all_considered = load_stimuli(resolve(inputs["stimulus_root"]))
    side = np.asarray([int(row["side"]) == 2 for row in all_rows])
    stimulus_rows = [row for row, keep in zip(all_rows, side) if keep]
    vectors = all_vectors[side, :4]
    considered = all_considered[side, :4]
    if len(stimulus_rows) != 285:
        raise RuntimeError(f"Expected 285 frozen 2x2 stimuli, found {len(stimulus_rows)}")

    pairs = [tuple(int(value) for value in pair) for pair in config["topology"]["x_neighbor_pairs"]]
    transform = modal_transform(pairs)
    impedance_rows = []
    impedance_payload = {"modal_transform": transform.tolist(), "frequencies": []}
    unsupported_fractions = []
    for frequency, antenna in zip(target_frequencies, antenna_matrices):
        impedance = s_to_z(antenna, float(config["reference_impedance_ohm"]))
        modal = transform.T @ impedance @ transform
        supported = np.diag(np.diag(modal))
        residual = modal - supported
        unsupported = float(np.linalg.norm(residual) ** 2 / max(np.linalg.norm(modal) ** 2, EPS))
        unsupported_fractions.append(unsupported)
        resistance = 0.5 * (impedance + impedance.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(resistance)
        eigenvalues = np.maximum(np.real(eigenvalues), 1.0e-9)
        whitening = math.sqrt(float(config["reference_impedance_ohm"])) * eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.conj().T
        ideal_series = float(config["reference_impedance_ohm"]) * np.eye(4) - impedance
        impedance_payload["frequencies"].append(
            {
                "frequency_ghz": float(frequency),
                "antenna_impedance": complex_matrix_payload(impedance),
                "modal_impedance": complex_matrix_payload(modal),
                "supported_modal_impedance": complex_matrix_payload(supported),
                "unsupported_modal_impedance": complex_matrix_payload(residual),
                "unsupported_modal_energy_fraction": unsupported,
                "full_matrix_resistance_whitening_transform": complex_matrix_payload(whitening),
                "ideal_series_target_50ohm_minus_load": complex_matrix_payload(ideal_series),
            }
        )
        for mode_index, mode_name in enumerate(config["topology"]["mode_order"]):
            value = modal[mode_index, mode_index]
            impedance_rows.append(
                {
                    "frequency_ghz": float(frequency),
                    "mode_index": mode_index,
                    "mode_name": mode_name,
                    "resistance_ohm": float(np.real(value)),
                    "reactance_ohm": float(np.imag(value)),
                    "magnitude_ohm": float(abs(value)),
                    "phase_deg": float(np.angle(value, deg=True)),
                    "unsupported_modal_energy_fraction": unsupported,
                }
            )
    write_json(out / "load_modal_impedance_audit.json", impedance_payload)
    write_csv(out / "modal_impedance_by_frequency.csv", impedance_rows)
    topology_summary = {
        "maximum_unsupported_modal_energy_fraction": max(unsupported_fractions),
        "gate": float(config["gates"]["maximum_unsupported_modal_energy_fraction"]),
        "gate_pass": max(unsupported_fractions) <= float(config["gates"]["maximum_unsupported_modal_energy_fraction"]),
    }
    write_json(out / "local_topology_support_gate.json", topology_summary)

    current = evaluate_networks(current_matrices, np.asarray([ideal_through_s8()] * 3), antenna_matrices, target_frequencies, stimulus_rows, vectors, considered)
    current_worst = sorted(current["source_rows"], key=lambda row: float(row["active_rl_db"]))[:30]
    write_csv(out / "current_v122_frequency_metrics.csv", current["frequency_rows"])
    write_csv(out / "current_v122_worst_active_events.csv", current_worst)

    z0 = float(config["reference_impedance_ohm"])
    gates = config["gates"]
    ideal_bounds_cfg = config["topology"]["ideal_local_mode_bounds"]
    ideal_bounds = (
        [ideal_bounds_cfg["mode_impedance_ohm"]] * 4
        + [ideal_bounds_cfg["mode_electrical_length_deg_at_10ghz"]] * 4
        + [ideal_bounds_cfg["mode_shunt_susceptance_s_at_10ghz"]] * 4
        + [ideal_bounds_cfg["mode_input_loading_fraction"]] * 4
    )

    def evaluate_ideal(parameters: np.ndarray) -> dict[str, Any]:
        corrections = np.asarray([ideal_correction_s8(parameters, float(frequency), z0, transform) for frequency in target_frequencies])
        return evaluate_networks(corrections, launch_matrices, antenna_matrices, target_frequencies, stimulus_rows, vectors, considered)

    def ideal_objective(parameters: np.ndarray) -> float:
        return objective_value(evaluate_ideal(parameters)["summary"], gates, True)

    optimization = config["optimization"]
    ideal_result = differential_evolution(
        ideal_objective,
        bounds=ideal_bounds,
        seed=int(config["seed"]),
        maxiter=int(args.ideal_maxiter or optimization["ideal_maxiter"]),
        popsize=int(optimization["ideal_popsize"]),
        tol=float(optimization["tolerance"]),
        polish=bool(optimization["polish"]),
        workers=1,
    )
    ideal = evaluate_ideal(np.asarray(ideal_result.x))
    ideal_checks = gate_summary(ideal["summary"], gates, True)
    ideal_summary = {
        **ideal["summary"],
        "parameters": np.asarray(ideal_result.x).tolist(),
        "objective": float(ideal_result.fun),
        "iterations": int(ideal_result.nit),
        "evaluations": int(ideal_result.nfev),
        "gate_checks": ideal_checks,
        "gate_pass": all(ideal_checks.values()),
    }
    write_json(out / "ideal_local_modal_upper_bound.json", ideal_summary)
    write_csv(out / "ideal_local_modal_frequency_metrics.csv", ideal["frequency_rows"])

    physical_cfg = config["topology"]["physical_common_parameter_bounds"]
    physical_bounds = [
        physical_cfg["even_mode_impedance_ohm"],
        physical_cfg["odd_mode_impedance_ohm"],
        physical_cfg["even_mode_electrical_length_deg_at_10ghz"],
        physical_cfg["odd_to_even_electrical_length_ratio"],
        physical_cfg["symmetric_ground_susceptance_s_at_10ghz"],
        physical_cfg["pair_bridge_susceptance_s_at_10ghz"],
        physical_cfg["input_loading_fraction"],
    ]
    finite_q = config["topology"]["finite_q"]

    def evaluate_physical(parameters: np.ndarray, q_scale: float = 1.0) -> dict[str, Any]:
        corrections = np.asarray([physical_correction_s8(parameters, float(frequency), z0, transform, finite_q, q_scale) for frequency in target_frequencies])
        return evaluate_networks(corrections, launch_matrices, antenna_matrices, target_frequencies, stimulus_rows, vectors, considered)

    def physical_objective(parameters: np.ndarray) -> float:
        values = [objective_value(evaluate_physical(item, q_scale)["summary"], gates, False) for item, q_scale in design_scenarios(parameters)]
        impedance_gap = max(float(gates["minimum_even_minus_odd_impedance_ohm"]) - (float(parameters[0]) - float(parameters[1])), 0.0)
        return float(max(values) + 0.10 * np.mean(values) + 500.0 * impedance_gap**2)

    physical_result = differential_evolution(
        physical_objective,
        bounds=physical_bounds,
        seed=int(config["seed"]) + 1,
        maxiter=int(args.physical_maxiter or optimization["physical_maxiter"]),
        popsize=int(optimization["physical_popsize"]),
        tol=float(optimization["tolerance"]),
        polish=bool(optimization["polish"]),
        workers=1,
    )
    parameters = np.asarray(physical_result.x)
    physical = evaluate_physical(parameters)
    physical_checks = gate_summary(physical["summary"], gates, False)
    physical_checks["even_odd_impedance_order"] = bool(
        parameters[0] - parameters[1] >= float(gates["minimum_even_minus_odd_impedance_ohm"])
    )
    physical_summary = {
        **physical["summary"],
        "parameters": parameters.tolist(),
        "components": component_description(parameters),
        "objective": float(physical_result.fun),
        "iterations": int(physical_result.nit),
        "evaluations": int(physical_result.nfev),
        "gate_checks": physical_checks,
        "gate_pass": all(physical_checks.values()),
    }
    write_json(out / "finite_q_physical_upper_bound.json", physical_summary)
    write_csv(out / "finite_q_physical_frequency_metrics.csv", physical["frequency_rows"])
    write_csv(out / "finite_q_physical_worst_active_events.csv", sorted(physical["source_rows"], key=lambda row: float(row["active_rl_db"]))[:30])

    rng = np.random.default_rng(int(config["seed"]) + 2)
    tolerance_cfg = config["tolerance"]
    tolerance_rows = []
    count = int(args.tolerance_samples or tolerance_cfg["sample_count"])
    clip = float(tolerance_cfg["clip_sigma"])
    for index in range(count):
        normal = np.clip(rng.normal(size=7), -clip, clip)
        varied = parameters.copy()
        varied[0] *= 1.0 + float(tolerance_cfg["impedance_sigma_fraction"]) * normal[0]
        varied[1] *= 1.0 + float(tolerance_cfg["impedance_sigma_fraction"]) * normal[1]
        varied[2] *= 1.0 + float(tolerance_cfg["electrical_length_sigma_fraction"]) * normal[2]
        varied[3] += 0.5 * float(tolerance_cfg["electrical_length_sigma_fraction"]) * normal[3]
        varied[4] *= 1.0 + float(tolerance_cfg["loading_sigma_fraction"]) * normal[4]
        varied[5] *= 1.0 + float(tolerance_cfg["loading_sigma_fraction"]) * normal[5]
        varied[6] += 0.02 * normal[6]
        for parameter_index, bounds in enumerate(physical_bounds):
            varied[parameter_index] = np.clip(varied[parameter_index], float(bounds[0]), float(bounds[1]))
        q_scale = float(np.clip(1.0 + float(tolerance_cfg["q_sigma_fraction"]) * rng.normal(), 0.7, 1.3))
        metrics = evaluate_physical(varied, q_scale)["summary"]
        joint = bool(
            metrics["active_rl_min_db"] >= float(gates["minimum_tolerance_active_rl_db"])
            and metrics["total_rl_min_db"] >= float(gates["minimum_tolerance_total_rl_db"])
            and metrics["passive_rl_min_db"] >= float(gates["minimum_tolerance_passive_rl_db"])
            and metrics["correction_block_efficiency_min"] >= float(gates["minimum_correction_block_efficiency"])
            and metrics["launch_plus_block_efficiency_min"] >= float(gates["minimum_launch_plus_block_efficiency"])
            and metrics["actual_load_insertion_efficiency_min"] >= float(gates["minimum_actual_load_insertion_efficiency"])
            and varied[0] - varied[1] >= float(gates["minimum_even_minus_odd_impedance_ohm"])
        )
        tolerance_rows.append({"sample": index, "q_scale": q_scale, **metrics, "joint_gate_pass": joint})
    write_csv(out / "tolerance_1000_metrics.csv", tolerance_rows)
    pass_rate = float(np.mean([row["joint_gate_pass"] for row in tolerance_rows]))
    tolerance_summary = {
        "sample_count": count,
        "joint_pass_count": int(sum(row["joint_gate_pass"] for row in tolerance_rows)),
        "joint_pass_rate": pass_rate,
        "required_pass_rate": float(gates["minimum_tolerance_joint_pass_rate"]),
        "gate_pass": pass_rate >= float(gates["minimum_tolerance_joint_pass_rate"]),
        "worst_active_rl_db": min(row["active_rl_min_db"] for row in tolerance_rows),
        "worst_total_rl_db": min(row["total_rl_min_db"] for row in tolerance_rows),
        "worst_correction_block_efficiency": min(row["correction_block_efficiency_min"] for row in tolerance_rows),
        "worst_launch_plus_block_efficiency": min(row["launch_plus_block_efficiency_min"] for row in tolerance_rows),
    }
    write_json(out / "tolerance_summary.json", tolerance_summary)

    all_pass = bool(topology_summary["gate_pass"] and ideal_summary["gate_pass"] and physical_summary["gate_pass"] and tolerance_summary["gate_pass"])
    if not topology_summary["gate_pass"]:
        decision_name = "stop_local_topology_unsupported_modal_energy"
    elif not ideal_summary["gate_pass"]:
        decision_name = "stop_ideal_local_modal_upper_bound_failed"
    elif not physical_summary["gate_pass"]:
        decision_name = "stop_finite_q_manufacturable_upper_bound_failed"
    elif not tolerance_summary["gate_pass"]:
        decision_name = "stop_tolerance_reserve_failed"
    else:
        decision_name = "allow_one_10ghz_network_only_s8_smoke"
    decision = {
        "stage": "v1.23_circuit_upper_bound_complete",
        "decision": decision_name,
        "topology_support_gate_pass": topology_summary["gate_pass"],
        "ideal_local_modal_gate_pass": ideal_summary["gate_pass"],
        "finite_q_physical_gate_pass": physical_summary["gate_pass"],
        "tolerance_gate_pass": tolerance_summary["gate_pass"],
        "all_circuit_gates_pass": all_pass,
        "allow_initial_10ghz_network_only_s8": all_pass,
        "allow_three_frequency_hfss": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    np.savez_compressed(
        out / "selected_circuit_operator.npz",
        frequencies_ghz=target_frequencies,
        physical_parameters=parameters,
        modal_transform=transform,
        correction_s8=np.asarray([physical_correction_s8(parameters, float(frequency), z0, transform, finite_q) for frequency in target_frequencies]),
        launch_s8=launch_matrices,
        antenna_s4=antenna_matrices,
    )
    finalized = finalize_existing(config, out)
    print(json.dumps({"decision": finalized["decision"], "topology": topology_summary, "ideal": ideal_summary, "physical": physical_summary, "tolerance": finalized["tolerance"]}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
