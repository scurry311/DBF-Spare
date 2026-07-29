#!/usr/bin/env python3
"""Design a realizable three-frequency S-L-S matching network on frozen commands."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from run_v16_robust_drift_oracle import load_npz, ri_to_complex
from validate_trusted_eep_hfss_residuals import series_network_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v21_three_frequency_broadband_matching.json"
DEFAULT_FROZEN = ROOT / "hfss_outputs" / "v21_frozen_v112_replay_20260729_run03" / "frozen_v112_replay_candidates.npz"
DEFAULT_NOMINAL = ROOT / "hfss_outputs" / "fixed_mesh_eep256_20260723_run05" / "grounded_patch_eep_operator_256port.npz"
DEFAULT_LOW = ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_20260729_run01" / "operator" / "grounded_patch_eep_operator_256port.npz"
DEFAULT_HIGH = ROOT / "hfss_outputs" / "v19_perturbed_operator_frequency_high_20260729_run01" / "operator" / "grounded_patch_eep_operator_256port.npz"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v21_three_frequency_broadband_match_20260729_run02"
Z0 = 50.0
EPS = 1.0e-12


@dataclass
class FrequencySources:
    name: str
    frequency_ghz: float
    raw_s: np.ndarray
    old_map: np.ndarray
    external: np.ndarray
    considered: np.ndarray
    denominator: np.ndarray
    candidate_index: np.ndarray
    old_antenna: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--low-operator", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--high-operator", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--maxiter", type=int, default=0)
    parser.add_argument("--popsize", type=int, default=0)
    parser.add_argument(
        "--variants", nargs="+", choices=("uniform_sls", "geometry3_sls"), default=None
    )
    parser.add_argument("--evaluate-initial-only", action="store_true")
    parser.add_argument("--initial-network-matrices", type=Path)
    parser.add_argument("--initial-variant", default="uniform_sls")
    parser.add_argument(
        "--method",
        choices=("differential_evolution", "coordinate"),
        default="differential_evolution",
    )
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


def geometry_classes(element_ixiy: np.ndarray) -> tuple[np.ndarray, list[str]]:
    ix = np.asarray(element_ixiy[:, 0], dtype=int)
    iy = np.asarray(element_ixiy[:, 1], dtype=int)
    x_edge = (ix == 0) | (ix == 15)
    y_edge = (iy == 0) | (iy == 15)
    labels = np.full(len(ix), 2, dtype=np.int8)
    labels[x_edge ^ y_edge] = 1
    labels[x_edge & y_edge] = 0
    return labels, ["corner", "edge", "interior"]


def frequency_scaled_reactance(reference: np.ndarray, ratio: float) -> np.ndarray:
    return np.where(reference >= 0.0, reference * ratio, reference / ratio)


def frequency_scaled_susceptance(reference: np.ndarray, ratio: float) -> np.ndarray:
    return np.where(reference >= 0.0, reference * ratio, reference / ratio)


def sls_port_s(
    parameters: np.ndarray,
    labels: np.ndarray,
    frequency_ghz: float,
    reference_frequency_ghz: float,
    q_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = np.asarray(parameters, dtype=float).reshape(-1, 3)
    x1_ref = table[labels, 0]
    b_ref = table[labels, 1]
    x3_ref = table[labels, 2]
    ratio = float(frequency_ghz / reference_frequency_ghz)
    x1 = frequency_scaled_reactance(x1_ref, ratio)
    x3 = frequency_scaled_reactance(x3_ref, ratio)
    susceptance = frequency_scaled_susceptance(b_ref, ratio)
    z1 = np.abs(x1) / q_value + 1j * x1
    z3 = np.abs(x3) / q_value + 1j * x3
    y2 = np.abs(susceptance) / q_value + 1j * susceptance
    a = 1.0 + z1 * y2
    b = (1.0 + z1 * y2) * z3 + z1
    c = y2
    d = 1.0 + y2 * z3
    denominator = a + b / Z0 + c * Z0 + d
    determinant = a * d - b * c
    s11 = (a + b / Z0 - c * Z0 - d) / denominator
    s21 = 2.0 / denominator
    s12 = 2.0 * determinant / denominator
    s22 = (-a + b / Z0 - c * Z0 + d) / denominator
    return s11, s12, s21, s22


def compose_network(
    raw_s: np.ndarray,
    parameters: np.ndarray,
    labels: np.ndarray,
    frequency_ghz: float,
    reference_frequency_ghz: float,
    q_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s11, s12, s21, s22 = sls_port_s(
        parameters, labels, frequency_ghz, reference_frequency_ghz, q_value
    )
    identity = np.eye(raw_s.shape[0], dtype=np.complex128)
    antenna_map = np.linalg.solve(identity - s22[:, None] * raw_s, np.diag(s21))
    antenna_reflection_map = raw_s @ antenna_map
    composite = np.diag(s11) + s12[:, None] * antenna_reflection_map
    return composite, antenna_map, antenna_reflection_map


def source_bundle(
    state_tasks: np.ndarray,
    masks: np.ndarray,
    candidate_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources: list[np.ndarray] = []
    considered: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    owners: list[int] = []
    for local_index, candidate in enumerate(candidate_indices):
        k_value = state_tasks[local_index].shape[1]
        mask = masks[candidate]
        candidate_tasks = state_tasks[local_index]
        values = [np.sum(candidate_tasks, axis=1)]
        values.extend(candidate_tasks[:, task] for task in range(k_value))
        for source_index, source in enumerate(values):
            amplitude = np.abs(source)
            threshold = 1.0e-8 if source_index == 0 else max(float(np.max(amplitude)) * 0.1, EPS)
            sources.append(source)
            considered.append(mask & (amplitude >= threshold))
            denominators.append(np.maximum(amplitude, threshold))
            owners.append(int(candidate))
    return (
        np.stack(sources, axis=1),
        np.stack(considered, axis=1),
        np.stack(denominators, axis=1),
        np.asarray(owners, dtype=np.int64),
    )


def prepare_frequency_sources(
    frozen: dict[str, np.ndarray],
    operators: list[dict[str, np.ndarray]],
) -> list[FrequencySources]:
    state_tasks_all = ri_to_complex(frozen["state_tasks_real_imag"])
    masks = np.asarray(frozen["masks"], dtype=bool)
    k_values = np.asarray(frozen["k_values"], dtype=int)
    state_groups = ((0,), (1, 2), (3, 4))
    result: list[FrequencySources] = []
    for operator, states in zip(operators, state_groups):
        columns: list[np.ndarray] = []
        flags: list[np.ndarray] = []
        denoms: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        for state_index in states:
            ragged = [
                np.asarray(state_tasks_all[index, state_index, :, : k_values[index]], np.complex128)
                for index in range(len(k_values))
            ]
            for candidate, tasks in enumerate(ragged):
                x, c, d, o = source_bundle(
                    tasks[None, ...], masks, np.asarray([candidate], dtype=int)
                )
                columns.append(x)
                flags.append(c)
                denoms.append(d)
                owners.append(o)
        external = np.concatenate(columns, axis=1)
        considered = np.concatenate(flags, axis=1)
        denominator = np.concatenate(denoms, axis=1)
        candidate_index = np.concatenate(owners)
        raw_s = np.asarray(operator["s_raw"], dtype=np.complex128)
        old_s, old_map, _series = series_network_map(
            raw_s, float(operator["frequency_ghz"]) * 1.0e9
        )
        if np.max(np.abs(old_s - np.asarray(operator["s_matched"]))) > 1.0e-6:
            raise RuntimeError("Old series-match reconstruction mismatch")
        result.append(
            FrequencySources(
                name=("nominal", "frequency_low", "frequency_high")[len(result)],
                frequency_ghz=float(operator["frequency_ghz"]),
                raw_s=raw_s,
                old_map=old_map,
                external=external,
                considered=considered,
                denominator=denominator,
                candidate_index=candidate_index,
                old_antenna=old_map @ external,
            )
        )
    return result


def source_metrics(
    composite: np.ndarray,
    antenna_map: np.ndarray,
    antenna_reflection_map: np.ndarray,
    bundle: FrequencySources,
    *,
    compute_passivity: bool,
) -> dict[str, np.ndarray | float]:
    reflected = composite @ bundle.external
    gamma = np.abs(reflected) / bundle.denominator
    gamma[~bundle.considered] = 0.0
    worst_gamma = np.max(gamma, axis=0)
    worst_rl = -20.0 * np.log10(np.maximum(worst_gamma, 1.0e-15))
    input_power = np.sum(np.abs(bundle.external) ** 2, axis=0)
    reflected_power = np.sum(np.abs(reflected) ** 2, axis=0)
    total_rl = -10.0 * np.log10(
        np.maximum(reflected_power / np.maximum(input_power, EPS), 1.0e-15)
    )
    source_floor = np.minimum(worst_rl, total_rl)
    antenna_incident = antenna_map @ bundle.external
    antenna_reflected = antenna_reflection_map @ bundle.external
    antenna_accepted = np.sum(np.abs(antenna_incident) ** 2, axis=0) - np.sum(
        np.abs(antenna_reflected) ** 2, axis=0
    )
    external_accepted = input_power - reflected_power
    efficiency = antenna_accepted / np.maximum(external_accepted, EPS)
    inner = np.sum(np.conjugate(bundle.old_antenna) * antenna_incident, axis=0)
    correlation = np.abs(inner) / np.maximum(
        np.linalg.norm(bundle.old_antenna, axis=0)
        * np.linalg.norm(antenna_incident, axis=0),
        EPS,
    )
    norm_ratio_db = 20.0 * np.log10(
        np.maximum(
            np.linalg.norm(antenna_incident, axis=0)
            / np.maximum(np.linalg.norm(bundle.old_antenna, axis=0), EPS),
            EPS,
        )
    )
    return {
        "source_floor": source_floor,
        "efficiency": efficiency,
        "correlation": correlation,
        "norm_ratio_db": norm_ratio_db,
        "passive_rl_min_db": float(
            np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(composite)), 1.0e-15)))
        ),
        "passivity_sigma_max": (
            float(np.max(np.linalg.svd(composite, compute_uv=False)))
            if compute_passivity
            else float("nan")
        ),
    }


def evaluate_parameters(
    parameters: np.ndarray,
    labels: np.ndarray,
    bundles: list[FrequencySources],
    k_values: np.ndarray,
    reference_frequency_ghz: float,
    q_value: float,
    *,
    compute_passivity: bool = True,
) -> dict[str, Any]:
    scene_count = len(k_values)
    candidate_floor = np.full(scene_count, np.inf, dtype=float)
    candidate_efficiency = np.full(scene_count, np.inf, dtype=float)
    passive_rl: list[float] = []
    passivity: list[float] = []
    correlations: list[float] = []
    norm_changes: list[float] = []
    matrices: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    for bundle in bundles:
        composite, antenna_map, antenna_reflection_map = compose_network(
            bundle.raw_s,
            parameters,
            labels,
            bundle.frequency_ghz,
            reference_frequency_ghz,
            q_value,
        )
        metrics = source_metrics(
            composite,
            antenna_map,
            antenna_reflection_map,
            bundle,
            compute_passivity=compute_passivity,
        )
        for candidate in range(scene_count):
            members = bundle.candidate_index == candidate
            candidate_floor[candidate] = min(
                candidate_floor[candidate], float(np.min(metrics["source_floor"][members]))
            )
            candidate_efficiency[candidate] = min(
                candidate_efficiency[candidate], float(np.min(metrics["efficiency"][members]))
            )
        passive_rl.append(float(metrics["passive_rl_min_db"]))
        if compute_passivity:
            passivity.append(float(metrics["passivity_sigma_max"]))
        correlations.extend(np.asarray(metrics["correlation"], dtype=float).tolist())
        norm_changes.extend(np.asarray(metrics["norm_ratio_db"], dtype=float).tolist())
        matrices.append(composite)
        maps.append(antenna_map)
    strict_active = candidate_floor >= 10.0
    reserve = candidate_floor >= 11.0
    return {
        "candidate_floor": candidate_floor,
        "candidate_efficiency": candidate_efficiency,
        "active10_count": int(np.sum(strict_active)),
        "reserve11_count": int(np.sum(reserve)),
        "active10_by_k": {
            k: int(np.sum(strict_active[k_values == k])) for k in (2, 4, 6)
        },
        "reserve11_by_k": {k: int(np.sum(reserve[k_values == k])) for k in (2, 4, 6)},
        "worst_active_rl_db": float(np.min(candidate_floor)),
        "p10_active_rl_db": float(np.quantile(candidate_floor, 0.10)),
        "minimum_network_efficiency": float(np.min(candidate_efficiency)),
        "minimum_passive_rl_db": float(np.min(passive_rl)),
        "maximum_passivity_sigma": float(np.max(passivity)) if passivity else float("nan"),
        "minimum_old_map_correlation": float(np.min(correlations)),
        "maximum_old_map_norm_change_db": float(np.max(np.abs(norm_changes))),
        "s_external": np.stack(matrices),
        "antenna_map": np.stack(maps),
    }


def objective_value(result: dict[str, Any], k_values: np.ndarray) -> float:
    floor = np.asarray(result["candidate_floor"], dtype=float)
    retained: list[np.ndarray] = []
    for k_value, required in ((2, 6), (4, 5)):
        stratum = np.sort(floor[k_values == k_value])[::-1]
        retained.append(stratum[:required])
    focus_floor = np.concatenate(retained)
    deficit10 = np.maximum(10.0 - focus_floor, 0.0)
    deficit11 = np.maximum(11.0 - focus_floor, 0.0)
    reserve_deficit = max(11.0 - float(np.max(floor)), 0.0)
    passive_deficit = max(10.5 - float(result["minimum_passive_rl_db"]), 0.0)
    efficiency_deficit = max(0.95 - float(result["minimum_network_efficiency"]), 0.0)
    correlation_deficit = max(0.985 - float(result["minimum_old_map_correlation"]), 0.0)
    norm_excess = max(float(result["maximum_old_map_norm_change_db"]) - 1.0, 0.0)
    cardinality_penalty = (
        6.0 * max(6 - int(result["active10_by_k"][2]), 0)
        + 6.0 * max(5 - int(result["active10_by_k"][4]), 0)
        + 2.0 * max(1 - int(result["reserve11_count"]), 0)
    )
    return float(
        cardinality_penalty
        + 18.0 * np.mean(deficit10**2)
        + 2.5 * np.mean(deficit11**2)
        + 10.0 * np.max(deficit10) ** 2
        + 4.0 * reserve_deficit**2
        + 4.0 * passive_deficit**2
        + 500.0 * efficiency_deficit**2
        + 250.0 * correlation_deficit**2
        + 2.0 * norm_excess**2
        - 0.15 * np.mean(np.minimum(focus_floor, 12.0))
    )


def variant_labels(variant: str, geometry: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if variant == "uniform_sls":
        return np.zeros_like(geometry), ["all"]
    return geometry, ["corner", "edge", "interior"]


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite broadband match: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frozen = load_npz(args.frozen)
    operators = [load_npz(path) for path in (args.nominal_operator, args.low_operator, args.high_operator)]
    element_ixiy = np.asarray(operators[0]["element_ixiy"], dtype=int)
    geometry, _geometry_names = geometry_classes(element_ixiy)
    bundles = prepare_frequency_sources(frozen, operators)
    k_values = np.asarray(frozen["k_values"], dtype=int)
    network = protocol["network"]
    optimization = protocol["optimization"]
    reference_frequency = float(network["reference_frequency_ghz"])
    q_value = float(network["component_q"])
    variants = list(args.variants or network["variants"])
    maxiter = int(args.maxiter or optimization["differential_evolution_maxiter"])
    popsize = int(args.popsize or optimization["differential_evolution_popsize"])
    x_old = 2.0 * np.pi * reference_frequency * 1.0e9 * 0.533e-9
    external_initial: np.ndarray | None = None
    if args.initial_network_matrices:
        initial_payload = load_npz(args.initial_network_matrices)
        names = [str(value) for value in initial_payload["variant_names"]]
        if args.initial_variant not in names:
            raise ValueError(f"Initial variant {args.initial_variant!r} not found in {names}")
        external_initial = np.asarray(
            initial_payload["parameters"][names.index(args.initial_variant)], dtype=float
        )
        external_initial = external_initial[np.isfinite(external_initial)]
    all_results: list[dict[str, Any]] = []
    matrix_names: list[str] = []
    matrix_values: list[np.ndarray] = []
    map_values: list[np.ndarray] = []
    parameter_values: list[np.ndarray] = []
    parameter_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    baseline_matrices: list[np.ndarray] = []
    baseline_maps: list[np.ndarray] = []
    for bundle in bundles:
        s_matrix, antenna_map, _series = series_network_map(
            bundle.raw_s, bundle.frequency_ghz * 1.0e9
        )
        baseline_matrices.append(s_matrix)
        baseline_maps.append(antenna_map)
    baseline_parameters = np.asarray([x_old, 0.0, 0.0])
    baseline = evaluate_parameters(
        baseline_parameters,
        np.zeros(256, dtype=np.int8),
        bundles,
        k_values,
        reference_frequency,
        50.0,
    )
    baseline["s_external"] = np.stack(baseline_matrices)
    baseline["antenna_map"] = np.stack(baseline_maps)
    baseline["variant"] = "old_series_l_0p533nH_q50"
    baseline["objective"] = objective_value(baseline, k_values)
    all_results.append(baseline)

    for variant_index, variant in enumerate(variants):
        labels, class_names = variant_labels(variant, geometry)
        class_count = len(class_names)
        initial_base = (
            external_initial[:3]
            if external_initial is not None and external_initial.size >= 3
            else np.asarray([x_old, 0.0, 0.0])
        )
        initial = np.tile(initial_base, class_count)
        x_bounds = tuple(float(value) for value in network["series_reactance_bounds_ohm"])
        b_bounds = tuple(float(value) for value in network["shunt_susceptance_bounds_siemens"])
        bounds = [bound for _ in class_names for bound in (x_bounds, b_bounds, x_bounds)]
        evaluations = 0
        started = time.time()

        def objective(vector: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            result = evaluate_parameters(
                vector,
                labels,
                bundles,
                k_values,
                reference_frequency,
                q_value,
                compute_passivity=False,
            )
            return objective_value(result, k_values)

        if args.evaluate_initial_only:
            selected = initial
            optimizer_success = True
            optimizer_message = "initial_only"
            optimizer_value = objective(initial)
        elif args.method == "coordinate":
            selected = initial.copy()
            optimizer_value = objective(selected)
            lower = np.asarray([bound[0] for bound in bounds], dtype=float)
            upper = np.asarray([bound[1] for bound in bounds], dtype=float)
            steps = np.tile(np.asarray([10.0, 0.003, 10.0]), class_count)
            for sweep in range(maxiter):
                sweep_start = optimizer_value
                for parameter_index in range(len(selected)):
                    trials: list[tuple[float, np.ndarray]] = []
                    for multiplier in (-1.0, -0.5, 0.0, 0.5, 1.0):
                        trial = selected.copy()
                        trial[parameter_index] = np.clip(
                            trial[parameter_index] + multiplier * steps[parameter_index],
                            lower[parameter_index],
                            upper[parameter_index],
                        )
                        trials.append((objective(trial), trial))
                    optimizer_value, selected = min(trials, key=lambda item: item[0])
                result = evaluate_parameters(
                    selected,
                    labels,
                    bundles,
                    k_values,
                    reference_frequency,
                    q_value,
                    compute_passivity=False,
                )
                print(
                    f"{variant} coordinate={sweep + 1}/{maxiter} eval={evaluations} "
                    f"objective={optimizer_value:.4f} K2={result['active10_by_k'][2]}/7 "
                    f"K4={result['active10_by_k'][4]}/6 reserve={result['reserve11_count']} "
                    f"RL={result['minimum_passive_rl_db']:.2f} eff={result['minimum_network_efficiency']:.4f}",
                    flush=True,
                )
                steps *= 0.55
                if sweep_start - optimizer_value < 1.0e-5 and float(np.max(steps[::3])) < 0.5:
                    break
            optimizer_success = True
            optimizer_message = "bounded_coordinate_search"
        else:
            def callback(vector: np.ndarray, convergence: float) -> bool:
                result = evaluate_parameters(
                    vector,
                    labels,
                    bundles,
                    k_values,
                    reference_frequency,
                    q_value,
                    compute_passivity=False,
                )
                print(
                    f"{variant} eval={evaluations} conv={convergence:.4g} "
                    f"K2={result['active10_by_k'][2]}/7 K4={result['active10_by_k'][4]}/6 "
                    f"reserve={result['reserve11_count']} RL={result['minimum_passive_rl_db']:.2f} "
                    f"eff={result['minimum_network_efficiency']:.4f}",
                    flush=True,
                )
                return False

            optimized = differential_evolution(
                objective,
                bounds=bounds,
                seed=int(protocol["seed"]) + variant_index * 1009,
                maxiter=maxiter,
                popsize=popsize,
                polish=bool(optimization["polish"]),
                workers=int(optimization["workers"]),
                updating="immediate",
                x0=initial,
                callback=callback,
                tol=1.0e-3,
            )
            selected = np.asarray(optimized.x, dtype=float)
            optimizer_success = bool(optimized.success)
            optimizer_message = str(optimized.message)
            optimizer_value = float(optimized.fun)
        result = evaluate_parameters(
            selected, labels, bundles, k_values, reference_frequency, q_value
        )
        result.update(
            {
                "variant": variant,
                "objective": float(optimizer_value),
                "optimizer_success": optimizer_success,
                "optimizer_message": optimizer_message,
                "optimizer_evaluations": evaluations,
                "elapsed_seconds": time.time() - started,
                "parameters": selected,
                "class_names": class_names,
            }
        )
        all_results.append(result)

    for result in all_results:
        name = str(result["variant"])
        matrix_names.append(name)
        matrix_values.append(np.asarray(result["s_external"], np.complex64))
        map_values.append(np.asarray(result["antenna_map"], np.complex64))
        params = np.asarray(result.get("parameters", baseline_parameters), dtype=float)
        padded = np.full(9, np.nan, dtype=float)
        padded[: len(params)] = params
        parameter_values.append(padded)
        row = {
            "variant": name,
            "objective": float(result["objective"]),
            "active10_count": int(result["active10_count"]),
            "reserve11_count": int(result["reserve11_count"]),
            "k2_active10_count": int(result["active10_by_k"][2]),
            "k4_active10_count": int(result["active10_by_k"][4]),
            "k6_active10_count": int(result["active10_by_k"][6]),
            "k2_reserve11_count": int(result["reserve11_by_k"][2]),
            "k4_reserve11_count": int(result["reserve11_by_k"][4]),
            "k6_reserve11_count": int(result["reserve11_by_k"][6]),
            "worst_active_rl_db": float(result["worst_active_rl_db"]),
            "p10_active_rl_db": float(result["p10_active_rl_db"]),
            "minimum_passive_rl_db": float(result["minimum_passive_rl_db"]),
            "minimum_network_efficiency": float(result["minimum_network_efficiency"]),
            "maximum_passivity_sigma": float(result["maximum_passivity_sigma"]),
            "minimum_old_map_correlation": float(result["minimum_old_map_correlation"]),
            "maximum_old_map_norm_change_db": float(result["maximum_old_map_norm_change_db"]),
            "optimizer_success": result.get("optimizer_success", True),
            "optimizer_message": result.get("optimizer_message", "baseline"),
            "optimizer_evaluations": result.get("optimizer_evaluations", 0),
            "elapsed_seconds": result.get("elapsed_seconds", 0.0),
        }
        parameter_rows.append(row)
        floors = np.asarray(result["candidate_floor"], dtype=float)
        efficiencies = np.asarray(result["candidate_efficiency"], dtype=float)
        for candidate in range(len(k_values)):
            candidate_rows.append(
                {
                    "variant": name,
                    "freeze_index": candidate,
                    "sample_index": int(frozen["sample_index"][candidate]),
                    "k": int(k_values[candidate]),
                    "ratio": float(frozen["ratio"][candidate]),
                    "active_rl_floor_db": float(floors[candidate]),
                    "active10_pass": int(floors[candidate] >= 10.0),
                    "reserve11_pass": int(floors[candidate] >= 11.0),
                    "minimum_network_efficiency": float(efficiencies[candidate]),
                }
            )
        classes = result.get("class_names", ["all"])
        table = params.reshape(-1, 3)
        for class_index, class_name in enumerate(classes):
            parameter_rows.append(
                {
                    "variant": name,
                    "row_type": "component_parameters",
                    "class_name": class_name,
                    "source_series_reactance_ohm_at_10ghz": float(table[class_index, 0]),
                    "shunt_susceptance_siemens_at_10ghz": float(table[class_index, 1]),
                    "antenna_series_reactance_ohm_at_10ghz": float(table[class_index, 2]),
                }
            )
    write_csv(args.out_dir / "network_variant_summary.csv", parameter_rows)
    write_csv(args.out_dir / "candidate_active_rl_replay.csv", candidate_rows)
    np.savez_compressed(
        args.out_dir / "network_matrices.npz",
        variant_names=np.asarray(matrix_names),
        frequencies_ghz=np.asarray([bundle.frequency_ghz for bundle in bundles], np.float32),
        s_external=np.stack(matrix_values),
        antenna_map=np.stack(map_values),
        parameters=np.stack(parameter_values),
        geometry_class=geometry,
        geometry_class_names=np.asarray(["corner", "edge", "interior"]),
        topology=np.asarray(network["topology"]),
        component_q=np.asarray(q_value),
    )
    summary_rows = [row for row in parameter_rows if row.get("row_type", "summary") == "summary"]
    best = max(
        summary_rows[1:] if len(summary_rows) > 1 else summary_rows,
        key=lambda row: (
            int(row["k2_active10_count"] >= 6),
            int(row["k4_active10_count"] >= 5),
            int(row["reserve11_count"]),
            int(row["active10_count"]),
            float(row["minimum_passive_rl_db"]),
        ),
    )
    summary = {
        "protocol": protocol["protocol"],
        "frozen_scene_count": len(k_values),
        "variants_evaluated": matrix_names,
        "selected_for_eep_replay": best["variant"],
        "selected_active10_count": best["active10_count"],
        "selected_reserve11_count": best["reserve11_count"],
        "selected_k2_active10_count": best["k2_active10_count"],
        "selected_k4_active10_count": best["k4_active10_count"],
        "selected_minimum_passive_rl_db": best["minimum_passive_rl_db"],
        "selected_minimum_network_efficiency": best["minimum_network_efficiency"],
        "physical_scope": "frequency-dependent finite-Q circuit cascade on three HFSS raw S256/EEP operators",
        "embedded_hfss_validated": False,
        "mask_or_weight_changes": False,
        "thresholds_changed": False,
        "hfss_allowed": False,
        "critic_training_allowed": False,
    }
    (args.out_dir / "design_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
