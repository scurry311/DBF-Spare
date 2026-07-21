"""Design and smoke-test tiled 2x2/4x4 even/odd matching networks on HFSS S16.

The antenna S matrices are real HFSS exports. The mode splitters and L sections
are circuit-domain, lossless single-frequency models. A passing result permits
a 16x16 antenna rebuild, but it is not an HFSS validation of the feed network.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

from analyze_full_s256p_active_return import parse_touchstone
from design_eep_port_match import component_description
from design_port_class_matching import compose_nonuniform_network


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_OUT = ROOT / "hfss_outputs" / "modal_decoupling_20260714_run01" / "s16_design"
S16_CANDIDATES = {
    "short_plain": (
        ROOT
        / "hfss_outputs"
        / "geometry_feed_smoke_20260714_run03"
        / "short_plain"
        / "short_plain.s16p"
    ),
    "tload_2p0_l12p6": (
        ROOT
        / "hfss_outputs"
        / "geometry_feed_smoke_20260714_run02"
        / "short_tload_2p0"
        / "short_tload_2p0.s16p"
    ),
    "tload_3p0_l12p6": (
        ROOT
        / "hfss_outputs"
        / "geometry_feed_smoke_20260714_run02"
        / "short_tload_3p0"
        / "short_tload_3p0.s16p"
    ),
    "tload_1p5_l11p5": (
        ROOT
        / "hfss_outputs"
        / "geometry_feed_smoke_20260714_run03"
        / "retuned_tload_1p5_l11p5"
        / "retuned_tload_1p5_l11p5.s16p"
    ),
    "tload_3p0_l10p4": (
        ROOT
        / "hfss_outputs"
        / "geometry_feed_smoke_20260714_run03"
        / "retuned_tload_3p0_l10p4"
        / "retuned_tload_3p0_l10p4.s16p"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--joint-sweeps", type=int, default=2)
    parser.add_argument("--return-loss-db", type=float, default=10.0)
    parser.add_argument("--coupling-db", type=float, default=-15.0)
    parser.add_argument("--significant-power-relative-db", type=float, default=-30.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dct_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        for mode in range(size):
            scale = math.sqrt(1.0 / size) if mode == 0 else math.sqrt(2.0 / size)
            matrix[row, mode] = scale * math.cos(math.pi * (row + 0.5) * mode / size)
    return matrix


def mode_transform(topology: str) -> np.ndarray:
    if topology == "tiled_2x2_even_odd":
        local = np.kron(dct_matrix(2), dct_matrix(2))
        transform = np.zeros((16, 16), dtype=np.float64)
        for x0 in (0, 2):
            for y0 in (0, 2):
                indices = [(x0 + dx) * 4 + y0 + dy for dx in range(2) for dy in range(2)]
                transform[np.ix_(indices, indices)] = local
        return transform
    if topology == "full_4x4_even_odd":
        return np.kron(dct_matrix(4), dct_matrix(4))
    raise ValueError(topology)


def nearest_coupling(matrix: np.ndarray) -> tuple[float, float]:
    side = int(round(math.sqrt(matrix.shape[0])))
    if side * side != matrix.shape[0]:
        raise ValueError(f"Expected a square planar port grid, got {matrix.shape[0]} ports")
    x_values: list[float] = []
    y_values: list[float] = []
    for ix in range(side):
        for iy in range(side):
            index = ix * side + iy
            if ix < side - 1:
                x_values.extend(
                    (abs(matrix[index, index + side]), abs(matrix[index + side, index]))
                )
            if iy < side - 1:
                y_values.extend((abs(matrix[index, index + 1]), abs(matrix[index + 1, index])))
    return (
        float(20.0 * np.log10(max(max(x_values), 1.0e-15))),
        float(20.0 * np.log10(max(max(y_values), 1.0e-15))),
    )


def passive_metrics(matrix: np.ndarray) -> dict[str, float]:
    return_loss = -20.0 * np.log10(np.maximum(np.abs(np.diag(matrix)), 1.0e-15))
    off_diagonal = np.abs(matrix - np.diag(np.diag(matrix)))
    x_worst, y_worst = nearest_coupling(matrix)
    return {
        "passive_rl_min_db": float(np.min(return_loss)),
        "passive_rl_median_db": float(np.median(return_loss)),
        "mutual_worst_db": float(20.0 * np.log10(max(float(np.max(off_diagonal)), 1.0e-15))),
        "nearest_x_worst_db": x_worst,
        "nearest_y_worst_db": y_worst,
        "reciprocity_max_abs": float(np.max(np.abs(matrix - matrix.T))),
        "passivity_sigma_max": float(np.linalg.svd(matrix, compute_uv=False).max()),
    }


def optimize_scalar_mode_matches(
    modal_s: np.ndarray, z0: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    series_x: list[float] = []
    shunt_b: list[float] = []
    for mode_index, gamma in enumerate(np.diag(modal_s)):
        impedance = z0 * (1.0 + gamma) / (1.0 - gamma)

        def objective(parameters: np.ndarray) -> float:
            transformed = 1.0 / (1.0 / impedance + 1j * parameters[1]) + 1j * parameters[0]
            reflection = (transformed - z0) / (transformed + z0)
            return float(abs(reflection))

        result = differential_evolution(
            objective,
            bounds=[(-300.0, 300.0), (-0.05, 0.05)],
            seed=seed + mode_index,
            popsize=8,
            maxiter=180,
            tol=1.0e-9,
            polish=True,
        )
        series_x.append(float(result.x[0]))
        shunt_b.append(float(result.x[1]))
    return np.asarray(series_x), np.asarray(shunt_b)


def compose_modal_network(
    antenna_s: np.ndarray,
    transform: np.ndarray,
    series_x: np.ndarray,
    shunt_b: np.ndarray,
    z0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    modal_s = transform.T @ antenna_s @ transform
    composite_modal, antenna_wave_map_modal, network_parameters = compose_nonuniform_network(
        modal_s, series_x, shunt_b, z0
    )
    composite_external = transform @ composite_modal @ transform.T
    antenna_wave_map = transform @ antenna_wave_map_modal @ transform.T
    return composite_external, modal_s, antenna_wave_map, network_parameters


def load_scenarios(dataset_path: Path) -> dict[str, np.ndarray]:
    dataset = np.load(dataset_path, allow_pickle=False)
    element_ixiy = np.asarray(dataset["element_ixiy"], dtype=int)
    selected = np.flatnonzero(
        (element_ixiy[:, 0] >= 6)
        & (element_ixiy[:, 0] <= 9)
        & (element_ixiy[:, 1] >= 6)
        & (element_ixiy[:, 1] <= 9)
    )
    selected = selected[np.lexsort((element_ixiy[selected, 1], element_ixiy[selected, 0]))]
    if selected.size != 16:
        raise RuntimeError(f"Expected 16 central ports, got {selected.size}")
    weight_ri = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float64)[:, selected]
    weights = weight_ri[:, :, 0] + 1j * weight_ri[:, :, 1]
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    weights = weights / np.maximum(norms, 1.0e-15)
    targets = np.asarray(dataset["targets_deg"], dtype=np.float64)
    k_values = np.asarray(dataset["k_values"], dtype=int)
    max_theta = np.asarray(
        [np.max(targets[index, : k_values[index], 0]) for index in range(len(k_values))],
        dtype=np.float64,
    )
    return {
        "weights": weights,
        "masks": np.asarray(dataset["masks"], dtype=bool)[:, selected],
        "k": k_values,
        "ratio": np.asarray(dataset["active_ratios_actual"], dtype=np.float64),
        "sample_index": np.arange(weights.shape[0], dtype=int),
        "max_theta": max_theta,
        "large_scan": max_theta >= 45.0,
        "selected_fullarray_indices": selected,
    }


def case_metrics(
    matrix: np.ndarray,
    weights: np.ndarray,
    masks: np.ndarray,
    threshold_db: float,
    significant_power_relative_db: float,
) -> dict[str, np.ndarray]:
    reflected = weights @ matrix.T
    active = masks & (np.abs(weights) > 1.0e-10)
    max_power = np.maximum(np.max(np.abs(weights) ** 2, axis=1, keepdims=True), 1.0e-30)
    significant_ratio = 10.0 ** (significant_power_relative_db / 10.0)
    significant = active & (np.abs(weights) ** 2 >= significant_ratio * max_power)
    gamma = np.abs(reflected) / np.maximum(np.abs(weights), 1.0e-15)
    worst_active_gamma = np.max(np.where(active, gamma, 0.0), axis=1)
    worst_significant_gamma = np.max(np.where(significant, gamma, 0.0), axis=1)
    incident_power = np.sum(np.abs(weights) ** 2, axis=1)
    reflected_power = np.sum(np.abs(reflected) ** 2, axis=1)
    total_gamma = np.sqrt(reflected_power / np.maximum(incident_power, 1.0e-30))
    rho = 10.0 ** (-threshold_db / 20.0)
    return {
        "worst_active_rl_db": -20.0 * np.log10(np.maximum(worst_active_gamma, 1.0e-15)),
        "worst_significant_rl_db": -20.0 * np.log10(np.maximum(worst_significant_gamma, 1.0e-15)),
        "total_rl_db": -20.0 * np.log10(np.maximum(total_gamma, 1.0e-15)),
        "all_active_pass": worst_active_gamma <= rho,
        "all_significant_pass": worst_significant_gamma <= rho,
        "total_pass": total_gamma <= rho,
        "worst_significant_gamma": worst_significant_gamma,
        "total_gamma": total_gamma,
    }


def stratified_training_indices(scenarios: dict[str, np.ndarray]) -> np.ndarray:
    selected: list[np.ndarray] = []
    for k_value in sorted(set(int(value) for value in scenarios["k"])):
        for ratio in sorted(set(round(float(value), 6) for value in scenarios["ratio"])):
            indices = np.flatnonzero(
                (scenarios["k"] == k_value) & np.isclose(scenarios["ratio"], ratio)
            )
            selected.append(indices[::2][:60])
    return np.unique(np.concatenate(selected))


def joint_refine(
    antenna_s: np.ndarray,
    transform: np.ndarray,
    initial_x: np.ndarray,
    initial_b: np.ndarray,
    z0: float,
    scenarios: dict[str, np.ndarray],
    sweeps: int,
    threshold_db: float,
    coupling_db: float,
    significant_power_relative_db: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    series_x = initial_x.copy()
    shunt_b = initial_b.copy()
    training_indices = stratified_training_indices(scenarios)
    weights = scenarios["weights"][training_indices]
    masks = scenarios["masks"][training_indices]
    rho = 10.0 ** (-threshold_db / 20.0)
    coupling_limit = 10.0 ** (coupling_db / 20.0)

    def objective(x_values: np.ndarray, b_values: np.ndarray) -> float:
        composite, _, _, _ = compose_modal_network(
            antenna_s, transform, x_values, b_values, z0
        )
        evaluated = case_metrics(
            composite, weights, masks, threshold_db, significant_power_relative_db
        )
        worst = evaluated["worst_significant_gamma"]
        total = evaluated["total_gamma"]
        passive = np.abs(np.diag(composite))
        off_diagonal = np.abs(composite - np.diag(np.diag(composite)))
        near_gate_failure = float(np.mean(expit((worst - rho) / 0.025)))
        total_penalty = float(np.mean(np.maximum(total - rho, 0.0) ** 2))
        passive_penalty = float(np.mean(np.maximum(passive - rho, 0.0) ** 2))
        coupling_penalty = float(max(float(np.max(off_diagonal)) - coupling_limit, 0.0) ** 2)
        return (
            near_gate_failure
            + 4.0 * total_penalty
            + 20.0 * passive_penalty
            + 20.0 * coupling_penalty
            + 0.02 * float(np.mean(worst**2))
        )

    history: list[dict[str, Any]] = []
    for sweep in range(sweeps):
        for mode_index in range(16):
            def local_objective(parameters: np.ndarray) -> float:
                candidate_x = series_x.copy()
                candidate_b = shunt_b.copy()
                candidate_x[mode_index] = float(parameters[0])
                candidate_b[mode_index] = float(parameters[1])
                return objective(candidate_x, candidate_b)

            result = minimize(
                local_objective,
                np.asarray([series_x[mode_index], shunt_b[mode_index]]),
                method="Powell",
                bounds=[(-300.0, 300.0), (-0.05, 0.05)],
                options={"maxiter": 70, "xtol": 1.0e-4, "ftol": 1.0e-7},
            )
            series_x[mode_index] = float(result.x[0])
            shunt_b[mode_index] = float(result.x[1])
        history.append(
            {
                "sweep": sweep + 1,
                "objective": objective(series_x, shunt_b),
                "series_x_min_ohm": float(np.min(series_x)),
                "series_x_max_ohm": float(np.max(series_x)),
                "shunt_b_min_siemens": float(np.min(shunt_b)),
                "shunt_b_max_siemens": float(np.max(shunt_b)),
            }
        )
    return series_x, shunt_b, history


def aggregate_case_metrics(
    model: str,
    evaluated: dict[str, np.ndarray],
    scenarios: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    for index in range(len(scenarios["k"])):
        case_rows.append(
            {
                "model": model,
                "sample_index": int(index),
                "k": int(scenarios["k"][index]),
                "ratio": float(scenarios["ratio"][index]),
                "max_theta_deg": float(scenarios["max_theta"][index]),
                "large_scan": int(scenarios["large_scan"][index]),
                "worst_active_rl_db": float(evaluated["worst_active_rl_db"][index]),
                "worst_significant_rl_db": float(evaluated["worst_significant_rl_db"][index]),
                "total_rl_db": float(evaluated["total_rl_db"][index]),
                "all_active_10db_pass": int(evaluated["all_active_pass"][index]),
                "all_significant_10db_pass": int(evaluated["all_significant_pass"][index]),
                "total_10db_pass": int(evaluated["total_pass"][index]),
            }
        )
    group_rows: list[dict[str, Any]] = []
    group_keys: list[tuple[str, np.ndarray]] = [("all", np.ones(len(case_rows), dtype=bool))]
    for k_value in sorted(set(int(value) for value in scenarios["k"])):
        group_keys.append((f"K{k_value}", scenarios["k"] == k_value))
        group_keys.append(
            (
                f"K{k_value}_large_scan",
                (scenarios["k"] == k_value) & scenarios["large_scan"],
            )
        )
        for ratio in sorted(set(round(float(value), 6) for value in scenarios["ratio"])):
            group_keys.append(
                (
                    f"K{k_value}_ratio{ratio:.1f}",
                    (scenarios["k"] == k_value) & np.isclose(scenarios["ratio"], ratio),
                )
            )
    for group_name, selection in group_keys:
        if not np.any(selection):
            continue
        group_rows.append(
            {
                "model": model,
                "group": group_name,
                "case_count": int(np.count_nonzero(selection)),
                "all_active_10db_pass_rate": float(np.mean(evaluated["all_active_pass"][selection])),
                "all_significant_10db_pass_rate": float(
                    np.mean(evaluated["all_significant_pass"][selection])
                ),
                "total_10db_pass_rate": float(np.mean(evaluated["total_pass"][selection])),
                "worst_active_rl_median_db": float(
                    np.median(evaluated["worst_active_rl_db"][selection])
                ),
                "worst_significant_rl_median_db": float(
                    np.median(evaluated["worst_significant_rl_db"][selection])
                ),
                "total_rl_median_db": float(np.median(evaluated["total_rl_db"][selection])),
            }
        )
    return case_rows, group_rows


def write_touchstone(path: Path, matrix: np.ndarray, frequency_hz: float, z0: float) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("! Circuit-cascaded S16; HFSS antenna S16 plus ideal modal feed network.\n")
        for index in range(matrix.shape[0]):
            handle.write(f"! Port[{index + 1}] = P{index:03d}\n")
        handle.write(f"# GHZ S RI R {z0:g}\n")
        values = [f"{frequency_hz / 1.0e9:.12g}"]
        for value in matrix.reshape(-1):
            values.extend((f"{value.real:.12e}", f"{value.imag:.12e}"))
        handle.write(" ".join(values) + "\n")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(args.dataset)
    topologies = ("tiled_2x2_even_odd", "full_4x4_even_odd")
    design_rows: list[dict[str, Any]] = []
    designs: list[dict[str, Any]] = []

    for candidate_index, (candidate_name, path) in enumerate(S16_CANDIDATES.items()):
        parsed = parse_touchstone(path)
        antenna_s = np.asarray(parsed["s_parameters"][0], dtype=np.complex128)
        z0 = float(parsed["reference_impedance_ohm"])
        raw_metrics = passive_metrics(antenna_s)
        for topology_index, topology in enumerate(topologies):
            transform = mode_transform(topology)
            modal_s = transform.T @ antenna_s @ transform
            series_x, shunt_b = optimize_scalar_mode_matches(
                modal_s, z0, int(args.seed) + candidate_index * 100 + topology_index * 20
            )
            composite, modal_s, antenna_wave_map, network_parameters = compose_modal_network(
                antenna_s, transform, series_x, shunt_b, z0
            )
            passive = passive_metrics(composite)
            evaluated = case_metrics(
                composite,
                scenarios["weights"],
                scenarios["masks"],
                float(args.return_loss_db),
                float(args.significant_power_relative_db),
            )
            row = {
                "candidate": candidate_name,
                "topology": topology,
                "stage": "independent_mode_match",
                **{f"raw_{key}": value for key, value in raw_metrics.items()},
                **passive,
                "all_active_10db_pass_rate": float(np.mean(evaluated["all_active_pass"])),
                "all_significant_10db_pass_rate": float(np.mean(evaluated["all_significant_pass"])),
                "total_10db_pass_rate": float(np.mean(evaluated["total_pass"])),
            }
            design_rows.append(row)
            designs.append(
                {
                    "candidate": candidate_name,
                    "path": path,
                    "topology": topology,
                    "antenna_s": antenna_s,
                    "transform": transform,
                    "series_x": series_x,
                    "shunt_b": shunt_b,
                    "composite": composite,
                    "modal_s": modal_s,
                    "antenna_wave_map": antenna_wave_map,
                    "network_parameters": network_parameters,
                    "z0": z0,
                    "frequency_hz": float(parsed["frequency_hz"][0]),
                    "row": row,
                }
            )

    eligible = [
        design
        for design in designs
        if design["row"]["passive_rl_min_db"] >= float(args.return_loss_db)
        and design["row"]["mutual_worst_db"] <= float(args.coupling_db)
        and design["row"]["passivity_sigma_max"] <= 1.0001
    ]
    if not eligible:
        raise RuntimeError("No independent modal design met the passive smoke gate")
    initial_best = max(
        eligible,
        key=lambda item: (
            item["row"]["all_significant_10db_pass_rate"],
            item["row"]["all_active_10db_pass_rate"],
            item["row"]["passive_rl_min_db"],
        ),
    )

    refined_x, refined_b, history = joint_refine(
        initial_best["antenna_s"],
        initial_best["transform"],
        initial_best["series_x"],
        initial_best["shunt_b"],
        initial_best["z0"],
        scenarios,
        int(args.joint_sweeps),
        float(args.return_loss_db),
        float(args.coupling_db),
        float(args.significant_power_relative_db),
    )
    refined_composite, refined_modal, refined_wave_map, refined_parameters = compose_modal_network(
        initial_best["antenna_s"],
        initial_best["transform"],
        refined_x,
        refined_b,
        initial_best["z0"],
    )
    refined_passive = passive_metrics(refined_composite)
    refined_cases = case_metrics(
        refined_composite,
        scenarios["weights"],
        scenarios["masks"],
        float(args.return_loss_db),
        float(args.significant_power_relative_db),
    )
    refined_row = {
        "candidate": initial_best["candidate"],
        "topology": initial_best["topology"],
        "stage": "joint_active_match_coupling_refine",
        **{
            f"raw_{key}": value
            for key, value in passive_metrics(initial_best["antenna_s"]).items()
        },
        **refined_passive,
        "all_active_10db_pass_rate": float(np.mean(refined_cases["all_active_pass"])),
        "all_significant_10db_pass_rate": float(np.mean(refined_cases["all_significant_pass"])),
        "total_10db_pass_rate": float(np.mean(refined_cases["total_pass"])),
    }
    design_rows.append(refined_row)

    initial_cases = case_metrics(
        initial_best["composite"],
        scenarios["weights"],
        scenarios["masks"],
        float(args.return_loss_db),
        float(args.significant_power_relative_db),
    )
    initial_is_feasible = bool(
        initial_best["row"]["passive_rl_min_db"] >= float(args.return_loss_db)
        and initial_best["row"]["mutual_worst_db"] <= float(args.coupling_db)
    )
    refined_is_feasible = bool(
        refined_passive["passive_rl_min_db"] >= float(args.return_loss_db)
        and refined_passive["mutual_worst_db"] <= float(args.coupling_db)
    )
    use_refined = bool(
        refined_is_feasible
        and (
            float(np.mean(refined_cases["all_significant_pass"])),
            float(np.mean(refined_cases["all_active_pass"])),
        )
        >= (
            float(np.mean(initial_cases["all_significant_pass"])),
            float(np.mean(initial_cases["all_active_pass"])),
        )
    )
    if use_refined:
        selected_x, selected_b = refined_x, refined_b
        selected_composite = refined_composite
        selected_modal = refined_modal
        selected_wave_map = refined_wave_map
        selected_parameters = refined_parameters
        selected_cases = refined_cases
        selected_passive = refined_passive
        selected_stage = "joint_active_match_coupling_refine"
    else:
        selected_x, selected_b = initial_best["series_x"], initial_best["shunt_b"]
        selected_composite = initial_best["composite"]
        selected_modal = initial_best["modal_s"]
        selected_wave_map = initial_best["antenna_wave_map"]
        selected_parameters = initial_best["network_parameters"]
        selected_cases = initial_cases
        selected_passive = passive_metrics(selected_composite)
        selected_stage = "independent_mode_match"

    case_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    raw_cases = case_metrics(
        initial_best["antenna_s"],
        scenarios["weights"],
        scenarios["masks"],
        float(args.return_loss_db),
        float(args.significant_power_relative_db),
    )
    for model_name, evaluated in (
        ("raw_s16", raw_cases),
        ("selected_modal_network_s16", selected_cases),
    ):
        cases, groups = aggregate_case_metrics(model_name, evaluated, scenarios)
        case_rows.extend(cases)
        group_rows.extend(groups)

    k_values = sorted(set(int(value) for value in scenarios["k"]))
    nonzero_k_feasible = all(
        bool(
            np.any(
                selected_cases["all_active_pass"]
                & selected_cases["total_pass"]
                & (scenarios["k"] == k_value)
            )
        )
        for k_value in k_values
    )
    nonzero_large_scan_feasible = all(
        bool(
            np.any(
                selected_cases["all_active_pass"]
                & selected_cases["total_pass"]
                & (scenarios["k"] == k_value)
                & scenarios["large_scan"]
            )
        )
        for k_value in k_values
    )
    allow_rebuild = bool(
        selected_passive["passive_rl_min_db"] >= float(args.return_loss_db)
        and selected_passive["mutual_worst_db"] <= float(args.coupling_db)
        and selected_passive["reciprocity_max_abs"] <= 1.0e-6
        and selected_passive["passivity_sigma_max"] <= 1.0001
        and float(np.mean(selected_cases["total_pass"])) >= 0.95
        and nonzero_k_feasible
        and nonzero_large_scan_feasible
    )

    mode_rows: list[dict[str, Any]] = []
    for mode_index in range(16):
        mx, my = divmod(mode_index, 4)
        components = component_description(
            float(selected_x[mode_index]),
            float(selected_b[mode_index]),
            float(initial_best["frequency_hz"]),
        )
        mode_rows.append(
            {
                "mode_index": mode_index,
                "mode_x": mx,
                "mode_y": my,
                "x_mirror_parity": "even" if mx % 2 == 0 else "odd",
                "y_mirror_parity": "even" if my % 2 == 0 else "odd",
                "series_reactance_ohm": float(selected_x[mode_index]),
                "shunt_susceptance_siemens": float(selected_b[mode_index]),
                "series_component": json.dumps(components["series"], separators=(",", ":")),
                "shunt_component": json.dumps(components["shunt"], separators=(",", ":")),
            }
        )

    write_csv(args.out_dir / "design_comparison.csv", design_rows)
    write_csv(args.out_dir / "active_return_case_metrics.csv", case_rows)
    write_csv(args.out_dir / "active_return_group_summary.csv", group_rows)
    write_csv(args.out_dir / "selected_mode_components.csv", mode_rows)
    write_csv(args.out_dir / "joint_optimization_history.csv", history)
    write_touchstone(
        args.out_dir / "selected_modal_network_composite.s16p",
        selected_composite,
        float(initial_best["frequency_hz"]),
        float(initial_best["z0"]),
    )
    np.savez_compressed(
        args.out_dir / "selected_modal_network_s16.npz",
        source_touchstone=np.asarray(str(initial_best["path"])),
        candidate=np.asarray(initial_best["candidate"]),
        topology=np.asarray(initial_best["topology"]),
        selected_stage=np.asarray(selected_stage),
        frequency_hz=np.asarray(initial_best["frequency_hz"]),
        reference_impedance_ohm=np.asarray(initial_best["z0"]),
        antenna_s=initial_best["antenna_s"].astype(np.complex64),
        mode_transform=initial_best["transform"].astype(np.float64),
        modal_antenna_s=selected_modal.astype(np.complex64),
        composite_s=selected_composite.astype(np.complex64),
        antenna_incident_wave_map=selected_wave_map.astype(np.complex64),
        matching_network_parameters=selected_parameters.astype(np.complex64),
        series_reactance_ohm=selected_x,
        shunt_susceptance_siemens=selected_b,
        selected_fullarray_indices=scenarios["selected_fullarray_indices"],
    )
    topology = {
        "name": initial_best["topology"],
        "scope": "one repeated 4x4 tile retaining 16 external PA channels",
        "forward_path": [
            "four separable 4-port DCT even/odd hybrids along y",
            "four separable 4-port DCT even/odd hybrids along x",
            "one L section per 2D mode",
            "inverse separable DCT hybrids to the 16 antenna ports",
        ],
        "full_array_replication": "repeat the same network on each of the sixteen 4x4 tiles",
        "single_frequency_only": True,
        "circuit_model": "ideal reciprocal lossless mode splitters and ideal lumped reactances",
        "not_yet_modeled": [
            "microstrip or stripline substrate",
            "hybrid insertion loss and phase imbalance",
            "lumped component Q and self resonance",
            "feed routing coupling",
        ],
        "mode_transform": initial_best["transform"].tolist(),
    }
    (args.out_dir / "network_topology.json").write_text(
        json.dumps(topology, indent=2), encoding="utf-8"
    )

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_is_real_hfss_s16": True,
        "selected_candidate": initial_best["candidate"],
        "selected_source_touchstone": str(initial_best["path"]),
        "selected_topology": initial_best["topology"],
        "selected_optimization_stage": selected_stage,
        "joint_refine_selected": use_refined,
        "return_loss_gate_db": float(args.return_loss_db),
        "coupling_gate_db": float(args.coupling_db),
        "significant_port_definition_db_relative_power": float(
            args.significant_power_relative_db
        ),
        "selected_passive_metrics": selected_passive,
        "raw_active_10db_pass_rate": float(np.mean(raw_cases["all_active_pass"])),
        "selected_active_10db_pass_rate": float(np.mean(selected_cases["all_active_pass"])),
        "selected_significant_10db_pass_rate": float(
            np.mean(selected_cases["all_significant_pass"])
        ),
        "selected_total_10db_pass_rate": float(np.mean(selected_cases["total_pass"])),
        "nonzero_engineering_feasible_set_for_each_k": nonzero_k_feasible,
        "nonzero_large_scan_feasible_set_for_each_k": nonzero_large_scan_feasible,
        "allow_16x16_rebuild": allow_rebuild,
        "decision": (
            "allow_new_geometry_fullarray_rebuild_and_raw_s256_export"
            if allow_rebuild
            else "block_fullarray_rebuild_due_to_s16_smoke_gate"
        ),
        "interpretation_limit": (
            "The selected S16 is an exact circuit cascade of real HFSS antenna S16 and an ideal "
            "single-frequency modal network. It is not an HFSS full-wave feed-layout result."
        ),
    }
    (args.out_dir / "design_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
