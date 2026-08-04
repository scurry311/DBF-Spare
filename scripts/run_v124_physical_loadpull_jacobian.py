#!/usr/bin/env python3
"""Run the preregistered v1.24 physical S8 load-pull/modal Jacobian audit."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import differential_evolution, lsq_linear

from design_v115_grounded_modal_network import terminate_network
from design_v119_multiport_post_decoupler import active_metrics, reordered_network
from run_v114_small_cell_broadband_feed import load_stimuli, memory_available_gb, parse_touchstone, profile_metrics
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v122_balanced_modal_branch import common_s_metrics, loaded_efficiencies, small_segment_diagnostics
from run_v123_physical_modal_transformer_smoke import (
    builder_text,
    cascade_s,
    component_values,
    network_metrics,
    solver_text,
    touchstone_port_names,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v124_physical_loadpull_jacobian_preregistered.json"
EPS = 1.0e-15


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in (
        "nominal_physical_config",
        "nominal_physical_s8",
        "target_circuit_operator",
        "trusted_antenna_s4",
        "stimulus_csv",
        "stimulus_npz",
    ):
        path = resolve(config["inputs"][key])
        observed = sha256_file(path)
        expected = str(config["inputs"][f"{key}_sha256"]).lower()
        if observed != expected:
            raise RuntimeError(f"Hash mismatch for {key}: {observed} != {expected}")
        rows.append({"role": key, "path": str(path), "size_bytes": path.stat().st_size, "sha256": observed})
    return rows


def variable_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in config["geometry_variables"]}


def apply_geometry(nominal: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    varied = copy.deepcopy(nominal)
    block = varied["single_local_block"]
    block["ground_load_x_mm"] = values["ground_load_x_mm"]
    block["bridge_height_mm"] = values["bridge_height_mm"]
    block["bridge_sheet_width_mm"] = values["bridge_sheet_width_mm"]
    common = values["bridge_common_x_mm"]
    half = values["bridge_half_stagger_mm"]
    block["bridge_x_mm_by_polarity"] = [common - half, common + half]
    return varied


def case_values(config: dict[str, Any], variable_name: str, sign: int) -> dict[str, float]:
    variables = variable_map(config)
    values = {name: float(item["nominal"]) for name, item in variables.items()}
    item = variables[variable_name]
    values[variable_name] += sign * float(item["step"])
    if not float(item["minimum"]) <= values[variable_name] <= float(item["maximum"]):
        raise ValueError(f"Preregistered perturbation is outside bounds for {variable_name}")
    return values


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.24 output: {out}")
    out.mkdir(parents=True)
    input_rows = verify_inputs(config)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag_commit = subprocess.run(
        ["git", "rev-list", "-n", "1", config["parent_tag"]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != config["parent_commit"] or tag_commit != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag_commit}")
    nominal = read_json(resolve(config["inputs"]["nominal_physical_config"]))
    nominal_components = component_values(nominal)
    cases = []
    expected_ports = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    for variable in config["geometry_variables"]:
        name = str(variable["name"])
        for sign, suffix in ((-1, "minus"), (1, "plus")):
            values = case_values(config, name, sign)
            varied = apply_geometry(nominal, values)
            varied_components = component_values(varied)
            for component_name, reference in nominal_components.items():
                if abs(float(varied_components[component_name]) - float(reference)) > 1.0e-14:
                    raise RuntimeError(f"Component drift in {name}/{suffix}: {component_name}")
            folder = out / "cases" / f"{name}_{suffix}"
            folder.mkdir(parents=True)
            project = folder / f"v124_{name}_{suffix}.aedt"
            touchstone = folder / f"v124_{name}_{suffix}.s8p"
            builder = folder / "build.vbs"
            solver = folder / "solve_export.vbs"
            builder.write_text(builder_text(project, touchstone, varied), encoding="ascii")
            solver.write_text(solver_text(project, touchstone), encoding="ascii")
            case = {
                "case_id": f"{name}_{suffix}",
                "variable": name,
                "sign": sign,
                "normalized_coordinate": float(sign),
                "geometry_values": values,
                "fixed_component_values": nominal_components,
                "project_path": str(project.resolve()),
                "touchstone_path": str(touchstone.resolve()),
                "builder_path": str(builder.resolve()),
                "solver_path": str(solver.resolve()),
                "expected_port_order": expected_ports,
            }
            write_json(folder / "case_preregistration.json", case)
            cases.append(case)
    write_json(out / "case_manifest.json", {"cases": cases})
    write_csv(out / "frozen_input_manifest.csv", input_rows)
    write_csv(out / "fixed_component_values.csv", [nominal_components])
    preregistration = {
        **config,
        "runtime_audit": {
            "head_commit": head,
            "tag_commit": tag_commit,
            "free_memory_gib": memory_available_gb(),
            "aedt_processes": aedt_processes(),
        },
        "case_count": len(cases),
        "nominal_component_values": nominal_components,
        "evidence_rules": {
            "central_difference_only": True,
            "component_values_and_q_frozen": True,
            "nominal_s8_reused_without_resolve": True,
            "no_hfss_confirmation_before_reachability_gate": True,
            "no_training_or_array_expansion": True,
        },
    }
    write_json(out / "preregistration.json", preregistration)
    decision = {
        "stage": "A_jacobian_preregistered",
        "allow_build": True,
        "allow_doe_solve": False,
        "allow_confirmation": False,
        "allow_feedpoint_branch": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "case_count": len(cases), "decision": decision}


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def run_builds(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_build"):
        raise RuntimeError("DOE builds are not authorized")
    rows = []
    for case in read_json(out / "case_manifest.json")["cases"]:
        require_no_aedt()
        free = memory_available_gb()
        if free < float(config["resources"]["minimum_free_memory_before_build_gib"]):
            raise MemoryError(f"Only {free:.2f} GiB free before {case['case_id']} build")
        project = Path(case["project_path"])
        if project.exists() and project.stat().st_size > 100:
            rows.append({"case_id": case["case_id"], "return_code": 0, "project_exists": True, "reused": True})
            continue
        log = project.parent / "build.log"
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [str(resolve(config["ansys_executable"])), "-RunScriptAndExit", case["builder_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        rows.append(
            {
                "case_id": case["case_id"],
                "return_code": int(result.returncode),
                "project_exists": project.exists() and project.stat().st_size > 100,
                "reused": False,
            }
        )
        write_csv(out / "build_progress.csv", rows)
    write_csv(out / "build_progress.csv", rows)
    return {"rows": rows}


def audit_builds(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest = read_json(out / "case_manifest.json")["cases"]
    build_rows = {row["case_id"]: row for row in csv.DictReader((out / "build_progress.csv").open(encoding="utf-8"))}
    nominal = read_json(resolve(config["inputs"]["nominal_physical_config"]))
    expected_components = component_values(nominal)
    rows = []
    for case in manifest:
        builder = Path(case["builder_path"]).read_text(encoding="ascii")
        folder = Path(case["project_path"]).parent
        log = (folder / "build.log").read_text(encoding="utf-8", errors="ignore") if (folder / "build.log").exists() else ""
        warnings = sum(
            log.lower().count(pattern)
            for pattern in ("script error", "invalid geometry", "small segment", "too many conductors touch lumped port")
        )
        fixed = all(
            abs(float(case["fixed_component_values"][key]) - float(expected_components[key])) <= 1.0e-14
            for key in expected_components
        )
        row = {
            "case_id": case["case_id"],
            "return_code": int(build_rows[case["case_id"]]["return_code"]),
            "project_valid": Path(case["project_path"]).exists() and Path(case["project_path"]).stat().st_size > 100,
            "differential_port_count": builder.count("AssignDifferentialPort oBoundary"),
            "ground_capacitor_count": len(set(re.findall(r"GroundCap_[NP]_\d+", builder))),
            "bridge_inductor_count": len(set(re.findall(r"BridgeL_\d+_[NP]", builder))),
            "bridge_via_count": len(set(re.findall(r"BridgeVia_\d+_[NP]_\d+", builder))),
            "fixed_component_values_match": fixed,
            "warning_count": warnings,
        }
        row["gate_pass"] = bool(
            row["return_code"] == 0
            and row["project_valid"]
            and row["differential_port_count"] == 8
            and row["ground_capacitor_count"] == 8
            and row["bridge_inductor_count"] == 4
            and row["bridge_via_count"] == 8
            and row["fixed_component_values_match"]
            and warnings == 0
        )
        rows.append(row)
    write_csv(out / "build_audit.csv", rows)
    gate = all(row["gate_pass"] for row in rows)
    decision = {
        "stage": "B_build_audit_complete",
        "build_gate_pass": gate,
        "allow_doe_solve": gate,
        "allow_confirmation": False,
        "allow_feedpoint_branch": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"rows": rows, "decision": decision}


def wait_for_resources(config: dict[str, Any]) -> float:
    deadline = time.time() + float(config["resources"]["memory_recovery_wait_seconds"])
    required = float(config["resources"]["minimum_free_memory_before_solve_gib"])
    while True:
        require_no_aedt()
        free = memory_available_gb()
        if free >= required:
            return free
        if time.time() >= deadline:
            raise MemoryError(f"Memory did not recover to {required:.2f} GiB; current {free:.2f} GiB")
        time.sleep(float(config["resources"]["poll_interval_seconds"]))


def run_doe(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_doe_solve"):
        raise RuntimeError("DOE solve is not authorized")
    rows = []
    progress_path = out / "solve_progress.csv"
    if progress_path.exists():
        rows = list(csv.DictReader(progress_path.open(encoding="utf-8")))
    completed = {str(row["case_id"]) for row in rows if str(row.get("touchstone_exists", "")).lower() == "true"}
    for case in read_json(out / "case_manifest.json")["cases"]:
        if case["case_id"] in completed:
            continue
        free = wait_for_resources(config)
        folder = Path(case["touchstone_path"]).parent
        code, aborted, minimum_free = run_process_with_memory_guard(
            [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", case["solver_path"]],
            folder / "solve_export.log",
            float(config["resources"]["abort_free_memory_during_solve_gib"]),
            float(config["resources"]["poll_interval_seconds"]),
        )
        touchstone = Path(case["touchstone_path"])
        row = {
            "case_id": case["case_id"],
            "variable": case["variable"],
            "sign": case["sign"],
            "return_code": code,
            "memory_aborted": aborted,
            "free_memory_gib_before": free,
            "minimum_free_memory_gib": minimum_free,
            "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
        }
        rows.append(row)
        write_csv(progress_path, rows)
        if code != 0 or aborted or not row["touchstone_exists"]:
            raise RuntimeError(f"DOE solve failed for {case['case_id']}: {row}")
    return {"completed_count": len(rows), "rows": rows}


def complex_vector(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value).ravel()
    return np.concatenate((flat.real, flat.imag))


def supported_reflection(value: np.ndarray) -> np.ndarray:
    selected: list[complex] = []
    pairs = ((0, 2), (2, 0), (3, 1), (1, 3))
    for block in (value[:4, :4], value[4:, 4:]):
        selected.extend(block[index, index] for index in range(4))
        selected.extend(block[first, second] for first, second in pairs)
    return complex_vector(np.asarray(selected))


def bounded_projection(jacobian: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    if float(np.linalg.norm(residual)) <= EPS:
        raise ValueError("Target residual is zero")
    solution = lsq_linear(jacobian, residual, bounds=(-1.0, 1.0), lsmr_tol="auto", max_iter=1000)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    nonzero = singular[singular > max(singular[0] * 1.0e-10, EPS)] if singular.size else np.asarray([])
    rank = int(nonzero.size)
    condition = float(nonzero[0] / nonzero[-1]) if nonzero.size else math.inf
    approximation = jacobian @ solution.x
    fraction = 1.0 - float(np.sum((residual - approximation) ** 2)) / float(np.sum(residual**2))
    unconstrained, *_ = np.linalg.lstsq(jacobian, residual, rcond=1.0e-10)
    unconstrained_approximation = jacobian @ unconstrained
    unconstrained_fraction = 1.0 - float(np.sum((residual - unconstrained_approximation) ** 2)) / float(np.sum(residual**2))
    return {
        "bounded_energy_fraction": max(0.0, min(1.0, fraction)),
        "unbounded_energy_fraction": max(0.0, min(1.0, unconstrained_fraction)),
        "rank": rank,
        "condition_number": condition,
        "bounded_coordinates": solution.x.tolist(),
        "unbounded_coordinates": unconstrained.tolist(),
        "bounded_coordinate_max_abs": float(np.max(np.abs(solution.x))),
        "bounded_solver_success": bool(solution.success),
    }


def predict_metrics(
    network: np.ndarray,
    antenna: np.ndarray,
    sources: np.ndarray,
    considered: np.ndarray,
) -> dict[str, float]:
    metrics = network_metrics(network, antenna, sources, considered)
    metrics["passivity_sigma"] = float(np.max(np.linalg.svd(network, compute_uv=False)))
    metrics["reciprocity_error"] = float(np.max(np.abs(network - network.T)))
    return metrics


def optimize_linear_network(
    nominal: np.ndarray,
    jacobian_s: np.ndarray,
    antenna: np.ndarray,
    sources: np.ndarray,
    considered: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    def evaluate(coordinates: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        network = nominal + np.tensordot(coordinates, jacobian_s, axes=(0, 0))
        return network, predict_metrics(network, antenna, sources, considered)

    def objective(coordinates: np.ndarray) -> float:
        _, metrics = evaluate(coordinates)
        penalties = (
            max(0.0, 11.0 - metrics["active_rl_min_db"]) ** 2
            + 0.5 * max(0.0, 11.0 - metrics["total_rl_min_db"]) ** 2
            + 200.0 * max(0.0, 0.95 - metrics["matched_efficiency_min"]) ** 2
            + 200.0 * max(0.0, 0.95 - metrics["actual_load_insertion_min"]) ** 2
            + 20.0 * max(0.0, metrics["passivity_sigma"] - 1.001) ** 2
        )
        return float(penalties - 0.02 * metrics["active_rl_min_db"] - 0.01 * metrics["total_rl_min_db"])

    result = differential_evolution(
        objective,
        [(-1.0, 1.0)] * jacobian_s.shape[0],
        seed=seed,
        maxiter=180,
        popsize=14,
        tol=1.0e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    network, metrics = evaluate(result.x)
    metrics["objective"] = float(result.fun)
    metrics["iterations"] = int(result.nit)
    metrics["evaluations"] = int(result.nfev)
    return result.x, network, metrics


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "case_manifest.json")["cases"]
    case_by_variable: dict[str, dict[int, dict[str, Any]]] = {}
    matrices: dict[str, np.ndarray] = {}
    numerical_rows = []
    numerical = config["numerical_gates"]
    for case in cases:
        touchstone = Path(case["touchstone_path"])
        if not touchstone.exists():
            raise RuntimeError(f"Missing DOE S8: {touchstone}")
        names = touchstone_port_names(touchstone)
        if set(names) != set(case["expected_port_order"]):
            raise RuntimeError(f"Unexpected port names for {case['case_id']}: {names}")
        frequencies, values = reordered_network(touchstone, case["expected_port_order"], 8)
        index = int(np.argmin(np.abs(frequencies - float(config["frequency_ghz"]))))
        matrix = values[index]
        matrices[case["case_id"]] = matrix
        case_by_variable.setdefault(case["variable"], {})[int(case["sign"])] = case
        profile = profile_metrics(touchstone.parent)
        segments = small_segment_diagnostics(touchstone.parent)
        common = common_s_metrics(np.asarray([matrix]))
        row = {
            "case_id": case["case_id"],
            "variable": case["variable"],
            "sign": case["sign"],
            **profile,
            **segments,
            **common,
        }
        row["gate_pass"] = bool(
            row.get("converged") is True
            and float(row.get("final_delta_s") or math.inf) <= float(numerical["maximum_final_delta_s"])
            and int(row.get("small_mesh_segment_count") or 0) <= int(numerical["maximum_small_mesh_segment_count"])
            and (
                row.get("small_mesh_segment_min_length_mm") is None
                or float(row["small_mesh_segment_min_length_mm"]) >= float(numerical["minimum_small_mesh_segment_length_mm"])
            )
            and row["reciprocity_error"] <= float(numerical["maximum_reciprocity_error"])
            and row["passivity_sigma"] <= float(numerical["maximum_passivity_sigma"])
        )
        numerical_rows.append(row)
    write_csv(out / "numerical_case_metrics.csv", numerical_rows)
    all_numerical_pass = all(row["gate_pass"] for row in numerical_rows)

    nominal_f, nominal_values = reordered_network(
        resolve(config["inputs"]["nominal_physical_s8"]),
        [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)],
        8,
    )
    nominal = nominal_values[int(np.argmin(np.abs(nominal_f - float(config["frequency_ghz"]))))]
    operator = np.load(resolve(config["inputs"]["target_circuit_operator"]), allow_pickle=False)
    target_index = int(np.argmin(np.abs(operator["frequencies_ghz"] - float(config["frequency_ghz"]))))
    target = cascade_s(operator["launch_s8"][target_index], operator["correction_s8"][target_index])
    antenna_f, antenna_values = parse_touchstone(resolve(config["inputs"]["trusted_antenna_s4"]), 4)
    antenna = antenna_values[int(np.argmin(np.abs(antenna_f - float(config["frequency_ghz"]))))]
    stimulus_rows, stimulus_vectors, stimulus_considered = load_stimuli(resolve(config["inputs"]["stimulus_root"]))
    selected = np.asarray(
        [int(row["side"]) == 2 and abs(float(row["frequency_ghz"]) - float(config["frequency_ghz"])) <= 1.0e-9 for row in stimulus_rows]
    )
    sources = stimulus_vectors[selected, :4].T
    considered = stimulus_considered[selected, :4].T
    nominal_external, _, _ = terminate_network(nominal, antenna)
    target_external, _, _ = terminate_network(target, antenna)
    nominal_response = (nominal_external @ sources)[considered]
    target_response = (target_external @ sources)[considered]
    transform = operator["modal_transform"].astype(complex)
    transform8 = np.zeros((8, 8), dtype=complex)
    transform8[:4, :4] = transform
    transform8[4:, 4:] = transform

    variables = [str(item["name"]) for item in config["geometry_variables"]]
    jacobian_s = []
    sensitivity_rows = []
    views: dict[str, tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]] = {
        "supported_reflection": (supported_reflection, supported_reflection(nominal), supported_reflection(target)),
        "load_operator": (
            lambda matrix: complex_vector(terminate_network(matrix, antenna)[0]),
            complex_vector(nominal_external),
            complex_vector(target_external),
        ),
        "active_response": (
            lambda matrix: complex_vector((terminate_network(matrix, antenna)[0] @ sources)[considered]),
            complex_vector(nominal_response),
            complex_vector(target_response),
        ),
        "modal_full_s8": (
            lambda matrix: complex_vector(transform8 @ matrix @ transform8.T),
            complex_vector(transform8 @ nominal @ transform8.T),
            complex_vector(transform8 @ target @ transform8.T),
        ),
        "full_s8": (complex_vector, complex_vector(nominal), complex_vector(target)),
    }
    view_columns: dict[str, list[np.ndarray]] = {name: [] for name in views}
    effective_count = 0
    for variable in variables:
        minus = matrices[case_by_variable[variable][-1]["case_id"]]
        plus = matrices[case_by_variable[variable][1]["case_id"]]
        derivative = (plus - minus) / 2.0
        jacobian_s.append(derivative)
        pair_signal = float(np.max(np.abs(plus - minus)))
        if pair_signal >= float(numerical["minimum_pair_signal_max_abs_delta_s"]):
            effective_count += 1
        full_nonlinearity = float(np.linalg.norm(plus + minus - 2.0 * nominal) / max(np.linalg.norm(plus - minus), EPS))
        row = {
            "variable": variable,
            "pair_signal_max_abs_delta_s": pair_signal,
            "full_s8_central_nonlinearity_ratio": full_nonlinearity,
            "effective": pair_signal >= float(numerical["minimum_pair_signal_max_abs_delta_s"]),
        }
        for name, (extractor, nominal_view, _) in views.items():
            plus_view = extractor(plus)
            minus_view = extractor(minus)
            view_columns[name].append((plus_view - minus_view) / 2.0)
            row[f"{name}_nonlinearity_ratio"] = float(
                np.linalg.norm(plus_view + minus_view - 2.0 * nominal_view)
                / max(np.linalg.norm(plus_view - minus_view), EPS)
            )
        sensitivity_rows.append(row)
    write_csv(out / "geometry_sensitivity.csv", sensitivity_rows)
    jacobian_s_array = np.asarray(jacobian_s)
    projection_rows = []
    projection_payload: dict[str, Any] = {}
    for name, (_, nominal_view, target_view) in views.items():
        jacobian = np.column_stack(view_columns[name])
        residual = target_view - nominal_view
        projection = bounded_projection(jacobian, residual)
        projection_payload[name] = projection
        projection_rows.append({"view": name, **{key: value for key, value in projection.items() if not isinstance(value, list)}})
    write_csv(out / "jacobian_reachability.csv", projection_rows)
    write_json(out / "jacobian_reachability.json", projection_payload)

    best_coordinates, predicted_network, predicted_metrics = optimize_linear_network(
        nominal,
        jacobian_s_array,
        antenna,
        sources,
        considered,
        int(config["seed"]),
    )
    nominal_metrics = predict_metrics(nominal, antenna, sources, considered)
    target_metrics = predict_metrics(target, antenna, sources, considered)
    metric_rows = [
        {"network": "nominal_physical", **nominal_metrics},
        {"network": "target_circuit", **target_metrics},
        {"network": "linear_predicted_best", **predicted_metrics},
    ]
    write_csv(out / "linear_predicted_metrics.csv", metric_rows)
    coordinate_rows = []
    variables_by_name = variable_map(config)
    for index, name in enumerate(variables):
        item = variables_by_name[name]
        coordinate_rows.append(
            {
                "variable": name,
                "normalized_coordinate": float(best_coordinates[index]),
                "predicted_value": float(item["nominal"]) + float(best_coordinates[index]) * float(item["step"]),
                "minimum": item["minimum"],
                "maximum": item["maximum"],
            }
        )
    write_csv(out / "linear_predicted_geometry.csv", coordinate_rows)
    np.savez_compressed(
        out / "jacobian_operators.npz",
        variable_names=np.asarray(variables),
        nominal_s8=nominal,
        target_s8=target,
        jacobian_s8=jacobian_s_array,
        predicted_coordinates=best_coordinates,
        predicted_s8=predicted_network,
    )

    reachability = config["reachability_gates"]
    max_nonlinearity = max(float(row["full_s8_central_nonlinearity_ratio"]) for row in sensitivity_rows)
    condition = float(projection_payload["active_response"]["condition_number"])
    gate_checks = {
        "all_numerical_cases": all_numerical_pass,
        "effective_variable_count": effective_count >= int(numerical["minimum_effective_geometry_variables"]),
        "central_linearity": max_nonlinearity <= float(numerical["maximum_central_nonlinearity_ratio"]),
        "jacobian_condition": condition <= float(numerical["maximum_jacobian_condition_number"]),
        "supported_reflection_reachability": projection_payload["supported_reflection"]["bounded_energy_fraction"]
        >= float(reachability["minimum_supported_reflection_energy_fraction"]),
        "load_operator_reachability": projection_payload["load_operator"]["bounded_energy_fraction"]
        >= float(reachability["minimum_load_operator_energy_fraction"]),
        "active_response_reachability": projection_payload["active_response"]["bounded_energy_fraction"]
        >= float(reachability["minimum_active_response_energy_fraction"]),
        "predicted_active_rl": predicted_metrics["active_rl_min_db"]
        >= float(reachability["minimum_linear_predicted_active_rl_db"]),
        "predicted_total_rl": predicted_metrics["total_rl_min_db"]
        >= float(reachability["minimum_linear_predicted_total_rl_db"]),
        "predicted_matched_efficiency": predicted_metrics["matched_efficiency_min"]
        >= float(reachability["minimum_linear_predicted_matched_efficiency"]),
        "predicted_actual_load_insertion": predicted_metrics["actual_load_insertion_min"]
        >= float(reachability["minimum_linear_predicted_actual_load_insertion"]),
    }
    gate_pass = all(gate_checks.values())
    summary = {
        "case_count": len(cases),
        "numerical_case_pass_count": sum(bool(row["gate_pass"]) for row in numerical_rows),
        "effective_geometry_variable_count": effective_count,
        "maximum_full_s8_central_nonlinearity_ratio": max_nonlinearity,
        "active_response_jacobian_condition_number": condition,
        "projection_energy_fractions": {
            name: float(value["bounded_energy_fraction"]) for name, value in projection_payload.items()
        },
        "nominal_metrics": nominal_metrics,
        "linear_predicted_best_metrics": predicted_metrics,
        "gate_checks": gate_checks,
        "failed_gates": [name for name, passed in gate_checks.items() if not passed],
        "geometry_reachability_gate_pass": gate_pass,
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "C_physical_jacobian_gate_complete",
        "geometry_reachability_gate_pass": gate_pass,
        "allow_one_predicted_geometry_confirmation": gate_pass,
        "stop_current_modal_block": not gate_pass,
        "transition_to_feedpoint_input_impedance": not gate_pass,
        "allow_three_frequency_hfss": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "The physical geometry Jacobian passes; one frozen 10 GHz predicted-geometry confirmation is authorized."
            if gate_pass
            else "The current local block cannot reach the required physical S8/load-pull direction: "
            + ", ".join(summary["failed_gates"])
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest = read_json(out / "case_manifest.json")["cases"] if (out / "case_manifest.json").exists() else []
    completed = sum(Path(case["touchstone_path"]).exists() and Path(case["touchstone_path"]).stat().st_size > 100 for case in manifest)
    return {
        "output_directory": str(out),
        "prepared": (out / "preregistration.json").exists(),
        "case_count": len(manifest),
        "touchstone_count": completed,
        "analysis_complete": (out / "stage_summary.json").exists(),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("prepare", "run-builds", "audit-builds", "run-doe", "analyze", "status"),
        default="status",
    )
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    actions = {
        "prepare": prepare,
        "run-builds": run_builds,
        "audit-builds": audit_builds,
        "run-doe": run_doe,
        "analyze": analyze,
        "status": status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
