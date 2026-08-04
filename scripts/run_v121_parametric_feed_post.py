#!/usr/bin/env python3
"""Run the preregistered v1.21 feed/launch/single-POST physical calibration."""

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
from typing import Any

import numpy as np
from scipy.stats import qmc

from design_v115_grounded_modal_network import terminate_network
from design_v119_multiport_post_decoupler import active_metrics, deembed_load, reordered_network
from design_v120_joint_feed_fanout_sparse_graph import sparse_pi_s8, unpack
from run_v114_small_cell_broadband_feed import load_stimuli, memory_available_gb, parse_touchstone, profile_metrics
from run_v115_physical_modal_feed_fixture import touchstone_port_names
from run_v1191_multiconductor_post_block import phase_align
from v121_shared_cad import (
    INTEGRATED_DESIGN_NAME,
    NETWORK_DESIGN_NAME,
    integrated_builder_text,
    integrated_solver_text,
    network_only_builder_text,
    parameter_map,
    physical_block,
    solver_text,
    validate_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v121_parametric_feed_post_preregistered.json"
WARNING_PATTERN = re.compile(
    r"small[ -]?segment|model validation error|invalid geometry|self[- ]intersection|"
    r"intersecting port|port .* not assigned|must have a selection|script error|fatal error",
    re.IGNORECASE,
)
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(command: list[str]) -> str:
    result = subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def aedt_processes() -> list[dict[str, Any]]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-Process ansysedt,ansysedtsv,HFSS -ErrorAction SilentlyContinue | "
        "Select-Object Id,ProcessName,WorkingSet64 | ConvertTo-Json -Compress",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    return payload if isinstance(payload, list) else [payload]


def output_root(config: dict[str, Any]) -> Path:
    return resolve(config["output_directory"])


def baseline_audit(config: dict[str, Any]) -> dict[str, Any]:
    expected = config["parent_commit"]
    head = git(["rev-parse", "HEAD"])
    tag_commit = git(["rev-list", "-n", "1", config["parent_tag"]])
    status = git(["status", "--porcelain"])
    executable = resolve(config["ansys_executable"])
    status_lines = status.splitlines()
    v121_paths = (
        "configs/v121_",
        "scripts/run_v121_",
        "scripts/v121_",
        config["output_directory"].replace("\\", "/"),
    )
    unrelated = [line for line in status_lines if not any(item in line.replace("\\", "/") for item in v121_paths)]
    result = {
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "expected_parent_commit": expected,
        "head_commit": head,
        "parent_tag": config["parent_tag"],
        "parent_tag_commit": tag_commit,
        "baseline_hash_match": head == expected and tag_commit == expected,
        "worktree_clean_before_v121": not unrelated,
        "worktree_status_at_audit": status_lines,
        "unrelated_worktree_status": unrelated,
        "ansys_executable": str(executable),
        "ansys_executable_exists": executable.exists(),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
    }
    if not result["baseline_hash_match"]:
        raise RuntimeError("HEAD/tag does not match the preregistered v1.20 baseline")
    if not executable.exists():
        raise FileNotFoundError(executable)
    return result


def parameter_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    observable = set(config["doe"]["network_observable_variables"])
    return [
        {
            "name": item["name"],
            "nominal": item["nominal"],
            "minimum": item["minimum"],
            "maximum": item["maximum"],
            "scope": item["scope"],
            "network_only_doe": item["name"] in observable,
            "manufacturing_range_frozen": True,
        }
        for item in config["variables"]
    ]


def lhs_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    variables = {item["name"]: item for item in config["variables"]}
    names = list(config["doe"]["network_observable_variables"])
    count = int(config["doe"]["network_only_sample_count"])
    nominal = parameter_map(config)
    sampler = qmc.LatinHypercube(d=len(names), seed=int(config["doe"]["seed"]), optimization="random-cd")
    unit = sampler.random(n=count - 1)
    lower = np.asarray([float(variables[name]["minimum"]) for name in names])
    upper = np.asarray([float(variables[name]["maximum"]) for name in names])
    scaled = qmc.scale(unit, lower, upper)
    rows: list[dict[str, Any]] = [{"candidate_id": "doe00_nominal", **nominal}]
    for index, vector in enumerate(scaled, start=1):
        values = dict(nominal)
        values.update({name: float(value) for name, value in zip(names, vector)})
        validate_parameters(config, values)
        # Project only the common POST correction onto the manufacturable
        # route set. The original LHS coordinates remain otherwise unchanged.
        for _ in range(201):
            try:
                integrated_builder_text(ROOT / "v121_route_feasibility_probe.aedt", config, values)
                break
            except ValueError as error:
                if "route target" not in str(error):
                    raise
                values["common_post_length_delta_mm"] += 0.01
                maximum = next(float(item["maximum"]) for item in config["variables"] if item["name"] == "common_post_length_delta_mm")
                if values["common_post_length_delta_mm"] > maximum + EPS:
                    raise RuntimeError("No route-feasible POST-length projection exists") from error
        else:
            raise RuntimeError("POST-length route projection did not converge")
        rows.append({"candidate_id": f"doe{index:02d}_lhs", **values})
    return rows


def apply_route_amendment(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    amendment_path = out / "preregistration_amendment01.json"
    if amendment_path.exists():
        raise FileExistsError(f"Refusing to overwrite preregistration amendment: {amendment_path}")
    source_hashes = read_json(out / "source_hashes.json")
    config_path = DEFAULT_CONFIG
    solved = list((out / "doe_10ghz_network_s8").glob("**/*.s8p")) if (out / "doe_10ghz_network_s8").exists() else []
    if solved:
        raise RuntimeError("Cannot amend the DOE after an S8 result exists")
    old_rows = list(csv.DictReader((out / "doe_candidates.csv").open(encoding="utf-8")))
    new_rows = lhs_candidates(config)
    changes = []
    old_by_id = {row["candidate_id"]: row for row in old_rows}
    for row in new_rows:
        old = old_by_id[row["candidate_id"]]
        for item in config["variables"]:
            name = item["name"]
            before = float(old[name])
            after = float(row[name])
            if abs(before - after) > 1.0e-12:
                changes.append({"candidate_id": row["candidate_id"], "parameter": name, "before": before, "after": after})
    amendment = {
        "amendment": "v1.21-preregistration-amendment-01-route-feasibility",
        "created_on": "2026-08-03",
        "timing": "after CAD-only construction rejection and before any S-parameter solve",
        "original_protocol_sha256": source_hashes["protocol_config_sha256"],
        "effective_protocol_sha256": sha256(config_path),
        "added_constraint": "route target length must not be shorter than the physical Manhattan base route",
        "performance_results_observed_before_amendment": False,
        "engineering_thresholds_changed": False,
        "parameter_ranges_changed": False,
        "doe_projection_changes": changes,
    }
    write_json(amendment_path, amendment)
    write_json(out / "config_effective_after_amendment01.json", config)
    write_csv(out / "doe_candidates_effective.csv", new_rows)
    return amendment


def apply_solver_amendment(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    amendment_path = out / "preregistration_amendment02.json"
    if amendment_path.exists():
        raise FileExistsError(f"Refusing to overwrite solver amendment: {amendment_path}")
    solved = list(out.glob("doe_10ghz_network_s8/**/*.s8p"))
    if solved:
        raise RuntimeError("Cannot amend the DOE solver after an S8 result exists")
    amendment = {
        "amendment": "v1.21-preregistration-amendment-02-iterative-resource-smoke",
        "created_on": "2026-08-03",
        "timing": "after one direct-solver memory abort and before any S-parameter result",
        "effective_protocol_sha256": sha256(DEFAULT_CONFIG),
        "physical_geometry_changed": False,
        "mesh_definition_changed": False,
        "engineering_thresholds_changed": False,
        "performance_results_observed_before_amendment": False,
        "failed_direct_resource_probe": {
            "candidate_id": "doe00_nominal",
            "initial_mesh_tetrahedra": 356315,
            "matrix_size": 1885474,
            "hfss_estimated_total_memory_gib": 14.55,
            "s8_exported": False,
            "stop_reason": "the preregistered 3 GiB free-memory abort guard was reached",
        },
        "numerical_change": {
            "from_solver": "Auto/Direct Solver",
            "to_solver": "Iterative Solver",
            "iterative_residual": float(config["doe"]["iterative_residual"]),
            "scope": "10 GHz network-only DOE; final Pareto candidates still require independent direct/DDM validation",
        },
    }
    write_json(amendment_path, amendment)
    write_json(out / "config_effective_after_amendment02.json", config)
    return amendment


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.21 output: {out}")
    out.mkdir(parents=True)
    audit = baseline_audit(config)
    config_path = DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else resolve("configs/v121_parametric_feed_post_preregistered.json")
    prereg = copy.deepcopy(config)
    prereg["runtime_baseline_audit"] = audit
    prereg["evidence_rules"] = {
        "network_only_s8_is_not_integrated_hfss": True,
        "integrated_s4_is_not_16x16_evidence": True,
        "circuit_or_cascade_metrics_are_not_hfss_metrics": True,
        "no_gate_or_parameter_changes_after_three_frequency_freeze": True,
    }
    write_json(out / "preregistration.json", prereg)
    write_json(out / "config_snapshot.json", config)
    write_json(out / "baseline_audit.json", audit)
    write_csv(out / "cad_parameter_table.csv", parameter_rows(config))
    write_csv(out / "doe_candidates.csv", lhs_candidates(config))
    initial = {
        "stage": "A_preregistered",
        "stage_a_pass": True,
        "allow_build_smoke": True,
        "allow_10ghz_network_doe": False,
        "allow_three_frequency_optimization": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": "Build smoke and memory gate are not yet complete.",
    }
    write_json(out / "stage_decision.json", initial)
    manifest = {
        "protocol_config": str(config_path.resolve()),
        "protocol_config_sha256": sha256(config_path),
        "v120_synthesis_sha256": sha256(resolve(config["synthesis_summary"])),
        "trusted_antenna_protocol_sha256": sha256(resolve(config["trusted_antenna_protocol"])),
    }
    write_json(out / "source_hashes.json", manifest)
    return {"output_directory": str(out), "audit": audit, "stage_decision": initial}


def smoke_parameter_sets(config: dict[str, Any]) -> list[dict[str, Any]]:
    nominal = parameter_map(config)
    low = {item["name"]: float(item["minimum"]) for item in config["variables"]}
    high = {item["name"]: float(item["maximum"]) for item in config["variables"]}
    # The simultaneous all-lower corner has a POST route shorter than its
    # Manhattan path. Preserve the lower geometry stress while using the
    # preregistered upper POST-length bound to keep the CAD manufacturable.
    low["common_post_length_delta_mm"] = high["common_post_length_delta_mm"]
    for values in (low, high):
        validate_parameters(config, values)
    return [
        {"candidate_id": "nominal", **nominal},
        {"candidate_id": "lower_safe", **low},
        {"candidate_id": "upper_bound", **high},
    ]


def prepare_case(
    config: dict[str, Any],
    candidate: dict[str, Any],
    model: str,
    folder: Path,
    frequencies: list[float],
    solver_type: str = "auto",
) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=False)
    values = parameter_map(config, {key: value for key, value in candidate.items() if key != "candidate_id"})
    if model == "network_s8":
        project = folder / "v121_network_only_s8.aedt"
        touchstone = folder / "v121_network_only_s8.s8p"
        builder = network_only_builder_text(project, config, values, frequencies, solver_type=solver_type)
        design_name = NETWORK_DESIGN_NAME
        expected_ports = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    elif model == "integrated_2x2":
        project = folder / "v121_integrated_2x2.aedt"
        touchstone = folder / "v121_integrated_2x2.s4p"
        builder = integrated_builder_text(project, config, values)
        design_name = INTEGRATED_DESIGN_NAME
        expected_ports = [f"PRE_{index}" for index in range(4)]
    else:
        raise ValueError(model)
    build = folder / "build.vbs"
    solve = folder / "solve_export.vbs"
    build.write_text(builder, encoding="ascii")
    solve_text = (
        integrated_solver_text(project, touchstone, folder, frequencies)
        if model == "integrated_2x2"
        else solver_text(project, touchstone, design_name)
    )
    solve.write_text(solve_text, encoding="ascii")
    manifest = {
        "candidate_id": candidate["candidate_id"],
        "model": model,
        "parameters": values,
        "frequencies_ghz": frequencies,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(build.resolve()),
        "solver_path": str(solve.resolve()),
        "design_name": design_name,
        "solver_type": solver_type,
        "expected_port_order": expected_ports,
        "fixed_graph": config["fixed_topology"]["manufacturable_graph_pairs"],
        "evidence_scope": "HFSS build definition only; no solved metric until a valid Touchstone export exists",
    }
    write_json(folder / "case_manifest.json", manifest)
    return manifest


def prepare_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    if not (out / "preregistration.json").exists():
        raise RuntimeError("Run preregister first")
    root = out / "build_smoke"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite build smoke: {root}")
    manifests = []
    for candidate in smoke_parameter_sets(config):
        for model in ("network_s8", "integrated_2x2"):
            manifests.append(
                prepare_case(config, candidate, model, root / candidate["candidate_id"] / model, [10.0])
            )
    write_json(root / "smoke_manifest.json", {"cases": manifests})
    return {"prepared_cases": len(manifests), "root": str(root)}


def run_process_with_memory_guard(
    command: list[str],
    log_path: Path,
    abort_gib: float | None,
    poll_seconds: float,
) -> tuple[int, bool, float]:
    minimum = math.inf
    low_checks = 0
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        while process.poll() is None:
            free = memory_available_gb()
            if math.isfinite(free):
                minimum = min(minimum, free)
                if abort_gib is not None:
                    low_checks = low_checks + 1 if free < abort_gib else 0
            if abort_gib is not None and low_checks >= 3:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                process.wait(timeout=30)
                return 99, True, minimum
            time.sleep(poll_seconds)
    return int(process.returncode), False, minimum


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def run_build_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    root = out / "build_smoke"
    smoke = read_json(root / "smoke_manifest.json")
    require_no_aedt()
    free = memory_available_gb()
    minimum = float(config["resources"]["minimum_free_memory_before_build_smoke_gib"])
    if math.isfinite(free) and free < minimum:
        raise MemoryError(f"Only {free:.2f} GiB free; build smoke requires {minimum:.2f} GiB")
    rows = []
    executable = str(resolve(config["ansys_executable"]))
    for manifest in smoke["cases"]:
        require_no_aedt()
        folder = Path(manifest["builder_path"]).parent
        started = time.time()
        code, aborted, minimum_observed = run_process_with_memory_guard(
            [executable, "-RunScriptAndExit", manifest["builder_path"]],
            folder / "build.log",
            None,
            float(config["resources"]["poll_interval_seconds"]),
        )
        project = Path(manifest["project_path"])
        rows.append({
            "candidate_id": manifest["candidate_id"],
            "model": manifest["model"],
            "return_code": code,
            "memory_aborted": aborted,
            "minimum_free_memory_gib": None if not math.isfinite(minimum_observed) else minimum_observed,
            "elapsed_seconds": time.time() - started,
            "project_exists": project.exists(),
            "project_bytes": project.stat().st_size if project.exists() else 0,
        })
    write_csv(root / "build_execution.csv", rows)
    return {"cases": rows}


def audit_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    root = out / "build_smoke"
    smoke = read_json(root / "smoke_manifest.json")
    executions = {
        (row["candidate_id"], row["model"]): row
        for row in list(csv.DictReader((root / "build_execution.csv").open(encoding="utf-8")))
    }
    expected_graph = [[0, 2], [2, 3], [3, 1]]
    rows = []
    for manifest in smoke["cases"]:
        folder = Path(manifest["builder_path"]).parent
        builder = Path(manifest["builder_path"]).read_text(encoding="ascii")
        log = (folder / "build.log").read_text(encoding="utf-8", errors="replace") if (folder / "build.log").exists() else ""
        execution = executions.get((manifest["candidate_id"], manifest["model"]), {})
        assigned_ports = re.findall(r'^AssignPort oBoundary, "(PRE|POST)_([0-3])"', builder, flags=re.MULTILINE)
        port_names = [f"{prefix}_{index}" for prefix, index in assigned_ports]
        expected_ports = manifest["expected_port_order"]
        graph_sheets = len(re.findall(r'^CreateSheet oEditor, "(?:Input|Output)GraphSheet_[0-2]"', builder, flags=re.MULTILINE))
        warning_matches = sorted(set(match.group(0) for match in WARNING_PATTERN.finditer(log)))
        project = Path(manifest["project_path"])
        row = {
            "candidate_id": manifest["candidate_id"],
            "model": manifest["model"],
            "return_code": int(execution.get("return_code", -1)),
            "project_valid": project.exists() and project.stat().st_size > 100,
            "port_set_valid": set(port_names) == set(expected_ports) and len(port_names) == len(expected_ports),
            "graph_valid": manifest["fixed_graph"] == expected_graph and graph_sheets == 6,
            "single_stage_valid": builder.count("SeriesSheet_") >= 4 and "SecondStage" not in builder,
            "small_segment_or_geometry_warnings": json.dumps(warning_matches),
            "warning_free": not warning_matches,
        }
        row["build_smoke_pass"] = bool(
            row["return_code"] == 0
            and row["project_valid"]
            and row["port_set_valid"]
            and row["graph_valid"]
            and row["single_stage_valid"]
            and row["warning_free"]
        )
        rows.append(row)
    write_csv(root / "build_smoke_audit.csv", rows)
    passed = all(row["build_smoke_pass"] for row in rows)
    free = memory_available_gb()
    memory_pass = not math.isfinite(free) or free >= float(config["resources"]["minimum_free_memory_before_hfss_gib"])
    decision = {
        "stage": "B_build_smoke_complete",
        "build_smoke_pass": passed,
        "build_smoke_pass_count": sum(row["build_smoke_pass"] for row in rows),
        "build_smoke_case_count": len(rows),
        "free_memory_gib": free,
        "formal_hfss_memory_gate_pass": memory_pass,
        "allow_10ghz_network_doe": passed and memory_pass,
        "allow_three_frequency_optimization": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": "10 GHz DOE is authorized only when all six build cases pass and at least 13 GiB is free.",
    }
    write_json(out / "stage_decision.json", decision)
    write_json(root / "build_smoke_summary.json", {"rows": rows, "decision": decision})
    return {"rows": rows, "decision": decision}


def prepare_doe(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    decision = read_json(out / "stage_decision.json")
    if not decision.get("build_smoke_pass"):
        raise RuntimeError("Build smoke did not pass")
    root = out / "doe_10ghz_network_s8"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite DOE: {root}")
    candidate_file = out / "doe_candidates_effective.csv"
    if not candidate_file.exists():
        candidate_file = out / "doe_candidates.csv"
    rows = list(csv.DictReader(candidate_file.open(encoding="utf-8")))
    manifests = []
    for row in rows:
        candidate = {"candidate_id": row["candidate_id"]}
        candidate.update({item["name"]: float(row[item["name"]]) for item in config["variables"]})
        manifests.append(
            prepare_case(
                config,
                candidate,
                "network_s8",
                root / row["candidate_id"],
                [10.0],
                solver_type=str(config["doe"].get("network_solver_type", "auto")),
            )
        )
    write_json(root / "doe_manifest.json", {"cases": manifests})
    return {"prepared_cases": len(manifests), "root": str(root)}


def refresh_resource_gate(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    summary_path = out / "build_smoke" / "build_smoke_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Build-smoke audit is missing")
    build_pass = bool(read_json(summary_path)["decision"]["build_smoke_pass"])
    require_no_aedt()
    free = memory_available_gb()
    minimum = float(config["resources"]["minimum_free_memory_before_hfss_gib"])
    memory_pass = not math.isfinite(free) or free >= minimum
    timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    check = {
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_smoke_pass": build_pass,
        "free_memory_gib": free,
        "minimum_free_memory_before_hfss_gib": minimum,
        "formal_hfss_memory_gate_pass": memory_pass,
        "aedt_processes": [],
        "allow_10ghz_network_doe": build_pass and memory_pass,
    }
    write_json(out / "resource_gate_checks" / f"resource_gate_{timestamp}.json", check)
    decision = {
        "stage": "B_resource_gate_refreshed",
        **check,
        "allow_three_frequency_optimization": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": "Only the resource authorization is refreshed; parameters and engineering gates remain frozen.",
    }
    write_json(out / "stage_decision.json", decision)
    return decision


def run_doe(config: dict[str, Any], maximum_new_cases: int | None = None) -> dict[str, Any]:
    out = output_root(config)
    decision = refresh_resource_gate(config)
    if not decision.get("allow_10ghz_network_doe"):
        raise RuntimeError("Formal DOE is locked by the build-smoke or memory gate")
    require_no_aedt()
    root = out / "doe_10ghz_network_s8"
    cases = read_json(root / "doe_manifest.json")["cases"]
    executable = str(resolve(config["ansys_executable"]))
    minimum_start = float(config["resources"]["minimum_free_memory_before_hfss_gib"])
    abort = float(config["resources"]["abort_free_memory_during_solve_gib"])
    poll = float(config["resources"]["poll_interval_seconds"])
    rows = []
    new_cases = 0
    for manifest in cases:
        touchstone = Path(manifest["touchstone_path"])
        if touchstone.exists() and touchstone.stat().st_size > 100:
            rows.append({"candidate_id": manifest["candidate_id"], "status": "already_complete"})
            continue
        if maximum_new_cases is not None and new_cases >= maximum_new_cases:
            break
        require_no_aedt()
        free = memory_available_gb()
        if math.isfinite(free) and free < minimum_start:
            rows.append({"candidate_id": manifest["candidate_id"], "status": "stopped_before_start_low_memory", "free_memory_gib": free})
            break
        folder = Path(manifest["builder_path"]).parent
        new_cases += 1
        build_code, _, build_min = run_process_with_memory_guard([executable, "-RunScriptAndExit", manifest["builder_path"]], folder / "build.log", abort, poll)
        solve_code = None
        solve_aborted = False
        solve_min = math.inf
        if build_code == 0:
            solve_code, solve_aborted, solve_min = run_process_with_memory_guard([executable, "-ng", "-RunScriptAndExit", manifest["solver_path"]], folder / "solve_export.log", abort, poll)
        rows.append({
            "candidate_id": manifest["candidate_id"],
            "status": "complete" if touchstone.exists() and touchstone.stat().st_size > 100 else "failed",
            "build_return_code": build_code,
            "solve_return_code": solve_code,
            "solve_memory_aborted": solve_aborted,
            "minimum_free_memory_gib": min(build_min, solve_min),
        })
        write_csv(root / "doe_execution.csv", rows)
        if solve_aborted:
            break
    return {"cases": rows}


def run_doe_smoke(config: dict[str, Any]) -> dict[str, Any]:
    return run_doe(config, maximum_new_cases=1)


def _minimum_passive_rl(matrix: np.ndarray) -> float:
    return float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matrix)), EPS))))


def analyze_network_case(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    touchstone = Path(manifest["touchstone_path"])
    names = touchstone_port_names(touchstone)
    desired_names = manifest["expected_port_order"]
    if set(names) != set(desired_names):
        raise RuntimeError(f"Unexpected ports in {touchstone}: {names}")
    frequencies, physical = reordered_network(touchstone, desired_names, 8)
    index = int(np.argmin(np.abs(frequencies - 10.0)))
    physical_matrix = physical[index]
    synthesis_config = read_json(resolve(config["synthesis_config"]))
    synthesis = read_json(resolve(config["synthesis_summary"]))
    graph = [tuple(int(value) for value in pair) for pair in config["fixed_topology"]["manufacturable_graph_pairs"]]
    _, series_ground, series_pair, input_ground, input_pair, output_ground, output_pair = unpack(np.asarray(synthesis["optimized_parameters"], dtype=float))
    target = sparse_pi_s8(float(frequencies[index]), series_ground, series_pair, input_ground, input_pair, output_ground, output_pair, graph, synthesis_config)
    aligned, phases, target_delta = phase_align(physical_matrix, target)
    feed_f, feed = reordered_network(resolve(synthesis_config["validated_feed_s8"]), desired_names, 8)
    integrated_f, integrated = reordered_network(resolve(synthesis_config["integrated_v118_s4"]), [f"PRE_{item}" for item in range(4)], 4)
    antenna_f, antenna = parse_touchstone(resolve(synthesis_config["trusted_antenna_s4"]), 4)
    feed_index = int(np.argmin(np.abs(feed_f - frequencies[index])))
    integrated_index = int(np.argmin(np.abs(integrated_f - frequencies[index])))
    antenna_index = int(np.argmin(np.abs(antenna_f - frequencies[index])))
    effective_load = deembed_load(integrated[integrated_index], feed[feed_index])
    desired = terminate_network(feed[feed_index], antenna[antenna_index])[0]
    post = terminate_network(aligned, effective_load)[0]
    corrected = terminate_network(feed[feed_index], post)[0]
    stimulus_rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    selected = np.asarray([
        int(row["side"]) == 2 and abs(float(row["frequency_ghz"]) - float(frequencies[index])) <= 1.0e-9
        for row in stimulus_rows
    ])
    active_rl, total_rl = active_metrics(corrected, vectors[selected, :4].T, considered[selected, :4].T)
    matched_s, load_incident, load_reflected = terminate_network(aligned, np.zeros((4, 4), dtype=complex))
    accepted = 1.0 - np.sum(np.abs(matched_s) ** 2, axis=0)
    delivered = np.sum(np.abs(load_incident) ** 2, axis=0) - np.sum(np.abs(load_reflected) ** 2, axis=0)
    profile = profile_metrics(touchstone.parent)
    return {
        "candidate_id": manifest["candidate_id"],
        "frequency_ghz": float(frequencies[index]),
        "final_delta_s": profile.get("final_delta_s"),
        "converged": profile.get("converged"),
        "tetrahedra": profile.get("tetrahedra"),
        "peak_memory_gib": profile.get("peak_memory_gb"),
        "reciprocity_error": float(np.max(np.abs(physical_matrix - physical_matrix.T))),
        "passivity_sigma": float(np.max(np.linalg.svd(physical_matrix, compute_uv=False))),
        "passive_rl_min_db": _minimum_passive_rl(aligned),
        "active_rl_min_db": active_rl,
        "total_rl_min_db": total_rl,
        "network_efficiency_min": float(np.min(delivered / np.maximum(accepted, EPS))),
        "physical_vs_target_s8_max_abs_delta": target_delta,
        "corrected_vs_reference_max_abs_delta_s": float(np.max(np.abs(corrected - desired))),
        "frozen_excitation_count": int(np.sum(selected)),
        "reference_phase_deg": json.dumps(phases),
        **manifest["parameters"],
    }


def pareto_flags(rows: list[dict[str, Any]]) -> list[bool]:
    objectives = np.asarray([
        [
            -float(row["active_rl_min_db"]),
            -float(row["total_rl_min_db"]),
            -float(row["passive_rl_min_db"]),
            -float(row["network_efficiency_min"]),
            float(row["physical_vs_target_s8_max_abs_delta"]),
        ]
        for row in rows
    ])
    flags = []
    for index, point in enumerate(objectives):
        dominated = any(np.all(other <= point) and np.any(other < point) for other_index, other in enumerate(objectives) if other_index != index)
        flags.append(not dominated)
    return flags


def sensitivity_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = list(config["doe"]["network_observable_variables"])
    metrics = [
        "active_rl_min_db",
        "total_rl_min_db",
        "passive_rl_min_db",
        "network_efficiency_min",
        "physical_vs_target_s8_max_abs_delta",
    ]
    matrix = np.asarray([[float(row[name]) for name in names] for row in rows])
    center = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), EPS)
    normalized = (matrix - center) / scale
    design = np.column_stack([np.ones(len(rows)), normalized])
    output = []
    for metric in metrics:
        target = np.asarray([float(row[metric]) for row in rows])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0][1:]
        for name, coefficient in zip(names, coefficients):
            output.append({"metric": metric, "parameter": name, "standardized_jacobian": float(coefficient), "absolute_rank_basis": abs(float(coefficient))})
    return output


def network_gate_pass(config: dict[str, Any], row: dict[str, Any]) -> bool:
    gates = config["gates"]
    return bool(
        row["converged"] is True
        and float(row["final_delta_s"] or math.inf) <= float(gates["maximum_final_delta_s"])
        and row["reciprocity_error"] <= float(gates["maximum_reciprocity_error"])
        and row["passivity_sigma"] <= float(gates["maximum_passivity_sigma"])
        and row["network_efficiency_min"] >= float(gates["minimum_network_efficiency"])
        and row["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
        and row["active_rl_min_db"] >= float(gates["minimum_active_rl_db"])
        and row["total_rl_min_db"] >= float(gates["minimum_total_rl_db"])
        and row["physical_vs_target_s8_max_abs_delta"]
        <= float(gates["maximum_physical_vs_target_s8_abs_delta"])
    )


def analyze_doe(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    root = out / "doe_10ghz_network_s8"
    manifests = read_json(root / "doe_manifest.json")["cases"]
    complete = [item for item in manifests if Path(item["touchstone_path"]).exists() and Path(item["touchstone_path"]).stat().st_size > 100]
    if len(complete) < 12:
        raise RuntimeError(f"Only {len(complete)} complete DOE cases; at least 12 are required")
    rows = [analyze_network_case(config, manifest) for manifest in complete]
    flags = pareto_flags(rows)
    for row, flag in zip(rows, flags):
        row["pareto_nondominated"] = flag
        row["network_gate_pass_10ghz"] = network_gate_pass(config, row)
    write_csv(root / "doe_physical_metrics.csv", rows)
    write_csv(root / "physical_sensitivity_jacobian.csv", sensitivity_rows(config, rows))
    pareto = sorted(
        [row for row in rows if row["pareto_nondominated"]],
        key=lambda row: (
            not row["network_gate_pass_10ghz"],
            row["physical_vs_target_s8_max_abs_delta"],
            -row["active_rl_min_db"],
        ),
    )
    write_csv(root / "pareto_candidates.csv", pareto)
    pass_count = sum(row["network_gate_pass_10ghz"] for row in rows)
    decision = {
        "stage": "C_10ghz_physical_sensitivity_complete",
        "complete_case_count": len(rows),
        "network_gate_pass_count": pass_count,
        "pareto_candidate_count": len(pareto),
        "allow_three_frequency_optimization": pass_count > 0 and len(pareto) >= 3,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"decision": decision, "best_rows": pareto[:5]}


def local_refinement_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    nominal = parameter_map(config)
    definitions = [
        (
            "local16_gradient_corner",
            {
                "common_trace_width_mm": 0.45,
                "outer_edge_gap_mm": 0.16,
                "center_edge_gap_mm": 0.35,
                "shunt_reference_offset_mm": 0.25,
                "common_post_length_delta_mm": 1.0,
            },
            "Jacobian corner maximizing predicted active, total and passive RL while reducing target-S8 error.",
        ),
        (
            "local17_efficiency_balance",
            {
                "common_trace_width_mm": 0.45,
                "outer_edge_gap_mm": 0.16,
                "center_edge_gap_mm": 0.22,
                "shunt_reference_offset_mm": 0.25,
                "common_post_length_delta_mm": 1.0,
            },
            "Same high-RL corner with nominal center gap retained to reduce the observed efficiency tradeoff.",
        ),
        (
            "local18_doe09_shunt_fix",
            {
                "common_trace_width_mm": 0.50,
                "outer_edge_gap_mm": 0.145,
                "center_edge_gap_mm": 0.32,
                "shunt_reference_offset_mm": 0.30,
                "common_post_length_delta_mm": 0.95,
            },
            "Local refinement of the best-active-RL DOE09 point, primarily correcting its adverse shunt offset.",
        ),
        (
            "local19_conservative_knee",
            {
                "common_trace_width_mm": 0.48,
                "outer_edge_gap_mm": 0.145,
                "center_edge_gap_mm": 0.30,
                "shunt_reference_offset_mm": 0.40,
                "common_post_length_delta_mm": 0.80,
            },
            "Conservative Pareto-knee interpolation that avoids placing every variable on a bound.",
        ),
    ]
    candidates = []
    for candidate_id, updates, rationale in definitions:
        values = dict(nominal)
        values.update(updates)
        validate_parameters(config, values)
        integrated_builder_text(ROOT / "v121_local_route_feasibility_probe.aedt", config, values)
        candidates.append({"candidate_id": candidate_id, **values, "selection_rationale": rationale})
    return candidates


def linear_prediction_rows(
    config: dict[str, Any],
    observed: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = list(config["doe"]["network_observable_variables"])
    metrics = [
        "active_rl_min_db",
        "total_rl_min_db",
        "passive_rl_min_db",
        "network_efficiency_min",
        "physical_vs_target_s8_max_abs_delta",
    ]
    matrix = np.asarray([[float(row[name]) for name in names] for row in observed])
    center = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), EPS)
    design = np.column_stack([np.ones(len(observed)), (matrix - center) / scale])
    candidate_matrix = np.asarray([[float(row[name]) for name in names] for row in candidates])
    candidate_design = np.column_stack([np.ones(len(candidates)), (candidate_matrix - center) / scale])
    predictions: dict[str, np.ndarray] = {}
    for metric in metrics:
        target = np.asarray([float(row[metric]) for row in observed])
        predictions[metric] = candidate_design @ np.linalg.lstsq(design, target, rcond=None)[0]
    rows = []
    for index, candidate in enumerate(candidates):
        row = dict(candidate)
        for metric in metrics:
            row[f"linear_predicted_{metric}"] = float(predictions[metric][index])
        rows.append(row)
    return rows


def prepare_local_refinement(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    doe_root = out / "doe_10ghz_network_s8"
    metrics_path = doe_root / "doe_physical_metrics.csv"
    decision = read_json(out / "stage_decision.json")
    if decision.get("stage") != "C_10ghz_physical_sensitivity_complete":
        raise RuntimeError("Complete and analyze the original 10 GHz DOE first")
    if int(decision.get("network_gate_pass_count", 0)) != 0:
        raise RuntimeError("Local rescue is only authorized when the original DOE has no gate pass")
    root = out / "doe_10ghz_local_refinement01"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite local refinement: {root}")
    observed = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    candidates = local_refinement_candidates(config)
    predicted = linear_prediction_rows(config, observed, candidates)
    plan = {
        "protocol": "v1.21-10ghz-local-refinement-stop-gate",
        "created_on": "2026-08-04",
        "source_case_count": len(observed),
        "frozen_candidate_count": len(candidates),
        "physical_topology_changed": False,
        "mesh_definition_changed": False,
        "engineering_thresholds_changed": False,
        "solver_type": str(config["doe"].get("network_solver_type", "auto")),
        "selection_basis": "Observed 16-case Jacobian and Pareto set; exactly four directional candidates before the preregistered 20-case review.",
        "stop_rule": config["stop_conditions"],
        "candidate_ids": [row["candidate_id"] for row in candidates],
    }
    write_json(root / "local_refinement_preregistration.json", plan)
    write_csv(root / "local_refinement_candidates_with_predictions.csv", predicted)
    manifests = []
    for candidate in candidates:
        physical_candidate = {key: value for key, value in candidate.items() if key != "selection_rationale"}
        manifests.append(
            prepare_case(
                config,
                physical_candidate,
                "network_s8",
                root / candidate["candidate_id"],
                [10.0],
                solver_type=str(config["doe"].get("network_solver_type", "auto")),
            )
        )
    write_json(root / "local_refinement_manifest.json", {"cases": manifests})
    return {"prepared_cases": len(manifests), "root": str(root), "predictions": predicted}


def run_local_refinement(config: dict[str, Any]) -> dict[str, Any]:
    root = output_root(config) / "doe_10ghz_local_refinement01"
    return run_case_collection(config, root, "local_refinement_manifest.json")


def analyze_local_refinement(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    doe_root = out / "doe_10ghz_network_s8"
    local_root = out / "doe_10ghz_local_refinement01"
    local_manifests = read_json(local_root / "local_refinement_manifest.json")["cases"]
    complete = [
        manifest
        for manifest in local_manifests
        if Path(manifest["touchstone_path"]).exists() and Path(manifest["touchstone_path"]).stat().st_size > 100
    ]
    if len(complete) != len(local_manifests):
        raise RuntimeError(f"Only {len(complete)}/{len(local_manifests)} local-refinement cases are complete")
    original = list(csv.DictReader((doe_root / "doe_physical_metrics.csv").open(encoding="utf-8")))
    local = [analyze_network_case(config, manifest) for manifest in complete]
    combined: list[dict[str, Any]] = []
    for source, rows in (("original_doe", original), ("local_refinement01", local)):
        for row in rows:
            converted = dict(row)
            for key in (
                "frequency_ghz",
                "final_delta_s",
                "reciprocity_error",
                "passivity_sigma",
                "passive_rl_min_db",
                "active_rl_min_db",
                "total_rl_min_db",
                "network_efficiency_min",
                "physical_vs_target_s8_max_abs_delta",
                "corrected_vs_reference_max_abs_delta_s",
            ):
                if converted.get(key) not in (None, ""):
                    converted[key] = float(converted[key])
            converted["converged"] = converted.get("converged") is True or str(converted.get("converged")).lower() == "true"
            converted["source_stage"] = source
            combined.append(converted)
    flags = pareto_flags(combined)
    for row, flag in zip(combined, flags):
        row["pareto_nondominated"] = flag
        row["network_gate_pass_10ghz"] = network_gate_pass(config, row)
    write_csv(local_root / "local_refinement_physical_metrics.csv", [row for row in combined if row["source_stage"] == "local_refinement01"])
    write_csv(local_root / "combined_20case_physical_metrics.csv", combined)
    pareto = sorted(
        [row for row in combined if row["pareto_nondominated"]],
        key=lambda row: (not row["network_gate_pass_10ghz"], row["physical_vs_target_s8_max_abs_delta"], -row["active_rl_min_db"]),
    )
    write_csv(local_root / "combined_20case_pareto.csv", pareto)
    best_active = max(float(row["active_rl_min_db"]) for row in combined)
    best_efficiency = max(float(row["network_efficiency_min"]) for row in combined)
    best_target_delta = min(float(row["physical_vs_target_s8_max_abs_delta"]) for row in combined)
    pass_count = sum(bool(row["network_gate_pass_10ghz"]) for row in combined)
    stop = config["stop_conditions"]
    review_reached = len(combined) >= int(stop["review_after_physical_candidates"])
    stop_reasons = []
    if best_active < float(stop["stop_if_best_active_rl_below_db"]):
        stop_reasons.append("best_active_rl_below_stop_threshold")
    if best_efficiency < float(stop["stop_if_best_efficiency_below"]):
        stop_reasons.append("best_network_efficiency_below_stop_threshold")
    if best_target_delta > float(stop["stop_if_best_physical_vs_target_above"]):
        stop_reasons.append("best_physical_vs_target_delta_above_stop_threshold")
    stop_topology = review_reached and pass_count == 0 and bool(stop_reasons)
    decision = {
        "stage": "C_local_refinement_stop_gate_complete",
        "complete_case_count": len(combined),
        "local_refinement_case_count": len(local),
        "network_gate_pass_count": pass_count,
        "pareto_candidate_count": len(pareto),
        "best_active_rl_db": best_active,
        "best_network_efficiency": best_efficiency,
        "best_physical_vs_target_s8_abs_delta": best_target_delta,
        "review_after_physical_candidates_reached": review_reached,
        "stop_current_post_topology": stop_topology,
        "stop_reasons": stop_reasons,
        "allow_three_frequency_optimization": pass_count > 0 and len(pareto) >= 3 and not stop_topology,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    write_json(local_root / "stop_gate_summary.json", decision)
    return {"decision": decision, "local_rows": local, "best_rows": pareto[:5]}


def run_case_collection(config: dict[str, Any], root: Path, manifest_name: str) -> dict[str, Any]:
    require_no_aedt()
    cases = read_json(root / manifest_name)["cases"]
    executable = str(resolve(config["ansys_executable"]))
    minimum_start = float(config["resources"]["minimum_free_memory_before_hfss_gib"])
    abort = float(config["resources"]["abort_free_memory_during_solve_gib"])
    poll = float(config["resources"]["poll_interval_seconds"])
    rows = []
    for manifest in cases:
        touchstone = Path(manifest["touchstone_path"])
        if touchstone.exists() and touchstone.stat().st_size > 100:
            rows.append({"candidate_id": manifest["candidate_id"], "status": "already_complete"})
            continue
        require_no_aedt()
        free = memory_available_gb()
        if math.isfinite(free) and free < minimum_start:
            rows.append({"candidate_id": manifest["candidate_id"], "status": "stopped_before_start_low_memory", "free_memory_gib": free})
            break
        folder = Path(manifest["builder_path"]).parent
        build_code, build_aborted, build_min = run_process_with_memory_guard(
            [executable, "-RunScriptAndExit", manifest["builder_path"]],
            folder / "build.log",
            abort,
            poll,
        )
        solve_code = None
        solve_aborted = False
        solve_min = math.inf
        if build_code == 0 and not build_aborted:
            solve_code, solve_aborted, solve_min = run_process_with_memory_guard(
                [executable, "-ng", "-RunScriptAndExit", manifest["solver_path"]],
                folder / "solve_export.log",
                abort,
                poll,
            )
        rows.append({
            "candidate_id": manifest["candidate_id"],
            "solver_type": manifest.get("solver_type", "auto"),
            "status": "complete" if touchstone.exists() and touchstone.stat().st_size > 100 else "failed",
            "build_return_code": build_code,
            "solve_return_code": solve_code,
            "memory_aborted": build_aborted or solve_aborted,
            "minimum_free_memory_gib": min(build_min, solve_min),
        })
        write_csv(root / "execution.csv", rows)
        if build_aborted or solve_aborted:
            break
    return {"cases": rows}


def analyze_network_case_multifrequency(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    touchstone = Path(manifest["touchstone_path"])
    desired_names = manifest["expected_port_order"]
    names = touchstone_port_names(touchstone)
    if set(names) != set(desired_names):
        raise RuntimeError(f"Unexpected ports in {touchstone}: {names}")
    frequencies, physical = reordered_network(touchstone, desired_names, 8)
    synthesis_config = read_json(resolve(config["synthesis_config"]))
    synthesis = read_json(resolve(config["synthesis_summary"]))
    graph = [tuple(int(value) for value in pair) for pair in config["fixed_topology"]["manufacturable_graph_pairs"]]
    _, series_ground, series_pair, input_ground, input_pair, output_ground, output_pair = unpack(np.asarray(synthesis["optimized_parameters"], dtype=float))
    feed_f, feed = reordered_network(resolve(synthesis_config["validated_feed_s8"]), desired_names, 8)
    integrated_f, integrated = reordered_network(resolve(synthesis_config["integrated_v118_s4"]), [f"PRE_{item}" for item in range(4)], 4)
    antenna_f, antenna = parse_touchstone(resolve(synthesis_config["trusted_antenna_s4"]), 4)
    stimulus_rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    rows = []
    aligned_matrices = []
    for physical_index, frequency in enumerate(frequencies):
        target = sparse_pi_s8(float(frequency), series_ground, series_pair, input_ground, input_pair, output_ground, output_pair, graph, synthesis_config)
        aligned, phases, target_delta = phase_align(physical[physical_index], target)
        aligned_matrices.append(aligned)
        feed_index = int(np.argmin(np.abs(feed_f - frequency)))
        integrated_index = int(np.argmin(np.abs(integrated_f - frequency)))
        antenna_index = int(np.argmin(np.abs(antenna_f - frequency)))
        effective_load = deembed_load(integrated[integrated_index], feed[feed_index])
        desired = terminate_network(feed[feed_index], antenna[antenna_index])[0]
        post = terminate_network(aligned, effective_load)[0]
        corrected = terminate_network(feed[feed_index], post)[0]
        selected = np.asarray([
            int(row["side"]) == 2 and abs(float(row["frequency_ghz"]) - float(frequency)) <= 1.0e-9
            for row in stimulus_rows
        ])
        active_rl, total_rl = active_metrics(corrected, vectors[selected, :4].T, considered[selected, :4].T)
        matched_s, load_incident, load_reflected = terminate_network(aligned, np.zeros((4, 4), dtype=complex))
        accepted = 1.0 - np.sum(np.abs(matched_s) ** 2, axis=0)
        delivered = np.sum(np.abs(load_incident) ** 2, axis=0) - np.sum(np.abs(load_reflected) ** 2, axis=0)
        rows.append({
            "candidate_id": manifest["candidate_id"],
            "frequency_ghz": float(frequency),
            "passive_rl_min_db": _minimum_passive_rl(aligned),
            "active_rl_min_db": active_rl,
            "total_rl_min_db": total_rl,
            "network_efficiency_min": float(np.min(delivered / np.maximum(accepted, EPS))),
            "physical_vs_target_s8_max_abs_delta": target_delta,
            "corrected_vs_reference_max_abs_delta_s": float(np.max(np.abs(corrected - desired))),
            "frozen_excitation_count": int(np.sum(selected)),
            "reference_phase_deg": json.dumps(phases),
        })
    aligned_stack = np.stack(aligned_matrices)
    profile = profile_metrics(touchstone.parent)
    summary = {
        "candidate_id": manifest["candidate_id"],
        "solver_type": manifest.get("solver_type", "auto"),
        "converged": profile.get("converged"),
        "final_delta_s": profile.get("final_delta_s"),
        "tetrahedra": profile.get("tetrahedra"),
        "peak_memory_gib": profile.get("peak_memory_gb"),
        "reciprocity_error_max": float(np.max(np.abs(aligned_stack - np.transpose(aligned_stack, (0, 2, 1))))),
        "passivity_sigma_max": float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in aligned_stack)),
        "passive_rl_min_db": min(row["passive_rl_min_db"] for row in rows),
        "active_rl_min_db": min(row["active_rl_min_db"] for row in rows),
        "total_rl_min_db": min(row["total_rl_min_db"] for row in rows),
        "network_efficiency_min": min(row["network_efficiency_min"] for row in rows),
        "physical_vs_target_s8_max_abs_delta": max(row["physical_vs_target_s8_max_abs_delta"] for row in rows),
        "corrected_vs_reference_max_abs_delta_s": max(row["corrected_vs_reference_max_abs_delta_s"] for row in rows),
        **manifest["parameters"],
    }
    gates = config["gates"]
    summary["three_frequency_network_gate_pass"] = bool(
        summary["converged"] is True
        and float(summary["final_delta_s"] or math.inf) <= float(gates["maximum_final_delta_s"])
        and summary["reciprocity_error_max"] <= float(gates["maximum_reciprocity_error"])
        and summary["passivity_sigma_max"] <= float(gates["maximum_passivity_sigma"])
        and summary["network_efficiency_min"] >= float(gates["minimum_network_efficiency"])
        and summary["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
        and summary["active_rl_min_db"] >= float(gates["minimum_active_rl_db"])
        and summary["total_rl_min_db"] >= float(gates["minimum_total_rl_db"])
        and summary["physical_vs_target_s8_max_abs_delta"] <= float(gates["maximum_physical_vs_target_s8_abs_delta"])
    )
    write_csv(touchstone.parent / "frequency_metrics.csv", rows)
    write_json(touchstone.parent / "analysis.json", summary)
    return summary, rows


def prepare_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_three_frequency_optimization"):
        raise RuntimeError("Three-frequency optimization is not authorized")
    source = list(csv.DictReader((out / "doe_10ghz_network_s8" / "pareto_candidates.csv").open(encoding="utf-8")))
    source.sort(key=lambda row: (str(row["network_gate_pass_10ghz"]).lower() != "true", float(row["physical_vs_target_s8_max_abs_delta"]), -float(row["active_rl_min_db"])))
    count = min(int(config["doe"]["three_frequency_pareto_candidate_count_max"]), len(source))
    minimum = int(config["doe"]["three_frequency_pareto_candidate_count_min"])
    if count < minimum:
        raise RuntimeError(f"Only {count} Pareto candidates; {minimum} are required")
    root = out / "pareto_three_frequency_s8"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite three-frequency cases: {root}")
    manifests = []
    for row in source[:count]:
        candidate = {"candidate_id": row["candidate_id"]}
        candidate.update({item["name"]: float(row[item["name"]]) for item in config["variables"]})
        manifests.append(prepare_case(config, candidate, "network_s8", root / row["candidate_id"], [float(item) for item in config["frequencies_ghz"]]))
    write_json(root / "three_frequency_manifest.json", {"cases": manifests})
    return {"prepared_cases": len(manifests), "root": str(root)}


def run_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    if not read_json(out / "stage_decision.json").get("allow_three_frequency_optimization"):
        raise RuntimeError("Three-frequency optimization is not authorized")
    return run_case_collection(config, out / "pareto_three_frequency_s8", "three_frequency_manifest.json")


def analyze_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    root = out / "pareto_three_frequency_s8"
    manifests = read_json(root / "three_frequency_manifest.json")["cases"]
    complete = [item for item in manifests if Path(item["touchstone_path"]).exists() and Path(item["touchstone_path"]).stat().st_size > 100]
    minimum = int(config["doe"]["three_frequency_pareto_candidate_count_min"])
    if len(complete) < minimum:
        raise RuntimeError(f"Only {len(complete)} complete three-frequency cases; {minimum} are required")
    summaries = [analyze_network_case_multifrequency(config, manifest)[0] for manifest in complete]
    summaries.sort(key=lambda row: (not row["three_frequency_network_gate_pass"], row["physical_vs_target_s8_max_abs_delta"], -row["active_rl_min_db"]))
    write_csv(root / "three_frequency_summary.csv", summaries)
    pass_count = sum(row["three_frequency_network_gate_pass"] for row in summaries)
    best = summaries[0]
    decision = {
        "stage": "D_three_frequency_s8_complete",
        "three_frequency_case_count": len(summaries),
        "three_frequency_gate_pass_count": pass_count,
        "selected_candidate_id": best["candidate_id"] if pass_count else None,
        "allow_independent_direct_ddm": pass_count > 0,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"decision": decision, "candidates": summaries}


def prepare_independent(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_independent_direct_ddm"):
        raise RuntimeError("Independent direct/DDM validation is not authorized")
    selected = decision["selected_candidate_id"]
    source_manifests = read_json(out / "pareto_three_frequency_s8" / "three_frequency_manifest.json")["cases"]
    source = next(item for item in source_manifests if item["candidate_id"] == selected)
    root = out / "independent_direct_ddm"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite independent validation: {root}")
    manifests = []
    for solver_name in ("direct", "ddm"):
        candidate = {"candidate_id": f"{selected}_{solver_name}", **source["parameters"]}
        manifests.append(prepare_case(config, candidate, "network_s8", root / solver_name, [float(item) for item in config["frequencies_ghz"]], solver_type=solver_name))
    write_json(root / "independent_manifest.json", {"selected_candidate_id": selected, "cases": manifests})
    return {"prepared_cases": len(manifests), "root": str(root)}


def run_independent(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    if not read_json(out / "stage_decision.json").get("allow_independent_direct_ddm"):
        raise RuntimeError("Independent direct/DDM validation is not authorized")
    return run_case_collection(config, out / "independent_direct_ddm", "independent_manifest.json")


def analyze_independent(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    root = out / "independent_direct_ddm"
    manifests = read_json(root / "independent_manifest.json")["cases"]
    if len(manifests) != 2 or not all(Path(item["touchstone_path"]).exists() and Path(item["touchstone_path"]).stat().st_size > 100 for item in manifests):
        raise RuntimeError("Both independent direct and DDM Touchstone files are required")
    summaries = [analyze_network_case_multifrequency(config, manifest)[0] for manifest in manifests]
    desired = manifests[0]["expected_port_order"]
    direct_f, direct_s = reordered_network(Path(manifests[0]["touchstone_path"]), desired, 8)
    ddm_f, ddm_s = reordered_network(Path(manifests[1]["touchstone_path"]), desired, 8)
    if not np.allclose(direct_f, ddm_f):
        raise RuntimeError("Direct and DDM frequency grids differ")
    delta = float(np.max(np.abs(direct_s - ddm_s)))
    gates = config["gates"]
    gate = bool(all(row["three_frequency_network_gate_pass"] for row in summaries) and delta <= float(gates["maximum_independent_direct_ddm_abs_delta_s"]))
    comparison = {
        "direct_vs_ddm_max_abs_delta_s": delta,
        "preferred_delta_s_gate_pass": delta <= float(gates["preferred_independent_direct_ddm_abs_delta_s"]),
        "required_delta_s_gate_pass": delta <= float(gates["maximum_independent_direct_ddm_abs_delta_s"]),
        "direct_summary": summaries[0],
        "ddm_summary": summaries[1],
        "independent_validation_pass": gate,
    }
    write_json(root / "direct_ddm_comparison.json", comparison)
    decision = {
        "stage": "E_independent_direct_ddm_complete",
        "independent_validation_pass": gate,
        "selected_candidate_id": read_json(root / "independent_manifest.json")["selected_candidate_id"],
        "allow_integrated_2x2": gate,
        "allow_scene_replay": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"decision": decision, "comparison": comparison}


def prepare_integrated(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_integrated_2x2"):
        raise RuntimeError("Integrated 2x2 solve is not authorized")
    independent = read_json(out / "independent_direct_ddm" / "independent_manifest.json")
    source = independent["cases"][0]
    candidate = {"candidate_id": independent["selected_candidate_id"], **source["parameters"]}
    root = out / "integrated_2x2_final_smoke"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite integrated smoke: {root}")
    manifest = prepare_case(config, candidate, "integrated_2x2", root / "direct01", [float(item) for item in config["frequencies_ghz"]], solver_type="direct")
    write_json(root / "integrated_manifest.json", {"cases": [manifest]})
    return {"prepared_cases": 1, "root": str(root)}


def run_integrated(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    if not read_json(out / "stage_decision.json").get("allow_integrated_2x2"):
        raise RuntimeError("Integrated 2x2 solve is not authorized")
    return run_case_collection(config, out / "integrated_2x2_final_smoke", "integrated_manifest.json")


def _read_efficiency(path: Path) -> float:
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    if len(rows) < 2 or len(rows[-1]) < 2:
        raise ValueError(f"Invalid efficiency export: {path}")
    return float(rows[-1][1])


def analyze_integrated(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    root = out / "integrated_2x2_final_smoke"
    manifest = read_json(root / "integrated_manifest.json")["cases"][0]
    touchstone = Path(manifest["touchstone_path"])
    if not touchstone.exists() or touchstone.stat().st_size <= 100:
        raise RuntimeError("Integrated S4 export is incomplete")
    desired_ports = [f"PRE_{index}" for index in range(4)]
    frequencies, integrated = reordered_network(touchstone, desired_ports, 4)
    independent = read_json(out / "independent_direct_ddm" / "independent_manifest.json")
    direct_manifest = next(item for item in independent["cases"] if item["solver_type"] == "direct")
    network_f, network = reordered_network(Path(direct_manifest["touchstone_path"]), direct_manifest["expected_port_order"], 8)
    antenna_f, antenna = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    if not np.allclose(frequencies, network_f) or not np.allclose(frequencies, antenna_f):
        raise RuntimeError("Integrated, network, and antenna frequency grids differ")
    stimulus_rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    frequency_rows = []
    expected_eep = []
    efficiencies = []
    for index, frequency in enumerate(frequencies):
        cascade = terminate_network(network[index], antenna[index])[0]
        selected = np.asarray([
            int(row["side"]) == 2 and abs(float(row["frequency_ghz"]) - float(frequency)) <= 1.0e-9
            for row in stimulus_rows
        ])
        active_rl, total_rl = active_metrics(integrated[index], vectors[selected, :4].T, considered[selected, :4].T)
        code = f"{float(frequency):.2f}".replace(".", "p")
        for port in range(4):
            eep_path = touchstone.parent / f"eep_pre_{port}_{code}.csv"
            efficiency_path = touchstone.parent / f"efficiency_pre_{port}_{code}.csv"
            expected_eep.append(eep_path)
            if efficiency_path.exists() and efficiency_path.stat().st_size > 20:
                efficiencies.append(_read_efficiency(efficiency_path))
        frequency_rows.append({
            "frequency_ghz": float(frequency),
            "passive_rl_min_db": _minimum_passive_rl(integrated[index]),
            "active_rl_min_db": active_rl,
            "total_rl_min_db": total_rl,
            "integrated_vs_s8_cascade_max_abs_delta_s": float(np.max(np.abs(integrated[index] - cascade))),
            "frozen_excitation_count": int(np.sum(selected)),
        })
    profile = profile_metrics(touchstone.parent)
    eep_complete_count = sum(path.exists() and path.stat().st_size > 100 for path in expected_eep)
    summary = {
        "candidate_id": manifest["candidate_id"],
        "converged": profile.get("converged"),
        "final_delta_s": profile.get("final_delta_s"),
        "tetrahedra": profile.get("tetrahedra"),
        "peak_memory_gib": profile.get("peak_memory_gb"),
        "reciprocity_error_max": float(np.max(np.abs(integrated - np.transpose(integrated, (0, 2, 1))))),
        "passivity_sigma_max": float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in integrated)),
        "passive_rl_min_db": min(row["passive_rl_min_db"] for row in frequency_rows),
        "active_rl_min_db": min(row["active_rl_min_db"] for row in frequency_rows),
        "total_rl_min_db": min(row["total_rl_min_db"] for row in frequency_rows),
        "integrated_vs_s8_cascade_max_abs_delta_s": max(row["integrated_vs_s8_cascade_max_abs_delta_s"] for row in frequency_rows),
        "eep_expected_count": len(expected_eep),
        "eep_complete_count": eep_complete_count,
        "radiation_efficiency_min": min(efficiencies) if efficiencies else None,
    }
    gates = config["gates"]
    summary["integrated_2x2_gate_pass"] = bool(
        summary["converged"] is True
        and float(summary["final_delta_s"] or math.inf) <= float(gates["maximum_final_delta_s"])
        and summary["reciprocity_error_max"] <= float(gates["maximum_reciprocity_error"])
        and summary["passivity_sigma_max"] <= float(gates["maximum_passivity_sigma"])
        and summary["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
        and summary["active_rl_min_db"] >= float(gates["minimum_active_rl_db"])
        and summary["total_rl_min_db"] >= float(gates["minimum_total_rl_db"])
        and summary["integrated_vs_s8_cascade_max_abs_delta_s"] <= float(gates["maximum_integrated_vs_s8_cascade_abs_delta_s"])
        and summary["eep_complete_count"] == summary["eep_expected_count"]
        and summary["radiation_efficiency_min"] is not None
        and summary["radiation_efficiency_min"] >= float(gates["minimum_network_efficiency"])
    )
    write_csv(root / "integrated_frequency_metrics.csv", frequency_rows)
    write_json(root / "integrated_analysis.json", summary)
    decision = {
        "stage": "E_integrated_2x2_complete",
        "integrated_2x2_gate_pass": summary["integrated_2x2_gate_pass"],
        "allow_frozen_20_scene_replay": summary["integrated_2x2_gate_pass"],
        "allow_new_s256_eep": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": "A passing integrated 2x2 authorizes frozen-scene replay only; larger arrays and labels require the replay gate.",
    }
    write_json(out / "stage_decision.json", decision)
    return {"decision": decision, "analysis": summary}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    decision = read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None
    return {
        "output_directory": str(out),
        "preregistered": (out / "preregistration.json").exists(),
        "build_smoke_prepared": (out / "build_smoke" / "smoke_manifest.json").exists(),
        "build_smoke_executed": (out / "build_smoke" / "build_execution.csv").exists(),
        "doe_prepared": (out / "doe_10ghz_network_s8" / "doe_manifest.json").exists(),
        "doe_metrics_complete": (out / "doe_10ghz_network_s8" / "doe_physical_metrics.csv").exists(),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=(
            "preregister",
            "apply-route-amendment",
            "apply-solver-amendment",
            "prepare-smoke",
            "run-build-smoke",
            "audit-smoke",
            "prepare-doe",
            "refresh-resource-gate",
            "run-doe",
            "run-doe-smoke",
            "analyze-doe",
            "prepare-local-refinement",
            "run-local-refinement",
            "analyze-local-refinement",
            "prepare-three-frequency",
            "run-three-frequency",
            "analyze-three-frequency",
            "prepare-independent",
            "run-independent",
            "analyze-independent",
            "prepare-integrated",
            "run-integrated",
            "analyze-integrated",
            "status",
        ),
        default="status",
    )
    args = parser.parse_args()
    config = read_json(args.config)
    actions = {
        "preregister": preregister,
        "apply-route-amendment": apply_route_amendment,
        "apply-solver-amendment": apply_solver_amendment,
        "prepare-smoke": prepare_smoke,
        "run-build-smoke": run_build_smoke,
        "audit-smoke": audit_smoke,
        "prepare-doe": prepare_doe,
        "refresh-resource-gate": refresh_resource_gate,
        "run-doe": run_doe,
        "run-doe-smoke": run_doe_smoke,
        "analyze-doe": analyze_doe,
        "prepare-local-refinement": prepare_local_refinement,
        "run-local-refinement": run_local_refinement,
        "analyze-local-refinement": analyze_local_refinement,
        "prepare-three-frequency": prepare_three_frequency,
        "run-three-frequency": run_three_frequency,
        "analyze-three-frequency": analyze_three_frequency,
        "prepare-independent": prepare_independent,
        "run-independent": run_independent,
        "analyze-independent": analyze_independent,
        "prepare-integrated": prepare_integrated,
        "run-integrated": run_integrated,
        "analyze-integrated": analyze_integrated,
        "status": status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
