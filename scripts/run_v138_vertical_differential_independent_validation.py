#!/usr/bin/env python3
"""Validate v1.37 with independent direct/DDM meshes, then three-frequency efficiency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from analyze_v137_vertical_mesh_audit import segment_audit
from run_v114_small_cell_broadband_feed import (
    efficiency_from_csv,
    memory_available_gb,
    parse_touchstone,
    profile_metrics,
)
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v125_feedpoint_input_impedance import topology_warning_count, write_csv, write_json
from run_v128_true_balanced_dual_resonant import vp
from run_v130_fixed_reference_cps_transformer import read_json, resolve
from run_v132_vertical_differential_launch import DESIGN_NAME, builder_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v138_vertical_differential_independent_validation_preregistered.json"
EPS = 1.0e-15


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solver_suffix(solver_type: str) -> str:
    options = {
        "direct": ', "DrivenSolverType:=", "Direct Solver"',
        "ddm": ', "DrivenSolverType:=", "Domain Decomposition", "IterativeResidual:=", 0.000001, "DDMSolverResidual:=", 0.000001',
    }
    if solver_type not in options:
        raise ValueError(f"Unsupported solver type: {solver_type}")
    return options[solver_type]


def independent_builder_text(
    project: Path,
    geometry: dict[str, Any],
    frequency_ghz: float,
    solver_type: str,
) -> str:
    text = builder_text(project, geometry, frequency_ghz)
    needle = '"PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)'
    replacement = needle[:-1] + solver_suffix(solver_type) + ")"
    if text.count(needle) != 1:
        raise RuntimeError("Unable to inject the frozen direct/DDM solver option")
    return text.replace(needle, replacement)


def solver_text(
    project: Path,
    touchstone: Path,
    frequency_ghz: float,
    efficiency_csv: Path | None,
) -> str:
    report = ""
    if efficiency_csv is not None:
        report = f'''Set oReport = oDesign.GetModule("ReportSetup")
reportName = "V138_RadiationEfficiency"
On Error Resume Next
oReport.DeleteReports Array(reportName)
Err.Clear
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("{frequency_ghz:g}GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(efficiency_csv)}"
On Error GoTo 0'''
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, reportName
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
{report}
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def prepare_case(
    root: Path,
    case_id: str,
    geometry: dict[str, Any],
    frequency_ghz: float,
    solver_type: str,
    export_efficiency: bool,
) -> dict[str, Any]:
    folder = root / case_id
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite v1.38 case: {folder}")
    folder.mkdir(parents=True)
    project = folder / f"v138_{case_id}.aedt"
    touchstone = folder / f"v138_{case_id}.s1p"
    efficiency = folder / "radiation_efficiency.csv" if export_efficiency else None
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    source = independent_builder_text(project, geometry, frequency_ghz, solver_type)
    builder.write_text(source, encoding="ascii")
    solver.write_text(solver_text(project, touchstone, frequency_ghz, efficiency), encoding="ascii")
    case = {
        "case_id": case_id,
        "frequency_ghz": frequency_ghz,
        "solver_type": solver_type,
        "geometry": geometry,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "efficiency_csv_path": str(efficiency.resolve()) if efficiency else None,
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "builder_sha256": sha256(builder),
        "solver_sha256": sha256(solver),
        "cad_audit": {
            "differential_port_definition_count": source.count("AssignDifferentialPortZ oBoundary"),
            "reference_ground_count": source.count('"ReferenceGround"') + source.count('"Ground"'),
            "finite_conductivity_sheet": "AssignFiniteCond" in source,
            "square_post_pair": all(token in source for token in ('"ViaN"', '"ViaP"')),
            "partitioned_substrate": "SubstrateRowCenter" in source,
            "solver_option_count": source.count("DrivenSolverType:="),
        },
    }
    write_json(folder / "case_manifest.json", case)
    return case


def frozen_parent(config: dict[str, Any]) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    tag = subprocess.run(
        ["git", "rev-list", "-n", "1", config["parent_tag"]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    return {"head_commit": head, "tag_commit": tag}


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def wait_for_memory(config: dict[str, Any]) -> float:
    required = float(config["resources"]["minimum_free_memory_before_1x1_gib"])
    deadline = time.time() + float(config["resources"]["memory_recovery_wait_seconds"])
    while True:
        require_no_aedt()
        free = memory_available_gb()
        if free >= required:
            return free
        if time.time() >= deadline:
            raise MemoryError(f"Memory did not recover to {required:.2f} GiB; current {free:.2f} GiB")
        time.sleep(float(config["resources"]["poll_interval_seconds"]))


def run_cases(
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    progress_path: Path,
) -> dict[str, Any]:
    rows = list(csv.DictReader(progress_path.open(encoding="utf-8-sig"))) if progress_path.exists() else []
    complete = {row["case_id"] for row in rows if row.get("touchstone_exists", "").lower() == "true"}
    executable = str(resolve(config["ansys_executable"]))
    abort = float(config["resources"]["abort_free_memory_during_solve_gib"])
    poll = float(config["resources"]["poll_interval_seconds"])
    for case in cases:
        if case["case_id"] in complete:
            continue
        free = wait_for_memory(config)
        folder = Path(case["project_path"]).parent
        with (folder / "build.log").open("w", encoding="utf-8") as handle:
            build = subprocess.run(
                [executable, "-RunScriptAndExit", case["builder_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        cad = case["cad_audit"]
        build_pass = bool(
            build.returncode == 0
            and Path(case["project_path"]).exists()
            and cad["differential_port_definition_count"] == 1
            and cad["reference_ground_count"] == 0
            and cad["finite_conductivity_sheet"]
            and cad["square_post_pair"]
            and cad["partitioned_substrate"]
            and cad["solver_option_count"] == 1
            and topology_warning_count(folder) == 0
        )
        if not build_pass:
            raise RuntimeError(f"Build gate failed for {case['case_id']}")
        code, aborted, minimum_free = run_process_with_memory_guard(
            [executable, "-ng", "-RunScriptAndExit", case["solver_path"]],
            folder / "solve_export.log",
            abort,
            poll,
        )
        touchstone = Path(case["touchstone_path"])
        row = {
            "case_id": case["case_id"],
            "solver_type": case["solver_type"],
            "build_return_code": build.returncode,
            "solve_return_code": code,
            "memory_aborted": aborted,
            "free_memory_gib_before": free,
            "minimum_free_memory_gib": minimum_free,
            "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
            "efficiency_exists": bool(
                case.get("efficiency_csv_path")
                and Path(case["efficiency_csv_path"]).exists()
                and Path(case["efficiency_csv_path"]).stat().st_size > 10
            ),
        }
        rows.append(row)
        write_csv(progress_path, rows)
        if code != 0 or aborted or not row["touchstone_exists"]:
            raise RuntimeError(f"Solve failed for {case['case_id']}: {row}")
    return {"completed_count": len(rows), "rows": rows}


def case_metrics(case: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], complex]:
    folder = Path(case["project_path"]).parent
    frequencies, matrices = parse_touchstone(Path(case["touchstone_path"]), 1)
    target = float(case["frequency_ghz"])
    index = int(np.argmin(np.abs(frequencies - target)))
    s11 = complex(matrices[index, 0, 0])
    impedance = 50.0 * (1.0 + s11) / (1.0 - s11)
    profile = profile_metrics(folder)
    bodies, lengths = segment_audit(folder)
    conductive = {"PrimaryN", "PrimaryP", "PadN", "PadP", "ViaN", "ViaP"}
    conductor_count = sum(count for body, count in bodies.items() if body in conductive)
    allowed = set(config["gates"]["allowed_residual_small_segment_bodies"])
    unexpected = sorted(set(bodies) - conductive - allowed)
    efficiency_path = case.get("efficiency_csv_path")
    efficiency = efficiency_from_csv(Path(efficiency_path)) if efficiency_path else None
    row = {
        "case_id": case["case_id"],
        "solver_type": case["solver_type"],
        "frequency_ghz": float(frequencies[index]),
        **profile,
        "s11_real": float(s11.real),
        "s11_imag": float(s11.imag),
        "s11_magnitude": float(abs(s11)),
        "passive_rl_db": float(-20.0 * np.log10(max(abs(s11), EPS))),
        "input_resistance_ohm": float(impedance.real),
        "input_reactance_ohm": float(impedance.imag),
        "input_impedance_magnitude_ohm": float(abs(impedance)),
        "radiation_efficiency": efficiency,
        "total_small_segment_message_count": int(sum(bodies.values())),
        "conductor_small_segment_message_count": int(conductor_count),
        "minimum_segment_length_mm": min(lengths) if lengths else None,
        "maximum_segment_length_mm": max(lengths) if lengths else None,
        "body_counts_json": json.dumps(dict(sorted(bodies.items())), separators=(",", ":")),
        "unexpected_small_segment_bodies_json": json.dumps(unexpected, separators=(",", ":")),
        "allowed_residual_bodies_only": not unexpected,
        "topology_warning_count": topology_warning_count(folder),
    }
    return row, s11


def preregister_crosscheck(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.38 output: {out}")
    parent = frozen_parent(config)
    prior = read_json(config["inputs"]["v137_stage_decision"])
    if not prior.get("conductor_geometry_gate_pass") or not prior.get("matching_numerical_gate_pass"):
        raise RuntimeError("v1.37 does not authorize an independent solver audit")
    if prior.get("strict_total_small_segment_gate_pass"):
        raise RuntimeError("v1.38 is reserved for auditing the known residual dielectric/air messages")
    root = out / "independent_10ghz"
    cases = [
        prepare_case(
            root,
            item["case_id"],
            dict(config["frozen_geometry"]),
            float(config["frequency_ghz"]),
            item["solver_type"],
            False,
        )
        for item in config["independent_solver_cases"]
    ]
    write_json(root / "case_manifest.json", {"cases": cases})
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {
                **parent,
                "free_memory_gib": memory_available_gb(),
                "aedt_processes": aedt_processes(),
                "runner_sha256": sha256(Path(__file__)),
            },
            "evidence_rules": {
                "three_new_projects_with_independent_adaptive_meshes": True,
                "two_direct_repeats_and_one_ddm_repeat": True,
                "geometry_frequency_and_thresholds_frozen": True,
                "no_total_warning_threshold_relaxation_before_cross_solver_evidence": True,
                "array_eep_labels_and_critic_locked": True,
            },
        },
    )
    decision = {
        "stage": "A_independent_crosscheck_preregistered",
        "allow_crosscheck_run": True,
        "allow_three_frequency_efficiency": False,
        "allow_2x2": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "case_count": len(cases), "decision": decision}


def run_crosscheck(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_crosscheck_run"):
        raise RuntimeError("Independent crosscheck is not authorized")
    root = out / "independent_10ghz"
    cases = read_json(root / "case_manifest.json")["cases"]
    return run_cases(config, cases, root / "run_progress.csv")


def analyze_crosscheck(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    root = out / "independent_10ghz"
    cases = read_json(root / "case_manifest.json")["cases"]
    rows: list[dict[str, Any]] = []
    s_values: dict[str, complex] = {}
    z_values: dict[str, complex] = {}
    for case in cases:
        row, s11 = case_metrics(case, config)
        rows.append(row)
        s_values[case["case_id"]] = s11
        z_values[case["case_id"]] = 50.0 * (1.0 + s11) / (1.0 - s11)
    comparisons = []
    for left, right in combinations(cases, 2):
        left_id = left["case_id"]
        right_id = right["case_id"]
        comparisons.append(
            {
                "left_case_id": left_id,
                "right_case_id": right_id,
                "abs_delta_s": float(abs(s_values[left_id] - s_values[right_id])),
                "passive_rl_delta_db": float(
                    abs(
                        next(row["passive_rl_db"] for row in rows if row["case_id"] == left_id)
                        - next(row["passive_rl_db"] for row in rows if row["case_id"] == right_id)
                    )
                ),
                "input_impedance_difference_ohm": float(abs(z_values[left_id] - z_values[right_id])),
            }
        )
    write_csv(root / "solver_metrics.csv", rows)
    write_csv(root / "pairwise_comparison.csv", comparisons)
    gates = config["gates"]
    max_delta_s = max(row["abs_delta_s"] for row in comparisons)
    rl_span = max(row["passive_rl_db"] for row in rows) - min(row["passive_rl_db"] for row in rows)
    max_z_delta = max(row["input_impedance_difference_ohm"] for row in comparisons)
    physical_gate = bool(
        len(rows) == 3
        and all(row.get("converged") is True for row in rows)
        and all(float(row.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"]) for row in rows)
        and all(row["passive_rl_db"] >= float(gates["minimum_passive_rl_db"]) for row in rows)
        and all(row["conductor_small_segment_message_count"] <= int(gates["maximum_conductor_small_segment_count"]) for row in rows)
        and all(row["allowed_residual_bodies_only"] for row in rows)
        and all(row["topology_warning_count"] <= int(gates["maximum_port_topology_warning_count"]) for row in rows)
    )
    benign_gate = bool(
        physical_gate
        and max_delta_s <= float(gates["maximum_benign_pairwise_abs_delta_s"])
        and rl_span <= float(gates["maximum_passive_rl_span_db"])
        and max_z_delta <= float(gates["maximum_input_impedance_difference_ohm"])
    )
    summary = {
        "case_count": len(rows),
        "minimum_passive_rl_db": min(row["passive_rl_db"] for row in rows),
        "maximum_pairwise_abs_delta_s": max_delta_s,
        "required_pairwise_gate_pass": max_delta_s <= float(gates["maximum_required_pairwise_abs_delta_s"]),
        "benign_pairwise_gate_pass": max_delta_s <= float(gates["maximum_benign_pairwise_abs_delta_s"]),
        "passive_rl_span_db": rl_span,
        "maximum_input_impedance_difference_ohm": max_z_delta,
        "maximum_conductor_small_segment_count": max(row["conductor_small_segment_message_count"] for row in rows),
        "residual_messages_confined_to_allowed_bodies": all(row["allowed_residual_bodies_only"] for row in rows),
        "physical_gate_pass": physical_gate,
        "residual_warning_benign_gate_pass": benign_gate,
    }
    write_json(root / "crosscheck_summary.json", summary)
    decision = {
        "stage": "B_independent_crosscheck_complete",
        "residual_warning_benign_gate_pass": benign_gate,
        "allow_three_frequency_efficiency": benign_gate,
        "allow_2x2": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "Independent direct/direct/DDM meshes agree within the preregistered S, RL, impedance, convergence, and body-attribution limits."
            if benign_gate
            else "The residual dielectric/air warnings are not yet proven benign; three-frequency efficiency remains locked."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": rows, "comparisons": comparisons, "decision": decision}


def prepare_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_three_frequency_efficiency"):
        raise RuntimeError("Three-frequency efficiency is not authorized")
    root = out / "three_frequency_efficiency"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite v1.38 three-frequency stage: {root}")
    cases = []
    for frequency in config["frequencies_ghz"]:
        value = float(frequency)
        case_id = f"direct_f{value:.2f}".replace(".", "p")
        cases.append(
            prepare_case(
                root,
                case_id,
                dict(config["frozen_geometry"]),
                value,
                "direct",
                True,
            )
        )
    write_json(root / "case_manifest.json", {"cases": cases})
    decision = read_json(out / "stage_decision.json")
    decision.update({"stage": "C_three_frequency_efficiency_prepared", "allow_three_frequency_run": True})
    write_json(out / "stage_decision.json", decision)
    return {"prepared_count": len(cases), "root": str(root)}


def run_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_three_frequency_run"):
        raise RuntimeError("Three-frequency run is not authorized")
    root = out / "three_frequency_efficiency"
    cases = read_json(root / "case_manifest.json")["cases"]
    return run_cases(config, cases, root / "run_progress.csv")


def analyze_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    root = out / "three_frequency_efficiency"
    cases = read_json(root / "case_manifest.json")["cases"]
    rows = [case_metrics(case, config)[0] for case in cases]
    rows.sort(key=lambda row: row["frequency_ghz"])
    write_csv(root / "three_frequency_efficiency_metrics.csv", rows)
    gates = config["gates"]
    gate = bool(
        len(rows) == 3
        and all(row.get("converged") is True for row in rows)
        and all(float(row.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"]) for row in rows)
        and all(row["passive_rl_db"] >= float(gates["minimum_passive_rl_db"]) for row in rows)
        and all(row["radiation_efficiency"] is not None and row["radiation_efficiency"] >= float(gates["minimum_radiation_efficiency"]) for row in rows)
        and all(row["conductor_small_segment_message_count"] <= int(gates["maximum_conductor_small_segment_count"]) for row in rows)
        and all(row["allowed_residual_bodies_only"] for row in rows)
        and all(row["topology_warning_count"] <= int(gates["maximum_port_topology_warning_count"]) for row in rows)
    )
    summary = {
        "frequency_count": len(rows),
        "minimum_passive_rl_db": min(row["passive_rl_db"] for row in rows),
        "minimum_radiation_efficiency": min(row["radiation_efficiency"] for row in rows if row["radiation_efficiency"] is not None) if any(row["radiation_efficiency"] is not None for row in rows) else None,
        "maximum_final_delta_s": max(float(row.get("final_delta_s") or math.inf) for row in rows),
        "maximum_conductor_small_segment_count": max(row["conductor_small_segment_message_count"] for row in rows),
        "three_frequency_efficiency_gate_pass": gate,
    }
    write_json(root / "stage_summary.json", summary)
    decision = {
        "stage": "D_three_frequency_efficiency_complete",
        "residual_warning_benign_gate_pass": read_json(out / "independent_10ghz" / "crosscheck_summary.json")["residual_warning_benign_gate_pass"],
        "three_frequency_efficiency_gate_pass": gate,
        "allow_2x2": gate,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "The frozen finite-sheet differential input passed independent solver consistency and three-frequency passive-RL/efficiency gates; only physical 2x2 is authorized next."
            if gate
            else "Three-frequency passive-RL or radiation-efficiency failed; all array and learning stages remain locked."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": rows, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    return {
        "output_directory": str(out),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=(
            "preregister-crosscheck",
            "run-crosscheck",
            "analyze-crosscheck",
            "prepare-three-frequency",
            "run-three-frequency",
            "analyze-three-frequency",
            "status",
        ),
        default="status",
    )
    args = parser.parse_args()
    config = read_json(args.config)
    actions = {
        "preregister-crosscheck": preregister_crosscheck,
        "run-crosscheck": run_crosscheck,
        "analyze-crosscheck": analyze_crosscheck,
        "prepare-three-frequency": prepare_three_frequency,
        "run-three-frequency": run_three_frequency,
        "analyze-three-frequency": analyze_three_frequency,
        "status": status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
