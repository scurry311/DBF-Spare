#!/usr/bin/env python3
"""Screen a true differential, direct-gap, dual-resonant 1x1 radiator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import memory_available_gb, parse_touchstone, profile_metrics
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v125_feedpoint_input_impedance import topology_warning_count, write_csv, write_json
from run_v128_true_balanced_dual_resonant import helpers, vp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v129_true_balanced_radiator_screen_preregistered.json"
DESIGN_NAME = "V129_TrueBalanced_DirectGap"
EPS = 1.0e-15


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def builder_text(project: Path, geometry: dict[str, Any], frequency_ghz: float) -> str:
    g = geometry
    board_x = float(g["board_x_mm"])
    board_y = float(g["board_y_mm"])
    h = float(g["substrate_thickness_mm"])
    copper = float(g["copper_thickness_mm"])
    gap = float(g["primary_inner_gap_mm"])
    primary_l = float(g["primary_arm_length_mm"])
    primary_w = float(g["primary_arm_width_mm"])
    secondary_l = float(g["secondary_arm_length_mm"])
    secondary_w = float(g["secondary_arm_width_mm"])
    secondary_y = float(g["secondary_arm_offset_y_mm"])
    neck_l = float(g["secondary_neck_length_x_mm"])
    margin = float(g["port_sheet_margin_mm"])
    mesh = float(g["local_mesh_max_length_mm"])
    refine = float(g["adaptive_refinement_percent"])
    max_passes = int(g["maximum_passes"])
    primary_bottom = -primary_w / 2.0
    secondary_bottom = secondary_y - secondary_w / 2.0
    neck_height = secondary_bottom - primary_bottom + 0.02
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V129", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(g['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(g['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board_x/2:.7f}, {-board_y/2:.7f}, {-h:.7f}, {board_x:.7f}, {board_y:.7f}, {h:.7f}, "RO5880_V129", True

' Negative fork: the primary and secondary arms are one conductor.
CreateBox oEditor, "PrimaryN", {-gap/2-primary_l:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "SecondaryN", {-gap/2-secondary_l:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "NeckN", {-gap/2-neck_l:.7f}, {primary_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}, {copper:.7f}, "copper", False
UniteSelection oEditor, "PrimaryN,SecondaryN,NeckN"

' Positive fork is the exact x mirror; there is no reference ground.
CreateBox oEditor, "PrimaryP", {gap/2:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "SecondaryP", {gap/2:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "NeckP", {gap/2:.7f}, {primary_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}, {copper:.7f}, "copper", False
UniteSelection oEditor, "PrimaryP,SecondaryP,NeckP"

' The reference plane is the physical radiator gap, isolating radiator impedance from any transformer.
CreateSheetY oEditor, "PortSheet_DIFF", {-gap/2-margin:.7f}, 0, {-margin:.7f}, {gap+2*margin:.7f}, {copper+2*margin:.7f}
AssignDifferentialPort oBoundary, "P_DIFF", "PortSheet_DIFF", {-gap/2:.7f}, {gap/2:.7f}, 0, {copper/2:.7f}

CreateBox oEditor, "AirRegion", {-board_x/2-12:.7f}, {-board_y/2-12:.7f}, {-h-12:.7f}, {board_x+24:.7f}, {board_y+24:.7f}, {h+copper+24:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:ForkMesh_0p180mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array("PrimaryN", "PrimaryP"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency_ghz:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {max_passes}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {refine:.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V129", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{helpers()}
'''


def solver_text(project: Path, touchstone: Path) -> str:
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, vars, variation
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
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def prepare_case(out: Path, geometry: dict[str, Any], frequency_ghz: float) -> dict[str, Any]:
    case_id = str(geometry["candidate_id"])
    folder = out / "direct_gap_1x1" / case_id
    folder.mkdir(parents=True)
    project = folder / f"v129_{case_id}.aedt"
    touchstone = folder / f"v129_{case_id}.s1p"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    build_source = builder_text(project, geometry, frequency_ghz)
    builder.write_text(build_source, encoding="ascii")
    solver.write_text(solver_text(project, touchstone), encoding="ascii")
    case = {
        "case_id": case_id,
        "frequency_ghz": frequency_ghz,
        "geometry": geometry,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "cad_audit": {
            "differential_port_definition_count": build_source.count("AssignDifferentialPort oBoundary"),
            "reference_ground_count": build_source.count('"ReferenceGround"') + build_source.count('"Ground"'),
            "negative_positive_conductor_pair": all(token in build_source for token in ('"PrimaryN"', '"PrimaryP"')),
            "secondary_resonator_objects": all(token in build_source for token in ('"SecondaryN"', '"SecondaryP"')),
            "direct_gap_reference_plane": "physical radiator gap" in build_source,
        },
    }
    write_json(folder / "case_manifest.json", case)
    return case


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.29 output: {out}")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    base = config["frozen_geometry"]
    cases = [prepare_case(out, {**base, **candidate}, float(config["frequency_ghz"])) for candidate in config["candidates"]]
    write_json(out / "case_manifest.json", {"cases": cases})
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {"head_commit": head, "tag_commit": tag, "free_memory_gib": memory_available_gb(), "aedt_processes": aedt_processes()},
            "evidence_rules": {
                "all_six_geometries_fixed_before_solve": True,
                "true_differential_port_without_reference_ground": True,
                "radiator_reference_plane_excludes_transformer": True,
                "array_and_learning_stages_locked": True
            },
        },
    )
    decision = {
        "stage": "A_direct_gap_screen_preregistered",
        "allow_run": True,
        "allow_fixed_reference_transformer": False,
        "allow_three_frequency": False,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "case_count": len(cases), "decision": decision}


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


def run(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_run"):
        raise RuntimeError("Direct-gap screen is not authorized")
    cases = read_json(out / "case_manifest.json")["cases"]
    progress = out / "run_progress.csv"
    rows = list(csv.DictReader(progress.open(encoding="utf-8"))) if progress.exists() else []
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
            build = subprocess.run([executable, "-RunScriptAndExit", case["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        warnings = topology_warning_count(folder)
        cad = case["cad_audit"]
        build_pass = bool(
            build.returncode == 0
            and Path(case["project_path"]).exists()
            and warnings == 0
            and cad["differential_port_definition_count"] == 1
            and cad["reference_ground_count"] == 0
            and cad["negative_positive_conductor_pair"]
            and cad["secondary_resonator_objects"]
            and cad["direct_gap_reference_plane"]
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
            "build_return_code": build.returncode,
            "solve_return_code": code,
            "memory_aborted": aborted,
            "free_memory_gib_before": free,
            "minimum_free_memory_gib": minimum_free,
            "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
            "topology_warning_count": warnings,
        }
        rows.append(row)
        write_csv(progress, rows)
        if code != 0 or aborted or not row["touchstone_exists"]:
            raise RuntimeError(f"Solve failed for {case['case_id']}: {row}")
    return {"completed_count": len(rows), "rows": rows}


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "case_manifest.json")["cases"]
    gates = config["gates"]
    rows: list[dict[str, Any]] = []
    for case in cases:
        folder = Path(case["touchstone_path"]).parent
        frequencies, matrices = parse_touchstone(Path(case["touchstone_path"]), 1)
        index = int(np.argmin(np.abs(frequencies - float(case["frequency_ghz"]))))
        s11 = matrices[index, 0, 0]
        impedance = 50.0 * (1.0 + s11) / (1.0 - s11)
        passive_rl = float(-20.0 * np.log10(max(float(abs(s11)), EPS)))
        profile = profile_metrics(folder)
        numerical_pass = bool(
            profile.get("converged") is True
            and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
            and topology_warning_count(folder) <= int(gates["maximum_port_topology_warning_count"])
        )
        impedance_reachable = bool(
            float(gates["minimum_input_resistance_ohm"]) <= impedance.real <= float(gates["maximum_input_resistance_ohm"])
            and abs(impedance.imag) <= float(gates["maximum_absolute_input_reactance_ohm"])
        )
        rows.append(
            {
                "case_id": case["case_id"],
                **case["geometry"],
                **profile,
                "frequency_ghz": float(frequencies[index]),
                "s11_real": float(s11.real),
                "s11_imag": float(s11.imag),
                "s11_magnitude": float(abs(s11)),
                "s11_phase_deg": float(np.angle(s11, deg=True)),
                "input_resistance_ohm": float(impedance.real),
                "input_reactance_ohm": float(impedance.imag),
                "passive_rl_db": passive_rl,
                "topology_warning_count": topology_warning_count(folder),
                "numerical_gate_pass": numerical_pass,
                "impedance_transform_reachable": impedance_reachable,
                "screen_gate_pass": numerical_pass and passive_rl >= float(gates["minimum_screen_passive_rl_db"]),
            }
        )
    rows.sort(key=lambda item: (not item["screen_gate_pass"], not item["impedance_transform_reachable"], -float(item["passive_rl_db"])))
    write_csv(out / "candidate_metrics.csv", rows)
    passing = [row for row in rows if row["screen_gate_pass"]]
    reachable = [row for row in rows if row["numerical_gate_pass"] and row["impedance_transform_reachable"]]
    selected = passing[0] if passing else (reachable[0] if reachable else None)
    allow_transformer = selected is not None
    summary = {
        "candidate_count": len(rows),
        "numerically_valid_count": sum(bool(row["numerical_gate_pass"]) for row in rows),
        "screen_gate_pass_count": len(passing),
        "transformer_reachable_count": len(reachable),
        "selected_radiator_candidate": selected["case_id"] if selected else None,
        "best_passive_rl_db": max(float(row["passive_rl_db"]) for row in rows),
        "allow_fixed_reference_transformer": allow_transformer,
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "B_direct_gap_screen_complete",
        "allow_run": True,
        "allow_fixed_reference_transformer": allow_transformer,
        "allow_three_frequency": False,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "A numerically credible radiator impedance is inside the preregistered transformer-reachable region; a fixed-reference three-section CPS transformer screen is authorized."
            if allow_transformer
            else "The direct-gap dual-resonant radiator is outside the preregistered transformable impedance region; transformer, array, EEP, and learning work remain locked."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": rows, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "case_manifest.json")["cases"] if (out / "case_manifest.json").exists() else []
    return {
        "output_directory": str(out),
        "touchstone_count": sum(Path(case["touchstone_path"]).exists() for case in cases),
        "case_count": len(cases),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=str)
    parser.add_argument("--mode", choices=("preregister", "run", "analyze", "status"), default="status")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.output_directory:
        config["output_directory"] = args.output_directory
    actions = {"preregister": preregister, "run": run, "analyze": analyze, "status": status}
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
