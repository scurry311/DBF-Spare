#!/usr/bin/env python3
"""Screen an independently variable, fixed-reference CPS impedance transformer."""

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
DEFAULT_CONFIG = ROOT / "configs" / "v130_fixed_reference_cps_transformer_preregistered.json"
DESIGN_NAME = "V130_FixedReference_CPS"
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
    pitch = float(g["feed_pair_center_pitch_mm"])
    input_w = float(g["input_trace_width_mm"])
    post_w = float(g["post_trace_width_mm"])
    feed_start = float(g["feed_y_start_mm"])
    feed_end = float(g["feed_y_end_mm"])
    transformer_center = float(g["transformer_center_y_mm"])
    transformer_w = float(g["transformer_width_mm"])
    transformer_l = float(g["transformer_length_mm"])
    transformer_start = transformer_center - transformer_l / 2.0
    transformer_end = transformer_center + transformer_l / 2.0
    overlap = float(g["segment_overlap_mm"])
    input_end = transformer_start + overlap
    post_start = transformer_end - overlap
    input_l = input_end - feed_start
    post_l = feed_end - post_start
    if input_l <= 0.6 or post_l <= 0.2 or transformer_w >= pitch:
        raise ValueError(f"Invalid fixed-reference section geometry: input={input_l}, post={post_l}, width={transformer_w}")
    port_y = float(g["port_y_mm"])
    if not (feed_start < port_y < transformer_start):
        raise ValueError("Fixed port must remain inside the frozen input section")
    margin = float(g["port_sheet_margin_mm"])
    mesh = float(g["local_mesh_max_length_mm"])
    refine = float(g["adaptive_refinement_percent"])
    max_passes = int(g["maximum_passes"])
    primary_bottom = -primary_w / 2.0
    secondary_bottom = secondary_y - secondary_w / 2.0
    neck_height = secondary_bottom - primary_bottom + 0.02
    left_center = -pitch / 2.0
    right_center = pitch / 2.0
    input_gap = pitch - input_w
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V130", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(g['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(g['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board_x/2:.7f}, {-board_y/2:.7f}, {-h:.7f}, {board_x:.7f}, {board_y:.7f}, {h:.7f}, "RO5880_V130", True

' Negative conductor: frozen input, variable transformer, compensating post, and frozen radiator.
CreateBox oEditor, "InputN", {left_center-input_w/2:.7f}, {feed_start:.7f}, 0, {input_w:.7f}, {input_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "TransformerN", {left_center-transformer_w/2:.7f}, {transformer_start:.7f}, 0, {transformer_w:.7f}, {transformer_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "PostN", {left_center-post_w/2:.7f}, {post_start:.7f}, 0, {post_w:.7f}, {post_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "PrimaryN", {-gap/2-primary_l:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "SecondaryN", {-gap/2-secondary_l:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "NeckN", {-gap/2-neck_l:.7f}, {primary_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}, {copper:.7f}, "copper", False
UniteSelection oEditor, "InputN,TransformerN,PostN,PrimaryN,SecondaryN,NeckN"

' Positive conductor is the exact x mirror; no reference ground is present.
CreateBox oEditor, "InputP", {right_center-input_w/2:.7f}, {feed_start:.7f}, 0, {input_w:.7f}, {input_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "TransformerP", {right_center-transformer_w/2:.7f}, {transformer_start:.7f}, 0, {transformer_w:.7f}, {transformer_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "PostP", {right_center-post_w/2:.7f}, {post_start:.7f}, 0, {post_w:.7f}, {post_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "PrimaryP", {gap/2:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "SecondaryP", {gap/2:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "NeckP", {gap/2:.7f}, {primary_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}, {copper:.7f}, "copper", False
UniteSelection oEditor, "InputP,TransformerP,PostP,PrimaryP,SecondaryP,NeckP"

' The port reference plane and total feed path are invariant across all width/length candidates.
CreateSheetY oEditor, "PortSheet_DIFF", {-input_gap/2-margin:.7f}, {port_y:.7f}, {-margin:.7f}, {input_gap+2*margin:.7f}, {copper+2*margin:.7f}
AssignDifferentialPort oBoundary, "P_DIFF", "PortSheet_DIFF", {-input_gap/2:.7f}, {input_gap/2:.7f}, {port_y:.7f}, {copper/2:.7f}

CreateBox oEditor, "AirRegion", {-board_x/2-12:.7f}, {-board_y/2-12:.7f}, {-h-12:.7f}, {board_x+24:.7f}, {board_y+24:.7f}, {h+copper+24:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:CPSMesh_0p180mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array("InputN", "InputP"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency_ghz:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {max_passes}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {refine:.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V130", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
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
    folder = out / "fixed_reference_cps_1x1" / case_id
    folder.mkdir(parents=True)
    project = folder / f"v130_{case_id}.aedt"
    touchstone = folder / f"v130_{case_id}.s1p"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    source = builder_text(project, geometry, frequency_ghz)
    builder.write_text(source, encoding="ascii")
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
            "differential_port_definition_count": source.count("AssignDifferentialPort oBoundary"),
            "reference_ground_count": source.count('"ReferenceGround"') + source.count('"Ground"'),
            "three_section_pair": all(token in source for token in ('"InputN"', '"TransformerN"', '"PostN"', '"InputP"', '"TransformerP"', '"PostP"')),
            "fixed_reference_statement": "reference plane and total feed path are invariant" in source,
        },
    }
    write_json(folder / "case_manifest.json", case)
    return case


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.30 output: {out}")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    radiator = read_json(config["inputs"]["radiator_screen_summary"])
    if not radiator.get("allow_fixed_reference_transformer") or radiator.get("selected_radiator_candidate") != config["inputs"]["required_selected_radiator_candidate"]:
        raise RuntimeError("The direct-gap radiator gate did not authorize this frozen transformer screen")
    base = config["frozen_geometry"]
    cases = [prepare_case(out, {**base, **candidate}, float(config["frequency_ghz"])) for candidate in config["candidates"]]
    write_json(out / "case_manifest.json", {"cases": cases})
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {"head_commit": head, "tag_commit": tag, "free_memory_gib": memory_available_gb(), "aedt_processes": aedt_processes()},
            "evidence_rules": {
                "radiator_geometry_frozen_from_v129": True,
                "port_reference_plane_fixed": True,
                "total_feed_path_fixed": True,
                "transformer_width_and_length_independent": True,
                "all_nine_candidates_fixed_before_solve": True,
                "array_and_learning_stages_locked": True
            },
        },
    )
    decision = {
        "stage": "A_fixed_reference_cps_preregistered",
        "allow_run": True,
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
        raise RuntimeError("Fixed-reference CPS screen is not authorized")
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
            and cad["three_section_pair"]
            and cad["fixed_reference_statement"]
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
        passed = bool(
            profile.get("converged") is True
            and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
            and topology_warning_count(folder) <= int(gates["maximum_port_topology_warning_count"])
            and passive_rl >= float(gates["minimum_screen_passive_rl_db"])
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "transformer_width_mm": case["geometry"]["transformer_width_mm"],
                "transformer_length_mm": case["geometry"]["transformer_length_mm"],
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
                "screen_gate_pass": passed,
            }
        )
    rows.sort(key=lambda item: (not item["screen_gate_pass"], -float(item["passive_rl_db"])))
    write_csv(out / "candidate_metrics.csv", rows)
    passing = [row for row in rows if row["screen_gate_pass"]]
    preferred = [row for row in rows if row["passive_rl_db"] >= float(gates["preferred_passive_rl_db"])]
    summary = {
        "candidate_count": len(rows),
        "screen_gate_pass_count": len(passing),
        "preferred_15db_count": len(preferred),
        "selected_candidate": rows[0]["case_id"] if passing else None,
        "best_passive_rl_db": rows[0]["passive_rl_db"],
        "allow_three_frequency": bool(passing),
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "B_fixed_reference_cps_screen_complete",
        "allow_run": True,
        "allow_three_frequency": bool(passing),
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "At least one fixed-reference CPS candidate retained 10 dB matching; only three-frequency 1x1 verification is authorized."
            if passing
            else "The independent CPS transformer screen failed at 10 GHz; all array and learning stages remain locked."
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
