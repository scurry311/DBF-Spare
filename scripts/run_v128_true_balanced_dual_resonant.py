#!/usr/bin/env python3
"""Build and gate a true-balanced dual-resonant 1x1 HFSS radiator smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import memory_available_gb, parse_touchstone, profile_metrics
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v125_feedpoint_input_impedance import topology_warning_count, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v128_true_balanced_dual_resonant_preregistered.json"
DESIGN_NAME = "V128_TrueBalanced_DualResonant"
EPS = 1.0e-15


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_config(path: str | Path) -> dict[str, Any]:
    config = read_json(path)
    parent = config.get("extends")
    return merge_dict(read_config(parent), config) if parent else config


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def helpers() -> str:
    return r'''Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(220 150 55)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheetY(editor, objName, x, y, z, width, height)
    ' For a Y-normal rectangle AEDT maps Width to Z and Height to X.
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(height), "Height:=", Mm(width), "WhichAxis:=", "Y"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub UniteSelection(editor, names)
    editor.Unite Array("NAME:Selections", "Selections:=", names), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub AssignDifferentialPort(boundary, portName, sheetName, xNegative, xPositive, y, z)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(xNegative), Mm(y), Mm(z)), "End:=", Array(Mm(xPositive), Mm(y), Mm(z))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function'''


def builder_text(project: Path, geometry: dict[str, Any], frequency_ghz: float) -> str:
    g = geometry
    board_x = float(g["board_x_mm"])
    board_y = float(g["board_y_mm"])
    h = float(g["substrate_thickness_mm"])
    copper = float(g["copper_thickness_mm"])
    input_w = float(g["input_trace_width_mm"])
    pitch = float(g["pair_center_pitch_mm"])
    input_gap = pitch - input_w
    input_y = float(g["input_y_start_mm"])
    input_l = float(g["input_segment_length_mm"])
    transformer_w = float(g["transformer_width_mm"])
    transformer_l = float(g["transformer_length_mm"])
    overlap = float(g["segment_overlap_mm"])
    transformer_y = input_y + input_l - overlap
    primary_gap = float(g["primary_inner_gap_mm"])
    primary_l = float(g["primary_arm_length_mm"])
    primary_w = float(g["primary_arm_width_mm"])
    secondary_l = float(g["secondary_arm_length_mm"])
    secondary_w = float(g["secondary_arm_width_mm"])
    secondary_y = float(g["secondary_arm_offset_y_mm"])
    neck_l = float(g["secondary_neck_length_x_mm"])
    port_y = float(g["port_y_mm"])
    port_margin = float(g.get("port_sheet_margin_mm", 0.05))
    mesh = float(g["local_mesh_max_length_mm"])
    refine = float(g["adaptive_refinement_percent"])
    max_passes = int(g["maximum_passes"])
    transformer_end = transformer_y + transformer_l
    primary_bottom = -primary_w / 2.0
    secondary_bottom = secondary_y - secondary_w / 2.0
    neck_bottom = primary_bottom
    neck_height = secondary_bottom - neck_bottom + 0.02
    right_center = pitch / 2.0
    left_center = -pitch / 2.0
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V128", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(g['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(g['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board_x/2:.7f}, {-board_y/2:.7f}, {-h:.7f}, {board_x:.7f}, {board_y:.7f}, {h:.7f}, "RO5880_V128", True

' Negative conductor: input CPS, independent transformer, primary arm, and secondary resonant arm.
CreateBox oEditor, "InputN", {left_center-input_w/2:.7f}, {input_y:.7f}, 0, {input_w:.7f}, {input_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "TransformerN", {left_center-transformer_w/2:.7f}, {transformer_y:.7f}, 0, {transformer_w:.7f}, {transformer_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "PrimaryN", {-primary_gap/2-primary_l:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "SecondaryN", {-primary_gap/2-secondary_l:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "NeckN", {-primary_gap/2-neck_l:.7f}, {neck_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}, {copper:.7f}, "copper", False
UniteSelection oEditor, "InputN,TransformerN,PrimaryN,SecondaryN,NeckN"

' Positive conductor is the exact x-mirror; no reference ground is present.
CreateBox oEditor, "InputP", {right_center-input_w/2:.7f}, {input_y:.7f}, 0, {input_w:.7f}, {input_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "TransformerP", {right_center-transformer_w/2:.7f}, {transformer_y:.7f}, 0, {transformer_w:.7f}, {transformer_l:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "PrimaryP", {primary_gap/2:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "SecondaryP", {primary_gap/2:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "NeckP", {primary_gap/2:.7f}, {neck_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}, {copper:.7f}, "copper", False
UniteSelection oEditor, "InputP,TransformerP,PrimaryP,SecondaryP,NeckP"

' One two-terminal differential port spans only the CPS gap.
CreateSheetY oEditor, "PortSheet_DIFF", {-input_gap/2-port_margin:.7f}, {port_y:.7f}, {-port_margin:.7f}, {input_gap+2*port_margin:.7f}, {copper+2*port_margin:.7f}
AssignDifferentialPort oBoundary, "P_DIFF", "PortSheet_DIFF", {-input_gap/2:.7f}, {input_gap/2:.7f}, {port_y:.7f}, {copper/2:.7f}

CreateBox oEditor, "AirRegion", {-board_x/2-12:.7f}, {-board_y/2-12:.7f}, {-h-12:.7f}, {board_x+24:.7f}, {board_y+24:.7f}, {h+copper+24:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:BalancedConductorMesh_0p180mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array("InputN", "InputP"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency_ghz:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {max_passes}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {refine:.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V128", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{helpers()}
'''


def solver_text(project: Path, touchstone: Path, efficiency: Path, frequency_ghz: float) -> str:
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
Set oReport = oDesign.GetModule("ReportSetup")
reportName = "V128_RadiationEfficiency"
On Error Resume Next
oReport.DeleteReports Array(reportName)
Err.Clear
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V128"), Array("Freq:=", Array("{frequency_ghz:g}GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(efficiency)}"
On Error GoTo 0
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def prepare_case(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    geometry = config["central_geometry"]
    case_id = str(geometry["candidate_id"])
    folder = output / "center_1x1" / case_id
    folder.mkdir(parents=True)
    project = folder / f"v128_{case_id}.aedt"
    touchstone = folder / f"v128_{case_id}.s1p"
    efficiency = folder / "radiation_efficiency.csv"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    text = builder_text(project, geometry, float(config["frequency_ghz"]))
    builder.write_text(text, encoding="ascii")
    solver.write_text(solver_text(project, touchstone, efficiency, float(config["frequency_ghz"])), encoding="ascii")
    case = {
        "case_id": case_id,
        "geometry": geometry,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "efficiency_path": str(efficiency.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "cad_audit": {
            "differential_port_definition_count": text.count("AssignDifferentialPort oBoundary"),
            "reference_ground_count": text.count('"ReferenceGround"') + text.count('"Ground"'),
            "negative_positive_conductor_pair": all(token in text for token in ('"InputN"', '"InputP"')),
            "independent_transformer_objects": all(token in text for token in ('"TransformerN"', '"TransformerP"')),
            "secondary_resonator_objects": all(token in text for token in ('"SecondaryN"', '"SecondaryP"')),
        },
    }
    write_json(output / "case_manifest.json", case)
    return case


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["output_directory"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.28 output: {output}")
    output.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    case = prepare_case(output, config)
    write_json(
        output / "preregistration.json",
        {
            **config,
            "runtime_audit": {"head_commit": head, "tag_commit": tag, "free_memory_gib": memory_available_gb(), "aedt_processes": aedt_processes()},
            "evidence_rules": {
                "true_two_conductor_port_without_reference_ground": True,
                "transformer_and_secondary_resonance_variables_separate": True,
                "center_gate_precedes_all_doe_and_array_work": True,
            },
        },
    )
    decision = {
        "stage": "A_preregistered",
        "allow_build": True,
        "allow_center_solve": False,
        "allow_jacobian": False,
        "allow_three_frequency": False,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(output / "stage_decision.json", decision)
    return {"output_directory": str(output), "case": case, "decision": decision}


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def build(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["output_directory"])
    decision = read_json(output / "stage_decision.json")
    if not decision.get("allow_build"):
        raise RuntimeError("Build is not authorized")
    require_no_aedt()
    minimum = float(config["resources"]["minimum_free_memory_before_build_gib"])
    free = memory_available_gb()
    if free < minimum:
        raise MemoryError(f"Only {free:.2f} GiB free; build requires {minimum:.2f} GiB")
    case = read_json(output / "case_manifest.json")
    executable = str(resolve(config["ansys_executable"]))
    folder = Path(case["project_path"]).parent
    with (folder / "build.log").open("w", encoding="utf-8") as handle:
        process = subprocess.run([executable, "-RunScriptAndExit", case["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    warnings = topology_warning_count(folder)
    audit = {
        "build_return_code": process.returncode,
        "project_exists": Path(case["project_path"]).exists(),
        "topology_warning_count": warnings,
        **case["cad_audit"],
    }
    gates = config["gates"]
    passed = bool(
        process.returncode == 0
        and audit["project_exists"]
        and warnings <= int(gates["maximum_port_topology_warning_count"])
        and audit["differential_port_definition_count"] == int(gates["required_differential_port_count"])
        and audit["reference_ground_count"] == int(gates["required_reference_ground_count"])
        and audit["negative_positive_conductor_pair"]
        and audit["independent_transformer_objects"]
        and audit["secondary_resonator_objects"]
    )
    audit["build_gate_pass"] = passed
    write_json(output / "build_audit.json", audit)
    decision.update({"stage": "B_build_complete", "allow_center_solve": passed})
    write_json(output / "stage_decision.json", decision)
    return {"audit": audit, "decision": decision}


def run_center(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["output_directory"])
    decision = read_json(output / "stage_decision.json")
    if not decision.get("allow_center_solve"):
        raise RuntimeError("Center solve is not authorized")
    require_no_aedt()
    free = memory_available_gb()
    minimum = float(config["resources"]["minimum_free_memory_before_solve_gib"])
    if free < minimum:
        raise MemoryError(f"Only {free:.2f} GiB free; center solve requires {minimum:.2f} GiB")
    case = read_json(output / "case_manifest.json")
    folder = Path(case["project_path"]).parent
    code, aborted, minimum_free = run_process_with_memory_guard(
        [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", case["solver_path"]],
        folder / "solve_export.log",
        float(config["resources"]["abort_free_memory_during_solve_gib"]),
        float(config["resources"]["poll_interval_seconds"]),
    )
    result = {
        "solve_return_code": code,
        "memory_aborted": aborted,
        "free_memory_gib_before": free,
        "minimum_free_memory_gib": minimum_free,
        "touchstone_exists": Path(case["touchstone_path"]).exists() and Path(case["touchstone_path"]).stat().st_size > 100,
        "efficiency_report_exists": Path(case["efficiency_path"]).exists(),
        "topology_warning_count": topology_warning_count(folder),
    }
    write_json(output / "run_audit.json", result)
    return result


def read_efficiency(path: Path) -> float | None:
    if not path.exists():
        return None
    values: list[float] = []
    with path.open(encoding="utf-8-sig", errors="ignore", newline="") as handle:
        for row in csv.reader(handle):
            for item in row:
                try:
                    value = float(item)
                except ValueError:
                    continue
                if 0.0 <= value <= 1.5:
                    values.append(value)
    return values[-1] if values else None


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["output_directory"])
    case = read_json(output / "case_manifest.json")
    folder = Path(case["project_path"]).parent
    touchstone = Path(case["touchstone_path"])
    frequencies, matrices = parse_touchstone(touchstone, 1)
    index = int(np.argmin(np.abs(frequencies - float(config["frequency_ghz"]))))
    s11 = matrices[index, 0, 0]
    impedance = 50.0 * (1.0 + s11) / (1.0 - s11)
    passive_rl = float(-20.0 * np.log10(max(float(abs(s11)), EPS)))
    efficiency = read_efficiency(Path(case["efficiency_path"]))
    profile = profile_metrics(folder)
    run_audit = read_json(output / "run_audit.json")
    gates = config["gates"]
    passed = bool(
        run_audit["solve_return_code"] == 0
        and not run_audit["memory_aborted"]
        and profile.get("converged") is True
        and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
        and passive_rl >= float(gates["minimum_center_passive_rl_db"])
        and efficiency is not None
        and efficiency >= float(gates["minimum_radiation_efficiency"])
        and run_audit["topology_warning_count"] <= int(gates["maximum_port_topology_warning_count"])
    )
    metrics = {
        "case_id": case["case_id"],
        **profile,
        "frequency_ghz": float(frequencies[index]),
        "s11_real": float(s11.real),
        "s11_imag": float(s11.imag),
        "s11_magnitude": float(abs(s11)),
        "passive_rl_db": passive_rl,
        "input_resistance_ohm": float(impedance.real),
        "input_reactance_ohm": float(impedance.imag),
        "radiation_efficiency": efficiency,
        "topology_warning_count": run_audit["topology_warning_count"],
        "center_gate_pass": passed,
    }
    write_csv(output / "center_metrics.csv", [metrics])
    write_json(output / "center_summary.json", metrics)
    decision = read_json(output / "stage_decision.json")
    decision.update(
        {
            "stage": "C_center_complete",
            "allow_jacobian": passed,
            "allow_three_frequency": passed,
            "allow_2x2": False,
            "allow_4x4": False,
            "allow_16x16": False,
            "allow_eep_export": False,
            "allow_training_labels": False,
            "allow_critic_training": False,
            "reason": (
                "The true-balanced center passed; only the preregistered 1x1 Jacobian and three-frequency verification are authorized."
                if passed
                else "The true-balanced center failed its 1x1 physical gate; all downstream work remains locked."
            ),
        }
    )
    write_json(output / "stage_decision.json", decision)
    return {"metrics": metrics, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["output_directory"])
    return {
        "output_directory": str(output),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(output / "stage_decision.json") if (output / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=str)
    parser.add_argument("--mode", choices=("preregister", "build", "run-center", "analyze", "status"), default="status")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.output_directory:
        config["output_directory"] = args.output_directory
    actions = {"preregister": preregister, "build": build, "run-center": run_center, "analyze": analyze, "status": status}
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
