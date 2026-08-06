#!/usr/bin/env python3
"""Screen a short true-differential via pair with independent capacitive pads."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import memory_available_gb, parse_touchstone, profile_metrics
from run_v121_parametric_feed_post import aedt_processes
from run_v125_feedpoint_input_impedance import topology_warning_count, write_csv, write_json
from run_v128_true_balanced_dual_resonant import vp
from run_v130_fixed_reference_cps_transformer import read_json, resolve, run, status


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v132_vertical_differential_launch_preregistered.json"
DESIGN_NAME = "V132_VerticalDifferential"
EPS = 1.0e-15


def helpers() -> str:
    return r'''Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(220 150 55)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateVia(editor, objName, x, y, z, radius, height, numSides)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", CStr(numSides)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """copper""", "SolveInside:=", False)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateMetalSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub UniteSelection(editor, names)
    editor.Unite Array("NAME:Selections", "Selections:=", names), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub SubtractKeepObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", True)
End Sub
Sub AssignDifferentialPortZ(boundary, portName, sheetName, xNegative, xPositive, y, z)
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
    gap = float(g["primary_inner_gap_mm"])
    primary_l = float(g["primary_arm_length_mm"])
    primary_w = float(g["primary_arm_width_mm"])
    secondary_l = float(g["secondary_arm_length_mm"])
    secondary_w = float(g["secondary_arm_width_mm"])
    secondary_y = float(g["secondary_arm_offset_y_mm"])
    neck_l = float(g["secondary_neck_length_x_mm"])
    neck_overlap = float(g.get("secondary_neck_overlap_mm", 0.02))
    pitch = float(g["via_pair_pitch_mm"])
    via_radius = float(g["via_radius_mm"])
    via_num_sides = int(g.get("via_num_sides", 24))
    vertical_shape = str(g.get("vertical_conductor_shape", "round_via"))
    finite_sheet = str(g.get("planar_conductor_representation", "volume_copper")) == "finite_conductivity_sheet"
    partitioned_substrate = str(g.get("substrate_hole_representation", "boolean_subtract")) == "partitioned_boxes"
    pad_w = float(g["bottom_pad_width_mm"])
    pad_y = float(g["bottom_pad_length_y_mm"])
    pad_gap = pitch - pad_w
    if pad_gap < 0.08 or via_radius * 2.0 > pad_w or pitch / 2.0 - via_radius < gap / 2.0 - 1.0e-9:
        raise ValueError(f"Invalid differential via/pad geometry: gap={pad_gap}, pitch={pitch}, radius={via_radius}")
    margin = float(g["port_sheet_margin_mm"])
    mesh = float(g["local_mesh_max_length_mm"])
    refine = float(g["adaptive_refinement_percent"])
    max_passes = int(g["maximum_passes"])
    primary_bottom = -primary_w / 2.0
    secondary_bottom = secondary_y - secondary_w / 2.0
    neck_height = secondary_bottom - primary_bottom + neck_overlap
    left_center = -pitch / 2.0
    right_center = pitch / 2.0
    bottom_z = -h if finite_sheet else -h - copper
    via_height = h if finite_sheet else h + 2.0 * copper
    post_width = 2.0 * via_radius
    if partitioned_substrate:
        if vertical_shape != "square_post":
            raise ValueError("Partitioned substrate holes require square vertical posts")
        board_left = -board_x / 2.0
        board_bottom = -board_y / 2.0
        post_half = post_width / 2.0
        left_hole_min = left_center - post_half
        left_hole_max = left_center + post_half
        right_hole_min = right_center - post_half
        right_hole_max = right_center + post_half
        substrate_creation = f'''CreateBox oEditor, "Substrate", {board_left:.7f}, {board_bottom:.7f}, {-h:.7f}, {board_x:.7f}, {board_y/2-post_half:.7f}, {h:.7f}, "RO5880_V132", True
CreateBox oEditor, "SubstrateTop", {board_left:.7f}, {post_half:.7f}, {-h:.7f}, {board_x:.7f}, {board_y/2-post_half:.7f}, {h:.7f}, "RO5880_V132", True
CreateBox oEditor, "SubstrateRowLeft", {board_left:.7f}, {-post_half:.7f}, {-h:.7f}, {left_hole_min-board_left:.7f}, {post_width:.7f}, {h:.7f}, "RO5880_V132", True
CreateBox oEditor, "SubstrateRowCenter", {left_hole_max:.7f}, {-post_half:.7f}, {-h:.7f}, {right_hole_min-left_hole_max:.7f}, {post_width:.7f}, {h:.7f}, "RO5880_V132", True
CreateBox oEditor, "SubstrateRowRight", {right_hole_max:.7f}, {-post_half:.7f}, {-h:.7f}, {board_x/2-right_hole_max:.7f}, {post_width:.7f}, {h:.7f}, "RO5880_V132", True
UniteSelection oEditor, "Substrate,SubstrateTop,SubstrateRowLeft,SubstrateRowCenter,SubstrateRowRight"'''
        clear_negative_post = ""
        clear_positive_post = ""
    else:
        substrate_creation = f'CreateBox oEditor, "Substrate", {-board_x/2:.7f}, {-board_y/2:.7f}, {-h:.7f}, {board_x:.7f}, {board_y:.7f}, {h:.7f}, "RO5880_V132", True'
        clear_negative_post = 'SubtractKeepObject oEditor, "Substrate", "ViaN"'
        clear_positive_post = 'SubtractKeepObject oEditor, "Substrate", "ViaP"'
    if vertical_shape == "square_post":
        create_negative_vertical = f'CreateBox oEditor, "ViaN", {left_center-post_width/2:.7f}, {-post_width/2:.7f}, {bottom_z:.7f}, {post_width:.7f}, {post_width:.7f}, {via_height:.7f}, "copper", False'
        create_positive_vertical = f'CreateBox oEditor, "ViaP", {right_center-post_width/2:.7f}, {-post_width/2:.7f}, {bottom_z:.7f}, {post_width:.7f}, {post_width:.7f}, {via_height:.7f}, "copper", False'
    elif vertical_shape == "round_via":
        create_negative_vertical = f'CreateVia oEditor, "ViaN", {left_center:.7f}, 0, {bottom_z:.7f}, {via_radius:.7f}, {via_height:.7f}, {via_num_sides}'
        create_positive_vertical = f'CreateVia oEditor, "ViaP", {right_center:.7f}, 0, {bottom_z:.7f}, {via_radius:.7f}, {via_height:.7f}, {via_num_sides}'
    else:
        raise ValueError(f"Unsupported vertical conductor shape: {vertical_shape}")
    if finite_sheet:
        negative_conductor = f'''CreateMetalSheetZ oEditor, "PrimaryN", {-gap/2-primary_l:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}
CreateMetalSheetZ oEditor, "SecondaryN", {-gap/2-secondary_l:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}
CreateMetalSheetZ oEditor, "NeckN", {-gap/2-neck_l:.7f}, {primary_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}
UniteSelection oEditor, "PrimaryN,SecondaryN,NeckN"
{create_negative_vertical}
CreateMetalSheetZ oEditor, "PadN", {left_center-pad_w/2:.7f}, {-pad_y/2:.7f}, {bottom_z:.7f}, {pad_w:.7f}, {pad_y:.7f}
{clear_negative_post}'''
        positive_conductor = f'''CreateMetalSheetZ oEditor, "PrimaryP", {gap/2:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}
CreateMetalSheetZ oEditor, "SecondaryP", {gap/2:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}
CreateMetalSheetZ oEditor, "NeckP", {gap/2:.7f}, {primary_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}
UniteSelection oEditor, "PrimaryP,SecondaryP,NeckP"
{create_positive_vertical}
CreateMetalSheetZ oEditor, "PadP", {right_center-pad_w/2:.7f}, {-pad_y/2:.7f}, {bottom_z:.7f}, {pad_w:.7f}, {pad_y:.7f}
{clear_positive_post}'''
        finite_conductivity = f'oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", "Objects:=", Array("PrimaryN", "PrimaryP", "PadN", "PadP"), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)'
    else:
        negative_conductor = f'''CreateBox oEditor, "PrimaryN", {-gap/2-primary_l:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "SecondaryN", {-gap/2-secondary_l:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "NeckN", {-gap/2-neck_l:.7f}, {primary_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}, {copper:.7f}, "copper", False
{create_negative_vertical}
CreateBox oEditor, "PadN", {left_center-pad_w/2:.7f}, {-pad_y/2:.7f}, {bottom_z:.7f}, {pad_w:.7f}, {pad_y:.7f}, {copper:.7f}, "copper", False
{clear_negative_post}
UniteSelection oEditor, "PrimaryN,SecondaryN,NeckN,ViaN,PadN"'''
        positive_conductor = f'''CreateBox oEditor, "PrimaryP", {gap/2:.7f}, {primary_bottom:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "SecondaryP", {gap/2:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "NeckP", {gap/2:.7f}, {primary_bottom:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}, {copper:.7f}, "copper", False
{create_positive_vertical}
CreateBox oEditor, "PadP", {right_center-pad_w/2:.7f}, {-pad_y/2:.7f}, {bottom_z:.7f}, {pad_w:.7f}, {pad_y:.7f}, {copper:.7f}, "copper", False
{clear_positive_post}
UniteSelection oEditor, "PrimaryP,SecondaryP,NeckP,ViaP,PadP"'''
        finite_conductivity = ""
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V132", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(g['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(g['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
{substrate_creation}

' Negative conductor: frozen fork, one short via, and one bottom capacitive launch pad.
{negative_conductor}

' Positive conductor is the exact x mirror; no common reference ground exists.
{positive_conductor}
{finite_conductivity}

' The bottom reference plane is fixed; pad gap/area and via radius are independent matching controls.
CreateSheetZ oEditor, "PortSheet_DIFF", {-pad_gap/2-margin:.7f}, {-pad_y/2:.7f}, {bottom_z:.7f}, {pad_gap+2*margin:.7f}, {pad_y:.7f}
AssignDifferentialPortZ oBoundary, "P_DIFF", "PortSheet_DIFF", {-pad_gap/2:.7f}, {pad_gap/2:.7f}, 0, {bottom_z:.7f}

CreateBox oEditor, "AirRegion", {-board_x/2-12:.7f}, {-board_y/2-12:.7f}, {bottom_z-12:.7f}, {board_x+24:.7f}, {board_y+24:.7f}, {h+2*copper+24:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:ViaPadMesh_0p100mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array("PrimaryN", "PrimaryP"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency_ghz:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {max_passes}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {refine:.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V132", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
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
    folder = out / "vertical_diff_1x1" / case_id
    folder.mkdir(parents=True)
    project = folder / f"v132_{case_id}.aedt"
    touchstone = folder / f"v132_{case_id}.s1p"
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
            "differential_port_definition_count": source.count("AssignDifferentialPortZ oBoundary"),
            "reference_ground_count": source.count('"ReferenceGround"') + source.count('"Ground"'),
            "vertical_via_pair": all(token in source for token in ('"ViaN"', '"ViaP"')),
            "independent_bottom_pad_pair": all(token in source for token in ('"PadN"', '"PadP"')),
            "fixed_bottom_reference_plane": "bottom reference plane is fixed" in source,
            "three_section_pair": all(token in source for token in ('"ViaN"', '"ViaP"', '"PadN"', '"PadP"')),
            "fixed_reference_statement": "bottom reference plane is fixed" in source,
        },
    }
    write_json(folder / "case_manifest.json", case)
    return case


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.32 output: {out}")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    direct = read_json(config["inputs"]["selected_direct_radiator"])
    stopped = read_json(config["inputs"]["stopped_planar_cps"])
    if direct.get("selected_radiator_candidate") != "direct_d1_both_short" or stopped.get("allow_three_frequency"):
        raise RuntimeError("v1.29/v1.31 decisions do not authorize the vertical differential branch")
    base = config["frozen_geometry"]
    cases = [prepare_case(out, {**base, **candidate}, float(config["frequency_ghz"])) for candidate in config["candidates"]]
    write_json(out / "case_manifest.json", {"cases": cases})
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {"head_commit": head, "tag_commit": tag, "free_memory_gib": memory_available_gb(), "aedt_processes": aedt_processes()},
            "evidence_rules": {
                "radiator_frozen_from_v129": True,
                "planar_cps_stopped_after_v130_v131": True,
                "true_vertical_differential_pair_without_ground": True,
                "via_inductance_and_pad_capacitance_independent": True,
                "all_six_candidates_fixed_before_solve": True,
                "array_and_learning_stages_locked": True
            },
        },
    )
    decision = {
        "stage": "A_vertical_differential_preregistered",
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
            and int(profile.get("small_mesh_segment_count") or 0) <= int(gates.get("maximum_small_mesh_segment_count", 10**9))
            and topology_warning_count(folder) <= int(gates["maximum_port_topology_warning_count"])
            and passive_rl >= float(gates["minimum_screen_passive_rl_db"])
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "via_pair_pitch_mm": case["geometry"]["via_pair_pitch_mm"],
                "via_radius_mm": case["geometry"]["via_radius_mm"],
                "bottom_pad_width_mm": case["geometry"]["bottom_pad_width_mm"],
                "bottom_pad_length_y_mm": case["geometry"]["bottom_pad_length_y_mm"],
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
        "stage": "B_vertical_differential_complete",
        "allow_run": True,
        "allow_three_frequency": bool(passing),
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "The short vertical differential launch passed 10 dB; only three-frequency 1x1 verification is authorized."
            if passing
            else "The vertical differential launch failed at 10 GHz; all downstream work remains locked."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": rows, "decision": decision}


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
