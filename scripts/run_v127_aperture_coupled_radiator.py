#!/usr/bin/env python3
"""Build and test the v1.27 aperture-coupled dual-slot radiator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import (
    efficiency_from_csv,
    memory_available_gb,
    parse_touchstone,
    profile_metrics,
)
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v125_feedpoint_input_impedance import topology_warning_count, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v127_aperture_coupled_radiator_preregistered.json"
DESIGN_NAME = "V127_ApertureCoupled_DualSlot"
EPS = 1.0e-15


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def vbs_helpers() -> str:
    return r'''Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(170 130 70)", "Transparency:=", 0.2, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheetY(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Y"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.1, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.1, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateCylinderZ(editor, objName, x, y, z, radius, height, material, solveInside)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", "24"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(210 150 50)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub SubtractObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", False)
End Sub
Sub SubtractKeepObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", True)
End Sub
Sub UniteSelection(editor, selectionText)
    editor.Unite Array("NAME:Selections", "Selections:=", selectionText), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub AssignPort(boundary, portName, sheetName, x, y, z1, z2)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x), Mm(y), Mm(z1)), "End:=", Array(Mm(x), Mm(y), Mm(z2))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Sub AssignHorizontalPort(boundary, portName, sheetName, x1, x2, y, z)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x1), Mm(y), Mm(z)), "End:=", Array(Mm(x2), Mm(y), Mm(z))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function
Function Pad3(value)
    Pad3 = Right("000" & CStr(value), 3)
End Function'''


def builder_text(project: Path, side: int, geometry: dict[str, Any], frequency_ghz: float) -> str:
    spacing = float(geometry["spacing_mm"])
    top_h = float(geometry["top_substrate_thickness_mm"])
    feed_h = float(geometry["feed_substrate_thickness_mm"])
    copper = float(geometry["copper_thickness_mm"])
    patch_w = float(geometry["patch_width_mm"])
    patch_l = float(geometry["patch_length_mm"])
    slot_l = float(geometry["dual_slot_length_mm"])
    slot_w = float(geometry["dual_slot_width_mm"])
    slot_sep = float(geometry["dual_slot_center_separation_mm"])
    aperture_l = float(geometry["coupling_aperture_length_mm"])
    aperture_w = float(geometry["coupling_aperture_width_mm"])
    aperture_y = float(geometry["coupling_aperture_y_offset_mm"])
    line_w = float(geometry["feed_line_width_mm"])
    line_start = float(geometry["feed_line_start_offset_mm"])
    stub_l = float(geometry["open_stub_length_mm"])
    port_margin = float(geometry.get("port_sheet_margin_mm", 0.1))
    launch_gap = float(geometry.get("launch_gap_mm", 0.2))
    ground_pad_w = float(geometry.get("launch_ground_pad_width_mm", 1.0))
    ground_pad_l = float(geometry.get("launch_ground_pad_length_mm", 0.8))
    launch_via_r = float(geometry.get("launch_via_radius_mm", 0.2))
    launch_port_margin = float(geometry.get("launch_port_margin_mm", 0.05))
    mesh = float(geometry["local_mesh_max_length_mm"])
    refine = float(geometry["adaptive_refinement_percent"])
    board = max(15.0, (side - 1) * spacing + 15.0)
    ground_z = -top_h - copper
    feed_bottom = ground_z - feed_h
    line_z = feed_bottom - copper
    line_length = -line_start + aperture_y + stub_l
    mesh_names = ", ".join(
        name
        for index in range(side * side)
        for name in (
            f'"TopSlotMesh_{index:03d}"',
            f'"TopApertureMesh_{index:03d}"',
            f'"BottomApertureMesh_{index:03d}"',
            f'"LaunchMesh_{index:03d}"',
        )
    )
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Dim ix, iy, idx, xc, yc, patchBottom, slotOffset, lineStartY, apertureY, nameBase
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V127", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(geometry['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(geometry['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "TopSubstrate", {-board/2:.7f}, {-board/2:.7f}, {-top_h:.7f}, {board:.7f}, {board:.7f}, {top_h:.7f}, "RO5880_V127", True
CreateBox oEditor, "Ground", {-board/2:.7f}, {-board/2:.7f}, {ground_z:.7f}, {board:.7f}, {board:.7f}, {copper:.7f}, "copper", False
CreateBox oEditor, "FeedSubstrate", {-board/2:.7f}, {-board/2:.7f}, {feed_bottom:.7f}, {board:.7f}, {board:.7f}, {feed_h:.7f}, "RO5880_V127", True
idx = 0
For ix = 0 To {side - 1}
    For iy = 0 To {side - 1}
        xc = (ix - 0.5 * ({side - 1})) * {spacing:.7f}
        yc = (iy - 0.5 * ({side - 1})) * {spacing:.7f}
        patchBottom = yc - {patch_l/2:.7f}
        slotOffset = {slot_sep/2:.7f}
        apertureY = yc + {aperture_y:.7f}
        lineStartY = yc + {line_start:.7f}
        nameBase = Pad3(idx)
        CreateBox oEditor, "TopSlotMesh_" & nameBase, xc - {slot_sep/2+slot_w/2+0.6:.7f}, patchBottom - 0.25, {-top_h:.7f}, {2*(slot_sep/2+slot_w/2+0.6):.7f}, {slot_l+0.5:.7f}, {top_h:.7f}, "RO5880_V127", True
        SubtractKeepObject oEditor, "TopSubstrate", "TopSlotMesh_" & nameBase
        CreateBox oEditor, "TopApertureMesh_" & nameBase, xc - {aperture_l/2+0.5:.7f}, apertureY - {aperture_w/2+0.6:.7f}, {-top_h:.7f}, {aperture_l+1.0:.7f}, {aperture_w+1.2:.7f}, {top_h:.7f}, "RO5880_V127", True
        SubtractKeepObject oEditor, "TopSubstrate", "TopApertureMesh_" & nameBase
        SubtractKeepObject oEditor, "TopSlotMesh_" & nameBase, "TopApertureMesh_" & nameBase
        CreateBox oEditor, "BottomApertureMesh_" & nameBase, xc - {aperture_l/2+0.5:.7f}, apertureY - {aperture_w/2+0.6:.7f}, {feed_bottom:.7f}, {aperture_l+1.0:.7f}, {aperture_w+1.2:.7f}, {feed_h:.7f}, "RO5880_V127", True
        SubtractKeepObject oEditor, "FeedSubstrate", "BottomApertureMesh_" & nameBase
        CreateBox oEditor, "LaunchMesh_" & nameBase, xc - {line_w/2+0.4:.7f}, lineStartY - 0.3, {feed_bottom:.7f}, {line_w+launch_gap+ground_pad_w+0.8:.7f}, {ground_pad_l+0.6:.7f}, {feed_h:.7f}, "RO5880_V127", True
        SubtractKeepObject oEditor, "FeedSubstrate", "LaunchMesh_" & nameBase
        CreateBox oEditor, "Patch_" & nameBase, xc - {patch_w/2:.7f}, patchBottom, 0, {patch_w:.7f}, {patch_l:.7f}, {copper:.7f}, "copper", False
        CreateBox oEditor, "DualSlotL_" & nameBase, xc - slotOffset - {slot_w/2:.7f}, patchBottom - 0.01, -0.01, {slot_w:.7f}, {slot_l+0.02:.7f}, {copper+0.02:.7f}, "vacuum", True
        SubtractObject oEditor, "Patch_" & nameBase, "DualSlotL_" & nameBase
        CreateBox oEditor, "DualSlotR_" & nameBase, xc + slotOffset - {slot_w/2:.7f}, patchBottom - 0.01, -0.01, {slot_w:.7f}, {slot_l+0.02:.7f}, {copper+0.02:.7f}, "vacuum", True
        SubtractObject oEditor, "Patch_" & nameBase, "DualSlotR_" & nameBase
        CreateBox oEditor, "Aperture_" & nameBase, xc - {aperture_l/2:.7f}, apertureY - {aperture_w/2:.7f}, {ground_z-0.01:.7f}, {aperture_l:.7f}, {aperture_w:.7f}, {copper+0.02:.7f}, "vacuum", True
        SubtractObject oEditor, "Ground", "Aperture_" & nameBase
        CreateBox oEditor, "FeedLine_" & nameBase, xc - {line_w/2:.7f}, lineStartY, {line_z:.7f}, {line_w:.7f}, {line_length:.7f}, {copper:.7f}, "copper", False
        CreateBox oEditor, "GroundPad_" & nameBase, xc + {line_w/2+launch_gap:.7f}, lineStartY, {line_z:.7f}, {ground_pad_w:.7f}, {ground_pad_l:.7f}, {copper:.7f}, "copper", False
        CreateCylinderZ oEditor, "GroundVia_" & nameBase, xc + {line_w/2+launch_gap+ground_pad_w/2:.7f}, lineStartY + {ground_pad_l/2:.7f}, {line_z:.7f}, {launch_via_r:.7f}, {-top_h-line_z:.7f}, "copper", False
        SubtractKeepObject oEditor, "FeedSubstrate", "GroundVia_" & nameBase
        SubtractKeepObject oEditor, "LaunchMesh_" & nameBase, "GroundVia_" & nameBase
        UniteSelection oEditor, "Ground,GroundPad_" & nameBase & ",GroundVia_" & nameBase
        CreateSheetZ oEditor, "PortSheet_" & nameBase, xc + {line_w/2-launch_port_margin:.7f}, lineStartY + {ground_pad_l/2-0.2:.7f}, {feed_bottom:.7f}, {launch_gap+2*launch_port_margin:.7f}, 0.4000000
        AssignHorizontalPort oBoundary, "P" & nameBase, "PortSheet_" & nameBase, xc + {line_w/2:.7f}, xc + {line_w/2+launch_gap:.7f}, lineStartY + {ground_pad_l/2:.7f}, {feed_bottom:.7f}
        idx = idx + 1
    Next
Next
CreateBox oEditor, "AirRegion", {-board/2-15:.7f}, {-board/2-15:.7f}, {line_z-8:.7f}, {board+30:.7f}, {board+30:.7f}, {top_h+feed_h+copper*2+23:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:JointInputMesh_0p180mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({mesh_names}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency_ghz:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", 24, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {refine:.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V127", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{vbs_helpers()}
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
reportName = "V127_RadiationEfficiency"
On Error Resume Next
oReport.DeleteReports Array(reportName)
Err.Clear
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V127"), Array("Freq:=", Array("{frequency_ghz:g}GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(efficiency)}"
On Error GoTo 0
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def prepare_case(out: Path, geometry: dict[str, Any], frequency_ghz: float) -> dict[str, Any]:
    case_id = str(geometry["candidate_id"])
    folder = out / "1x1" / case_id
    folder.mkdir(parents=True)
    project = folder / f"v127_1x1_{case_id}.aedt"
    touchstone = folder / f"v127_1x1_{case_id}.s1p"
    efficiency = folder / "radiation_efficiency.csv"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    builder.write_text(builder_text(project, 1, geometry, frequency_ghz), encoding="ascii")
    solver.write_text(solver_text(project, touchstone, efficiency, frequency_ghz), encoding="ascii")
    case = {
        "case_id": case_id,
        "side": 1,
        "frequency_ghz": frequency_ghz,
        "geometry": geometry,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "efficiency_csv_path": str(efficiency.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "evidence_scope": "physical aperture-coupled dual-slot patch; center 1x1 smoke only",
    }
    write_json(folder / "case_manifest.json", case)
    return case


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.27 output: {out}")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    input_rows = []
    for key in ("v126_physical_s4", "v126_diagonal_bound"):
        path = resolve(config["inputs"][key])
        observed = sha256_file(path)
        expected = config["inputs"][f"{key}_sha256"]
        if observed != expected:
            raise RuntimeError(f"Input hash mismatch for {key}")
        input_rows.append({"role": key, "path": str(path), "sha256": observed, "size_bytes": path.stat().st_size})
    case = prepare_case(out, config["central_geometry"], float(config["frequency_ghz"]))
    write_json(out / "center_case_manifest.json", {"case": case})
    write_json(out / "frozen_input_manifest.json", {"inputs": input_rows})
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {
                "head_commit": head,
                "tag_commit": tag,
                "free_memory_gib": memory_available_gb(),
                "aedt_processes": aedt_processes(),
            },
            "evidence_rules": {
                "center_gate_before_doe": True,
                "one_by_one_gate_before_s4_jacobian": True,
                "sii_and_nearest_sij_are_joint_targets": True,
                "no_existing_bridge_elements": True,
            },
        },
    )
    decision = {
        "stage": "A_aperture_center_preregistered",
        "allow_center_build": True,
        "allow_center_run": False,
        "allow_joint_1x1_doe": False,
        "allow_2x2_jacobian": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "case": case, "decision": decision}


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def build(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_center_build"):
        raise RuntimeError("Center build is not authorized")
    require_no_aedt()
    case = read_json(out / "center_case_manifest.json")["case"]
    folder = Path(case["project_path"]).parent
    with (folder / "build.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(resolve(config["ansys_executable"])), "-RunScriptAndExit", case["builder_path"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    warnings = topology_warning_count(folder)
    passed = result.returncode == 0 and Path(case["project_path"]).exists() and warnings == 0
    audit = {
        "build_return_code": result.returncode,
        "project_exists": Path(case["project_path"]).exists(),
        "topology_warning_count": warnings,
        "build_gate_pass": passed,
    }
    write_json(out / "center_build_audit.json", audit)
    decision = {
        **decision,
        "stage": "B_aperture_center_build_complete",
        "allow_center_build": False,
        "allow_center_run": passed,
    }
    write_json(out / "stage_decision.json", decision)
    if not passed:
        raise RuntimeError(f"Center build gate failed: {audit}")
    return {"audit": audit, "decision": decision}


def run(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_center_run"):
        raise RuntimeError("Center solve is not authorized")
    require_no_aedt()
    free = memory_available_gb()
    required = float(config["resources"]["minimum_free_memory_before_1x1_gib"])
    if free < required:
        raise MemoryError(f"Only {free:.2f} GiB free; {required:.2f} GiB required")
    case = read_json(out / "center_case_manifest.json")["case"]
    folder = Path(case["project_path"]).parent
    code, aborted, minimum_free = run_process_with_memory_guard(
        [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", case["solver_path"]],
        folder / "solve_export.log",
        float(config["resources"]["abort_free_memory_during_solve_gib"]),
        float(config["resources"]["poll_interval_seconds"]),
    )
    touchstone = Path(case["touchstone_path"])
    result = {
        "solve_return_code": code,
        "memory_aborted": aborted,
        "free_memory_gib_before": free,
        "minimum_free_memory_gib": minimum_free,
        "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
        "topology_warning_count": topology_warning_count(folder),
    }
    write_json(out / "center_run_audit.json", result)
    if code != 0 or aborted or not result["touchstone_exists"] or result["topology_warning_count"] > 0:
        raise RuntimeError(f"Center solve gate failed: {result}")
    decision["stage"] = "C_aperture_center_solve_complete"
    decision["allow_center_run"] = False
    write_json(out / "stage_decision.json", decision)
    return {"run": result, "decision": decision}


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    case = read_json(out / "center_case_manifest.json")["case"]
    folder = Path(case["touchstone_path"]).parent
    frequencies, matrices = parse_touchstone(Path(case["touchstone_path"]), 1)
    target = float(case["frequency_ghz"])
    index = int(np.argmin(np.abs(frequencies - target)))
    matrix = matrices[index]
    passive_rl = float(-20.0 * np.log10(max(float(abs(matrix[0, 0])), EPS)))
    efficiency = efficiency_from_csv(Path(case["efficiency_csv_path"]))
    profile = profile_metrics(folder)
    gates = config["gates"]
    passed = bool(
        profile.get("converged") is True
        and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
        and passive_rl >= float(gates["minimum_center_passive_rl_db"])
        and efficiency is not None
        and float(efficiency) >= float(gates["minimum_radiation_efficiency"])
        and topology_warning_count(folder) <= int(gates["maximum_port_topology_warning_count"])
    )
    summary = {
        **profile,
        "frequency_ghz": float(frequencies[index]),
        "passive_rl_db": passive_rl,
        "passivity_sigma": float(abs(matrix[0, 0])),
        "radiation_efficiency": efficiency,
        "topology_warning_count": topology_warning_count(folder),
        "center_gate_pass": passed,
    }
    write_json(out / "center_stage_summary.json", summary)
    decision = {
        "stage": "D_aperture_center_gate_complete",
        "center_gate_pass": passed,
        "allow_joint_1x1_doe": passed,
        "allow_2x2_jacobian": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "Center aperture-coupled radiator is numerically and physically credible; a small joint 1x1 DOE is authorized."
            if passed
            else "Center aperture-coupled radiator failed the 1x1 physical gate; 2x2 and all downstream work remain locked."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "decision": decision}


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
    parser.add_argument("--output-directory", type=str)
    parser.add_argument("--mode", choices=("preregister", "build", "run", "analyze", "status"), default="status")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    if args.output_directory:
        config["output_directory"] = args.output_directory
    actions = {"preregister": preregister, "build": build, "run": run, "analyze": analyze, "status": status}
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
