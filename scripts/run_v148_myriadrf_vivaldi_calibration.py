#!/usr/bin/env python3
"""Build and gate a PCB-derived MyriadRF Vivaldi HFSS calibration fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
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
from run_v125_feedpoint_input_impedance import topology_warning_count


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v148_myriadrf_vivaldi_calibration_preregistered.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def balanced_blocks(text: str, token: str) -> list[str]:
    blocks: list[str] = []
    cursor = 0
    while True:
        start = text.find(token, cursor)
        if start < 0:
            return blocks
        depth = 0
        end = None
        for index in range(start, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise ValueError(f"Unbalanced KiCad block beginning at {start}")
        blocks.append(text[start:end])
        cursor = end


def parse_board(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="strict")
    edges = [
        tuple(map(float, match))
        for match in re.findall(
            r"\(gr_line \(start ([\d.\-]+) ([\d.\-]+)\) "
            r"\(end ([\d.\-]+) ([\d.\-]+)\).*?\(layer Edge\.Cuts\)",
            text,
        )
    ]
    points = [(x1, y1) for x1, y1, _, _ in edges] + [(x2, y2) for _, _, x2, y2 in edges]
    x0 = min(x for x, _ in points)
    y0 = min(y for _, y in points)
    width = max(x for x, _ in points) - x0
    height = max(y for _, y in points) - y0
    zones = []
    for index, block in enumerate(balanced_blocks(text, "(zone ")):
        layer_match = re.search(r"\(layer ([FB]\.Cu)\)", block)
        filled = balanced_blocks(block, "(filled_polygon")
        if layer_match is None or len(filled) != 1:
            raise ValueError(f"Zone {index} does not have one filled polygon")
        vertices = [
            (float(x) - x0, float(y) - y0)
            for x, y in re.findall(r"\(xy\s+([\d.\-]+)\s+([\d.\-]+)\)", filled[0])
        ]
        if len(vertices) < 3:
            raise ValueError(f"Zone {index} has fewer than three vertices")
        zones.append({"index": index, "layer": layer_match.group(1), "vertices_mm": vertices})
    vias = [
        {
            "x_mm": float(x) - x0,
            "y_mm": float(y) - y0,
            "outer_diameter_mm": float(size),
            "drill_diameter_mm": float(drill),
        }
        for x, y, size, drill in re.findall(
            r"\(via \(at ([\d.\-]+) ([\d.\-]+)\) \(size ([\d.\-]+)\) "
            r"\(drill ([\d.\-]+)\)",
            text,
        )
    ]
    thickness_match = re.search(r"\(thickness ([\d.\-]+)\)", text)
    return {
        "source_path": str(path),
        "source_sha256": sha256(path),
        "edge_origin_mm": [x0, y0],
        "board_width_mm": width,
        "board_height_mm": height,
        "kicad_general_thickness_mm": float(thickness_match.group(1)) if thickness_match else None,
        "zones": zones,
        "vias": vias,
        "sma_module_count": len(re.findall(r"\(module SMA_H", text)),
        "model_limitations": [
            "The report-specified 0.5 mm RO4350B thickness overrides the stale 1.6 mm KiCad general field.",
            "The SMA body is not included; the HFSS reference plane is frozen at the center-pad inner board edge.",
            "Copper thickness is not specified by the public PCB archive and is frozen as a preregistered assumption.",
        ],
    }


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def solver_suffix(solver_type: str) -> str:
    if solver_type == "direct":
        return ', "DrivenSolverType:=", "Direct Solver"'
    if solver_type == "ddm":
        return ', "DrivenSolverType:=", "Domain Decomposition", "IterativeResidual:=", 0.000001, "DDMSolverResidual:=", 0.000001'
    raise ValueError(f"Unknown solver type: {solver_type}")


def polygon_vbs(name: str, vertices: list[tuple[float, float]], z: float) -> str:
    point_rows = [
        f'Array("NAME:PLPoint", "X:=", Mm({x:.7f}), "Y:=", Mm({y:.7f}), "Z:=", Mm({z:.7f}))'
        for x, y in vertices
    ]
    segment_rows = [
        f'Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", {index}, "NoOfPoints:=", 2)'
        for index in range(len(vertices) - 1)
    ]
    return (
        "oEditor.CreatePolyline Array( _\n"
        "    \"NAME:PolylineParameters\", \"IsPolylineCovered:=\", True, \"IsPolylineClosed:=\", True, _\n"
        "    Array(\"NAME:PolylinePoints\", _\n        " + ", _\n        ".join(point_rows) + "), _\n"
        "    Array(\"NAME:PolylineSegments\", _\n        " + ", _\n        ".join(segment_rows) + "), _\n"
        "    Array(\"NAME:PolylineXSection\", \"XSectionType:=\", \"None\", \"XSectionOrient:=\", \"Auto\", "
        "\"XSectionWidth:=\", \"0mm\", \"XSectionTopWidth:=\", \"0mm\", \"XSectionHeight:=\", \"0mm\", "
        "\"XSectionNumSegments:=\", \"0\", \"XSectionBendType:=\", \"Corner\")), _\n"
        f'    Array("NAME:Attributes", "Name:=", "{name}", "Flags:=", "", "Color:=", "(230 160 60)", '
        '"Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)'
    )


VBS_HELPERS = r'''
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function

Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(120 150 170)", "Transparency:=", 0.72, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub

Sub CreateCylinderZ(editor, objName, x, y, z, radius, height, material, solveInside)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", "0"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub

Sub CreateSheetX(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "X"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub

Sub SubtractObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", False)
End Sub
'''


def builder_text(project: Path, config: dict[str, Any], board: dict[str, Any], solver_type: str) -> str:
    design = config["design"]
    sweep = config["sweep"]
    width = float(board["board_width_mm"])
    height = float(board["board_height_mm"])
    thickness = float(design["substrate_thickness_mm"])
    copper = float(design["copper_thickness_mm"])
    polygons = []
    conductor_names = []
    for zone in board["zones"]:
        name = f"{zone['layer'].replace('.', '')}_Zone_{zone['index']:02d}"
        z = thickness if zone["layer"] == "F.Cu" else 0.0
        polygons.append(polygon_vbs(name, zone["vertices_mm"], z))
        conductor_names.append(name)
    via_lines = []
    for index, via in enumerate(board["vias"]):
        x = float(via["x_mm"])
        y = float(via["y_mm"])
        outer = float(via["outer_diameter_mm"]) / 2.0
        drill = float(via["drill_diameter_mm"]) / 2.0
        via_lines.extend([
            f'CreateCylinderZ oEditor, "ViaCavity_{index}", {x:.7f}, {y:.7f}, 0, {outer:.7f}, {thickness:.7f}, "vacuum", True',
            f'SubtractObject oEditor, "Substrate", "ViaCavity_{index}"',
            f'CreateCylinderZ oEditor, "Via_{index}", {x:.7f}, {y:.7f}, 0, {outer:.7f}, {thickness:.7f}, "copper", False',
            f'CreateCylinderZ oEditor, "ViaDrill_{index}", {x:.7f}, {y:.7f}, 0, {drill:.7f}, {thickness:.7f}, "vacuum", True',
            f'SubtractObject oEditor, "Via_{index}", "ViaDrill_{index}"',
        ])
    finite_objects = ", ".join(f'"{name}"' for name in conductor_names)
    # A 0.18 mm operation on the complete 180-vertex artwork produced more
    # than two million tetrahedra on this host.  Keep the deterministic local
    # constraint at the reference plane; lambda/adaptive refinement resolves
    # the full copper artwork.
    mesh_objects = '"PortSheet"'
    x_in = float(design["port_x_mm_from_board_edge"])
    y_center = float(design["port_center_y_mm_from_board_edge"])
    port_width = float(design["port_width_mm"])
    air_x0 = -float(design["air_margin_input_mm"])
    air_y0 = -float(design["air_margin_lateral_mm"])
    air_z0 = -float(design["air_margin_z_mm"])
    air_dx = width + float(design["air_margin_input_mm"]) + float(design["air_margin_aperture_mm"])
    air_dy = height + 2.0 * float(design["air_margin_lateral_mm"])
    air_dz = thickness + 2.0 * float(design["air_margin_z_mm"])
    gate = [float(value) for value in sweep["gate_frequencies_ghz"]]
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO4350B_V148", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(design['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(design['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{design['name']}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{design['name']}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")

' Exact filled copper polygons from the frozen MyriadRF KiCad file.
CreateBox oEditor, "Substrate", 0, 0, 0, {width:.7f}, {height:.7f}, {thickness:.7f}, "RO4350B_V148", True
{chr(10).join(polygons)}
{chr(10).join(via_lines)}

' Connector body is excluded. The reference plane is the SMA center-pad inner board edge.
CreateSheetX oEditor, "PortSheet", {x_in:.7f}, {y_center-port_width/2.0:.7f}, 0, {port_width:.7f}, {thickness:.7f}
oBoundary.AssignLumpedPort Array("NAME:P1", "Objects:=", Array("PortSheet"), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array("{x_in:.7f}mm", "{y_center:.7f}mm", "{thickness:.7f}mm"), "End:=", Array("{x_in:.7f}mm", "{y_center:.7f}mm", "0mm")), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
oBoundary.AssignFiniteCond Array("NAME:CopperFiniteConductivity", "Objects:=", Array({finite_objects}), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)
CreateBox oEditor, "AirRegion", {air_x0:.7f}, {air_y0:.7f}, {air_z0:.7f}, {air_dx:.7f}, {air_dy:.7f}, {air_dz:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:PortAndArtworkMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({mesh_objects}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(design['local_port_mesh_mm']):.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{float(sweep['adaptive_frequency_ghz']):g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {int(design['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {float(design['adaptive_refinement_percent']):g}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False{solver_suffix(solver_type)})
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Broad", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "{float(sweep['broad_start_ghz']):g}GHz", "RangeEnd:=", "{float(sweep['broad_stop_ghz']):g}GHz", "RangeCount:=", {int(sweep['broad_count'])}, "Type:=", "Interpolating", "SaveFields:=", False, "SaveRadFields:=", False, "InterpTolerance:=", 0.5, "InterpMaxSolns:=", 250, "InterpMinSolns:=", 0, "InterpMinSubranges:=", 1, "InterpUseS:=", True, "InterpUsePortImped:=", True, "InterpUsePropConst:=", True, "UseDerivativeConvergence:=", False, "InterpDerivTolerance:=", 0.2, "UseFullBasis:=", True, "EnforcePassivity:=", True, "PassivityErrorTolerance:=", 0.0001, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "{gate[0]:g}GHz", "RangeEnd:=", "{gate[2]:g}GHz", "RangeCount:=", 3, "Type:=", "Discrete", "SaveFields:=", True, "SaveRadFields:=", True, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V148", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{VBS_HELPERS}
'''


def solver_text(project: Path, broad: Path, gate: Path, efficiency: Path, gain: Path, design_name: str) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, reportName
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{design_name}")
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Broad"), 3, "{vp(broad)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Gate3"), 3, "{vp(gate)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
Set oReport = oDesign.GetModule("ReportSetup")
reportName = "V148_RadiationEfficiency"
On Error Resume Next
oReport.DeleteReports Array(reportName)
Err.Clear
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : Sweep_Gate3", Array("Context:=", "InfiniteSphere_V148"), Array("Freq:=", Array("10GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(efficiency)}"
Err.Clear
reportName = "V148_EndfireGain"
oReport.DeleteReports Array(reportName)
Err.Clear
oReport.CreateReport reportName, "Far Fields", "Data Table", "Setup_10GHz : Sweep_Gate3", Array("Context:=", "InfiniteSphere_V148"), Array("Freq:=", Array("10GHz"), "Theta:=", Array("90deg"), "Phi:=", Array("0deg")), Array("X Component:=", "Freq", "Y Component:=", Array("dB(GainTotal)"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(gain)}"
On Error GoTo 0
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def paths(config: dict[str, Any]) -> dict[str, Path]:
    root = resolve(config["output_directory"])
    folder = root / "nominal_direct"
    return {
        "root": root,
        "folder": folder,
        "project": folder / "v148_myriadrf_vivaldi_nominal_direct.aedt",
        "build": folder / "build.vbs",
        "solve": folder / "solve_export.vbs",
        "broad": folder / "v148_myriadrf_vivaldi_broad.s1p",
        "gate": folder / "v148_myriadrf_vivaldi_gate3.s1p",
        "efficiency": folder / "radiation_efficiency.csv",
        "gain": folder / "endfire_gain.csv",
    }


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    p = paths(config)
    if p["root"].exists() and any(p["root"].iterdir()):
        raise FileExistsError(f"Refusing to overwrite calibration output: {p['root']}")
    p["folder"].mkdir(parents=True)
    source = config["source"]
    repo = resolve(source["repository"])
    actual_commit = git(repo, "rev-parse", "HEAD")
    if actual_commit != source["commit"]:
        raise RuntimeError(f"MyriadRF source commit mismatch: {actual_commit}")
    source_paths = {key: resolve(value) for key, value in source.items() if key not in {"repository", "commit"}}
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    board = parse_board(source_paths["board"])
    build = builder_text(p["project"], config, board, "direct")
    p["build"].write_text(build, encoding="ascii")
    p["solve"].write_text(
        solver_text(p["project"], p["broad"], p["gate"], p["efficiency"], p["gain"], config["design"]["name"]),
        encoding="ascii",
    )
    batch = p["root"] / "serial_batchoptions.acf"
    batch.write_text(
        "$begin 'Config'\n"
        f"'Desktop/Settings/ProjectOptions/NumberOfProcessors'={int(config['resources']['number_of_processors'])}\n"
        "$end 'Config'\n",
        encoding="ascii",
    )
    manifest = {
        "source_commit": actual_commit,
        "source_files": {key: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for key, path in source_paths.items()},
        "board": board,
        "case": {key: str(value) for key, value in p.items() if key not in {"root", "folder"}},
        "batch_options": str(batch),
        "static_cad_audit": {
            "zone_count": len(board["zones"]),
            "front_zone_count": sum(zone["layer"] == "F.Cu" for zone in board["zones"]),
            "back_zone_count": sum(zone["layer"] == "B.Cu" for zone in board["zones"]),
            "total_polygon_vertices": sum(len(zone["vertices_mm"]) for zone in board["zones"]),
            "plated_via_count": len(board["vias"]),
            "port_count": build.count('AssignLumpedPort Array("NAME:P1"'),
            "radiation_boundary_count": build.count('AssignRadiation Array("NAME:Rad_AirRegion"'),
            "finite_conductivity": "AssignFiniteCond" in build,
            "broad_sweep_count": build.count('"NAME:Sweep_Broad"'),
            "gate_sweep_count": build.count('"NAME:Sweep_Gate3"'),
            "deterministic_local_mesh_objects": ["PortSheet"],
        },
        "evidence_policy": config["evidence_policy"],
    }
    write_json(p["root"] / "preregistration.json", config)
    write_json(p["root"] / "source_and_cad_manifest.json", manifest)
    write_json(p["root"] / "stage_decision.json", {
        "stage": "calibration_prepared",
        "allow_build_smoke": True,
        "allow_nominal_solve": False,
        **config["locks"],
    })
    return manifest


def require_no_aedt() -> None:
    running = aedt_processes()
    if running:
        raise RuntimeError(f"Concurrent AEDT/HFSS is prohibited: {running}")


def build(config: dict[str, Any]) -> dict[str, Any]:
    require_no_aedt()
    p = paths(config)
    manifest = read_json(p["root"] / "source_and_cad_manifest.json")
    with (p["folder"] / "build.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(resolve(config["ansys_executable"])), "-RunScriptAndExit", str(p["build"])],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    audit = manifest["static_cad_audit"]
    passed = bool(
        result.returncode == 0
        and p["project"].exists()
        and audit["zone_count"] == 4
        and audit["plated_via_count"] == 2
        and audit["port_count"] == 1
        and audit["radiation_boundary_count"] == 1
        and audit["finite_conductivity"]
        and topology_warning_count(p["folder"]) == 0
    )
    summary = {
        "return_code": result.returncode,
        "project_exists": p["project"].exists(),
        "project_bytes": p["project"].stat().st_size if p["project"].exists() else 0,
        "topology_warning_count": topology_warning_count(p["folder"]),
        "static_cad_audit": audit,
        "build_smoke_gate_pass": passed,
        "free_memory_gib_after": memory_available_gb(),
        "aedt_processes_after": aedt_processes(),
    }
    write_json(p["folder"] / "build_audit.json", summary)
    write_json(p["root"] / "stage_decision.json", {
        "stage": "calibration_build_complete",
        "allow_build_smoke": True,
        "allow_nominal_solve": passed,
        **config["locks"],
    })
    return summary


def solve(config: dict[str, Any]) -> dict[str, Any]:
    require_no_aedt()
    p = paths(config)
    decision = read_json(p["root"] / "stage_decision.json")
    if not decision.get("allow_nominal_solve"):
        raise RuntimeError("Calibration solve is locked by the build-smoke gate")
    free = memory_available_gb()
    required = float(config["resources"]["minimum_free_memory_before_solve_gib"])
    if free < required:
        raise MemoryError(f"Need {required:.2f} GiB free, found {free:.2f} GiB")
    batch = p["root"] / "serial_batchoptions.acf"
    code, aborted, minimum = run_process_with_memory_guard(
        [
            str(resolve(config["ansys_executable"])),
            "-ng",
            "-BatchOptions",
            str(batch),
            "-BatchSolve",
            str(p["project"]),
        ],
        p["folder"] / "batch_solve.log",
        float(config["resources"]["abort_free_memory_during_solve_gib"]),
        float(config["resources"]["poll_interval_seconds"]),
    )
    export_code = None
    if code == 0 and not aborted:
        require_no_aedt()
        export_code, export_aborted, export_minimum = run_process_with_memory_guard(
            [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", str(p["solve"])],
            p["folder"] / "solve_export.log",
            float(config["resources"]["abort_free_memory_during_solve_gib"]),
            float(config["resources"]["poll_interval_seconds"]),
        )
        aborted = aborted or export_aborted
        minimum = min(minimum, export_minimum)
    final_code = code if code != 0 else int(export_code or 0)
    summary = {
        "batch_solve_return_code": code,
        "export_return_code": export_code,
        "return_code": final_code,
        "memory_aborted": aborted,
        "free_memory_gib_before": free,
        "minimum_free_memory_gib": minimum,
        "broad_touchstone_exists": p["broad"].exists() and p["broad"].stat().st_size > 100,
        "gate_touchstone_exists": p["gate"].exists() and p["gate"].stat().st_size > 100,
        "efficiency_exists": p["efficiency"].exists(),
        "gain_exists": p["gain"].exists(),
    }
    write_json(p["folder"] / "run_progress.json", summary)
    return summary


def postprocess(config: dict[str, Any]) -> dict[str, Any]:
    """Re-export reports from existing fields without rebuilding or solving."""
    require_no_aedt()
    p = paths(config)
    if not p["project"].exists() or not p["gate"].exists():
        raise FileNotFoundError("A solved project and gate Touchstone are required")
    script = p["folder"] / "postprocess_existing_fields.vbs"
    script.write_text(
        solver_text(
            p["project"], p["broad"], p["gate"], p["efficiency"], p["gain"], config["design"]["name"]
        ),
        encoding="ascii",
    )
    code, aborted, minimum = run_process_with_memory_guard(
        [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", str(script)],
        p["folder"] / "postprocess_existing_fields.log",
        float(config["resources"]["abort_free_memory_during_solve_gib"]),
        float(config["resources"]["poll_interval_seconds"]),
    )
    summary = {
        "return_code": code,
        "memory_aborted": aborted,
        "minimum_free_memory_gib": minimum,
        "efficiency_exists": p["efficiency"].exists(),
        "gain_exists": p["gain"].exists(),
        "new_hfss_solve_started": False,
    }
    write_json(p["folder"] / "postprocess_existing_fields_summary.json", summary)
    return summary


def scalar_from_csv(path: Path) -> float | None:
    if not path.exists():
        return None
    values = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[1:]:
        for token in line.split(",")[1:]:
            try:
                values.append(float(token.strip().strip('"')))
            except ValueError:
                continue
    return values[0] if values else None


def mesh_warning_audit(folder: Path) -> dict[str, Any]:
    texts = [path.read_text(encoding="utf-8", errors="ignore") for path in folder.rglob("*.g3derr")]
    joined = "\n".join(texts)
    body_names = re.findall(r"Small mesh segment detected on body\s*:\s*([^\r\n]+)", joined)
    unique = sorted(set(name.strip() for name in body_names))
    cutoff_values = [
        float(value)
        for value in re.findall(r"cutoff frequency for S-Matrix Only Solve is ([\d.eE+\-]+)MHz", joined)
    ]
    return {
        "record_count": len(body_names),
        "unique_bodies": unique,
        "conductor_or_port_warning": any(name != "AirRegion" for name in unique),
        "maximum_reported_cutoff_mhz": max(cutoff_values) if cutoff_values else None,
        "classification": "numerically benign air-region sliver" if unique == ["AirRegion"] else "requires geometry review",
    }


def curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frequency, matrices = parse_touchstone(path, 1)
    rl = -20.0 * np.log10(np.maximum(np.abs(matrices[:, 0, 0]), 1.0e-15))
    return frequency, rl


def contiguous_band(frequency: np.ndarray, rl: np.ndarray, center: float) -> tuple[float, float, float]:
    passed = rl >= 10.0
    index = int(np.argmin(np.abs(frequency - center)))
    if not passed[index]:
        return 0.0, float(frequency[index]), float(frequency[index])
    left = index
    right = index
    while left > 0 and passed[left - 1]:
        left -= 1
    while right + 1 < len(passed) and passed[right + 1]:
        right += 1
    return float(frequency[right] - frequency[left]), float(frequency[left]), float(frequency[right])


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    p = paths(config)
    if not p["broad"].exists() or not p["gate"].exists():
        raise FileNotFoundError("Calibration Touchstone exports are incomplete")
    broad_frequency, broad_rl = curve(p["broad"])
    gate_frequency, gate_rl = curve(p["gate"])
    center = float(config["sweep"]["adaptive_frequency_ghz"])
    center_index = int(np.argmin(np.abs(gate_frequency - center)))
    bandwidth, low, high = contiguous_band(broad_frequency, broad_rl, center)
    efficiency = efficiency_from_csv(p["efficiency"])
    gain = scalar_from_csv(p["gain"])
    profile = profile_metrics(p["folder"])
    warning_audit = mesh_warning_audit(p["folder"])
    gates = config["gates"]
    numerical_pass = bool(
        profile.get("converged") is True
        and float(profile.get("final_delta_s") or np.inf) <= float(gates["maximum_final_delta_s"])
        and not warning_audit["conductor_or_port_warning"]
    )
    simulated_reference_consistency = bool(
        float(np.min(gate_rl)) >= float(gates["minimum_gate3_passive_rl_db"])
        and float(gate_rl[center_index]) >= float(gates["minimum_center_passive_rl_db"])
        and bandwidth >= float(gates["minimum_contiguous_10db_bandwidth_ghz"])
        and efficiency is not None
        and efficiency >= float(gates["minimum_radiation_efficiency"])
        and gain is not None
        and abs(gain - float(gates["reported_simulated_gain_10ghz_dbi"])) <= float(gates["maximum_simulated_gain_error_db"])
    )
    measured_validation_pass = False
    summary = {
        "version": config["version"],
        **profile,
        "topology_warning_count": topology_warning_count(p["folder"]),
        "mesh_warning_audit": warning_audit,
        "gate_frequencies_ghz": [float(value) for value in gate_frequency],
        "gate_passive_rl_db": [float(value) for value in gate_rl],
        "gate_worst_passive_rl_db": float(np.min(gate_rl)),
        "center_passive_rl_db": float(gate_rl[center_index]),
        "contiguous_10db_bandwidth_ghz": bandwidth,
        "contiguous_10db_band_low_ghz": low,
        "contiguous_10db_band_high_ghz": high,
        "radiation_efficiency": efficiency,
        "endfire_gain_10ghz_dbi": gain,
        "reported_simulated_gain_error_db": None if gain is None else gain - float(gates["reported_simulated_gain_10ghz_dbi"]),
        "numerical_gate_pass": numerical_pass,
        "simulated_reference_consistency_gate_pass": simulated_reference_consistency,
        "measured_validation_gate_pass": measured_validation_pass,
        "engineering_label_allowed": False,
        "training_label_allowed": False,
        "evidence_reason": "The public report has a measured S11 plot but no raw Touchstone data; connector and reference-plane regression cannot be independently reproduced.",
    }
    write_json(p["folder"] / "stage_summary.json", summary)
    write_json(p["root"] / "stage_decision.json", {
        "stage": "myriadrf_calibration_analyzed",
        "myriadrf_numerical_gate_pass": numerical_pass,
        "myriadrf_simulated_reference_consistency_gate_pass": simulated_reference_consistency,
        "myriadrf_measured_validation_gate_pass": False,
        "allow_marlin_1x1": False,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_s256_export": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "next_action": "Obtain AEDT 2025.2 or an unencrypted Marlin project with substrate/port definitions, plus raw MyriadRF VNA data, before reopening the physical branch.",
    })
    return summary


def status(config: dict[str, Any]) -> dict[str, Any]:
    p = paths(config)
    return {
        "output_directory": str(p["root"]),
        "prepared": (p["root"] / "preregistration.json").exists(),
        "built": p["project"].exists(),
        "solved": p["broad"].exists() and p["gate"].exists(),
        "analyzed": (p["folder"] / "stage_summary.json").exists(),
        "decision": read_json(p["root"] / "stage_decision.json") if (p["root"] / "stage_decision.json").exists() else None,
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("prepare", "build", "solve", "postprocess", "analyze", "run", "status"), required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if args.mode == "run":
        result = {"build": build(config)}
        result["solve"] = solve(config)
        if result["solve"]["return_code"] == 0 and not result["solve"]["memory_aborted"]:
            result["analysis"] = analyze(config)
    else:
        result = {"prepare": prepare, "build": build, "solve": solve, "postprocess": postprocess, "analyze": analyze, "status": status}[args.mode](config)
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
