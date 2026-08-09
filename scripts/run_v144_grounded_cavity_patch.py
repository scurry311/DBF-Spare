#!/usr/bin/env python3
"""Build and gate a grounded shallow-cavity probe-fed patch at 1x1 and 2x2.

The cavity walls are responsible for x/y coupling suppression.  Patch length
and probe offset remain independent controls for resonance and input
resistance.  This stage is deliberately limited to a physical 1x1 prescreen
and one frozen-stimulus 2x2 S4 smoke.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
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
from run_v125_feedpoint_input_impedance import topology_warning_count, write_csv, write_json
from run_v139_physical_2x2_differential_array import (
    active_rows,
    grouped_active_summary,
    radiation_matrix,
    touchstone_port_names,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v144_grounded_cavity_patch_preregistered.json"
DESIGN_NAME = "V144_GroundedShallowCavityPatch"
EPS = 1.0e-15


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def ri_to_complex(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    return array[..., 0] + 1j * array[..., 1]


def complex_ri(value: np.ndarray) -> np.ndarray:
    return np.stack((value.real, value.imag), axis=-1)


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    parent = config.get("extends")
    if not parent:
        return config
    base = load_config(resolve(parent))

    def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        output = dict(left)
        for key, value in right.items():
            if key == "extends":
                continue
            if isinstance(value, dict) and isinstance(output.get(key), dict):
                output[key] = merge(output[key], value)
            else:
                output[key] = value
        return output

    return merge(base, config)


def output_root(config: dict[str, Any]) -> Path:
    return resolve(config["output_directory"])


def centers(side: int, spacing: float) -> list[tuple[str, float, float]]:
    if side == 1:
        return [("P00", 0.0, 0.0)]
    if side == 2:
        half = spacing / 2.0
        return [
            ("P00", -half, -half),
            ("P10", half, -half),
            ("P01", -half, half),
            ("P11", half, half),
        ]
    raise ValueError(f"Unsupported side: {side}")


def solver_suffix(solver_type: str) -> str:
    if solver_type == "direct":
        return ', "DrivenSolverType:=", "Direct Solver"'
    if solver_type == "ddm":
        return (
            ', "DrivenSolverType:=", "Domain Decomposition", '
            '"IterativeResidual:=", 0.000001, "DDMSolverResidual:=", 0.000001'
        )
    raise ValueError(f"Unsupported solver type: {solver_type}")


def vbs_helpers() -> str:
    return r'''
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function
Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(210 145 55)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateCylinderZ(editor, objName, x, y, z, radius, height, material, solveInside)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", "20"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(220 155 45)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateSheetY(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(height), "Height:=", Mm(width), "WhichAxis:=", "Y"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub SubtractObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", False)
End Sub
Sub UniteSelection(editor, names)
    editor.Unite Array("NAME:Selections", "Selections:=", names), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub AssignPort(boundary, portName, sheetName, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
'''


def cavity_geometry_text(
    side: int, geometry: dict[str, Any]
) -> tuple[str, list[str], list[str]]:
    spacing_x = float(geometry["spacing_x_mm"])
    spacing_y = float(geometry["spacing_y_mm"])
    if abs(spacing_x - spacing_y) > 1.0e-9:
        raise ValueError("v1.44 currently requires square cells")
    spacing = spacing_x
    board_x = side * spacing_x
    board_y = side * spacing_y
    wall = float(geometry["cavity_wall_thickness_mm"])
    h = float(geometry["substrate_thickness_mm"])
    copper = float(geometry["copper_thickness_mm"])
    above = float(geometry["cavity_wall_height_above_patch_mm"])
    patch_x = float(geometry["patch_width_x_mm"])
    patch_y = float(geometry["patch_length_y_mm"])
    feed_offset_x = float(geometry.get("feed_offset_x_mm", 0.0))
    feed_offset_y = float(geometry["feed_offset_y_mm"])
    probe = float(geometry["probe_radius_mm"])
    coax_inner = float(geometry["coax_inner_radius_mm"])
    coax_outer = float(geometry["coax_outer_radius_mm"])
    coax_drop = float(geometry["coax_drop_mm"])
    coax_dielectric_er = float(geometry.get("coax_dielectric_er", 1.0))
    port_width = float(geometry["port_width_mm"])
    coax_bottom = -h - copper - coax_drop
    tongue_enabled = bool(geometry.get("feed_tongue_enabled", False))
    tongue_width = float(geometry.get("feed_tongue_width_mm", 0.60))
    notch_width = float(geometry.get("feed_notch_width_mm", 1.00))
    notch_depth = float(geometry.get("feed_notch_depth_mm", 1.00))
    tongue_overlap = float(geometry.get("feed_tongue_overlap_mm", 0.10))
    series_l_enabled = bool(geometry.get("series_inductor_enabled", False))
    series_l_nh = float(geometry.get("series_inductance_nh", 0.0))
    series_l_q = float(geometry.get("series_inductor_q", 50.0))
    series_gap = float(geometry.get("series_component_gap_mm", 0.05))

    if patch_x >= spacing_x - 2.0 * wall or patch_y >= spacing_y - 2.0 * wall:
        raise ValueError("Patch does not clear the grounded cavity wall")
    if abs(feed_offset_x) + coax_outer >= spacing_x / 2.0 - wall:
        raise ValueError("Probe/coax does not clear the x cavity wall")
    if abs(feed_offset_y) + coax_outer >= spacing_y / 2.0 - wall:
        raise ValueError("Probe/coax does not clear the grounded cavity wall")
    if abs(feed_offset_x) + probe > patch_x / 2.0 + 1.0e-9:
        raise ValueError("Probe does not remain fully inside the patch in x")
    if abs(feed_offset_y) + probe > patch_y / 2.0 + 1.0e-9:
        raise ValueError("Probe does not remain fully inside the patch in y")
    if tongue_enabled:
        patch_bottom = -patch_y / 2.0
        tongue_start = patch_bottom + 0.05
        tongue_stop = patch_bottom + notch_depth + tongue_overlap
        if tongue_width + 0.10 > notch_width:
            raise ValueError("Feed tongue requires at least 0.10 mm side clearance")
        if abs(feed_offset_x) + probe > tongue_width / 2.0 + 1.0e-9:
            raise ValueError("Probe does not remain fully inside the feed tongue")
        if not (tongue_start + probe <= feed_offset_y <= tongue_stop - probe):
            raise ValueError("Probe is outside the usable feed-tongue length")

    lines: list[str] = []
    patch_names: list[str] = []
    port_names: list[str] = []
    cell_inner_x = spacing_x - wall
    cell_inner_y = spacing_y - wall
    for name, cx, cy in centers(side, spacing):
        substrate = f"Substrate_{name}"
        lines.append(
            f'CreateBox oEditor, "{substrate}", {cx-cell_inner_x/2:.7f}, '
            f'{cy-cell_inner_y/2:.7f}, {-h:.7f}, {cell_inner_x:.7f}, '
            f'{cell_inner_y:.7f}, {h:.7f}, "RO5880_V144", True'
        )

    lines.append(
        f'CreateBox oEditor, "Ground", {-board_x/2:.7f}, {-board_y/2:.7f}, '
        f'{-h-copper:.7f}, {board_x:.7f}, {board_y:.7f}, {copper:.7f}, "copper", False'
    )
    wall_names: list[str] = []
    x_boundaries = [-board_x / 2.0 + i * spacing_x for i in range(side + 1)]
    y_boundaries = [-board_y / 2.0 + i * spacing_y for i in range(side + 1)]
    for index, x in enumerate(x_boundaries):
        name = f"CavityWallX_{index:02d}"
        wall_names.append(name)
        lines.append(
            f'CreateBox oEditor, "{name}", {x-wall/2:.7f}, {-board_y/2-wall/2:.7f}, '
            f'{-h:.7f}, {wall:.7f}, {board_y+wall:.7f}, {h+above:.7f}, "copper", False'
        )
    for index, y in enumerate(y_boundaries):
        name = f"CavityWallY_{index:02d}"
        wall_names.append(name)
        lines.append(
            f'CreateBox oEditor, "{name}", {-board_x/2-wall/2:.7f}, {y-wall/2:.7f}, '
            f'{-h:.7f}, {board_x+wall:.7f}, {wall:.7f}, {h+above:.7f}, "copper", False'
        )
    lines.append(f'UniteSelection oEditor, "Ground,{",".join(wall_names)}"')

    for name, cx, cy in centers(side, spacing):
        feed_x = cx + feed_offset_x
        feed_y = cy + feed_offset_y
        substrate = f"Substrate_{name}"
        patch = f"Patch_{name}"
        patch_names.append(patch)
        port_names.append(name)
        patch_lines = [
            f'CreateBox oEditor, "{patch}", {cx-patch_x/2:.7f}, {cy-patch_y/2:.7f}, '
            f'0, {patch_x:.7f}, {patch_y:.7f}, {copper:.7f}, "copper", False'
        ]
        if tongue_enabled:
            patch_bottom = cy - patch_y / 2.0
            tongue_start = patch_bottom + 0.05
            tongue_length = notch_depth + tongue_overlap - 0.05
            patch_lines.extend(
                [
                    f'CreateBox oEditor, "FeedNotch_{name}", {cx-notch_width/2:.7f}, '
                    f'{patch_bottom-0.01:.7f}, {-0.01:.7f}, {notch_width:.7f}, '
                    f'{notch_depth+0.01:.7f}, {copper+0.02:.7f}, "vacuum", True',
                    f'SubtractObject oEditor, "{patch}", "FeedNotch_{name}"',
                    f'CreateBox oEditor, "FeedTongue_{name}", {cx-tongue_width/2:.7f}, '
                    f'{tongue_start:.7f}, 0, {tongue_width:.7f}, {tongue_length:.7f}, '
                    f'{copper:.7f}, "copper", False',
                    f'UniteSelection oEditor, "{patch},FeedTongue_{name}"',
                ]
            )
        if series_l_enabled:
            split_z = -h / 2.0
            resistance = 2.0 * math.pi * 10.0 * series_l_nh / series_l_q
            probe_lines = [
                f'CreateCylinderZ oEditor, "ProbeLower_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                f'{coax_bottom:.7f}, {probe:.7f}, {split_z-coax_bottom:.7f}, "copper", False',
                f'CreateCylinderZ oEditor, "ProbeUpper_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                f'{split_z+series_gap:.7f}, {probe:.7f}, {copper-split_z-series_gap:.7f}, "copper", False',
                f'UniteSelection oEditor, "{patch},ProbeUpper_{name}"',
                f'CreateSheetY oEditor, "SeriesElement_{name}", {feed_x-probe:.7f}, {feed_y:.7f}, '
                f'{split_z:.7f}, {2*probe:.7f}, {series_gap:.7f}',
                f'oBoundary.AssignLumpedRLC Array("NAME:SeriesL_{name}", "Objects:=", Array("SeriesElement_{name}"), '
                f'Array("NAME:CurrentLine", "Start:=", Array(Mm({feed_x:.7f}), Mm({feed_y:.7f}), Mm({split_z:.7f})), '
                f'"End:=", Array(Mm({feed_x:.7f}), Mm({feed_y:.7f}), Mm({split_z+series_gap:.7f}))), '
                f'"RLC Type:=", "Serial", "UseResist:=", True, "Resistance:=", "{resistance:.9g}ohm", '
                f'"UseInduct:=", True, "Inductance:=", "{series_l_nh:.9g}nH", '
                f'"UseCap:=", False, "Capacitance:=", "0pF")',
            ]
        else:
            probe_lines = [
                f'CreateCylinderZ oEditor, "Probe_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                f'{coax_bottom:.7f}, {probe:.7f}, {h+2*copper+coax_drop:.7f}, "copper", False',
                f'UniteSelection oEditor, "{patch},Probe_{name}"',
            ]
        lines.extend(
            [
                f'CreateCylinderZ oEditor, "GroundHole_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                f'{-h-copper-0.05:.7f}, {coax_inner:.7f}, {copper+0.10:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "Ground", "GroundHole_{name}"',
                f'CreateCylinderZ oEditor, "SubstrateHole_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                f'{-h-0.01:.7f}, {probe+0.01:.7f}, {h+0.02:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "{substrate}", "SubstrateHole_{name}"',
                *patch_lines,
                *probe_lines,
                f'CreateCylinderZ oEditor, "Outer_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                f'{coax_bottom:.7f}, {coax_outer:.7f}, {coax_drop+copper/2:.7f}, "copper", False',
                f'CreateCylinderZ oEditor, "OuterCut_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                f'{coax_bottom-0.02:.7f}, {coax_inner:.7f}, {coax_drop+copper+0.04:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "Outer_{name}", "OuterCut_{name}"',
                *(
                    [
                        f'CreateCylinderZ oEditor, "MatchFill_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                        f'{coax_bottom:.7f}, {coax_inner:.7f}, {coax_drop+copper:.7f}, "MatchCeramic_V144", True',
                        f'CreateCylinderZ oEditor, "MatchFillCut_{name}", {feed_x:.7f}, {feed_y:.7f}, '
                        f'{coax_bottom-0.01:.7f}, {probe+0.005:.7f}, {coax_drop+copper+0.02:.7f}, "vacuum", True',
                        f'SubtractObject oEditor, "MatchFill_{name}", "MatchFillCut_{name}"',
                    ]
                    if coax_dielectric_er > 1.001
                    else []
                ),
                f'UniteSelection oEditor, "Ground,Outer_{name}"',
                f'CreateSheetZ oEditor, "PortSheet_{name}", {feed_x+probe-0.02:.7f}, '
                f'{feed_y-port_width/2:.7f}, {coax_bottom:.7f}, '
                f'{coax_inner-probe+0.04:.7f}, {port_width:.7f}',
                f'AssignPort oBoundary, "{name}", "PortSheet_{name}", '
                f'{feed_x+probe+0.01:.7f}, {feed_y:.7f}, {coax_bottom:.7f}, '
                f'{feed_x+coax_inner-0.01:.7f}, {feed_y:.7f}, {coax_bottom:.7f}',
            ]
        )
    return "\n".join(lines), patch_names, port_names


def builder_text(
    project: Path,
    side: int,
    geometry: dict[str, Any],
    frequency_ghz: float,
    solver_type: str,
) -> str:
    body, patch_names, port_names = cavity_geometry_text(side, geometry)
    board_x = side * float(geometry["spacing_x_mm"])
    board_y = side * float(geometry["spacing_y_mm"])
    air_padding = float(geometry.get("air_padding_mm", 15.0))
    h = float(geometry["substrate_thickness_mm"])
    copper = float(geometry["copper_thickness_mm"])
    coax_drop = float(geometry["coax_drop_mm"])
    mesh = float(geometry["local_mesh_max_length_mm"])
    mesh_objects = ", ".join(f'"{name}"' for name in patch_names)
    match_material = ""
    if float(geometry.get("coax_dielectric_er", 1.0)) > 1.001:
        match_material = (
            'oProject.GetDefinitionManager().AddMaterial Array("NAME:MatchCeramic_V144", '
            '"CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, '
            f'"permittivity:=", "{float(geometry["coax_dielectric_er"]):g}", '
            f'"dielectric_loss_tangent:=", "{float(geometry.get("coax_dielectric_loss_tangent", 0.001)):g}")'
        )
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V144", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(geometry['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(geometry['loss_tangent']):g}")
{match_material}
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
{body}
CreateBox oEditor, "AirRegion", {-board_x/2-air_padding:.7f}, {-board_y/2-air_padding:.7f}, {-h-copper-coax_drop-air_padding:.7f}, {board_x+2*air_padding:.7f}, {board_y+2*air_padding:.7f}, {h+copper+coax_drop+2*air_padding:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:UnifiedPatchProbeMesh_0p18mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({mesh_objects}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency_ghz:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {int(geometry['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {float(geometry['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False{solver_suffix(solver_type)})
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V144", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{vbs_helpers()}
'''


def calibration_states(nports: int) -> list[tuple[str, np.ndarray]]:
    states: list[tuple[str, np.ndarray]] = []
    for i in range(nports):
        vector = np.zeros(nports, dtype=complex)
        vector[i] = 1.0
        states.append((f"basis_{i}", vector))
    for i in range(nports):
        for j in range(i + 1, nports):
            vector = np.zeros(nports, dtype=complex)
            vector[i] = vector[j] = 1.0
            states.append((f"pair_re_{i}_{j}", vector))
            vector = np.zeros(nports, dtype=complex)
            vector[i], vector[j] = 1.0, 1j
            states.append((f"pair_im_{i}_{j}", vector))
    return states


def solver_text(
    project: Path,
    touchstone: Path,
    folder: Path,
    port_names: list[str],
    frequency_ghz: float,
) -> str:
    blocks: list[str] = []
    for name, vector in calibration_states(len(port_names)):
        magnitudes = [f'"{abs(value) ** 2:.12g}W"' for value in vector]
        phases = [f'"{math.degrees(np.angle(value)):.12g}deg"' for value in vector]
        blocks.append(
            f'ApplyCalibration oSolutions, ports, Array({",".join(magnitudes)}), Array({",".join(phases)})\n'
            f'ExportEfficiency oReport, "{name}", "{vp(folder / ("efficiency_" + name + ".csv"))}", "{frequency_ghz:g}GHz"'
        )
    port_array = ",".join(f'"{name}"' for name in port_names)
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, ports
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
ports = Array({port_array})
{chr(10).join(blocks)}
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ApplyCalibration(solModule, portArray, magnitudeArray, phaseArray)
    Dim sources, editArgs(), i, j, sourceName, portName, magnitude, phase
    sources = solModule.GetAllSources()
    ReDim editArgs(UBound(sources) + 1)
    editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)
    For i = LBound(sources) To UBound(sources)
        sourceName = CStr(sources(i))
        magnitude = "0W"
        phase = "0deg"
        For j = LBound(portArray) To UBound(portArray)
            portName = CStr(portArray(j))
            If LCase(Split(sourceName, ":")(0)) = LCase(portName) Then
                magnitude = CStr(magnitudeArray(j))
                phase = CStr(phaseArray(j))
            End If
        Next
        editArgs(i + 1) = Array("Name:=", sourceName, "Magnitude:=", magnitude, "Phase:=", phase)
    Next
    solModule.EditSources editArgs
End Sub
Sub ExportEfficiency(reportModule, stateName, outputPath, frequencyValue)
    Dim reportName
    reportName = "V144_Eff_" & stateName
    On Error Resume Next
    reportModule.DeleteReports Array(reportName)
    Err.Clear
    reportModule.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V144"), Array("Freq:=", Array(frequencyValue)), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
    If Err.Number = 0 Then reportModule.ExportToFile reportName, outputPath
    On Error GoTo 0
End Sub
'''


def case_geometry(
    config: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    return {
        **config["fixed_geometry"],
        **config["array"],
        **{key: value for key, value in candidate.items() if key != "candidate_id"},
    }


def prepare_case(
    root: Path,
    case_id: str,
    side: int,
    geometry: dict[str, Any],
    frequency_ghz: float,
    solver_type: str,
) -> dict[str, Any]:
    folder = root / case_id
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite v1.44 case: {folder}")
    folder.mkdir(parents=True)
    nports = side * side
    project = folder / f"v144_{case_id}.aedt"
    touchstone = folder / f"v144_{case_id}.s{nports}p"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    port_names = [name for name, _, _ in centers(side, float(geometry["spacing_x_mm"]))]
    source = builder_text(project, side, geometry, frequency_ghz, solver_type)
    builder.write_text(source, encoding="ascii")
    solver.write_text(
        solver_text(project, touchstone, folder, port_names, frequency_ghz),
        encoding="ascii",
    )
    case = {
        "case_id": case_id,
        "side": side,
        "port_count": nports,
        "frequency_ghz": frequency_ghz,
        "solver_type": solver_type,
        "port_order": port_names,
        "geometry": geometry,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "builder_sha256": sha256(builder),
        "solver_sha256": sha256(solver),
        "cad_audit": {
            "grounded_cavity_wall_object_count": source.count('CreateBox oEditor, "CavityWall'),
            "probe_feed_count": source.count('CreateCylinderZ oEditor, "Probe_'),
            "lumped_port_count": source.count("AssignPort oBoundary"),
            "input_impedance_control": ["feed_offset_x_mm", "feed_offset_y_mm"],
            "coupling_control": [
                "cavity_wall_thickness_mm",
                "cavity_wall_height_above_patch_mm",
            ],
            "external_matching_stage_count": 0,
            "element_integrated_coax_transformer_count": source.count('CreateCylinderZ oEditor, "MatchFill_'),
            "element_integrated_feed_tongue_count": source.count('CreateBox oEditor, "FeedTongue_'),
            "element_series_rlc_count": source.count("AssignLumpedRLC"),
        },
    }
    write_json(folder / "case_manifest.json", case)
    return case


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    if git("rev-parse", "HEAD") != config["parent_commit"]:
        raise RuntimeError("Current HEAD does not match the preregistered v1.43 baseline")
    if git("rev-list", "-n", "1", config["parent_tag"]) != config["parent_commit"]:
        raise RuntimeError("v1.43 tag does not resolve to the preregistered parent")
    executable = resolve(config["ansys_executable"])
    if not executable.exists():
        raise FileNotFoundError(executable)
    inputs = {name: resolve(path) for name, path in config["inputs"].items()}
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen input(s): {missing}")
    out.mkdir(parents=True)
    stimuli = out / "stimuli"
    stimuli.mkdir()
    shutil.copy2(inputs["frozen_stimuli_manifest"], stimuli / "stimuli_manifest.csv")
    shutil.copy2(inputs["frozen_stimuli_vectors"], stimuli / "stimuli_vectors.npz")
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    frozen_copy_hashes = {
        "stimuli_manifest": sha256(stimuli / "stimuli_manifest.csv"),
        "stimuli_vectors": sha256(stimuli / "stimuli_vectors.npz"),
    }
    audit = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "head_commit": git("rev-parse", "HEAD"),
        "parent_tag": config["parent_tag"],
        "ansys_executable": str(executable),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "input_sha256": input_hashes,
        "frozen_copy_sha256": frozen_copy_hashes,
        "input_hash_copy_match": (
            input_hashes["frozen_stimuli_manifest"] == frozen_copy_hashes["stimuli_manifest"]
            and input_hashes["frozen_stimuli_vectors"] == frozen_copy_hashes["stimuli_vectors"]
        ),
    }
    write_json(out / "preregistration.json", config)
    write_json(out / "baseline_audit.json", audit)
    decision = {
        "stage": "A_preregistered",
        "allow_1x1_prescreen": True,
        "allow_2x2": False,
        "allow_neutralization_layer": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": "The v1.44 grounded shallow-cavity physical branch is preregistered; only the 1x1 impedance prescreen is open.",
    }
    write_json(out / "stage_decision.json", decision)
    return {"audit": audit, "decision": decision}


def prepare_1x1(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_1x1_prescreen"):
        raise RuntimeError("1x1 prescreen is not authorized")
    root = out / "one_by_one_prescreen"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite 1x1 prescreen: {root}")
    root.mkdir()
    cases = []
    for candidate in config["one_by_one_candidates"]:
        geometry = case_geometry(config, candidate)
        cases.append(
            prepare_case(
                root,
                candidate["candidate_id"],
                1,
                geometry,
                float(config["frequency_ghz"]),
                "direct",
            )
        )
    write_json(root / "case_manifest.json", {"cases": cases})
    return {"prepared_cases": len(cases), "root": str(root)}


def run_case(config: dict[str, Any], case: dict[str, Any], minimum_free_gib: float) -> dict[str, Any]:
    require_no_aedt()
    free = memory_available_gb()
    if math.isfinite(free) and free < minimum_free_gib:
        raise MemoryError(
            f"Only {free:.2f} GiB free; case requires {minimum_free_gib:.2f} GiB"
        )
    executable = str(resolve(config["ansys_executable"]))
    folder = Path(case["project_path"]).parent
    started = time.time()
    with (folder / "build.log").open("w", encoding="utf-8") as handle:
        build = subprocess.run(
            [executable, "-RunScriptAndExit", case["builder_path"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    solve_code: int | None = None
    aborted = False
    minimum_observed = free
    if build.returncode == 0 and Path(case["project_path"]).exists():
        require_no_aedt()
        solve_code, aborted, minimum_observed = run_process_with_memory_guard(
            [executable, "-ng", "-RunScriptAndExit", case["solver_path"]],
            folder / "solve_export.log",
            float(config["resources"]["abort_free_memory_during_solve_gib"]),
            float(config["resources"]["poll_interval_seconds"]),
        )
    result = {
        "case_id": case["case_id"],
        "build_return_code": int(build.returncode),
        "solve_return_code": solve_code,
        "memory_aborted": aborted,
        "free_memory_gib_before": free,
        "minimum_free_memory_gib": minimum_observed,
        "elapsed_seconds": time.time() - started,
        "project_exists": Path(case["project_path"]).exists(),
        "touchstone_exists": Path(case["touchstone_path"]).exists(),
        "topology_warning_count": topology_warning_count(folder),
    }
    write_json(folder / "run_audit.json", result)
    return result


def run_1x1(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    root = out / "one_by_one_prescreen"
    manifest = read_json(root / "case_manifest.json")
    rows = []
    minimum = float(config["resources"]["minimum_free_memory_before_1x1_gib"])
    for case in manifest["cases"]:
        rows.append(run_case(config, case, minimum))
    write_csv(root / "execution.csv", rows)
    return {"cases": rows}


def target_self_impedance(config: dict[str, Any]) -> complex:
    with np.load(resolve(config["inputs"]["target_s4"]), allow_pickle=False) as data:
        target = ri_to_complex(data["target_s4_real_imag"])
    gamma = complex(np.mean(np.diag(target)))
    return 50.0 * (1.0 + gamma) / (1.0 - gamma)


def analyze_1x1(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    root = out / "one_by_one_prescreen"
    cases = read_json(root / "case_manifest.json")["cases"]
    target_z = target_self_impedance(config)
    gates = config["gates"]
    rows: list[dict[str, Any]] = []
    for case in cases:
        folder = Path(case["project_path"]).parent
        run = read_json(folder / "run_audit.json")
        profile = profile_metrics(folder)
        touchstone = Path(case["touchstone_path"])
        row: dict[str, Any] = {
            "case_id": case["case_id"],
            **case["geometry"],
            **run,
            **profile,
        }
        if touchstone.exists() and touchstone.stat().st_size > 100:
            frequencies, matrices = parse_touchstone(touchstone, 1)
            index = int(np.argmin(np.abs(frequencies - float(case["frequency_ghz"]))))
            s11 = complex(matrices[index, 0, 0])
            impedance = 50.0 * (1.0 + s11) / (1.0 - s11)
            efficiency = efficiency_from_csv(folder / "efficiency_basis_0.csv")
            row.update(
                {
                    "frequency_ghz": float(frequencies[index]),
                    "s11_real": float(s11.real),
                    "s11_imag": float(s11.imag),
                    "passive_rl_db": float(-20.0 * np.log10(max(abs(s11), EPS))),
                    "input_resistance_ohm": float(impedance.real),
                    "input_reactance_ohm": float(impedance.imag),
                    "target_impedance_error_ohm": float(abs(impedance - target_z)),
                    "radiation_efficiency": efficiency,
                }
            )
        else:
            row.update(
                {
                    "passive_rl_db": -math.inf,
                    "target_impedance_error_ohm": math.inf,
                    "radiation_efficiency": None,
                }
            )
        row["one_by_one_gate_pass"] = bool(
            row.get("build_return_code") == 0
            and row.get("solve_return_code") == 0
            and not row.get("memory_aborted")
            and row.get("converged") is True
            and float(row.get("final_delta_s") or math.inf)
            <= float(gates["maximum_final_delta_s"])
            and float(row["passive_rl_db"])
            >= float(gates["minimum_1x1_passive_rl_db"])
            and row.get("radiation_efficiency") is not None
            and float(row["radiation_efficiency"])
            >= float(gates["minimum_system_efficiency"])
            and int(row.get("topology_warning_count", 999))
            <= int(gates["maximum_port_topology_warning_count"])
        )
        row["selection_score"] = (
            float(row["passive_rl_db"])
            - 0.15 * float(row["target_impedance_error_ohm"])
            + 2.0 * float(row.get("radiation_efficiency") or 0.0)
        )
        rows.append(row)
        write_json(folder / "case_summary.json", row)
    write_csv(root / "one_by_one_metrics.csv", rows)
    passing = [row for row in rows if row["one_by_one_gate_pass"]]
    selected = max(passing, key=lambda row: float(row["selection_score"])) if passing else None
    decision = read_json(out / "stage_decision.json")
    decision.update(
        {
            "stage": "B_one_by_one_complete",
            "one_by_one_pass_count": len(passing),
            "selected_candidate_id": selected["case_id"] if selected else None,
            "allow_2x2": selected is not None,
            "allow_neutralization_layer": False,
            "reason": (
                "At least one grounded cavity element passed the physical impedance/efficiency gate; one frozen 2x2 S4 smoke is authorized."
                if selected
                else "No grounded cavity element passed the 1x1 impedance/efficiency gate; 2x2 remains locked and a new impedance-tuning run is required."
            ),
        }
    )
    write_json(out / "stage_decision.json", decision)
    return {"metrics": rows, "selected": selected, "decision": decision}


def prepare_2x2(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_2x2"):
        raise RuntimeError("2x2 is not authorized by the 1x1 gate")
    candidate_id = str(decision["selected_candidate_id"])
    source_cases = read_json(out / "one_by_one_prescreen" / "case_manifest.json")["cases"]
    source = next(case for case in source_cases if case["case_id"] == candidate_id)
    root = out / "two_by_two_initial"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite 2x2 initial smoke: {root}")
    root.mkdir()
    case = prepare_case(
        root,
        f"{candidate_id}_direct",
        2,
        source["geometry"],
        float(config["frequency_ghz"]),
        "direct",
    )
    write_json(root / "case_manifest.json", {"cases": [case]})
    return case


def run_2x2(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    case = read_json(out / "two_by_two_initial" / "case_manifest.json")["cases"][0]
    minimum = float(config["resources"]["minimum_free_memory_before_2x2_gib"])
    result = run_case(config, case, minimum)
    write_csv(out / "two_by_two_initial" / "execution.csv", [result])
    return result


def reordered_s4(case: dict[str, Any]) -> tuple[float, np.ndarray, list[str]]:
    path = Path(case["touchstone_path"])
    frequencies, matrices = parse_touchstone(path, 4)
    index = int(np.argmin(np.abs(frequencies - float(case["frequency_ghz"]))))
    s = matrices[index]
    expected = list(case["port_order"])
    exported = touchstone_port_names(path)
    if exported and sorted(exported) != sorted(expected):
        raise RuntimeError(f"Unexpected Touchstone port names: {exported}")
    if exported and exported != expected:
        permutation = [exported.index(name) for name in expected]
        s = s[np.ix_(permutation, permutation)]
    return float(frequencies[index]), s, exported


def coupling_metrics(s: np.ndarray, port_order: list[str]) -> list[dict[str, Any]]:
    pairs = {
        "x": [(0, 1), (2, 3)],
        "y": [(0, 2), (1, 3)],
        "diagonal": [(0, 3), (1, 2)],
    }
    rows = []
    for group, indices in pairs.items():
        for i, j in indices:
            rows.append(
                {
                    "group": group,
                    "port_i": port_order[i],
                    "port_j": port_order[j],
                    "sij_real": float(s[i, j].real),
                    "sij_imag": float(s[i, j].imag),
                    "coupling_db": float(20.0 * np.log10(max(abs(s[i, j]), EPS))),
                }
            )
    return rows


def modal_metrics(s: np.ndarray) -> list[dict[str, Any]]:
    modes = {
        "even": [1, 1, 1, 1],
        "x_odd": [1, -1, 1, -1],
        "y_odd": [1, 1, -1, -1],
        "checker": [1, -1, -1, 1],
    }
    rows = []
    for name, raw in modes.items():
        vector = np.asarray(raw, complex) / 2.0
        gamma = complex(np.vdot(vector, s @ vector))
        impedance = 50.0 * (1.0 + gamma) / (1.0 - gamma)
        rows.append(
            {
                "mode": name,
                "gamma_real": float(gamma.real),
                "gamma_imag": float(gamma.imag),
                "rl_db": float(-20.0 * np.log10(max(abs(gamma), EPS))),
                "resistance_ohm": float(impedance.real),
                "reactance_ohm": float(impedance.imag),
            }
        )
    return rows


def analyze_2x2(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    case = read_json(out / "two_by_two_initial" / "case_manifest.json")["cases"][0]
    folder = Path(case["project_path"]).parent
    run = read_json(folder / "run_audit.json")
    frequency, s, exported = reordered_s4(case)
    radiation, calibration = radiation_matrix(folder, s)
    active = active_rows(out, s, radiation, frequency)
    grouped = grouped_active_summary(active)
    coupling = coupling_metrics(s, list(case["port_order"]))
    modal = modal_metrics(s)
    profile = profile_metrics(folder)
    passive_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s)), EPS))
    reciprocity = float(np.max(np.abs(s - s.T)))
    passivity = float(np.linalg.svd(s, compute_uv=False)[0])
    min_active = min(float(row["active_rl_db"]) for row in active)
    min_total = min(float(row["total_rl_db"]) for row in active)
    min_efficiency = min(float(row["system_efficiency"]) for row in active)
    group_worst = {
        group: max(float(row["coupling_db"]) for row in coupling if row["group"] == group)
        for group in ("x", "y", "diagonal")
    }
    old_coupling = float(config["references"]["old_grounded_patch_nearest_coupling_db"])
    worst_neighbor = max(group_worst["x"], group_worst["y"])
    improvement = old_coupling - worst_neighbor
    with np.load(resolve(config["inputs"]["target_s4"]), allow_pickle=False) as data:
        target_s4 = ri_to_complex(data["target_s4_real_imag"])
    gates = config["gates"]
    numerical_gate = bool(
        run["build_return_code"] == 0
        and run["solve_return_code"] == 0
        and not run["memory_aborted"]
        and profile.get("converged") is True
        and float(profile.get("final_delta_s") or math.inf)
        <= float(gates["maximum_final_delta_s"])
        and reciprocity <= float(gates["maximum_reciprocity_error"])
        and passivity <= float(gates["maximum_passivity_sigma"])
        and int(run["topology_warning_count"])
        <= int(gates["maximum_port_topology_warning_count"])
    )
    coupling_gate = bool(
        group_worst["x"] <= float(gates["maximum_x_neighbor_coupling_db"])
        and group_worst["y"] <= float(gates["maximum_y_neighbor_coupling_db"])
        and group_worst["diagonal"] <= float(gates["maximum_diagonal_coupling_db"])
        and improvement >= float(gates["minimum_coupling_improvement_vs_old_patch_db"])
    )
    strict = bool(
        numerical_gate
        and coupling_gate
        and float(passive_rl.min()) >= float(gates["minimum_2x2_passive_rl_db"])
        and min_active >= float(gates["minimum_active_rl_design_db"])
        and min_total >= float(gates["minimum_total_rl_db"])
        and min_efficiency >= float(gates["minimum_system_efficiency"])
    )
    summary = {
        "case_id": case["case_id"],
        "frequency_ghz": frequency,
        **profile,
        "exported_port_order": exported,
        "effective_port_order": case["port_order"],
        "maximum_reciprocity_error": reciprocity,
        "maximum_passivity_sigma": passivity,
        "minimum_passive_rl_db": float(passive_rl.min()),
        "minimum_active_rl_db": min_active,
        "minimum_total_rl_db": min_total,
        "minimum_system_efficiency": min_efficiency,
        "worst_x_coupling_db": group_worst["x"],
        "worst_y_coupling_db": group_worst["y"],
        "worst_diagonal_coupling_db": group_worst["diagonal"],
        "coupling_improvement_vs_old_patch_db": improvement,
        "physical_vs_target_s4_max_abs_delta": float(np.max(np.abs(s - target_s4))),
        "stimulus_count": len(active),
        "active_rl_ge_10_count": sum(float(row["active_rl_db"]) >= 10.0 for row in active),
        "active_rl_ge_11_count": sum(float(row["active_rl_db"]) >= 11.0 for row in active),
        "numerical_gate_pass": numerical_gate,
        "coupling_gate_pass": coupling_gate,
        "strict_two_by_two_gate_pass": strict,
    }
    write_json(folder / "case_summary.json", summary)
    write_csv(folder / "active_metrics.csv", active)
    write_csv(folder / "active_group_summary.csv", grouped)
    write_csv(folder / "coupling_metrics.csv", coupling)
    write_csv(folder / "modal_impedance.csv", modal)
    write_csv(folder / "radiation_calibration.csv", calibration)
    np.savez_compressed(
        folder / "physical_operators.npz",
        s_real_imag=complex_ri(s),
        radiation_real_imag=complex_ri(radiation),
        target_s4_real_imag=complex_ri(target_s4),
        port_order=np.asarray(case["port_order"]),
    )
    decision = read_json(out / "stage_decision.json")
    decision.update(
        {
            "stage": "C_two_by_two_complete",
            "strict_two_by_two_gate_pass": strict,
            "allow_independent_crosscheck": strict,
            "allow_neutralization_layer": not strict and numerical_gate,
            "allow_4x4": False,
            "allow_16x16": False,
            "allow_eep_export": False,
            "allow_training_labels": False,
            "allow_critic_training": False,
            "reason": (
                "The grounded cavity 2x2 passed matching, coupling, frozen active-RL, total-RL, efficiency and numerical gates; only an independent direct/DDM crosscheck may follow."
                if strict
                else (
                    "The cavity model is numerically valid but misses a physical gate; one new-run grounded neutralization-layer rescue is allowed, while array scaling and learning remain locked."
                    if numerical_gate
                    else "The cavity model failed its numerical gate; no geometry rescue or array scaling is authorized until the solve is repaired."
                )
            ),
        }
    )
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "grouped": grouped, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = output_root(config)
    return {
        "output_directory": str(out),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "preregistered": (out / "preregistration.json").exists(),
        "one_by_one_prepared": (out / "one_by_one_prescreen" / "case_manifest.json").exists(),
        "one_by_one_analyzed": (out / "one_by_one_prescreen" / "one_by_one_metrics.csv").exists(),
        "two_by_two_prepared": (out / "two_by_two_initial" / "case_manifest.json").exists(),
        "two_by_two_analyzed": any(out.glob("two_by_two_initial/*/case_summary.json")),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=(
            "preregister",
            "prepare-1x1",
            "run-1x1",
            "analyze-1x1",
            "prepare-2x2",
            "run-2x2",
            "analyze-2x2",
            "status",
        ),
        default="status",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    actions = {
        "preregister": preregister,
        "prepare-1x1": prepare_1x1,
        "run-1x1": run_1x1,
        "analyze-1x1": analyze_1x1,
        "prepare-2x2": prepare_2x2,
        "run-2x2": run_2x2,
        "analyze-2x2": analyze_2x2,
        "status": status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
