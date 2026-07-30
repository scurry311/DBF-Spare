#!/usr/bin/env python3
"""Build and gate an integrated physical 2x2 feed-network/antenna HFSS smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from design_v115_grounded_modal_network import terminate_network
from run_v114_small_cell_broadband_feed import (
    load_stimuli,
    memory_available_gb,
    parse_touchstone,
    profile_metrics,
)
from run_v115_physical_modal_feed_fixture import touchstone_port_names


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v117_integrated_2x2_smoke.json"
DESIGN_NAME = "V117_Integrated_Feed_Antenna_2x2"
ETA0 = 376.730313668
EPS = 1.0e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "export", "analyze", "status"), default="status")
    parser.add_argument("--replicate", type=int, choices=(1, 2), default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def frequency_code(frequency: float) -> str:
    return f"{frequency:.2f}".replace(".", "p")


def wait_for_stable_file(path: Path, timeout_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    previous = -1
    stable = 0
    while time.monotonic() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 100 and size == previous:
                stable += 1
                if stable >= 2:
                    return True
            else:
                stable = 0
            previous = size
        time.sleep(1.0)
    return False


def run_solver_with_memory_guard(
    command: list[str],
    log_path: Path,
    abort_free_memory_gb: float,
) -> tuple[int, bool, float]:
    minimum_observed = math.inf
    low_memory_checks = 0
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            free_memory = memory_available_gb()
            if math.isfinite(free_memory):
                minimum_observed = min(minimum_observed, free_memory)
                low_memory_checks = low_memory_checks + 1 if free_memory < abort_free_memory_gb else 0
            if low_memory_checks >= 3:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                process.wait(timeout=30)
                return 99, True, minimum_observed
            time.sleep(5.0)
    return int(process.returncode), False, minimum_observed


def helpers_vbs() -> str:
    return r'''Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(170 130 70)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateCylinderZ(editor, objName, x, y, z, radius, height, material, solveInside)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", "24"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(210 150 50)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheet(editor, objName, axisName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", axisName), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.1, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub SubtractObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", False)
End Sub
Sub SubtractKeepObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", True)
End Sub
Sub UniteObjects(editor, firstName, secondName)
    editor.Unite Array("NAME:Selections", "Selections:=", firstName & "," & secondName), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub AssignPort(boundary, portName, sheetName, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Sub AssignRLCExpr(boundary, boundaryName, sheetName, rlcType, useR, resistanceExpr, useL, inductanceExpr, useC, capacitanceExpr, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedRLC Array("NAME:" & boundaryName, "Objects:=", Array(sheetName), Array("NAME:CurrentLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "RLC Type:=", rlcType, "UseResist:=", useR, "Resistance:=", resistanceExpr, "UseInduct:=", useL, "Inductance:=", inductanceExpr, "UseCap:=", useC, "Capacitance:=", capacitanceExpr)
End Sub
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function'''


def builder_text(project: Path, config: dict[str, Any], antenna: dict[str, Any]) -> str:
    physical = antenna["physical_topology"]
    candidate = antenna["one_by_one_candidates"][0]
    network = config["feed_network"]
    mesh = config["mesh"]
    spacing = float(physical["spacing_mm"])
    board = 30.0
    antenna_h = float(physical["substrate_thickness_mm"])
    antenna_copper = float(physical["copper_thickness_mm"])
    network_h = float(network["substrate_thickness_mm"])
    network_copper = float(network["copper_thickness_mm"])
    ground_bottom = -antenna_h - antenna_copper
    signal_top = ground_bottom - network_h
    signal_bottom = signal_top - network_copper
    patch_w = float(physical["patch_width_mm"])
    patch_l = float(physical["patch_length_mm"])
    slot_l = float(candidate["slot_length_mm"])
    slot_w = float(candidate["slot_width_mm"])
    tongue = float(candidate["tongue_width_mm"])
    feed_inset = float(candidate["feed_inset_from_edge_mm"])
    trace_w = float(network["trace_width_mm"])
    substrate_margin = float(network["routed_substrate_margin_mm"])
    substrate_ribbon = trace_w + 2.0 * substrate_margin
    offset = float(network["x_offset_mm"])
    pre_x = offset + float(network["pre_reference_x_local_mm"])
    post_x = offset + float(network["post_reference_x_local_mm"])
    gap_center = offset + float(network["series_gap_center_x_local_mm"])
    gap_length = float(network["series_gap_length_mm"])
    cap_x = offset + float(network["ground_cap_x_local_mm"])
    bridge_x = offset + float(network["bridge_x_local_mm"])
    bridge_width = float(network["bridge_sheet_width_mm"])
    y_channels = [float(value) for value in network["channel_y_mm_by_port"]]
    left_end = gap_center - gap_length / 2.0
    right_start = gap_center + gap_length / 2.0
    omega = 2.0 * math.pi * 10.0e9
    series_l = float(network["series_inductor_nh"])
    ground_c = float(network["ground_capacitor_pf"])
    bridge_l = float(network["bridge_inductor_nh"])
    series_r = omega * series_l * 1.0e-9 / float(network["series_q"])
    ground_r = float(network["ground_capacitor_q"]) / (omega * ground_c * 1.0e-12)
    bridge_r = float(network["bridge_inductor_q"]) * omega * bridge_l * 1.0e-9
    probe_radius = 0.25
    hole_radius = 0.60
    component_names: list[str] = []
    fanout_names: list[str] = []
    antenna_mesh_names: list[str] = []
    geometry: list[str] = []
    assignments: list[str] = []
    feed_points: list[tuple[float, float]] = []
    for ix in range(2):
        for iy in range(2):
            xc = (ix - 0.5) * spacing
            yc = (iy - 0.5) * spacing
            feed_y = yc - patch_l / 2.0 + feed_inset
            feed_points.append((xc, feed_y))
    for port, (feed_x, feed_y) in enumerate(feed_points):
        y_value = y_channels[port]
        patch_bottom = (-0.5 if port % 2 == 0 else 0.5) * spacing - patch_l / 2.0
        slot_offset = 0.5 * (tongue + slot_w)
        pre_name = f"TracePre_{port}"
        post_name = f"TracePost_{port}"
        series_name = f"SeriesSheet_{port}"
        cap_name = f"GroundCapSheet_{port}"
        fan_h = f"FanH_{port}"
        fan_v = f"FanV_{port}"
        patch_name = f"Patch_{port}"
        probe_name = f"Probe_{port}"
        feed_mesh = f"FeedMesh_{port}"
        port_sheet = f"PrePortSheet_{port}"
        geometry.extend(
            [
                f'CreateBox oEditor, "NetworkSubH_{port}", {post_x-substrate_margin:.7f}, {y_value-substrate_ribbon/2:.7f}, {signal_top:.7f}, {feed_x-post_x+2*substrate_margin:.7f}, {substrate_ribbon:.7f}, {network_h:.7f}, "RO5880_V117_Network", True',
                f'UniteObjects oEditor, "NetworkSubstrate", "NetworkSubH_{port}"',
                f'CreateBox oEditor, "NetworkSubV_{port}", {feed_x-substrate_ribbon/2:.7f}, {min(y_value, feed_y)-substrate_margin:.7f}, {signal_top:.7f}, {substrate_ribbon:.7f}, {abs(feed_y-y_value)+2*substrate_margin:.7f}, {network_h:.7f}, "RO5880_V117_Network", True',
                f'UniteObjects oEditor, "NetworkSubstrate", "NetworkSubV_{port}"',
                f'CreateBox oEditor, "{pre_name}", {pre_x:.7f}, {y_value-trace_w/2:.7f}, {signal_bottom:.7f}, {left_end-pre_x:.7f}, {trace_w:.7f}, {network_copper:.7f}, "copper", False',
                f'CreateBox oEditor, "{post_name}", {right_start:.7f}, {y_value-trace_w/2:.7f}, {signal_bottom:.7f}, {post_x-right_start:.7f}, {trace_w:.7f}, {network_copper:.7f}, "copper", False',
                f'CreateSheet oEditor, "{series_name}", "Z", {left_end:.7f}, {y_value-trace_w/2:.7f}, {signal_top:.7f}, {gap_length:.7f}, {trace_w:.7f}',
                f'CreateSheet oEditor, "{cap_name}", "X", {cap_x:.7f}, {y_value-trace_w/4:.7f}, {signal_top:.7f}, {trace_w/2:.7f}, {network_h:.7f}',
                f'CreateSheet oEditor, "{port_sheet}", "X", {pre_x:.7f}, {y_value-trace_w/2:.7f}, {signal_top:.7f}, {trace_w:.7f}, {network_h:.7f}',
                f'CreateBox oEditor, "{fan_h}", {post_x:.7f}, {y_value-trace_w/2:.7f}, {signal_bottom:.7f}, {feed_x-post_x:.7f}, {trace_w:.7f}, {network_copper:.7f}, "copper", False',
                f'CreateBox oEditor, "{fan_v}", {feed_x-trace_w/2:.7f}, {min(y_value, feed_y):.7f}, {signal_bottom:.7f}, {trace_w:.7f}, {abs(feed_y-y_value):.7f}, {network_copper:.7f}, "copper", False',
                f'CreateBox oEditor, "{feed_mesh}", {feed_x-(tongue+2*slot_w+1.2)/2:.7f}, {patch_bottom-0.10:.7f}, {-antenna_h:.7f}, {tongue+2*slot_w+1.2:.7f}, {slot_l+0.5:.7f}, {antenna_h:.7f}, "RO5880_V117_Antenna", True',
                f'SubtractKeepObject oEditor, "AntennaSubstrate", "{feed_mesh}"',
                f'CreateCylinderZ oEditor, "GroundHole_{port}", {feed_x:.7f}, {feed_y:.7f}, {ground_bottom-0.05:.7f}, {hole_radius:.7f}, {antenna_copper+0.10:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "SharedGround", "GroundHole_{port}"',
                f'CreateCylinderZ oEditor, "AntennaHole_{port}", {feed_x:.7f}, {feed_y:.7f}, {-antenna_h-0.01:.7f}, {probe_radius+0.01:.7f}, {antenna_h+0.02:.7f}, "vacuum", True',
                f'SubtractKeepObject oEditor, "AntennaSubstrate", "AntennaHole_{port}"',
                f'SubtractObject oEditor, "{feed_mesh}", "AntennaHole_{port}"',
                f'CreateCylinderZ oEditor, "NetworkHole_{port}", {feed_x:.7f}, {feed_y:.7f}, {signal_top-0.01:.7f}, {probe_radius+0.01:.7f}, {network_h+0.02:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "NetworkSubstrate", "NetworkHole_{port}"',
                f'CreateBox oEditor, "{patch_name}", {feed_x-patch_w/2:.7f}, {patch_bottom:.7f}, 0, {patch_w:.7f}, {patch_l:.7f}, {antenna_copper:.7f}, "copper", False',
                f'CreateBox oEditor, "SlotL_{port}", {feed_x-slot_offset-slot_w/2:.7f}, {patch_bottom-0.01:.7f}, -0.01, {slot_w:.7f}, {slot_l+0.02:.7f}, {antenna_copper+0.02:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "{patch_name}", "SlotL_{port}"',
                f'CreateBox oEditor, "SlotR_{port}", {feed_x+slot_offset-slot_w/2:.7f}, {patch_bottom-0.01:.7f}, -0.01, {slot_w:.7f}, {slot_l+0.02:.7f}, {antenna_copper+0.02:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "{patch_name}", "SlotR_{port}"',
                f'CreateCylinderZ oEditor, "{probe_name}", {feed_x:.7f}, {feed_y:.7f}, {signal_bottom:.7f}, {probe_radius:.7f}, {-signal_bottom+antenna_copper:.7f}, "copper", False',
                f'UniteObjects oEditor, "{patch_name}", "{probe_name}"',
                f'UniteObjects oEditor, "{post_name}", "{fan_h}"',
                f'UniteObjects oEditor, "{post_name}", "{fan_v}"',
                f'UniteObjects oEditor, "{post_name}", "{patch_name}"',
            ]
        )
        assignments.extend(
            [
                f'AssignRLCExpr oBoundary, "SeriesL_{port}", "{series_name}", "Serial", True, "v_series_r", True, "v_series_l", False, "1pF", {left_end:.7f}, {y_value:.7f}, {signal_top:.7f}, {right_start:.7f}, {y_value:.7f}, {signal_top:.7f}',
                f'AssignRLCExpr oBoundary, "GroundC_{port}", "{cap_name}", "Parallel", True, "v_ground_r", False, "1nH", True, "v_ground_c", {cap_x:.7f}, {y_value:.7f}, {signal_top:.7f}, {cap_x:.7f}, {y_value:.7f}, {ground_bottom:.7f}',
                f'AssignPort oBoundary, "PRE_{port}", "{port_sheet}", {pre_x:.7f}, {y_value:.7f}, {signal_top:.7f}, {pre_x:.7f}, {y_value:.7f}, {ground_bottom:.7f}',
            ]
        )
        component_names.extend((pre_name, series_name, cap_name))
        fanout_names.append(post_name)
        antenna_mesh_names.append(feed_mesh)
    for pair_index, (first, second) in enumerate(((0, 2), (1, 3))):
        low, high = sorted((y_channels[first], y_channels[second]))
        y0 = low + trace_w / 2.0
        height = high - low - trace_w
        name = f"BridgeSheet_{pair_index}"
        geometry.append(
            f'CreateSheet oEditor, "{name}", "Z", {bridge_x-bridge_width/2:.7f}, {y0:.7f}, {signal_top:.7f}, {bridge_width:.7f}, {height:.7f}'
        )
        assignments.append(
            f'AssignRLCExpr oBoundary, "BridgeL_{pair_index}", "{name}", "Parallel", True, "v_bridge_r", True, "v_bridge_l", False, "1pF", {bridge_x:.7f}, {y0:.7f}, {signal_top:.7f}, {bridge_x:.7f}, {y0+height:.7f}, {signal_top:.7f}'
        )
        component_names.append(name)
    component_items = ", ".join(f'"{name}"' for name in component_names)
    fanout_items = ", ".join(f'"{name}"' for name in fanout_names)
    antenna_items = ", ".join(f'"{name}"' for name in antenna_mesh_names)
    variable_block = f'''oDesign.ChangeProperty Array( _
    "NAME:AllTabs", _
    Array( _
        "NAME:LocalVariableTab", _
        Array("NAME:PropServers", "LocalVariables"), _
        Array( _
            "NAME:NewProps", _
            Array("NAME:v_series_l", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{series_l:.10f}nH"), _
            Array("NAME:v_series_r", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{series_r:.10f}ohm"), _
            Array("NAME:v_ground_c", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{ground_c:.10f}pF"), _
            Array("NAME:v_ground_r", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{ground_r:.10f}ohm"), _
            Array("NAME:v_bridge_l", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{bridge_l:.10f}nH"), _
            Array("NAME:v_bridge_r", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{bridge_r:.10f}ohm") _
        ) _
    ) _
)'''
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V117_Antenna", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{physical['relative_permittivity']}", "dielectric_loss_tangent:=", "{physical['loss_tangent']}")
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V117_Network", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{network['relative_permittivity']}", "dielectric_loss_tangent:=", "{network['loss_tangent']}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
{variable_block}
CreateBox oEditor, "AntennaSubstrate", {-board/2:.7f}, {-board/2:.7f}, {-antenna_h:.7f}, {board:.7f}, {board:.7f}, {antenna_h:.7f}, "RO5880_V117_Antenna", True
CreateBox oEditor, "SharedGround", {-board/2:.7f}, {-board/2:.7f}, {ground_bottom:.7f}, {board:.7f}, {board:.7f}, {antenna_copper:.7f}, "copper", False
CreateBox oEditor, "NetworkSubstrate", {pre_x-substrate_margin:.7f}, {-3.75-substrate_margin:.7f}, {signal_top:.7f}, {post_x-pre_x+2*substrate_margin:.7f}, {7.5+2*substrate_margin:.7f}, {network_h:.7f}, "RO5880_V117_Network", True
{chr(10).join(geometry)}
{chr(10).join(assignments)}
CreateBox oEditor, "AirRegion", {-board/2-15:.7f}, {-board/2-15:.7f}, -16, {board+30:.7f}, {board+30:.7f}, 32, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:V117_NetworkComponentMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({component_items}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(mesh['network_component_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
oMesh.AssignLengthOp Array("NAME:V117_AntennaFeedMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({antenna_items}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(mesh['antenna_feed_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", {float(config['gates']['maximum_final_delta_s']):.7f}, "MaximumPasses:=", {int(mesh['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", {int(mesh['minimum_converged_passes'])}, "PercentRefinement:=", {float(mesh['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "PortAccuracy:=", 2)
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "9.96GHz", "RangeEnd:=", "10.04GHz", "RangeCount:=", 3, "Type:=", "Discrete", "SaveFields:=", True, "SaveRadFields:=", True, "InterpTolerance:=", 0.5, "InterpMaxSolns:=", 250, "InterpMinSolns:=", 0, "InterpMinSubranges:=", 1, "InterpUseS:=", True, "InterpUsePortImped:=", True, "InterpUsePropConst:=", True, "UseDerivativeConvergence:=", False, "InterpDerivTolerance:=", 0.2, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V117", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{helpers_vbs()}
'''


def solver_text(
    project: Path,
    touchstone: Path,
    folder: Path,
    frequencies: list[float],
    solve_network: bool = True,
) -> str:
    frequency_loop = []
    for frequency in frequencies:
        code = frequency_code(frequency)
        frequency_loop.append(
            f'ExportPortFields oSolutions, oReport, ports, "{frequency:g}GHz", "{vp(folder)}", "{code}"'
        )
    solve_block = ""
    if solve_network:
        solve_block = f'''oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")
vars = oSolutions.ListVariations("Setup_10GHz:Sweep_Gate3")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Gate3"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False'''
    else:
        solve_block = '''Set oSolutions = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")'''
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, ports
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
{solve_block}
ports = Array("PRE_0", "PRE_1", "PRE_2", "PRE_3")
{chr(10).join(frequency_loop)}
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ExportPortFields(solModule, reportModule, portArray, frequencyValue, outDir, frequencyCode)
    Dim i, portName, reportName, fieldPath, efficiencyPath
    For i = LBound(portArray) To UBound(portArray)
        portName = CStr(portArray(i))
        ApplySinglePort solModule, portName
        reportName = "V117_EEP_" & frequencyCode & "_" & portName
        fieldPath = outDir & "\\eep_" & LCase(portName) & "_" & frequencyCode & ".csv"
        DeleteIfExists reportModule, reportName
        reportModule.CreateReport reportName, "Far Fields", "Rectangular Contour Plot", "Setup_10GHz : Sweep_Gate3", Array("Context:=", "InfiniteSphere_V117"), Array("Freq:=", Array(frequencyValue), "Theta:=", Array("All"), "Phi:=", Array("All")), Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", Array("re(rETheta)", "im(rETheta)", "re(rEPhi)", "im(rEPhi)")), Array()
        reportModule.ExportToFile reportName, fieldPath
        DeleteIfExists reportModule, reportName
        reportName = "V117_Eff_" & frequencyCode & "_" & portName
        efficiencyPath = outDir & "\\efficiency_" & LCase(portName) & "_" & frequencyCode & ".csv"
        reportModule.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : Sweep_Gate3", Array("Context:=", "InfiniteSphere_V117"), Array("Freq:=", Array(frequencyValue)), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
        reportModule.ExportToFile reportName, efficiencyPath
        DeleteIfExists reportModule, reportName
    Next
End Sub
Sub ApplySinglePort(solModule, selectedPort)
    Dim sources, editArgs(), i, sourceName, magnitude
    sources = solModule.GetAllSources()
    ReDim editArgs(UBound(sources) + 1)
    editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)
    For i = LBound(sources) To UBound(sources)
        sourceName = CStr(sources(i))
        magnitude = "0W"
        If LCase(Split(sourceName, ":")(0)) = LCase(selectedPort) Then magnitude = "1W"
        editArgs(i + 1) = Array("Name:=", sourceName, "Magnitude:=", magnitude, "Phase:=", "0deg")
    Next
    solModule.EditSources editArgs
End Sub
Sub DeleteIfExists(reportModule, reportName)
    On Error Resume Next
    reportModule.DeleteReports Array(reportName)
    On Error GoTo 0
End Sub
'''


def case_folder(out_dir: Path, replicate: int) -> Path:
    return out_dir / f"integrated_2x2_direct{replicate:02d}"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    out_dir = resolve(config["output_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "config_snapshot.json").exists():
        shutil.copy2(args.config, out_dir / "config_snapshot.json")
        shutil.copy2(resolve(config["trusted_antenna_protocol"]), out_dir / "antenna_protocol_snapshot.json")
    if args.replicate == 2:
        first = case_folder(out_dir, 1) / "analysis.json"
        if not first.exists() or not read_json(first).get("integrated_gate_pass"):
            raise RuntimeError("Independent repeat is locked until direct01 passes")
    folder = case_folder(out_dir, args.replicate)
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite integrated smoke: {folder}")
    folder.mkdir(parents=True)
    project = folder / f"v117_integrated_2x2_direct{args.replicate:02d}.aedt"
    touchstone = folder / f"v117_integrated_2x2_direct{args.replicate:02d}.s4p"
    antenna = read_json(resolve(config["trusted_antenna_protocol"]))
    build = folder / "build.vbs"
    solve = folder / "solve_export.vbs"
    build.write_text(builder_text(project, config, antenna), encoding="ascii")
    solve.write_text(solver_text(project, touchstone, folder, config["frequencies_ghz"]), encoding="ascii")
    manifest = {
        "replicate": args.replicate,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(build.resolve()),
        "solver_path": str(solve.resolve()),
        "pre_reference_ports": [f"PRE_{index}" for index in range(4)],
        "post_reference_plane": "boundary between compact network and physical fanout; no excitation port",
    }
    write_json(folder / "case_manifest.json", manifest)
    return {"prepared": True, "case": manifest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    out_dir = resolve(config["output_directory"])
    folder = case_folder(out_dir, args.replicate)
    manifest = read_json(folder / "case_manifest.json")
    touchstone = Path(manifest["touchstone_path"])
    if touchstone.exists() and touchstone.stat().st_size > 100:
        return {"status": "already_complete", "replicate": args.replicate}
    free_memory = memory_available_gb()
    minimum = float(config["mesh"]["minimum_free_memory_gb"])
    if math.isfinite(free_memory) and free_memory < minimum:
        raise MemoryError(f"Only {free_memory:.2f} GiB free; {minimum:.2f} GiB required")
    ansys = resolve(config["ansys_executable"])
    with (folder / "build.log").open("w", encoding="utf-8") as handle:
        build = subprocess.run(
            [str(ansys), "-RunScriptAndExit", manifest["builder_path"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    project_ready = wait_for_stable_file(Path(manifest["project_path"]))
    solve_code = None
    memory_guard_aborted = False
    minimum_free_memory_during_solve = None
    if build.returncode == 0 and project_ready:
        solve_code, memory_guard_aborted, observed = run_solver_with_memory_guard(
            [str(ansys), "-ng", "-RunScriptAndExit", manifest["solver_path"]],
            folder / "solve_export.log",
            float(config["mesh"]["abort_free_memory_gb"]),
        )
        minimum_free_memory_during_solve = observed if math.isfinite(observed) else None
    summary = {
        "replicate": args.replicate,
        "build_return_code": int(build.returncode),
        "project_ready_before_solve": project_ready,
        "solve_return_code": solve_code,
        "memory_guard_aborted": memory_guard_aborted,
        "minimum_free_memory_gb_during_solve": minimum_free_memory_during_solve,
        "free_memory_gb_before": free_memory,
    }
    write_json(folder / "run_summary.json", summary)
    return summary


def export_fields(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    out_dir = resolve(config["output_directory"])
    folder = case_folder(out_dir, args.replicate)
    manifest = read_json(folder / "case_manifest.json")
    project = Path(manifest["project_path"])
    touchstone = Path(manifest["touchstone_path"])
    if not project.exists() or not touchstone.exists():
        raise RuntimeError("A solved integrated project and S4 are required before field export")
    expected = [
        folder / f"eep_pre_{port}_{frequency_code(float(frequency))}.csv"
        for frequency in config["frequencies_ghz"]
        for port in range(4)
    ]
    if all(path.exists() and path.stat().st_size > 1000 for path in expected):
        return {"status": "already_complete", "field_count": len(expected)}
    script = folder / "export_fields_only.vbs"
    script.write_text(
        solver_text(project, touchstone, folder, config["frequencies_ghz"], solve_network=False),
        encoding="ascii",
    )
    ansys = resolve(config["ansys_executable"])
    code, memory_aborted, minimum_free = run_solver_with_memory_guard(
        [str(ansys), "-ng", "-RunScriptAndExit", str(script.resolve())],
        folder / "export_fields_only.log",
        float(config["mesh"]["abort_free_memory_gb"]),
    )
    summary = {
        "return_code": code,
        "memory_guard_aborted": memory_aborted,
        "minimum_free_memory_gb": minimum_free if math.isfinite(minimum_free) else None,
        "eep_files": sum(path.exists() and path.stat().st_size > 1000 for path in expected),
        "expected_eep_files": len(expected),
    }
    write_json(folder / "field_export_summary.json", summary)
    return summary


def numeric_rows(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        for row in reader:
            try:
                rows.append([float(value) for value in row])
            except (ValueError, TypeError):
                continue
    return header, np.asarray(rows, dtype=float)


def field_scale(header: str) -> float:
    lower = header.lower()
    if "[mv]" in lower:
        return 1.0e-3
    if "[uv]" in lower:
        return 1.0e-6
    return 1.0


def load_radiation_matrix(folder: Path, frequency: float) -> tuple[np.ndarray, dict[str, Any]]:
    code = frequency_code(frequency)
    fields = []
    theta_reference = None
    phi_reference = None
    for port in range(4):
        path = folder / f"eep_pre_{port}_{code}.csv"
        header, values = numeric_rows(path)
        if values.shape[1] < 6:
            raise RuntimeError(f"Incomplete EEP export: {path}")
        theta = values[:, 0]
        phi = values[:, 1]
        scale_theta = field_scale(header[2])
        scale_phi = field_scale(header[4])
        etheta = (values[:, 2] + 1j * values[:, 3]) * scale_theta
        ephi = (values[:, 4] + 1j * values[:, 5]) * scale_phi
        if theta_reference is None:
            theta_reference, phi_reference = theta, phi
        elif not np.allclose(theta_reference, theta) or not np.allclose(phi_reference, phi):
            raise RuntimeError("EEP angular grids differ between PRE ports")
        fields.append(np.column_stack((etheta, ephi)))
    theta = np.asarray(theta_reference)
    phi = np.asarray(phi_reference)
    keep = phi < 360.0 - 1.0e-9
    theta = theta[keep]
    phi = phi[keep]
    field = np.stack(fields, axis=1)[keep]
    theta_unique = np.unique(theta)
    phi_unique = np.unique(phi)
    dtheta = math.radians(float(np.median(np.diff(theta_unique))))
    dphi = math.radians(float(np.median(np.diff(phi_unique))))
    weights = np.sin(np.deg2rad(theta)) * dtheta * dphi
    radiation = np.zeros((4, 4), dtype=complex)
    for component in range(2):
        matrix = field[:, :, component]
        radiation += matrix.conj().T @ (weights[:, None] * matrix)
    radiation /= 2.0 * ETA0
    radiation = 0.5 * (radiation + radiation.conj().T)
    return radiation, {
        "theta_count": int(theta_unique.size),
        "phi_count_without_duplicate_360": int(phi_unique.size),
        "minimum_radiation_eigenvalue": float(np.min(np.linalg.eigvalsh(radiation)).real),
    }


def load_port_efficiencies(folder: Path, frequency: float) -> np.ndarray:
    code = frequency_code(frequency)
    values = []
    for port in range(4):
        _, rows = numeric_rows(folder / f"efficiency_pre_{port}_{code}.csv")
        if rows.size == 0:
            raise RuntimeError(f"Missing radiation efficiency for PRE_{port} at {frequency} GHz")
        values.append(float(rows[-1, -1]))
    return np.asarray(values)


def reordered_network(path: Path, desired: list[str], nports: int) -> tuple[np.ndarray, np.ndarray]:
    frequencies, network = parse_touchstone(path, nports)
    names = touchstone_port_names(path)
    if len(names) != nports or set(names) != set(desired):
        raise RuntimeError(f"Unexpected port names in {path}: {names}")
    order = [names.index(name) for name in desired]
    return frequencies, network[:, order][:, :, order]


def analyze_case(folder: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(folder / "case_manifest.json")
    run_summary = read_json(folder / "run_summary.json") if (folder / "run_summary.json").exists() else {}
    field_summary = (
        read_json(folder / "field_export_summary.json")
        if (folder / "field_export_summary.json").exists()
        else {}
    )
    result = {
        **run_summary,
        **profile_metrics(folder),
        "field_export_return_code": field_summary.get("return_code"),
        "field_export_file_count": field_summary.get("eep_files", 0),
        "field_export_recovered": bool(
            field_summary.get("return_code") == 0 and field_summary.get("eep_files") == 12
        ),
    }
    touchstone = Path(manifest["touchstone_path"])
    if not touchstone.exists() or touchstone.stat().st_size < 100:
        result["electromagnetic_solve_complete"] = False
        result["integrated_gate_pass"] = False
        write_json(folder / "analysis.json", result)
        return result
    desired_pre = manifest["pre_reference_ports"]
    frequencies, integrated_s4 = reordered_network(touchstone, desired_pre, 4)
    antenna_frequencies, antenna_s4 = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    desired_s8 = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    feed_frequencies, feed_s8 = reordered_network(resolve(config["validated_feed_s8"]), desired_s8, 8)
    if not (
        np.allclose(frequencies, config["frequencies_ghz"], atol=1.0e-9)
        and np.allclose(frequencies, antenna_frequencies, atol=1.0e-9)
        and np.allclose(frequencies, feed_frequencies, atol=1.0e-9)
    ):
        raise RuntimeError("Integrated, antenna, and feed frequency grids do not match")
    cascade_s4 = np.stack(
        [terminate_network(feed_s8[index], antenna_s4[index])[0] for index in range(len(frequencies))]
    )
    integrated_vs_cascade = float(np.max(np.abs(integrated_s4 - cascade_s4)))
    reciprocity = float(np.max(np.abs(integrated_s4 - np.transpose(integrated_s4, (0, 2, 1)))))
    passivity = float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in integrated_s4))
    rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    frequency_rows = []
    source_rows = []
    eep_power_errors = []
    for frequency_index, frequency in enumerate(frequencies):
        s_matrix = integrated_s4[frequency_index]
        radiation, radiation_meta = load_radiation_matrix(folder, float(frequency))
        efficiencies = load_port_efficiencies(folder, float(frequency))
        accepted_per_port = 1.0 - np.sum(np.abs(s_matrix) ** 2, axis=0)
        expected_radiated_per_port = efficiencies * accepted_per_port
        eep_diagonal = np.real(np.diag(radiation))
        errors = np.abs(eep_diagonal - expected_radiated_per_port) / np.maximum(
            np.abs(expected_radiated_per_port), EPS
        )
        eep_power_errors.extend(errors.tolist())
        indices = [
            index for index, row in enumerate(rows)
            if abs(float(row["frequency_ghz"]) - float(frequency)) <= 1.0e-9
        ]
        sources = vectors[indices].T
        active = considered[indices].T
        reflected = s_matrix @ sources
        gamma = np.where(active, np.abs(reflected) / np.maximum(np.abs(sources), EPS), 0.0)
        active_rl = -20.0 * np.log10(np.maximum(np.max(gamma, axis=0), EPS))
        incident = np.sum(np.abs(sources) ** 2, axis=0)
        reflected_power = np.sum(np.abs(reflected) ** 2, axis=0)
        accepted = incident - reflected_power
        radiated = np.real(np.einsum("in,ij,jn->n", sources.conj(), radiation, sources))
        integrated_efficiency = radiated / np.maximum(accepted, EPS)
        transducer = radiated / np.maximum(incident, EPS)
        total_rl = -10.0 * np.log10(np.maximum(reflected_power / incident, EPS))
        passive_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s_matrix)), EPS))
        frequency_rows.append(
            {
                "frequency_ghz": float(frequency),
                "passive_rl_min_db": float(np.min(passive_rl)),
                "active_rl_min_db": float(np.min(active_rl)),
                "total_rl_min_db": float(np.min(total_rl)),
                "integrated_accepted_to_radiated_efficiency_min": float(np.min(integrated_efficiency)),
                "integrated_transducer_efficiency_min": float(np.min(transducer)),
                "single_port_radiation_efficiency_min": float(np.min(efficiencies)),
                "eep_power_relative_error_max": float(np.max(errors)),
                **radiation_meta,
            }
        )
        for local, global_index in enumerate(indices):
            source_rows.append(
                {
                    **rows[global_index],
                    "active_rl_db": float(active_rl[local]),
                    "total_rl_db": float(total_rl[local]),
                    "incident_power": float(incident[local]),
                    "accepted_power": float(accepted[local]),
                    "radiated_power_from_eep": float(radiated[local]),
                    "integrated_accepted_to_radiated_efficiency": float(integrated_efficiency[local]),
                    "integrated_transducer_efficiency": float(transducer[local]),
                }
            )
    summary = {
        "passive_rl_min_db": min(row["passive_rl_min_db"] for row in frequency_rows),
        "active_rl_min_db": min(row["active_rl_min_db"] for row in frequency_rows),
        "total_rl_min_db": min(row["total_rl_min_db"] for row in frequency_rows),
        "integrated_accepted_to_radiated_efficiency_min": min(
            row["integrated_accepted_to_radiated_efficiency_min"] for row in frequency_rows
        ),
        "integrated_transducer_efficiency_min": min(
            row["integrated_transducer_efficiency_min"] for row in frequency_rows
        ),
        "single_port_radiation_efficiency_min": min(
            row["single_port_radiation_efficiency_min"] for row in frequency_rows
        ),
        "eep_power_relative_error_max": max(eep_power_errors),
        "integrated_vs_cascade_max_abs_delta_s": integrated_vs_cascade,
        "reciprocity_error_max": reciprocity,
        "passivity_sigma_max": passivity,
        "network_insertion_efficiency_from_validated_s8": read_json(
            resolve(config["validated_feed_analysis"])
        )["actual_load_insertion_efficiency_min"],
    }
    result.update(summary)
    result["electromagnetic_solve_complete"] = bool(
        result.get("converged") is True and touchstone.exists() and result["field_export_recovered"]
    )
    gates = config["gates"]
    result["engineering_gate_pass"] = bool(
        result.get("converged") is True
        and float(result.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
        and reciprocity <= float(gates["maximum_reciprocity_error"])
        and passivity <= float(gates["maximum_passivity_sigma"])
        and summary["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
        and summary["active_rl_min_db"] >= float(gates["minimum_active_rl_db"])
        and summary["total_rl_min_db"] >= float(gates["minimum_total_rl_db"])
        and summary["integrated_accepted_to_radiated_efficiency_min"]
        >= float(gates["minimum_integrated_accepted_to_radiated_efficiency"])
        and summary["integrated_transducer_efficiency_min"]
        >= float(gates["minimum_integrated_transducer_efficiency"])
        and summary["eep_power_relative_error_max"] <= float(gates["maximum_eep_power_relative_error"])
        and integrated_vs_cascade <= float(gates["maximum_integrated_vs_cascade_abs_delta_s"])
    )
    result["design_reserve_gate_pass"] = bool(
        result["engineering_gate_pass"]
        and summary["active_rl_min_db"] >= float(gates["design_active_rl_db"])
    )
    result["integrated_gate_pass"] = result["engineering_gate_pass"]
    write_csv(folder / "frequency_metrics.csv", frequency_rows)
    write_csv(folder / "stimulus_metrics.csv", source_rows)
    write_json(folder / "analysis.json", result)
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    out_dir = resolve(config["output_directory"])
    results = []
    for folder in sorted(out_dir.glob("integrated_2x2_direct*")):
        if (folder / "case_manifest.json").exists():
            results.append(analyze_case(folder, config))
    first = results[0] if results else None
    second = results[1] if len(results) > 1 else None
    repeat_delta = None
    repeat_pass = False
    if first and second:
        first_manifest = read_json(case_folder(out_dir, 1) / "case_manifest.json")
        second_manifest = read_json(case_folder(out_dir, 2) / "case_manifest.json")
        _, first_s = reordered_network(Path(first_manifest["touchstone_path"]), first_manifest["pre_reference_ports"], 4)
        _, second_s = reordered_network(Path(second_manifest["touchstone_path"]), second_manifest["pre_reference_ports"], 4)
        repeat_delta = float(np.max(np.abs(first_s - second_s)))
        repeat_pass = bool(
            second.get("integrated_gate_pass")
            and repeat_delta <= float(config["gates"]["maximum_independent_repeat_abs_delta_s"])
        )
    decision = {
        "first_integrated_gate_pass": bool(first and first.get("integrated_gate_pass")),
        "first_design_reserve_gate_pass": bool(first and first.get("design_reserve_gate_pass")),
        "allow_independent_repeat": bool(first and first.get("integrated_gate_pass") and second is None),
        "independent_repeat_max_abs_delta_s": repeat_delta,
        "independent_repeat_pass": repeat_pass,
        "allow_4x4": repeat_pass,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out_dir / "stage_decision.json", decision)
    write_csv(out_dir / "case_metrics.csv", results)
    return {"results": results, "decision": decision}


def status(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    out_dir = resolve(config["output_directory"])
    cases = []
    if out_dir.exists():
        for folder in sorted(out_dir.glob("integrated_2x2_direct*")):
            manifest_path = folder / "case_manifest.json"
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                cases.append(
                    {
                        "replicate": manifest["replicate"],
                        "project_exists": Path(manifest["project_path"]).exists(),
                        "touchstone_exists": Path(manifest["touchstone_path"]).exists(),
                        "analysis_exists": (folder / "analysis.json").exists(),
                    }
                )
    return {
        "cases": cases,
        "decision": read_json(out_dir / "stage_decision.json") if (out_dir / "stage_decision.json").exists() else None,
        "free_memory_gb": memory_available_gb(),
    }


def main() -> None:
    args = parse_args()
    result = {
        "prepare": prepare,
        "run": run,
        "export": export_fields,
        "analyze": analyze,
        "status": status,
    }[args.mode](args)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
