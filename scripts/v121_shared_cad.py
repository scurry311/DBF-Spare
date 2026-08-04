#!/usr/bin/env python3
"""Shared v1.21 physical CAD generators for network-only and integrated 2x2 gates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from design_v120_joint_feed_fanout_sparse_graph import unpack
from run_v117_integrated_2x2_smoke import helpers_vbs, vp
from run_v120_sparse_graph_physical_front_gate import (
    builder_text as v120_network_builder_text,
    rlc_assignment,
    vb_bool,
)


ROOT = Path(__file__).resolve().parents[1]
NETWORK_DESIGN_NAME = "V120_SparseGraph_PhysicalFront_S8"
INTEGRATED_DESIGN_NAME = "V121_ParametricFeedPost_Integrated2x2"
EPS = 1.0e-15


def resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_map(config: dict[str, Any], values: dict[str, float] | None = None) -> dict[str, float]:
    result = {item["name"]: float(item["nominal"]) for item in config["variables"]}
    if values:
        unknown = set(values) - set(result)
        if unknown:
            raise KeyError(f"Unknown v1.21 parameters: {sorted(unknown)}")
        result.update({key: float(value) for key, value in values.items()})
    validate_parameters(config, result)
    return result


def validate_parameters(config: dict[str, Any], values: dict[str, float]) -> None:
    ranges = {item["name"]: item for item in config["variables"]}
    for name, value in values.items():
        item = ranges[name]
        if value < float(item["minimum"]) - EPS or value > float(item["maximum"]) + EPS:
            raise ValueError(f"{name}={value:g} is outside [{item['minimum']}, {item['maximum']}]")
    constraints = config["geometric_constraints"]
    clearance = float(values["ground_clearance_radius_mm"]) - float(values["probe_radius_mm"])
    if clearance < float(constraints["minimum_clearance_over_probe_mm"]):
        raise ValueError("Ground clearance is too close to the probe radius")
    pad_margin = float(values["launch_pad_radius_mm"]) - 0.5 * float(values["common_trace_width_mm"])
    if pad_margin < float(constraints["minimum_pad_over_trace_half_width_mm"]):
        raise ValueError("Launch pad radius is too small for the selected trace width")


def channel_positions(config: dict[str, Any], values: dict[str, float]) -> list[float]:
    topology = config["fixed_topology"]
    order = [int(item) for item in topology["port_order_from_negative_y"]]
    width = float(values["common_trace_width_mm"])
    gaps = [
        float(values["outer_edge_gap_mm"]),
        float(values["center_edge_gap_mm"]),
        float(values["outer_edge_gap_mm"]),
    ]
    coordinates = [0.0]
    for gap in gaps:
        coordinates.append(coordinates[-1] + width + gap)
    center = 0.5 * (coordinates[0] + coordinates[-1])
    result = [0.0] * 4
    for index, port in enumerate(order):
        result[port] = coordinates[index] - center
    return result


def physical_block(config: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    topology = config["fixed_topology"]
    length = float(topology["base_trace_length_mm"]) + float(values["common_post_length_delta_mm"])
    width = float(values["common_trace_width_mm"])
    positions = channel_positions(config, values)
    board_width = max(5.2, max(positions) - min(positions) + width + 1.6)
    offset = float(values["shunt_reference_offset_mm"])
    return {
        "substrate_relative_permittivity": float(topology["network_substrate_relative_permittivity"]),
        "substrate_loss_tangent": float(topology["network_substrate_loss_tangent"]),
        "substrate_thickness_mm": float(topology["network_substrate_thickness_mm"]),
        "copper_thickness_mm": float(topology["network_copper_thickness_mm"]),
        "board_length_mm": length + 2.0,
        "board_width_mm": board_width,
        "trace_length_mm": length,
        "trace_width_mm_by_port": [width] * 4,
        "port_order_from_negative_y": [int(item) for item in topology["port_order_from_negative_y"]],
        "adjacent_gap_mm": [
            float(values["outer_edge_gap_mm"]),
            float(values["center_edge_gap_mm"]),
            float(values["outer_edge_gap_mm"]),
        ],
        "series_plane_x_mm": 0.0,
        "input_shunt_x_mm": -length / 2.0 + offset,
        "input_graph_x_mm": -length / 2.0 + offset + 0.30,
        "output_graph_x_mm": length / 2.0 - offset - 0.30,
        "output_shunt_x_mm": length / 2.0 - offset,
        "component_sheet_width_mm": float(topology["component_sheet_width_mm"]),
        "mesh_max_length_mm": float(config["mesh"]["network_component_max_length_mm"]),
        "adaptive_refinement_percent": float(config["mesh"]["adaptive_refinement_percent"]),
        "maximum_passes": int(config["mesh"]["maximum_passes"]),
        "minimum_converged_passes": int(config["mesh"]["minimum_converged_passes"]),
        "minimum_free_memory_gb": float(config["resources"]["minimum_free_memory_before_hfss_gib"]),
    }


def network_only_builder_text(
    project: Path,
    config: dict[str, Any],
    values: dict[str, float],
    frequencies_ghz: list[float],
    solver_type: str = "auto",
) -> str:
    local = {
        "synthesis_config": config["synthesis_config"],
        "synthesis_summary": config["synthesis_summary"],
        "physical_block": physical_block(config, values),
    }
    text = v120_network_builder_text(project, local)
    start = float(min(frequencies_ghz))
    stop = float(max(frequencies_ghz))
    count = len(frequencies_ghz)
    text = text.replace('"RangeStart:=", "9.96GHz"', f'"RangeStart:=", "{start:g}GHz"')
    text = text.replace('"RangeEnd:=", "10.04GHz"', f'"RangeEnd:=", "{stop:g}GHz"')
    text = text.replace('"RangeCount:=", 3', f'"RangeCount:=", {count}')
    solver_options = {
        "auto": "",
        "direct": ', "DrivenSolverType:=", "Direct Solver"',
        "iterative": ', "DrivenSolverType:=", "Iterative Solver", "IterativeResidual:=", 0.000001',
        "ddm": ', "DrivenSolverType:=", "Domain Decomposition", "IterativeResidual:=", 0.000001, "DDMSolverResidual:=", 0.000001',
    }
    if solver_type not in solver_options:
        raise ValueError(f"Unsupported HFSS solver type: {solver_type}")
    if solver_options[solver_type]:
        text = text.replace('"PortAccuracy:=", 2)', f'"PortAccuracy:=", 2{solver_options[solver_type]})')
    return text


def solver_text(project: Path, touchstone: Path, design_name: str) -> str:
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, vars, variation
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{design_name}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:Sweep_Gate3")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Gate3"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def integrated_solver_text(
    project: Path,
    touchstone: Path,
    folder: Path,
    frequencies_ghz: list[float],
) -> str:
    exports = []
    for frequency in frequencies_ghz:
        code = f"{frequency:.2f}".replace(".", "p")
        exports.append(f'ExportPortFields oSolutions, oReport, ports, "{frequency:g}GHz", "{vp(folder)}", "{code}"')
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, ports
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{INTEGRATED_DESIGN_NAME}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")
vars = oSolutions.ListVariations("Setup_10GHz:Sweep_Gate3")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Gate3"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
ports = Array("PRE_0", "PRE_1", "PRE_2", "PRE_3")
{chr(10).join(exports)}
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ExportPortFields(solModule, reportModule, portArray, frequencyValue, outDir, frequencyCode)
    Dim i, portName, reportName, fieldPath, efficiencyPath
    For i = LBound(portArray) To UBound(portArray)
        portName = CStr(portArray(i))
        ApplySinglePort solModule, portName
        reportName = "V121_EEP_" & frequencyCode & "_" & portName
        fieldPath = outDir & "\\eep_" & LCase(portName) & "_" & frequencyCode & ".csv"
        DeleteIfExists reportModule, reportName
        reportModule.CreateReport reportName, "Far Fields", "Rectangular Contour Plot", "Setup_10GHz : Sweep_Gate3", Array("Context:=", "InfiniteSphere_V121"), Array("Freq:=", Array(frequencyValue), "Theta:=", Array("All"), "Phi:=", Array("All")), Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", Array("re(rETheta)", "im(rETheta)", "re(rEPhi)", "im(rEPhi)")), Array()
        reportModule.ExportToFile reportName, fieldPath
        DeleteIfExists reportModule, reportName
        reportName = "V121_Eff_" & frequencyCode & "_" & portName
        efficiencyPath = outDir & "\\efficiency_" & LCase(portName) & "_" & frequencyCode & ".csv"
        reportModule.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : Sweep_Gate3", Array("Context:=", "InfiniteSphere_V121"), Array("Freq:=", Array(frequencyValue)), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
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


def _route_plan(
    config: dict[str, Any],
    values: dict[str, float],
    y_channels: list[float],
    post_x: float,
) -> list[dict[str, Any]]:
    antenna = read_json(resolve(config["trusted_antenna_protocol"]))
    physical = antenna["physical_topology"]
    synthesis = read_json(resolve(config["synthesis_summary"]))
    route_lengths = [float(item) + float(values["common_post_length_delta_mm"]) for item in synthesis["physical_translation"]["route_length_mm_by_port"]]
    spacing = float(physical["spacing_mm"])
    patch_l = float(physical["patch_length_mm"])
    lanes = [1.0, 2.5, 3.5, 5.0]
    routes: list[dict[str, Any]] = []
    for port in range(4):
        ix, iy = divmod(port, 2)
        patch_x = (ix - 0.5) * spacing
        patch_y = (iy - 0.5) * spacing
        feed_x = patch_x + float(values["feed_x_offset_mm"])
        feed_y = patch_y - patch_l / 2.0 + float(values["feed_inset_mm"])
        lane = lanes[port]
        base = abs(lane - post_x) + abs(feed_x - lane) + abs(feed_y - y_channels[port])
        target = route_lengths[port]
        if target < base - 1.0e-8:
            raise ValueError(f"Port {port} route target {target:.4f} mm is shorter than base {base:.4f} mm")
        detour = 0.5 * (target - base)
        detour_y = min(y_channels[port], feed_y) - detour if iy == 0 else max(y_channels[port], feed_y) + detour
        points = [(post_x, y_channels[port]), (lane, y_channels[port]), (lane, detour_y), (feed_x, detour_y), (feed_x, feed_y)]
        routes.append({"port": port, "target_length_mm": target, "feed_x_mm": feed_x, "feed_y_mm": feed_y, "points_mm": points})
    return routes


def integrated_builder_text(project: Path, config: dict[str, Any], values: dict[str, float]) -> str:
    antenna = read_json(resolve(config["trusted_antenna_protocol"]))
    physical = antenna["physical_topology"]
    patch_candidate = antenna["one_by_one_candidates"][0]
    synthesis_config = read_json(resolve(config["synthesis_config"]))
    synthesis = read_json(resolve(config["synthesis_summary"]))
    graph = [tuple(int(item) for item in pair) for pair in config["fixed_topology"]["manufacturable_graph_pairs"]]
    _, series_ground, _, input_ground, input_pair, output_ground, output_pair = unpack(np.asarray(synthesis["optimized_parameters"], dtype=float))
    q = synthesis_config["network"]
    block = physical_block(config, values)
    y_channels = channel_positions(config, values)
    trace_l = float(block["trace_length_mm"])
    stage_center_x = -9.0
    pre_x = stage_center_x - trace_l / 2.0
    post_x = stage_center_x + trace_l / 2.0
    series_x = stage_center_x
    input_shunt_x = pre_x + float(values["shunt_reference_offset_mm"])
    input_graph_x = input_shunt_x + 0.30
    output_shunt_x = post_x - float(values["shunt_reference_offset_mm"])
    output_graph_x = output_shunt_x - 0.30
    board = 30.0
    antenna_h = float(physical["substrate_thickness_mm"])
    antenna_copper = float(physical["copper_thickness_mm"])
    network_h = float(config["fixed_topology"]["network_substrate_thickness_mm"])
    network_copper = float(config["fixed_topology"]["network_copper_thickness_mm"])
    ground_bottom = -antenna_h - antenna_copper
    signal_top = ground_bottom - network_h
    signal_bottom = signal_top - network_copper
    trace_w = float(values["common_trace_width_mm"])
    sheet_w = float(config["fixed_topology"]["component_sheet_width_mm"])
    probe_r = float(values["probe_radius_mm"])
    clearance_r = float(values["ground_clearance_radius_mm"])
    pad_r = float(values["launch_pad_radius_mm"])
    launch_l = float(values["launch_taper_length_mm"])
    patch_w = float(physical["patch_width_mm"])
    patch_l = float(physical["patch_length_mm"])
    slot_l = float(patch_candidate["slot_length_mm"])
    slot_w = float(patch_candidate["slot_width_mm"])
    tongue = float(patch_candidate["tongue_width_mm"])
    routes = _route_plan(config, values, y_channels, post_x)
    route_by_port = {int(item["port"]): item for item in routes}
    geometry: list[str] = []
    assignments: list[str] = []
    component_names: list[str] = []
    fanout_names: list[str] = []

    for port in range(4):
        y = y_channels[port]
        left = f"TraceLeft_{port}"
        right = f"TraceRight_{port}"
        series_sheet = f"SeriesSheet_{port}"
        pre_sheet = f"PrePortSheet_{port}"
        geometry.extend([
            f'CreateBox oEditor, "{left}", {pre_x:.7f}, {y-trace_w/2:.7f}, {signal_bottom:.7f}, {series_x-sheet_w/2-pre_x:.7f}, {trace_w:.7f}, {network_copper:.7f}, "copper", False',
            f'CreateBox oEditor, "{right}", {series_x+sheet_w/2:.7f}, {y-trace_w/2:.7f}, {signal_bottom:.7f}, {post_x-series_x-sheet_w/2:.7f}, {trace_w:.7f}, {network_copper:.7f}, "copper", False',
            f'CreateSheet oEditor, "{series_sheet}", "Z", {series_x-sheet_w/2:.7f}, {y-trace_w/2:.7f}, {signal_top:.7f}, {sheet_w:.7f}, {trace_w:.7f}',
            f'CreateSheet oEditor, "{pre_sheet}", "X", {pre_x:.7f}, {y-trace_w/2:.7f}, {signal_top:.7f}, {trace_w:.7f}, {network_h:.7f}',
        ])
        item = rlc_assignment(float(series_ground[port]), "series", float(q["series_inductor_q"]), float(q["series_capacitor_q"]))
        assignments.extend([
            f'AssignRLCExpr oBoundary, "Series_{port}", "{series_sheet}", "Serial", True, "{item["r_ohm"]:.10f}ohm", {vb_bool(item["use_l"])}, "{item["l_nh"]:.10f}nH", {vb_bool(item["use_c"])}, "{item["c_pf"]:.10f}pF", {series_x-sheet_w/2:.7f}, {y:.7f}, {signal_top:.7f}, {series_x+sheet_w/2:.7f}, {y:.7f}, {signal_top:.7f}',
            f'AssignPort oBoundary, "PRE_{port}", "{pre_sheet}", {pre_x:.7f}, {y:.7f}, {signal_top:.7f}, {pre_x:.7f}, {y:.7f}, {ground_bottom:.7f}',
        ])
        component_names.extend([left, right, series_sheet])

    for plane, plane_x, values_array in (
        ("Input", input_shunt_x, input_ground),
        ("Output", output_shunt_x, output_ground),
    ):
        for port, value in enumerate(values_array):
            if abs(float(value)) <= EPS:
                continue
            y = y_channels[port]
            name = f"{plane}GroundSheet_{port}"
            item = rlc_assignment(float(value), "shunt", float(q["shunt_inductor_q"]), float(q["shunt_capacitor_q"]))
            geometry.append(f'CreateSheet oEditor, "{name}", "X", {plane_x:.7f}, {y-trace_w/4:.7f}, {signal_top:.7f}, {trace_w/2:.7f}, {network_h:.7f}')
            assignments.append(f'AssignRLCExpr oBoundary, "{plane}Ground_{port}", "{name}", "Parallel", True, "{item["r_ohm"]:.10f}ohm", {vb_bool(item["use_l"])}, "{item["l_nh"]:.10f}nH", {vb_bool(item["use_c"])}, "{item["c_pf"]:.10f}pF", {plane_x:.7f}, {y:.7f}, {signal_top:.7f}, {plane_x:.7f}, {y:.7f}, {ground_bottom:.7f}')
            component_names.append(name)

    for plane, plane_x, values_array in (
        ("Input", input_graph_x, input_pair),
        ("Output", output_graph_x, output_pair),
    ):
        for edge, (value, pair) in enumerate(zip(values_array, graph)):
            if abs(float(value)) <= EPS:
                continue
            first, second = pair
            low, high = sorted((y_channels[first], y_channels[second]))
            name = f"{plane}GraphSheet_{edge}"
            item = rlc_assignment(float(value), "shunt", float(q["shunt_inductor_q"]), float(q["shunt_capacitor_q"]))
            height = high - low - trace_w
            geometry.append(f'CreateSheet oEditor, "{name}", "Z", {plane_x-sheet_w/2:.7f}, {low+trace_w/2:.7f}, {signal_top:.7f}, {sheet_w:.7f}, {height:.7f}')
            assignments.append(f'AssignRLCExpr oBoundary, "{plane}Graph_{edge}", "{name}", "Parallel", True, "{item["r_ohm"]:.10f}ohm", {vb_bool(item["use_l"])}, "{item["l_nh"]:.10f}nH", {vb_bool(item["use_c"])}, "{item["c_pf"]:.10f}pF", {plane_x:.7f}, {low+trace_w/2:.7f}, {signal_top:.7f}, {plane_x:.7f}, {high-trace_w/2:.7f}, {signal_top:.7f}')
            component_names.append(name)

    for port in range(4):
        route = route_by_port[port]
        points = [tuple(point) for point in route["points_mm"]]
        right = f"TraceRight_{port}"
        for index, (first, second) in enumerate(zip(points, points[1:])):
            x1, y1 = first
            x2, y2 = second
            name = f"Route_{port}_{index}"
            if abs(y2 - y1) <= EPS:
                geometry.append(f'CreateBox oEditor, "{name}", {min(x1,x2)-trace_w/2:.7f}, {y1-trace_w/2:.7f}, {signal_bottom:.7f}, {abs(x2-x1)+trace_w:.7f}, {trace_w:.7f}, {network_copper:.7f}, "copper", False')
            else:
                geometry.append(f'CreateBox oEditor, "{name}", {x1-trace_w/2:.7f}, {min(y1,y2)-trace_w/2:.7f}, {signal_bottom:.7f}, {trace_w:.7f}, {abs(y2-y1)+trace_w:.7f}, {network_copper:.7f}, "copper", False')
            geometry.append(f'UniteObjects oEditor, "{right}", "{name}"')
        feed_x = float(route["feed_x_mm"])
        feed_y = float(route["feed_y_mm"])
        iy = port % 2
        sign = 1.0 if iy == 1 else -1.0
        launch_start = feed_y - sign * launch_l
        launch_mid = feed_y - sign * launch_l / 2.0
        mid_w = 0.5 * (trace_w + 2.0 * pad_r)
        geometry.extend([
            f'CreateBox oEditor, "LaunchNarrow_{port}", {feed_x-trace_w/2:.7f}, {min(launch_start,launch_mid)-trace_w/2:.7f}, {signal_bottom:.7f}, {trace_w:.7f}, {abs(launch_mid-launch_start)+trace_w:.7f}, {network_copper:.7f}, "copper", False',
            f'UniteObjects oEditor, "{right}", "LaunchNarrow_{port}"',
            f'CreateBox oEditor, "LaunchWide_{port}", {feed_x-mid_w/2:.7f}, {min(launch_mid,feed_y)-mid_w/2:.7f}, {signal_bottom:.7f}, {mid_w:.7f}, {abs(feed_y-launch_mid)+mid_w:.7f}, {network_copper:.7f}, "copper", False',
            f'UniteObjects oEditor, "{right}", "LaunchWide_{port}"',
            f'CreateCylinderZ oEditor, "LaunchPad_{port}", {feed_x:.7f}, {feed_y:.7f}, {signal_bottom:.7f}, {pad_r:.7f}, {network_copper:.7f}, "copper", False',
            f'UniteObjects oEditor, "{right}", "LaunchPad_{port}"',
        ])
        patch_x = (-0.5 if port < 2 else 0.5) * float(physical["spacing_mm"])
        patch_y = (-0.5 if port % 2 == 0 else 0.5) * float(physical["spacing_mm"])
        patch_bottom = patch_y - patch_l / 2.0
        slot_offset = 0.5 * (tongue + slot_w)
        patch_name = f"Patch_{port}"
        probe_name = f"Probe_{port}"
        geometry.extend([
            f'CreateCylinderZ oEditor, "GroundHole_{port}", {feed_x:.7f}, {feed_y:.7f}, {ground_bottom-0.01:.7f}, {clearance_r:.7f}, {antenna_copper+0.02:.7f}, "vacuum", True',
            f'SubtractObject oEditor, "SharedGround", "GroundHole_{port}"',
            f'CreateCylinderZ oEditor, "NetworkHole_{port}", {feed_x:.7f}, {feed_y:.7f}, {signal_top-0.01:.7f}, {probe_r+0.01:.7f}, {network_h+0.02:.7f}, "vacuum", True',
            f'SubtractObject oEditor, "NetworkSubstrate", "NetworkHole_{port}"',
            f'CreateCylinderZ oEditor, "AntennaHole_{port}", {feed_x:.7f}, {feed_y:.7f}, {-antenna_h-0.01:.7f}, {probe_r+0.01:.7f}, {antenna_h+0.02:.7f}, "vacuum", True',
            f'SubtractObject oEditor, "AntennaSubstrate", "AntennaHole_{port}"',
            f'CreateBox oEditor, "{patch_name}", {patch_x-patch_w/2:.7f}, {patch_bottom:.7f}, 0, {patch_w:.7f}, {patch_l:.7f}, {antenna_copper:.7f}, "copper", False',
            f'CreateBox oEditor, "SlotL_{port}", {feed_x-slot_offset-slot_w/2:.7f}, {patch_bottom-0.01:.7f}, -0.01, {slot_w:.7f}, {slot_l+0.02:.7f}, {antenna_copper+0.02:.7f}, "vacuum", True',
            f'SubtractObject oEditor, "{patch_name}", "SlotL_{port}"',
            f'CreateBox oEditor, "SlotR_{port}", {feed_x+slot_offset-slot_w/2:.7f}, {patch_bottom-0.01:.7f}, -0.01, {slot_w:.7f}, {slot_l+0.02:.7f}, {antenna_copper+0.02:.7f}, "vacuum", True',
            f'SubtractObject oEditor, "{patch_name}", "SlotR_{port}"',
            f'CreateCylinderZ oEditor, "{probe_name}", {feed_x:.7f}, {feed_y:.7f}, {signal_bottom:.7f}, {probe_r:.7f}, {-signal_bottom+antenna_copper:.7f}, "copper", False',
            f'UniteObjects oEditor, "{patch_name}", "{probe_name}"',
            f'UniteObjects oEditor, "{right}", "{patch_name}"',
        ])
        fanout_names.append(right)

    component_items = ", ".join(f'"{name}"' for name in component_names)
    fanout_items = ", ".join(f'"{name}"' for name in fanout_names)
    mesh = config["mesh"]
    frequencies = [float(item) for item in config["frequencies_ghz"]]
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V121_Antenna", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{physical['relative_permittivity']}", "dielectric_loss_tangent:=", "{physical['loss_tangent']}")
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V121_Network", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{config['fixed_topology']['network_substrate_relative_permittivity']}", "dielectric_loss_tangent:=", "{config['fixed_topology']['network_substrate_loss_tangent']}")
oProject.InsertDesign "HFSS", "{INTEGRATED_DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{INTEGRATED_DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "AntennaSubstrate", {-board/2:.7f}, {-board/2:.7f}, {-antenna_h:.7f}, {board:.7f}, {board:.7f}, {antenna_h:.7f}, "RO5880_V121_Antenna", True
CreateBox oEditor, "SharedGround", {-board/2:.7f}, {-board/2:.7f}, {ground_bottom:.7f}, {board:.7f}, {board:.7f}, {antenna_copper:.7f}, "copper", False
CreateBox oEditor, "NetworkSubstrate", {-board/2:.7f}, {-board/2:.7f}, {signal_top:.7f}, {board:.7f}, {board:.7f}, {network_h:.7f}, "RO5880_V121_Network", True
{chr(10).join(geometry)}
{chr(10).join(assignments)}
CreateBox oEditor, "AirRegion", {-board/2-15:.7f}, {-board/2-15:.7f}, -16, {board+30:.7f}, {board+30:.7f}, 32, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:V121_NetworkComponentMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({component_items}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(mesh['network_component_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
oMesh.AssignLengthOp Array("NAME:V121_FanoutAndFeedMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({fanout_items}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{min(float(mesh['fanout_and_probe_max_length_mm']), float(mesh['antenna_feed_max_length_mm'])):.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", {float(config['gates']['maximum_final_delta_s']):.7f}, "MaximumPasses:=", {int(mesh['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", {int(mesh['minimum_converged_passes'])}, "PercentRefinement:=", {float(mesh['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "PortAccuracy:=", 2)
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "{min(frequencies):g}GHz", "RangeEnd:=", "{max(frequencies):g}GHz", "RangeCount:=", {len(frequencies)}, "Type:=", "Discrete", "SaveFields:=", True, "SaveRadFields:=", True, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V121", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{helpers_vbs()}
'''
