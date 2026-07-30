#!/usr/bin/env python3
"""Build, solve, and gate one physical dual-reference-plane v1.15 S8 fixture."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v115_grounded_modal_network.json"
DEFAULT_DESIGN = ROOT / "hfss_outputs" / "v115_grounded_modal_network_20260730_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v115_physical_modal_feed_fixture_20260730_run01"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "V115_Physical_Modal_Feed_S8"
EPS = 1.0e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="status")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--physical-design", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def touchstone_port_names(path: Path) -> list[str]:
    names: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"!\s*Port\[(\d+)\]\s*=\s*(\S+)", line)
        if match:
            names[int(match.group(1)) - 1] = match.group(2)
    return [names[index] for index in sorted(names)]


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def vb_bool(value: bool) -> str:
    return "True" if value else "False"


def helpers_vbs() -> str:
    return r'''Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(190 120 50)", "Transparency:=", 0.1, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheet(editor, objName, axisName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", axisName), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub AssignPort(boundary, portName, sheetName, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Sub AssignRLC(boundary, boundaryName, sheetName, rlcType, useR, resistanceOhm, useL, inductanceNh, useC, capacitancePf, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedRLC Array("NAME:" & boundaryName, "Objects:=", Array(sheetName), Array("NAME:CurrentLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "RLC Type:=", rlcType, "UseResist:=", useR, "Resistance:=", CStr(resistanceOhm) & "ohm", "UseInduct:=", useL, "Inductance:=", CStr(inductanceNh) & "nH", "UseCap:=", useC, "Capacitance:=", CStr(capacitancePf) & "pF")
End Sub
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function'''


def builder_text(project: Path, protocol: dict[str, Any], selected: dict[str, Any]) -> str:
    fixture = {**protocol["physical_fixture"], **selected.get("physical_fixture_override", {})}
    h = float(fixture["substrate_thickness_mm"])
    copper = float(fixture["copper_thickness_mm"])
    board_l = float(fixture["board_length_mm"])
    board_w = float(fixture["board_width_mm"])
    trace_w = float(fixture["trace_width_mm"])
    y_values = [float(value) for value in fixture["channel_y_mm_by_antenna_port"]]
    pre_x = float(fixture["pre_reference_x_mm"])
    post_x = float(fixture["post_reference_x_mm"])
    gap_center = float(fixture["series_gap_center_x_mm"])
    gap_length = float(fixture["series_gap_length_mm"])
    cap_x = float(fixture["ground_cap_x_mm"])
    bridge_x = float(fixture["bridge_x_mm"])
    bridge_width = float(fixture["bridge_sheet_width_mm"])
    mesh = float(fixture["mesh_max_length_mm"])
    refine = float(fixture["adaptive_refinement_percent"])
    max_passes = int(fixture["maximum_passes"])
    min_converged = int(fixture["minimum_converged_passes"])
    left_end = gap_center - gap_length / 2.0
    right_start = gap_center + gap_length / 2.0

    omega = 2.0 * math.pi * 10.0e9
    component_override = selected.get("physical_component_values", {})
    if component_override:
        series_l_nh = float(component_override["series_inductor_nh"])
        ground_c_pf = float(component_override["ground_capacitor_pf"])
        bridge_l_nh = float(component_override["bridge_inductor_nh"])
        series_x = omega * series_l_nh * 1.0e-9
        ground_b = omega * ground_c_pf * 1.0e-12
        bridge_b = -1.0 / (omega * bridge_l_nh * 1.0e-9)
    else:
        series_x = float(selected["series_reactance_ohm_at_10ghz"])
        ground_b = float(selected["ground_susceptance_siemens_at_10ghz"])
        bridge_b = float(selected["bridge_susceptance_siemens_at_10ghz"])
        series_l_nh = series_x / omega * 1.0e9
        ground_c_pf = ground_b / omega * 1.0e12
        bridge_l_nh = -1.0 / (omega * bridge_b) * 1.0e9
    series_r = abs(series_x) / float(protocol["network"]["series_q"])
    ground_parallel_r = float(protocol["network"]["ground_capacitor_q"]) / abs(ground_b)
    bridge_parallel_r = float(protocol["network"]["bridge_inductor_q"]) / abs(bridge_b)
    if min(series_l_nh, ground_c_pf, bridge_l_nh) <= 0.0:
        raise ValueError("The physical fixture requires the selected grounded-lowpass component signs")

    geometry: list[str] = []
    assignments: list[str] = []
    mesh_names: list[str] = []
    for port, y_value in enumerate(y_values):
        pre_name = f"TracePre_{port}"
        post_name = f"TracePost_{port}"
        series_name = f"SeriesSheet_{port}"
        cap_name = f"GroundCapSheet_{port}"
        pre_port = f"PrePortSheet_{port}"
        post_port = f"PostPortSheet_{port}"
        geometry.extend(
            [
                f'CreateBox oEditor, "{pre_name}", {pre_x:.7f}, {y_value-trace_w/2:.7f}, 0, {left_end-pre_x:.7f}, {trace_w:.7f}, {copper:.7f}, "copper", False',
                f'CreateBox oEditor, "{post_name}", {right_start:.7f}, {y_value-trace_w/2:.7f}, 0, {post_x-right_start:.7f}, {trace_w:.7f}, {copper:.7f}, "copper", False',
                f'CreateSheet oEditor, "{series_name}", "Z", {left_end:.7f}, {y_value-trace_w/2:.7f}, 0, {gap_length:.7f}, {trace_w:.7f}',
                f'CreateSheet oEditor, "{cap_name}", "X", {cap_x:.7f}, {y_value-trace_w/4:.7f}, {-h:.7f}, {trace_w/2:.7f}, {h:.7f}',
                f'CreateSheet oEditor, "{pre_port}", "X", {pre_x:.7f}, {y_value-trace_w/2:.7f}, {-h:.7f}, {trace_w:.7f}, {h:.7f}',
                f'CreateSheet oEditor, "{post_port}", "X", {post_x:.7f}, {y_value-trace_w/2:.7f}, {-h:.7f}, {trace_w:.7f}, {h:.7f}',
            ]
        )
        assignments.extend(
            [
                f'AssignRLC oBoundary, "SeriesL_{port}", "{series_name}", "Serial", True, {series_r:.10f}, True, {series_l_nh:.10f}, False, 1, {left_end:.7f}, {y_value:.7f}, 0, {right_start:.7f}, {y_value:.7f}, 0',
                f'AssignRLC oBoundary, "GroundC_{port}", "{cap_name}", "Parallel", True, {ground_parallel_r:.10f}, False, 1, True, {ground_c_pf:.10f}, {cap_x:.7f}, {y_value:.7f}, 0, {cap_x:.7f}, {y_value:.7f}, {-h:.7f}',
            ]
        )
        mesh_names.extend((pre_name, post_name, series_name, cap_name))

    pair_rows = [(0, 2), (1, 3)]
    for pair_index, (first, second) in enumerate(pair_rows):
        lower_y, upper_y = sorted((y_values[first], y_values[second]))
        gap_y0 = lower_y + trace_w / 2.0
        gap_height = upper_y - lower_y - trace_w
        bridge_name = f"BridgeSheet_{pair_index}"
        geometry.append(
            f'CreateSheet oEditor, "{bridge_name}", "Z", {bridge_x-bridge_width/2:.7f}, {gap_y0:.7f}, 0, {bridge_width:.7f}, {gap_height:.7f}'
        )
        assignments.append(
            f'AssignRLC oBoundary, "BridgeL_{pair_index}", "{bridge_name}", "Parallel", True, {bridge_parallel_r:.10f}, True, {bridge_l_nh:.10f}, False, 1, {bridge_x:.7f}, {gap_y0:.7f}, 0, {bridge_x:.7f}, {gap_y0+gap_height:.7f}, 0'
        )
        mesh_names.append(bridge_name)

    # Assign all PRE ports before POST ports so the exported S8 partition is deterministic.
    for port, y_value in enumerate(y_values):
        assignments.append(
            f'AssignPort oBoundary, "PRE_{port}", "PrePortSheet_{port}", {pre_x:.7f}, {y_value:.7f}, 0, {pre_x:.7f}, {y_value:.7f}, {-h:.7f}'
        )
    for port, y_value in enumerate(y_values):
        assignments.append(
            f'AssignPort oBoundary, "POST_{port}", "PostPortSheet_{port}", {post_x:.7f}, {y_value:.7f}, 0, {post_x:.7f}, {y_value:.7f}, {-h:.7f}'
        )
    mesh_objects = ", ".join(f'"{name}"' for name in mesh_names)
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V115", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{fixture['relative_permittivity']}", "dielectric_loss_tangent:=", "{fixture['loss_tangent']}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h:.7f}, {board_l:.7f}, {board_w:.7f}, {h:.7f}, "RO5880_V115", True
CreateBox oEditor, "Ground", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h-copper:.7f}, {board_l:.7f}, {board_w:.7f}, {copper:.7f}, "copper", False
{chr(10).join(geometry)}
{chr(10).join(assignments)}
CreateBox oEditor, "AirRegion", {-board_l/2-5:.7f}, {-board_w/2-5:.7f}, -5, {board_l+10:.7f}, {board_w+10:.7f}, 10, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:V115_ComponentAndTraceMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({mesh_objects}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {max_passes}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", {min_converged}, "PercentRefinement:=", {refine:.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "PortAccuracy:=", 2)
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "9.96GHz", "RangeEnd:=", "10.04GHz", "RangeCount:=", 3, "Type:=", "Discrete", "SaveFields:=", False, "SaveRadFields:=", False, "InterpTolerance:=", 0.5, "InterpMaxSolns:=", 250, "InterpMinSolns:=", 0, "InterpMinSubranges:=", 1, "InterpUseS:=", True, "InterpUsePortImped:=", True, "InterpUsePropConst:=", True, "UseDerivativeConvergence:=", False, "InterpDerivTolerance:=", 0.2, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{helpers_vbs()}
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
vars = oSolutions.ListVariations("Setup_10GHz:Sweep_Gate3")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Gate3"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def case_dir(out_dir: Path, replicate: int) -> Path:
    return out_dir / f"physical_s8_direct{replicate:02d}"


def prepare_case(out_dir: Path, protocol: dict[str, Any], selected: dict[str, Any], replicate: int) -> dict[str, Any]:
    folder = case_dir(out_dir, replicate)
    if folder.exists():
        return load_json(folder / "case_manifest.json")
    folder.mkdir(parents=True)
    project = folder / f"v115_physical_modal_feed_s8_direct{replicate:02d}.aedt"
    touchstone = folder / f"v115_physical_modal_feed_s8_direct{replicate:02d}.s8p"
    build_script = folder / "build.vbs"
    solve_script = folder / "solve_export.vbs"
    build_script.write_text(builder_text(project, protocol, selected), encoding="ascii")
    solve_script.write_text(solver_text(project, touchstone), encoding="ascii")
    manifest = {
        "replicate": replicate,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(build_script.resolve()),
        "solver_path": str(solve_script.resolve()),
        "pre_reference_ports": [f"PRE_{index}" for index in range(4)],
        "post_reference_ports": [f"POST_{index}" for index in range(4)],
        "selected_variant": selected["variant"],
    }
    write_json(folder / "case_manifest.json", manifest)
    return manifest


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite physical fixture: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_json(args.protocol)
    decision = load_json(args.design_dir / "stage_decision.json")
    selected = (
        load_json(args.physical_design)
        if args.physical_design
        else load_json(args.design_dir / "selected_network.json")
    )
    if not decision.get("allow_physical_2x2"):
        raise RuntimeError("The v1.15 circuit gate does not authorize a physical fixture")
    shutil.copy2(args.protocol, args.out_dir / "protocol_snapshot.json")
    if args.physical_design:
        shutil.copy2(args.physical_design, args.out_dir / "physical_design_snapshot.json")
        write_json(args.out_dir / "selected_network_snapshot.json", selected)
    else:
        shutil.copy2(args.design_dir / "selected_network.json", args.out_dir / "selected_network_snapshot.json")
    shutil.copy2(
        args.design_dir / "selected_dual_reference_plane_network.npz",
        args.out_dir / "selected_dual_reference_plane_network_snapshot.npz",
    )
    manifest = prepare_case(args.out_dir, protocol, selected, 1)
    write_json(
        args.out_dir / "stage_decision_rev01.json",
        {
            "decision": "run_one_physical_s8_direct_fixture",
            "allow_replicate": False,
            "allow_4x4": False,
            "allow_16x16": False,
            "allow_hfss_training_labels": False,
            "allow_critic_training": False,
        },
    )
    return {"prepared": True, "case": manifest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    folders = sorted(path.parent for path in args.out_dir.glob("physical_s8_direct*/case_manifest.json"))
    if not folders:
        raise RuntimeError("No prepared physical S8 fixture")
    minimum_memory = float(load_json(args.out_dir / "protocol_snapshot.json")["physical_fixture"]["minimum_free_memory_gb"])
    results = []
    for folder in folders:
        manifest = load_json(folder / "case_manifest.json")
        touchstone = Path(manifest["touchstone_path"])
        if touchstone.exists() and touchstone.stat().st_size > 100:
            results.append({"replicate": manifest["replicate"], "status": "already_complete"})
            continue
        free_memory = memory_available_gb()
        if math.isfinite(free_memory) and free_memory < minimum_memory:
            raise MemoryError(f"Only {free_memory:.2f} GiB free; {minimum_memory:.2f} GiB required")
        with (folder / "build.log").open("w", encoding="utf-8") as handle:
            build = subprocess.run(
                [str(args.ansys_exe), "-RunScriptAndExit", manifest["builder_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        solve_code = None
        if build.returncode == 0:
            with (folder / "solve_export.log").open("w", encoding="utf-8") as handle:
                solve = subprocess.run(
                    [str(args.ansys_exe), "-ng", "-RunScriptAndExit", manifest["solver_path"]],
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            solve_code = int(solve.returncode)
        summary = {
            "replicate": int(manifest["replicate"]),
            "build_return_code": int(build.returncode),
            "solve_return_code": solve_code,
            "free_memory_gb_before": free_memory,
        }
        write_json(folder / "run_summary.json", summary)
        results.append(summary)
    return {"run_results": results}


def evaluate_physical_s8(
    network_s8: np.ndarray,
    protocol: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    z0 = float(protocol["reference_impedance_ohm"])
    antenna_frequencies, antenna_s4 = parse_touchstone(ROOT / protocol["trusted_antenna_s4"], 4)
    rows, vectors, considered = load_stimuli(ROOT / protocol["trusted_stimulus_root"])
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    source_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    for frequency_index, frequency in enumerate(antenna_frequencies):
        external_s, incident_map, reflected_map = terminate_network(
            network_s8[frequency_index], antenna_s4[frequency_index]
        )
        indices = [
            index for index, row in enumerate(rows) if abs(float(row["frequency_ghz"]) - frequency) <= 1.0e-9
        ]
        sources = vectors[indices].T
        active = considered[indices].T
        reflected = external_s @ sources
        gamma = np.where(active, np.abs(reflected) / np.maximum(np.abs(sources), EPS), 0.0)
        active_rl = -20.0 * np.log10(np.maximum(np.max(gamma, axis=0), EPS))
        incident_power = np.sum(np.abs(sources) ** 2, axis=0)
        reflected_power = np.sum(np.abs(reflected) ** 2, axis=0)
        total_rl = -10.0 * np.log10(np.maximum(reflected_power / incident_power, EPS))
        antenna_incident = incident_map @ sources
        antenna_reflected = reflected_map @ sources
        external_accepted = incident_power - reflected_power
        antenna_accepted = np.sum(np.abs(antenna_incident) ** 2, axis=0) - np.sum(
            np.abs(antenna_reflected) ** 2, axis=0
        )
        insertion = antenna_accepted / np.maximum(external_accepted, EPS)
        transducer = antenna_accepted / np.maximum(incident_power, EPS)
        matched_s, matched_incident, matched_reflected = terminate_network(
            network_s8[frequency_index], np.zeros((4, 4), dtype=complex)
        )
        matched_external_accepted = 1.0 - np.sum(np.abs(matched_s) ** 2, axis=0)
        matched_delivered = np.sum(np.abs(matched_incident) ** 2, axis=0) - np.sum(
            np.abs(matched_reflected) ** 2, axis=0
        )
        matched_efficiency = matched_delivered / np.maximum(matched_external_accepted, EPS)
        passive_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(external_s)), EPS))
        frequency_rows.append(
            {
                "frequency_ghz": float(frequency),
                "passive_rl_min_db": float(np.min(passive_rl)),
                "active_rl_min_db": float(np.min(active_rl)),
                "total_rl_min_db": float(np.min(total_rl)),
                "actual_load_insertion_efficiency_min": float(np.min(insertion)),
                "actual_load_transducer_efficiency_min": float(np.min(transducer)),
                "matched_load_network_efficiency_min": float(np.min(matched_efficiency)),
            }
        )
        for local, global_index in enumerate(indices):
            source_rows.append(
                {
                    **rows[global_index],
                    "active_rl_db": float(active_rl[local]),
                    "total_rl_db": float(total_rl[local]),
                    "external_incident_power": float(incident_power[local]),
                    "external_accepted_power": float(external_accepted[local]),
                    "antenna_accepted_power": float(antenna_accepted[local]),
                    "network_dissipated_power": float(external_accepted[local] - antenna_accepted[local]),
                    "actual_load_insertion_efficiency": float(insertion[local]),
                    "actual_load_transducer_efficiency": float(transducer[local]),
                }
            )
    summary = {
        "passive_rl_min_db": min(row["passive_rl_min_db"] for row in frequency_rows),
        "active_rl_min_db": min(row["active_rl_min_db"] for row in frequency_rows),
        "total_rl_min_db": min(row["total_rl_min_db"] for row in frequency_rows),
        "actual_load_insertion_efficiency_min": min(row["actual_load_insertion_efficiency_min"] for row in frequency_rows),
        "actual_load_transducer_efficiency_min": min(row["actual_load_transducer_efficiency_min"] for row in frequency_rows),
        "matched_load_network_efficiency_min": min(row["matched_load_network_efficiency_min"] for row in frequency_rows),
        "frequency_rows": frequency_rows,
    }
    return summary, source_rows


def analyze_case(folder: Path, protocol: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    manifest = load_json(folder / "case_manifest.json")
    run_summary = load_json(folder / "run_summary.json") if (folder / "run_summary.json").exists() else {}
    touchstone = Path(manifest["touchstone_path"])
    result = {"replicate": int(manifest["replicate"]), **run_summary, **profile_metrics(folder)}
    if not touchstone.exists() or touchstone.stat().st_size < 100:
        result["physical_gate_pass"] = False
        write_json(folder / "analysis.json", result)
        return result
    frequencies, physical_s8 = parse_touchstone(touchstone, 8)
    exported_names = touchstone_port_names(touchstone)
    desired_names = manifest["pre_reference_ports"] + manifest["post_reference_ports"]
    if len(exported_names) != 8 or set(exported_names) != set(desired_names):
        raise RuntimeError(f"Unexpected physical S8 port names: {exported_names}")
    permutation = [exported_names.index(name) for name in desired_names]
    physical_s8 = physical_s8[:, permutation][:, :, permutation]
    if not np.allclose(frequencies, protocol["frequencies_ghz"], atol=1.0e-9):
        raise RuntimeError("Physical S8 frequency grid mismatch")
    reciprocity = float(np.max(np.abs(physical_s8 - np.transpose(physical_s8, (0, 2, 1)))))
    passivity = float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in physical_s8))
    circuit = np.load(out_dir / "selected_dual_reference_plane_network_snapshot.npz", allow_pickle=False)
    target_s8 = np.asarray(circuit["network_s8"], dtype=complex)
    physical_vs_target = float(np.max(np.abs(physical_s8 - target_s8)))
    metrics, source_rows = evaluate_physical_s8(physical_s8, protocol, out_dir)
    write_csv(folder / "physical_frequency_metrics.csv", metrics.pop("frequency_rows"))
    write_csv(folder / "physical_stimulus_metrics.csv", source_rows)
    gates = protocol["gates"]
    result.update(
        {
            **metrics,
            "reciprocity_error_max": reciprocity,
            "passivity_sigma_max": passivity,
            "physical_vs_lumped_target_s8_max_abs": physical_vs_target,
            "touchstone_exported_port_order": exported_names,
            "analysis_port_order": desired_names,
        }
    )
    result["physical_gate_pass"] = bool(
        result.get("converged") is True
        and float(result.get("final_delta_s") or math.inf) <= float(gates["maximum_physical_final_delta_s"])
        and reciprocity <= float(gates["maximum_physical_reciprocity_error"])
        and passivity <= float(gates["maximum_physical_passivity_sigma"])
        and result["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
        and result["active_rl_min_db"] >= float(gates["minimum_representative_active_rl_db"])
        and result["total_rl_min_db"] >= float(gates["minimum_representative_total_rl_db"])
        and result["actual_load_insertion_efficiency_min"] >= float(gates["minimum_actual_load_insertion_efficiency"])
        and result["matched_load_network_efficiency_min"] >= float(gates["minimum_matched_load_network_efficiency"])
    )
    write_json(folder / "analysis.json", result)
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.out_dir / "protocol_snapshot.json")
    folders = sorted(path.parent for path in args.out_dir.glob("physical_s8_direct*/case_manifest.json"))
    results = [analyze_case(folder, protocol, args.out_dir) for folder in folders]
    write_csv(args.out_dir / "physical_case_metrics.csv", results)
    first = next((row for row in results if int(row["replicate"]) == 1), None)
    second = next((row for row in results if int(row["replicate"]) == 2), None)
    if first is None or not first.get("physical_gate_pass"):
        decision = {
            "decision": "stop_physical_s8_gate_failed",
            "allow_independent_repeat": False,
            "allow_4x4": False,
            "allow_16x16": False,
            "allow_hfss_training_labels": False,
            "allow_critic_training": False,
        }
    elif second is None:
        selected = load_json(args.out_dir / "selected_network_snapshot.json")
        prepare_case(args.out_dir, protocol, selected, 2)
        decision = {
            "decision": "run_independent_physical_s8_repeat",
            "allow_independent_repeat": True,
            "allow_4x4": False,
            "allow_16x16": False,
        }
    else:
        first_manifest = load_json(case_dir(args.out_dir, 1) / "case_manifest.json")
        second_manifest = load_json(case_dir(args.out_dir, 2) / "case_manifest.json")
        first_f, first_s = parse_touchstone(Path(first_manifest["touchstone_path"]), 8)
        second_f, second_s = parse_touchstone(Path(second_manifest["touchstone_path"]), 8)
        repeat_delta = float(np.max(np.abs(first_s - second_s)))
        repeat_pass = bool(
            np.allclose(first_f, second_f, atol=1.0e-12)
            and repeat_delta <= float(protocol["gates"]["maximum_independent_repeat_delta_s"])
        )
        decision = {
            "decision": "physical_2x2_feed_gate_pass" if second.get("physical_gate_pass") and repeat_pass else "stop_physical_repeat_gate_failed",
            "independent_repeat_max_abs_delta_s": repeat_delta,
            "independent_repeat_gate_pass": repeat_pass,
            "allow_4x4": False,
            "allow_16x16": False,
            "allow_hfss_training_labels": False,
            "allow_critic_training": False,
        }
    revisions = len(list(args.out_dir.glob("stage_decision_rev*.json"))) + 1
    write_json(args.out_dir / f"stage_decision_rev{revisions:02d}.json", decision)
    return {"results": results, "stage_decision": decision}


def status(args: argparse.Namespace) -> dict[str, Any]:
    cases = []
    if args.out_dir.exists():
        for manifest_path in sorted(args.out_dir.glob("physical_s8_direct*/case_manifest.json")):
            manifest = load_json(manifest_path)
            folder = manifest_path.parent
            cases.append(
                {
                    "replicate": manifest["replicate"],
                    "touchstone_exists": Path(manifest["touchstone_path"]).exists(),
                    "analyzed": (folder / "analysis.json").exists(),
                }
            )
    decisions = sorted(args.out_dir.glob("stage_decision_rev*.json")) if args.out_dir.exists() else []
    return {
        "prepared": (args.out_dir / "protocol_snapshot.json").exists(),
        "cases": cases,
        "latest_decision": load_json(decisions[-1]) if decisions else None,
        "free_memory_gb": memory_available_gb(),
    }


def main() -> None:
    args = parse_args()
    result = {"prepare": prepare, "run": run, "analyze": analyze, "status": status}[args.mode](args)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
