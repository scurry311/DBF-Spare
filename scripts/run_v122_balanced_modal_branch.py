#!/usr/bin/env python3
"""Build and gate the v1.22 balanced launch and one-section modal branch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from design_v115_grounded_modal_network import terminate_network
from design_v119_multiport_post_decoupler import active_metrics, reordered_network
from run_v114_small_cell_broadband_feed import load_stimuli, memory_available_gb, parse_touchstone, profile_metrics
from run_v115_physical_modal_feed_fixture import touchstone_port_names
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v122_balanced_modal_branch_preregistered.json"
DESIGN_NAME = "V122_Balanced_Modal_Branch"
EPS = 1.0e-15


def resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    parent = payload.get("extends")
    if not parent:
        return payload
    return deep_merge(load_config(resolve(str(parent))), payload)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def vbs_helpers() -> str:
    return r'''Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(210 150 55)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheetX(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "X"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(210 150 55)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub AssignDifferentialPort(boundary, portName, sheetName, x, yNegative, yPositive, z)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x), Mm(yNegative), Mm(z)), "End:=", Array(Mm(x), Mm(yPositive), Mm(z))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Sub UnitePair(editor, firstName, secondName)
    editor.Unite Array("NAME:Selections", "Selections:=", firstName & "," & secondName), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function'''


def channel_centers(config: dict[str, Any], side: int) -> list[tuple[int, float]]:
    if side == 1:
        return [(0, 0.0)]
    physical = config["physical_model"]
    values = [float(value) for value in physical["channel_y_mm_by_antenna_port"]]
    return [(int(port), values[int(port)]) for port in physical["physical_channel_order_negative_to_positive_y"]]


def builder_text(project: Path, config: dict[str, Any], side: int) -> str:
    physical = config["physical_model"]
    mesh = config["mesh"]
    h = float(physical["substrate_thickness_mm"])
    copper = float(physical["copper_thickness_mm"])
    board_l = float(physical["board_length_mm"])
    board_w = float(physical["board_width_1x1_mm"] if side == 1 else physical["board_width_2x2_mm"])
    pre_x = float(physical["pre_reference_x_mm"])
    post_x = float(physical["post_reference_x_mm"])
    pad_l = float(physical["pre_pad_length_mm"])
    overlap = float(physical["pad_trace_overlap_mm"])
    pitch = float(physical["pair_center_pitch_mm"])
    pad_w = float(physical["pre_pad_width_mm"])
    trace_w = float(physical["main_trace_width_mm"])
    finite_sheet = physical.get("conductor_representation") == "finite_conductivity_sheet"
    port_sheet_z = (0.0 if finite_sheet else copper) if "port_sheet_height_mm" in physical else 0.0
    port_sheet_height = float(physical.get("port_sheet_height_mm", copper))
    main_x = pre_x + pad_l - overlap
    geometry: list[str] = []
    assignments: list[str] = []
    trace_names: list[str] = []
    branch_names: list[str] = []
    centers = channel_centers(config, side)
    for port, center_y in centers:
        negative_y = center_y - pitch / 2.0
        positive_y = center_y + pitch / 2.0
        for polarity, y_value in (("N", negative_y), ("P", positive_y)):
            trace = f"Trace{polarity}_{port}"
            pad = f"Pad{polarity}_{port}"
            if finite_sheet and abs(pad_w - trace_w) > EPS:
                raise ValueError("The finite-conductivity sheet model requires one constant-width conductor per polarity")
            if finite_sheet:
                geometry.append(
                    f'CreateSheetZ oEditor, "{trace}", {pre_x:.7f}, {y_value-trace_w/2:.7f}, 0, {post_x-pre_x:.7f}, {trace_w:.7f}'
                )
            elif abs(pad_w - trace_w) <= EPS:
                geometry.append(
                    f'CreateBox oEditor, "{trace}", {pre_x:.7f}, {y_value-trace_w/2:.7f}, 0, {post_x-pre_x:.7f}, {trace_w:.7f}, {copper:.7f}, "copper", False'
                )
            else:
                geometry.extend(
                    [
                        f'CreateBox oEditor, "{trace}", {main_x:.7f}, {y_value-trace_w/2:.7f}, 0, {post_x-main_x:.7f}, {trace_w:.7f}, {copper:.7f}, "copper", False',
                        f'CreateBox oEditor, "{pad}", {pre_x:.7f}, {y_value-pad_w/2:.7f}, 0, {pad_l:.7f}, {pad_w:.7f}, {copper:.7f}, "copper", False',
                        f'UnitePair oEditor, "{trace}", "{pad}"',
                    ]
                )
            trace_names.append(trace)
        geometry.extend(
            [
                f'CreateSheetX oEditor, "PrePortSheet_{port}", {pre_x:.7f}, {negative_y:.7f}, {port_sheet_z:.7f}, {pitch:.7f}, {port_sheet_height:.7f}',
                f'CreateSheetX oEditor, "PostPortSheet_{port}", {post_x:.7f}, {negative_y:.7f}, {port_sheet_z:.7f}, {pitch:.7f}, {port_sheet_height:.7f}',
            ]
        )
        assignments.append(
            f'AssignDifferentialPort oBoundary, "PRE_{port}", "PrePortSheet_{port}", {pre_x:.7f}, {negative_y:.7f}, {positive_y:.7f}, {port_sheet_z:.7f}'
        )
    for port, center_y in centers:
        negative_y = center_y - pitch / 2.0
        positive_y = center_y + pitch / 2.0
        assignments.append(
            f'AssignDifferentialPort oBoundary, "POST_{port}", "PostPortSheet_{port}", {post_x:.7f}, {negative_y:.7f}, {positive_y:.7f}, {port_sheet_z:.7f}'
        )
    if side == 2:
        resonator_x = float(physical["modal_resonator_center_x_mm"])
        resonator_l = float(physical["modal_resonator_length_mm"])
        resonator_w = float(physical["modal_resonator_width_mm"])
        y_by_port = {port: center for port, center in centers}
        for index, pair in enumerate(physical["local_x_neighbor_pairs"]):
            first, second = (int(pair[0]), int(pair[1]))
            center_y = 0.5 * (y_by_port[first] + y_by_port[second])
            name = f"ModalFloatingBranch_{index}"
            geometry.append(
                (
                    f'CreateSheetZ oEditor, "{name}", {resonator_x-resonator_l/2:.7f}, {center_y-resonator_w/2:.7f}, 0, {resonator_l:.7f}, {resonator_w:.7f}'
                    if finite_sheet
                    else f'CreateBox oEditor, "{name}", {resonator_x-resonator_l/2:.7f}, {center_y-resonator_w/2:.7f}, 0, {resonator_l:.7f}, {resonator_w:.7f}, {copper:.7f}, "copper", False'
                )
            )
            branch_names.append(name)
    trace_mesh = ", ".join(f'"{name}"' for name in trace_names)
    branch_mesh = ", ".join(f'"{name}"' for name in branch_names)
    solver = (
        ', "DrivenSolverType:=", "Iterative Solver", "IterativeResidual:=", '
        f'{float(mesh["iterative_residual"]):.10g}'
        if str(mesh["solver_type"]).lower() == "iterative"
        else ""
    )
    branch_mesh_statement = ""
    if branch_names:
        branch_mesh_statement = f'oMesh.AssignLengthOp Array("NAME:ModalBranchMesh_0p080mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({branch_mesh}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(mesh["modal_branch_max_length_mm"]):.7f}mm", "UseAdvSizing:=", False)'
    ground_statement = (
        f'CreateSheetZ oEditor, "ReferenceGround", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h:.7f}, {board_l:.7f}, {board_w:.7f}'
        if finite_sheet
        else f'CreateBox oEditor, "ReferenceGround", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h-copper:.7f}, {board_l:.7f}, {board_w:.7f}, {copper:.7f}, "copper", False'
    )
    finite_conductivity_statement = ""
    if finite_sheet:
        finite_objects = ", ".join(f'"{name}"' for name in ["ReferenceGround", *trace_names, *branch_names])
        finite_conductivity_statement = f'oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", "Objects:=", Array({finite_objects}), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)'
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:{physical['substrate_material']}", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{physical['relative_permittivity']}", "dielectric_loss_tangent:=", "{physical['loss_tangent']}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h:.7f}, {board_l:.7f}, {board_w:.7f}, {h:.7f}, "{physical['substrate_material']}", True
{ground_statement}
{chr(10).join(geometry)}
{finite_conductivity_statement}
{chr(10).join(assignments)}
CreateBox oEditor, "AirRegion", {-board_l/2-4:.7f}, {-board_w/2-4:.7f}, -3, {board_l+8:.7f}, {board_w+8:.7f}, 6.5, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:DifferentialTraceMesh_0p120mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({trace_mesh}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(mesh['differential_trace_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
{branch_mesh_statement}
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", {float(config['gates']['maximum_final_delta_s']):.7f}, "MaximumPasses:=", {int(mesh['maximum_passes'])}, "MinimumPasses:=", {int(mesh['minimum_passes'])}, "MinimumConvergedPasses:=", {int(mesh['minimum_converged_passes'])}, "PercentRefinement:=", {float(mesh['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "PortAccuracy:=", 2{solver})
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "9.96GHz", "RangeEnd:=", "10.04GHz", "RangeCount:=", 3, "Type:=", "Discrete", "SaveFields:=", False, "SaveRadFields:=", False, "InterpTolerance:=", 0.5, "InterpMaxSolns:=", 250, "InterpMinSolns:=", 0, "InterpMinSubranges:=", 1, "InterpUseS:=", True, "InterpUsePortImped:=", True, "InterpUsePropConst:=", True, "UseDerivativeConvergence:=", False, "InterpDerivTolerance:=", 0.2, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{vbs_helpers()}
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


def single_frequency_solver_text(project: Path, touchstone: Path) -> str:
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


def frequency_token(frequency_ghz: float) -> str:
    return f"{frequency_ghz:.2f}".replace(".", "p")


def prepare_serial_modal_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    out = resolve(config["output_directory"])
    root = out / "modal_2x2_s8_serial"
    root.mkdir(parents=True, exist_ok=True)
    existing = root / "serial_case_manifest.json"
    if existing.exists():
        return read_json(existing)["cases"]
    expected_order = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    cases = []
    for frequency in (float(value) for value in config["frequencies_ghz"]):
        token = frequency_token(frequency)
        folder = root / f"f_{token}ghz"
        folder.mkdir(parents=True, exist_ok=False)
        project = folder / f"v122_balanced_2x2_f_{token}ghz.aedt"
        touchstone = folder / f"v122_balanced_2x2_f_{token}ghz.s8p"
        builder = folder / "build.vbs"
        solver = folder / "solve_export.vbs"
        build_text = builder_text(project, config, 2)
        build_text = build_text.replace('"Frequency:=", "10GHz"', f'"Frequency:=", "{frequency:g}GHz"')
        build_text = re.sub(r'^oAnalysis\.InsertFrequencySweep .*$', '', build_text, flags=re.MULTILINE)
        builder.write_text(build_text, encoding="ascii")
        solver.write_text(single_frequency_solver_text(project, touchstone), encoding="ascii")
        cases.append(
            {
                "frequency_ghz": frequency,
                "project_path": str(project.resolve()),
                "touchstone_path": str(touchstone.resolve()),
                "builder_path": str(builder.resolve()),
                "solver_path": str(solver.resolve()),
                "expected_port_order": expected_order,
                "solution": "Setup_10GHz:LastAdaptive",
            }
        )
    payload = {
        "execution_mode": "exact_single_frequency_adaptive_serial",
        "reason": "The original discrete three-frequency sweep exceeded the memory stop line by launching concurrent hf3d workers.",
        "physical_geometry_changed": False,
        "engineering_thresholds_changed": False,
        "cases": cases,
    }
    write_json(existing, payload)
    return cases


def run_modal_serial(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_2x2_solve"):
        raise RuntimeError("The serial 2x2 solve is not authorized")
    require_no_aedt()
    executable = str(resolve(config["ansys_executable"]))
    minimum = float(config["resources"]["minimum_free_memory_before_solve_gib"])
    abort = float(config["resources"]["abort_free_memory_during_solve_gib"])
    poll = float(config["resources"]["poll_interval_seconds"])
    rows = []
    for case in prepare_serial_modal_cases(config):
        require_no_aedt()
        folder = Path(case["builder_path"]).parent
        project = Path(case["project_path"])
        touchstone = Path(case["touchstone_path"])
        if not project.exists():
            code, memory_aborted, minimum_free = run_process_with_memory_guard(
                [executable, "-RunScriptAndExit", case["builder_path"]],
                folder / "build.log",
                None,
                poll,
            )
            if code != 0 or memory_aborted or not project.exists():
                rows.append({"frequency_ghz": case["frequency_ghz"], "stage": "build", "return_code": code, "memory_aborted": memory_aborted, "minimum_free_memory_gib": minimum_free})
                break
        if touchstone.exists() and touchstone.stat().st_size > 100:
            rows.append({"frequency_ghz": case["frequency_ghz"], "stage": "solve", "status": "already_complete", "touchstone_exists": True})
            continue
        free = memory_available_gb()
        if free < minimum:
            raise RuntimeError(f"Only {free:.2f} GiB free; {minimum:.2f} GiB required before {case['frequency_ghz']} GHz")
        code, memory_aborted, minimum_free = run_process_with_memory_guard(
            [executable, "-ng", "-RunScriptAndExit", case["solver_path"]],
            folder / "solve_export.log",
            abort,
            poll,
        )
        row = {
            "frequency_ghz": case["frequency_ghz"],
            "stage": "solve",
            "return_code": code,
            "memory_aborted": memory_aborted,
            "minimum_free_memory_gib": minimum_free,
            "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
        }
        write_json(folder / "run_summary.json", row)
        rows.append(row)
        if code != 0 or memory_aborted or not row["touchstone_exists"]:
            break
    write_csv(out / "modal_2x2_s8_serial" / "execution.csv", rows)
    return {"execution_mode": "exact_single_frequency_adaptive_serial", "cases": rows}


def prepare_case(config: dict[str, Any], side: int, folder: Path) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=False)
    n = side * side
    project = folder / f"v122_balanced_{side}x{side}.aedt"
    touchstone = folder / f"v122_balanced_{side}x{side}.s{2*n}p"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    builder.write_text(builder_text(project, config, side), encoding="ascii")
    solver.write_text(solver_text(project, touchstone), encoding="ascii")
    manifest = {
        "side": side,
        "port_count": 2 * n,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "expected_port_order": [f"PRE_{index}" for index in range(n)] + [f"POST_{index}" for index in range(n)],
        "evidence_scope": "physical differential network-only HFSS; not an integrated antenna result",
    }
    write_json(folder / "case_manifest.json", manifest)
    return manifest


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.22 output: {out}")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Parent mismatch: HEAD={head}, tag={tag}")
    preregistration = {
        **config,
        "runtime_audit": {
            "head_commit": head,
            "parent_tag_commit": tag,
            "aedt_processes": aedt_processes(),
            "free_memory_gib": memory_available_gb(),
        },
        "evidence_rules": {
            "matched_or_trusted_s4_termination_is_not_integrated_hfss": True,
            "no_2x2_solve_before_1x1_gate": True,
            "no_second_decoupling_stage": True,
            "no_training_labels_or_critic": True,
        },
    }
    write_json(out / "preregistration.json", preregistration)
    cases = {
        "launch_1x1": prepare_case(config, 1, out / "launch_1x1_s2"),
        "modal_2x2": prepare_case(config, 2, out / "modal_2x2_s8"),
    }
    write_json(out / "case_manifest.json", cases)
    decision = {
        "stage": "A_preregistered",
        "allow_build_smoke": True,
        "allow_1x1_solve": False,
        "allow_2x2_solve": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "cases": cases, "decision": decision}


def apply_mesh_selection_amendment(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    amendment = out / "preregistration_amendment01_mesh_selection.json"
    if amendment.exists():
        raise FileExistsError(f"Refusing to overwrite amendment: {amendment}")
    cases = read_json(out / "case_manifest.json")
    if (out / "build_smoke_execution.csv").exists() or any(Path(item["project_path"]).exists() for item in cases.values()):
        raise RuntimeError("Mesh selection can only be amended before the first AEDT build")
    for manifest in cases.values():
        Path(manifest["builder_path"]).write_text(builder_text(Path(manifest["project_path"]), config, int(manifest["side"])), encoding="ascii")
    payload = {
        "amendment": "v1.22-preregistration-amendment-01-disjoint-mesh-selections",
        "created_on": "2026-08-04",
        "timing": "after generated-script inspection and before any AEDT build or electromagnetic result",
        "physical_geometry_changed": False,
        "engineering_thresholds_changed": False,
        "mesh_scales_changed": False,
        "change": "Remove floating modal branches from the 0.12 mm trace mesh selection because they already have a dedicated 0.08 mm mesh operation.",
    }
    write_json(amendment, payload)
    return payload


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def run_build_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    require_no_aedt()
    free = memory_available_gb()
    if free < float(config["resources"]["minimum_free_memory_before_build_gib"]):
        raise RuntimeError(f"Only {free:.2f} GiB free before build smoke")
    executable = str(resolve(config["ansys_executable"]))
    cases = read_json(out / "case_manifest.json")
    rows = []
    for name, manifest in cases.items():
        require_no_aedt()
        folder = Path(manifest["builder_path"]).parent
        with (folder / "build.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [executable, "-RunScriptAndExit", manifest["builder_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        rows.append({"case": name, "return_code": int(result.returncode), "project_exists": Path(manifest["project_path"]).exists()})
    write_csv(out / "build_smoke_execution.csv", rows)
    return {"cases": rows}


def audit_build_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "case_manifest.json")
    executions = {row["case"]: row for row in csv.DictReader((out / "build_smoke_execution.csv").open(encoding="utf-8"))}
    rows = []
    for name, manifest in cases.items():
        folder = Path(manifest["builder_path"]).parent
        builder = Path(manifest["builder_path"]).read_text(encoding="ascii")
        log = (folder / "build.log").read_text(encoding="utf-8", errors="ignore")
        port_names = [item for item in manifest["expected_port_order"] if f'"{item}"' in builder]
        warning_count = sum(
            log.lower().count(pattern)
            for pattern in ("script error", "invalid geometry", "small segment", "too many conductors touch lumped port")
        )
        row = {
            "case": name,
            "return_code": int(executions[name]["return_code"]),
            "project_valid": Path(manifest["project_path"]).exists() and Path(manifest["project_path"]).stat().st_size > 100,
            "expected_port_count": len(manifest["expected_port_order"]),
            "builder_port_count": len(port_names),
            "differential_port_definition_count": builder.count("AssignDifferentialPort oBoundary"),
            "modal_branch_count": len(set(re.findall(r'ModalFloatingBranch_\d+', builder))),
            "warning_count": warning_count,
        }
        row["gate_pass"] = bool(
            row["return_code"] == 0
            and row["project_valid"]
            and row["builder_port_count"] == row["expected_port_count"]
            and row["differential_port_definition_count"] == row["expected_port_count"]
            and warning_count == 0
        )
        rows.append(row)
    write_csv(out / "build_smoke_audit.csv", rows)
    build_pass = all(row["gate_pass"] for row in rows)
    decision = {
        "stage": "B_build_smoke_complete",
        "build_smoke_pass": build_pass,
        "allow_1x1_solve": build_pass,
        "allow_2x2_solve": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"rows": rows, "decision": decision}


def run_solve(config: dict[str, Any], case_name: str, authorization: str) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    decision = read_json(out / "stage_decision.json")
    if not decision.get(authorization):
        raise RuntimeError(f"{case_name} is not authorized by {authorization}")
    require_no_aedt()
    free = memory_available_gb()
    minimum = float(config["resources"]["minimum_free_memory_before_solve_gib"])
    if free < minimum:
        raise RuntimeError(f"Only {free:.2f} GiB free; {minimum:.2f} GiB required")
    manifest = read_json(out / "case_manifest.json")[case_name]
    touchstone = Path(manifest["touchstone_path"])
    if touchstone.exists() and touchstone.stat().st_size > 100:
        return {"case": case_name, "status": "already_complete"}
    folder = Path(manifest["solver_path"]).parent
    code, aborted, minimum_free = run_process_with_memory_guard(
        [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", manifest["solver_path"]],
        folder / "solve_export.log",
        float(config["resources"]["abort_free_memory_during_solve_gib"]),
        float(config["resources"]["poll_interval_seconds"]),
    )
    result = {
        "case": case_name,
        "return_code": code,
        "memory_aborted": aborted,
        "minimum_free_memory_gib": minimum_free,
        "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
    }
    write_json(folder / "run_summary.json", result)
    return result


def common_s_metrics(matrices: np.ndarray) -> dict[str, float]:
    reciprocity = max(float(np.max(np.abs(matrix - matrix.T))) for matrix in matrices)
    passivity = max(float(np.max(np.linalg.svd(matrix, compute_uv=False))) for matrix in matrices)
    return {"reciprocity_error": reciprocity, "passivity_sigma": passivity}


def small_segment_diagnostics(folder: Path) -> dict[str, float | None]:
    lengths: list[float] = []
    for path in folder.rglob("*.g3derr"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        lengths.extend(
            float(value)
            for value in re.findall(r"Segment length\(s\)\s*:\s*([0-9.]+)mm", text)
        )
    return {
        "small_mesh_segment_min_length_mm": min(lengths) if lengths else None,
        "small_mesh_segment_max_length_mm": max(lengths) if lengths else None,
    }


def analyze_launch(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest = read_json(out / "case_manifest.json")["launch_1x1"]
    touchstone = Path(manifest["touchstone_path"])
    if not touchstone.exists():
        raise RuntimeError("The 1x1 Touchstone is missing")
    frequencies, matrices = reordered_network(touchstone, manifest["expected_port_order"], 2)
    target = np.asarray(config["frequencies_ghz"], dtype=float)
    indices = [int(np.argmin(np.abs(frequencies - value))) for value in target]
    selected = matrices[indices]
    rows = []
    for frequency, matrix in zip(target, selected):
        accepted = max(1.0 - abs(matrix[0, 0]) ** 2, EPS)
        delivered = abs(matrix[1, 0]) ** 2
        rows.append(
            {
                "frequency_ghz": float(frequency),
                "input_rl_db": float(-20.0 * np.log10(max(abs(matrix[0, 0]), EPS))),
                "output_rl_db": float(-20.0 * np.log10(max(abs(matrix[1, 1]), EPS))),
                "insertion_efficiency": float(delivered / accepted),
                "transducer_efficiency": float(delivered),
                "s21_db": float(10.0 * np.log10(max(delivered, EPS))),
            }
        )
    write_csv(out / "launch_1x1_s2" / "frequency_metrics.csv", rows)
    profile = profile_metrics(touchstone.parent)
    segment_diagnostics = small_segment_diagnostics(touchstone.parent)
    common = common_s_metrics(selected)
    gates = config["gates"]
    summary = {
        **profile,
        **segment_diagnostics,
        **common,
        "input_rl_min_db": min(row["input_rl_db"] for row in rows),
        "output_rl_min_db": min(row["output_rl_db"] for row in rows),
        "insertion_efficiency_min": min(row["insertion_efficiency"] for row in rows),
        "transducer_efficiency_min": min(row["transducer_efficiency"] for row in rows),
    }
    summary["gate_pass"] = bool(
        summary.get("converged") is True
        and float(summary.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
        and int(summary.get("small_mesh_segment_count") or 0)
        <= int(gates["maximum_small_mesh_segments_per_differential_channel"])
        and (
            summary.get("small_mesh_segment_min_length_mm") is None
            or float(summary["small_mesh_segment_min_length_mm"])
            >= float(gates["minimum_small_mesh_segment_length_mm"])
        )
        and summary["reciprocity_error"] <= float(gates["maximum_reciprocity_error"])
        and summary["passivity_sigma"] <= float(gates["maximum_passivity_sigma"])
        and summary["input_rl_min_db"] >= float(gates["minimum_1x1_input_rl_db"])
        and summary["output_rl_min_db"] >= float(gates["minimum_1x1_output_rl_db"])
        and summary["insertion_efficiency_min"] >= float(gates["minimum_1x1_insertion_efficiency"])
        and summary["transducer_efficiency_min"] >= float(gates["minimum_1x1_transducer_efficiency"])
    )
    write_json(out / "launch_1x1_s2" / "analysis.json", summary)
    decision = {
        "stage": "C_1x1_launch_gate_complete",
        "launch_gate_pass": summary["gate_pass"],
        "allow_2x2_solve": summary["gate_pass"],
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "frequency_rows": rows, "decision": decision}


def loaded_efficiencies(
    network: np.ndarray,
    antenna: np.ndarray,
    sources: np.ndarray,
) -> tuple[float, float]:
    external, load_incident_map, load_reflected_map = terminate_network(network, antenna)
    source_reflected = external @ sources
    load_incident = load_incident_map @ sources
    load_reflected = load_reflected_map @ sources
    source_incident_power = np.sum(np.abs(sources) ** 2, axis=0)
    source_accepted = source_incident_power - np.sum(np.abs(source_reflected) ** 2, axis=0)
    load_accepted = np.sum(np.abs(load_incident) ** 2, axis=0) - np.sum(np.abs(load_reflected) ** 2, axis=0)
    insertion = load_accepted / np.maximum(source_accepted, EPS)
    transducer = load_accepted / np.maximum(source_incident_power, EPS)
    return float(np.min(insertion)), float(np.min(transducer))


def analyze_modal(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    serial_manifest = out / "modal_2x2_s8_serial" / "serial_case_manifest.json"
    profiles = []
    diagnostics = []
    if serial_manifest.exists():
        selected_rows = []
        for case in read_json(serial_manifest)["cases"]:
            touchstone = Path(case["touchstone_path"])
            if not touchstone.exists():
                raise RuntimeError(f"The serial 2x2 Touchstone is missing: {touchstone}")
            frequencies, matrices = reordered_network(touchstone, case["expected_port_order"], 8)
            selected_rows.append(matrices[int(np.argmin(np.abs(frequencies - float(case["frequency_ghz"]))))])
            profiles.append(profile_metrics(touchstone.parent))
            diagnostics.append(small_segment_diagnostics(touchstone.parent))
        selected_network = np.asarray(selected_rows)
        target = np.asarray([float(case["frequency_ghz"]) for case in read_json(serial_manifest)["cases"]])
        analysis_folder = out / "modal_2x2_s8_serial"
        execution_mode = "exact_single_frequency_adaptive_serial"
    else:
        manifest = read_json(out / "case_manifest.json")["modal_2x2"]
        touchstone = Path(manifest["touchstone_path"])
        if not touchstone.exists():
            raise RuntimeError("The 2x2 Touchstone is missing")
        frequencies, matrices = reordered_network(touchstone, manifest["expected_port_order"], 8)
        target = np.asarray(config["frequencies_ghz"], dtype=float)
        selected_network = np.asarray([matrices[int(np.argmin(np.abs(frequencies - value)))] for value in target])
        profiles = [profile_metrics(touchstone.parent)]
        diagnostics = [small_segment_diagnostics(touchstone.parent)]
        analysis_folder = out / "modal_2x2_s8"
        execution_mode = "discrete_three_frequency_sweep"
    antenna_f, antenna = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    stimulus_rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    rows = []
    all_sources = []
    all_considered = []
    external_by_frequency = []
    for frequency, network in zip(target, selected_network):
        load = antenna[int(np.argmin(np.abs(antenna_f - frequency)))]
        external, _, _ = terminate_network(network, load)
        selected = np.asarray(
            [int(row["side"]) == 2 and abs(float(row["frequency_ghz"]) - frequency) <= 1.0e-9 for row in stimulus_rows]
        )
        sources = vectors[selected, :4].T
        active = considered[selected, :4].T
        active_rl, total_rl = active_metrics(external, sources, active)
        matched_external, load_incident, load_reflected = terminate_network(network, np.zeros((4, 4), dtype=complex))
        accepted = 1.0 - np.sum(np.abs(matched_external) ** 2, axis=0)
        delivered = np.sum(np.abs(load_incident) ** 2, axis=0) - np.sum(np.abs(load_reflected) ** 2, axis=0)
        insertion, transducer = loaded_efficiencies(network, load, sources)
        rows.append(
            {
                "frequency_ghz": float(frequency),
                "matched_load_passive_rl_min_db": float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matched_external)), EPS)))),
                "representative_active_rl_min_db": active_rl,
                "representative_total_rl_min_db": total_rl,
                "matched_load_network_efficiency_min": float(np.min(delivered / np.maximum(accepted, EPS))),
                "actual_load_insertion_efficiency_min": insertion,
                "actual_load_transducer_efficiency_min": transducer,
                "representative_source_count": int(np.sum(selected)),
            }
        )
        all_sources.append(sources)
        all_considered.append(active)
        external_by_frequency.append(external)
    write_csv(analysis_folder / "frequency_metrics.csv", rows)
    profile = {
        "pass_count": min(int(item.get("pass_count") or 0) for item in profiles),
        "final_delta_s": max(float(item.get("final_delta_s") or math.inf) for item in profiles),
        "converged": all(item.get("converged") is True for item in profiles),
        "small_mesh_segment_count": max(int(item.get("small_mesh_segment_count") or 0) for item in profiles),
        "peak_solver_memory_gb": max(float(item.get("peak_solver_memory_gb") or 0.0) for item in profiles),
        "maximum_tetrahedra": max(int(item.get("maximum_tetrahedra") or 0) for item in profiles),
    }
    segment_diagnostics = {
        "small_mesh_segment_min_length_mm": min(
            (float(item["small_mesh_segment_min_length_mm"]) for item in diagnostics if item["small_mesh_segment_min_length_mm"] is not None),
            default=None,
        ),
        "small_mesh_segment_max_length_mm": max(
            (float(item["small_mesh_segment_max_length_mm"]) for item in diagnostics if item["small_mesh_segment_max_length_mm"] is not None),
            default=None,
        ),
    }
    common = common_s_metrics(selected_network)
    gates = config["gates"]
    summary = {
        **profile,
        **segment_diagnostics,
        **common,
        "solver_execution_mode": execution_mode,
        "per_frequency_profiles": profiles,
        "matched_load_passive_rl_min_db": min(row["matched_load_passive_rl_min_db"] for row in rows),
        "representative_active_rl_min_db": min(row["representative_active_rl_min_db"] for row in rows),
        "representative_total_rl_min_db": min(row["representative_total_rl_min_db"] for row in rows),
        "matched_load_network_efficiency_min": min(row["matched_load_network_efficiency_min"] for row in rows),
        "actual_load_insertion_efficiency_min": min(row["actual_load_insertion_efficiency_min"] for row in rows),
        "actual_load_transducer_efficiency_min": min(row["actual_load_transducer_efficiency_min"] for row in rows),
    }
    gate_checks = {
        "converged": summary.get("converged") is True,
        "final_delta_s": float(summary.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"]),
        "mesh_segment_count": int(summary.get("small_mesh_segment_count") or 0)
        <= 4 * int(gates["maximum_small_mesh_segments_per_differential_channel"]),
        "mesh_segment_length": summary.get("small_mesh_segment_min_length_mm") is None
        or float(summary["small_mesh_segment_min_length_mm"]) >= float(gates["minimum_small_mesh_segment_length_mm"]),
        "reciprocity": summary["reciprocity_error"] <= float(gates["maximum_reciprocity_error"]),
        "passivity": summary["passivity_sigma"] <= float(gates["maximum_passivity_sigma"]),
        "passive_rl": summary["matched_load_passive_rl_min_db"] >= float(gates["minimum_2x2_matched_load_passive_rl_db"]),
        "active_rl": summary["representative_active_rl_min_db"] >= float(gates["minimum_2x2_representative_active_rl_db"]),
        "total_rl": summary["representative_total_rl_min_db"] >= float(gates["minimum_2x2_representative_total_rl_db"]),
        "network_efficiency": summary["matched_load_network_efficiency_min"] >= float(gates["minimum_2x2_matched_load_network_efficiency"]),
        "actual_load_insertion_efficiency": summary["actual_load_insertion_efficiency_min"] >= float(gates["minimum_2x2_actual_load_insertion_efficiency"]),
        "actual_load_transducer_efficiency": summary["actual_load_transducer_efficiency_min"] >= float(gates["minimum_2x2_actual_load_transducer_efficiency"]),
    }
    summary["gate_checks"] = gate_checks
    summary["failed_gates"] = [name for name, passed in gate_checks.items() if not passed]
    summary["engineering_margins"] = {
        "active_rl_db": summary["representative_active_rl_min_db"] - float(gates["minimum_2x2_representative_active_rl_db"]),
        "total_rl_db": summary["representative_total_rl_min_db"] - float(gates["minimum_2x2_representative_total_rl_db"]),
        "actual_load_insertion_efficiency": summary["actual_load_insertion_efficiency_min"] - float(gates["minimum_2x2_actual_load_insertion_efficiency"]),
        "actual_load_transducer_efficiency": summary["actual_load_transducer_efficiency_min"] - float(gates["minimum_2x2_actual_load_transducer_efficiency"]),
    }
    summary["gate_pass"] = all(gate_checks.values())
    write_json(analysis_folder / "analysis.json", summary)
    decision = {
        "stage": "D_2x2_modal_branch_gate_complete",
        "modal_s8_gate_pass": summary["gate_pass"],
        "allow_independent_repeat": summary["gate_pass"],
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "The S8 physical gate failed: " + ", ".join(summary["failed_gates"])
            if not summary["gate_pass"]
            else "The S8 gate passed, but integrated 2x2 remains locked until an independent repeat passes Delta S <= 0.05."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "frequency_rows": rows, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    return {
        "output_directory": str(out),
        "preregistered": (out / "preregistration.json").exists(),
        "build_smoke_complete": (out / "build_smoke_audit.csv").exists(),
        "launch_s2_complete": len(list((out / "launch_1x1_s2").glob("*.s2p"))) > 0 if out.exists() else False,
        "modal_s8_complete": len(list((out / "modal_2x2_s8").glob("*.s8p"))) > 0 if out.exists() else False,
        "modal_s8_serial_complete": len(list((out / "modal_2x2_s8_serial").glob("f_*ghz/*.s8p"))) == len(config["frequencies_ghz"]) if out.exists() else False,
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=(
            "preregister",
            "apply-mesh-selection-amendment",
            "run-build-smoke",
            "audit-build-smoke",
            "run-launch",
            "analyze-launch",
            "run-modal",
            "run-modal-serial",
            "analyze-modal",
            "status",
        ),
        default="status",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    actions = {
        "preregister": preregister,
        "apply-mesh-selection-amendment": apply_mesh_selection_amendment,
        "run-build-smoke": run_build_smoke,
        "audit-build-smoke": audit_build_smoke,
        "run-launch": lambda item: run_solve(item, "launch_1x1", "allow_1x1_solve"),
        "analyze-launch": analyze_launch,
        "run-modal": lambda item: run_solve(item, "modal_2x2", "allow_2x2_solve"),
        "run-modal-serial": run_modal_serial,
        "analyze-modal": analyze_modal,
        "status": status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
