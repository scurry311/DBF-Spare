#!/usr/bin/env python3
"""Reproduce and independently verify a literature-backed 10 GHz balun dipole."""

from __future__ import annotations

import argparse
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
from run_v128_true_balanced_dual_resonant import vp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v147_literature_10ghz_balun_dipole_preregistered.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def solver_suffix(solver_type: str) -> str:
    if solver_type == "direct":
        return ', "DrivenSolverType:=", "Direct Solver"'
    if solver_type == "ddm":
        return ', "DrivenSolverType:=", "Domain Decomposition", "IterativeResidual:=", 0.000001, "DDMSolverResidual:=", 0.000001'
    raise ValueError(f"Unsupported solver type: {solver_type}")


def vbs_helpers() -> str:
    return r'''
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function

Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(120 150 170)", "Transparency:=", 0.65, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub

Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub

Sub CreateSheetY(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Y"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub

Sub CreateCircleZ(editor, objName, x, y, z, radius)
    editor.CreateCircle Array("NAME:CircleParameters", "IsCovered:=", True, "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "WhichAxis:=", "Z", "NumSegments:=", "0"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub

Sub SubtractSelection(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", False)
End Sub

Sub UniteSelection(editor, names)
    editor.Unite Array("NAME:Selections", "Selections:=", names), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
'''


def builder_text(project: Path, config: dict[str, Any], solver_type: str) -> str:
    g = config["geometry"]
    sweep = config["sweep"]
    name = str(config["design_name"])
    t = float(g["substrate_thickness_mm"])
    half_t = t / 2.0
    board_w = float(g["substrate_width_mm"])
    board_h = float(g["substrate_height_mm"])
    reflector = float(g["reflector_size_mm"])
    dipole_l = float(g["dipole_total_length_mm"])
    dipole_w = float(g["dipole_arm_width_mm"])
    slot = float(g["slotline_width_mm"])
    slot_l = float(g["slotline_length_mm"])
    leg_separation = float(g["microstrip_leg_separation_mm"])
    back_ground_w = float(g["back_slotline_ground_width_mm"])
    balun_l = float(g["balun_length_mm"])
    branch_l = float(g["match_branch_length_mm"])
    ms_a = float(g["microstrip_a_width_mm"])
    ms_b = float(g["microstrip_b_width_mm"])
    center_z = float(g["radiator_center_height_mm"])
    arm_l = (dipole_l - slot) / 2.0
    back_half = (back_ground_w - slot) / 2.0
    dipole_z = center_z - dipole_w / 2.0
    back_top = center_z + dipole_w / 2.0
    feed_x = -leg_separation / 2.0
    stub_x = leg_separation / 2.0
    feed_left = feed_x - ms_a / 2.0
    stub_left = stub_x - ms_b / 2.0
    bridge_left = feed_left
    bridge_width = stub_x + ms_b / 2.0 - bridge_left
    bridge_z = balun_l - ms_b
    stub_bottom = balun_l - branch_l
    clearance_x = float(g["ground_port_clearance_x_mm"])
    clearance_y = float(g["ground_port_clearance_y_mm"])
    port_reference_x = float(g["port_reference_x_mm"])
    if not (feed_left < port_reference_x < feed_x + ms_a / 2.0):
        raise ValueError("port_reference_x_mm must lie on the front microstrip feed leg")
    if back_ground_w <= leg_separation + max(ms_a, ms_b):
        raise ValueError("back_slotline_ground_width_mm must cover both microstrip legs")
    copper = float(g["copper_thickness_mm"])
    mesh = float(g["local_mesh_max_length_mm"])
    frequency = float(sweep["adaptive_frequency_ghz"])
    broad_start = float(sweep["broad_start_ghz"])
    broad_stop = float(sweep["broad_stop_ghz"])
    broad_count = int(sweep["broad_count"])
    gate = [float(item) for item in sweep["gate_frequencies_ghz"]]
    if len(gate) != 3 or not math.isclose(gate[1], frequency, abs_tol=1e-9):
        raise ValueError("gate_frequencies_ghz must contain [low, adaptive, high]")
    conductor_names = ["Ground", "BackLeft", "BackRight", "MSFeed"]
    conductors = ", ".join(f'"{item}"' for item in conductor_names)
    local_mesh = ", ".join(f'"{item}"' for item in ["BackLeft", "BackRight", "MSFeed", "PortSheet"])
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V147", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(g['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(g['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{name}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{name}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")

' Literature-backed vertical RO5880 printed dipole and integrated microstrip/slotline balun.
CreateBox oEditor, "Substrate", {-board_w/2:.7f}, {-half_t:.7f}, 0, {board_w:.7f}, {t:.7f}, {board_h:.7f}, "RO5880_V147", True
CreateCircleZ oEditor, "Ground", 0, 0, 0, {reflector/2:.7f}
CreateSheetZ oEditor, "PortClearance", {feed_x-clearance_x/2:.7f}, {-half_t:.7f}, 0, {clearance_x:.7f}, {clearance_y:.7f}
SubtractSelection oEditor, "Ground", "PortClearance"

' Back-side broad slotline ground conductors and the dipole arms are one continuous copper topology.
CreateSheetY oEditor, "BackLeft", {-back_ground_w/2:.7f}, {-half_t:.7f}, 0, {back_half:.7f}, {back_top:.7f}
CreateSheetY oEditor, "BackRight", {slot/2:.7f}, {-half_t:.7f}, 0, {back_half:.7f}, {back_top:.7f}
CreateSheetY oEditor, "DipoleLeft", {-dipole_l/2:.7f}, {-half_t:.7f}, {dipole_z:.7f}, {arm_l:.7f}, {dipole_w:.7f}
CreateSheetY oEditor, "DipoleRight", {slot/2:.7f}, {-half_t:.7f}, {dipole_z:.7f}, {arm_l:.7f}, {dipole_w:.7f}
UniteSelection oEditor, "BackLeft,DipoleLeft"
UniteSelection oEditor, "BackRight,DipoleRight"

' Front-side U-shaped microstrip-to-slotline transition: line a, transverse coupling bridge, and open line b.
CreateSheetY oEditor, "MSFeed", {feed_left:.7f}, {half_t:.7f}, 0, {ms_a:.7f}, {balun_l:.7f}
CreateSheetY oEditor, "MSBridge", {bridge_left:.7f}, {half_t:.7f}, {bridge_z:.7f}, {bridge_width:.7f}, {ms_b:.7f}
CreateSheetY oEditor, "MSOpenStub", {stub_left:.7f}, {half_t:.7f}, {stub_bottom:.7f}, {ms_b:.7f}, {branch_l:.7f}
UniteSelection oEditor, "MSFeed,MSBridge,MSOpenStub"

' Lumped 50-ohm reference at the reflector plane, between the front line and connected back ground.
CreateSheetZ oEditor, "PortSheet", {feed_left:.7f}, {-half_t:.7f}, 0, {ms_a:.7f}, {t:.7f}
oBoundary.AssignLumpedPort Array("NAME:P1", "Objects:=", Array("PortSheet"), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array("{port_reference_x:.7f}mm", "{-half_t:.7f}mm", "0mm"), "End:=", Array("{port_reference_x:.7f}mm", "{half_t:.7f}mm", "0mm")), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")

oBoundary.AssignFiniteCond Array("NAME:CopperFiniteConductivity", "Objects:=", Array({conductors}), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)
CreateBox oEditor, "AirRegion", -25, -25, -15, 50, 50, 45, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:UnifiedBalunSlotMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({local_mesh}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {int(g['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {float(g['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False{solver_suffix(solver_type)})
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Broad", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "{broad_start:g}GHz", "RangeEnd:=", "{broad_stop:g}GHz", "RangeCount:=", {broad_count}, "Type:=", "Interpolating", "SaveFields:=", False, "SaveRadFields:=", False, "InterpTolerance:=", 0.5, "InterpMaxSolns:=", 250, "InterpMinSolns:=", 0, "InterpMinSubranges:=", 1, "InterpUseS:=", True, "InterpUsePortImped:=", True, "InterpUsePropConst:=", True, "UseDerivativeConvergence:=", False, "InterpDerivTolerance:=", 0.2, "UseFullBasis:=", True, "EnforcePassivity:=", True, "PassivityErrorTolerance:=", 0.0001, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "{gate[0]:g}GHz", "RangeEnd:=", "{gate[2]:g}GHz", "RangeCount:=", 3, "Type:=", "Discrete", "SaveFields:=", True, "SaveRadFields:=", True, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V147", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{vbs_helpers()}
'''


def solver_text(project: Path, broad: Path, gate3: Path, efficiency: Path, design_name: str) -> str:
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, reportName
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{design_name}")
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Broad"), 3, "{vp(broad)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Gate3"), 3, "{vp(gate3)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
Set oReport = oDesign.GetModule("ReportSetup")
reportName = "V147_RadiationEfficiency"
On Error Resume Next
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V147"), Array("Freq:=", Array("10GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(efficiency)}"
On Error GoTo 0
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Concurrent AEDT/HFSS is not allowed: {processes}")


def case_record(config: dict[str, Any], case_name: str) -> dict[str, Any]:
    matches = [item for item in config["cases"] if item["name"] == case_name]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate case: {case_name}")
    return matches[0]


def case_paths(config: dict[str, Any], case_name: str) -> dict[str, Path]:
    folder = resolve(config["output_directory"]) / case_name
    return {
        "folder": folder,
        "project": folder / f"v147_{case_name}.aedt",
        "build": folder / "build.vbs",
        "solve": folder / "solve_export.vbs",
        "broad": folder / f"v147_{case_name}_broad.s1p",
        "gate3": folder / f"v147_{case_name}_gate3.s1p",
        "efficiency": folder / "radiation_efficiency.csv",
    }


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.47 output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    batch_options = out / "serial_frequency_batchoptions.acf"
    batch_options.write_text(
        "$begin 'Config'\n"
        f"'Desktop/Settings/ProjectOptions/NumberOfProcessors'={int(config['resources']['number_of_processors'])}\n"
        f"'HFSS/NumCoresPerDistributedTask'={int(config['resources']['cores_per_distributed_task'])}\n"
        "$end 'Config'\n",
        encoding="ascii",
    )
    manifest: dict[str, Any] = {"cases": {}}
    for case in config["cases"]:
        paths = case_paths(config, case["name"])
        paths["folder"].mkdir(parents=True)
        build_source = builder_text(paths["project"], config, str(case["solver_type"]))
        paths["build"].write_text(build_source, encoding="ascii")
        paths["solve"].write_text(
            solver_text(paths["project"], paths["broad"], paths["gate3"], paths["efficiency"], str(config["design_name"])),
            encoding="ascii",
        )
        manifest["cases"][case["name"]] = {
            **case,
            **{key: str(value) for key, value in paths.items()},
            "batch_options": str(batch_options),
            "cad_audit": {
                "port_count": build_source.count('AssignLumpedPort Array("NAME:P1"'),
                "broad_sweep_count": build_source.count('"NAME:Sweep_Broad"'),
                "gate3_sweep_count": build_source.count('"NAME:Sweep_Gate3"'),
                "finite_conductivity": "AssignFiniteCond" in build_source,
                "solver_type": case["solver_type"],
            },
        }
    write_json(out / "preregistration.json", {**config, "manifest": manifest})
    write_json(out / "stage_decision.json", {
        "stage": "prepared",
        "allow_nominal_build": True,
        "allow_independent_validation": False,
        "allow_2x2": False,
        **config["locks"],
    })
    return {"output_directory": str(out), "case_count": len(config["cases"]), "manifest": manifest}


def build_case(config: dict[str, Any], case_name: str) -> dict[str, Any]:
    require_no_aedt()
    out = resolve(config["output_directory"])
    prereg = read_json(out / "preregistration.json")
    item = prereg["manifest"]["cases"][case_name]
    if case_name != "nominal_direct":
        decision = read_json(out / "stage_decision.json")
        if not decision.get("allow_independent_validation"):
            raise RuntimeError("Independent 1x1 validation is locked until nominal frequency-sweep gate passes")
    folder = Path(item["folder"])
    with (folder / "build.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(resolve(config["ansys_executable"])), "-RunScriptAndExit", item["build"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    audit = item["cad_audit"]
    passed = bool(
        result.returncode == 0
        and Path(item["project"]).exists()
        and audit["port_count"] == 1
        and audit["broad_sweep_count"] == 1
        and audit["gate3_sweep_count"] == 1
        and audit["finite_conductivity"]
        and topology_warning_count(folder) == 0
    )
    row = {
        "case": case_name,
        "return_code": int(result.returncode),
        "project_exists": Path(item["project"]).exists(),
        "topology_warning_count": topology_warning_count(folder),
        "cad_audit": audit,
        "build_gate_pass": passed,
    }
    write_json(folder / "build_audit.json", row)
    if not passed:
        raise RuntimeError(f"v1.47 build failed: {row}")
    return row


def solve_case(config: dict[str, Any], case_name: str) -> dict[str, Any]:
    require_no_aedt()
    out = resolve(config["output_directory"])
    item = read_json(out / "preregistration.json")["manifest"]["cases"][case_name]
    folder = Path(item["folder"])
    required = float(config["resources"]["minimum_free_memory_before_solve_gib"])
    free = memory_available_gb()
    if free < required:
        raise MemoryError(f"Need {required:.2f} GiB free, found {free:.2f} GiB")
    code, aborted, minimum = run_process_with_memory_guard(
        [
            str(resolve(config["ansys_executable"])),
            "-ng",
            "-Distributed",
            "-MachineList",
            "list=scurry:1:4:90%:0",
            "excludeTypes=Frequencies",
            "-BatchOptions",
            item["batch_options"],
            "-BatchSolve",
            item["project"],
        ],
        folder / "batch_solve.log",
        float(config["resources"]["abort_free_memory_during_solve_gib"]),
        float(config["resources"]["poll_interval_seconds"]),
    )
    export_code = None
    export_aborted = False
    export_minimum = math.inf
    if code == 0 and not aborted:
        require_no_aedt()
        export_code, export_aborted, export_minimum = run_process_with_memory_guard(
            [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", item["solve"]],
            folder / "solve_export.log",
            float(config["resources"]["abort_free_memory_during_solve_gib"]),
            float(config["resources"]["poll_interval_seconds"]),
        )
        minimum = min(minimum, export_minimum)
    final_code = code if code != 0 else int(export_code or 0)
    aborted = bool(aborted or export_aborted)
    row = {
        "case": case_name,
        "return_code": final_code,
        "batch_solve_return_code": code,
        "export_return_code": export_code,
        "memory_aborted": aborted,
        "free_memory_gib_before": free,
        "minimum_free_memory_gib": minimum,
        "broad_touchstone_exists": Path(item["broad"]).exists() and Path(item["broad"]).stat().st_size > 100,
        "gate3_touchstone_exists": Path(item["gate3"]).exists() and Path(item["gate3"]).stat().st_size > 100,
    }
    write_json(folder / "run_progress.json", row)
    if final_code != 0 or aborted or not row["broad_touchstone_exists"] or not row["gate3_touchstone_exists"]:
        raise RuntimeError(f"v1.47 solve failed: {row}")
    return row


def rl_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency, matrices = parse_touchstone(path, 1)
    gamma = matrices[:, 0, 0]
    rl = -20.0 * np.log10(np.maximum(np.abs(gamma), 1.0e-15))
    return frequency, gamma, rl


def contiguous_bandwidth(frequency: np.ndarray, rl: np.ndarray, threshold: float, center: float) -> tuple[float, float, float]:
    passed = rl >= threshold
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


def analyze_case(config: dict[str, Any], case_name: str) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    item = read_json(out / "preregistration.json")["manifest"]["cases"][case_name]
    folder = Path(item["folder"])
    broad_f, broad_g, broad_rl = rl_curve(Path(item["broad"]))
    gate_f, gate_g, gate_rl = rl_curve(Path(item["gate3"]))
    center_idx = int(np.argmin(np.abs(gate_f - float(config["sweep"]["adaptive_frequency_ghz"]))))
    bandwidth, band_low, band_high = contiguous_bandwidth(broad_f, broad_rl, 10.0, float(config["sweep"]["adaptive_frequency_ghz"]))
    efficiency = efficiency_from_csv(Path(item["efficiency"]))
    profile = profile_metrics(folder)
    gates = config["gates"]
    passed = bool(
        profile.get("converged") is True
        and float(profile.get("final_delta_s") or np.inf) <= float(gates["maximum_final_delta_s"])
        and float(np.min(gate_rl)) >= float(gates["minimum_gate3_passive_rl_db"])
        and float(gate_rl[center_idx]) >= float(gates["minimum_center_passive_rl_db"])
        and bandwidth >= float(gates["minimum_contiguous_10db_bandwidth_ghz"])
        and efficiency is not None
        and efficiency >= float(gates["minimum_radiation_efficiency"])
        and topology_warning_count(folder) <= int(gates["maximum_topology_warning_count"])
    )
    summary = {
        "case": case_name,
        "solver_type": item["solver_type"],
        **profile,
        "gate3_frequencies_ghz": [float(item) for item in gate_f],
        "gate3_passive_rl_db": [float(item) for item in gate_rl],
        "gate3_worst_passive_rl_db": float(np.min(gate_rl)),
        "center_passive_rl_db": float(gate_rl[center_idx]),
        "center_s11_real": float(gate_g[center_idx].real),
        "center_s11_imag": float(gate_g[center_idx].imag),
        "broad_minimum_rl_db": float(np.min(broad_rl)),
        "broad_maximum_rl_db": float(np.max(broad_rl)),
        "contiguous_10db_bandwidth_ghz": bandwidth,
        "contiguous_10db_band_low_ghz": band_low,
        "contiguous_10db_band_high_ghz": band_high,
        "radiation_efficiency": efficiency,
        "topology_warning_count": topology_warning_count(folder),
        "one_by_one_case_gate_pass": passed,
    }
    write_json(folder / "stage_summary.json", summary)
    if case_name == "nominal_direct":
        write_json(out / "stage_decision.json", {
            "stage": "nominal_analyzed",
            "nominal_frequency_sweep_gate_pass": passed,
            "allow_independent_validation": passed,
            "allow_2x2": False,
            **config["locks"],
            "next_action": "Run frozen independent direct and DDM rebuilds." if passed else "Stop before independent validation and audit the literature geometry/reference-plane interpretation.",
        })
    return summary


def complex_interpolate(frequency: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    real = np.interp(target, frequency, values.real)
    imag = np.interp(target, frequency, values.imag)
    return real + 1j * imag


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    summaries: dict[str, Any] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in config["cases"]:
        name = str(case["name"])
        paths = case_paths(config, name)
        summary_path = paths["folder"] / "stage_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing analyzed case: {name}")
        summaries[name] = read_json(summary_path)
        frequency, gamma, _ = rl_curve(paths["broad"])
        curves[name] = (frequency, gamma)
    reference_f, reference_g = curves["nominal_direct"]
    direct_g = complex_interpolate(*curves["independent_direct"], reference_f)
    ddm_g = complex_interpolate(*curves["independent_ddm"], reference_f)
    nominal_direct_delta = float(np.max(np.abs(reference_g - direct_g)))
    direct_ddm_delta = float(np.max(np.abs(direct_g - ddm_g)))
    gates = config["gates"]
    independent_pass = bool(
        all(bool(item["one_by_one_case_gate_pass"]) for item in summaries.values())
        and direct_ddm_delta <= float(gates["maximum_direct_ddm_s_difference"])
        and nominal_direct_delta <= float(gates["maximum_direct_ddm_s_difference"])
    )
    result = {
        "version": config["version"],
        "case_summaries": summaries,
        "nominal_vs_independent_direct_max_abs_delta_s": nominal_direct_delta,
        "independent_direct_vs_ddm_max_abs_delta_s": direct_ddm_delta,
        "independent_1x1_gate_pass": independent_pass,
        "allow_2x2_active_rl": independent_pass,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "next_action": "Build a frozen 2x2 at 16 mm pitch and evaluate passive coupling plus frozen-stimulus active RL." if independent_pass else "Stop before 2x2 and reject or repair the 1x1 physical interpretation.",
    }
    write_json(out / "stage_summary.json", result)
    write_json(out / "stage_decision.json", {"stage": "independent_1x1_complete", **result})
    return result


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    result: dict[str, Any] = {"output_directory": str(out), "prepared": (out / "preregistration.json").exists(), "cases": {}}
    for case in config["cases"]:
        paths = case_paths(config, str(case["name"]))
        result["cases"][case["name"]] = {
            "project": paths["project"].exists(),
            "broad_touchstone": paths["broad"].exists(),
            "gate3_touchstone": paths["gate3"].exists(),
            "analyzed": (paths["folder"] / "stage_summary.json").exists(),
        }
    result["finalized"] = (out / "stage_summary.json").exists()
    result["aedt_processes"] = aedt_processes()
    result["free_memory_gib"] = memory_available_gb()
    return result


def run_case(config: dict[str, Any], case_name: str) -> dict[str, Any]:
    build_result = build_case(config, case_name)
    solve_result = solve_case(config, case_name)
    analysis_result = analyze_case(config, case_name)
    return {"build": build_result, "solve": solve_result, "analysis": analysis_result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("prepare", "build", "solve", "analyze", "run-case", "finalize", "status"), required=True)
    parser.add_argument("--case", choices=("nominal_direct", "independent_direct", "independent_ddm"))
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if args.mode == "prepare":
        result = prepare(config)
    elif args.mode == "finalize":
        result = finalize(config)
    elif args.mode == "status":
        result = status(config)
    else:
        if args.case is None:
            parser.error(f"--case is required for --mode {args.mode}")
        action = {"build": build_case, "solve": solve_case, "analyze": analyze_case, "run-case": run_case}[args.mode]
        result = action(config, args.case)
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
