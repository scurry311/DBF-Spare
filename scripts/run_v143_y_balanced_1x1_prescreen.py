#!/usr/bin/env python3
"""Build, solve and gate the v1.43 y-balanced 1x1 input prescreen."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_v137_vertical_mesh_audit import segment_audit
from run_v114_small_cell_broadband_feed import (
    efficiency_from_csv,
    memory_available_gb,
    parse_touchstone,
    profile_metrics,
)
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v125_feedpoint_input_impedance import topology_warning_count, write_json
from run_v128_true_balanced_dual_resonant import vp
from run_v132_vertical_differential_launch import helpers
from run_v139_physical_2x2_differential_array import element_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v143_y_balanced_1x1_prescreen.json"
DESIGN_NAME = "V143_YBalanced_1x1"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def substrate_text(g: dict[str, Any]) -> str:
    h = float(g["substrate_thickness_mm"])
    radius = float(g["via_radius_mm"])
    pitch = float(g["via_pair_pitch_mm"])
    holes = [(-pitch / 2.0 - radius, -pitch / 2.0 + radius), (pitch / 2.0 - radius, pitch / 2.0 + radius)]
    xs = sorted({-7.5, 7.5, *(value for pair in holes for value in pair)})
    ys = [-7.5, -radius, radius, 7.5]
    blocks = []
    for ix, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        for iy, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            in_hole = abs(my) < radius and any(left < mx < right for left, right in holes)
            if in_hole:
                continue
            blocks.append(
                f'CreateBox oEditor, "SubstratePart_{ix:02d}_{iy:02d}", {x0:.7f}, {y0:.7f}, {-h:.7f}, {x1-x0:.7f}, {y1-y0:.7f}, {h:.7f}, "RO5880_V143", True'
            )
    return "\n".join(blocks)


def builder_text(project: Path, g: dict[str, Any], frequency: float) -> str:
    element, names = element_text(g, "P_DIFF", 0.0, 0.0)
    sheets = [name for name in names if name.startswith("Primary") or name.startswith("Pad")]
    primary = [name for name in names if name.startswith("Primary")]
    sheet_objects = ", ".join(f'"{name}"' for name in sheets)
    primary_objects = ", ".join(f'"{name}"' for name in primary)
    h = float(g["substrate_thickness_mm"])
    copper = float(g["copper_thickness_mm"])
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V143", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(g['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(g['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
{substrate_text(g)}
{element}
oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", "Objects:=", Array({sheet_objects}), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)
CreateBox oEditor, "AirRegion", -19.5, -19.5, {-h-12:.7f}, 39, 39, {h+24.035:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:YBalancedInputMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({primary_objects}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(g['local_mesh_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {int(g['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {float(g['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False, "DrivenSolverType:=", "Direct Solver")
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V143", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{helpers()}
'''


def solver_text(project: Path, touchstone: Path, efficiency: Path, frequency: float) -> str:
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
reportName = "V143_RadiationEfficiency"
On Error Resume Next
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V143"), Array("Freq:=", Array("{frequency:g}GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(efficiency)}"
On Error GoTo 0
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Concurrent AEDT/HFSS is not allowed: {processes}")


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.43 1x1 output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    project = out / "v143_y_balanced_1x1.aedt"
    touchstone = out / "v143_y_balanced_1x1.s1p"
    efficiency = out / "radiation_efficiency.csv"
    build = out / "build.vbs"
    solve = out / "solve_export.vbs"
    source = builder_text(project, config["geometry"], float(config["frequency_ghz"]))
    build.write_text(source, encoding="ascii")
    solve.write_text(solver_text(project, touchstone, efficiency, float(config["frequency_ghz"])), encoding="ascii")
    manifest = {
        "project": str(project),
        "touchstone": str(touchstone),
        "efficiency": str(efficiency),
        "build": str(build),
        "solve": str(solve),
        "cad_audit": {
            "differential_port_count": source.count("AssignDifferentialPortZ oBoundary"),
            "lower_branch_count": source.count("SecondaryLower"),
            "reference_ground_count": source.count('"ReferenceGround"') + source.count('"Ground"'),
            "finite_conductivity": "AssignFiniteCond" in source,
            "direct_solver_count": source.count("DrivenSolverType:="),
        },
    }
    write_json(out / "preregistration.json", {**config, "manifest": manifest})
    write_json(out / "stage_decision.json", {"stage": "prepared", "allow_build": True, "allow_solve": False, **config["scope"]})
    return manifest


def build(config: dict[str, Any]) -> dict[str, Any]:
    require_no_aedt()
    out = resolve(config["output_directory"])
    manifest = read_json(out / "preregistration.json")["manifest"]
    with (out / "build.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run([str(resolve(config["ansys_executable"])), "-RunScriptAndExit", manifest["build"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    audit = manifest["cad_audit"]
    passed = bool(result.returncode == 0 and Path(manifest["project"]).exists() and audit["differential_port_count"] == 1 and audit["lower_branch_count"] >= 4 and audit["reference_ground_count"] == 0 and audit["finite_conductivity"] and audit["direct_solver_count"] == 1 and topology_warning_count(out) == 0)
    result_row = {"return_code": result.returncode, "project_exists": Path(manifest["project"]).exists(), "topology_warning_count": topology_warning_count(out), "build_gate_pass": passed, "cad_audit": audit}
    write_json(out / "build_audit.json", result_row)
    if not passed:
        raise RuntimeError(f"1x1 build failed: {result_row}")
    write_json(out / "stage_decision.json", {"stage": "build_complete", "allow_build": False, "allow_solve": True, **config["scope"]})
    return result_row


def solve(config: dict[str, Any]) -> dict[str, Any]:
    require_no_aedt()
    out = resolve(config["output_directory"])
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_solve"):
        raise RuntimeError("1x1 solve is not authorized")
    required = float(config["resources"]["minimum_free_memory_before_solve_gib"])
    free = memory_available_gb()
    if free < required:
        raise MemoryError(f"Need {required:.2f} GiB free, found {free:.2f} GiB")
    manifest = read_json(out / "preregistration.json")["manifest"]
    code, aborted, minimum = run_process_with_memory_guard(
        [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", manifest["solve"]],
        out / "solve_export.log",
        float(config["resources"]["abort_free_memory_during_solve_gib"]),
        float(config["resources"]["poll_interval_seconds"]),
    )
    touchstone = Path(manifest["touchstone"])
    row = {"return_code": code, "memory_aborted": aborted, "free_memory_gib_before": free, "minimum_free_memory_gib": minimum, "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100}
    write_json(out / "run_progress.json", row)
    if code != 0 or aborted or not row["touchstone_exists"]:
        raise RuntimeError(f"1x1 solve failed: {row}")
    return row


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest = read_json(out / "preregistration.json")["manifest"]
    frequency, matrices = parse_touchstone(Path(manifest["touchstone"]), 1)
    index = int(np.argmin(np.abs(frequency - float(config["frequency_ghz"]))))
    gamma = complex(matrices[index, 0, 0])
    impedance = 50.0 * (1.0 + gamma) / (1.0 - gamma)
    rl = float(-20.0 * np.log10(max(abs(gamma), 1.0e-15)))
    efficiency = efficiency_from_csv(Path(manifest["efficiency"]))
    profile = profile_metrics(out)
    bodies, lengths = segment_audit(out)
    conductor_count = sum(value for name, value in bodies.items() if name.startswith(("Primary", "Pad", "Via")))
    target = complex(float(config["target"]["input_resistance_ohm"]), float(config["target"]["input_reactance_ohm"]))
    impedance_error = abs(impedance - target)
    gates = config["gates"]
    passed = bool(profile.get("converged") is True and float(profile.get("final_delta_s") or np.inf) <= float(gates["maximum_final_delta_s"]) and rl >= float(gates["minimum_passive_rl_db"]) and efficiency is not None and efficiency >= float(gates["minimum_radiation_efficiency"]) and conductor_count <= int(gates["maximum_conductor_small_segment_count"]) and topology_warning_count(out) <= int(gates["maximum_topology_warning_count"]) and impedance_error <= float(config["target"]["maximum_complex_impedance_error_ohm"]))
    summary = {**profile, "frequency_ghz": float(frequency[index]), "s11_real": gamma.real, "s11_imag": gamma.imag, "passive_rl_db": rl, "input_resistance_ohm": impedance.real, "input_reactance_ohm": impedance.imag, "target_impedance_error_ohm": float(impedance_error), "radiation_efficiency": efficiency, "conductor_small_segment_count": int(conductor_count), "minimum_segment_length_mm": min(lengths) if lengths else None, "topology_warning_count": topology_warning_count(out), "input_prescreen_gate_pass": passed, "allow_2x2": False, "allow_eep_or_training": False}
    write_json(out / "stage_summary.json", summary)
    write_json(out / "stage_decision.json", {"stage": "analyzed", "input_prescreen_gate_pass": passed, "allow_2x2": False, "allow_4x4": False, "allow_16x16": False, "allow_eep_export": False, "allow_training_labels": False, "allow_critic_training": False, "next_action": "Use this 1x1 result only to tune self impedance; a separate memory-feasible 2x2 is required for coupling and active-RL."})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("prepare", "build", "solve", "analyze"), required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    actions = {"prepare": prepare, "build": build, "solve": solve, "analyze": analyze}
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
