"""Build and validate a 10 GHz grounded microstrip-to-CPS printed-dipole smoke.

The fixture replaces free-space twin leads with a manufacturable topology:
top copper microstrip, bottom reference ground, a plated ground via, a CPS
transition, and a printed dipole on a low-loss dielectric.  It is intentionally
one fixed 1x1 design, not an impedance-parameter sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "hfss_outputs" / "cps_microstrip_balun_smoke_20260717_run01"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "CPS_Microstrip_Balun_10GHz"

# Rogers 5880-like substrate and analytically seeded 50-ohm microstrip.
CONFIG: dict[str, float] = {
    "frequency_ghz": 10.0,
    "er": 2.2,
    "tan_delta": 0.0009,
    "substrate_thickness_mm": 0.787,
    "board_x_mm": 30.0,
    "board_y_mm": 32.0,
    "microstrip_width_mm": 2.40,
    "transformer_width_mm": 0.90,
    "microstrip_length_mm": 7.0,
    "transition_length_mm": 5.0,
    "cps_width_mm": 0.60,
    "cps_gap_mm": 0.40,
    "cps_length_mm": 4.0,
    "via_radius_mm": 0.20,
    "dipole_length_mm": 12.6,
    "dipole_gap_mm": 0.50,
    "dipole_width_mm": 0.80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="prepare")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def vbs_builder(project: Path) -> str:
    c = CONFIG
    h = c["substrate_thickness_mm"]
    bw = c["board_x_mm"]
    by = c["board_y_mm"]
    cps_center = (c["cps_gap_mm"] + c["cps_width_mm"]) / 2.0
    arm_length = (c["dipole_length_mm"] - c["dipole_gap_mm"]) / 2.0
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oDefinitionManager
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
Set oDefinitionManager = oProject.GetDefinitionManager()
oDefinitionManager.AddMaterial Array("NAME:RO5880_Custom", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{c['er']}", "dielectric_loss_tangent:=", "{c['tan_delta']}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")

CreateBox oEditor, "Substrate", {-bw/2:.6f}, {-by/2:.6f}, {-h:.6f}, {bw:.6f}, {by:.6f}, {h:.6f}, "RO5880_Custom", True
CreateSheetZ oEditor, "Ground", {-bw/2:.6f}, {-by/2:.6f}, {-h:.6f}, {bw:.6f}, {by:.6f}, "pec"

' 50-ohm microstrip input and a fixed high-impedance transition section.
CreateSheetZ oEditor, "MS_Wide", {-c['microstrip_width_mm']/2:.6f}, {-by/2:.6f}, 0, {c['microstrip_width_mm']:.6f}, {c['microstrip_length_mm'] + 0.2:.6f}, "pec"
CreateSheetZ oEditor, "MS_Transformer", {-c['transformer_width_mm']/2:.6f}, {-by/2+c['microstrip_length_mm']:.6f}, 0, {c['transformer_width_mm']:.6f}, {c['transition_length_mm'] + 0.2:.6f}, "pec"

' CPS conductors: signal arm is fed by top microstrip, return arm by grounded via.
CreateSheetZ oEditor, "CPS_Pos", {cps_center-c['cps_width_mm']/2:.6f}, {-by/2+c['microstrip_length_mm']+c['transition_length_mm']:.6f}, 0, {c['cps_width_mm']:.6f}, {c['cps_length_mm']:.6f}, "pec"
CreateSheetZ oEditor, "CPS_Neg", {-cps_center-c['cps_width_mm']/2:.6f}, {-by/2+c['microstrip_length_mm']+c['transition_length_mm']:.6f}, 0, {c['cps_width_mm']:.6f}, {c['cps_length_mm']:.6f}, "pec"
CreateVia oEditor, "GroundVia", {-cps_center:.6f}, {-by/2+c['microstrip_length_mm']+c['transition_length_mm']:.6f}, {-h:.6f}, {c['via_radius_mm']:.6f}, {h:.6f}

' Printed dipole, both halves are tied to the corresponding CPS strip.
CreateSheetZ oEditor, "Dipole_Right", {c['dipole_gap_mm']/2:.6f}, {-c['dipole_width_mm']/2:.6f}, 0, {arm_length:.6f}, {c['dipole_width_mm']:.6f}, "pec"
CreateSheetZ oEditor, "Dipole_Left", {-arm_length-c['dipole_gap_mm']/2:.6f}, {-c['dipole_width_mm']/2:.6f}, 0, {arm_length:.6f}, {c['dipole_width_mm']:.6f}, "pec"

' Microstrip lumped input: XZ port sheet touches top signal and bottom reference ground.
CreatePortXZ oEditor, "PortSheet_IN", -2.0, {-by/2 + 0.2:.6f}, {-h - 0.2:.6f}, 4.0, {h + 0.4:.6f}
AssignPort oBoundary, "P_IN", "PortSheet_IN", 0, {-by/2 + 0.2:.6f}, 0, 0, {-by/2 + 0.2:.6f}, {-h:.6f}

CreateBox oEditor, "AirRegion", -25, -27, -18, 50, 54, 36, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", 8, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", 15, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Function Mm(value)
    Mm = CStr(Round(CDbl(value), 6)) & "mm"
End Function

Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(120 150 170)", "Transparency:=", 0.65, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub

Sub CreateSheetZ(editor, objName, x, y, z, width, height, material)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", False)
End Sub

Sub CreateVia(editor, objName, x, y, z, radius, height)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", "16"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """pec""", "SolveInside:=", False)
End Sub

Sub CreatePortXZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Y"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub

Sub AssignPort(boundary, portName, sheetName, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
'''


def write_solve_export(path: Path, project: Path, touchstone: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Option Explicit",
                "Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, vars, variation",
                'Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")',
                "Set oDesktop = oAnsoftApp.GetAppDesktop()",
                f'oDesktop.OpenProject "{vp(project)}"',
                f'Set oProject = oDesktop.SetActiveProject("{project.stem}")',
                f'Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")',
                'oDesign.Analyze "Setup_10GHz"',
                "oProject.Save",
                'Set oSolutions = oDesign.GetModule("Solutions")',
                'vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")',
                "variation = CStr(vars(LBound(vars)))",
                f'oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False',
                "oDesktop.CloseProject oProject.GetName()",
                "oDesktop.QuitApplication",
            ]
        )
        + "\n",
        encoding="ascii",
    )


def profile_metrics(folder: Path) -> dict[str, Any]:
    selected: tuple[Path, str, list[float]] | None = None
    for profile in folder.rglob("*.profile"):
        text = profile.read_text(encoding="utf-8", errors="ignore")
        values: list[float] = []
        for line in text.splitlines():
            if "Max Mag. Delta S" not in line:
                continue
            for token in line.split("Delta S", 1)[1].split(","):
                try:
                    values.append(float(token.replace("\\", "").replace("'", "").strip()))
                    break
                except ValueError:
                    continue
        candidate = (profile, text, values)
        if selected is None or len(values) > len(selected[2]):
            selected = candidate
    profile, text, values = selected if selected is not None else (None, "", [])
    return {
        "profile": str(profile) if profile else "",
        "pass_count": len(values),
        "final_delta_s": values[-1] if values else None,
        "converged": "Adaptive Passes converged" in text and "did not converge" not in text,
        "small_mesh_segment_count": sum(item.read_text(encoding="utf-8", errors="ignore").lower().count("small mesh segment") for item in folder.rglob("*.g3derr")),
    }


def parse_s11(path: Path) -> float:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith(("!", "#", "[")):
            values.extend(float(value) for value in line.split())
    return float(-20.0 * np.log10(max(values[1], 1.0e-15)))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    name = "cps_microstrip_balun_1x1"
    folder = args.out_dir / name
    folder.mkdir()
    project = folder / f"{name}.aedt"
    touchstone = folder / f"{name}.s1p"
    builder = folder / f"build_{name}.vbs"
    solver = folder / f"solve_export_{name}.vbs"
    builder.write_text(vbs_builder(project), encoding="ascii")
    write_solve_export(solver, project, touchstone)
    row = {"name": name, **CONFIG, "project_path": str(project), "touchstone_path": str(touchstone), "builder_path": str(builder), "solver_path": str(solver)}
    write_csv(args.out_dir / "candidate_manifest.csv", row)
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "scope": "fixed 1x1 grounded microstrip-to-CPS printed-dipole smoke", "configuration": row, "gate": "converged Delta S <= 0.05; valid S1; input RL >= 10 dB; no port-topology errors", "next_step": "only then instantiate the same topology in 2x2"}
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.out_dir / "candidate_manifest.csv"
    if not manifest.exists():
        prepare(args)
    row = next(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    folder = Path(row["project_path"]).parent
    with (folder / "build.log").open("w", encoding="utf-8", errors="replace") as handle:
        build = subprocess.run([str(args.ansys_exe), "-RunScriptAndExit", row["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    solve_code: int | None = None
    if build.returncode == 0:
        with (folder / "solve_export.log").open("w", encoding="utf-8", errors="replace") as handle:
            solve = subprocess.run([str(args.ansys_exe), "-ng", "-RunScriptAndExit", row["solver_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        solve_code = int(solve.returncode)
    result = {"build_return_code": int(build.returncode), "solve_return_code": solve_code}
    (args.out_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    row = next(csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")))
    folder = Path(row["project_path"]).parent
    touchstone = Path(row["touchstone_path"])
    run_summary = json.loads((args.out_dir / "run_summary.json").read_text(encoding="utf-8")) if (args.out_dir / "run_summary.json").exists() else {}
    metric: dict[str, Any] = {"name": row["name"], **run_summary, **profile_metrics(folder), "touchstone_exists": touchstone.exists()}
    solve_log = folder / f"solve_export_{row['name']}.log"
    log_text = solve_log.read_text(encoding="utf-8", errors="ignore") if solve_log.exists() else ""
    metric["port_topology_warning_count"] = log_text.count("Too many conductors touch lumped port")
    if touchstone.exists() and touchstone.stat().st_size > 100:
        metric["input_rl_db"] = parse_s11(touchstone)
    metric["gate_pass"] = int(metric.get("build_return_code") == 0 and metric.get("solve_return_code") == 0 and metric.get("converged") is True and (metric.get("final_delta_s") or float("inf")) <= 0.05 and metric["touchstone_exists"] and metric["port_topology_warning_count"] == 0 and float(metric.get("input_rl_db", -float("inf"))) >= 10.0)
    write_csv(args.out_dir / "validation_metrics.csv", metric)
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "metric": metric, "decision": "allow_2x2_cps_microstrip_fixture" if metric["gate_pass"] else "block_2x2_and_refine_printed_feed_geometry", "active_2400_started": False}
    (args.out_dir / "stage_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def status(args: argparse.Namespace) -> dict[str, Any]:
    return {"prepared": (args.out_dir / "candidate_manifest.csv").exists(), "run_complete": (args.out_dir / "run_summary.json").exists(), "analyzed": (args.out_dir / "stage_summary.json").exists()}


def main() -> None:
    args = parse_args()
    result = {"prepare": prepare, "run": run, "analyze": analyze, "status": status}[args.mode](args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
