"""Build staged 10 GHz grounded rectangular-patch URA models for training.

The shared-ground, direct discrete-probe topology keeps exactly one 50-ohm
port per element, so existing 16x16 masks, complex weights, EEP operators and
full-wave dataset schemas remain compatible.  Use 1x1 then 4x4 before 16x16.
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
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DEFAULT_OUT = ROOT / "hfss_outputs" / "grounded_patch_rebuild_20260717_run01"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
CONFIG = {
    "frequency_ghz": 10.0,
    "spacing_x_mm": 15.0,
    "spacing_y_mm": 15.0,
    "er": 2.2,
    "tan_delta": 0.0009,
    "substrate_thickness_mm": 0.787,
    "patch_width_x_mm": 11.8,
    "patch_length_y_mm": 9.35,
    "feed_offset_y_mm": -3.5,
    "copper_thickness_mm": 0.035,
    "probe_radius_mm": 0.25,
    "coax_inner_radius_mm": 0.60,
    "coax_outer_radius_mm": 1.10,
    "coax_drop_mm": 0.80,
    "port_width_mm": 0.30,
    "series_match_inductance_nh": 0.533,
    "series_match_q": 50.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("prepare", "build", "solve", "run", "analyze", "status"),
        default="prepare",
    )
    parser.add_argument("--side", type=int, choices=(1, 4, 8, 16), default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--feed-offset-y-mm", type=float)
    parser.add_argument("--patch-length-y-mm", type=float)
    parser.add_argument("--patch-width-x-mm", type=float)
    parser.add_argument("--feed-model", choices=("coax", "direct"), default="coax")
    return parser.parse_args()


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(CONFIG)
    config["feed_model"] = args.feed_model
    overrides = {
        "feed_offset_y_mm": args.feed_offset_y_mm,
        "patch_length_y_mm": args.patch_length_y_mm,
        "patch_width_x_mm": args.patch_width_x_mm,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = float(value)
    return config


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def write_csv(path: Path, row: dict[str, Any] | list[dict[str, Any]]) -> None:
    rows = row if isinstance(row, list) else [row]
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def builder_text(project: Path, side: int, c: dict[str, Any]) -> str:
    h = c["substrate_thickness_mm"]
    copper = c["copper_thickness_mm"]
    probe = c["probe_radius_mm"]
    coax_inner = c["coax_inner_radius_mm"]
    coax_outer = c["coax_outer_radius_mm"]
    coax_drop = c["coax_drop_mm"]
    coax_bottom = -h - copper - coax_drop
    board_x = max(15.0, (side - 1) * c["spacing_x_mm"] + 15.0)
    board_y = max(15.0, (side - 1) * c["spacing_y_mm"] + 15.0)
    sweep_setup = ''
    if side == 1:
        sweep_setup = 'oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_8_12GHz", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "8GHz", "RangeEnd:=", "12GHz", "RangeCount:=", 81, "Type:=", "Interpolating", "SaveFields:=", False, "SaveRadFields:=", False, "InterpTolerance:=", 0.5, "InterpMaxSolns:=", 250, "InterpMinSolns:=", 0, "InterpMinSubranges:=", 1, "InterpUseS:=", True, "InterpUsePortImped:=", True, "InterpUsePropConst:=", True, "UseDerivativeConvergence:=", False, "InterpDerivTolerance:=", 0.2, "UseFullBasis:=", True, "EnforcePassivity:=", True, "PassivityErrorTolerance:=", 0.0001, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")'
    if c.get("feed_model") == "direct":
        feed_geometry = f'''        CreateBox oEditor, "Patch_" & nameBase, xc - {c['patch_width_x_mm']/2:.6f}, yc - {c['patch_length_y_mm']/2:.6f}, 0, {c['patch_width_x_mm']:.6f}, {c['patch_length_y_mm']:.6f}, {copper:.6f}, "pec", False
        CreateSheetX oEditor, "PortSheet_" & nameBase, xc, feedY - {c['port_width_mm']/2:.6f}, {-h:.6f}, {c['port_width_mm']:.6f}, {h:.6f}
        AssignPort oBoundary, "P" & nameBase, "PortSheet_" & nameBase, xc, feedY, {-h+0.01:.6f}, xc, feedY, -0.01'''
    else:
        feed_geometry = f'''        CreateCylinderZ oEditor, "GroundHole_" & nameBase, xc, feedY, {-h-copper-0.05:.6f}, {coax_inner:.6f}, {copper+0.10:.6f}, "vacuum", True
        SubtractObject oEditor, "Ground", "GroundHole_" & nameBase
        CreateCylinderZ oEditor, "SubstrateHole_" & nameBase, xc, feedY, {-h-0.01:.6f}, {probe+0.01:.6f}, {h+0.02:.6f}, "vacuum", True
        SubtractObject oEditor, "Substrate", "SubstrateHole_" & nameBase
        CreateBox oEditor, "Patch_" & nameBase, xc - {c['patch_width_x_mm']/2:.6f}, yc - {c['patch_length_y_mm']/2:.6f}, 0, {c['patch_width_x_mm']:.6f}, {c['patch_length_y_mm']:.6f}, {copper:.6f}, "pec", False
        CreateCylinderZ oEditor, "Probe_" & nameBase, xc, feedY, {coax_bottom:.6f}, {probe:.6f}, {h + 2*copper + coax_drop:.6f}, "pec", False
        UniteObjects oEditor, "Patch_" & nameBase, "Probe_" & nameBase
        CreateCylinderZ oEditor, "Outer_" & nameBase, xc, feedY, {coax_bottom:.6f}, {coax_outer:.6f}, {coax_drop+copper/2:.6f}, "pec", False
        CreateCylinderZ oEditor, "OuterCut_" & nameBase, xc, feedY, {coax_bottom-0.02:.6f}, {coax_inner:.6f}, {coax_drop+copper+0.04:.6f}, "vacuum", True
        SubtractObject oEditor, "Outer_" & nameBase, "OuterCut_" & nameBase
        UniteObjects oEditor, "Ground", "Outer_" & nameBase
        CreateSheetZ oEditor, "PortSheet_" & nameBase, xc + {probe-0.02:.6f}, feedY - {c['port_width_mm']/2:.6f}, {coax_bottom:.6f}, {coax_inner-probe+0.04:.6f}, {c['port_width_mm']:.6f}, "vacuum"
        AssignPort oBoundary, "P" & nameBase, "PortSheet_" & nameBase, xc + {probe+0.01:.6f}, feedY, {coax_bottom:.6f}, xc + {coax_inner-0.01:.6f}, feedY, {coax_bottom:.6f}'''
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDefinitionManager, oDesign, oEditor, oBoundary, oAnalysis, oRad
Dim nx, ny, ix, iy, idx, xc, yc, feedY, nameBase
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
nx = {side}
ny = {side}

CreateBox oEditor, "Substrate", {-board_x/2:.6f}, {-board_y/2:.6f}, {-h:.6f}, {board_x:.6f}, {board_y:.6f}, {h:.6f}, "RO5880_Custom", True
CreateBox oEditor, "Ground", {-board_x/2:.6f}, {-board_y/2:.6f}, {-h-copper:.6f}, {board_x:.6f}, {board_y:.6f}, {copper:.6f}, "pec", False
idx = 0
For ix = 0 To nx - 1
    For iy = 0 To ny - 1
        xc = (ix - 0.5 * (nx - 1)) * {c['spacing_x_mm']:.6f}
        yc = (iy - 0.5 * (ny - 1)) * {c['spacing_y_mm']:.6f}
        feedY = yc + {c['feed_offset_y_mm']:.6f}
        nameBase = Pad3(idx)
{feed_geometry}
        idx = idx + 1
    Next
Next

CreateBox oEditor, "AirRegion", {-board_x/2-15:.6f}, {-board_y/2-15:.6f}, -15, {board_x+30:.6f}, {board_y+30:.6f}, 30, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", 20, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", 15, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
{sweep_setup}
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_Theta0_90_Phi0_360", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "90deg", "ThetaStep:=", "1deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "2deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Function Mm(value)
    Mm = CStr(Round(CDbl(value), 6)) & "mm"
End Function
Function Pad3(value)
    Pad3 = Right("000" & CStr(value), 3)
End Function
Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(120 150 170)", "Transparency:=", 0.65, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height, material)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", True)
End Sub
Sub CreateSheetX(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "X"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateCylinderZ(editor, objName, x, y, z, radius, height, material, solveInside)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", "20"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub SubtractObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", False)
End Sub
Sub UniteObjects(editor, firstName, secondName)
    editor.Unite Array("NAME:Selections", "Selections:=", firstName & "," & secondName), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub AssignPort(boundary, portName, sheetName, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
'''


def solve_export_text(project: Path, touchstone: Path, sweep_touchstone: Path | None) -> str:
    sweep_export = ""
    if sweep_touchstone is not None:
        sweep_export = f'oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_8_12GHz"), 3, "{vp(sweep_touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False'
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
{sweep_export}
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def profile_metrics(folder: Path) -> dict[str, Any]:
    selected: tuple[Path, str, list[float]] | None = None
    for profile in folder.rglob("*.profile"):
        text = profile.read_text(encoding="utf-8", errors="ignore")
        values: list[float] = []
        for line in text.splitlines():
            if "Max Mag. Delta S" in line:
                for token in line.split("Delta S", 1)[1].split(","):
                    try:
                        values.append(float(token.replace("\\", "").replace("'", "").strip()))
                        break
                    except ValueError:
                        continue
        if selected is None or len(values) > len(selected[2]):
            selected = (profile, text, values)
    profile, text, values = selected if selected else (None, "", [])
    return {"profile": str(profile) if profile else "", "pass_count": len(values), "final_delta_s": values[-1] if values else None, "converged": "Adaptive Passes converged" in text and "did not converge" not in text, "small_mesh_segment_count": sum(item.read_text(encoding="utf-8", errors="ignore").lower().count("small mesh segment") for item in folder.rglob("*.g3derr"))}


def parse_touchstone(path: Path, nports: int) -> np.ndarray:
    tokens: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith(("!", "#", "[")):
            tokens.extend(float(value) for value in line.split())
    values = np.asarray(tokens[1:], dtype=float).reshape(nports * nports, 2)
    return (values[:, 0] * np.exp(1j * np.deg2rad(values[:, 1]))).reshape(nports, nports)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    config = effective_config(args)
    name = f"grounded_patch_{args.side}x{args.side}"
    folder = args.out_dir / name
    folder.mkdir()
    nports = args.side * args.side
    project = folder / f"{name}.aedt"
    touchstone = folder / f"{name}.s{nports}p"
    sweep_touchstone = folder / f"{name}_sweep.s1p" if args.side == 1 else None
    builder = folder / f"build_{name}.vbs"
    solver = folder / f"solve_export_{name}.vbs"
    builder.write_text(builder_text(project, args.side, config), encoding="ascii")
    solver.write_text(solve_export_text(project, touchstone, sweep_touchstone), encoding="ascii")
    row = {"name": name, "side": args.side, "port_count": nports, **config, "project_path": str(project), "touchstone_path": str(touchstone), "sweep_touchstone_path": str(sweep_touchstone) if sweep_touchstone else "", "builder_path": str(builder), "solver_path": str(solver)}
    write_csv(args.out_dir / "candidate_manifest.csv", row)
    summary = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "scope": f"grounded rectangular patch {args.side}x{args.side} gate", "configuration": row, "gate": "converged Delta S <= 0.05; valid Sn; finite-Q matched passive RL >= 10 dB; no port-topology warning", "compatibility": "one port per element; 16x16 mask/weight/EEP/training schemas unchanged"}
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.out_dir / "candidate_manifest.csv").exists():
        prepare(args)
    row = next(csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")))
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


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.out_dir / "candidate_manifest.csv").exists():
        prepare(args)
    row = next(csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")))
    folder = Path(row["project_path"]).parent
    with (folder / "build.log").open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.run(
            [str(args.ansys_exe), "-RunScriptAndExit", row["builder_path"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {"build_return_code": int(process.returncode), "solve_return_code": None}
    (args.out_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def solve(args: argparse.Namespace) -> dict[str, Any]:
    row = next(csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")))
    project = Path(row["project_path"])
    if not project.exists():
        raise FileNotFoundError(f"Build the AEDT project before solving: {project}")
    folder = project.parent
    with (folder / "solve_export.log").open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.run(
            [str(args.ansys_exe), "-ng", "-RunScriptAndExit", row["solver_path"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    prior = (
        json.loads((args.out_dir / "run_summary.json").read_text(encoding="utf-8"))
        if (args.out_dir / "run_summary.json").exists()
        else {}
    )
    result = {
        "build_return_code": prior.get("build_return_code", 0),
        "solve_return_code": int(process.returncode),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    row = next(csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")))
    folder = Path(row["project_path"]).parent
    touchstone = Path(row["touchstone_path"])
    run_summary = json.loads((args.out_dir / "run_summary.json").read_text(encoding="utf-8")) if (args.out_dir / "run_summary.json").exists() else {}
    metric: dict[str, Any] = {"name": row["name"], "side": int(row["side"]), **run_summary, **profile_metrics(folder), "touchstone_exists": touchstone.exists()}
    log_path = folder / f"solve_export_{row['name']}.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    topology_patterns = (
        "Too many conductors touch lumped port",
        "'0' conductors touch lumped port",
        "'1' conductors touch lumped port",
        "typically a lumped port contains 2 conductors",
        "Both endpoints of port lines must lie on port",
    )
    metric["port_topology_warning_count"] = sum(log_text.count(pattern) for pattern in topology_patterns)
    if touchstone.exists() and touchstone.stat().st_size > 100:
        s = parse_touchstone(touchstone, int(row["port_count"]))
        rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s)), 1.0e-15))
        metric["passive_rl_min_db"] = float(np.min(rl))
        metric["passive_rl_median_db"] = float(np.median(rl))
        metric["passive_rl_10db_port_pass_rate"] = float(np.mean(rl >= 10.0))
        nports = int(row["port_count"])
        ident = np.eye(nports, dtype=np.complex128)
        z0 = 50.0
        z_ant = z0 * (ident + s) @ np.linalg.inv(ident - s)
        frequency_hz = float(row.get("frequency_ghz", CONFIG["frequency_ghz"])) * 1.0e9
        inductance_h = float(row.get("series_match_inductance_nh", CONFIG["series_match_inductance_nh"])) * 1.0e-9
        match_q = float(row.get("series_match_q", CONFIG["series_match_q"]))
        omega_l = 2.0 * np.pi * frequency_hz * inductance_h
        series_impedance = omega_l / match_q + 1j * omega_l
        z_matched = z_ant + series_impedance * ident
        s_matched = (z_matched - z0 * ident) @ np.linalg.inv(z_matched + z0 * ident)
        matched_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s_matched)), 1.0e-15))
        metric["series_match_inductance_nh"] = inductance_h * 1.0e9
        metric["series_match_q"] = match_q
        metric["series_match_resistance_ohm"] = float(omega_l / match_q)
        metric["matched_passive_rl_min_db"] = float(np.min(matched_rl))
        metric["matched_passive_rl_median_db"] = float(np.median(matched_rl))
        metric["matched_passive_rl_10db_port_pass_rate"] = float(np.mean(matched_rl >= 10.0))
        np.savez_compressed(folder / "matched_s_10ghz.npz", s_raw=s, s_matched=s_matched, z_ant=z_ant, z_matched=z_matched, frequency_hz=frequency_hz, series_inductance_h=inductance_h, series_q=match_q)
        if int(row["side"]) >= 4:
            side = int(row["side"])
            port_rows: list[dict[str, Any]] = []
            for port in range(nports):
                ix, iy = divmod(port, side)
                x_edge = ix in (0, side - 1)
                y_edge = iy in (0, side - 1)
                port_class = "corner" if x_edge and y_edge else ("edge" if x_edge or y_edge else "interior")
                neighbor_indices = []
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < side and 0 <= ny < side:
                        neighbor_indices.append(nx * side + ny)
                nearest = max((abs(s[port, other]) for other in neighbor_indices), default=0.0)
                mutual = max((abs(s[port, other]) for other in range(nports) if other != port), default=0.0)
                port_rows.append({
                    "port_index": port,
                    "ix": ix,
                    "iy": iy,
                    "port_class": port_class,
                    "z_self_real_ohm": float(np.real(z_ant[port, port])),
                    "z_self_imag_ohm": float(np.imag(z_ant[port, port])),
                    "raw_passive_rl_db": float(rl[port]),
                    "matched_passive_rl_db": float(matched_rl[port]),
                    "matched_rl_10db_pass": int(matched_rl[port] >= 10.0),
                    "nearest_coupling_db": float(20.0 * np.log10(max(nearest, 1.0e-15))),
                    "worst_mutual_coupling_db": float(20.0 * np.log10(max(mutual, 1.0e-15))),
                })
            write_csv(folder / "port_class_metrics.csv", port_rows)
            class_rows: list[dict[str, Any]] = []
            for port_class in ("corner", "edge", "interior"):
                members = [item for item in port_rows if item["port_class"] == port_class]
                class_rows.append({
                    "port_class": port_class,
                    "port_count": len(members),
                    "raw_rl_min_db": min(float(item["raw_passive_rl_db"]) for item in members),
                    "raw_rl_median_db": float(np.median([float(item["raw_passive_rl_db"]) for item in members])),
                    "matched_rl_min_db": min(float(item["matched_passive_rl_db"]) for item in members),
                    "matched_rl_median_db": float(np.median([float(item["matched_passive_rl_db"]) for item in members])),
                    "matched_rl_10db_pass_rate": float(np.mean([int(item["matched_rl_10db_pass"]) for item in members])),
                    "nearest_coupling_worst_db": max(float(item["nearest_coupling_db"]) for item in members),
                })
            write_csv(folder / "port_class_summary.csv", class_rows)
    metric["gate_pass"] = int(metric.get("build_return_code") == 0 and metric.get("solve_return_code") == 0 and metric.get("converged") is True and (metric.get("final_delta_s") or float("inf")) <= 0.05 and metric["touchstone_exists"] and metric["port_topology_warning_count"] == 0 and float(metric.get("matched_passive_rl_min_db", -float("inf"))) >= 10.0)
    write_csv(args.out_dir / "validation_metrics.csv", metric)
    if metric["gate_pass"]:
        next_stage = {
            1: "allow_4x4_grounded_patch",
            4: "allow_8x8_grounded_patch",
            8: "allow_16x16_grounded_patch",
            16: "allow_256port_eep_export",
        }[int(args.side)]
    else:
        next_stage = "block_next_stage"
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "metric": metric, "decision": next_stage, "training_started": False}
    (args.out_dir / "stage_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def status(args: argparse.Namespace) -> dict[str, Any]:
    return {"prepared": (args.out_dir / "candidate_manifest.csv").exists(), "run_complete": (args.out_dir / "run_summary.json").exists(), "analyzed": (args.out_dir / "stage_summary.json").exists()}


def main() -> None:
    args = parse_args()
    result = {
        "prepare": prepare,
        "build": build,
        "solve": solve,
        "run": run,
        "analyze": analyze,
        "status": status,
    }[args.mode](args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
