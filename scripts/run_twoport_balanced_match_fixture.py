"""HFSS 1x1/2x2 fixture for a physical balanced two-port feed/match section.

Each dipole pole is a single PEC BRep made from its radiating arm, a short
radiator-side twin-lead, and a fixed 7.5 mm stepped-impedance transformer.
Two differential lumped ports are assigned per element: input and antenna
plane.  The 1x1 fixture is the mandatory smoke before a matching 2x2 build.
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
SOURCE_BUILDER = ROOT / "scripts" / "build_ura16_quick.vbs"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DEFAULT_OUT = ROOT / "hfss_outputs" / "twoport_balanced_match_fixture_20260716_run01"
DESIGN_NAME = "URA16_Quick_10GHz"

# Fixed physical topology: no bar/cap change and no matching sweep.
CONFIG: dict[str, float] = {
    "dipole_length_mm": 12.6,
    "arm_radius_mm": 0.35,
    "gap_mm": 0.5,
    "radiator_lead_mm": 1.5,
    "transformer_length_mm": 7.5,
    "radiator_lead_radius_mm": 0.18,
    "transformer_radius_mm": 0.10,
    "spacing_mm": 15.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="prepare")
    parser.add_argument("--side", type=int, choices=(1, 2), default=1)
    parser.add_argument("--topology", choices=("radiator", "network_only"), default="radiator")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def strict_setup(text: str) -> str:
    for old, new in (
        ('"MaximumPasses:=", 4', '"MaximumPasses:=", 8'),
        ('"MinimumPasses:=", 1', '"MinimumPasses:=", 2'),
        ('"MinimumConvergedPasses:=", 1', '"MinimumConvergedPasses:=", 2'),
        ('"PercentRefinement:=", 20', '"PercentRefinement:=", 15'),
    ):
        text = replace_once(text, old, new)
    return text


def helpers() -> str:
    return r'''

Sub CreateZRod(editor, objName, xCenter, yCenter, zStart, radius, length)
    editor.CreateCylinder Array( _
            "NAME:CylinderParameters", _
            "XCenter:=", Mm(xCenter), _
            "YCenter:=", Mm(yCenter), _
            "ZCenter:=", Mm(zStart), _
            "Radius:=", Mm(radius), _
            "Height:=", Mm(length), _
            "WhichAxis:=", "Z", _
            "NumSides:=", "16" _
        ), _
        Array( _
            "NAME:Attributes", _
            "Name:=", objName, _
            "Flags:=", "", _
            "Color:=", "(230 160 60)", _
            "Transparency:=", 0, _
            "PartCoordinateSystem:=", "Global", _
            "MaterialValue:=", """pec""", _
            "SolveInside:=", False _
        )
End Sub

Sub CreateMatchedPole(editor, objName, feedName, matchName, armStart, feedX, yCenter, armRadius, armLength, leadLength, leadRadius, transformerLength, transformerRadius)
    CreateArm editor, objName, armStart, yCenter, 0.0, armRadius, armLength
    ' The upper lead overlaps the arm and the lower transformer on purpose.
    CreateZRod editor, feedName, feedX, yCenter, -leadLength, leadRadius, leadLength + armRadius
    CreateZRod editor, matchName, feedX, yCenter, -leadLength - transformerLength, transformerRadius, transformerLength + 0.20
    editor.Unite Array("NAME:Selections", "Selections:=", objName & "," & feedName & "," & matchName), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
'''


def network_only_helpers() -> str:
    return r'''

Sub CreateZPortSheet(editor, objName, xStart, yStart, zStart, width, height)
    editor.CreateRectangle Array( _
            "NAME:RectangleParameters", _
            "IsCovered:=", True, _
            "XStart:=", Mm(xStart), _
            "YStart:=", Mm(yStart), _
            "ZStart:=", Mm(zStart), _
            "Width:=", Mm(width), _
            "Height:=", Mm(height), _
            "WhichAxis:=", "Z" _
        ), _
        Array( _
            "NAME:Attributes", _
            "Name:=", objName, _
            "Flags:=", "", _
            "Color:=", "(128 128 128)", _
            "Transparency:=", 0, _
            "PartCoordinateSystem:=", "Global", _
            "MaterialValue:=", """vacuum""", _
            "SolveInside:=", True _
        )
End Sub

Sub CreateNetworkPole(editor, objName, leadName, matchName, xCenter, yCenter, leadLength, leadRadius, transformerLength, transformerRadius)
    CreateZRod editor, leadName, xCenter, yCenter, -leadLength, leadRadius, leadLength
    CreateZRod editor, matchName, xCenter, yCenter, -leadLength - transformerLength, transformerRadius, transformerLength + 0.20
    editor.Unite Array("NAME:Selections", "Selections:=", leadName & "," & matchName), Array("NAME:UniteParameters", "KeepOriginals:=", False)
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


def write_builder(path: Path, project: Path, side: int, topology: str) -> None:
    c = CONFIG
    text = SOURCE_BUILDER.read_text(encoding="utf-8-sig")
    text = replace_once(text, r'projectPath = rootDir & "\models\hfss\ura16_quick_10ghz.aedt"', f'projectPath = "{vp(project)}"')
    text = replace_once(text, "dipoleLengthMm = 15.0", f"dipoleLengthMm = {c['dipole_length_mm']:.6f}")
    text = replace_once(text, "radiusMm = 0.25", f"radiusMm = {c['arm_radius_mm']:.6f}")
    text = replace_once(text, "gapMm = 0.5", f"gapMm = {c['gap_mm']:.6f}")
    text = replace_once(text, "nx = 16", f"nx = {side}")
    text = replace_once(text, "ny = 16", f"ny = {side}")
    text = strict_setup(text)
    lead = c["radiator_lead_mm"]
    transformer = c["transformer_length_mm"]
    old = '''        CreateArm oEditor, leftName, xStartLeft, yc, 0.0, radiusMm, armLenMm
        CreateArm oEditor, rightName, xStartRight, yc, 0.0, radiusMm, armLenMm
        CreatePortSheet oEditor, sheetName, xc - gapMm / 2.0, yc, -portHeightMm / 2.0, gapMm, portHeightMm
        AssignLumpedPort oBoundary, portName, sheetName, xc - gapMm / 2.0, yc, 0.0, xc + gapMm / 2.0, yc, 0.0'''
    if topology == "radiator":
        new = f'''        CreateMatchedPole oEditor, leftName, "Lead_" & nameBase & "_L", "Match_" & nameBase & "_L", xStartLeft, xStartLeft + armLenMm, yc, radiusMm, armLenMm, {lead:.6f}, {c['radiator_lead_radius_mm']:.6f}, {transformer:.6f}, {c['transformer_radius_mm']:.6f}
        CreateMatchedPole oEditor, rightName, "Lead_" & nameBase & "_R", "Match_" & nameBase & "_R", xStartRight, xStartRight, yc, radiusMm, armLenMm, {lead:.6f}, {c['radiator_lead_radius_mm']:.6f}, {transformer:.6f}, {c['transformer_radius_mm']:.6f}
        CreatePortSheet oEditor, sheetName & "_OUT", xc - gapMm / 2.0, yc, -{lead + 0.5:.6f}, gapMm, portHeightMm
        AssignLumpedPort oBoundary, portName & "_OUT", sheetName & "_OUT", xc - gapMm / 2.0, yc, -{lead:.6f}, xc + gapMm / 2.0, yc, -{lead:.6f}
        CreatePortSheet oEditor, sheetName & "_IN", xc - gapMm / 2.0, yc, -{lead + transformer + 0.5:.6f}, gapMm, portHeightMm
        AssignLumpedPort oBoundary, portName & "_IN", sheetName & "_IN", xc - gapMm / 2.0, yc, -{lead + transformer:.6f}, xc + gapMm / 2.0, yc, -{lead + transformer:.6f}'''
        text += helpers()
    else:
        new = f'''        CreateNetworkPole oEditor, leftName, "Lead_" & nameBase & "_L", "Match_" & nameBase & "_L", xc - gapMm / 2.0, yc, {lead:.6f}, {c['radiator_lead_radius_mm']:.6f}, {transformer:.6f}, {c['transformer_radius_mm']:.6f}
        CreateNetworkPole oEditor, rightName, "Lead_" & nameBase & "_R", "Match_" & nameBase & "_R", xc + gapMm / 2.0, yc, {lead:.6f}, {c['radiator_lead_radius_mm']:.6f}, {transformer:.6f}, {c['transformer_radius_mm']:.6f}
        CreateZPortSheet oEditor, sheetName & "_OUT", xc - gapMm / 2.0, yc - portHeightMm / 2.0, 0.0, gapMm, portHeightMm
        AssignLumpedPort oBoundary, portName & "_OUT", sheetName & "_OUT", xc - gapMm / 2.0, yc, 0.0, xc + gapMm / 2.0, yc, 0.0
        CreateZPortSheet oEditor, sheetName & "_IN", xc - gapMm / 2.0, yc - portHeightMm / 2.0, -{lead + transformer:.6f}, gapMm, portHeightMm
        AssignLumpedPort oBoundary, portName & "_IN", sheetName & "_IN", xc - gapMm / 2.0, yc, -{lead + transformer:.6f}, xc + gapMm / 2.0, yc, -{lead + transformer:.6f}'''
        text += helpers() + network_only_helpers()
    text = replace_once(text, old, new)
    path.write_text(text, encoding="ascii")


def parse_touchstone(path: Path) -> np.ndarray:
    tokens: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith(("!", "#", "[")):
            tokens.extend(float(value) for value in line.split())
    nports = int(re.search(r"\.s(\d+)p$", path.name, flags=re.I).group(1))
    values = np.asarray(tokens[1:], dtype=float).reshape(nports * nports, 2)
    return (values[:, 0] * np.exp(1j * np.deg2rad(values[:, 1]))).reshape(nports, nports)


def profile_metrics(folder: Path) -> dict[str, Any]:
    selected: tuple[Path, str, list[float]] | None = None
    for profile in folder.rglob("*.profile"):
        text = profile.read_text(encoding="utf-8", errors="ignore")
        deltas: list[float] = []
        for line in text.splitlines():
            if "Max Mag. Delta S" not in line:
                continue
            for token in line.split("Delta S", 1)[1].split(","):
                try:
                    deltas.append(float(token.replace("\\", "").replace("'", "").strip()))
                    break
                except ValueError:
                    continue
        candidate = (profile, text, deltas)
        if selected is None or len(deltas) > len(selected[2]):
            selected = candidate
    profile, text, deltas = selected if selected is not None else (None, "", [])
    warning_count = sum(item.read_text(encoding="utf-8", errors="ignore").lower().count("small mesh segment") for item in folder.rglob("*.g3derr"))
    return {
        "profile": str(profile) if profile else "",
        "pass_count": len(deltas),
        "final_delta_s": deltas[-1] if deltas else None,
        "converged": "Adaptive Passes converged" in text and "did not converge" not in text,
        "small_mesh_segment_count": warning_count,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    name = f"twoport_balanced_{args.topology}_{args.side}x{args.side}"
    folder = args.out_dir / name
    folder.mkdir()
    nports = 2 * args.side * args.side
    project = folder / f"{name}.aedt"
    touchstone = folder / f"{name}.s{nports}p"
    builder = folder / f"build_{name}.vbs"
    solver = folder / f"solve_export_{name}.vbs"
    write_builder(builder, project, args.side, args.topology)
    write_solve_export(solver, project, touchstone)
    row = {"name": name, "side": args.side, "topology": args.topology, "port_count": nports, **CONFIG, "project_path": str(project), "touchstone_path": str(touchstone), "builder_path": str(builder), "solver_path": str(solver)}
    write_csv(args.out_dir / "candidate_manifest.csv", [row])
    scope = "physical two-differential-port network fixture with terminal ports" if args.topology == "network_only" else "physical two-differential-port balanced feed/match fixture with radiator"
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "scope": scope, "configuration": row, "gate": "converged Delta S <= 0.05; valid touchstone; zero small-mesh and port-topology warnings", "next_step": "allow 2x2 only after 1x1 gate pass"}
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
    metric: dict[str, Any] = {"name": row["name"], "side": int(row["side"]), **run_summary, **profile_metrics(folder), "touchstone_exists": touchstone.exists()}
    solve_log = folder / f"solve_export_{row['name']}.log"
    metric["port_topology_warning_count"] = (
        solve_log.read_text(encoding="utf-8", errors="ignore").count("Too many conductors touch lumped port")
        if solve_log.exists()
        else 0
    )
    if touchstone.exists() and touchstone.stat().st_size > 200:
        s = parse_touchstone(touchstone)
        rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s)), 1.0e-15))
        metric["port_rl_min_db"] = float(np.min(rl))
        metric["port_rl_median_db"] = float(np.median(rl))
        if s.shape == (2, 2):
            metric["input_rl_db"] = float(rl[0])
            metric["antenna_plane_rl_db"] = float(rl[1])
            metric["through_s21_db"] = float(20.0 * np.log10(max(abs(s[1, 0]), 1.0e-15)))
    metric["gate_pass"] = int(metric.get("build_return_code") == 0 and metric.get("solve_return_code") == 0 and metric.get("converged") is True and (metric.get("final_delta_s") or float("inf")) <= 0.05 and metric["touchstone_exists"] and metric.get("small_mesh_segment_count") == 0 and metric["port_topology_warning_count"] == 0)
    write_csv(args.out_dir / "validation_metrics.csv", [metric])
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "metric": metric, "decision": "allow_2x2_fixture" if args.side == 1 and metric["gate_pass"] else "block_next_fixture_stage", "active_2400_started": False}
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
