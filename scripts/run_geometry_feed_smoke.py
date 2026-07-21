"""Run 4x4 HFSS geometry/feed smoke candidates before a 16x16 rebuild.

The candidates retain 15 mm pitch. Transverse T-shaped PEC end loading lets
the x-directed arms be shortened to enlarge collinear tip clearance while
recovering electrical length. This is a geometry smoke, not a fabricated
matching-network sign-off.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_full_s256p_active_return import parse_touchstone


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BUILDER = ROOT / "scripts" / "build_ura16_quick.vbs"
DEFAULT_OUT = ROOT / "hfss_outputs" / "geometry_feed_smoke_20260714_run01"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "URA16_Quick_10GHz"
CANDIDATES = (
    {"name": "baseline_v2", "length": 14.2, "radius": 0.25, "hat_length": 0.0, "hat_radius": 0.0},
    {"name": "short_plain", "length": 12.6, "radius": 0.35, "hat_length": 0.0, "hat_radius": 0.0},
    {"name": "retuned_tload_1p5_l11p5", "length": 11.5, "radius": 0.35, "hat_length": 1.5, "hat_radius": 0.25},
    {"name": "retuned_tload_2p0_l11p0", "length": 11.0, "radius": 0.35, "hat_length": 2.0, "hat_radius": 0.25},
    {"name": "retuned_tload_3p0_l10p4", "length": 10.4, "radius": 0.35, "hat_length": 3.0, "hat_radius": 0.25},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="prepare")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--overwrite-control-files", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


def write_candidate_builder(path: Path, project: Path, candidate: dict[str, float | str]) -> None:
    text = SOURCE_BUILDER.read_text(encoding="utf-8-sig")
    text = replace_once(
        text,
        r'projectPath = rootDir & "\models\hfss\ura16_quick_10ghz.aedt"',
        f'projectPath = "{vp(project)}"',
    )
    text = replace_once(text, "dipoleLengthMm = 15.0", f"dipoleLengthMm = {float(candidate['length']):.6f}")
    text = replace_once(text, "radiusMm = 0.25", f"radiusMm = {float(candidate['radius']):.6f}")
    text = replace_once(text, "nx = 16", "nx = 4")
    text = replace_once(text, "ny = 16", "ny = 4")
    hat_length = float(candidate["hat_length"])
    if hat_length > 0.0:
        marker = '        CreateArm oEditor, rightName, xStartRight, yc, 0.0, radiusMm, armLenMm'
        hats = (
            marker
            + f'\n        CreateHat oEditor, "Hat_" & nameBase & "_L", xStartLeft, yc - {hat_length / 2.0:.6f}, 0.0, {float(candidate["hat_radius"]):.6f}, {hat_length:.6f}'
            + f'\n        CreateHat oEditor, "Hat_" & nameBase & "_R", xStartRight + armLenMm, yc - {hat_length / 2.0:.6f}, 0.0, {float(candidate["hat_radius"]):.6f}, {hat_length:.6f}'
            + '\n        oEditor.Unite Array("NAME:Selections", "Selections:=", leftName & ",Hat_" & nameBase & "_L"), Array("NAME:UniteParameters", "KeepOriginals:=", False)'
            + '\n        oEditor.Unite Array("NAME:Selections", "Selections:=", rightName & ",Hat_" & nameBase & "_R"), Array("NAME:UniteParameters", "KeepOriginals:=", False)'
        )
        text = replace_once(text, marker, hats)
        text += '''

Sub CreateHat(editor, objName, xCenter, yStart, zCenter, radius, length)
    editor.CreateCylinder Array( _
            "NAME:CylinderParameters", _
            "XCenter:=", Mm(xCenter), _
            "YCenter:=", Mm(yStart), _
            "ZCenter:=", Mm(zCenter), _
            "Radius:=", Mm(radius), _
            "Height:=", Mm(length), _
            "WhichAxis:=", "Y", _
            "NumSides:=", "12" _
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
'''
    path.write_text(text, encoding="ascii")


def write_solve_export(path: Path, project: Path, touchstone: Path) -> None:
    lines = [
        "Option Explicit",
        "Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, vars, variation",
        "Set oAnsoftApp = CreateObject(\"Ansoft.ElectronicsDesktop\")",
        "Set oDesktop = oAnsoftApp.GetAppDesktop()",
        f'oDesktop.OpenProject "{vp(project)}"',
        f'Set oProject = oDesktop.SetActiveProject("{project.stem}")',
        f'Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")',
        'oDesign.Analyze "Setup_10GHz"',
        "oProject.Save",
        "Set oSolutions = oDesign.GetModule(\"Solutions\")",
        'vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")',
        "variation = CStr(vars(LBound(vars)))",
        f'oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False',
        "oDesktop.CloseProject oProject.GetName()",
        "oDesktop.QuitApplication",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite_control_files:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        name = str(candidate["name"])
        folder = args.out_dir / name
        folder.mkdir(exist_ok=True)
        project = folder / f"{name}.aedt"
        touchstone = folder / f"{name}.s16p"
        builder = folder / f"build_{name}.vbs"
        solver = folder / f"solve_export_{name}.vbs"
        write_candidate_builder(builder, project, candidate)
        write_solve_export(solver, project, touchstone)
        rows.append({**candidate, "project_path": str(project), "touchstone_path": str(touchstone), "builder": str(builder), "solver": str(solver)})
    write_csv(args.out_dir / "candidate_manifest.csv", rows)
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "4x4 HFSS geometry smoke only; no 16x16 full-array solve",
        "frequency_ghz": 10.0,
        "spacing_mm": 15.0,
        "candidate_count": len(rows),
        "promotion_gate": "minimum passive RL >= 10 dB and x-nearest coupling improves >= 3 dB from baseline",
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.out_dir / "candidate_manifest.csv"
    if not manifest.exists():
        prepare(args)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    status: list[dict[str, Any]] = []
    for row in rows:
        folder = Path(row["project_path"]).parent
        build_log, solve_log = folder / "build.log", folder / "solve_export.log"
        with build_log.open("w", encoding="utf-8", errors="replace") as handle:
            build = subprocess.run([str(args.ansys_exe), "-RunScriptAndExit", row["builder"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        code = int(build.returncode)
        if code == 0:
            with solve_log.open("w", encoding="utf-8", errors="replace") as handle:
                solve = subprocess.run([str(args.ansys_exe), "-ng", "-RunScriptAndExit", row["solver"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            code = int(solve.returncode)
        status.append({"name": row["name"], "return_code": code, "build_log": str(build_log), "solve_log": str(solve_log)})
    write_csv(args.out_dir / "run_status.csv", status)
    return {"runs": status, "all_zero_exit": bool(all(row["return_code"] == 0 for row in status))}


def nearest_metrics(matrix: np.ndarray) -> tuple[float, float, float, float]:
    x_values: list[float] = []
    y_values: list[float] = []
    for ix in range(4):
        for iy in range(4):
            index = ix * 4 + iy
            if iy < 3:
                y_values.extend((abs(matrix[index, index + 1]), abs(matrix[index + 1, index])))
            if ix < 3:
                x_values.extend((abs(matrix[index, index + 4]), abs(matrix[index + 4, index])))
    return tuple(float(20.0 * np.log10(value)) for value in (max(x_values), np.median(x_values), max(y_values), np.median(y_values)))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")):
        touchstone = Path(item["touchstone_path"])
        if not touchstone.exists() or touchstone.stat().st_size < 1000:
            rows.append({"name": item["name"], "status": "missing_touchstone"})
            continue
        parsed = parse_touchstone(touchstone)
        matrix = np.asarray(parsed["s_parameters"], dtype=np.complex128)
        if matrix.shape != (1, 16, 16):
            rows.append({"name": item["name"], "status": f"unexpected_shape_{matrix.shape}"})
            continue
        s = matrix[0]
        passive = -20.0 * np.log10(np.maximum(abs(np.diag(s)), 1.0e-15))
        off = abs(s - np.diag(np.diag(s)))
        x_max, x_median, y_max, y_median = nearest_metrics(s)
        rows.append({"name": item["name"], "status": "complete", "passive_rl_min_db": float(np.min(passive)), "passive_rl_median_db": float(np.median(passive)), "passive_rl_10db_port_pass_rate": float(np.mean(passive >= 10.0)), "mutual_worst_db": float(20.0 * np.log10(np.max(off))), "nearest_x_worst_db": x_max, "nearest_x_median_db": x_median, "nearest_y_worst_db": y_max, "nearest_y_median_db": y_median})
    baseline = next((row for row in rows if row.get("name") == "baseline_v2" and row.get("status") == "complete"), None)
    for row in rows:
        if baseline is None or row.get("status") != "complete":
            row["promoted"] = 0
        else:
            improvement = float(baseline["nearest_x_worst_db"]) - float(row["nearest_x_worst_db"])
            row["nearest_x_coupling_improvement_db"] = improvement
            row["promoted"] = int(float(row["passive_rl_min_db"]) >= 10.0 and improvement >= 3.0)
    write_csv(args.out_dir / "smoke_metrics.csv", rows)
    promoted = [row["name"] for row in rows if int(row.get("promoted", 0))]
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "complete_candidate_count": len([row for row in rows if row.get("status") == "complete"]), "promoted_candidates": promoted, "allow_16x16_rebuild": bool(promoted), "decision": "allow_full_array_candidate_rebuild" if promoted else "block_full_array_rebuild_due_to_smoke_gate_failure", "limitations": ["A 4x4 smoke cannot certify 16x16 active impedance.", "T loading is an embedded PEC geometry change but has no feed/balun loss or bandwidth model.", "No PSLL, isolation, or full-array active-RL conclusion is made here."]}
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def status(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    manifest = args.out_dir / "candidate_manifest.csv"
    if manifest.exists():
        for item in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            touchstone = Path(item["touchstone_path"])
            rows.append({"name": item["name"], "touchstone_exists": touchstone.exists(), "size_bytes": touchstone.stat().st_size if touchstone.exists() else 0})
    return {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "candidates": rows}


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "run":
        result = run(args)
    elif args.mode == "analyze":
        result = analyze(args)
    else:
        result = status(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
