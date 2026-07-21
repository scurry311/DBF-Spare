"""Validate a fixed 4x4 dipole array with an explicit balanced twin-lead feed.

This replaces the fragile end-load variants with one stable radiating arm per
polarity.  Each arm and its vertical feed wire are united into a single PEC
body; the two pole bodies intentionally remain separate for differential
excitation.  It is a passive S16 smoke only, not a matching-network sign-off.
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
DEFAULT_OUT = ROOT / "hfss_outputs" / "balanced_feed_smoke_20260716_run01"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "URA16_Quick_10GHz"

# Fixed from the previously converged short-plain 4x4 reference.  These are
# deliberately not a new bar/cap sweep.
CONFIG: dict[str, float | int | str] = {
    "name": "short_plain_l12p6_balanced_twinlead_4x4",
    "dipole_length_mm": 12.6,
    "arm_radius_mm": 0.35,
    "gap_mm": 0.5,
    "feed_drop_mm": 3.0,
    # Keep 0.14 mm physical clearance between the two 0.5 mm-spaced poles.
    "feed_radius_mm": 0.18,
    "spacing_x_mm": 15.0,
    "spacing_y_mm": 15.0,
    "side": 4,
}


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


def write_solve_export(path: Path, project: Path, touchstone: Path) -> None:
    lines = [
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
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_touchstone(path: Path) -> dict[str, np.ndarray]:
    """Parse the single-frequency HFSS Touchstone v1 MA export used here."""
    tokens: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("!", "#", "[")):
            continue
        tokens.extend(float(item) for item in stripped.split())
    nports = int(re.search(r"\.s(\d+)p$", path.name, flags=re.I).group(1))
    expected = 1 + 2 * nports * nports
    if len(tokens) != expected:
        raise ValueError(f"Expected {expected} Touchstone values, found {len(tokens)}")
    values = np.asarray(tokens[1:], dtype=float).reshape(nports * nports, 2)
    magnitude = values[:, 0]
    phase_rad = np.deg2rad(values[:, 1])
    matrix = (magnitude * np.exp(1j * phase_rad)).reshape(nports, nports)
    return {"s_parameters": matrix[None, :, :]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="prepare")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


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


def balanced_feed_helpers() -> str:
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

Sub CreateBalancedPole(editor, objName, feedName, armStart, feedX, yCenter, armRadius, armLength, feedDrop, feedRadius)
    CreateArm editor, objName, armStart, yCenter, 0.0, armRadius, armLength
    ' The feed wire intentionally penetrates the arm by one arm radius.
    CreateZRod editor, feedName, feedX, yCenter, -feedDrop, feedRadius, feedDrop + armRadius
    editor.Unite Array("NAME:Selections", "Selections:=", objName & "," & feedName), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
'''


def write_builder(path: Path, project: Path) -> None:
    c = CONFIG
    text = SOURCE_BUILDER.read_text(encoding="utf-8-sig")
    text = replace_once(
        text,
        r'projectPath = rootDir & "\models\hfss\ura16_quick_10ghz.aedt"',
        f'projectPath = "{vp(project)}"',
    )
    text = replace_once(text, "dipoleLengthMm = 15.0", f"dipoleLengthMm = {float(c['dipole_length_mm']):.6f}")
    text = replace_once(text, "radiusMm = 0.25", f"radiusMm = {float(c['arm_radius_mm']):.6f}")
    text = replace_once(text, "gapMm = 0.5", f"gapMm = {float(c['gap_mm']):.6f}")
    text = replace_once(text, "nx = 16", f"nx = {int(c['side'])}")
    text = replace_once(text, "ny = 16", f"ny = {int(c['side'])}")
    text = strict_setup(text)

    feed_drop = float(c["feed_drop_mm"])
    feed_radius = float(c["feed_radius_mm"])
    gap = float(c["gap_mm"])
    old = '''        CreateArm oEditor, leftName, xStartLeft, yc, 0.0, radiusMm, armLenMm
        CreateArm oEditor, rightName, xStartRight, yc, 0.0, radiusMm, armLenMm
        CreatePortSheet oEditor, sheetName, xc - gapMm / 2.0, yc, -portHeightMm / 2.0, gapMm, portHeightMm
        AssignLumpedPort oBoundary, portName, sheetName, xc - gapMm / 2.0, yc, 0.0, xc + gapMm / 2.0, yc, 0.0'''
    new = f'''        CreateBalancedPole oEditor, leftName, "Feed_" & nameBase & "_L", xStartLeft, xStartLeft + armLenMm, yc, radiusMm, armLenMm, {feed_drop:.6f}, {feed_radius:.6f}
        CreateBalancedPole oEditor, rightName, "Feed_" & nameBase & "_R", xStartRight, xStartRight, yc, radiusMm, armLenMm, {feed_drop:.6f}, {feed_radius:.6f}
        CreatePortSheet oEditor, sheetName, xc - gapMm / 2.0, yc, -{feed_drop + 0.5:.6f}, gapMm, portHeightMm
        AssignLumpedPort oBoundary, portName, sheetName, xc - gapMm / 2.0, yc, -{feed_drop:.6f}, xc + gapMm / 2.0, yc, -{feed_drop:.6f}'''
    text = replace_once(text, old, new)
    text += balanced_feed_helpers()
    path.write_text(text, encoding="ascii")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    name = str(CONFIG["name"])
    folder = args.out_dir / name
    folder.mkdir()
    project = folder / f"{name}.aedt"
    touchstone = folder / f"{name}.s16p"
    builder = folder / f"build_{name}.vbs"
    solver = folder / f"solve_export_{name}.vbs"
    write_builder(builder, project)
    write_solve_export(solver, project, touchstone)
    row = {**CONFIG, "project_path": str(project), "touchstone_path": str(touchstone), "builder_path": str(builder), "solver_path": str(solver)}
    write_csv(args.out_dir / "candidate_manifest.csv", [row])
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "fixed 4x4 explicit balanced twin-lead feed smoke; no end-load parameter scan",
        "configuration": CONFIG,
        "acceptance_gate": "converged Delta S <= 0.05; valid S16; min passive RL >= 10 dB; zero small-mesh-segment warnings",
        "next_step_on_pass": "independent 8x8 smoke only",
        "next_step_on_fail": "do not start 8x8; inspect feed-port geometry and model a physical matching network",
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def profile_metrics(folder: Path) -> dict[str, Any]:
    profiles = list(folder.rglob("*.profile"))
    parsed_profiles: list[tuple[Path, str, list[float]]] = []
    for profile in profiles:
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
        parsed_profiles.append((profile, text, deltas))
    # HFSS can emit a zero-iteration variation profile beside LastAdaptive.
    # Select the profile that actually records the most adaptive iterations.
    selected = max(parsed_profiles, key=lambda item: (len(item[2]), item[0].stat().st_mtime), default=None)
    profile, text, deltas = selected if selected is not None else (None, "", [])
    converged = "Adaptive Passes converged" in text and "did not converge" not in text
    errors = list(folder.rglob("*.g3derr"))
    warning_count = sum(len(re.findall(r"small mesh segment", item.read_text(encoding="utf-8", errors="ignore"), flags=re.I)) for item in errors)
    return {
        "profile": str(profile) if profile is not None else "",
        "pass_count": len(deltas),
        "final_delta_s": deltas[-1] if deltas else None,
        "converged": converged,
        "small_mesh_segment_count": warning_count,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.out_dir / "candidate_manifest.csv"
    if not manifest.exists():
        prepare(args)
    row = next(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    folder = Path(row["project_path"]).parent
    build_log = folder / "build.log"
    solve_log = folder / "solve_export.log"
    with build_log.open("w", encoding="utf-8", errors="replace") as handle:
        build = subprocess.run([str(args.ansys_exe), "-RunScriptAndExit", row["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    solve_code: int | None = None
    if build.returncode == 0:
        with solve_log.open("w", encoding="utf-8", errors="replace") as handle:
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
    if touchstone.exists() and touchstone.stat().st_size > 1000:
        parsed = parse_touchstone(touchstone)
        s = np.asarray(parsed["s_parameters"], dtype=np.complex128)
        if s.shape == (1, 16, 16):
            passive = -20.0 * np.log10(np.maximum(np.abs(np.diag(s[0])), 1.0e-15))
            metric["passive_rl_min_db"] = float(np.min(passive))
            metric["passive_rl_median_db"] = float(np.median(passive))
            metric["passive_rl_10db_port_pass_rate"] = float(np.mean(passive >= 10.0))
    metric["gate_pass"] = int(
        metric.get("build_return_code") == 0
        and metric.get("solve_return_code") == 0
        and metric.get("converged") is True
        and (metric.get("final_delta_s") or float("inf")) <= 0.05
        and metric.get("touchstone_exists") is True
        and metric.get("small_mesh_segment_count") == 0
        and float(metric.get("passive_rl_min_db", -float("inf"))) >= 10.0
    )
    write_csv(args.out_dir / "validation_metrics.csv", [metric])
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "4x4_explicit_balanced_feed",
        "metric": metric,
        "decision": "allow_independent_8x8_smoke" if metric["gate_pass"] else "block_8x8_and_refine_feed_or_matching_model",
        "active_2400_started": False,
    }
    (args.out_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def status(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prepared": (args.out_dir / "candidate_manifest.csv").exists(),
        "run_complete": (args.out_dir / "run_summary.json").exists(),
        "analyzed": (args.out_dir / "stage_summary.json").exists(),
    }


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
