"""Strict-convergence 8x8 HFSS smoke for rounded end-loading geometries."""

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

from analyze_full_s256p_active_return import parse_touchstone
from design_modal_subarray_network import passive_metrics
from run_geometry_feed_smoke import replace_once, vp, write_solve_export


ROOT = Path(__file__).resolve().parent
SOURCE_BUILDER = ROOT / "build_ura16_quick.vbs"
DEFAULT_OUT = ROOT / "hfss_outputs" / "embedded8x8_modal_smoke_20260715_run03"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")

CANDIDATES: tuple[dict[str, Any], ...] = (
    {"name": "short_plain_control", "kind": "plain", "length": 12.6, "radius": 0.35},
    {
        "name": "smooth_blended_l11p2_bar2p0",
        "kind": "smooth_blended",
        "length": 11.2,
        "radius": 0.35,
        "bar_length": 2.0,
        "bar_radius": 0.22,
        "blend_radius": 0.48,
        "cap_radius": 0.26,
    },
    {
        "name": "smooth_blended_l10p8_bar2p4",
        "kind": "smooth_blended",
        "length": 10.8,
        "radius": 0.35,
        "bar_length": 2.4,
        "bar_radius": 0.22,
        "blend_radius": 0.52,
        "cap_radius": 0.28,
    },
    {
        "name": "smooth_compact_l10p4_bar3p0_dx16p0",
        "kind": "smooth_blended",
        "length": 10.4,
        "radius": 0.35,
        "bar_length": 3.0,
        "bar_radius": 0.24,
        "blend_radius": 0.54,
        "cap_radius": 0.30,
        "spacing_x": 16.0,
        "spacing_y": 15.0,
    },
)
DEFAULT_RUN = ("short_plain_control", "smooth_blended_l11p2_bar2p0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="status")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--candidate", action="append", choices=[str(item["name"]) for item in CANDIDATES])
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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


def smooth_geometry(candidate: dict[str, Any]) -> str:
    half = float(candidate["bar_length"]) / 2.0
    bar_radius = float(candidate["bar_radius"])
    blend = float(candidate["blend_radius"])
    cap = float(candidate["cap_radius"])
    length = float(candidate["bar_length"])
    return f'''        CreateHat oEditor, "Bar_" & nameBase & "_L", xStartLeft, yc - {half:.6f}, 0.0, {bar_radius:.6f}, {length:.6f}
        CreateHat oEditor, "Bar_" & nameBase & "_R", xStartRight + armLenMm, yc - {half:.6f}, 0.0, {bar_radius:.6f}, {length:.6f}
        CreateSphere oEditor, "Blend_" & nameBase & "_L", xStartLeft, yc, 0.0, {blend:.6f}
        CreateSphere oEditor, "Blend_" & nameBase & "_R", xStartRight + armLenMm, yc, 0.0, {blend:.6f}
        CreateSphere oEditor, "Cap_" & nameBase & "_L0", xStartLeft, yc - {half:.6f}, 0.0, {cap:.6f}
        CreateSphere oEditor, "Cap_" & nameBase & "_L1", xStartLeft, yc + {half:.6f}, 0.0, {cap:.6f}
        CreateSphere oEditor, "Cap_" & nameBase & "_R0", xStartRight + armLenMm, yc - {half:.6f}, 0.0, {cap:.6f}
        CreateSphere oEditor, "Cap_" & nameBase & "_R1", xStartRight + armLenMm, yc + {half:.6f}, 0.0, {cap:.6f}
        oEditor.Unite Array("NAME:Selections", "Selections:=", leftName & ",Bar_" & nameBase & "_L,Blend_" & nameBase & "_L,Cap_" & nameBase & "_L0,Cap_" & nameBase & "_L1"), Array("NAME:UniteParameters", "KeepOriginals:=", False)
        oEditor.Unite Array("NAME:Selections", "Selections:=", rightName & ",Bar_" & nameBase & "_R,Blend_" & nameBase & "_R,Cap_" & nameBase & "_R0,Cap_" & nameBase & "_R1"), Array("NAME:UniteParameters", "KeepOriginals:=", False)'''


def sanitized_smooth_geometry(candidate: dict[str, Any]) -> str:
    """Return a non-tangent, cylinder-only version of the end loading.

    The previous sphere/bar construction contained three-way tangencies that
    produced sliver faces at 8x8 scale.  These sleeves deliberately overlap
    both arm and bar before one Unite operation per dipole half.
    """
    half = float(candidate["bar_length"]) / 2.0
    bar_radius = float(candidate["bar_radius"])
    cap = float(candidate["cap_radius"])
    blend = float(candidate["blend_radius"])
    bar_length = float(candidate["bar_length"])
    sleeve_length = float(candidate.get("blend_sleeve_length", 0.60))
    sleeve_start = -sleeve_length / 2.0
    cap_length = 2.0 * cap
    return f'''        CreateHat oEditor, "BarCore_" & nameBase & "_L", xStartLeft, yc - {half:.6f}, 0.0, {bar_radius:.6f}, {bar_length:.6f}
        CreateHat oEditor, "BarCore_" & nameBase & "_R", xStartRight + armLenMm, yc - {half:.6f}, 0.0, {bar_radius:.6f}, {bar_length:.6f}
        CreateHat oEditor, "Cap_" & nameBase & "_L0", xStartLeft, yc - {half + cap:.6f}, 0.0, {cap:.6f}, {cap_length:.6f}
        CreateHat oEditor, "Cap_" & nameBase & "_L1", xStartLeft, yc + {half - cap:.6f}, 0.0, {cap:.6f}, {cap_length:.6f}
        CreateHat oEditor, "Cap_" & nameBase & "_R0", xStartRight + armLenMm, yc - {half + cap:.6f}, 0.0, {cap:.6f}, {cap_length:.6f}
        CreateHat oEditor, "Cap_" & nameBase & "_R1", xStartRight + armLenMm, yc + {half - cap:.6f}, 0.0, {cap:.6f}, {cap_length:.6f}
        CreateXSleeve oEditor, "BlendSleeve_" & nameBase & "_L", xStartLeft + {sleeve_start:.6f}, yc, 0.0, {blend:.6f}, {sleeve_length:.6f}
        CreateXSleeve oEditor, "BlendSleeve_" & nameBase & "_R", xStartRight + armLenMm + {sleeve_start:.6f}, yc, 0.0, {blend:.6f}, {sleeve_length:.6f}
        oEditor.Unite Array("NAME:Selections", "Selections:=", leftName & ",BarCore_" & nameBase & "_L,Cap_" & nameBase & "_L0,Cap_" & nameBase & "_L1,BlendSleeve_" & nameBase & "_L"), Array("NAME:UniteParameters", "KeepOriginals:=", False)
        oEditor.Unite Array("NAME:Selections", "Selections:=", rightName & ",BarCore_" & nameBase & "_R,Cap_" & nameBase & "_R0,Cap_" & nameBase & "_R1,BlendSleeve_" & nameBase & "_R"), Array("NAME:UniteParameters", "KeepOriginals:=", False)'''


def t_equivalent_geometry(candidate: dict[str, Any]) -> str:
    """Use one robust transverse solid per arm instead of four tangent solids.

    The unchanged bar/cap/blend values define the equivalent loading envelope:
    cap sets the terminal radius and added transverse length, while blend sets
    a finite inboard overlap with the dipole arm.
    """
    bar_radius = float(candidate["bar_radius"])
    cap = float(candidate["cap_radius"])
    blend = float(candidate["blend_radius"])
    radius = max(bar_radius, cap)
    length = float(candidate["bar_length"]) + 2.0 * cap
    overlap = max(0.10, min(0.40, blend - bar_radius))
    half = length / 2.0
    return f'''        CreateHat oEditor, "TEq_" & nameBase & "_L", xStartLeft + {overlap:.6f}, yc - {half:.6f}, 0.0, {radius:.6f}, {length:.6f}
        CreateHat oEditor, "TEq_" & nameBase & "_R", xStartRight + armLenMm - {overlap:.6f}, yc - {half:.6f}, 0.0, {radius:.6f}, {length:.6f}
        oEditor.Unite Array("NAME:Selections", "Selections:=", leftName & ",TEq_" & nameBase & "_L"), Array("NAME:UniteParameters", "KeepOriginals:=", False)
        oEditor.Unite Array("NAME:Selections", "Selections:=", rightName & ",TEq_" & nameBase & "_R"), Array("NAME:UniteParameters", "KeepOriginals:=", False)'''


def planar_monolithic_geometry(candidate: dict[str, Any]) -> str:
    """Create each loaded half-dipole as one closed PEC sheet.

    This avoids all conductor/end-load boolean operations.  The fixed cap and
    blend values define the transverse width and inboard attachment position.
    """
    arm_half_width = float(candidate["radius"])
    cap = float(candidate["cap_radius"])
    blend = float(candidate["blend_radius"])
    bar_length = float(candidate["bar_length"])
    bar_half = (bar_length + 2.0 * cap) / 2.0
    overlap = max(0.10, min(0.40, blend - float(candidate["bar_radius"])))
    return f'''        CreatePlanarTLeft oEditor, leftName, xStartLeft, yc, armLenMm, {arm_half_width:.6f}, {cap:.6f}, {bar_half:.6f}, {overlap:.6f}
        CreatePlanarTRight oEditor, rightName, xStartRight, yc, armLenMm, {arm_half_width:.6f}, {cap:.6f}, {bar_half:.6f}, {overlap:.6f}'''


HAT_HELPER = '''

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


PLANAR_T_HELPER = '''

Sub CreatePlanarTLeft(editor, objName, xOuter, yCenter, armLength, armHalfWidth, barHalfWidthX, barHalfLengthY, overlap)
    Dim xInner, xBarInner, z0
    xInner = xOuter + armLength
    xBarInner = xOuter + overlap + barHalfWidthX
    z0 = 0.0
    editor.CreatePolyline Array( _
        "NAME:PolylineParameters", _
        "IsPolylineCovered:=", True, _
        "IsPolylineClosed:=", True, _
        Array("NAME:PolylinePoints", _
            Array("NAME:PLPoint", "X:=", Mm(xOuter), "Y:=", Mm(yCenter - barHalfLengthY), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xBarInner), "Y:=", Mm(yCenter - barHalfLengthY), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xBarInner), "Y:=", Mm(yCenter - armHalfWidth), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xInner), "Y:=", Mm(yCenter - armHalfWidth), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xInner), "Y:=", Mm(yCenter + armHalfWidth), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xBarInner), "Y:=", Mm(yCenter + armHalfWidth), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xBarInner), "Y:=", Mm(yCenter + barHalfLengthY), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xOuter), "Y:=", Mm(yCenter + barHalfLengthY), "Z:=", Mm(z0))), _
        Array("NAME:PolylineSegments", _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 0, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 1, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 2, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 3, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 4, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 5, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 6, "NoOfPoints:=", 2)), _
        Array("NAME:PolylineXSection", "XSectionType:=", "None", "XSectionOrient:=", "Auto", "XSectionWidth:=", "0mm", "XSectionTopWidth:=", "0mm", "XSectionHeight:=", "0mm", "XSectionNumSegments:=", "0", "XSectionBendType:=", "Corner")), _
        Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """pec""", "SolveInside:=", False)
End Sub

Sub CreatePlanarTRight(editor, objName, xInner, yCenter, armLength, armHalfWidth, barHalfWidthX, barHalfLengthY, overlap)
    Dim xOuter, xBarInner, z0
    xOuter = xInner + armLength
    xBarInner = xOuter - overlap - barHalfWidthX
    z0 = 0.0
    editor.CreatePolyline Array( _
        "NAME:PolylineParameters", _
        "IsPolylineCovered:=", True, _
        "IsPolylineClosed:=", True, _
        Array("NAME:PolylinePoints", _
            Array("NAME:PLPoint", "X:=", Mm(xOuter), "Y:=", Mm(yCenter - barHalfLengthY), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xBarInner), "Y:=", Mm(yCenter - barHalfLengthY), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xBarInner), "Y:=", Mm(yCenter - armHalfWidth), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xInner), "Y:=", Mm(yCenter - armHalfWidth), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xInner), "Y:=", Mm(yCenter + armHalfWidth), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xBarInner), "Y:=", Mm(yCenter + armHalfWidth), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xBarInner), "Y:=", Mm(yCenter + barHalfLengthY), "Z:=", Mm(z0)), _
            Array("NAME:PLPoint", "X:=", Mm(xOuter), "Y:=", Mm(yCenter + barHalfLengthY), "Z:=", Mm(z0))), _
        Array("NAME:PolylineSegments", _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 0, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 1, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 2, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 3, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 4, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 5, "NoOfPoints:=", 2), _
            Array("NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 6, "NoOfPoints:=", 2)), _
        Array("NAME:PolylineXSection", "XSectionType:=", "None", "XSectionOrient:=", "Auto", "XSectionWidth:=", "0mm", "XSectionTopWidth:=", "0mm", "XSectionHeight:=", "0mm", "XSectionNumSegments:=", "0", "XSectionBendType:=", "Corner")), _
        Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """pec""", "SolveInside:=", False)
End Sub
'''


X_SLEEVE_HELPER = '''

Sub CreateXSleeve(editor, objName, xStart, yCenter, zCenter, radius, length)
    editor.CreateCylinder Array( _
            "NAME:CylinderParameters", _
            "XCenter:=", Mm(xStart), _
            "YCenter:=", Mm(yCenter), _
            "ZCenter:=", Mm(zCenter), _
            "Radius:=", Mm(radius), _
            "Height:=", Mm(length), _
            "WhichAxis:=", "X", _
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
'''


def write_builder(path: Path, project: Path, candidate: dict[str, Any]) -> None:
    text = SOURCE_BUILDER.read_text(encoding="utf-8-sig")
    text = replace_once(text, r'projectPath = "D:\codex_workspace\hfss_ura16_quick_model\ura16_quick_10ghz.aedt"', f'projectPath = "{vp(project)}"')
    text = replace_once(text, "dipoleLengthMm = 15.0", f"dipoleLengthMm = {float(candidate['length']):.6f}")
    text = replace_once(text, "radiusMm = 0.25", f"radiusMm = {float(candidate['radius']):.6f}")
    spacing_x = float(candidate.get("spacing_x", 15.0))
    spacing_y = float(candidate.get("spacing_y", 15.0))
    text = replace_once(
        text,
        "Dim f0GHz, lambdaMm, spacingMm, dipoleLengthMm, gapMm, armLenMm, radiusMm",
        "Dim f0GHz, lambdaMm, spacingMm, spacingYmm, dipoleLengthMm, gapMm, armLenMm, radiusMm",
    )
    text = replace_once(text, "spacingMm = 15.0", f"spacingMm = {spacing_x:.6f}\nspacingYmm = {spacing_y:.6f}")
    text = replace_once(text, "airY = (ny - 1) * spacingMm + dipoleLengthMm + 2 * airPadMm", "airY = (ny - 1) * spacingYmm + dipoleLengthMm + 2 * airPadMm")
    text = replace_once(text, "yc = (iy - 0.5 * (ny - 1)) * spacingMm", "yc = (iy - 0.5 * (ny - 1)) * spacingYmm")
    side = int(candidate.get("side", 8))
    if side < 2:
        raise ValueError(f"Array side must be at least two, got {side}")
    text = replace_once(text, "nx = 16", f"nx = {side}")
    text = replace_once(text, "ny = 16", f"ny = {side}")
    text = strict_setup(text)
    if candidate["kind"] in ("smooth_blended", "smooth_blended_sanitized", "smooth_blended_t_equivalent", "planar_t_monolithic", "planar_t_sheet"):
        marker = '        CreateArm oEditor, rightName, xStartRight, yc, 0.0, radiusMm, armLenMm'
        if candidate["kind"] == "smooth_blended":
            geometry = smooth_geometry(candidate)
        elif candidate["kind"] == "smooth_blended_sanitized":
            geometry = sanitized_smooth_geometry(candidate)
        elif candidate["kind"] == "smooth_blended_t_equivalent":
            geometry = t_equivalent_geometry(candidate)
        else:
            geometry = planar_monolithic_geometry(candidate)
            both_arms = '        CreateArm oEditor, leftName, xStartLeft, yc, 0.0, radiusMm, armLenMm\n' + marker
            text = replace_once(text, both_arms, geometry)
            text += PLANAR_T_HELPER
            path.write_text(text, encoding="ascii")
            return
        text = replace_once(text, marker, marker + "\n" + geometry)
        text += HAT_HELPER
        if candidate["kind"] == "smooth_blended_sanitized":
            text += X_SLEEVE_HELPER
        else:
            text += '''

Sub CreateSphere(editor, objName, xCenter, yCenter, zCenter, radius)
    editor.CreateSphere Array( _
            "NAME:SphereParameters", _
            "XCenter:=", Mm(xCenter), _
            "YCenter:=", Mm(yCenter), _
            "ZCenter:=", Mm(zCenter), _
            "Radius:=", Mm(radius) _
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


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        name = str(candidate["name"])
        folder = args.out_dir / name
        folder.mkdir()
        project = folder / f"{name}.aedt"
        touchstone = folder / f"{name}.s64p"
        builder = folder / f"build_{name}.vbs"
        solver = folder / f"solve_export_{name}.vbs"
        write_builder(builder, project, candidate)
        write_solve_export(solver, project, touchstone)
        rows.append({**candidate, "project_path": str(project), "touchstone_path": str(touchstone), "builder_path": str(builder), "solver_path": str(solver)})
    write_csv(args.out_dir / "candidate_manifest.csv", rows)
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "scope": "8x8 embedded smoke only", "delta_s_gate": 0.05, "maximum_passes": 8, "default_run_candidates": list(DEFAULT_RUN)}
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def chosen_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    manifest = args.out_dir / "candidate_manifest.csv"
    if not manifest.exists():
        prepare(args)
    names = set(args.candidate or DEFAULT_RUN)
    return [row for row in csv.DictReader(manifest.open(encoding="utf-8-sig")) if row["name"] in names]


def run(args: argparse.Namespace) -> dict[str, Any]:
    statuses: list[dict[str, Any]] = []
    for row in chosen_rows(args):
        folder = Path(row["project_path"]).parent
        started = time.time()
        with (folder / "build.log").open("w", encoding="utf-8", errors="replace") as handle:
            build = subprocess.run([str(args.ansys_exe), "-RunScriptAndExit", row["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        code = int(build.returncode)
        if code == 0:
            with (folder / "solve_export.log").open("w", encoding="utf-8", errors="replace") as handle:
                solve = subprocess.run([str(args.ansys_exe), "-ng", "-RunScriptAndExit", row["solver_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            code = int(solve.returncode)
        statuses.append({"name": row["name"], "return_code": code, "elapsed_seconds": time.time() - started})
    write_csv(args.out_dir / "run_status.csv", statuses)
    return {"runs": statuses, "all_zero_exit": bool(statuses and all(row["return_code"] == 0 for row in statuses))}


def convergence(project: Path) -> dict[str, Any]:
    profiles = list(project.with_suffix(".aedtresults").rglob("*.profile"))
    if not profiles:
        return {"profile": "", "pass_count": 0, "final_delta_s": float("nan"), "converged": False}
    profile = max(profiles, key=lambda item: item.stat().st_mtime)
    content = profile.read_text(encoding="utf-8", errors="replace")
    values = [float(value) for value in re.findall(r"Max Mag\. Delta S\\',\s*([0-9.eE+-]+)", content)]
    final = values[-1] if values else float("nan")
    return {"profile": str(profile), "pass_count": len(re.findall(r"Name='Adaptive Pass \d+'", content)), "final_delta_s": final, "converged": bool(values and final <= 0.05 and "Adaptive Passes did not converge" not in content)}


def indices(x0: int, y0: int) -> list[int]:
    return [(x0 + dx) * 8 + y0 + dy for dx in range(4) for dy in range(4)]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.out_dir / "candidate_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for item in csv.DictReader(manifest.open(encoding="utf-8-sig")):
        touchstone = Path(item["touchstone_path"])
        info = convergence(Path(item["project_path"]))
        row: dict[str, Any] = {"name": item["name"], "kind": item["kind"], "touchstone_exists": touchstone.exists(), "pass_count": info["pass_count"], "final_delta_s": info["final_delta_s"], "converged": int(info["converged"]), "profile": info["profile"]}
        if touchstone.exists() and touchstone.stat().st_size > 1000:
            s = np.asarray(parse_touchstone(touchstone)["s_parameters"], dtype=np.complex128)
            if s.shape == (1, 64, 64):
                row.update(passive_metrics(s[0]))
                for class_name, (x0, y0) in {"corner": (0, 0), "edge": (0, 2), "interior": (2, 2)}.items():
                    selected = indices(x0, y0)
                    class_rows.append({"name": item["name"], "class": class_name, "window_x0": x0, "window_y0": y0, **passive_metrics(s[0][np.ix_(selected, selected)])})
            else:
                row["status"] = f"unexpected_shape_{s.shape}"
        rows.append(row)
    write_csv(args.out_dir / "s64_geometry_metrics.csv", rows)
    write_csv(args.out_dir / "embedded_class_window_metrics.csv", class_rows)
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "converged_candidates": [row["name"] for row in rows if row["converged"] == 1], "allow_modal_network_evaluation": bool(any(row["converged"] == 1 for row in rows)), "decision": "allow_modal_network_evaluation" if any(row["converged"] == 1 for row in rows) else "block_modal_network_evaluation_until_delta_s_gate_passes"}
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def status(args: argparse.Namespace) -> dict[str, Any]:
    return {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "output_exists": args.out_dir.exists(), "manifest_exists": (args.out_dir / "candidate_manifest.csv").exists()}


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
