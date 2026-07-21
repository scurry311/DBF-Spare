#!/usr/bin/env python3
"""Create physically equivalent local dielectric feed regions for all 256 ports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DEFAULT_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_portmesh_staged_convergence_20260718_run02"
)
DEFAULT_OUT = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_feednbr_source_20260719_run01"
)
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
SETUP_NAME = "Setup_10GHz"
NPORTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mesh-mm", type=float, default=0.18)
    parser.add_argument("--neighborhood-mm", type=float, default=1.2)
    parser.add_argument("--percent-refinement", type=float, default=5.0)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def parse_mm(value: str) -> float:
    match = re.fullmatch(r"\s*([+\-0-9.eE]+)mm\s*", value)
    if not match:
        raise ValueError(f"Expected millimetre value, got {value!r}")
    return float(match.group(1))


def port_centers(project_text: str) -> list[tuple[int, float, float, float, float]]:
    pattern = re.compile(
        r"Name='PortSheet_(\d{3})'.*?\$begin 'RectangleParameters'"
        r".*?XStart='([^']+)'.*?YStart='([^']+)'.*?ZStart='([^']+)'"
        r".*?Width='([^']+)'.*?Height='([^']+)'.*?WhichAxis='X'",
        re.DOTALL,
    )
    rows: list[tuple[int, float, float, float, float]] = []
    for match in pattern.finditer(project_text):
        index = int(match.group(1))
        x = parse_mm(match.group(2))
        y_start = parse_mm(match.group(3))
        z_start = parse_mm(match.group(4))
        width = parse_mm(match.group(5))
        height = parse_mm(match.group(6))
        rows.append((index, x, y_start + width / 2.0, z_start, height))
    rows.sort()
    if [row[0] for row in rows] != list(range(NPORTS)):
        raise ValueError(f"Parsed {len(rows)} ordered port sheets, expected {NPORTS}")
    return rows


def box_block(index: int, x: float, y: float, z: float, height: float, width: float) -> str:
    half = width / 2.0
    return f'''oEditor.CreateBox Array( _
    "NAME:BoxParameters", _
    "XPosition:=", "{x - half:.9g}mm", _
    "YPosition:=", "{y - half:.9g}mm", _
    "ZPosition:=", "{z:.9g}mm", _
    "XSize:=", "{width:.9g}mm", _
    "YSize:=", "{width:.9g}mm", _
    "ZSize:=", "{height:.9g}mm"), _
    Array("NAME:Attributes", "Name:=", "FeedNbr_{index:03d}", _
    "Flags:=", "", "Color:=", "(80 180 120)", "Transparency:=", 0.75, _
    "PartCoordinateSystem:=", "Global", "UDMId:=", "", "MaterialValue:=", _
    Chr(34) & "RO5880_Custom" & Chr(34), "SurfaceMaterialValue:=", Chr(34) & Chr(34), _
    "SolveInside:=", True, "ShellElement:=", False, "IsMaterialEditable:=", True, _
    "UseMaterialAppearance:=", False, "IsLightweight:=", False)
'''


def build_vbs(
    project: Path,
    rows: list[tuple[int, float, float, float, float]],
    mesh_mm: float,
    neighborhood_mm: float,
    percent_refinement: float,
) -> str:
    boxes = "".join(
        box_block(index, x, y, z, height, neighborhood_mm)
        for index, x, y, z, height in rows
    )
    names = ",".join(f"FeedNbr_{index:03d}" for index in range(NPORTS))
    object_array = ", ".join(f'"FeedNbr_{index:03d}"' for index in range(NPORTS))
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oMesh, oAnalysis
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{PROJECT_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
{boxes}
oEditor.Subtract Array("NAME:Selections", "Blank Parts:=", "Substrate", _
    "Tool Parts:=", "{names}"), _
    Array("NAME:SubtractParameters", "KeepOriginals:=", True)
Set oMesh = oDesign.GetModule("MeshSetup")
On Error Resume Next
oMesh.DeleteOp Array("FeedNeighborhoodUniform_0p180mm")
On Error GoTo 0
oMesh.AssignLengthOp Array( _
    "NAME:FeedNeighborhoodUniform_0p180mm", _
    "RefineInside:=", False, _
    "Enabled:=", True, _
    "Objects:=", Array({object_array}), _
    "RestrictElem:=", False, _
    "NumMaxElem:=", "1000", _
    "RestrictLength:=", True, _
    "MaxLength:=", "{mesh_mm:.6f}mm", _
    "UseAdvSizing:=", False)
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
oAnalysis.EditSetup "{SETUP_NAME}", Array( _
    "NAME:{SETUP_NAME}", "SolveType:=", "Single", "Frequency:=", "10GHz", _
    "MaxDeltaS:=", 0.05, "MaximumPasses:=", 30, "MinimumPasses:=", 2, _
    "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {percent_refinement:.6g}, _
    "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, _
    "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    source_project = source_run / PROJECT_NAME / f"{PROJECT_NAME}.aedt"
    pass08 = source_run / "stages" / "pass08"
    out_dir = args.out_dir.resolve()
    if not source_project.exists():
        raise FileNotFoundError(source_project)
    if not (pass08 / "grounded_patch_16x16.s256p").exists():
        raise FileNotFoundError("Valid pass08 Touchstone is required")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out_dir}")
    if abs(args.mesh_mm - 0.18) > 1.0e-12:
        raise ValueError("This validation branch is locked to a 0.18 mm local mesh")
    if abs(args.percent_refinement - 5.0) > 1.0e-12:
        raise ValueError("This validation branch is locked to 5 percent refinement")
    if not 0.8 <= args.neighborhood_mm <= 1.5:
        raise ValueError("Feed neighborhood must remain within the audited 0.8-1.5 mm range")

    text = source_project.read_text(encoding="utf-8", errors="ignore")
    rows = port_centers(text)
    spacings_x = sorted(
        {round(abs(a[1] - b[1]), 9) for a in rows for b in rows if 0 < abs(a[1] - b[1]) < 20}
    )
    spacings_y = sorted(
        {round(abs(a[2] - b[2]), 9) for a in rows for b in rows if 0 < abs(a[2] - b[2]) < 20}
    )
    if spacings_x[:1] != [15.0] or spacings_y[:1] != [15.0]:
        raise ValueError(f"Unexpected array spacing: x={spacings_x[:3]}, y={spacings_y[:3]}")

    project_dir = out_dir / PROJECT_NAME
    project_dir.mkdir(parents=True)
    project = project_dir / f"{PROJECT_NAME}.aedt"
    shutil.copy2(source_project, project)
    reference_dir = out_dir / "reference_pass08"
    reference_dir.mkdir()
    for name in ("grounded_patch_16x16.s256p", "stage_metrics.json", "DV1332_S1287_V0.profile"):
        shutil.copy2(pass08 / name, reference_dir / name)

    vbs = project_dir / "apply_uniform_feed_neighborhood_mesh.vbs"
    log = project_dir / "apply_uniform_feed_neighborhood_mesh.log"
    vbs.write_text(
        build_vbs(project, rows, args.mesh_mm, args.neighborhood_mm, args.percent_refinement),
        encoding="ascii",
    )
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.run(
            [str(args.ansys_exe), "-ng", "-RunScriptAndExit", str(vbs.resolve())],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    output_text = project.read_text(encoding="utf-8", errors="ignore")
    feed_names = sorted(set(re.findall(r"Name='FeedNbr_(\d{3})'", output_text)))
    op_written = "FeedNeighborhoodUniform_0p180mm" in output_text
    port_op_preserved = "PortFeedUniform_0p180mm" in output_text
    log_text = log.read_text(encoding="utf-8", errors="ignore")
    error_count = log_text.lower().count("[error]")
    passed = bool(
        process.returncode == 0
        and error_count == 0
        and feed_names == [f"{index:03d}" for index in range(NPORTS)]
        and op_written
        and port_op_preserved
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source_project),
        "reference_pass08": str(reference_dir),
        "output_project": str(project),
        "feed_neighborhood_count": len(feed_names),
        "neighborhood_xy_mm": float(args.neighborhood_mm),
        "neighborhood_z_mm": float(rows[0][4]),
        "local_mesh_mm": float(args.mesh_mm),
        "feed_neighborhood_refine_inside": False,
        "percent_refinement": float(args.percent_refinement),
        "port_mesh_operation_preserved": port_op_preserved,
        "feed_mesh_operation_written": op_written,
        "apply_return_code": int(process.returncode),
        "log_error_count": error_count,
        "configuration_smoke_pass": passed,
        "training_labels_unlocked": False,
        "decision": "allow_clean_staged_solve" if passed else "block_staged_solve",
    }
    (out_dir / "feed_neighborhood_mesh_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out_dir / "feed_neighborhood_centers.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "x_mm", "y_mm", "z_start_mm", "z_size_mm"])
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
