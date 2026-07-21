#!/usr/bin/env python3
"""Add lightweight horizontal mesh-seed sheets around all 256 feed locations."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from prepare_16x16_feed_neighborhood_mesh import NPORTS, port_centers, vp


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
    / "grounded_patch_direct_16x16_feedsheet_source_20260719_run01"
)
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
SETUP_NAME = "Setup_10GHz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mesh-mm", type=float, default=0.18)
    parser.add_argument("--sheet-mm", type=float, default=1.2)
    parser.add_argument("--percent-refinement", type=float, default=5.0)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def sheet_block(index: int, x: float, y: float, z: float, size: float) -> str:
    half = size / 2.0
    return f'''oEditor.CreateRectangle Array( _
    "NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", "{x - half:.9g}mm", _
    "YStart:=", "{y - half:.9g}mm", "ZStart:=", "{z:.9g}mm", _
    "Width:=", "{size:.9g}mm", "Height:=", "{size:.9g}mm", "WhichAxis:=", "Z"), _
    Array("NAME:Attributes", "Name:=", "FeedSeed_{index:03d}", "Flags:=", "", _
    "Color:=", "(220 180 40)", "Transparency:=", 0.85, _
    "PartCoordinateSystem:=", "Global", "UDMId:=", "", "MaterialValue:=", _
    Chr(34) & "RO5880_Custom" & Chr(34), "SurfaceMaterialValue:=", Chr(34) & Chr(34), _
    "SolveInside:=", True, "ShellElement:=", False, "IsMaterialEditable:=", True, _
    "UseMaterialAppearance:=", False, "IsLightweight:=", False)
'''


def build_vbs(
    project: Path,
    rows: list[tuple[int, float, float, float, float]],
    mesh_mm: float,
    sheet_mm: float,
    percent_refinement: float,
) -> str:
    sheets = "".join(
        sheet_block(index, x, y, z_start + height / 2.0, sheet_mm)
        for index, x, y, z_start, height in rows
    )
    object_array = ", ".join(f'"FeedSeed_{index:03d}"' for index in range(NPORTS))
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oMesh, oAnalysis
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{PROJECT_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
{sheets}
Set oMesh = oDesign.GetModule("MeshSetup")
On Error Resume Next
oMesh.DeleteOp Array("FeedSheetUniform_0p180mm")
On Error GoTo 0
oMesh.AssignLengthOp Array( _
    "NAME:FeedSheetUniform_0p180mm", "RefineInside:=", False, "Enabled:=", True, _
    "Objects:=", Array({object_array}), "RestrictElem:=", False, _
    "NumMaxElem:=", "1000", "RestrictLength:=", True, _
    "MaxLength:=", "{mesh_mm:.6f}mm", "UseAdvSizing:=", False)
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
    if not source_project.exists() or not (pass08 / "grounded_patch_16x16.s256p").exists():
        raise FileNotFoundError("Source project and valid pass08 reference are required")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out_dir}")
    if abs(args.mesh_mm - 0.18) > 1.0e-12 or abs(args.percent_refinement - 5.0) > 1.0e-12:
        raise ValueError("This branch is locked to 0.18 mm and 5 percent refinement")
    if not 0.8 <= args.sheet_mm <= 1.5:
        raise ValueError("Feed sheet must remain within 0.8-1.5 mm")

    rows = port_centers(source_project.read_text(encoding="utf-8", errors="ignore"))
    project_dir = out_dir / PROJECT_NAME
    project_dir.mkdir(parents=True)
    project = project_dir / f"{PROJECT_NAME}.aedt"
    shutil.copy2(source_project, project)
    reference_dir = out_dir / "reference_pass08"
    reference_dir.mkdir()
    for name in ("grounded_patch_16x16.s256p", "stage_metrics.json", "DV1332_S1287_V0.profile"):
        shutil.copy2(pass08 / name, reference_dir / name)

    vbs = project_dir / "apply_uniform_feed_sheet_mesh.vbs"
    log = project_dir / "apply_uniform_feed_sheet_mesh.log"
    vbs.write_text(
        build_vbs(project, rows, args.mesh_mm, args.sheet_mm, args.percent_refinement),
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
    seed_names = sorted(set(re.findall(r"Name='FeedSeed_(\d{3})'", output_text)))
    log_errors = log.read_text(encoding="utf-8", errors="ignore").lower().count("[error]")
    passed = bool(
        process.returncode == 0
        and log_errors == 0
        and seed_names == [f"{index:03d}" for index in range(NPORTS)]
        and "FeedSheetUniform_0p180mm" in output_text
        and "PortFeedUniform_0p180mm" in output_text
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source_project),
        "reference_pass08": str(reference_dir),
        "output_project": str(project),
        "feed_seed_sheet_count": len(seed_names),
        "sheet_xy_mm": float(args.sheet_mm),
        "sheet_z_fraction_of_substrate": 0.5,
        "local_mesh_mm": float(args.mesh_mm),
        "percent_refinement": float(args.percent_refinement),
        "port_mesh_operation_preserved": "PortFeedUniform_0p180mm" in output_text,
        "feed_sheet_mesh_operation_written": "FeedSheetUniform_0p180mm" in output_text,
        "apply_return_code": int(process.returncode),
        "log_error_count": log_errors,
        "configuration_smoke_pass": passed,
        "training_labels_unlocked": False,
        "decision": "allow_clean_staged_solve" if passed else "block_staged_solve",
    }
    (out_dir / "feed_sheet_mesh_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
