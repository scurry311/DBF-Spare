#!/usr/bin/env python3
"""Clone the trusted 16x16 geometry and apply uniform local meshes to all ports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DEFAULT_SOURCE = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_staged_convergence_20260717_run01"
    / "grounded_patch_16x16"
    / "grounded_patch_16x16.aedt"
)
DEFAULT_OUT = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_portmesh_source_20260718_run01"
)
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
SETUP_NAME = "Setup_10GHz"
NPORTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--port-mesh-mm", type=float, default=0.10)
    parser.add_argument("--percent-refinement", type=float, default=10.0)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def operation_name(mesh_mm: float) -> str:
    return f"PortFeedUniform_{mesh_mm:.3f}mm".replace(".", "p")


def mesh_vbs(project: Path, mesh_mm: float, percent_refinement: float) -> str:
    objects = ", ".join(f'"PortSheet_{index:03d}"' for index in range(NPORTS))
    op_name = operation_name(mesh_mm)
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oMesh, oAnalysis
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{PROJECT_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oMesh = oDesign.GetModule("MeshSetup")
On Error Resume Next
oMesh.DeleteOp Array("{op_name}")
On Error GoTo 0
oMesh.AssignLengthOp Array( _
    "NAME:{op_name}", _
    "RefineInside:=", False, _
    "Enabled:=", True, _
    "Objects:=", Array({objects}), _
    "RestrictElem:=", False, _
    "NumMaxElem:=", "1000", _
    "RestrictLength:=", True, _
    "MaxLength:=", "{mesh_mm:.6f}mm", _
    "UseAdvSizing:=", False)
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
oAnalysis.EditSetup "{SETUP_NAME}", Array( _
    "NAME:{SETUP_NAME}", _
    "SolveType:=", "Single", _
    "Frequency:=", "10GHz", _
    "MaxDeltaS:=", 0.05, _
    "MaximumPasses:=", 30, _
    "MinimumPasses:=", 2, _
    "MinimumConvergedPasses:=", 2, _
    "PercentRefinement:=", {percent_refinement:.6g}, _
    "BasisOrder:=", 1, _
    "DoLambdaRefine:=", True, _
    "DoMaterialLambda:=", True, _
    "SetLambdaTarget:=", False, _
    "UseMaxTetIncrease:=", False, _
    "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, _
    "SetPortMinMaxTri:=", False)
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def main() -> None:
    args = parse_args()
    source = args.source_project.resolve()
    out_dir = args.out_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out_dir}")
    if not 0.04 <= args.port_mesh_mm <= 0.20:
        raise ValueError("port mesh must remain within the audited 0.04-0.20 mm range")
    if not 5.0 <= args.percent_refinement <= 15.0:
        raise ValueError("percent refinement must remain within 5-15 percent")

    source_text = source.read_text(encoding="utf-8", errors="ignore")
    source_ports = sorted(set(re.findall(r"Name='PortSheet_(\d{3})'", source_text)))
    if source_ports != [f"{index:03d}" for index in range(NPORTS)]:
        raise ValueError(f"Source project contains {len(source_ports)} unique port sheets, expected 256")

    project_dir = out_dir / PROJECT_NAME
    project_dir.mkdir(parents=True)
    project = project_dir / f"{PROJECT_NAME}.aedt"
    shutil.copy2(source, project)
    vbs = project_dir / "apply_uniform_port_mesh.vbs"
    log = project_dir / "apply_uniform_port_mesh.log"
    vbs.write_text(
        mesh_vbs(project, args.port_mesh_mm, args.percent_refinement), encoding="ascii"
    )
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.run(
            [str(args.ansys_exe), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    project_text = project.read_text(encoding="utf-8", errors="ignore")
    op_name = operation_name(args.port_mesh_mm)
    operation_written = op_name.lower() in project_text.lower()
    vbs_port_objects = sorted(set(re.findall(r'"PortSheet_(\d{3})"', vbs.read_text())))
    row = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source),
        "output_project": str(project),
        "operation_name": op_name,
        "port_mesh_mm": float(args.port_mesh_mm),
        "percent_refinement": float(args.percent_refinement),
        "source_port_sheet_count": len(source_ports),
        "vbs_port_sheet_count": len(vbs_port_objects),
        "apply_return_code": int(process.returncode),
        "operation_written_to_project": int(operation_written),
        "source_results_copied": 0,
    }
    with (out_dir / "port_mesh_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    passed = bool(
        process.returncode == 0
        and operation_written
        and len(source_ports) == NPORTS
        and len(vbs_port_objects) == NPORTS
    )
    summary = {
        **row,
        "configuration_smoke_pass": passed,
        "decision": "allow_clean_staged_solve" if passed else "block_staged_solve",
    }
    (out_dir / "port_mesh_prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
