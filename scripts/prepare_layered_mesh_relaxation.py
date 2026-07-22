#!/usr/bin/env python3
"""Create a local-memory layered mesh branch using 0.22/0.30 mm limits."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_layered_feedmesh_c021_h028_20260722_run02"
)
OUT_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_layered_feedmesh_c022_h030_20260722_run01"
)
ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
NPORTS = 256
CORE_OP = "FeedCoreUniform_0p220mm"
HALO_OP = "FeedHaloUniform_0p300mm"


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def blocks(text: str, pattern: str) -> list[str]:
    return [match.group(0) for match in re.finditer(pattern, text, re.DOTALL)]


def digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def build_vbs(project: Path) -> str:
    cores = ", ".join(f'"FeedCore_{index:03d}"' for index in range(NPORTS))
    halos = ", ".join(f'"FeedNbr_{index:03d}"' for index in range(NPORTS))
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oMesh
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{PROJECT_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oMesh = oDesign.GetModule("MeshSetup")
On Error Resume Next
oMesh.DeleteOp Array("FeedCoreUniform_0p210mm")
oMesh.DeleteOp Array("FeedHaloUniform_0p280mm")
oMesh.DeleteOp Array("{CORE_OP}")
oMesh.DeleteOp Array("{HALO_OP}")
On Error GoTo 0
oMesh.AssignLengthOp Array("NAME:{CORE_OP}", _
    "RefineInside:=", True, "Enabled:=", True, _
    "Objects:=", Array({cores}), "RestrictElem:=", False, _
    "NumMaxElem:=", "1000", "RestrictLength:=", True, _
    "MaxLength:=", "0.220000mm", "UseAdvSizing:=", False)
oMesh.AssignLengthOp Array("NAME:{HALO_OP}", _
    "RefineInside:=", True, "Enabled:=", True, _
    "Objects:=", Array({halos}), "RestrictElem:=", False, _
    "NumMaxElem:=", "1000", "RestrictLength:=", True, _
    "MaxLength:=", "0.300000mm", "UseAdvSizing:=", False)
On Error Resume Next
oDesign.DeleteFullVariation "All", False
On Error GoTo 0
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def main() -> None:
    if OUT_RUN.exists() and any(OUT_RUN.iterdir()):
        raise FileExistsError(OUT_RUN)
    source_project = SOURCE_RUN / PROJECT_NAME / f"{PROJECT_NAME}.aedt"
    source_text = source_project.read_text(encoding="utf-8", errors="ignore")
    project_dir = OUT_RUN / PROJECT_NAME
    project_dir.mkdir(parents=True)
    project = project_dir / f"{PROJECT_NAME}.aedt"
    shutil.copy2(source_project, project)
    reference_dir = OUT_RUN / "reference_pass05"
    reference_dir.mkdir()
    for path in (SOURCE_RUN / "reference_pass05").glob("*"):
        if path.is_file():
            shutil.copy2(path, reference_dir / path.name)
    vbs = project_dir / "apply_layered_mesh_c022_h030.vbs"
    log = project_dir / "apply_layered_mesh_c022_h030.log"
    vbs.write_text(build_vbs(project), encoding="ascii")
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(
            [str(ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    output_text = project.read_text(encoding="utf-8", errors="ignore")
    source_ports = blocks(source_text, r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'")
    output_ports = blocks(output_text, r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'")
    source_perfecte = blocks(source_text, r"\$begin 'PEC_GroundPatch_Sheets'.*?\$end 'PEC_GroundPatch_Sheets'")
    output_perfecte = blocks(output_text, r"\$begin 'PEC_GroundPatch_Sheets'.*?\$end 'PEC_GroundPatch_Sheets'")
    core_mesh = re.search(rf"\$begin '{CORE_OP}'(.*?)\$end '{CORE_OP}'", output_text, re.DOTALL)
    halo_mesh = re.search(rf"\$begin '{HALO_OP}'(.*?)\$end '{HALO_OP}'", output_text, re.DOTALL)
    passed = bool(
        result.returncode == 0
        and len(source_ports) == len(output_ports) == NPORTS
        and digest(source_ports) == digest(output_ports)
        and digest(source_perfecte) == digest(output_perfecte)
        and core_mesh
        and halo_mesh
        and "RefineInside=true" in core_mesh.group(1)
        and "MaxLength='0.22mm'" in core_mesh.group(1)
        and "RefineInside=true" in halo_mesh.group(1)
        and "MaxLength='0.3mm'" in halo_mesh.group(1)
        and "PortFeedUniform_0p180mm" in output_text
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source_project),
        "output_project": str(project),
        "port_count": len(output_ports),
        "port_definition_hash_unchanged": digest(source_ports) == digest(output_ports),
        "perfect_e_boundary_hash_unchanged": digest(source_perfecte) == digest(output_perfecte),
        "feed_core_count": len(set(re.findall(r"Name='FeedCore_(\d{3})'", output_text))),
        "feed_halo_count": len(set(re.findall(r"Name='FeedNbr_(\d{3})'", output_text))),
        "core_region_mm": [0.4, 0.4, 0.787],
        "core_mesh_mm": 0.22,
        "halo_region_mm": [0.8, 0.8, 0.787],
        "halo_mesh_mm": 0.30,
        "port_surface_mesh_mm": 0.18,
        "configuration_smoke_pass": passed,
        "training_labels_locked": True,
        "decision": "allow_layered_ddm6_smoke" if passed else "block_layered_ddm6_smoke",
    }
    (OUT_RUN / "layered_feedmesh_prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
