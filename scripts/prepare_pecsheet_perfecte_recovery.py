#!/usr/bin/env python3
"""Add an explicit Perfect E boundary to the zero-thickness conductor sheets."""

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
    / "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_20260722_run01"
)
OUT_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_perfecte_20260722_run01"
)
ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
BOUNDARY_NAME = "PEC_GroundPatch_Sheets"
NPORTS = 256


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def blocks(text: str, pattern: str) -> list[str]:
    return [match.group(0) for match in re.finditer(pattern, text, re.DOTALL)]


def digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def geometry_parts(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for block in re.findall(
        r"\$begin 'GeometryPart'.*?\$end 'GeometryPart'", text, re.DOTALL
    ):
        name = re.search(r"\bName='([^']+)'", block)
        if name:
            result[name.group(1)] = block
    return result


def build_vbs(project: Path) -> str:
    names = ["Ground", *[f"Patch_{index:03d}" for index in range(NPORTS)]]
    object_array = ", ".join(f'"{name}"' for name in names)
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oBoundary
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{PROJECT_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oBoundary = oDesign.GetModule("BoundarySetup")
On Error Resume Next
oBoundary.DeleteBoundaries Array("{BOUNDARY_NAME}")
On Error GoTo 0
oBoundary.AssignPerfectE Array("NAME:{BOUNDARY_NAME}", _
    "Objects:=", Array({object_array}), "InfGroundPlane:=", False)
On Error Resume Next
oDesign.DeleteFullVariation "All", False
On Error GoTo 0
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def main() -> None:
    source_project = SOURCE_RUN / PROJECT_NAME / f"{PROJECT_NAME}.aedt"
    source_text = source_project.read_text(encoding="utf-8", errors="ignore")
    project_dir = OUT_RUN / PROJECT_NAME
    project = project_dir / f"{PROJECT_NAME}.aedt"
    prior_summary = OUT_RUN / "pecsheet_perfecte_prepare_summary.json"
    resume_validation = bool(
        project.exists()
        and not project.with_suffix(".aedt.lock").exists()
        and prior_summary.exists()
        and not json.loads(prior_summary.read_text(encoding="utf-8")).get(
            "configuration_smoke_pass"
        )
    )
    if not resume_validation:
        if OUT_RUN.exists() and any(OUT_RUN.iterdir()):
            raise FileExistsError(OUT_RUN)
        project_dir.mkdir(parents=True)
        shutil.copy2(source_project, project)
        reference_dir = OUT_RUN / "reference_pass05"
        reference_dir.mkdir()
        for path in (SOURCE_RUN / "reference_pass05").glob("*"):
            if path.is_file():
                shutil.copy2(path, reference_dir / path.name)

        vbs = project_dir / "assign_ground_patch_perfecte.vbs"
        log = project_dir / "assign_ground_patch_perfecte.log"
        vbs.write_text(build_vbs(project), encoding="ascii")
        with log.open("w", encoding="utf-8", errors="replace") as handle:
            result = subprocess.run(
                [str(ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        apply_return_code = int(result.returncode)
    else:
        apply_return_code = 0

    output_text = project.read_text(encoding="utf-8", errors="ignore")
    source_ports = blocks(source_text, r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'")
    output_ports = blocks(output_text, r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'")
    boundary = re.search(
        rf"\$begin '{BOUNDARY_NAME}'(.*?)\$end '{BOUNDARY_NAME}'",
        output_text,
        re.DOTALL,
    )
    parts = geometry_parts(output_text)
    sheet_types = []
    for name in ["Ground", *[f"Patch_{index:03d}" for index in range(NPORTS)]]:
        part = parts.get(name)
        operation = re.search(r"OperationType='([^']+)'", part) if part else None
        sheet_types.append(operation.group(1) if operation else None)
    port_hash_unchanged = digest(source_ports) == digest(output_ports)
    passed = bool(
        apply_return_code == 0
        and len(source_ports) == len(output_ports) == NPORTS
        and port_hash_unchanged
        and boundary
        and all(kind == "Rectangle" for kind in sheet_types)
        and "FeedNeighborhoodUniform_0p180mm" in output_text
        and "PortFeedUniform_0p180mm" in output_text
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source_project),
        "output_project": str(project),
        "apply_return_code": apply_return_code,
        "resumed_validation_after_validator_fix": resume_validation,
        "port_count": len(output_ports),
        "port_definition_hash_unchanged": port_hash_unchanged,
        "perfect_e_boundary_present": boundary is not None,
        "perfect_e_object_count_expected": 257,
        "ground_patch_all_zero_thickness_sheets": all(
            kind == "Rectangle" for kind in sheet_types
        ),
        "feed_mesh_preserved": "FeedNeighborhoodUniform_0p180mm" in output_text,
        "port_mesh_preserved": "PortFeedUniform_0p180mm" in output_text,
        "configuration_smoke_pass": passed,
        "training_labels_locked": True,
        "decision": "allow_port_topology_smoke" if passed else "block_port_topology_smoke",
    }
    (OUT_RUN / "pecsheet_perfecte_prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
