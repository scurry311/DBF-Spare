#!/usr/bin/env python3
"""Create an isolated 16x16 branch with deterministic core/halo feed meshes."""

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
    / "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_perfecte_ddm6_20260722_run01"
)
OUT_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_layered_feedmesh_c021_h028_20260722_run02"
)
ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
NPORTS = 256
CORE_SIZE_MM = 0.4
CORE_MESH_MM = 0.21
HALO_SIZE_MM = 0.8
HALO_MESH_MM = 0.28
DEPTH_MM = 0.787
CORE_OP = "FeedCoreUniform_0p210mm"
HALO_OP = "FeedHaloUniform_0p280mm"
OLD_OP = "FeedNeighborhoodUniform_0p180mm"


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def blocks(text: str, pattern: str) -> list[str]:
    return [match.group(0) for match in re.finditer(pattern, text, re.DOTALL)]


def digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def feed_centers(text: str) -> list[tuple[int, float, float, float, float]]:
    result: list[tuple[int, float, float, float, float]] = []
    geometry = blocks(text, r"\$begin 'GeometryPart'.*?\$end 'GeometryPart'")
    for block in geometry:
        name = re.search(r"\bName='FeedNbr_(\d{3})'", block)
        if not name:
            continue
        values: dict[str, float] = {}
        for key in ("XPosition", "YPosition", "ZPosition", "XSize", "YSize", "ZSize"):
            match = re.search(rf"\b{key}='([+-]?[0-9.eE]+)mm'", block)
            if not match:
                raise ValueError(f"Missing {key} in FeedNbr_{name.group(1)}")
            values[key] = float(match.group(1))
        if abs(values["XSize"] - HALO_SIZE_MM) > 1.0e-9:
            raise ValueError(f"Unexpected FeedNbr X size: {values['XSize']}")
        if abs(values["YSize"] - HALO_SIZE_MM) > 1.0e-9:
            raise ValueError(f"Unexpected FeedNbr Y size: {values['YSize']}")
        if abs(values["ZSize"] - DEPTH_MM) > 1.0e-9:
            raise ValueError(f"Unexpected FeedNbr depth: {values['ZSize']}")
        result.append(
            (
                int(name.group(1)),
                values["XPosition"] + values["XSize"] / 2.0,
                values["YPosition"] + values["YSize"] / 2.0,
                values["ZPosition"],
                values["ZSize"],
            )
        )
    result.sort()
    if [row[0] for row in result] != list(range(NPORTS)):
        raise ValueError(f"Expected {NPORTS} FeedNbr objects, found {len(result)}")
    return result


def core_box(index: int, x: float, y: float, z: float, depth: float) -> str:
    half = CORE_SIZE_MM / 2.0
    return f'''oEditor.CreateBox Array( _
    "NAME:BoxParameters", "XPosition:=", "{x - half:.9g}mm", _
    "YPosition:=", "{y - half:.9g}mm", "ZPosition:=", "{z:.9g}mm", _
    "XSize:=", "{CORE_SIZE_MM:.9g}mm", "YSize:=", "{CORE_SIZE_MM:.9g}mm", _
    "ZSize:=", "{depth:.9g}mm"), _
    Array("NAME:Attributes", "Name:=", "FeedCore_{index:03d}", "Flags:=", "", _
    "Color:=", "(230 180 60)", "Transparency:=", 0.65, _
    "PartCoordinateSystem:=", "Global", "UDMId:=", "", _
    "MaterialValue:=", Chr(34) & "RO5880_Custom" & Chr(34), _
    "SurfaceMaterialValue:=", Chr(34) & Chr(34), "SolveInside:=", True, _
    "ShellElement:=", False, "IsMaterialEditable:=", True, _
    "UseMaterialAppearance:=", False, "IsLightweight:=", False)
'''


def build_vbs(project: Path, centers: list[tuple[int, float, float, float, float]]) -> str:
    boxes = "".join(core_box(*row) for row in centers)
    core_csv = ",".join(f"FeedCore_{index:03d}" for index in range(NPORTS))
    halo_array = ", ".join(f'"FeedNbr_{index:03d}"' for index in range(NPORTS))
    core_array = ", ".join(f'"FeedCore_{index:03d}"' for index in range(NPORTS))
    subtractions = "".join(
        f'''oEditor.Subtract Array("NAME:Selections", _
    "Blank Parts:=", "FeedNbr_{index:03d}", "Tool Parts:=", "FeedCore_{index:03d}"), _
    Array("NAME:SubtractParameters", "KeepOriginals:=", True)
'''
        for index in range(NPORTS)
    )
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oEditor, oMesh
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{PROJECT_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
Set oMesh = oDesign.GetModule("MeshSetup")
On Error Resume Next
oMesh.DeleteOp Array("{OLD_OP}")
oMesh.DeleteOp Array("{CORE_OP}")
oMesh.DeleteOp Array("{HALO_OP}")
oEditor.Delete Array("NAME:Selections", "Selections:=", "{core_csv}")
On Error GoTo 0
{boxes}
{subtractions}
oMesh.AssignLengthOp Array("NAME:{CORE_OP}", _
    "RefineInside:=", True, "Enabled:=", True, _
    "Objects:=", Array({core_array}), "RestrictElem:=", False, _
    "NumMaxElem:=", "1000", "RestrictLength:=", True, _
    "MaxLength:=", "{CORE_MESH_MM:.6f}mm", "UseAdvSizing:=", False)
oMesh.AssignLengthOp Array("NAME:{HALO_OP}", _
    "RefineInside:=", True, "Enabled:=", True, _
    "Objects:=", Array({halo_array}), "RestrictElem:=", False, _
    "NumMaxElem:=", "1000", "RestrictLength:=", True, _
    "MaxLength:=", "{HALO_MESH_MM:.6f}mm", "UseAdvSizing:=", False)
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
    centers = feed_centers(source_text)
    project_dir = OUT_RUN / PROJECT_NAME
    project_dir.mkdir(parents=True)
    project = project_dir / f"{PROJECT_NAME}.aedt"
    shutil.copy2(source_project, project)
    reference_dir = OUT_RUN / "reference_pass05"
    reference_dir.mkdir()
    for path in (SOURCE_RUN / "reference_pass05").glob("*"):
        if path.is_file():
            shutil.copy2(path, reference_dir / path.name)

    vbs = project_dir / "apply_layered_feedmesh.vbs"
    log = project_dir / "apply_layered_feedmesh.log"
    vbs.write_text(build_vbs(project, centers), encoding="ascii")
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
    source_perfecte = blocks(
        source_text, r"\$begin 'PEC_GroundPatch_Sheets'.*?\$end 'PEC_GroundPatch_Sheets'"
    )
    output_perfecte = blocks(
        output_text, r"\$begin 'PEC_GroundPatch_Sheets'.*?\$end 'PEC_GroundPatch_Sheets'"
    )
    cores = sorted(set(re.findall(r"Name='FeedCore_(\d{3})'", output_text)))
    halos = sorted(set(re.findall(r"Name='FeedNbr_(\d{3})'", output_text)))
    core_mesh = re.search(rf"\$begin '{CORE_OP}'(.*?)\$end '{CORE_OP}'", output_text, re.DOTALL)
    halo_mesh = re.search(rf"\$begin '{HALO_OP}'(.*?)\$end '{HALO_OP}'", output_text, re.DOTALL)
    port_hash_unchanged = digest(source_ports) == digest(output_ports)
    perfecte_hash_unchanged = digest(source_perfecte) == digest(output_perfecte)
    expected_ids = [f"{index:03d}" for index in range(NPORTS)]
    passed = bool(
        result.returncode == 0
        and len(source_ports) == len(output_ports) == NPORTS
        and port_hash_unchanged
        and perfecte_hash_unchanged
        and cores == halos == expected_ids
        and core_mesh
        and halo_mesh
        and "RefineInside=true" in core_mesh.group(1)
        and "MaxLength='0.21mm'" in core_mesh.group(1)
        and "RefineInside=true" in halo_mesh.group(1)
        and "MaxLength='0.28mm'" in halo_mesh.group(1)
        and "PortFeedUniform_0p180mm" in output_text
        and OLD_OP not in output_text
        and project.stat().st_size < 50_000_000
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source_project),
        "output_project": str(project),
        "apply_return_code": int(result.returncode),
        "port_count": len(output_ports),
        "port_definition_hash_unchanged": port_hash_unchanged,
        "perfect_e_boundary_hash_unchanged": perfecte_hash_unchanged,
        "feed_core_count": len(cores),
        "feed_halo_count": len(halos),
        "core_region_mm": [CORE_SIZE_MM, CORE_SIZE_MM, DEPTH_MM],
        "core_mesh_mm": CORE_MESH_MM,
        "halo_region_mm": [HALO_SIZE_MM, HALO_SIZE_MM, DEPTH_MM],
        "halo_mesh_mm": HALO_MESH_MM,
        "port_surface_mesh_mm": 0.18,
        "old_full_depth_0p18_op_removed": OLD_OP not in output_text,
        "project_size_bytes": project.stat().st_size,
        "project_size_gate_bytes": 50_000_000,
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
