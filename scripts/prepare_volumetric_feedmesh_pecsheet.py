#!/usr/bin/env python3
"""Create an isolated volumetric-feed branch with zero-thickness PEC conductors."""

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
    / "grounded_patch_direct_16x16_volumetric_feedmesh_20260722_run01"
)
OUT_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_20260722_run01"
)
ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
NPORTS = 256
MESH_OP = "FeedNeighborhoodUniform_0p180mm"


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def geometry_parts(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for block in re.findall(
        r"\$begin 'GeometryPart'.*?\$end 'GeometryPart'", text, re.DOTALL
    ):
        name = re.search(r"\bName='([^']+)'", block)
        if name:
            result[name.group(1)] = block
    return result


def box_parameters(block: str) -> dict[str, float]:
    keys = ("XPosition", "YPosition", "ZPosition", "XSize", "YSize", "ZSize")
    values: dict[str, float] = {}
    for key in keys:
        match = re.search(rf"\b{key}='([+-]?[0-9.eE]+)mm'", block)
        if not match:
            raise ValueError(f"Missing {key} in geometry block")
        values[key] = float(match.group(1))
    return values


def rectangle(
    name: str, x: float, y: float, z: float, width: float, height: float
) -> str:
    return f'''oEditor.CreateRectangle Array( _
    "NAME:RectangleParameters", "IsCovered:=", True, _
    "XStart:=", "{x:.9g}mm", "YStart:=", "{y:.9g}mm", _
    "ZStart:=", "{z:.9g}mm", "Width:=", "{width:.9g}mm", _
    "Height:=", "{height:.9g}mm", "WhichAxis:=", "Z"), _
    Array("NAME:Attributes", "Name:=", "{name}", "Flags:=", "", _
    "Color:=", "(120 150 170)", "Transparency:=", 0.65, _
    "PartCoordinateSystem:=", "Global", "UDMId:=", "", _
    "MaterialValue:=", Chr(34) & "pec" & Chr(34), _
    "SurfaceMaterialValue:=", Chr(34) & Chr(34), "SolveInside:=", False, _
    "ShellElement:=", False, "IsMaterialEditable:=", True, _
    "UseMaterialAppearance:=", False, "IsLightweight:=", False)
'''


def digest(blocks: list[str]) -> str:
    return hashlib.sha256("\n".join(blocks).encode("utf-8")).hexdigest()


def named_blocks(text: str, pattern: str) -> list[str]:
    return [m.group(0) for m in re.finditer(pattern, text, re.DOTALL)]


def build_vbs(project: Path, source_text: str) -> str:
    parts = geometry_parts(source_text)
    names = ["Ground", *[f"Patch_{i:03d}" for i in range(NPORTS)]]
    missing = [name for name in names if name not in parts]
    if missing:
        raise ValueError(f"Missing conductor objects: {missing[:5]}")

    ground = box_parameters(parts["Ground"])
    if abs(ground["ZSize"] - 0.035) > 1.0e-9:
        raise ValueError(f"Unexpected Ground thickness: {ground['ZSize']}")
    # Preserve the conductor/substrate interface, not the outer metal surface.
    ground_z = ground["ZPosition"] + ground["ZSize"]
    commands = [
        rectangle(
            "Ground",
            ground["XPosition"],
            ground["YPosition"],
            ground_z,
            ground["XSize"],
            ground["YSize"],
        )
    ]
    for index in range(NPORTS):
        name = f"Patch_{index:03d}"
        patch = box_parameters(parts[name])
        if abs(patch["ZSize"] - 0.035) > 1.0e-9:
            raise ValueError(f"Unexpected {name} thickness: {patch['ZSize']}")
        commands.append(
            rectangle(
                name,
                patch["XPosition"],
                patch["YPosition"],
                patch["ZPosition"],
                patch["XSize"],
                patch["YSize"],
            )
        )
    selections = ",".join(names)
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oEditor
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{PROJECT_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.Delete Array("NAME:Selections", "Selections:=", "{selections}")
{''.join(commands)}
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
    resume_validation = bool(
        project.exists()
        and not project.with_suffix(".aedt.lock").exists()
        and not (OUT_RUN / "pecsheet_prepare_summary.json").exists()
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

        vbs = project_dir / "convert_ground_patch_to_pec_sheets.vbs"
        log = project_dir / "convert_ground_patch_to_pec_sheets.log"
        vbs.write_text(build_vbs(project, source_text), encoding="ascii")
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
    source_ports = named_blocks(source_text, r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'")
    output_ports = named_blocks(output_text, r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'")
    source_parts = geometry_parts(source_text)
    output_parts = geometry_parts(output_text)
    conductor_names = ["Ground", *[f"Patch_{i:03d}" for i in range(NPORTS)]]
    conductor_types = {
        name: re.search(r"OperationType='([^']+)'", output_parts.get(name, "")).group(1)
        if re.search(r"OperationType='([^']+)'", output_parts.get(name, ""))
        else None
        for name in conductor_names
    }
    feed_regions = sorted(set(re.findall(r"Name='FeedNbr_(\d{3})'", output_text)))
    mesh = re.search(rf"\$begin '{MESH_OP}'(.*?)\$end '{MESH_OP}'", output_text, re.DOTALL)
    port_hash_unchanged = digest(source_ports) == digest(output_ports)
    all_sheets = all(value == "Rectangle" for value in conductor_types.values())
    passed = bool(
        apply_return_code == 0
        and len(source_ports) == len(output_ports) == NPORTS
        and port_hash_unchanged
        and len(feed_regions) == NPORTS
        and mesh
        and "RefineInside=true" in mesh.group(1)
        and "MaxLength='0.18mm'" in mesh.group(1)
        and "PortFeedUniform_0p180mm" in output_text
        and all_sheets
        and all(name in source_parts for name in conductor_names)
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source_project),
        "output_project": str(project),
        "apply_return_code": apply_return_code,
        "resumed_validation_after_external_timeout": resume_validation,
        "port_count": len(output_ports),
        "port_definition_hash_unchanged": port_hash_unchanged,
        "feed_neighborhood_count": len(feed_regions),
        "feed_neighborhood_mesh_mm": 0.18,
        "feed_neighborhood_refine_inside": bool(mesh and "RefineInside=true" in mesh.group(1)),
        "ground_sheet_z_mm": ground_z_from_text(output_parts.get("Ground", "")),
        "pec_sheet_conductor_count": sum(value == "Rectangle" for value in conductor_types.values()),
        "ground_patch_all_zero_thickness_sheets": all_sheets,
        "configuration_smoke_pass": passed,
        "training_labels_locked": True,
        "decision": "allow_initial_mesh_smoke" if passed else "block_initial_mesh_smoke",
    }
    (OUT_RUN / "pecsheet_prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


def ground_z_from_text(block: str) -> float | None:
    match = re.search(r"\bZStart='([+-]?[0-9.eE]+)mm'", block)
    return float(match.group(1)) if match else None


if __name__ == "__main__":
    main()
