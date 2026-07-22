#!/usr/bin/env python3
"""Build a clean 16x16 branch with full-depth deterministic feed mesh regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from prepare_16x16_feed_neighborhood_mesh import NPORTS, port_centers, vp


ROOT = Path(__file__).resolve().parents[1]
ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
MESH_OP = "FeedNeighborhoodUniform_0p180mm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mesh-mm", type=float, default=0.18)
    parser.add_argument("--region-mm", type=float, default=0.8)
    parser.add_argument("--ansys-exe", type=Path, default=ANSYS)
    return parser.parse_args()


def box_block(index: int, x: float, y: float, z: float, height: float, size: float) -> str:
    half = size / 2.0
    return f'''oEditor.CreateBox Array( _
    "NAME:BoxParameters", "XPosition:=", "{x - half:.9g}mm", _
    "YPosition:=", "{y - half:.9g}mm", "ZPosition:=", "{z:.9g}mm", _
    "XSize:=", "{size:.9g}mm", "YSize:=", "{size:.9g}mm", _
    "ZSize:=", "{height:.9g}mm"), _
    Array("NAME:Attributes", "Name:=", "FeedNbr_{index:03d}", "Flags:=", "", _
    "Color:=", "(80 180 120)", "Transparency:=", 0.75, _
    "PartCoordinateSystem:=", "Global", "UDMId:=", "", _
    "MaterialValue:=", Chr(34) & "RO5880_Custom" & Chr(34), _
    "SurfaceMaterialValue:=", Chr(34) & Chr(34), "SolveInside:=", True, _
    "ShellElement:=", False, "IsMaterialEditable:=", True, _
    "UseMaterialAppearance:=", False, "IsLightweight:=", False)
'''


def build_vbs(project: Path, rows: list[tuple[int, float, float, float, float]], mesh_mm: float, region_mm: float) -> str:
    boxes = "".join(box_block(index, x, y, z, height, region_mm) for index, x, y, z, height in rows)
    old_names = ",".join(f"FeedSeed_{index:03d}" for index in range(NPORTS))
    new_names = ",".join(f"FeedNbr_{index:03d}" for index in range(NPORTS))
    object_array = ", ".join(f'"FeedNbr_{index:03d}"' for index in range(NPORTS))
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
oMesh.DeleteOp Array("FeedSheetUniform_0p180mm")
oMesh.DeleteOp Array("{MESH_OP}")
oEditor.Delete Array("NAME:Selections", "Selections:=", "{old_names}")
On Error GoTo 0
{boxes}
oEditor.Subtract Array("NAME:Selections", "Blank Parts:=", "Substrate", _
    "Tool Parts:=", "{new_names}"), _
    Array("NAME:SubtractParameters", "KeepOriginals:=", True)
oMesh.AssignLengthOp Array("NAME:{MESH_OP}", _
    "RefineInside:=", True, "Enabled:=", True, _
    "Objects:=", Array({object_array}), "RestrictElem:=", False, _
    "NumMaxElem:=", "1000", "RestrictLength:=", True, _
    "MaxLength:=", "{mesh_mm:.6f}mm", "UseAdvSizing:=", False)
On Error Resume Next
oDesign.DeleteFullVariation "All", False
On Error GoTo 0
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def blocks(text: str, pattern: str) -> list[str]:
    return [match.group(0) for match in re.finditer(pattern, text, re.DOTALL)]


def digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(out_dir)
    if abs(args.mesh_mm - 0.18) > 1.0e-12 or abs(args.region_mm - 0.8) > 1.0e-12:
        raise ValueError("This branch is locked to 0.18 mm mesh and 0.8 mm regions")
    source_project = source_run / PROJECT_NAME / f"{PROJECT_NAME}.aedt"
    source_text = source_project.read_text(encoding="utf-8", errors="ignore")
    rows = port_centers(source_text)
    project_dir = out_dir / PROJECT_NAME
    project_dir.mkdir(parents=True)
    project = project_dir / f"{PROJECT_NAME}.aedt"
    shutil.copy2(source_project, project)
    reference = out_dir / "reference_pass05"
    reference.mkdir()
    for path in (
        source_run / "stages" / "pass05" / "stage_metrics.json",
        source_run / "stages" / "pass05" / "grounded_patch_16x16_ddm_pass05.s256p",
        source_run / "stages" / "pass05" / "DV1332_S2193_V2409.profile",
    ):
        if path.exists():
            shutil.copy2(path, reference / path.name)
    vbs = project_dir / "apply_volumetric_feedmesh.vbs"
    log = project_dir / "apply_volumetric_feedmesh.log"
    vbs.write_text(build_vbs(project, rows, args.mesh_mm, args.region_mm), encoding="ascii")
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(
            [str(args.ansys_exe), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    output_text = project.read_text(encoding="utf-8", errors="ignore")
    port_pattern = r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'"
    source_ports = blocks(source_text, port_pattern)
    output_ports = blocks(output_text, port_pattern)
    protected_names = ["Ground", *[f"Patch_{index:03d}" for index in range(NPORTS)]]
    protected_unchanged = all(
        f"Name='{name}'" in source_text and f"Name='{name}'" in output_text
        for name in protected_names
    )
    feed_nbrs = sorted(set(re.findall(r"Name='FeedNbr_(\d{3})'", output_text)))
    feed_seeds = sorted(set(re.findall(r"Name='FeedSeed_(\d{3})'", output_text)))
    mesh_block = re.search(rf"\$begin '{MESH_OP}'(.*?)\$end '{MESH_OP}'", output_text, re.DOTALL)
    passed = bool(
        result.returncode == 0
        and len(source_ports) == len(output_ports) == NPORTS
        and digest(source_ports) == digest(output_ports)
        and feed_nbrs == [f"{index:03d}" for index in range(NPORTS)]
        and not feed_seeds
        and mesh_block is not None
        and "RefineInside=true" in mesh_block.group(1)
        and "MaxLength='0.18mm'" in mesh_block.group(1)
        and "PortFeedUniform_0p180mm" in output_text
        and "FeedSheetUniform_0p180mm" not in output_text
        and protected_unchanged
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source_project),
        "output_project": str(project),
        "apply_return_code": int(result.returncode),
        "source_port_count": len(source_ports),
        "output_port_count": len(output_ports),
        "port_definition_hash_unchanged": digest(source_ports) == digest(output_ports),
        "feed_seed_count": len(feed_seeds),
        "volumetric_feed_region_count": len(feed_nbrs),
        "region_xy_mm": float(args.region_mm),
        "region_z_mm": float(rows[0][4]),
        "mesh_mm": float(args.mesh_mm),
        "refine_inside": bool(mesh_block and "RefineInside=true" in mesh_block.group(1)),
        "port_mesh_preserved": "PortFeedUniform_0p180mm" in output_text,
        "ground_and_patch_names_preserved": protected_unchanged,
        "configuration_smoke_pass": passed,
        "training_labels_locked": True,
        "decision": "allow_initial_mesh_smoke" if passed else "block_initial_mesh_smoke",
    }
    (out_dir / "volumetric_feedmesh_prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
