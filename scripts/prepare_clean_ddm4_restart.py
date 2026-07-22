#!/usr/bin/env python3
"""Prepare a clean DDM4 branch without changing array geometry or port definitions."""

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
    / "grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02"
)
OUT_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_clean_ddm4_surfacefeed_20260722_run01"
)
ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
PROJECT_NAME = "grounded_patch_16x16"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
NPORTS = 256


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def blocks(text: str, pattern: str) -> list[str]:
    return [match.group(0) for match in re.finditer(pattern, text, re.DOTALL)]


def digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def cleanup_vbs(project: Path) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oAnalysis
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{PROJECT_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
On Error Resume Next
oAnalysis.DeleteSetups Array("Setup_DDM_Recovery")
oAnalysis.DeleteSetups Array("Setup_VolFeed_DDM")
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
    summary_path = OUT_RUN / "clean_ddm_restart_prepare_summary.json"
    retry_blocked_cleanup = bool(
        project.exists()
        and summary_path.exists()
        and not json.loads(summary_path.read_text(encoding="utf-8")).get(
            "configuration_smoke_pass"
        )
        and not project.with_suffix(".aedt.lock").exists()
    )
    if not retry_blocked_cleanup:
        if OUT_RUN.exists() and any(OUT_RUN.iterdir()):
            raise FileExistsError(OUT_RUN)
        project_dir.mkdir(parents=True)
        shutil.copy2(source_project, project)

        reference_dir = OUT_RUN / "reference_pass05"
        reference_dir.mkdir()
        for path in (SOURCE_RUN / "stages" / "pass05").glob("*"):
            if path.is_file():
                shutil.copy2(path, reference_dir / path.name)

    vbs = project_dir / "clear_history_for_clean_ddm4.vbs"
    log = project_dir / "clear_history_for_clean_ddm4.log"
    vbs.write_text(cleanup_vbs(project), encoding="ascii")
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(
            [str(ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    output_text = project.read_text(encoding="utf-8", errors="ignore")
    port_pattern = r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'"
    source_ports = blocks(source_text, port_pattern)
    output_ports = blocks(output_text, port_pattern)
    source_mesh = {
        name: blocks(source_text, rf"\$begin '{name}'.*?\$end '{name}'")
        for name in ("PortFeedUniform_0p180mm", "FeedSheetUniform_0p180mm")
    }
    output_mesh = {
        name: blocks(output_text, rf"\$begin '{name}'.*?\$end '{name}'")
        for name in source_mesh
    }
    feed_seeds = sorted(set(re.findall(r"Name='FeedSeed_(\d{3})'", output_text)))
    passed = bool(
        result.returncode == 0
        and len(source_ports) == len(output_ports) == NPORTS
        and digest(source_ports) == digest(output_ports)
        and all(digest(source_mesh[name]) == digest(output_mesh[name]) for name in source_mesh)
        and feed_seeds == [f"{index:03d}" for index in range(NPORTS)]
        and "FeedNeighborhoodUniform_0p180mm" not in output_text
        and "Setup_DDM_Recovery" not in output_text
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration_mode": "clean_surface_mesh_ddm4_restart",
        "source_project": str(source_project),
        "output_project": str(project),
        "apply_return_code": int(result.returncode),
        "retried_blocked_cleanup": retry_blocked_cleanup,
        "port_count": len(output_ports),
        "port_definition_hash_unchanged": digest(source_ports) == digest(output_ports),
        "surface_mesh_hash_unchanged": all(
            digest(source_mesh[name]) == digest(output_mesh[name]) for name in source_mesh
        ),
        "feed_seed_count": len(feed_seeds),
        "volumetric_feed_region_count": 0,
        "port_surface_mesh_mm": 0.18,
        "percent_refinement": 5.0,
        "ddm_tasks": 4,
        "requested_cores": 4,
        "max_total_domain_memory_gb": 18.5,
        "max_matrix_size": 5_800_000,
        "old_meshlink_setup_removed": "Setup_DDM_Recovery" not in output_text,
        "configuration_smoke_pass": passed,
        "training_labels_locked": True,
        "decision": "allow_clean_ddm4_smoke" if passed else "block_clean_ddm4_smoke",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
