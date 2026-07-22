#!/usr/bin/env python3
"""Copy the validated Perfect E sheet model into an isolated DDM6 branch."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_perfecte_20260722_run01"
)
OUT_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_perfecte_ddm6_20260722_run01"
)
PROJECT_NAME = "grounded_patch_16x16"
NPORTS = 256


def digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def port_blocks(text: str) -> list[str]:
    return [
        match.group(0)
        for match in re.finditer(r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'", text, re.DOTALL)
    ]


def main() -> None:
    if OUT_RUN.exists() and any(OUT_RUN.iterdir()):
        raise FileExistsError(OUT_RUN)
    source_project = SOURCE_RUN / PROJECT_NAME / f"{PROJECT_NAME}.aedt"
    source_text = source_project.read_text(encoding="utf-8", errors="ignore")
    project_dir = OUT_RUN / PROJECT_NAME
    project_dir.mkdir(parents=True)
    project = project_dir / f"{PROJECT_NAME}.aedt"
    shutil.copy2(source_project, project)
    output_text = project.read_text(encoding="utf-8", errors="ignore")

    reference_dir = OUT_RUN / "reference_pass05"
    reference_dir.mkdir()
    for path in (SOURCE_RUN / "reference_pass05").glob("*"):
        if path.is_file():
            shutil.copy2(path, reference_dir / path.name)

    source_ports = port_blocks(source_text)
    output_ports = port_blocks(output_text)
    feed_regions = sorted(set(re.findall(r"Name='FeedNbr_(\d{3})'", output_text)))
    passed = bool(
        len(source_ports) == len(output_ports) == NPORTS
        and digest(source_ports) == digest(output_ports)
        and len(feed_regions) == NPORTS
        and "FeedNeighborhoodUniform_0p180mm" in output_text
        and "PortFeedUniform_0p180mm" in output_text
        and "$begin 'PEC_GroundPatch_Sheets'" in output_text
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_project": str(source_project),
        "output_project": str(project),
        "port_count": len(output_ports),
        "port_definition_hash_unchanged": digest(source_ports) == digest(output_ports),
        "feed_neighborhood_count": len(feed_regions),
        "perfect_e_boundary_present": "$begin 'PEC_GroundPatch_Sheets'" in output_text,
        "requested_ddm_tasks": 6,
        "requested_cores": 12,
        "configuration_smoke_pass": passed,
        "training_labels_locked": True,
        "decision": "allow_ddm6_resource_smoke" if passed else "block_ddm6_resource_smoke",
    }
    (OUT_RUN / "ddm_domain_prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
