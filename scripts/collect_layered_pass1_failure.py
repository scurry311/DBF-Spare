#!/usr/bin/env python3
"""Preserve the detached layered pass-1 late-OOM evidence and lock labels."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_layered_feedmesh_c021_h028_20260722_run02"
)


def number(pattern: str, text: str, cast=int):
    match = re.search(pattern, text, re.DOTALL)
    return cast(match.group(1)) if match else None


def main() -> None:
    stage = RUN / "stages" / "pass01_failed_late_oom"
    if stage.exists() and any(stage.iterdir()):
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    profiles = list((RUN / "grounded_patch_16x16").rglob("*.profile"))
    profile = max(profiles, key=lambda path: path.stat().st_mtime_ns)
    messages = RUN / "volfeed_pass01_messages.txt"
    profile_text = profile.read_text(encoding="utf-8", errors="ignore").replace("\\'", "'")
    message_text = messages.read_text(encoding="utf-8", errors="ignore")
    shutil.copy2(profile, stage / profile.name)
    shutil.copy2(messages, stage / messages.name)

    zero_ports = sorted(set(re.findall(r"'0' conductors touch lumped port 'P(\d{3})'", message_text)))
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "pass01_failed_late_oom",
        "normal_completion": "Normal Completion" in profile_text,
        "elapsed_time": number(r"Elapsed Time',\s*'([^']+)'", profile_text, str),
        "manual_refine_tetrahedra": number(
            r"ProfileItem\('Manual Refine'.*?'Tetrahedra',\s*(\d+)", profile_text
        ),
        "port_refine_tetrahedra": number(
            r"ProfileItem\('Port Refine'.*?'Tetrahedra',\s*(\d+)", profile_text
        ),
        "ddm_domains": number(r"'Domain',\s*(\d+)", profile_text),
        "ddm_tasks": 6,
        "matrix_size_observed_before_cleanup": 1271954,
        "solver_memory_estimate_gb_observed_before_cleanup": 20.34575,
        "solver_memory_observation_source": "temporary solverUsagefile.txt inspected before AEDT cleanup",
        "zero_conductor_unique_ports": len(zero_ports),
        "late_oom_message_present": "Out of memory" in message_text,
        "touchstone_generated": False,
        "small_segment_report_available": False,
        "small_segment_gate_pass": None,
        "numerical_smatrix_valid": None,
        "reference_delta_s_gate_pass": None,
        "reference_rl_delta_gate_pass": None,
        "training_labels_locked": True,
        "decision": "stop_pass01_late_oom_relax_layered_mesh_within_allowed_range",
    }
    (stage / "failure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    status_path = RUN / "volumetric_ddm_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": "stopped_pass01_late_oom",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pass01_failure": summary,
            "training_labels_locked": True,
        }
    )
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
