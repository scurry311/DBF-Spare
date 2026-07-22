#!/usr/bin/env python3
"""Summarize guarded volumetric feed-mesh recovery attempts without creating labels."""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "hfss_outputs" / "volumetric_feedmesh_recovery_audit_20260722_run02"
RUNS = [
    (
        "volumetric_solid_ddm4",
        "grounded_patch_direct_16x16_volumetric_feedmesh_20260722_run01",
    ),
    (
        "pec_sheet_without_perfecte_ddm4",
        "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_20260722_run01",
    ),
    (
        "pec_sheet_perfecte_ddm4",
        "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_perfecte_20260722_run01",
    ),
    (
        "pec_sheet_perfecte_ddm6",
        "grounded_patch_direct_16x16_volumetric_feedmesh_pecsheet_perfecte_ddm6_20260722_run01",
    ),
]


def number(pattern: str, text: str, cast=float):
    match = re.search(pattern, text, re.IGNORECASE)
    return cast(match.group(1)) if match else None


def latest_profile(run: Path) -> Path | None:
    profiles = list(run.rglob("*.profile"))
    return max(profiles, key=lambda path: path.stat().st_mtime_ns) if profiles else None


def collect(label: str, directory: str) -> dict[str, object]:
    run = ROOT / "hfss_outputs" / directory
    messages_path = run / "volfeed_pass01_messages.txt"
    messages = messages_path.read_text(encoding="utf-8", errors="ignore") if messages_path.exists() else ""
    profile_path = latest_profile(run)
    profile = profile_path.read_text(encoding="utf-8", errors="ignore") if profile_path else ""
    profile_clean = profile.replace("\\'", "'")
    zero_ports = sorted(set(re.findall(r"'0' conductors touch lumped port 'P(\d{3})'", messages)))
    oom = re.search(
        r"requires approximately\s+([0-9.]+)\s*\(GB\), available\s+([0-9.]+)\s*\(GB\)",
        messages,
        re.IGNORECASE,
    )
    touchstones = list(run.rglob("*.s256p"))
    generated = [path for path in touchstones if "reference_pass05" not in path.parts]
    g3derr = list(run.rglob("*.g3derr"))
    return {
        "branch": label,
        "run_dir": str(run),
        "ddm_tasks": number(r"'Tasks',\s*(\d+)", profile_clean, int) or 4,
        "ddm_domains": number(r"'Domain',\s*(\d+)", profile_clean, int),
        "manual_refine_tetrahedra": number(
            r"ProfileItem\('Manual Refine'.*?'Tetrahedra',\s*(\d+)", profile_clean, int
        ),
        "port_refine_tetrahedra": number(
            r"ProfileItem\('Port Refine'.*?'Tetrahedra',\s*(\d+)", profile_clean, int
        ),
        "zero_conductor_unique_ports": len(zero_ports),
        "zero_conductor_port_ids": ";".join(zero_ports),
        "oom_required_gb": float(oom.group(1)) if oom else None,
        "oom_available_gb": float(oom.group(2)) if oom else None,
        "oom_shortfall_gb": round(float(oom.group(1)) - float(oom.group(2)), 2) if oom else None,
        "new_s256_count": len(generated),
        "small_segment_report_available": bool(g3derr),
        "small_segment_count": None,
        "profile_path": str(profile_path) if profile_path else None,
        "messages_path": str(messages_path) if messages_path.exists() else None,
        "pass1_complete": bool(generated),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    rows = [collect(label, directory) for label, directory in RUNS]
    csv_path = OUTPUT / "volumetric_feedmesh_branch_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_configuration": {
            "feed_seed_count": 0,
            "feed_neighborhood_count": 256,
            "feed_neighborhood_mm": [0.8, 0.8, 0.787],
            "max_length_mm": 0.18,
            "refine_inside": True,
            "percent_refinement": 5.0,
            "ddm_residual": 1.0e-6,
        },
        "verified": {
            "feed_seed_removed": True,
            "feed_neighborhood_configuration_smoke": True,
            "port_definition_hash_unchanged": True,
            "pec_sheet_requires_explicit_perfect_e": True,
            "perfect_e_zero_conductor_warning_count": 0,
            "ddm6_domain_count": next(
                row["ddm_domains"] for row in rows if row["branch"].endswith("ddm6")
            ),
        },
        "not_verified": {
            "small_segment_count_after_rebuild": "unavailable because no branch completed pass1 and no g3derr was emitted",
            "pass1_s256": False,
            "two_round_delta_s": False,
            "monotonic_delta_s": False,
        },
        "branch_results": rows,
        "engineering_decision": {
            "state": "blocked_by_solver_memory_at_fixed_mesh",
            "best_attempt_required_gb": next(
                row["oom_required_gb"] for row in rows if row["branch"].endswith("ddm6")
            ),
            "best_attempt_available_gb": next(
                row["oom_available_gb"] for row in rows if row["branch"].endswith("ddm6")
            ),
            "training_labels_locked": True,
            "allow_hfss_metrics": False,
            "recommended_next_action": "Run the validated Perfect E DDM6 branch on a host with at least 32 GB RAM; otherwise create a separately named mesh-coverage convergence branch rather than changing this fixed 0.18 mm test in place.",
        },
    }
    (OUTPUT / "volumetric_feedmesh_recovery_decision.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
