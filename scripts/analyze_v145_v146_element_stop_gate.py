#!/usr/bin/env python3
"""Aggregate the v1.45/v1.46 physical 1x1 element stop gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "hfss_outputs" / "v146_element_topology_stop_gate_20260810"
BASELINE = ROOT / "baselines" / "2026-08-10-v146-aperture-balanced-element-stop-gate"
SOURCES = [
    ("aperture_coupled", "v145_run03_center", "hfss_outputs/v145_aperture_coupled_patch_20260810_run03_center/one_by_one_prescreen/one_by_one_metrics.csv"),
    ("aperture_coupled", "v145_run04_doe", "hfss_outputs/v145_aperture_coupled_patch_20260810_run04_doe/one_by_one_prescreen/one_by_one_metrics.csv"),
    ("aperture_coupled", "v145_run05_strong_aperture", "hfss_outputs/v145_aperture_coupled_patch_20260810_run05_strong_aperture/one_by_one_prescreen/one_by_one_metrics.csv"),
    ("aperture_coupled_transformer", "v145_run06_transformer", "hfss_outputs/v145_aperture_coupled_patch_20260810_run06_transformer/one_by_one_prescreen/one_by_one_metrics.csv"),
    ("aperture_coupled_transformer", "v145_run07_isolated_transformer", "hfss_outputs/v145_aperture_coupled_patch_20260810_run07_isolated_transformer/one_by_one_prescreen/one_by_one_metrics.csv"),
    ("ground_backed_differential", "v146_run01", "hfss_outputs/v146_ground_backed_differential_patch_20260810_run01/one_by_one_prescreen/one_by_one_metrics.csv"),
    ("ground_backed_differential", "v146_run02_low_profile", "hfss_outputs/v146_ground_backed_differential_patch_20260810_run02_low_profile/one_by_one_prescreen/one_by_one_metrics.csv"),
]
ABORTS = [
    "hfss_outputs/v145_aperture_coupled_patch_20260810_run01/one_by_one_prescreen/acp00_center/run_audit.json",
    "hfss_outputs/v145_aperture_coupled_patch_20260810_run02_sheet/one_by_one_prescreen/acp00_center/run_audit.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def truth(value: str) -> bool:
    return str(value).strip().lower() == "true"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def branch_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    complete = [
        row
        for row in rows
        if truth(row.get("converged", ""))
        and int(float(row.get("solve_return_code", "-1"))) == 0
        and truth(row.get("touchstone_exists", ""))
    ]
    best = max(complete, key=lambda row: as_float(row, "passive_rl_db"))
    return {
        "complete_case_count": len(complete),
        "numerical_gate_count": sum(
            as_float(row, "final_delta_s") <= 0.05
            and int(float(row.get("topology_warning_count", "999"))) == 0
            for row in complete
        ),
        "efficiency_gate_count": sum(
            as_float(row, "radiation_efficiency") >= 0.95 for row in complete
        ),
        "passive_rl_10db_count": sum(
            as_float(row, "passive_rl_db") >= 10.0 for row in complete
        ),
        "passive_rl_15db_count": sum(
            as_float(row, "passive_rl_db") >= 15.0 for row in complete
        ),
        "best_case_id": best["case_id"],
        "best_passive_rl_db": as_float(best, "passive_rl_db"),
        "best_input_resistance_ohm": as_float(best, "input_resistance_ohm"),
        "best_input_reactance_ohm": as_float(best, "input_reactance_ohm"),
        "best_radiation_efficiency": as_float(best, "radiation_efficiency"),
        "best_final_delta_s": as_float(best, "final_delta_s"),
        "minimum_observed_free_memory_gib": min(
            as_float(row, "minimum_free_memory_gib") for row in complete
        ),
        "maximum_tetrahedra": int(
            max(as_float(row, "maximum_tetrahedra") for row in complete)
        ),
    }


def main() -> None:
    for directory in (OUTPUT, BASELINE):
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
    OUTPUT.mkdir(parents=True)
    snapshots = BASELINE / "snapshots"
    snapshots.mkdir(parents=True)

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = ["branch", "source_run"]
    for branch, source_run, relative in SOURCES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for source in reader:
                rows.append({"branch": branch, "source_run": source_run, **source})

    combined = OUTPUT / "complete_1x1_metrics.csv"
    with combined.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_branch = {
        branch: branch_summary([row for row in rows if row["branch"] == branch])
        for branch in sorted({row["branch"] for row in rows})
    }
    topology = {
        "aperture_family": branch_summary(
            [row for row in rows if row["branch"].startswith("aperture_coupled")]
        ),
        "ground_backed_differential_family": branch_summary(
            [row for row in rows if row["branch"] == "ground_backed_differential"]
        ),
    }
    aborts = []
    for relative in ABORTS:
        path = ROOT / relative
        if path.exists():
            aborts.append({"source": relative, **read_json(path)})

    summary = {
        "stage": "v1.45_v1.46_element_topology_stop_gate",
        "evidence_level": "B_physical_10ghz_1x1_only",
        "complete_case_count": len(rows),
        "branch_summary": by_branch,
        "topology_summary": topology,
        "memory_abort_events": aborts,
        "gate": {
            "minimum_1x1_passive_rl_db": 15.0,
            "minimum_radiation_efficiency": 0.95,
            "maximum_final_delta_s": 0.05,
            "maximum_topology_warning_count": 0,
        },
        "one_by_one_gate_pass_count": sum(
            truth(row.get("one_by_one_gate_pass", "")) for row in rows
        ),
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "decision": (
            "Stop both the standard aperture-coupled/single-transformer family "
            "and the ground-backed true-differential split-patch family before "
            "2x2. Neither family forms a 15 dB physical 1x1 feasible set."
        ),
    }
    write_json(OUTPUT / "stage_summary.json", summary)
    write_json(OUTPUT / "stage_decision.json", summary)

    shutil.copy2(combined, snapshots / combined.name)
    shutil.copy2(OUTPUT / "stage_summary.json", snapshots / "stage_summary.json")
    write_json(
        BASELINE / "baseline_metadata.json",
        {
            "baseline_id": "v1.46.0-aperture-balanced-element-stop-gate",
            "created_on": "2026-08-10",
            "parent_commit": "0a61925f1f5b314809fd7b30bdf83f8a4a3487b5",
            "evidence_level": summary["evidence_level"],
            "raw_output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        },
    )
    manifest = []
    for path in sorted(BASELINE.rglob("*")):
        if path.is_file():
            manifest.append(
                {
                    "path": str(path.relative_to(BASELINE)).replace("\\", "/"),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_json(BASELINE / "artifact_manifest.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
