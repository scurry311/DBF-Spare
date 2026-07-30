#!/usr/bin/env python3
"""Freeze the resource-safe v1.17 integrated physical 2x2 smoke evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "hfss_outputs" / "v117_integrated_2x2_smoke_20260730_run03"
CASE = SOURCE / "integrated_2x2_direct01"
BASELINE = ROOT / "baselines" / "2026-07-30-v117-integrated-2x2-smoke"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".s4p", ".s8p"}:
        lines = [line.rstrip() for line in source.read_text(encoding="utf-8").splitlines()]
        destination.write_text("\n".join(lines) + "\n", encoding="ascii")
    else:
        shutil.copy2(source, destination)


def write_manifest() -> None:
    rows = []
    for path in sorted(item for item in BASELINE.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": path.relative_to(BASELINE).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    with (BASELINE / "artifact_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if BASELINE.exists():
        raise FileExistsError(f"Refusing to overwrite baseline: {BASELINE}")
    analysis = read_json(CASE / "analysis.json")
    decision = read_json(SOURCE / "stage_decision.json")
    if not analysis.get("electromagnetic_solve_complete"):
        raise RuntimeError("The integrated electromagnetic solve and field recovery are incomplete")
    if decision.get("first_integrated_gate_pass"):
        raise RuntimeError("This baseline is intended to preserve the failed integrated gate")
    snapshots = BASELINE / "snapshots"
    snapshots.mkdir(parents=True)
    for source, destination in (
        (SOURCE / "config_snapshot.json", snapshots / "config_snapshot.json"),
        (SOURCE / "antenna_protocol_snapshot.json", snapshots / "antenna_protocol_snapshot.json"),
        (SOURCE / "stage_decision.json", snapshots / "stage_decision.json"),
        (SOURCE / "case_metrics.csv", snapshots / "case_metrics.csv"),
        (CASE / "analysis.json", snapshots / "analysis.json"),
        (CASE / "frequency_metrics.csv", snapshots / "frequency_metrics.csv"),
        (CASE / "stimulus_metrics.csv", snapshots / "stimulus_metrics.csv"),
        (CASE / "run_summary.json", snapshots / "run_summary.json"),
        (CASE / "field_export_summary.json", snapshots / "field_export_summary.json"),
        (CASE / "v117_integrated_2x2_direct01.s4p", snapshots / "integrated_direct01.s4p"),
    ):
        copy_artifact(source, destination)
    for path in sorted(CASE.glob("eep_*.csv")):
        copy_artifact(path, snapshots / "eep" / path.name)
    for path in sorted(CASE.glob("efficiency_*.csv")):
        copy_artifact(path, snapshots / "efficiency" / path.name)
    attempts = [
        {
            "run": "run01",
            "status": "stopped_before_solve",
            "reason": "intersecting fanout and united patch/probe conductors",
            "valid_label": 0,
        },
        {
            "run": "run02",
            "status": "memory_guard_manual_stop",
            "reason": "542245 tetrahedra and 15.05 GiB peak solver memory",
            "valid_label": 0,
        },
        {
            "run": "run03",
            "status": "electromagnetic_complete_gate_failed",
            "reason": "resource-safe solve; physical matching and cascade-consistency gates failed",
            "valid_label": 1,
        },
    ]
    with (snapshots / "development_attempts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=attempts[0].keys())
        writer.writeheader()
        writer.writerows(attempts)
    metadata = {
        "version": "v1.17.0-integrated-2x2-smoke",
        "created_on": "2026-07-30",
        "parent_tag": "v1.16.0-physical-s8-hfss-optimization",
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_output": str(SOURCE.resolve()),
        "evidence_level": "integrated physical feed-network and 2x2 antenna HFSS full-wave",
        "training_label_authorized": False,
    }
    write_json(BASELINE / "baseline_metadata.json", metadata)
    report = f"""# v1.17 Integrated Physical 2x2 Smoke

This baseline physically connects the v1.16 grounded x-modal network to four
dual-slot probe-fed patches. The compact network is placed below the shared
ground, followed by routed microstrip fanout and four physical probes. Only
the four source-facing PRE ports remain as excitations.

## Numerical Credibility

| Metric | Result | Gate |
|---|---:|---:|
| Adaptive passes | {analysis['pass_count']} | converged |
| Final Delta S | {analysis['final_delta_s']:.5f} | <= 0.05 |
| Tetrahedra | {analysis['maximum_tetrahedra']:,} | resource audit |
| Peak solver memory | {analysis['peak_solver_memory_gb']:.2f} GiB | resource audit |
| Reciprocity error | {analysis['reciprocity_error_max']:.3e} | <= 1e-4 |
| Passivity sigma | {analysis['passivity_sigma_max']:.3f} | <= 1.001 |
| EEP power relative error | {100*analysis['eep_power_relative_error_max']:.2f}% | <= 5% |

The electromagnetic solve completed normally. A postprocessing variable-name
error was recovered by reopening the saved fields without rerunning HFSS; all
12 EEP files were exported.

## Engineering Gate

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Passive RL | {analysis['passive_rl_min_db']:.3f} dB | >= 12 dB | Failed |
| Active RL | {analysis['active_rl_min_db']:.3f} dB | >= 11 dB | Failed |
| Total RL | {analysis['total_rl_min_db']:.3f} dB | >= 11 dB | Failed |
| Accepted-to-radiated efficiency | {100*analysis['integrated_accepted_to_radiated_efficiency_min']:.2f}% | >= 95% | Failed |
| Transducer efficiency | {100*analysis['integrated_transducer_efficiency_min']:.2f}% | >= 85% | Failed |
| Integrated versus S8+S4 max abs Delta S | {analysis['integrated_vs_cascade_max_abs_delta_s']:.3f} | <= 0.05 | Failed |

## Decision

No independent repeat, 4x4/16x16 expansion, HFSS labels, or critic training is
authorized. The validated S8 network itself is not the failed element; the
new POST-to-probe fanout changes electrical lengths and port impedances. The
next hardware experiment must equalize routed electrical lengths and optimize
the POST transition using this integrated S4, while keeping the v1.16 network
components and all engineering thresholds frozen.
"""
    (BASELINE / "BASELINE.md").write_text(report, encoding="ascii")
    write_manifest()
    print(json.dumps({"baseline": str(BASELINE), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
