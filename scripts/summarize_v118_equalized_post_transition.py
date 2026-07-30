#!/usr/bin/env python3
"""Freeze the v1.18 equalized POST-transition S4 pre-repeat gate."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "hfss_outputs" / "v118_equalized_post_transition_20260730_run01"
CANDIDATE = SOURCE / "candidates" / "eq23p393_launch0p95_pad0p50"
CASE = CANDIDATE / "integrated_2x2_direct01"
BASELINE = ROOT / "baselines" / "2026-07-30-v118-equalized-post-transition"


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
    analysis = read_json(CASE / "s4_gate_analysis.json")
    decision = read_json(CANDIDATE / "stage_decision.json")
    if not analysis.get("converged") or analysis.get("post_transition_gate_pass"):
        raise RuntimeError("Expected a converged candidate that failed the pre-repeat gate")
    snapshots = BASELINE / "snapshots"
    snapshots.mkdir(parents=True)
    artifacts = (
        (CANDIDATE / "config_snapshot.json", snapshots / "config_snapshot.json"),
        (CANDIDATE / "antenna_protocol_snapshot.json", snapshots / "antenna_protocol_snapshot.json"),
        (CANDIDATE / "stage_decision.json", snapshots / "stage_decision.json"),
        (CASE / "case_manifest.json", snapshots / "case_manifest.json"),
        (CASE / "run_summary.json", snapshots / "run_summary.json"),
        (CASE / "field_export_summary.json", snapshots / "field_export_summary.json"),
        (CASE / "s4_gate_analysis.json", snapshots / "s4_gate_analysis.json"),
        (CASE / "frequency_s4_gate_metrics.csv", snapshots / "frequency_s4_gate_metrics.csv"),
        (CASE / "v117_integrated_2x2_direct01.s4p", snapshots / "equalized_direct01.s4p"),
        (SOURCE / "proxy_fit_summary.json", snapshots / "proxy_fit_summary.json"),
        (SOURCE / "proxy_candidate_screen.csv", snapshots / "proxy_candidate_screen.csv"),
        (SOURCE / "per_port_transition_synthesis.json", snapshots / "per_port_transition_synthesis.json"),
        (SOURCE / "per_port_lsection_synthesis.json", snapshots / "per_port_lsection_synthesis.json"),
    )
    for source, destination in artifacts:
        copy_artifact(source, destination)
    metadata = {
        "version": "v1.18.0-equalized-post-transition",
        "created_on": "2026-07-30",
        "parent_tag": "v1.17.0-integrated-2x2-smoke",
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_output": str(SOURCE.resolve()),
        "evidence_level": "integrated physical equal-length 2x2 HFSS S4 pre-repeat gate",
        "independent_repeat_authorized": False,
        "training_label_authorized": False,
    }
    write_json(BASELINE / "baseline_metadata.json", metadata)
    report = f"""# v1.18 Equalized POST Transition

The v1.16 RLC values, Q factors, substrate, copper, trace width, channel
locations, and modal bridge values are frozen. The compact network is moved as
a rigid body and four non-crossing routes are made equal at
`{analysis['equalized_route_length_mm']:.4f} mm`. A common stepped probe launch
is included and the integrated physical 2x2 is solved at 9.96/10/10.04 GHz.

## Numerical Credibility

| Metric | Result | Gate |
|---|---:|---:|
| Adaptive passes | {analysis['pass_count']} | converged |
| Final Delta S | {analysis['final_delta_s']:.5f} | <= 0.05 |
| Tetrahedra | {analysis['maximum_tetrahedra']:,} | resource audit |
| Peak solver memory | {analysis['peak_solver_memory_gb']:.2f} GiB | resource audit |
| Minimum free memory | {analysis['minimum_free_memory_gb_during_solve']:.2f} GiB | > 1.5 GiB abort line |
| Reciprocity error | {analysis['reciprocity_error_max']:.3e} | <= 1e-4 |
| Passivity sigma | {analysis['passivity_sigma_max']:.3f} | <= 1.001 |

## Pre-Repeat Gate

| Metric | v1.17 | v1.18 | Gate | Decision |
|---|---:|---:|---:|---|
| Active RL | 1.601 dB | {analysis['active_rl_min_db']:.3f} dB | >= 11 dB | Failed |
| Integrated versus cascade max abs Delta S | 0.307 | {analysis['integrated_vs_cascade_max_abs_delta_s']:.3f} | <= 0.05 | Failed |
| Passive RL | 8.251 dB | {analysis['passive_rl_min_db']:.3f} dB | diagnostic | Improved |
| Total RL | 8.215 dB | {analysis['total_rl_min_db']:.3f} dB | diagnostic | Improved |

Equalization is beneficial but insufficient. A common transition search and
per-port uncoupled stepped-line/L-section synthesis have no joint feasible
point. The best per-port stepped-line proxy reaches about 9.97 dB active RL
and 0.096 max abs Delta S, still outside both gates.

The EEP report export produced no files. This is recorded but is not a reason
to rerun the field solve because the S4 pre-repeat gate already failed.

## Decision

No independent repeat, 4x4/16x16 expansion, HFSS labels, or critic training is
authorized. The next physical topology must integrate a multiport POST
transition/decoupler; further common-length or uncoupled local launch sweeps
are stopped.
"""
    (BASELINE / "BASELINE.md").write_text(report, encoding="ascii")
    write_manifest()
    print(json.dumps({"baseline": str(BASELINE), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
