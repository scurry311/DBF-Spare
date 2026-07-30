#!/usr/bin/env python3
"""Freeze the v1.16 exact-HFSS physical S8 optimization evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "hfss_outputs" / "v116_physical_s8_hfss_optimization_20260730_run02"
BASELINE = ROOT / "baselines" / "2026-07-30-v116-physical-s8-hfss-optimization"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".s4p", ".s8p"}:
        lines = [line.rstrip() for line in source.read_text(encoding="utf-8", errors="strict").splitlines()]
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
    decision = read_json(SOURCE / "stage_decision.json")
    if not decision.get("independent_repeat_pass"):
        raise RuntimeError("The independent exact-HFSS repeat gate has not passed")
    best = read_json(SOURCE / "best_candidate.json")
    repeat = read_json(SOURCE / "candidates" / "repeat_00_independent" / "analysis.json")
    nominal = read_json(SOURCE / "candidates" / "coarse_00_nominal" / "analysis.json")
    BASELINE.mkdir(parents=True)
    snapshots = BASELINE / "snapshots"
    for name in (
        "config_snapshot.json",
        "base_protocol_snapshot.json",
        "base_physical_design_snapshot.json",
        "candidate_metrics.csv",
        "best_candidate.json",
        "repeat_source.json",
        "stage_decision.json",
    ):
        copy_artifact(SOURCE / name, snapshots / name)
    for candidate in sorted((SOURCE / "candidates").iterdir()):
        if not candidate.is_dir() or not (candidate / "analysis.json").exists():
            continue
        candidate_id = candidate.name
        copy_artifact(candidate / "analysis.json", snapshots / "analysis" / f"{candidate_id}.json")
        copy_artifact(candidate / f"{candidate_id}.s8p", snapshots / "s8" / f"{candidate_id}.s8p")
        copy_artifact(
            candidate / "physical_frequency_metrics.csv",
            snapshots / "frequency" / f"{candidate_id}.csv",
        )
    copy_artifact(
        SOURCE / "candidates" / best["candidate_id"] / "physical_stimulus_metrics.csv",
        snapshots / "best_physical_stimulus_metrics.csv",
    )
    metadata = {
        "version": "v1.16.0-physical-s8-hfss-optimization",
        "created_on": "2026-07-30",
        "parent_tag": "v1.15.0-grounded-modal-network",
        "source_output": str(SOURCE.resolve()),
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "candidate_count": 14,
        "evidence_level": "physical HFSS S8 cascaded with trusted antenna S4; not integrated antenna-network full-wave",
    }
    write_json(BASELINE / "baseline_metadata.json", metadata)
    component = best["component_values"]
    report = f"""# v1.16 Exact-HFSS Physical S8 Optimization

This baseline replaces low-order circuit extrapolation with exact HFSS solves
of a parameterized physical eight-port feed fixture. Four PRE and four POST
reference planes remain explicit, and all 285 frozen representative sources
are replayed after cascading each physical S8 with the trusted antenna S4.

## Search

Seven one-at-a-time coarse points and six local refinement points were solved
sequentially. The selected finite-Q component values are:

- series inductor: `{component['series_inductor_nh']:.6f} nH`;
- grounded shunt capacitor: `{component['ground_capacitor_pf']:.6f} pF`;
- x-pair bridge inductor: `{component['bridge_inductor_nh']:.6f} nH`.

## Result

| Metric | v1.15 nominal | v1.16 selected | Independent repeat |
|---|---:|---:|---:|
| Passive RL | {nominal['passive_rl_min_db']:.3f} dB | {best['passive_rl_min_db']:.3f} dB | {repeat['passive_rl_min_db']:.3f} dB |
| Active RL | {nominal['active_rl_min_db']:.3f} dB | {best['active_rl_min_db']:.3f} dB | {repeat['active_rl_min_db']:.3f} dB |
| Total RL | {nominal['total_rl_min_db']:.3f} dB | {best['total_rl_min_db']:.3f} dB | {repeat['total_rl_min_db']:.3f} dB |
| Actual-load insertion efficiency | {100*nominal['actual_load_insertion_efficiency_min']:.2f}% | {100*best['actual_load_insertion_efficiency_min']:.2f}% | {100*repeat['actual_load_insertion_efficiency_min']:.2f}% |
| Transducer efficiency | {100*nominal['actual_load_transducer_efficiency_min']:.2f}% | {100*best['actual_load_transducer_efficiency_min']:.2f}% | {100*repeat['actual_load_transducer_efficiency_min']:.2f}% |
| Final Delta S | {nominal['final_delta_s']:.5f} | {best['final_delta_s']:.5f} | {repeat['final_delta_s']:.5f} |

The selected design improves minimum active RL by
`{decision['best_active_rl_improvement_db']:.3f} dB`. The independent repeat
differs by only `{decision['independent_repeat_max_abs_delta_s']:.3e}` in
maximum absolute S and passes the unchanged 11.5 dB design gate.

## Decision

Retain the grounded x-modal topology. One integrated physical 2x2
antenna-network smoke is now authorized. A distributed even/odd hybrid is kept
only as a fallback. The 4x4/16x16 expansions, HFSS labels, and critic training
remain locked because this evidence is still an S8 plus trusted-S4 cascade,
not an integrated full-wave antenna-network validation.
"""
    (BASELINE / "BASELINE.md").write_text(report, encoding="ascii")
    write_manifest()
    print(json.dumps({"baseline": str(BASELINE), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
