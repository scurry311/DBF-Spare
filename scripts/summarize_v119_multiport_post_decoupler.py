#!/usr/bin/env python3
"""Freeze the v1.19 full-matrix circuit and physical POST S8 evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import parse_touchstone
from run_v115_physical_modal_feed_fixture import touchstone_port_names


ROOT = Path(__file__).resolve().parents[1]
CIRCUIT = ROOT / "hfss_outputs" / "v119_multiport_post_decoupler_20260730_run04"
SHORT = ROOT / "hfss_outputs" / "v1191_multiconductor_post_block_20260730_run01" / "physical_s8_direct01"
LONG = ROOT / "hfss_outputs" / "v1191_multiconductor_post_block_20260730_run02" / "physical_s8_direct01"
TWO = ROOT / "hfss_outputs" / "v1192_two_section_multiconductor_post_20260730_run02" / "physical_s8_direct01"
FAILED_BUILD = ROOT / "hfss_outputs" / "v1192_two_section_multiconductor_post_20260730_run01" / "physical_s8_direct01"
BASELINE = ROOT / "baselines" / "2026-07-30-v119-multiport-post-decoupler"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".s8p", ".csv", ".json", ".log"}:
        destination.write_text(source.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def modal_span(path: Path) -> float:
    _, matrices = parse_touchstone(path, 8)
    names = touchstone_port_names(path)
    desired = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    order = [names.index(name) for name in desired]
    matrix = matrices[1][order][:, order]
    phases = np.unwrap(np.angle(np.linalg.eigvals(matrix[4:, :4])))
    return float(np.ptp(phases) * 180.0 / np.pi)


def manifest() -> None:
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if BASELINE.exists():
        raise FileExistsError(f"Refusing to overwrite baseline: {BASELINE}")
    snapshots = BASELINE / "snapshots"
    snapshots.mkdir(parents=True)
    circuit = read_json(CIRCUIT / "synthesis_summary.json")
    short = read_json(SHORT / "analysis.json")
    long = read_json(LONG / "analysis.json")
    two = read_json(TWO / "analysis.json")
    artifacts = (
        (CIRCUIT / "config_snapshot.json", snapshots / "circuit_config.json"),
        (CIRCUIT / "synthesis_summary.json", snapshots / "circuit_synthesis.json"),
        (CIRCUIT / "stage_decision.json", snapshots / "circuit_decision.json"),
        (CIRCUIT / "components.csv", snapshots / "e96_components.csv"),
        (CIRCUIT / "tolerance_trials.csv", snapshots / "tolerance_trials.csv"),
        (CIRCUIT / "e96_modal_pi_target.s8p", snapshots / "e96_modal_pi_target.s8p"),
        (SHORT / "analysis.json", snapshots / "short_s8_analysis.json"),
        (SHORT / "frequency_gate_metrics.csv", snapshots / "short_s8_frequency.csv"),
        (SHORT / "v1191_multiconductor_post_direct01.s8p", snapshots / "short_physical.s8p"),
        (LONG / "analysis.json", snapshots / "long_s8_analysis.json"),
        (LONG / "frequency_gate_metrics.csv", snapshots / "long_s8_frequency.csv"),
        (LONG / "v1191_multiconductor_post_direct01.s8p", snapshots / "long_physical.s8p"),
        (TWO / "analysis.json", snapshots / "two_section_s8_analysis.json"),
        (TWO / "frequency_gate_metrics.csv", snapshots / "two_section_s8_frequency.csv"),
        (TWO / "v1191_multiconductor_post_direct01.s8p", snapshots / "two_section_physical.s8p"),
        (FAILED_BUILD / "build.log", snapshots / "two_section_failed_build.log"),
    )
    for source, destination in artifacts:
        copy(source, destination)
    comparison = [
        {
            "candidate": name,
            "final_delta_s": metrics["final_delta_s"],
            "peak_solver_memory_gb": metrics["peak_solver_memory_gb"],
            "network_efficiency_min": metrics["network_efficiency_min"],
            "offdiagonal_transmission_max": metrics["offdiagonal_transmission_max"],
            "modal_phase_span_deg": modal_span(path),
            "active_rl_min_db": metrics["active_rl_min_db"],
            "total_rl_min_db": metrics["total_rl_min_db"],
            "corrected_vs_target_max_abs_delta_s": metrics["corrected_vs_target_max_abs_delta_s"],
            "physical_s8_gate_pass": metrics["physical_s8_gate_pass"],
        }
        for name, metrics, path in (
            ("single_5p4mm", short, SHORT / "v1191_multiconductor_post_direct01.s8p"),
            ("single_10p8mm", long, LONG / "v1191_multiconductor_post_direct01.s8p"),
            ("two_section_9p0_plus_9p0mm", two, TWO / "v1191_multiconductor_post_direct01.s8p"),
        )
    ]
    with (BASELINE / "physical_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    decision = {
        "full_matrix_finite_q_circuit_gate_pass": circuit["tolerance_gate_pass"],
        "compact_realization_gate_pass": circuit["compact_realization_gate_pass"],
        "physical_s8_pass_count": sum(int(row["physical_s8_gate_pass"]) for row in comparison),
        "allow_integrated_2x2": False,
        "allow_independent_repeat": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "stop_condition": "full-matrix correction is circuit-feasible but not compactly realizable by the tested single-layer one/two-section POST blocks",
        "next_action": "reduce the correction burden by jointly redesigning the antenna feed and POST fanout under a realizable sparse coupling graph",
    }
    write_json(BASELINE / "stage_decision.json", decision)
    metadata = {
        "version": "v1.19.0-multiport-post-decoupler",
        "created_on": "2026-07-30",
        "parent_tag": "v1.18.0-equalized-post-transition",
        "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "evidence_level": "finite-Q full-matrix circuit target plus three standalone physical HFSS S8 candidates",
        "integrated_2x2_authorized": False,
        "training_label_authorized": False,
    }
    write_json(BASELINE / "baseline_metadata.json", metadata)
    report = f"""# v1.19 Full-Matrix Multiport POST Decoupler

The v1.18 integrated S4 is deembedded through the trusted v1.16 feed S8. A
reciprocal four-port pi correction network is synthesized with full
off-diagonal series coupling, input/output bridge terms, finite component Q,
E96 quantization, and the same 285 frozen representative excitations.

## Circuit Upper Bound

| Metric | Result | Gate |
|---|---:|---:|
| Active RL | {circuit['e96_quantized_metrics']['active_rl_min_db']:.3f} dB | >= 11 dB |
| Total RL | {circuit['e96_quantized_metrics']['total_rl_min_db']:.3f} dB | >= 11 dB |
| Passive RL | {circuit['e96_quantized_metrics']['passive_rl_min_db']:.3f} dB | >= 10 dB |
| Corrected versus target max abs Delta S | {circuit['e96_quantized_metrics']['integrated_vs_target_max_abs_delta_s']:.5f} | <= 0.05 |
| Matched-load network efficiency | {100*circuit['e96_matched_load_network_efficiency_min']:.2f}% | >= 95% |
| 1000-sample tolerance joint pass | {100*circuit['tolerance_joint_pass_rate']:.1f}% | >= 90% |

This is a circuit upper bound, not physical HFSS evidence. Twelve of twenty
equivalent discrete values are outside the declared 10 GHz package range. An
exact Givens implementation requires twelve transform stages and has only
{100*circuit['estimated_modal_transform_efficiency']:.2f}% estimated transform
efficiency, so the compact realization gate fails.

## Physical S8 Tests

| Candidate | Solver Delta S | Peak memory | Efficiency | Max off-diagonal transmission | Modal phase span | Active RL | Corrected Delta S |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5.4 mm one section | {short['final_delta_s']:.5f} | {short['peak_solver_memory_gb']:.2f} GiB | {100*short['network_efficiency_min']:.2f}% | {short['offdiagonal_transmission_max']:.3f} | {comparison[0]['modal_phase_span_deg']:.1f} deg | {short['active_rl_min_db']:.3f} dB | {short['corrected_vs_target_max_abs_delta_s']:.3f} |
| 10.8 mm one section | {long['final_delta_s']:.5f} | {long['peak_solver_memory_gb']:.2f} GiB | {100*long['network_efficiency_min']:.2f}% | {long['offdiagonal_transmission_max']:.3f} | {comparison[1]['modal_phase_span_deg']:.1f} deg | {long['active_rl_min_db']:.3f} dB | {long['corrected_vs_target_max_abs_delta_s']:.3f} |
| 9+9 mm noncommuting sections | {two['final_delta_s']:.5f} | {two['peak_solver_memory_gb']:.2f} GiB | {100*two['network_efficiency_min']:.2f}% | {two['offdiagonal_transmission_max']:.3f} | {comparison[2]['modal_phase_span_deg']:.1f} deg | {two['active_rl_min_db']:.3f} dB | {two['corrected_vs_target_max_abs_delta_s']:.3f} |

All three S8 solutions are converged, reciprocal, and passive. The longer
single section improves active RL but remains far below 11 dB. The two-section
network does not create the required approximately 53 degree modal spread,
and its efficiency drops below 95%. The phase alignment used in this screen
is optimistic because it is fitted at each frequency; failure therefore
cannot be blamed on an overly strict reference-plane convention.

## Decision

No independent repeat, integrated 2x2, array expansion, HFSS labels, or critic
training is authorized. Additional coupled sections are stopped because they
increase size and loss without supplying the required full coupling matrix.
The next hardware revision must reduce the required correction at its source:
jointly redesign the antenna feed and POST fanout while constraining the
network to a realizable sparse adjacency graph.
"""
    (BASELINE / "BASELINE.md").write_text(report, encoding="ascii")
    manifest()
    print(json.dumps({"baseline": str(BASELINE), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
