#!/usr/bin/env python3
"""Package the v1.15 dual-reference-plane circuit and physical S8 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "baselines" / "2026-07-30-v115-grounded-modal-network"
CIRCUIT = ROOT / "hfss_outputs" / "v115_grounded_modal_network_20260730_run01"
PHYSICAL_INITIAL = ROOT / "hfss_outputs" / "v115_physical_modal_feed_fixture_20260730_run02"
CALIBRATION_FIRST = ROOT / "hfss_outputs" / "v115_physical_aware_resynthesis_20260730_run01"
PHYSICAL_NEAR = ROOT / "hfss_outputs" / "v115_physical_modal_feed_fixture_20260730_run04"
CALIBRATION_SECOND = ROOT / "hfss_outputs" / "v115_physical_aware_resynthesis_20260730_run02"
PHYSICAL_FINAL = ROOT / "hfss_outputs" / "v115_physical_modal_feed_fixture_20260730_run05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def analysis(root: Path) -> dict[str, Any]:
    return load_json(root / "physical_s8_direct01" / "analysis.json")


def compact_physical(label: str, root: Path) -> dict[str, Any]:
    value = analysis(root)
    return {
        "stage": label,
        "passive_rl_min_db": value["passive_rl_min_db"],
        "active_rl_min_db": value["active_rl_min_db"],
        "total_rl_min_db": value["total_rl_min_db"],
        "actual_load_insertion_efficiency_min": value["actual_load_insertion_efficiency_min"],
        "actual_load_transducer_efficiency_min": value["actual_load_transducer_efficiency_min"],
        "matched_load_network_efficiency_min": value["matched_load_network_efficiency_min"],
        "final_delta_s": value["final_delta_s"],
        "peak_solver_memory_gb": value["peak_solver_memory_gb"],
        "reciprocity_error_max": value["reciprocity_error_max"],
        "passivity_sigma_max": value["passivity_sigma_max"],
        "physical_gate_pass": value["physical_gate_pass"],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite v1.15 baseline: {out_dir}")
    snapshots = out_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    circuit = load_json(CIRCUIT / "selected_network.json")
    progression = [
        {
            "stage": "finite_q_lumped_circuit",
            "passive_rl_min_db": circuit["passive_rl_min_db"],
            "active_rl_min_db": circuit["active_rl_min_db"],
            "total_rl_min_db": circuit["total_rl_min_db"],
            "actual_load_insertion_efficiency_min": circuit["actual_load_insertion_efficiency_min"],
            "actual_load_transducer_efficiency_min": circuit["actual_load_transducer_efficiency_min"],
            "matched_load_network_efficiency_min": circuit["matched_load_network_efficiency_min"],
            "physical_gate_pass": False,
            "evidence_class": "circuit synthesis only",
        },
        {**compact_physical("initial_physical_s8", PHYSICAL_INITIAL), "evidence_class": "physical HFSS S8 cascaded with trusted HFSS antenna S4"},
        {**compact_physical("physical_aware_near_gate", PHYSICAL_NEAR), "evidence_class": "physical HFSS S8 cascaded with trusted HFSS antenna S4"},
        {**compact_physical("reserve_target_confirmation", PHYSICAL_FINAL), "evidence_class": "physical HFSS S8 cascaded with trusted HFSS antenna S4"},
    ]
    write_csv(snapshots / "stage_progression.csv", progression)

    worst_rows: list[dict[str, Any]] = []
    for label, root in (("physical_aware_near_gate", PHYSICAL_NEAR), ("reserve_target_confirmation", PHYSICAL_FINAL)):
        rows = read_csv(root / "physical_s8_direct01" / "physical_stimulus_metrics.csv")
        rows.sort(key=lambda row: float(row["active_rl_db"]))
        worst_rows.extend({"stage": label, **row} for row in rows[:12])
    write_csv(snapshots / "worst_physical_stimuli.csv", worst_rows)

    copies = {
        ROOT / "configs" / "v115_grounded_modal_network.json": snapshots / "v115_grounded_modal_network.json",
        CIRCUIT / "variant_summary.csv": snapshots / "circuit_variant_summary.csv",
        CIRCUIT / "selected_network.json": snapshots / "circuit_selected_network.json",
        CIRCUIT / "selected_frequency_metrics.csv": snapshots / "circuit_frequency_metrics.csv",
        CIRCUIT / "selected_dual_reference_plane_network.npz": snapshots / "circuit_dual_reference_plane_network.npz",
        CALIBRATION_FIRST / "physical_aware_selected.json": snapshots / "physical_aware_design_01.json",
        CALIBRATION_SECOND / "physical_aware_selected.json": snapshots / "physical_aware_design_02.json",
        PHYSICAL_INITIAL / "physical_s8_direct01" / "v115_physical_modal_feed_s8_direct01.s8p": snapshots / "physical_initial.s8p",
        PHYSICAL_NEAR / "physical_s8_direct01" / "v115_physical_modal_feed_s8_direct01.s8p": snapshots / "physical_near_gate.s8p",
        PHYSICAL_FINAL / "physical_s8_direct01" / "v115_physical_modal_feed_s8_direct01.s8p": snapshots / "physical_final.s8p",
        PHYSICAL_INITIAL / "physical_s8_direct01" / "physical_frequency_metrics.csv": snapshots / "physical_initial_frequency_metrics.csv",
        PHYSICAL_NEAR / "physical_s8_direct01" / "physical_frequency_metrics.csv": snapshots / "physical_near_gate_frequency_metrics.csv",
        PHYSICAL_FINAL / "physical_s8_direct01" / "physical_frequency_metrics.csv": snapshots / "physical_final_frequency_metrics.csv",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)

    near = analysis(PHYSICAL_NEAR)
    final = analysis(PHYSICAL_FINAL)
    decision = {
        "version": "v1.15.0-grounded-modal-network",
        "circuit_gate_pass": True,
        "physical_dual_reference_plane_gate_pass": False,
        "best_physical_active_rl_db": float(near["active_rl_min_db"]),
        "best_physical_active_rl_deficit_db": 11.0 - float(near["active_rl_min_db"]),
        "physical_network_insertion_efficiency_gate_pass": True,
        "physical_integrated_antenna_network_fullwave_completed": False,
        "allow_independent_repeat": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_hfss_training_labels": False,
        "allow_critic_training": False,
        "next_required_experiment": "parameterized fullwave S8 optimization or distributed even_odd hybrid; no further low_order surrogate extrapolation",
    }
    write_json(snapshots / "stage_decision.json", decision)
    write_json(
        out_dir / "baseline_metadata.json",
        {
            "tag": out_dir.name,
            "version": decision["version"],
            "evidence_scope": "finite-Q circuit plus physical centralized-feed S8 cascaded with trusted antenna S4",
            "integrated_network_antenna_fullwave": False,
            "training_labels_locked": True,
            "hfss_physical_labels_allowed": False,
            "residual_critic_training_locked": True,
            "rebuild_16x16_allowed": False,
        },
    )
    report = f"""# v1.15 Grounded X-Modal Feed Network

This baseline separates four source-facing PRE reference planes from four
antenna-facing POST reference planes. Network insertion efficiency is computed
as antenna-plane accepted power divided by PRE-plane accepted power; transducer
efficiency additionally includes input mismatch.

## Circuit Synthesis

The selected finite-Q grounded-lowpass modal circuit reaches a minimum passive
RL of {float(circuit['passive_rl_min_db']):.3f} dB, representative active RL of
{float(circuit['active_rl_min_db']):.3f} dB, total RL of
{float(circuit['total_rl_min_db']):.3f} dB, and actual-load insertion efficiency
of {100.0 * float(circuit['actual_load_insertion_efficiency_min']):.2f}%. Its
reference-plane cascade error is {float(circuit['reference_plane_cascade_error_max']):.3e}.

## Physical S8 Evidence

| Stage | Passive RL | Active RL | Total RL | Insertion efficiency | Delta S |
|---|---:|---:|---:|---:|---:|
| Initial fixture | {float(progression[1]['passive_rl_min_db']):.3f} dB | {float(progression[1]['active_rl_min_db']):.3f} dB | {float(progression[1]['total_rl_min_db']):.3f} dB | {100.0 * float(progression[1]['actual_load_insertion_efficiency_min']):.2f}% | {float(progression[1]['final_delta_s']):.5f} |
| Physical-aware near gate | {float(near['passive_rl_min_db']):.3f} dB | {float(near['active_rl_min_db']):.3f} dB | {float(near['total_rl_min_db']):.3f} dB | {100.0 * float(near['actual_load_insertion_efficiency_min']):.2f}% | {float(near['final_delta_s']):.5f} |
| Reserve-target confirmation | {float(final['passive_rl_min_db']):.3f} dB | {float(final['active_rl_min_db']):.3f} dB | {float(final['total_rl_min_db']):.3f} dB | {100.0 * float(final['actual_load_insertion_efficiency_min']):.2f}% | {float(final['final_delta_s']):.5f} |

The best physical result misses the unchanged 11 dB active-RL gate by
{11.0 - float(near['active_rl_min_db']):.3f} dB. The subsequent reserve-target
geometry does not confirm the surrogate prediction. Both physical networks are
reciprocal, passive, converged, and above 95% insertion efficiency, so the
remaining failure is joint active matching rather than dissipative loss.

## Decision

The circuit concept is feasible, but this centralized lumped-layout realization
is not physically qualified. No independent repeat, integrated antenna-network
full-wave model, 4x4/16x16 expansion, training labels, or critic retraining is
authorized. Further work must use parameterized full-wave S8 optimization or a
distributed even/odd hybrid; low-order surrogate extrapolation is stopped.
"""
    (out_dir / "BASELINE.md").write_text(report, encoding="ascii")

    manifest_rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(out_dir)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(out_dir / "artifact_manifest.csv", manifest_rows)
    print(json.dumps({"baseline": str(out_dir), "artifacts": len(manifest_rows), **decision}, indent=2))


if __name__ == "__main__":
    main()
