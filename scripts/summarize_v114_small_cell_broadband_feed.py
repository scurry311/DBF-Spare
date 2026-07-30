#!/usr/bin/env python3
"""Build the immutable v1.14 small-cell broadband-feed evidence snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import parse_touchstone


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "baselines" / "2026-07-30-v114-small-cell-broadband-feed"
RUNS = {
    "preregistered": ROOT / "hfss_outputs" / "v114_small_cell_broadband_feed_20260730_run04",
    "tongue_development": ROOT / "hfss_outputs" / "v1141_small_cell_stepped_tongue_20260730_run05",
    "feed_depth": ROOT / "hfss_outputs" / "v1142_small_cell_feed_depth_20260730_run06",
    "surface_validation": ROOT / "hfss_outputs" / "v1143_small_cell_surface_mesh_20260730_run07",
    "x_decoupler": ROOT / "hfss_outputs" / "v1144_small_cell_x_decoupler_20260730_run08",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="ascii")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def analysis_rows(run: Path, side: int) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run / f"{side}x{side}").glob("*_direct*/analysis.json")):
        row = load_json(path)
        row["source_run"] = run.name
        row["analysis_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        rows.append(row)
    return rows


def one_by_one_rows() -> list[dict[str, Any]]:
    labels = {
        "preregistered": "preregistered topology screen",
        "tongue_development": "data-informed engineering development",
        "feed_depth": "feed-depth confirmation and independent direct repeat",
        "surface_validation": "surface-mesh cross-check and independent repeat",
    }
    rows: list[dict[str, Any]] = []
    for key in labels:
        for source in analysis_rows(RUNS[key], 1):
            rows.append(
                {
                    "evidence_stage": labels[key],
                    "candidate_id": source.get("candidate_id"),
                    "replicate": source.get("replicate"),
                    "passive_rl_min_db": source.get("passive_rl_min_db"),
                    "minimum_radiation_efficiency": source.get("minimum_radiation_efficiency"),
                    "final_delta_s": source.get("final_delta_s"),
                    "cross_mesh_max_abs_delta_s": source.get("cross_mesh_max_abs_delta_s"),
                    "repeat_or_base_gate_pass": source.get("base_gate_pass"),
                    "peak_solver_memory_gb": source.get("peak_solver_memory_gb"),
                    "maximum_tetrahedra": source.get("maximum_tetrahedra"),
                    "source_run": source["source_run"],
                }
            )
    return rows


def two_by_two_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, key in (
        ("trusted base feed", "surface_validation"),
        ("isolated x-strip development", "x_decoupler"),
    ):
        for source in analysis_rows(RUNS[key], 2):
            rows.append(
                {
                    "topology_stage": label,
                    "candidate_id": source.get("candidate_id"),
                    "passive_rl_min_db": source.get("passive_rl_min_db"),
                    "representative_active_rl_min_db": source.get("representative_active_rl_min_db"),
                    "representative_total_rl_min_db": source.get("representative_total_rl_min_db"),
                    "minimum_radiation_efficiency": source.get("minimum_radiation_efficiency"),
                    "final_delta_s": source.get("final_delta_s"),
                    "peak_solver_memory_gb": source.get("peak_solver_memory_gb"),
                    "maximum_tetrahedra": source.get("maximum_tetrahedra"),
                    "base_gate_pass": source.get("base_gate_pass"),
                    "source_run": source["source_run"],
                }
            )
    return rows


def repeat_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("feed_depth", "surface_validation"):
        analyses = analysis_rows(RUNS[key], 1)
        candidate_ids = sorted({str(row["candidate_id"]) for row in analyses})
        for candidate_id in candidate_ids:
            selected = sorted(
                (row for row in analyses if row["candidate_id"] == candidate_id),
                key=lambda row: int(row["replicate"]),
            )
            if len(selected) < 2:
                continue
            matrices = []
            frequency_grids = []
            for analysis in selected[:2]:
                analysis_path = ROOT / analysis["analysis_path"]
                frequencies, matrix = parse_touchstone(touchstone_for_analysis(analysis_path), 1)
                frequency_grids.append(frequencies)
                matrices.append(matrix)
            if not np.allclose(frequency_grids[0], frequency_grids[1], atol=1.0e-12):
                raise ValueError(f"Repeat frequency mismatch for {candidate_id}")
            delta = np.abs(matrices[0] - matrices[1])
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "side": 1,
                    "replicate_a": int(selected[0]["replicate"]),
                    "replicate_b": int(selected[1]["replicate"]),
                    "max_abs_delta_s": float(np.max(delta)),
                    "rms_abs_delta_s": float(np.sqrt(np.mean(delta**2))),
                    "maximum_allowed_abs_delta_s": 0.05,
                    "repeat_gate_pass": bool(float(np.max(delta)) <= 0.05),
                }
            )
    return rows


def touchstone_for_analysis(path: Path) -> Path:
    manifest = load_json(path.parent / "case_manifest.json")
    return Path(manifest["touchstone_path"])


def pair_mode_rl(matrix: np.ndarray, pair: tuple[int, int]) -> float:
    worst_gamma = 0.0
    for sign in (1.0, -1.0):
        source = np.zeros(4, dtype=complex)
        source[list(pair)] = (1.0, sign)
        reflected = matrix @ source
        gamma = np.max(np.abs(reflected[list(pair)]) / np.abs(source[list(pair)]))
        worst_gamma = max(worst_gamma, float(gamma))
    return float(-20.0 * np.log10(max(worst_gamma, 1.0e-12)))


def four_port_mode_rl(matrix: np.ndarray) -> float:
    worst_gamma = 0.0
    for x_sign in (1.0, -1.0):
        for y_sign in (1.0, -1.0):
            source = np.asarray(
                (1.0, y_sign, x_sign, x_sign * y_sign), dtype=complex
            )
            reflected = matrix @ source
            worst_gamma = max(worst_gamma, float(np.max(np.abs(reflected / source))))
    return float(-20.0 * np.log10(max(worst_gamma, 1.0e-12)))


def modal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    analyses = analysis_rows(RUNS["surface_validation"], 2) + analysis_rows(
        RUNS["x_decoupler"], 2
    )
    x_pairs = ((0, 2), (1, 3))
    y_pairs = ((0, 1), (2, 3))
    diagonal_pairs = ((0, 3), (1, 2))
    for analysis in analyses:
        analysis_path = ROOT / analysis["analysis_path"]
        frequencies, matrices = parse_touchstone(touchstone_for_analysis(analysis_path), 4)
        for frequency, matrix in zip(frequencies, matrices):
            coupling = lambda pairs: 20.0 * np.log10(
                max(max(abs(matrix[i, j]), abs(matrix[j, i])) for i, j in pairs)
            )
            rows.append(
                {
                    "candidate_id": analysis["candidate_id"],
                    "frequency_ghz": float(frequency),
                    "x_neighbor_coupling_worst_db": float(coupling(x_pairs)),
                    "y_neighbor_coupling_worst_db": float(coupling(y_pairs)),
                    "diagonal_coupling_worst_db": float(coupling(diagonal_pairs)),
                    "x_pair_even_odd_active_rl_worst_db": min(
                        pair_mode_rl(matrix, pair) for pair in x_pairs
                    ),
                    "four_port_even_odd_active_rl_worst_db": four_port_mode_rl(matrix),
                }
            )
    return rows


def failure_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("surface_validation", "x_decoupler"):
        for replay in sorted((RUNS[key] / "2x2").glob("*_direct*/representative_active_rl_replay.csv")):
            candidate = load_json(replay.parent / "case_manifest.json")["candidate"]["candidate_id"]
            for row in read_csv(replay):
                rows.append({"candidate_id": candidate, **row})
    rows.sort(key=lambda row: float(row["active_rl_db"]))
    return rows[:30]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(
    out_dir: Path,
    one_rows: list[dict[str, Any]],
    two_rows: list[dict[str, Any]],
    repeats: list[dict[str, Any]],
    modal: list[dict[str, Any]],
) -> None:
    st_h = next(row for row in one_rows if row["candidate_id"] == "st_h" and int(row["replicate"]) == 1)
    surface = next(row for row in one_rows if row["candidate_id"] == "st_h_surface" and int(row["replicate"]) == 1)
    repeat_max = max(float(row["max_abs_delta_s"]) for row in repeats)
    base = next(row for row in two_rows if row["candidate_id"] == "st_h_surface")
    x_rows = [row for row in two_rows if row["candidate_id"].startswith("dx_")]
    modal_at_high = [row for row in modal if abs(float(row["frequency_ghz"]) - 10.04) < 1.0e-6]
    x_best = max(
        (row for row in modal_at_high if row["candidate_id"].startswith("dx_")),
        key=lambda row: float(row["four_port_even_odd_active_rl_worst_db"]),
    )
    text = f"""# v1.14 Small-Cell Broadband-Feed Feasibility

This baseline evaluates physical broadband-feed feasibility only. It does not
rebuild the 16x16 array, generate HFSS training labels, alter engineering
thresholds, search masks, or retrain the residual critic.

## One-Cell Gate

The final stepped-tongue dual-slot feed (`st_h`) reaches a three-frequency
minimum passive RL of {float(st_h['passive_rl_min_db']):.3f} dB on the volumetric
0.18 mm feed mesh. The lower-memory surface-refined realization reaches
{float(surface['passive_rl_min_db']):.3f} dB. Its cross-mesh maximum |Delta S| is
{float(surface['cross_mesh_max_abs_delta_s']):.5f}; the independent direct repeat
has maximum |Delta S| {repeat_max:.3e}. Radiation efficiency is approximately
{100.0 * min(float(st_h['minimum_radiation_efficiency']), float(surface['minimum_radiation_efficiency'])):.2f}%.

The intermediate tongue candidates were chosen after inspecting preceding
impedance results and are therefore engineering development evidence, not an
independent candidate benchmark.

## Two-Cell Gate

| Metric | Trusted base feed | Required |
|---|---:|---:|
| Minimum passive RL | {float(base['passive_rl_min_db']):.3f} dB | >= 12 dB |
| Representative worst active RL | {float(base['representative_active_rl_min_db']):.3f} dB | >= 11 dB |
| Representative worst total RL | {float(base['representative_total_rl_min_db']):.3f} dB | >= 11 dB |
| Minimum radiation efficiency | {100.0 * float(base['minimum_radiation_efficiency']):.2f}% | >= 95% |
| Final Delta S | {float(base['final_delta_s']):.5f} | <= 0.05 |
| Peak solver memory | {float(base['peak_solver_memory_gb']):.2f} GiB | Diagnostic |

The passive, efficiency, and convergence gates pass, but representative active
matching fails. At 10.04 GHz the trusted base x-neighbor coupling is reported in
`modal_coupling_comparison.csv`; its worst four-port even/odd-mode active RL is
{float(next(row for row in modal_at_high if row['candidate_id'] == 'st_h_surface')['four_port_even_odd_active_rl_worst_db']):.3f} dB. The x-neighbor coupling drives the weak mode.
An independent 2x2 repeat is not opened after this active-RL stop condition.

## Local Decoupler Screen

{len(x_rows)} isolated x-strip candidates were solved. None passes the 2x2 base
gate. Even the best high-corner four-port modal result among these candidates is
{float(x_best['four_port_even_odd_active_rl_worst_db']):.3f} dB, so this ungrounded
parasitic-strip topology is rejected rather than expanded to 4x4.

## Decision

The 1x1 physical feed and deterministic surface-mesh method are credible, but
the 2x2 representative active-RL gate fails. Therefore 4x4, 16x16 S256/EEP,
candidate HFSS labels, mask search, and critic training remain locked. The next
hardware experiment must synthesize a controlled grounded or capacitively
loaded x-pair even/odd-mode decoupling network on the trusted S4, then validate
one physical 2x2 realization against the same frozen stimuli and unchanged
thresholds.
"""
    (out_dir / "BASELINE.md").write_text(text, encoding="ascii")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(path.is_file() for path in out_dir.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite existing baseline: {out_dir}")
    snapshots = out_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    one_rows = one_by_one_rows()
    two_rows = two_by_two_rows()
    repeats = repeat_rows()
    modal = modal_rows()
    failures = failure_rows()
    if not any(row["candidate_id"] == "st_h_surface" for row in two_rows):
        raise RuntimeError("Trusted 2x2 base analysis is missing")
    if len([row for row in two_rows if row["candidate_id"].startswith("dx_")]) != 3:
        raise RuntimeError("All three preregistered x-decoupler endpoints must be analyzed")

    write_csv(snapshots / "one_by_one_development.csv", one_rows)
    write_csv(snapshots / "two_by_two_gate.csv", two_rows)
    write_csv(snapshots / "repeat_crosscheck.csv", repeats)
    write_csv(snapshots / "modal_coupling_comparison.csv", modal)
    write_csv(snapshots / "active_failure_examples.csv", failures)

    decision = {
        "version": "v1.14.0-small-cell-broadband-feed",
        "thresholds_changed": False,
        "one_by_one_physical_gate_pass": True,
        "two_by_two_passive_efficiency_convergence_gate_pass": True,
        "two_by_two_representative_active_rl_gate_pass": False,
        "isolated_x_parasitic_strip_topology_accepted": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_hfss_candidate_labels": False,
        "allow_mask_or_weight_optimization": False,
        "allow_residual_critic_training": False,
        "next_required_experiment": "grounded_or_capacitively_loaded_x_pair_even_odd_modal_network_on_2x2",
    }
    write_json(snapshots / "stage_decision.json", decision)
    build_report(out_dir, one_rows, two_rows, repeats, modal)

    for config in (
        "v114_small_cell_broadband_feed_preregistered.json",
        "v1141_small_cell_stepped_tongue_refinement.json",
        "v1142_small_cell_feed_depth_confirmation.json",
        "v1143_small_cell_surface_mesh_crosscheck.json",
        "v1144_small_cell_x_modal_decoupler.json",
    ):
        shutil.copy2(ROOT / "configs" / config, snapshots / config)

    metadata = {
        "tag": out_dir.name,
        "version": decision["version"],
        "evidence_scope": "1x1 and 2x2 physical HFSS only",
        "training_labels_locked": True,
        "hfss_physical_labels_allowed": False,
        "residual_critic_training_locked": True,
        "four_by_four_allowed": False,
        "rebuild_16x16_allowed": False,
    }
    write_json(out_dir / "baseline_metadata.json", metadata)

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
    print(json.dumps({"baseline": str(out_dir), "artifact_count": len(manifest_rows), **decision}, indent=2))


if __name__ == "__main__":
    main()
