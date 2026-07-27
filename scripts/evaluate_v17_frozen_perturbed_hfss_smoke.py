#!/usr/bin/env python3
"""Evaluate the frozen v1.7 source-perturbed 16x16 HFSS smoke."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hfss-dir", type=Path, required=True)
    parser.add_argument("--active-audit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def integer(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def grouped_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    output: list[dict[str, Any]] = []
    for value, members in sorted(groups.items(), key=lambda item: item[0]):
        count = len(members)
        joint = sum(int(row["final_engineering_gate"]) for row in members)
        output.append(
            {
                "stratum": field,
                "value": value,
                "candidate_count": count,
                "pattern_gate_count": sum(int(row["fullwave_pattern_gate"]) for row in members),
                "semantic_active_rl_gate_count": sum(
                    int(row["semantic_active_rl_gate"]) for row in members
                ),
                "final_engineering_gate_count": joint,
                "final_engineering_gate_rate": joint / count,
                "all_nonzero_diagnostic_gate_count": sum(
                    int(row["all_nonzero_task_diagnostic_gate"]) for row in members
                ),
                "worst_hfss_psll_db": max(float(row["hfss_psll_db"]) for row in members),
                "minimum_hfss_nearest_iso_db": min(
                    float(row["hfss_nearest_iso_db"]) for row in members
                ),
                "minimum_hfss_local_iso_db": min(
                    float(row["hfss_local_iso_db"]) for row in members
                ),
                "minimum_combined_active_rl_db": min(
                    float(row["combined_worst_active_rl_db"]) for row in members
                ),
                "minimum_significant_task_active_rl_db": min(
                    float(row["worst_significant_case_active_rl_db"]) for row in members
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite smoke evaluation: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selection = read_csv(args.dataset_dir / "frozen_selection_manifest.csv")
    labels = read_csv(args.hfss_dir / "candidate_residual_labels.csv")
    active = read_csv(args.active_audit_dir / "candidate_active_rl_semantics.csv")
    analysis = json.loads((args.hfss_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    active_summary = json.loads(
        (args.active_audit_dir / "summary.json").read_text(encoding="utf-8")
    )
    prepared = json.loads((args.dataset_dir / "prepare_summary.json").read_text(encoding="utf-8"))

    selected_by_index = {integer(row, "candidate_index"): row for row in selection}
    labels_by_index = {integer(row, "candidate_index"): row for row in labels}
    active_by_index = {integer(row, "candidate_index"): row for row in active}
    indices = sorted(selected_by_index)
    if indices != sorted(labels_by_index) or indices != sorted(active_by_index):
        raise RuntimeError("Candidate indices disagree across frozen selection, HFSS, and RL audit")
    if len(indices) < 15 or len(indices) > 20:
        raise RuntimeError(f"Expected 15-20 frozen candidates, got {len(indices)}")

    combined: list[dict[str, Any]] = []
    for index in indices:
        source = selected_by_index[index]
        label = labels_by_index[index]
        rl = active_by_index[index]
        pattern_gate = integer(rl, "fullwave_pattern_gate")
        semantic_rl = integer(rl, "combined_plus_significant_task_active_rl_gate")
        final_gate = int(pattern_gate and semantic_rl)
        combined.append(
            {
                "candidate_index": index,
                "sample_index": integer(source, "sample_index"),
                "k": integer(source, "k_value"),
                "ratio": number(source, "ratio"),
                "active_count": integer(source, "active_count"),
                "selection_role": source["selection_role"],
                "large_scan": integer(label, "large_scan"),
                "min_target_separation_deg": number(source, "min_target_separation_deg"),
                "max_target_theta_deg": number(source, "max_target_theta_deg"),
                "frozen_E2_corner": source["frozen_E2_corner"],
                "E2_worst_margin_db": number(source, "E2_worst_margin_db"),
                "hfss_psll_db": number(label, "hfss_psll_db"),
                "hfss_nearest_iso_db": number(label, "hfss_nearest_iso_db"),
                "hfss_local_iso_db": number(label, "hfss_local_iso_db"),
                "hfss_mainlobe_gain_db": number(label, "hfss_mainlobe_gain_db"),
                "delta_psll_db": number(label, "delta_psll_db"),
                "delta_nearest_iso_db": number(label, "delta_nearest_iso_db"),
                "delta_local_iso_db": number(label, "delta_local_iso_db"),
                "delta_mainlobe_gain_db": number(label, "delta_mainlobe_gain_db"),
                "fullwave_pattern_gate": pattern_gate,
                "semantic_active_rl_gate": semantic_rl,
                "combined_worst_active_rl_db": number(rl, "combined_worst_active_rl_db"),
                "combined_total_rl_db": number(rl, "combined_total_rl_db"),
                "worst_significant_case_active_rl_db": number(
                    rl, "worst_significant_case_active_rl_db"
                ),
                "all_nonzero_task_diagnostic_gate": integer(
                    rl, "all_nonzero_task_active_rl_gate"
                ),
                "final_engineering_gate": final_gate,
            }
        )
    write_csv(args.out_dir / "candidate_final_gates.csv", combined)

    strata: list[dict[str, Any]] = []
    for field in ("k", "ratio", "selection_role", "large_scan"):
        strata.extend(grouped_rows(combined, field))
    write_csv(args.out_dir / "stratified_smoke_metrics.csv", strata)

    count = len(combined)
    final_count = sum(int(row["final_engineering_gate"]) for row in combined)
    k6_positive = sum(
        int(row["final_engineering_gate"]) for row in combined if int(row["k"]) == 6
    )
    sparse_multibeam_positive = sum(
        int(row["final_engineering_gate"])
        for row in combined
        if int(row["k"]) >= 2 and float(row["ratio"]) < 1.0
    )
    precision = final_count / count
    physical_corner_included = all(
        integer(row, "physical_16x16_operator_corner_included") == 1 for row in selection
    )
    smoke_pass = bool(
        analysis["complete_case_count"] == analysis["expected_case_count"]
        and analysis["all_no_scale_reconstruction_pass"]
        and precision >= float(args.minimum_precision)
        and k6_positive >= 1
        and sparse_multibeam_positive >= 5
    )
    summary = {
        "protocol": "v17-frozen-16x16-source-perturbed-hfss-smoke",
        "candidate_count": count,
        "independent_scene_count": len({int(row["sample_index"]) for row in combined}),
        "case_count": int(analysis["complete_case_count"]),
        "all_no_scale_reconstruction_pass": bool(analysis["all_no_scale_reconstruction_pass"]),
        "complex_nmse_max": float(analysis["complex_nmse_max"]),
        "magnitude_rmse_db_max": float(analysis["magnitude_rmse_db_max"]),
        "fullwave_pattern_gate_count": sum(int(row["fullwave_pattern_gate"]) for row in combined),
        "semantic_active_rl_gate_count": sum(
            int(row["semantic_active_rl_gate"]) for row in combined
        ),
        "final_engineering_gate_count": final_count,
        "accepted_candidate_hfss_precision": precision,
        "k6_positive_count": k6_positive,
        "sparse_multibeam_positive_count": sparse_multibeam_positive,
        "all_nonzero_task_diagnostic_gate_count": sum(
            int(row["all_nonzero_task_diagnostic_gate"]) for row in combined
        ),
        "all_nonzero_semantic_disagreement_count": int(
            active_summary["semantic_disagreement_count"]
        ),
        "worst_hfss_psll_db": max(float(row["hfss_psll_db"]) for row in combined),
        "minimum_hfss_nearest_iso_db": min(
            float(row["hfss_nearest_iso_db"]) for row in combined
        ),
        "minimum_hfss_local_iso_db": min(
            float(row["hfss_local_iso_db"]) for row in combined
        ),
        "minimum_combined_active_rl_db": min(
            float(row["combined_worst_active_rl_db"]) for row in combined
        ),
        "minimum_significant_task_active_rl_db": min(
            float(row["worst_significant_case_active_rl_db"]) for row in combined
        ),
        "maximum_absolute_delta_psll_db": max(
            abs(float(row["delta_psll_db"])) for row in combined
        ),
        "maximum_absolute_delta_nearest_iso_db": max(
            abs(float(row["delta_nearest_iso_db"])) for row in combined
        ),
        "maximum_absolute_delta_local_iso_db": max(
            abs(float(row["delta_local_iso_db"])) for row in combined
        ),
        "maximum_absolute_delta_mainlobe_gain_db": max(
            abs(float(row["delta_mainlobe_gain_db"])) for row in combined
        ),
        "smoke_gate_pass": smoke_pass,
        "physical_16x16_operator_corner_included": physical_corner_included,
        "hfss_scope": prepared["hfss_scope"],
        "critic_training_labels_allowed": False,
        "critic_label_blocker": (
            "The smoke reuses the nominal saved-field operator and perturbs source/calibration "
            "commands only; it does not expose hidden frequency, geometry, dielectric, or S-parameter drift."
        ),
        "thresholds_changed_after_hfss": False,
        "weights_changed_after_hfss": False,
    }
    (args.out_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Frozen 16x16 Perturbed HFSS Smoke",
        "",
        f"- Frozen candidates: {count} independent scenes, {analysis['complete_case_count']} complete HFSS cases.",
        f"- No-scale reconstruction: NMSE max {analysis['complex_nmse_max']:.3e}; magnitude RMSE max {analysis['magnitude_rmse_db_max']:.3e} dB.",
        f"- Strict pattern gate: {summary['fullwave_pattern_gate_count']}/{count}.",
        f"- Combined plus significant-task active-RL gate: {summary['semantic_active_rl_gate_count']}/{count}.",
        f"- Final engineering smoke gate: {final_count}/{count} ({precision:.1%}).",
        f"- K=6 positives: {k6_positive}; all sparse multibeam positives: {sparse_multibeam_positive}.",
        f"- Legacy all-nonzero task diagnostic: {summary['all_nonzero_task_diagnostic_gate_count']}/{count}; disagreements: {summary['all_nonzero_semantic_disagreement_count']}.",
        "",
        "## Decision",
        "",
        "The frozen small HFSS smoke passes." if smoke_pass else "The frozen small HFSS smoke fails.",
        "It validates the trusted nominal 16x16 saved-field basis under frozen source and calibration perturbations.",
        "It does not contain perturbed 16x16 frequency, geometry, dielectric, or S-parameter operators, so it is not a new hidden-physics critic label set.",
    ]
    (args.out_dir / "SMOKE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
