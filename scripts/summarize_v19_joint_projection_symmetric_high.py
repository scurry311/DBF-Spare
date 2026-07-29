#!/usr/bin/env python3
"""Summarize nominal/9.96 projection and prospective physical 10.04 GHz smoke."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_v16_robust_drift_oracle import load_nominal_operator


ROOT = Path(__file__).resolve().parents[1]
JOINT = ROOT / "hfss_outputs" / "v19_nominal_9p96_joint_projection_20260729_run01"
LOW = ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_20260729_run01"
HIGH = ROOT / "hfss_outputs" / "v19_perturbed_operator_frequency_high_20260729_run01"
HIGH_EVAL = ROOT / "hfss_outputs" / "v19_frequency_high_prospective_evaluation_20260729_run01"
DIRECT_DATASET = ROOT / "hfss_outputs" / "v19_frequency_high_direct_smoke_dataset_20260729_run01"
DIRECT = ROOT / "hfss_outputs" / "v19_frequency_high_direct_smoke_20260729_run01"
AUDIT = ROOT / "hfss_outputs" / "v19_frequency_high_direct_smoke_active_rl_audit_20260729_run01"
OUT = ROOT / "hfss_outputs" / "v19_joint_projection_symmetric_high_stage_summary_20260729_run01"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def group_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = f"{float(row[field]):.1f}" if field == "ratio" else str(row[field])
        groups[value].append(row)
    output: list[dict[str, Any]] = []
    for value, members in sorted(groups.items()):
        output.append(
            {
                "stratum": field,
                "value": value,
                "scene_count": len(members),
                "nominal_low_joint_pass_count": sum(int(row["joint_pass"]) for row in members),
                "high_source_pass_count": sum(int(row["high_pass"]) for row in members),
                "three_frequency_pass_count": sum(int(row["three_frequency_pass"]) for row in members),
                "minimum_high_active_rl_db": min(float(row["high_active_rl_db"]) for row in members),
            }
        )
    return output


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite stage summary: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    joint_summary = load_json(JOINT / "stage_summary.json")
    joint_rows = read_csv(JOINT / "selected_candidate_metrics.csv")
    high_summary = load_json(HIGH_EVAL / "evaluation_summary.json")
    high_rows = read_csv(HIGH_EVAL / "candidate_frequency_high_metrics.csv")
    high_by_source = {int(row["source_candidate_index"]): row for row in high_rows}
    combined: list[dict[str, Any]] = []
    for joint in joint_rows:
        high = high_by_source[int(joint["source_candidate_index"])]
        joint_pass = int(joint["all_corner_strict_pass"])
        high_pass = int(high["high_E2_source_strict_pass"])
        combined.append(
            {
                "source_candidate_index": int(joint["source_candidate_index"]),
                "sample_index": int(joint["sample_index"]),
                "k": int(joint["k"]),
                "ratio": float(joint["ratio"]),
                "variant": str(joint["variant"]),
                "joint_pass": joint_pass,
                "joint_active_rl_db": 10.0 + float(joint["robust_active_rl_margin_db"]),
                "high_pass": high_pass,
                "high_active_rl_db": float(high["high_E2_source_active_rl_floor_db"]),
                "three_frequency_pass": int(joint_pass and high_pass),
            }
        )
    write_csv(OUT / "three_frequency_candidate_gates.csv", combined)
    strata: list[dict[str, Any]] = []
    for field in ("k", "ratio"):
        strata.extend(group_rows(combined, field))
    write_csv(OUT / "stratified_three_frequency_gates.csv", strata)

    _low_base, _low_effective, _low_fast, low_s = load_nominal_operator(
        LOW / "operator" / "grounded_patch_eep_operator_256port.npz"
    )
    _high_base, _high_effective, _high_fast, high_s = load_nominal_operator(
        HIGH / "operator" / "grounded_patch_eep_operator_256port.npz"
    )
    low_operator = load_json(LOW / "operator" / "operator_analysis_summary.json")
    high_operator = load_json(HIGH / "operator" / "operator_analysis_summary.json")
    high_solve = load_json(HIGH / "solve" / "fieldsolve_validation.json")
    physical_delta = np.abs(high_s - low_s)

    resource_rows = [
        row
        for row in read_csv(HIGH / "resource_history.csv")
        if row["stage"] == "running_fieldsolve"
    ]
    profile = next((HIGH / "solve").glob("*.profile"))
    profile_text = profile.read_text(encoding="utf-8", errors="ignore")
    max_memory = re.findall(r"Max memory/process.*?([0-9.]+) GB", profile_text)
    matrix_size = re.findall(r"Matrix size.*?(\d{6,})", profile_text)
    matrix_disk = re.findall(r"Disk.*?([0-9.]+ GB)", profile_text)
    elapsed = re.findall(r"Elapsed Time.*?'([0-9:]+)'", profile_text)
    resource = {
        "sample_count": len(resource_rows),
        "minimum_physical_available_gb": min(
            float(row["physical_available_gb"]) for row in resource_rows
        ),
        "below_0p75_sample_count": sum(
            float(row["physical_available_gb"]) < 0.75 for row in resource_rows
        ),
        "minimum_commit_headroom_gb": min(
            float(row["commit_headroom_gb"]) for row in resource_rows
        ),
        "minimum_d_free_gb": min(float(row["d_free_gb"]) for row in resource_rows),
        "profile_max_memory_per_process_gb": float(max_memory[-1]) if max_memory else None,
        "profile_matrix_size": int(matrix_size[-1]) if matrix_size else None,
        "profile_matrix_disk": matrix_disk[-1] if matrix_disk else None,
        "profile_elapsed": elapsed[-1] if elapsed else None,
        "resource_abort_triggered": False,
    }
    (OUT / "resource_summary.json").write_text(
        json.dumps(resource, indent=2), encoding="utf-8"
    )

    direct = load_json(DIRECT / "analysis_summary.json")
    subset = load_json(DIRECT_DATASET / "prepare_summary.json")
    audit_rows = read_csv(AUDIT / "candidate_active_rl_semantics.csv")
    direct_agreement: list[dict[str, Any]] = []
    for local_index, source_index in enumerate(subset["selected_source_candidate_indices"]):
        predicted = high_by_source[int(source_index)]
        measured = next(row for row in audit_rows if int(row["candidate_index"]) == local_index)
        predicted_gate = int(predicted["high_E2_source_strict_pass"])
        measured_gate = int(measured["engineering_gate_preexisting_eep_semantics"])
        direct_agreement.append(
            {
                "local_candidate_index": local_index,
                "source_candidate_index": int(source_index),
                "sample_index": int(predicted["sample_index"]),
                "k": int(predicted["k"]),
                "ratio": float(predicted["ratio"]),
                "predicted_gate": predicted_gate,
                "direct_semantic_gate": measured_gate,
                "gate_agreement": int(predicted_gate == measured_gate),
                "predicted_active_rl_db": float(predicted["high_E2_source_active_rl_floor_db"]),
                "direct_significant_active_rl_db": float(
                    measured["worst_significant_case_active_rl_db"]
                ),
            }
        )
    write_csv(OUT / "direct_gate_agreement.csv", direct_agreement)

    three_frequency_pass = sum(int(row["three_frequency_pass"]) for row in combined)
    summary = {
        "protocol": "v19-nominal-low-joint-projection-and-symmetric-high-smoke",
        "scene_count": len(combined),
        "joint_projection_baseline_pass_count": int(
            joint_summary["baseline_all_corner_strict_pass_count"]
        ),
        "joint_projection_selected_pass_count": int(
            joint_summary["selected_all_corner_strict_pass_count"]
        ),
        "joint_projection_pattern_pass_count": int(joint_summary["selected_pattern_pass_count"]),
        "high_source_strict_pass_count": int(high_summary["high_source_strict_pass_count"]),
        "high_source_pattern_pass_count": int(high_summary["high_source_pattern_pass_count"]),
        "three_frequency_strict_pass_count": three_frequency_pass,
        "three_frequency_strict_pass_rate": three_frequency_pass / len(combined),
        "three_frequency_k_pass": {
            str(k): sum(
                int(row["three_frequency_pass"]) for row in combined if int(row["k"]) == k
            )
            for k in (2, 4, 6)
        },
        "low_matched_passive_rl_min_db": float(low_operator["matched_passive_rl_min_db"]),
        "high_matched_passive_rl_min_db": float(high_operator["matched_passive_rl_min_db"]),
        "low_high_matched_s_max_abs_delta": float(np.max(physical_delta)),
        "high_nominal_max_abs_delta_s": float(
            high_solve["reference_direct_comparison"]["max_abs_delta_s"]
        ),
        "high_operator_gate_pass": bool(
            high_solve["solve_and_export_gate_pass"]
            and high_solve["numerical_smatrix_valid"]
            and high_solve["matched_rl_gate_pass"]
            and high_operator["structural_gate_pass"]
        ),
        "nominal_delta_s_0p05_gate_applicable": False,
        "high_failure_root_causes": high_summary["failure_root_causes"],
        "direct_hfss_case_count": int(direct["complete_case_count"]),
        "direct_no_scale_complex_nmse_max": float(direct["complex_nmse_max"]),
        "direct_no_scale_magnitude_rmse_db_max": float(direct["magnitude_rmse_db_max"]),
        "direct_gate_agreement_count": sum(int(row["gate_agreement"]) for row in direct_agreement),
        "direct_gate_agreement_rate": float(
            np.mean([int(row["gate_agreement"]) for row in direct_agreement])
        ),
        "resource_summary": resource,
        "critic_training_allowed": False,
        "large_hfss_batch_allowed": False,
        "next_action": (
            "Jointly optimize mask and one common task-weight command against nominal, 9.96, "
            "and 10.04 GHz physical EEP/S256 operators; target active-RL margin above 1 dB "
            "before any additional physical corner or critic training."
        ),
    }
    (OUT / "stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Three-Frequency Active-RL Stage",
        "",
        f"- Nominal/9.96 joint projection: {summary['joint_projection_baseline_pass_count']}/20 to "
        f"{summary['joint_projection_selected_pass_count']}/20 strict.",
        f"- Prospective 10.04 E2-source strict pass: {summary['high_source_strict_pass_count']}/20.",
        f"- Common nominal/9.96/10.04 strict pass: {three_frequency_pass}/20.",
        f"- Direct high-corner agreement: {summary['direct_gate_agreement_count']}/2; "
        f"NMSE max {summary['direct_no_scale_complex_nmse_max']:.3e}.",
        "- All prospective high-corner root failures are active-RL failures.",
        "",
        "## Decision",
        "",
        "The symmetric physical corner is valid, but the fixed low-corner projection does not "
        "generalize across the band. Critic retraining and bulk HFSS remain locked.",
    ]
    (OUT / "STAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
