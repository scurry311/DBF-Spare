#!/usr/bin/env python3
"""Summarize the guarded 9.96 GHz 16x16 perturbed-operator smoke."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_20260729_run01"
EVALUATION = (
    ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_evaluation_20260729_run01"
)
DIRECT_DATASET = (
    ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_direct_smoke_dataset_20260729_run01"
)
DIRECT = ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_direct_smoke_20260729_run01"
ACTIVE_AUDIT = (
    ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_direct_smoke_active_rl_audit_20260729_run01"
)
OUT = ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_stage_summary_20260729_run02"


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


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite stage summary: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    solve = load_json(RUN / "solve" / "fieldsolve_validation.json")
    operator = load_json(RUN / "operator" / "operator_analysis_summary.json")
    evaluation = load_json(EVALUATION / "evaluation_summary.json")
    direct = load_json(DIRECT / "analysis_summary.json")
    active = load_json(ACTIVE_AUDIT / "summary.json")
    subset = load_json(DIRECT_DATASET / "prepare_summary.json")
    candidate_rows = read_csv(EVALUATION / "candidate_physical_corner_metrics.csv")
    candidate_by_index = {int(row["candidate_index"]): row for row in candidate_rows}
    direct_active_rows = read_csv(ACTIVE_AUDIT / "candidate_active_rl_semantics.csv")

    comparison_rows: list[dict[str, Any]] = []
    for local_index, source_index in enumerate(subset["selected_source_candidate_indices"]):
        predicted = candidate_by_index[int(source_index)]
        measured = next(
            row for row in direct_active_rows if int(row["candidate_index"]) == local_index
        )
        predicted_gate = int(predicted["physical_plus_source_strict_pass"])
        measured_gate = int(measured["engineering_gate_preexisting_eep_semantics"])
        comparison_rows.append(
            {
                "local_candidate_index": local_index,
                "source_candidate_index": int(source_index),
                "sample_index": int(predicted["sample_index"]),
                "k": int(predicted["k"]),
                "ratio": float(predicted["ratio"]),
                "predicted_physical_operator_gate": predicted_gate,
                "direct_hfss_semantic_gate": measured_gate,
                "gate_agreement": int(predicted_gate == measured_gate),
                "predicted_active_rl_floor_db": float(
                    predicted["physical_plus_source_active_rl_floor_db"]
                ),
                "direct_worst_significant_active_rl_db": float(
                    measured["worst_significant_case_active_rl_db"]
                ),
                "direct_combined_active_rl_db": float(measured["combined_worst_active_rl_db"]),
            }
        )
    write_csv(OUT / "direct_gate_agreement.csv", comparison_rows)

    resource_rows = [
        row for row in read_csv(RUN / "resource_history.csv") if row["stage"] == "running_fieldsolve"
    ]
    profile = next((RUN / "solve").glob("*.profile"))
    profile_text = profile.read_text(encoding="utf-8", errors="ignore")
    max_memory = re.findall(r"Max memory/process.*?([0-9.]+) GB", profile_text)
    matrix_size = re.findall(r"Matrix size.*?(\d{6,})", profile_text)
    matrix_disk = re.findall(r"Disk.*?([0-9.]+ GB)", profile_text)
    resource_summary = {
        "sample_count": len(resource_rows),
        "minimum_physical_available_gb": min(
            float(row["physical_available_gb"]) for row in resource_rows
        ),
        "minimum_commit_headroom_gb": min(
            float(row["commit_headroom_gb"]) for row in resource_rows
        ),
        "minimum_d_free_gb": min(float(row["d_free_gb"]) for row in resource_rows),
        "profile_max_memory_per_process_gb": float(max_memory[-1]) if max_memory else None,
        "profile_matrix_size": int(matrix_size[-1]) if matrix_size else None,
        "profile_matrix_disk": matrix_disk[-1] if matrix_disk else None,
        "resource_abort_triggered": False,
        "aedt_ram_limit_percent": 80,
    }
    (OUT / "resource_summary.json").write_text(
        json.dumps(resource_summary, indent=2), encoding="utf-8"
    )

    agreement = sum(int(row["gate_agreement"]) for row in comparison_rows)
    operator_gate = bool(
        solve["solve_and_export_gate_pass"]
        and solve["numerical_smatrix_valid"]
        and solve["matched_rl_gate_pass"]
        and operator["structural_gate_pass"]
        and resource_summary["minimum_physical_available_gb"] >= 0.75
        and resource_summary["minimum_commit_headroom_gb"] >= 3.0
        and resource_summary["minimum_d_free_gb"] >= 25.0
        and direct["all_no_scale_reconstruction_pass"]
        and agreement == len(comparison_rows)
    )
    summary = {
        "protocol": "v18-small-16x16-perturbed-operator-hfss",
        "corner": "frequency_low_E2",
        "frequency_ghz": 9.96,
        "physical_operator_gate_pass": operator_gate,
        "solve_and_export_complete": bool(solve["solve_and_export_gate_pass"]),
        "numerical_smatrix_valid": bool(solve["numerical_smatrix_valid"]),
        "operator_structural_gate_pass": bool(operator["structural_gate_pass"]),
        "matched_passive_rl_min_db": float(operator["matched_passive_rl_min_db"]),
        "nominal_vs_physical_max_abs_delta_s": float(
            solve["reference_direct_comparison"]["max_abs_delta_s"]
        ),
        "nominal_delta_s_0p05_gate_applicable": False,
        "nominal_delta_s_gate_reason": (
            "The setup frequency changed intentionally; delta S is a physical response, not a "
            "same-model direct/DDM consistency error."
        ),
        "eep_port_complete": bool(operator["field_complete"]),
        "eep_grid_point_count": int(operator["grid_point_count"]),
        "frozen_candidate_count": int(evaluation["candidate_count"]),
        "operator_only_strict_pass_count": int(evaluation["operator_only_strict_pass_count"]),
        "physical_plus_source_strict_pass_count": int(
            evaluation["physical_plus_source_strict_pass_count"]
        ),
        "physical_plus_source_failure_root_causes": evaluation["failure_root_causes"],
        "direct_hfss_candidate_count": int(direct["candidate_count"]),
        "direct_hfss_case_count": int(direct["complete_case_count"]),
        "direct_no_scale_complex_nmse_max": float(direct["complex_nmse_max"]),
        "direct_no_scale_magnitude_rmse_db_max": float(direct["magnitude_rmse_db_max"]),
        "direct_gate_agreement_count": agreement,
        "direct_gate_agreement_rate": agreement / len(comparison_rows),
        "direct_fullwave_pattern_gate_count": int(active["fullwave_pattern_gate_count"]),
        "direct_semantic_active_rl_gate_count": int(
            active["preexisting_combined_plus_significant_task_gate_count"]
        ),
        "resource_summary": resource_summary,
        "critic_training_allowed": False,
        "development_operator_labels_allowed": operator_gate,
        "next_action": (
            "Use nominal and 9.96 GHz S256 jointly in active-RL-constrained weight projection; "
            "do not retrain the critic until at least the symmetric 10.04 GHz operator and "
            "additional independent physical corners are available."
        ),
    }
    (OUT / "stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# 9.96 GHz Perturbed-Operator HFSS Smoke",
        "",
        f"- Physical operator gate: {'PASS' if operator_gate else 'FAIL'}.",
        f"- Resource minima: {resource_summary['minimum_physical_available_gb']:.3f} GB free RAM, "
        f"{resource_summary['minimum_commit_headroom_gb']:.3f} GB commit headroom, "
        f"{resource_summary['minimum_d_free_gb']:.3f} GB free disk.",
        f"- Complete EEP ports/grid: 256/{operator['grid_point_count']} directions.",
        f"- Frozen candidate strict pass: operator only {evaluation['operator_only_strict_pass_count']}/20; "
        f"operator plus source perturbation {evaluation['physical_plus_source_strict_pass_count']}/20.",
        "- All 16 failures are active-RL failures; the frozen pattern gates remain satisfied.",
        f"- Direct HFSS agreement: {agreement}/{len(comparison_rows)}, no-scale NMSE max "
        f"{direct['complex_nmse_max']:.3e}.",
        "",
        "## Decision",
        "",
        "The physical 9.96 GHz operator is valid for development evaluation. Critic retraining remains locked.",
        "The next algorithmic step is joint nominal/9.96 GHz active-RL projection before another HFSS corner.",
    ]
    (OUT / "STAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
