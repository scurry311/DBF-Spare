#!/usr/bin/env python3
"""Audit combined, significant-task, and all-nonzero active-RL semantics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from optimize_trusted_eep_s256_joint_weights import active_return


EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hfss-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold-db", type=float, default=10.0)
    parser.add_argument("--task-significant-relative-db", type=float, default=-20.0)
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


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite active-RL audit: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        dataset = {key: source[key] for key in source.files}
    with np.load(args.hfss_dir / "case_excitations.npz", allow_pickle=False) as source:
        cases = {key: source[key] for key in source.files}
    case_manifest = read_csv(args.hfss_dir / "case_manifest.csv")
    candidate_labels = {
        int(row["candidate_index"]): row
        for row in read_csv(args.hfss_dir / "candidate_residual_labels.csv")
    }
    s_matrix = np.asarray(cases["matched_s"], dtype=np.complex128)
    excitations = np.asarray(cases["hfss_actual_external_excitation"], dtype=np.complex128)
    masks = np.asarray(dataset["masks"], dtype=bool)
    threshold = float(args.threshold_db)
    task_relative = float(args.task_significant_relative_db)

    case_rows: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for position, source in enumerate(case_manifest):
        candidate = int(source["candidate_index"])
        case_kind = str(source["case_kind"])
        excitation = excitations[position]
        mask = masks[candidate]
        relative = None if case_kind == "combined" else task_relative
        significant = active_return(
            s_matrix,
            excitation,
            mask,
            relative_db=relative,
            threshold_db=threshold,
        )
        active_amplitude = np.abs(excitation[mask])
        maximum = max(float(np.max(active_amplitude)), EPS)
        nonzero = active_amplitude[active_amplitude >= 1.0e-8]
        minimum_relative_db = float(
            20.0 * np.log10(max(float(np.min(nonzero)), EPS) / maximum)
        )
        row = {
            "case_index": position,
            "case_id": source["case_id"],
            "candidate_index": candidate,
            "sample_index": int(source["sample_index"]),
            "k": int(source["k"]),
            "ratio": float(source["ratio"]),
            "case_kind": case_kind,
            "task_index": int(source["task_index"]),
            "all_nonzero_worst_active_rl_db": float(source["worst_active_rl_db"]),
            "all_nonzero_total_rl_db": float(source["total_rl_db"]),
            "all_nonzero_gate": int(
                float(source["worst_active_rl_db"]) >= threshold
                and float(source["total_rl_db"]) >= threshold
            ),
            "significant_worst_active_rl_db": float(significant["worst_active_rl_db"]),
            "significant_total_rl_db": float(significant["total_rl_db"]),
            "significant_gate": int(significant["gate_pass"]),
            "significant_evaluated_port_count": int(significant["evaluated_port_count"]),
            "nonzero_evaluated_port_count": int(nonzero.size),
            "minimum_nonzero_relative_amplitude_db": minimum_relative_db,
        }
        case_rows.append(row)
        grouped[candidate].append(row)
    write_csv(args.out_dir / "case_active_rl_semantics.csv", case_rows)

    candidate_rows: list[dict[str, Any]] = []
    for candidate in sorted(grouped):
        rows = grouped[candidate]
        combined = next(row for row in rows if row["case_kind"] == "combined")
        label = candidate_labels[candidate]
        all_case_gate = int(all(row["all_nonzero_gate"] for row in rows))
        significant_task_gate = int(all(row["significant_gate"] for row in rows))
        combined_gate = int(combined["all_nonzero_gate"])
        pattern_gate = int(label["strict_gate20"]) * int(label["mainlobe_gate"])
        candidate_rows.append(
            {
                "candidate_index": candidate,
                "sample_index": int(label["sample_index"]),
                "k": int(label["k"]),
                "ratio": float(label["ratio"]),
                "selection_role": label["selection_role"],
                "fullwave_pattern_gate": pattern_gate,
                "combined_active_rl_gate": combined_gate,
                "combined_plus_significant_task_active_rl_gate": significant_task_gate,
                "all_nonzero_task_active_rl_gate": all_case_gate,
                "engineering_gate_combined_operating_state": int(pattern_gate and combined_gate),
                "engineering_gate_preexisting_eep_semantics": int(
                    pattern_gate and significant_task_gate
                ),
                "engineering_gate_all_nonzero_task_diagnostic": int(
                    pattern_gate and all_case_gate
                ),
                "combined_worst_active_rl_db": float(
                    combined["all_nonzero_worst_active_rl_db"]
                ),
                "combined_total_rl_db": float(combined["all_nonzero_total_rl_db"]),
                "worst_significant_case_active_rl_db": min(
                    float(row["significant_worst_active_rl_db"]) for row in rows
                ),
                "worst_all_nonzero_case_active_rl_db": min(
                    float(row["all_nonzero_worst_active_rl_db"]) for row in rows
                ),
            }
        )
    write_csv(args.out_dir / "candidate_active_rl_semantics.csv", candidate_rows)
    count = len(candidate_rows)
    combined_pass = sum(row["engineering_gate_combined_operating_state"] for row in candidate_rows)
    preexisting_pass = sum(
        row["engineering_gate_preexisting_eep_semantics"] for row in candidate_rows
    )
    all_nonzero_pass = sum(
        row["engineering_gate_all_nonzero_task_diagnostic"] for row in candidate_rows
    )
    summary = {
        "candidate_count": count,
        "fullwave_pattern_gate_count": sum(row["fullwave_pattern_gate"] for row in candidate_rows),
        "combined_operating_state_engineering_gate_count": combined_pass,
        "combined_operating_state_engineering_gate_rate": combined_pass / count,
        "preexisting_combined_plus_significant_task_gate_count": preexisting_pass,
        "preexisting_combined_plus_significant_task_gate_rate": preexisting_pass / count,
        "all_nonzero_task_diagnostic_gate_count": all_nonzero_pass,
        "all_nonzero_task_diagnostic_gate_rate": all_nonzero_pass / count,
        "task_significant_relative_db": task_relative,
        "threshold_db": threshold,
        "semantic_disagreement_count": sum(
            row["engineering_gate_preexisting_eep_semantics"]
            != row["engineering_gate_all_nonzero_task_diagnostic"]
            for row in candidate_rows
        ),
        "decision": (
            "Use combined operating-state per-port and total active RL as the hardware gate. "
            "Retain -20 dB significant-task RL as a conservative decomposition diagnostic; "
            "do not gate hardware feasibility on near-zero task decomposition coefficients."
        ),
        "threshold_changed": False,
        "hfss_results_modified": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
