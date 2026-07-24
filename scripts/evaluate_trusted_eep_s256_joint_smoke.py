#!/usr/bin/env python3
"""Summarize the trusted 96-candidate joint-optimization smoke and gate HFSS."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPT = ROOT / "hfss_outputs" / "trusted_eep_s256_joint_optimization_20260724_run03"
DEFAULT_AUDIT = ROOT / "hfss_outputs" / "trusted_active_rl_audit_20260724_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_eep_s256_joint_smoke_decision_20260724_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-dir", type=Path, default=DEFAULT_OPT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def count(rows: list[dict[str, str]], key: str) -> int:
    return sum(integer(row, key) for row in rows)


def group_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        keys = (
            ("all", "all", "all"),
            (row["k"], "all", "all"),
            (row["k"], f"{value(row, 'ratio'):.1f}", "all"),
            (row["k"], f"{value(row, 'ratio'):.1f}", row["large_scan"]),
        )
        for key in keys:
            groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (k_value, ratio, large_scan), members in sorted(groups.items()):
        improvement = np.asarray(
            [
                value(row, "optimized_combined_worst_active_rl_db")
                - value(row, "baseline_combined_worst_active_rl_db")
                for row in members
            ]
        )
        output.append(
            {
                "k": k_value,
                "ratio": ratio,
                "large_scan": large_scan,
                "candidate_count": len(members),
                "combined_active_gate_count": count(members, "optimized_combined_active_gate"),
                "combined_active_gate_rate": float(
                    np.mean([integer(row, "optimized_combined_active_gate") for row in members])
                ),
                "robust_active_gate_count": count(members, "robust_active_RL_gate"),
                "robust_active_gate_rate": float(
                    np.mean([integer(row, "robust_active_RL_gate") for row in members])
                ),
                "gate15_count": count(members, "gate15"),
                "strict_gate20_count": count(members, "strict_gate20"),
                "mainlobe_gate_count": count(members, "mainlobe_gate"),
                "strict_engineering_gate_count": count(members, "strict_engineering_gate"),
                "strict_engineering_gate_rate": float(
                    np.mean([integer(row, "strict_engineering_gate") for row in members])
                ),
                "mean_active_rl_improvement_db": float(np.mean(improvement)),
                "max_active_rl_improvement_db": float(np.max(improvement)),
                "mean_equal_weakest_gain_power_ratio": float(
                    np.mean([value(row, "equal_weakest_gain_power_ratio") for row in members])
                ),
            }
        )
    return output


def scene_oracle(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[integer(row, "sample_index")].append(row)
    output = []
    for sample_index, members in sorted(groups.items()):
        sparse = [row for row in members if value(row, "ratio") < 0.999]
        positive = [row for row in sparse if integer(row, "strict_engineering_gate")]
        selected = min(positive, key=lambda row: value(row, "ratio")) if positive else None
        output.append(
            {
                "sample_index": sample_index,
                "k": max(integer(row, "k") for row in members),
                "candidate_count": len(members),
                "sparse_candidate_count": len(sparse),
                "sparse_oracle_pass": int(selected is not None),
                "minimum_feasible_ratio": value(selected, "ratio") if selected else "",
                "selected_candidate_index": integer(selected, "candidate_index") if selected else "",
            }
        )
    return output


def priority_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        strict = integer(row, "strict_engineering_gate") == 1
        sparse = value(row, "ratio") < 0.999
        robust = integer(row, "robust_active_RL_gate") == 1
        k_value = integer(row, "k")
        if strict and sparse:
            role = "sparse_joint_positive"
            priority = 100.0 - 10.0 * value(row, "ratio")
        elif strict:
            role = "ratio1_control_positive"
            priority = 50.0
        elif robust and k_value == 6:
            role = "k6_active_positive_direction_failure"
            priority = 40.0 + value(row, "optimized_local_iso_db") / 10.0
        elif robust:
            margins = (
                abs(value(row, "optimized_psll_db")) / 3.0,
                abs(value(row, "optimized_nearest_iso_db") - 25.0) / 5.0,
                abs(value(row, "optimized_local_iso_db") - 20.0) / 5.0,
            )
            role = "active_positive_direction_boundary"
            priority = 30.0 - min(margins)
        else:
            continue
        output.append(
            {
                "priority": priority,
                "role": role,
                "candidate_index": integer(row, "candidate_index"),
                "sample_index": integer(row, "sample_index"),
                "k": k_value,
                "ratio": value(row, "ratio"),
                "large_scan": integer(row, "large_scan"),
                "min_target_separation_deg": value(row, "min_target_separation_deg"),
                "combined_active_rl_db": value(row, "optimized_combined_worst_active_rl_db"),
                "task_significant_active_rl_db": value(
                    row, "optimized_all_tasks_significant_worst_active_rl_db"
                ),
                "psll_db": value(row, "optimized_psll_db"),
                "nearest_iso_db": value(row, "optimized_nearest_iso_db"),
                "local_iso_db": value(row, "optimized_local_iso_db"),
                "target_spread_db": value(row, "optimized_target_spread_db"),
                "pointing_error_deg": value(row, "optimized_pointing_error_deg"),
                "strict_engineering_gate": integer(row, "strict_engineering_gate"),
            }
        )
    return sorted(output, key=lambda row: float(row["priority"]), reverse=True)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite smoke decision: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.optimization_dir / "optimization_candidate_metrics.csv")
    audit = json.loads((args.audit_dir / "active_rl_audit_summary.json").read_text(encoding="utf-8"))
    groups = group_summary(rows)
    oracle = scene_oracle(rows)
    priorities = priority_rows(rows)
    write_csv(args.out_dir / "smoke_summary_by_k_ratio_scan.csv", groups)
    write_csv(args.out_dir / "scene_minimum_ratio_oracle.csv", oracle)
    write_csv(args.out_dir / "next_optimization_priority_candidates.csv", priorities)

    sparse_rows = [row for row in rows if value(row, "ratio") < 0.999]
    sparse_joint = [row for row in sparse_rows if integer(row, "strict_engineering_gate")]
    k6_rows = [row for row in rows if integer(row, "k") == 6]
    k6_strict = [row for row in k6_rows if integer(row, "strict_engineering_gate")]
    multi_sparse = [
        row for row in sparse_joint if integer(row, "k") in (2, 4, 6)
    ]
    decision = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(rows),
        "independent_scene_count": len({integer(row, "sample_index") for row in rows}),
        "audit_combined_strict_gate_before_optimization": int(
            audit["definitions"]["strict_mask"]["combined_gate_count"]
        ),
        "combined_active_gate_after_optimization": count(rows, "optimized_combined_active_gate"),
        "robust_active_gate_after_optimization": count(rows, "robust_active_RL_gate"),
        "strict_gate20_after_optimization": count(rows, "strict_gate20"),
        "strict_engineering_gate_after_optimization": count(rows, "strict_engineering_gate"),
        "sparse_strict_engineering_positive_count": len(sparse_joint),
        "sparse_multibeam_strict_engineering_positive_count": len(multi_sparse),
        "k6_strict_engineering_positive_count": len(k6_strict),
        "sparse_scene_oracle_pass_count": sum(int(row["sparse_oracle_pass"]) for row in oracle),
        "generate_new_hfss_training_labels": False,
        "allow_large_hfss_batch": False,
        "decision": "hold_hfss_labels_strengthen_dense_regional_eep_projection",
        "reasons": [
            "K=6 has no strict local-20 plus mainlobe plus active-RL joint positive",
            "sparse multibeam strict positives do not cover K=2 and are limited to one K=4 ratio-0.9 candidate",
            "the nominal EEP/HFSS basis is already numerically identical, so repeating these weights would not create learnable residual labels",
            "the discrete local-null constraints underpredict the full EEP local-5deg peak by roughly 5-10 dB in hard scenes",
        ],
        "next_gate": {
            "method": "dense local-5deg EEP operator plus combined-target equalities, then repeat the 96-candidate smoke",
            "required_k6_sparse_joint_positive_count": 1,
            "required_sparse_multibeam_joint_positive_count": 5,
            "required_before_hfss": "nonzero K=2/4/6 coverage and a 50-100-case oracle shortlist",
        },
    }
    (args.out_dir / "hfss_label_generation_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
