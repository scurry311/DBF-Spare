#!/usr/bin/env python3
"""Evaluate the gated dense-local EEP/HFSS shortlist without a 96-row assumption."""

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
DEFAULT_DENSE = ROOT / "hfss_outputs" / "trusted_dense_local_eep_joint_20260724_run02"
DEFAULT_DATASET = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_dataset_20260724_run01"
DEFAULT_HFSS = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_smoke_20260724_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_decision_20260724_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hfss-dir", type=Path, default=DEFAULT_HFSS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-hfss-cases", type=int, default=50)
    parser.add_argument("--max-hfss-cases", type=int, default=100)
    parser.add_argument("--min-sparse-multibeam", type=int, default=5)
    parser.add_argument("--min-sparse-k6", type=int, default=1)
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


def residual_stats(rows: list[dict[str, str]], key: str) -> dict[str, float]:
    values = np.asarray([value(row, key) for row in rows], dtype=float)
    return {
        "mean_db": float(np.mean(values)),
        "std_db": float(np.std(values)),
        "max_abs_db": float(np.max(np.abs(values))),
    }


def group_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        k_value = str(integer(row, "k"))
        ratio = f"{value(row, 'ratio'):.1f}"
        scan = str(integer(row, "large_scan"))
        for key in (
            ("all", "all", "all"),
            (k_value, "all", "all"),
            ("all", ratio, "all"),
            (k_value, ratio, "all"),
            (k_value, ratio, scan),
        ):
            groups[key].append(row)

    output: list[dict[str, Any]] = []
    for (k_value, ratio, scan), members in sorted(groups.items()):
        output.append(
            {
                "k": k_value,
                "ratio": ratio,
                "large_scan": scan,
                "candidate_count": len(members),
                "hfss_case_count": sum(1 + integer(row, "k") for row in members),
                "gate15_count": sum(integer(row, "gate15") for row in members),
                "strict_gate20_count": sum(integer(row, "strict_gate20") for row in members),
                "mainlobe_gate_count": sum(integer(row, "mainlobe_gate") for row in members),
                "active_RL_gate_count": sum(integer(row, "active_RL_gate") for row in members),
                "strict_engineering_gate_count": sum(
                    integer(row, "strict_engineering_gate") for row in members
                ),
                "strict_engineering_gate_rate": float(
                    np.mean([integer(row, "strict_engineering_gate") for row in members])
                ),
                "hfss_psll_mean_db": float(np.mean([value(row, "hfss_psll_db") for row in members])),
                "hfss_nearest_iso_min_db": float(
                    np.min([value(row, "hfss_nearest_iso_db") for row in members])
                ),
                "hfss_local_iso_min_db": float(
                    np.min([value(row, "hfss_local_iso_db") for row in members])
                ),
                "hfss_active_rl_min_db": float(
                    np.min([value(row, "all_case_worst_active_rl_db") for row in members])
                ),
                "delta_psll_max_abs_db": float(
                    np.max(np.abs([value(row, "delta_psll_db") for row in members]))
                ),
                "delta_local_iso_max_abs_db": float(
                    np.max(np.abs([value(row, "delta_local_iso_db") for row in members]))
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite smoke decision: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dense_summary = json.loads(
        (args.dense_dir / "dense_refinement_summary.json").read_text(encoding="utf-8")
    )
    prepare_summary = json.loads(
        (args.dataset_dir / "prepare_summary.json").read_text(encoding="utf-8")
    )
    hfss_summary = json.loads(
        (args.hfss_dir / "analysis_summary.json").read_text(encoding="utf-8")
    )
    rows = read_csv(args.hfss_dir / "candidate_residual_labels.csv")
    groups = group_rows(rows)

    case_count = int(hfss_summary["complete_case_count"])
    all_sparse = all(value(row, "ratio") < 0.999 for row in rows)
    all_multibeam = all(integer(row, "k") in (2, 4, 6) for row in rows)
    strict_positive_count = sum(integer(row, "strict_engineering_gate") for row in rows)
    sparse_multibeam_positive_count = sum(
        integer(row, "strict_engineering_gate")
        and value(row, "ratio") < 0.999
        and integer(row, "k") in (2, 4, 6)
        for row in rows
    )
    sparse_k6_positive_count = sum(
        integer(row, "strict_engineering_gate")
        and value(row, "ratio") < 0.999
        and integer(row, "k") == 6
        for row in rows
    )
    shortlist_open_gate = bool(
        int(dense_summary["sparse_multibeam_strict_positive_count"])
        >= int(args.min_sparse_multibeam)
        and int(dense_summary["sparse_k6_strict_positive_count"])
        >= int(args.min_sparse_k6)
        and int(args.min_hfss_cases) <= case_count <= int(args.max_hfss_cases)
    )
    physical_label_collection_allowed = bool(
        shortlist_open_gate
        and int(hfss_summary["expected_case_count"]) == case_count
        and bool(hfss_summary["all_no_scale_reconstruction_pass"])
        and bool(hfss_summary["scene_leakage_free"])
        and all_sparse
        and all_multibeam
        and strict_positive_count == len(rows)
    )
    residuals = {
        "delta_psll": residual_stats(rows, "delta_psll_db"),
        "delta_nearest_iso": residual_stats(rows, "delta_nearest_iso_db"),
        "delta_local_iso": residual_stats(rows, "delta_local_iso_db"),
        "delta_mainlobe_gain": residual_stats(rows, "delta_mainlobe_gain_db"),
    }
    residual_signal_max_std = max(item["std_db"] for item in residuals.values())
    residual_critic_training_allowed = bool(
        physical_label_collection_allowed
        and int(hfss_summary["hard_negative_count"]) > 0
        and residual_signal_max_std >= 0.05
    )

    decision = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dense_candidate_count": int(dense_summary["candidate_count"]),
        "dense_sparse_multibeam_strict_positive_count": int(
            dense_summary["sparse_multibeam_strict_positive_count"]
        ),
        "dense_sparse_k6_strict_positive_count": int(
            dense_summary["sparse_k6_strict_positive_count"]
        ),
        "hfss_candidate_count": len(rows),
        "hfss_expected_case_count": int(hfss_summary["expected_case_count"]),
        "hfss_complete_case_count": case_count,
        "hfss_strict_engineering_positive_count": strict_positive_count,
        "hfss_sparse_multibeam_positive_count": sparse_multibeam_positive_count,
        "hfss_sparse_k6_positive_count": sparse_k6_positive_count,
        "hfss_k_counts": prepare_summary["k_counts"],
        "ratio1_included": not all_sparse,
        "all_no_scale_reconstruction_pass": bool(
            hfss_summary["all_no_scale_reconstruction_pass"]
        ),
        "complex_nmse_max": float(hfss_summary["complex_nmse_max"]),
        "magnitude_rmse_db_max": float(hfss_summary["magnitude_rmse_db_max"]),
        "shortlist_open_gate_pass": shortlist_open_gate,
        "hfss_50_100_case_task_completed": bool(
            int(args.min_hfss_cases) <= case_count <= int(args.max_hfss_cases)
        ),
        "hfss_physical_label_collection_allowed": physical_label_collection_allowed,
        "residual_critic_training_allowed": residual_critic_training_allowed,
        "residuals": residuals,
        "legacy_labels_allowed_explanation": (
            "The generic validator's labels_allowed flag requires exactly 96 candidates. "
            "This gated shortlist intentionally contains 15 candidates and 65 HFSS cases; "
            "the decision above applies the experiment-specific acceptance conditions."
        ),
        "decision": (
            "accept_dense_local_hfss_positive_labels_hold_residual_critic"
            if physical_label_collection_allowed and not residual_critic_training_allowed
            else "accept_and_train_residual_critic"
            if residual_critic_training_allowed
            else "reject_dense_local_hfss_labels"
        ),
        "next_action": (
            "Retain these samples as trusted sparse positive and boundary labels. Add paired "
            "near-boundary perturbations and EEP-pass/HFSS-fail hard negatives before training "
            "the residual critic; the present residuals are numerical-noise scale."
        ),
    }
    write_csv(args.out_dir / "dense_joint_hfss_group_summary.csv", groups)
    write_csv(args.out_dir / "dense_joint_hfss_candidate_labels.csv", rows)
    (args.out_dir / "dense_joint_hfss_smoke_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
