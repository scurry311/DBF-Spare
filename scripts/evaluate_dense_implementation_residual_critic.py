#!/usr/bin/env python3
"""Evaluate promotion gates for the dense implementation-residual critic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "trusted_dense_implementation_residual_dataset_20260724_run01"
DEFAULT_HFSS = ROOT / "hfss_outputs" / "trusted_dense_boundary_hfss_20260724_run01"
DEFAULT_TRAIN = ROOT / "hfss_outputs" / "trusted_dense_implementation_residual_critic_20260724_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_dense_implementation_critic_decision_20260724_run01"
RESIDUAL_NAMES = ("psll_db", "iso_nearest_db", "iso_local_db", "peak_min_db", "peak_spread_db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hfss-dir", type=Path, default=DEFAULT_HFSS)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--auroc-min", type=float, default=0.88)
    parser.add_argument("--ece-max", type=float, default=0.08)
    parser.add_argument("--minimum-test-scenes", type=int, default=10)
    parser.add_argument("--minimum-mainlobe-negatives", type=int, default=5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(np.mean(array)), float(1.96 * np.std(array, ddof=1) / math.sqrt(array.size))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite critic decision: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_json(args.dataset_dir / "build_summary.json")
    hfss = load_json(args.hfss_dir / "analysis_summary.json")
    aggregate = load_json(args.train_dir / "five_seed_summary.json")
    seeds = [int(seed) for seed in aggregate["seeds"]]
    rows: list[dict[str, Any]] = []
    mainlobe_negative_counts: list[int] = []
    for seed in seeds:
        summary = load_json(args.train_dir / f"seed_{seed}" / "run_summary.json")
        test = summary["test"]
        with (args.train_dir / f"seed_{seed}" / "predictions.csv").open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            predictions = [row for row in csv.DictReader(handle) if row["split"] == "test"]
        mainlobe_negative_counts.append(
            sum(int(row["true_mainlobe_gate"]) == 0 for row in predictions)
        )
        row = {
            "seed": seed,
            "test_candidate_count": int(test["n"]),
            "test_scene_count": int(test["scene_count"]),
            "gate15_auroc": float(test["gate15_cal_auroc"]),
            "gate15_auprc": float(test["gate15_cal_auprc"]),
            "gate15_brier": float(test["gate15_cal_brier"]),
            "gate15_ece": float(test["gate15_cal_ece"]),
            "gate15_precision": float(test["gate15_cal_precision"]),
            "gate15_recall": float(test["gate15_cal_recall"]),
            "gate20_auroc": float(test["gate20_cal_auroc"]),
            "gate20_auprc": float(test["gate20_cal_auprc"]),
            "gate20_brier": float(test["gate20_cal_brier"]),
            "gate20_ece": float(test["gate20_cal_ece"]),
            "strict_engineering_auroc": float(test["strict_engineering_gate_cal_auroc"]),
            "strict_engineering_brier": float(test["strict_engineering_gate_cal_brier"]),
            "strict_engineering_ece": float(test["strict_engineering_gate_cal_ece"]),
            "rank_strict_engineering_rate": float(test["rank_strict_engineering_gate_rate"]),
            "conservative_strict_engineering_rate": float(
                test["conservative_strict_engineering_gate_rate"]
            ),
            "mainlobe_negative_count": mainlobe_negative_counts[-1],
        }
        for name in RESIDUAL_NAMES:
            row[f"residual_{name}_rmse"] = float(test[f"residual_{name}_rmse"])
        rows.append(row)
    write_csv(args.out_dir / "five_seed_test_metrics.csv", rows)

    aggregate_metrics: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        if key in {"seed", "test_candidate_count", "test_scene_count", "mainlobe_negative_count"}:
            continue
        mean, ci = mean_ci([float(row[key]) for row in rows])
        aggregate_metrics[key] = {"mean": mean, "ci95_half_width": ci}

    gate15_auc_pass = aggregate_metrics["gate15_auroc"]["mean"] >= float(args.auroc_min)
    gate20_auc_pass = aggregate_metrics["gate20_auroc"]["mean"] >= float(args.auroc_min)
    calibration_pass = bool(
        aggregate_metrics["gate15_ece"]["mean"] <= float(args.ece_max)
        and aggregate_metrics["gate20_ece"]["mean"] <= float(args.ece_max)
    )
    test_scene_count = min(int(row["test_scene_count"]) for row in rows)
    test_support_pass = test_scene_count >= int(args.minimum_test_scenes)
    mainlobe_support_pass = min(mainlobe_negative_counts) >= int(args.minimum_mainlobe_negatives)
    promotion = bool(
        dataset["residual_critic_training_allowed"]
        and hfss["all_no_scale_reconstruction_pass"]
        and int(hfss["hard_negative_count"]) >= 10
        and gate15_auc_pass
        and gate20_auc_pass
        and calibration_pass
        and test_support_pass
        and mainlobe_support_pass
    )
    decision = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_completed": True,
        "seed_count": len(seeds),
        "parameter_count": int(
            load_json(args.train_dir / f"seed_{aggregate['best_seed']}" / "run_summary.json")[
                "parameter_count"
            ]
        ),
        "candidate_count": int(dataset["candidate_count"]),
        "independent_scene_count": int(dataset["independent_scene_count"]),
        "test_scene_count": test_scene_count,
        "hfss_case_count": int(hfss["complete_case_count"]),
        "hfss_hard_negative_count": int(hfss["hard_negative_count"]),
        "actual_eep_to_hfss_complex_nmse_max": float(hfss["complex_nmse_max"]),
        "nominal_to_hfss_magnitude_rmse_db_max": float(
            hfss["nominal_to_direct_magnitude_rmse_db_max"]
        ),
        "aggregate_test_metrics": aggregate_metrics,
        "acceptance": {
            "gate15_auroc_pass": gate15_auc_pass,
            "gate20_auroc_pass": gate20_auc_pass,
            "calibration_ece_pass": calibration_pass,
            "test_scene_support_pass": test_support_pass,
            "mainlobe_negative_support_pass": mainlobe_support_pass,
        },
        "promote_to_engineering_critic": promotion,
        "checkpoint_status": "engineering_promoted" if promotion else "experimental_boundary_critic",
        "failure_reasons": [
            reason
            for failed, reason in (
                (not calibration_pass, "mean calibrated ECE exceeds 0.08"),
                (not test_support_pass, "scene-level test set has fewer than 10 independent scenes"),
                (not mainlobe_support_pass, "mainlobe gate has insufficient negative support"),
            )
            if failed
        ],
        "next_action": (
            "Use this checkpoint only for uncertainty-aware boundary ranking. Add at least "
            "45 new independent scenes with intermediate perturbation levels, low-ratio pairs, "
            "and at least 20 mainlobe-failure examples; then repeat grouped five-seed training."
        ),
    }
    (args.out_dir / "critic_acceptance.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
