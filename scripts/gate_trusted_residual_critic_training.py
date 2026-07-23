#!/usr/bin/env python3
"""Gate residual-critic training and create an auditable null baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "eep_hfss_validation"
)
RESIDUAL_NAMES = (
    "delta_psll_db",
    "delta_nearest_iso_db",
    "delta_local_iso_db",
    "delta_mainlobe_gain_db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--residual-std-floor-db", type=float, default=1.0e-3)
    parser.add_argument("--minimum-hard-negatives", type=int, default=10)
    parser.add_argument("--minimum-active-positives", type=int, default=10)
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


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def main() -> None:
    args = parse_args()
    rows = read_csv(args.dataset_dir / "candidate_residual_labels.csv")
    payload = np.load(args.dataset_dir / "residual_critic_dataset.npz", allow_pickle=False)
    split = np.asarray(payload["split"], dtype=np.int8)
    residuals = np.stack(
        [np.asarray(payload[name], dtype=np.float64) for name in RESIDUAL_NAMES], axis=1
    )
    train = split == 0
    train_mean = np.mean(residuals[train], axis=0)
    train_var = np.maximum(np.var(residuals[train], axis=0), 1.0e-12)
    global_std = np.std(residuals, axis=0)
    hard_negative_count = int(sum(int(row["hard_negative"]) for row in rows))
    hard_positive_count = int(sum(int(row["hard_positive"]) for row in rows))
    near_boundary_count = int(sum(int(row["near_boundary"]) for row in rows))
    active_positive_count = int(sum(int(row["active_RL_gate"]) for row in rows))
    strict_positive_count = int(sum(int(row["strict_engineering_gate"]) for row in rows))

    reasons: list[str] = []
    if float(np.max(global_std)) < float(args.residual_std_floor_db):
        reasons.append("all four HFSS-minus-EEP residuals are below the numerical-learning floor")
    if hard_negative_count < int(args.minimum_hard_negatives):
        reasons.append("insufficient EEP-pass/HFSS-fail hard negatives")
    if active_positive_count < int(args.minimum_active_positives):
        reasons.append("active-RL gate has insufficient positive samples")
    if strict_positive_count == 0:
        reasons.append("strict engineering gate has no positive samples")
    training_allowed = not reasons

    metric_rows: list[dict[str, Any]] = []
    for split_value, split_name in enumerate(("train", "val", "test")):
        indices = split == split_value
        for residual_index, name in enumerate(RESIDUAL_NAMES):
            target = residuals[indices, residual_index]
            constant = np.full_like(target, train_mean[residual_index])
            zero = np.zeros_like(target)
            metric_rows.append(
                {
                    "split": split_name,
                    "residual": name,
                    "n": int(target.size),
                    "target_std_db": float(np.std(target)) if target.size else float("nan"),
                    "train_mean_baseline_rmse_db": float(np.sqrt(np.mean((constant - target) ** 2))) if target.size else float("nan"),
                    "train_mean_baseline_mae_db": float(np.mean(np.abs(constant - target))) if target.size else float("nan"),
                    "zero_baseline_rmse_db": float(np.sqrt(np.mean(target**2))) if target.size else float("nan"),
                }
            )
    write_csv(args.dataset_dir / "null_residual_baseline_metrics.csv", metric_rows)

    np.savez_compressed(
        args.dataset_dir / "null_residual_critic_checkpoint.npz",
        model_type=np.asarray("constant_heteroscedastic_null_residual_baseline"),
        residual_names=np.asarray(RESIDUAL_NAMES),
        residual_mean=train_mean.astype(np.float32),
        residual_log_variance=np.log(train_var).astype(np.float32),
        train_candidate_count=np.asarray(int(np.sum(train)), dtype=np.int64),
        full_neural_training_allowed=np.asarray(training_allowed),
        baseline_scope=np.asarray("numerical reconstruction sanity baseline; not an engineering feasibility critic"),
    )

    priority_rows = sorted(
        rows,
        key=lambda row: (
            -int(float(row["training_priority"])),
            -int(row["hard_negative"]),
            -int(row["near_boundary"]),
            -int(row["hard_positive"]),
            -int(row["k"] == "6"),
            float(row["ratio"]),
        ),
    )
    write_csv(args.dataset_dir / "critic_priority_candidates.csv", priority_rows)

    groups: dict[tuple[int, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["k"]), round(float(row["ratio"]), 1), int(row["large_scan"]))].append(row)
    group_rows: list[dict[str, Any]] = []
    for (k_value, ratio, large_scan), group in sorted(groups.items()):
        values = lambda name: finite([float(row[name]) for row in group])
        group_rows.append(
            {
                "k": k_value,
                "ratio": ratio,
                "large_scan": large_scan,
                "n": len(group),
                "gate15_rate": float(np.mean([int(row["gate15"]) for row in group])),
                "strict_gate20_rate": float(np.mean([int(row["strict_gate20"]) for row in group])),
                "mainlobe_gate_rate": float(np.mean([int(row["mainlobe_gate"]) for row in group])),
                "active_RL_gate_rate": float(np.mean([int(row["active_RL_gate"]) for row in group])),
                "psll_mean_db": float(np.mean(values("hfss_psll_db"))),
                "nearest_iso_mean_db": float(np.mean(values("hfss_nearest_iso_db"))),
                "local_iso_mean_db": float(np.mean(values("hfss_local_iso_db"))),
                "worst_active_rl_mean_db": float(np.mean(values("all_case_worst_active_rl_db"))),
                "worst_active_rl_max_db": float(np.max(values("all_case_worst_active_rl_db"))),
            }
        )
    write_csv(args.dataset_dir / "fullwave_summary_by_k_ratio_scan.csv", group_rows)

    decision = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "full_neural_residual_critic_training_allowed": training_allowed,
        "decision": "train_residual_critic" if training_allowed else "hold_neural_critic_use_null_baseline",
        "reasons": reasons,
        "residual_std_db": {
            name: float(global_std[index]) for index, name in enumerate(RESIDUAL_NAMES)
        },
        "hard_negative_count": hard_negative_count,
        "hard_positive_count": hard_positive_count,
        "near_boundary_count": near_boundary_count,
        "active_RL_positive_count": active_positive_count,
        "strict_engineering_positive_count": strict_positive_count,
        "null_checkpoint": str(args.dataset_dir / "null_residual_critic_checkpoint.npz"),
        "required_next_labels": [
            "active-RL-constrained EEP/SOCP candidates with nonzero engineering positives",
            "physical geometry, termination, switch, frequency, or manufacturing perturbations that make HFSS differ from the nominal EEP",
            "EEP-pass/HFSS-fail hard negatives and matched hard positives from identical sample_index groups",
        ],
    }
    (args.dataset_dir / "critic_training_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
