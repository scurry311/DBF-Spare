#!/usr/bin/env python3
"""Fit scene-grouped pooled regularized-isotonic critic calibrators."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from train_fullwave_residual_critic_v2 import binary_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = (
    ROOT / "hfss_outputs" / "trusted_dense_implementation_residual_critic_20260725_run01"
)
DEFAULT_OUT = (
    ROOT / "hfss_outputs" / "trusted_dense_implementation_residual_calibration_20260725_run01"
)
GATES = ("gate15", "gate20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alphas", default="0.01,0.02,0.05,0.10,0.20")
    parser.add_argument("--auroc-min", type=float, default=0.88)
    parser.add_argument("--ece-max", type=float, default=0.08)
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


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(value / (1.0 - value))


def sigmoid(logit: np.ndarray) -> np.ndarray:
    value = np.asarray(logit, dtype=np.float64)
    return np.where(
        value >= 0.0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        return float(array[0]), 0.0
    return (
        float(np.mean(array)),
        float(1.96 * np.std(array, ddof=1) / math.sqrt(array.size)),
    )


def regularized_prediction(
    model: IsotonicRegression, raw_probability: np.ndarray, alpha: float
) -> np.ndarray:
    isotonic = np.asarray(model.predict(raw_probability), dtype=np.float64)
    return np.clip(
        (1.0 - float(alpha)) * isotonic + float(alpha) * raw_probability,
        1.0e-6,
        1.0 - 1.0e-6,
    )


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite calibration output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads((args.train_dir / "five_seed_summary.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in aggregate["seeds"]]
    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    seed_data: dict[int, dict[str, Any]] = {}
    reference_signature: list[tuple[int, str, int]] | None = None
    for seed in seeds:
        summary = json.loads(
            (args.train_dir / f"seed_{seed}" / "run_summary.json").read_text(encoding="utf-8")
        )
        rows = read_csv(args.train_dir / f"seed_{seed}" / "predictions.csv")
        signature = [
            (int(row["row_index"]), row["split"], int(row["sample_index"])) for row in rows
        ]
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError("Prediction rows differ across seeds")
        payload: dict[str, Any] = {
            "rows": rows,
            "split": np.asarray([row["split"] for row in rows]),
            "sample_index": np.asarray([int(row["sample_index"]) for row in rows]),
        }
        for gate in GATES:
            calibrated_probability = np.asarray(
                [float(row[f"prob_{gate}"]) for row in rows], dtype=np.float64
            )
            raw_logit = probability_to_logit(calibrated_probability) * float(
                summary["temperatures"][gate]
            )
            payload[f"raw_{gate}"] = sigmoid(raw_logit)
            payload[f"true_{gate}"] = np.asarray(
                [int(row[f"true_{gate}"]) for row in rows], dtype=np.int8
            )
        seed_data[seed] = payload

    calibrators: dict[str, dict[str, Any]] = {}
    cross_validation_rows: list[dict[str, Any]] = []
    for gate in GATES:
        validation_probability = np.concatenate(
            [
                seed_data[seed][f"raw_{gate}"][seed_data[seed]["split"] == "val"]
                for seed in seeds
            ]
        )
        validation_label = np.concatenate(
            [
                seed_data[seed][f"true_{gate}"][seed_data[seed]["split"] == "val"]
                for seed in seeds
            ]
        )
        validation_group = np.concatenate(
            [
                seed_data[seed]["sample_index"][seed_data[seed]["split"] == "val"]
                for seed in seeds
            ]
        )
        unique_groups = np.unique(validation_group)
        folds = min(int(args.folds), int(unique_groups.size))
        if folds < 2:
            raise ValueError("At least two independent validation scenes are required")
        group_kfold = GroupKFold(n_splits=folds)
        alpha_metrics: list[dict[str, Any]] = []
        for alpha in alphas:
            prediction = np.zeros(validation_probability.size, dtype=np.float64)
            for train_index, test_index in group_kfold.split(
                validation_probability, validation_label, validation_group
            ):
                model = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
                model.fit(validation_probability[train_index], validation_label[train_index])
                prediction[test_index] = regularized_prediction(
                    model, validation_probability[test_index], alpha
                )
            metrics = binary_metrics(validation_label, probability_to_logit(prediction))
            record = {"gate": gate, "alpha": alpha, **metrics}
            alpha_metrics.append(record)
            cross_validation_rows.append(record)
        # NLL is the proper-scoring-rule selection criterion; test labels are never used here.
        chosen = min(alpha_metrics, key=lambda row: (float(row["nll"]), float(row["alpha"])))
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        model.fit(validation_probability, validation_label)
        calibrators[gate] = {
            "alpha": float(chosen["alpha"]),
            "selection_metric": "scene-grouped validation OOF NLL",
            "selection_nll": float(chosen["nll"]),
            "validation_scene_count": int(unique_groups.size),
            "validation_prediction_count": int(validation_probability.size),
            "x_thresholds": np.asarray(model.X_thresholds_, dtype=float).tolist(),
            "y_thresholds": np.asarray(model.y_thresholds_, dtype=float).tolist(),
            "model": model,
        }
    write_csv(args.out_dir / "calibration_cross_validation.csv", cross_validation_rows)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in seeds:
        payload = seed_data[seed]
        test = payload["split"] == "test"
        row: dict[str, Any] = {
            "seed": seed,
            "test_candidate_count": int(np.sum(test)),
            "test_scene_count": int(np.unique(payload["sample_index"][test]).size),
        }
        calibrated_by_gate: dict[str, np.ndarray] = {}
        for gate in GATES:
            calibrator = calibrators[gate]
            probability = regularized_prediction(
                calibrator["model"],
                payload[f"raw_{gate}"],
                float(calibrator["alpha"]),
            )
            calibrated_by_gate[gate] = probability
            metrics = binary_metrics(
                payload[f"true_{gate}"][test], probability_to_logit(probability[test])
            )
            for name, value in metrics.items():
                row[f"{gate}_{name}"] = float(value)
        metric_rows.append(row)
        for index, source_row in enumerate(payload["rows"]):
            prediction_rows.append(
                {
                    "seed": seed,
                    "row_index": int(source_row["row_index"]),
                    "split": source_row["split"],
                    "sample_index": int(source_row["sample_index"]),
                    "true_gate15": int(source_row["true_gate15"]),
                    "raw_probability_gate15": float(payload["raw_gate15"][index]),
                    "pooled_probability_gate15": float(calibrated_by_gate["gate15"][index]),
                    "true_gate20": int(source_row["true_gate20"]),
                    "raw_probability_gate20": float(payload["raw_gate20"][index]),
                    "pooled_probability_gate20": float(calibrated_by_gate["gate20"][index]),
                }
            )
    write_csv(args.out_dir / "pooled_calibrated_five_seed_test_metrics.csv", metric_rows)
    write_csv(args.out_dir / "pooled_calibrated_predictions.csv", prediction_rows)

    aggregate_metrics: dict[str, dict[str, float]] = {}
    for gate in GATES:
        for metric in ("auroc", "auprc", "brier", "ece", "precision", "recall"):
            values = [float(row[f"{gate}_{metric}"]) for row in metric_rows]
            mean, ci = mean_ci(values)
            aggregate_metrics[f"{gate}_{metric}"] = {
                "mean": mean,
                "ci95_half_width": ci,
                "values": values,
            }
    pass_gate = bool(
        aggregate_metrics["gate15_auroc"]["mean"] >= float(args.auroc_min)
        and aggregate_metrics["gate20_auroc"]["mean"] >= float(args.auroc_min)
        and aggregate_metrics["gate15_ece"]["mean"] <= float(args.ece_max)
        and aggregate_metrics["gate20_ece"]["mean"] <= float(args.ece_max)
    )
    serializable_calibrators = {
        gate: {key: value for key, value in payload.items() if key != "model"}
        for gate, payload in calibrators.items()
    }
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_dir": str(args.train_dir),
        "seeds": seeds,
        "method": "pooled scene-grouped regularized isotonic calibration",
        "test_labels_used_for_calibrator_selection": False,
        "calibrators": serializable_calibrators,
        "aggregate_test_metrics": aggregate_metrics,
        "acceptance": {
            "auroc_min": float(args.auroc_min),
            "ece_max": float(args.ece_max),
            "gate15_auroc_pass": aggregate_metrics["gate15_auroc"]["mean"]
            >= float(args.auroc_min),
            "gate20_auroc_pass": aggregate_metrics["gate20_auroc"]["mean"]
            >= float(args.auroc_min),
            "gate15_ece_pass": aggregate_metrics["gate15_ece"]["mean"]
            <= float(args.ece_max),
            "gate20_ece_pass": aggregate_metrics["gate20_ece"]["mean"]
            <= float(args.ece_max),
        },
        "pooled_calibration_gate_pass": pass_gate,
    }
    (args.out_dir / "pooled_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
