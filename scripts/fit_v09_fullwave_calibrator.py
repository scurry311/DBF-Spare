#!/usr/bin/env python3
"""Fit a regularized Platt calibrator on independent v0.9 HFSS val scenes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_v09_physical_margin_critic import binary_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "v09_hfss_calibration_dataset_20260726_run01"
DEFAULT_HFSS = ROOT / "hfss_outputs" / "v09_hfss_calibration_20260726_run01"
DEFAULT_TEST = ROOT / "hfss_outputs" / "v09_fullwave_evaluation_20260726_run02" / "heldout_candidate_evaluation.csv"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_fullwave_calibrator_20260726_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hfss-dir", type=Path, default=DEFAULT_HFSS)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--regularization", type=float, default=0.02)
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


def transform(probability: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    probability = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    logit = np.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + np.exp(-(float(slope) * logit + float(intercept))))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v0.9 calibrator: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection = read_csv(args.dataset_dir / "calibration_selection.csv")
    labels = read_csv(args.hfss_dir / "candidate_residual_labels.csv")
    if len(selection) != len(labels):
        raise RuntimeError("Calibration selections and HFSS labels differ")
    val_probability = np.asarray(
        [float(row["uncalibrated_strict_probability"]) for row in selection], dtype=float
    )
    val_target = np.asarray(
        [int(row["strict_engineering_gate"]) for row in labels], dtype=int
    )
    if np.unique(val_target).size != 2:
        raise RuntimeError("HFSS calibration set lacks both strict classes")
    best = (float("inf"), 1.0, 0.0)
    for slope in np.linspace(0.25, 2.5, 91):
        for intercept in np.linspace(-2.0, 2.0, 161):
            calibrated = transform(val_probability, slope, intercept)
            nll = -np.mean(
                val_target * np.log(np.maximum(calibrated, 1.0e-9))
                + (1 - val_target) * np.log(np.maximum(1.0 - calibrated, 1.0e-9))
            )
            objective = float(nll) + float(args.regularization) * (
                (float(slope) - 1.0) ** 2 + float(intercept) ** 2
            )
            if objective < best[0]:
                best = (objective, float(slope), float(intercept))
    _, slope, intercept = best
    val_calibrated = transform(val_probability, slope, intercept)
    test_rows = read_csv(args.test_csv)
    test_probability = np.asarray(
        [float(row["mean_strict_probability"]) for row in test_rows], dtype=float
    )
    test_target = np.asarray(
        [int(row["hfss_strict_engineering_gate"]) for row in test_rows], dtype=int
    )
    test_calibrated = transform(test_probability, slope, intercept)
    val_rows = [
        {
            **selection[index],
            "hfss_strict_engineering_gate": int(val_target[index]),
            "calibrated_probability": float(val_calibrated[index]),
        }
        for index in range(len(selection))
    ]
    write_csv(args.out_dir / "validation_calibration.csv", val_rows)
    calibrated_test_rows = [
        {
            **row,
            "calibrated_strict_probability": float(test_calibrated[index]),
        }
        for index, row in enumerate(test_rows)
    ]
    write_csv(args.out_dir / "heldout_calibrated_predictions.csv", calibrated_test_rows)
    before = binary_metrics(test_probability, test_target)
    after = binary_metrics(test_calibrated, test_target)
    calibrator = {
        "type": "regularized_platt",
        "slope": slope,
        "intercept": intercept,
        "probability_threshold": 0.5,
        "regularization": float(args.regularization),
        "fit_scene_count": len(selection),
        "fit_candidate_count": len(selection),
        "fit_positive_count": int(np.sum(val_target)),
        "test_labels_used_for_fit": False,
    }
    (args.out_dir / "calibrator.json").write_text(
        json.dumps(calibrator, indent=2), encoding="utf-8"
    )
    summary = {
        "calibrator": calibrator,
        "validation_metrics": binary_metrics(val_calibrated, val_target),
        "heldout_before": before,
        "heldout_after": after,
        "acceptance": {
            "heldout_auroc_ge_0_88": after["auroc"] >= 0.88,
            "heldout_ece_le_0_08": after["ece"] <= 0.08,
            "heldout_brier_not_worse": after["brier"] <= before["brier"],
        },
    }
    summary["acceptance"]["calibrator_pass"] = bool(all(summary["acceptance"].values()))
    (args.out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
