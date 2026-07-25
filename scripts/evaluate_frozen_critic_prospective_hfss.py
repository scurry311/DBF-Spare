#!/usr/bin/env python3
"""Evaluate pre-registered frozen critic predictions against prospective HFSS labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from generate_expanded_independent_residual_scenes import target_hash
from train_fullwave_residual_critic_v2 import average_precision, binary_auc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "prospective_gate15_scenes_20260725_run01"
DEFAULT_FREEZE = ROOT / "hfss_outputs" / "prospective_frozen_critic_20260725_run01"
DEFAULT_HFSS = ROOT / "hfss_outputs" / "prospective_gate15_hfss_20260725_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "prospective_frozen_critic_evaluation_20260725_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--hfss-dir", type=Path, default=DEFAULT_HFSS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--auroc-min", type=float, default=0.88)
    parser.add_argument("--ece-max", type=float, default=0.08)
    parser.add_argument("--minimum-scenes", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_rows(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (probabilities >= low) & (
            probabilities <= high if index == bins - 1 else probabilities < high
        )
        rows.append(
            {
                "bin": index,
                "low": low,
                "high": high,
                "count": int(mask.sum()),
                "mean_probability": (
                    float(np.mean(probabilities[mask])) if np.any(mask) else float("nan")
                ),
                "positive_rate": (
                    float(np.mean(labels[mask])) if np.any(mask) else float("nan")
                ),
            }
        )
    return rows


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    predicted = probabilities >= 0.5
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum((~predicted) & (~positive)))
    fp = int(np.sum(predicted & (~positive)))
    fn = int(np.sum((~predicted) & positive))
    bins = calibration_rows(labels, probabilities)
    ece = sum(
        (row["count"] / max(labels.size, 1))
        * abs(row["mean_probability"] - row["positive_rate"])
        for row in bins
        if row["count"]
    )
    return {
        "count": int(labels.size),
        "positive_count": int(labels.sum()),
        "positive_rate": float(np.mean(labels)),
        "auroc": binary_auc(labels, probabilities),
        "auprc": average_precision(labels, probabilities),
        "accuracy": float(np.mean(predicted == positive)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": float(ece),
        "nll": float(
            -np.mean(
                labels * np.log(probabilities)
                + (1 - labels) * np.log(1.0 - probabilities)
            )
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def decision_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    decisions = np.asarray(decisions, dtype=np.int64)
    positive = labels == 1
    admitted = decisions == 1
    tp = int(np.sum(admitted & positive))
    tn = int(np.sum((~admitted) & (~positive)))
    fp = int(np.sum(admitted & (~positive)))
    fn = int(np.sum((~admitted) & positive))
    return {
        "admit_count": int(admitted.sum()),
        "actual_positive_count": int(positive.sum()),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float(np.mean(admitted == positive)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def grouped_bootstrap_intervals(
    rows: list[dict[str, Any]],
    gate_names: tuple[str, ...] = ("gate15", "gate20"),
    iterations: int = 2000,
    seed: int = 20260725,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["sample_index"])].append(index)
    scene_indices = list(grouped.values())
    rng = np.random.default_rng(seed)
    metric_names = ("auroc", "auprc", "brier", "ece", "precision", "recall")
    samples: dict[str, dict[str, list[float]]] = {
        gate: {metric: [] for metric in metric_names} for gate in gate_names
    }
    for _ in range(iterations):
        selected_scenes = rng.integers(0, len(scene_indices), size=len(scene_indices))
        selected = np.asarray(
            [index for scene in selected_scenes for index in scene_indices[int(scene)]],
            dtype=np.int64,
        )
        for gate in gate_names:
            labels = np.asarray([row[f"actual_{gate}"] for row in rows], dtype=np.int64)[selected]
            probabilities = np.asarray(
                [row[f"prob_{gate}"] for row in rows], dtype=float
            )[selected]
            metrics = probability_metrics(labels, probabilities)
            for metric in metric_names:
                value = float(metrics[metric])
                if math.isfinite(value):
                    samples[gate][metric].append(value)
    intervals: dict[str, dict[str, dict[str, float]]] = {}
    csv_rows: list[dict[str, Any]] = []
    for gate in gate_names:
        intervals[gate] = {}
        for metric in metric_names:
            values = np.asarray(samples[gate][metric], dtype=float)
            interval = {
                "lower_95": (
                    float(np.quantile(values, 0.025)) if values.size else float("nan")
                ),
                "upper_95": (
                    float(np.quantile(values, 0.975)) if values.size else float("nan")
                ),
                "bootstrap_valid_count": int(values.size),
            }
            intervals[gate][metric] = interval
            csv_rows.append({"gate": gate, "metric": metric, **interval})
    return intervals, csv_rows


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prospective evaluation: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.freeze_dir / "prospective_freeze_manifest.json"
    predictions_path = args.freeze_dir / "frozen_predictions_before_hfss.csv"
    selections_path = args.freeze_dir / "frozen_scene_selections_before_hfss.csv"
    labels_path = args.hfss_dir / "candidate_residual_labels.csv"
    analysis_path = args.hfss_dir / "analysis_summary.json"
    audit_path = args.hfss_dir / "gate15_boundary_hfss_summary.json"
    manifest = read_json(manifest_path)
    analysis = read_json(analysis_path)
    audit = read_json(audit_path)

    hash_checks = {
        "checkpoint": sha256(Path(manifest["checkpoint_path"]))
        == manifest["checkpoint_sha256"],
        "calibrator": sha256(Path(manifest["calibrator_path"]))
        == manifest["calibrator_sha256"],
        "training_dataset": sha256(Path(manifest["training_dataset_path"]))
        == manifest["training_dataset_sha256"],
        "prospective_dataset": sha256(args.dataset_dir / "dataset_arrays.npz")
        == manifest["prospective_dataset_sha256"],
        "frozen_predictions": sha256(predictions_path) == manifest["prediction_sha256"],
        "frozen_selections": sha256(selections_path) == manifest["selection_sha256"],
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"Frozen artifact hash mismatch: {hash_checks}")
    if manifest.get("hfss_results_read") is not False:
        raise RuntimeError("Freeze manifest does not prove pre-HFSS inference")
    if manifest.get("retraining_allowed_after_hfss") is not False:
        raise RuntimeError("Freeze manifest permits post-HFSS retraining")
    if int(manifest.get("training_target_overlap_count", -1)) != 0:
        raise RuntimeError("Prospective directions overlap training targets")

    with np.load(args.dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        prospective_dataset = {key: source[key] for key in source.files}
    with np.load(Path(manifest["training_dataset_path"]), allow_pickle=False) as source:
        training_dataset = {key: source[key] for key in source.files}
    prospective_target_hashes = {
        target_hash(
            prospective_dataset["targets_deg"][index][
                np.asarray(prospective_dataset["task_valid"][index], dtype=bool)
            ]
        )
        for index in range(int(prospective_dataset["candidate_indices"].size))
    }
    training_target_hashes = {
        target_hash(
            training_dataset["targets_deg"][index][
                np.asarray(training_dataset["task_valid"][index], dtype=bool)
            ]
        )
        for index in range(int(training_dataset["sample_index"].size))
    }
    recomputed_target_overlap = sorted(prospective_target_hashes & training_target_hashes)
    if recomputed_target_overlap:
        raise RuntimeError(f"Recomputed target leakage: {recomputed_target_overlap[:5]}")

    predictions = read_csv(predictions_path)
    selections = read_csv(selections_path)
    labels = read_csv(labels_path)
    labels_by_candidate = {int(row["candidate_index"]): row for row in labels}
    if len(labels_by_candidate) != len(labels):
        raise ValueError("Duplicate candidate indices in HFSS labels")
    if len(predictions) != int(manifest["candidate_count"]):
        raise ValueError("Frozen prediction count does not match freeze manifest")
    if len(labels) != len(predictions):
        raise ValueError("Prospective HFSS result count does not match frozen predictions")

    candidate_rows: list[dict[str, Any]] = []
    threshold_label_mismatch_count = 0
    thresholds = manifest["thresholds_frozen"]
    gate_specs = {
        "gate15": ("prob_gate15", "gate15"),
        "gate20": ("prob_gate20", "strict_gate20"),
        "mainlobe_gate": ("prob_mainlobe_gate", "mainlobe_gate"),
        "strict_engineering_gate": (
            "prob_strict_engineering_gate",
            "strict_engineering_gate",
        ),
    }
    for prediction in predictions:
        candidate = int(prediction["candidate_index"])
        label = labels_by_candidate.get(candidate)
        if label is None:
            raise ValueError(f"Missing HFSS label for candidate {candidate}")
        if int(prediction["sample_index"]) != int(label["sample_index"]):
            raise ValueError(f"Scene mismatch for candidate {candidate}")
        row: dict[str, Any] = {
            "candidate_index": candidate,
            "sample_index": int(prediction["sample_index"]),
            "scene_id": prediction["scene_id"],
            "target_hash": prediction["target_hash"],
            "boundary_type": prediction["boundary_type"],
            "boundary_side": prediction["boundary_side_audit_only"],
            "variant_kind": prediction["variant_kind"],
            "k": int(prediction["k"]),
            "active_ratio": float(prediction["active_ratio"]),
            "pattern15_admit": int(prediction["pattern15_admit"]),
            "strict_admit": int(prediction["strict_admit"]),
            "fullwave_complete": int(label["fullwave_complete"]),
        }
        for name, (probability_key, label_key) in gate_specs.items():
            row[f"prob_{name}"] = float(prediction[probability_key])
            row[f"actual_{name}"] = int(label[label_key])
        for metric, prediction_key, actual_key in (
            ("psll_db", "pred_psll_db", "hfss_psll_db"),
            ("nearest_iso_db", "pred_nearest_iso_db", "hfss_nearest_iso_db"),
            ("local_iso_db", "pred_local_iso_db", "hfss_local_iso_db"),
            ("mainlobe_gain_db", "pred_mainlobe_gain_db", "hfss_mainlobe_gain_db"),
            ("target_spread_db", "pred_target_spread_db", "hfss_target_spread_db"),
        ):
            row[f"pred_{metric}"] = float(prediction[prediction_key])
            row[f"actual_{metric}"] = float(label[actual_key])
            row[f"error_{metric}"] = row[f"pred_{metric}"] - row[f"actual_{metric}"]
        row["actual_active_rl_gate"] = int(label["active_RL_gate"])
        row["actual_worst_active_rl_db"] = float(label["all_case_worst_active_rl_db"])
        row["actual_worst_total_rl_db"] = float(label["all_case_worst_total_rl_db"])
        derived_gate15 = bool(
            row["fullwave_complete"]
            and row["actual_psll_db"] <= float(thresholds["psll_db_max"])
            and row["actual_nearest_iso_db"] >= float(thresholds["nearest_iso_db_min"])
            and row["actual_local_iso_db"] >= float(thresholds["local_iso_gate15_db_min"])
        )
        derived_gate20 = bool(
            row["fullwave_complete"]
            and row["actual_psll_db"] <= float(thresholds["psll_db_max"])
            and row["actual_nearest_iso_db"] >= float(thresholds["nearest_iso_db_min"])
            and row["actual_local_iso_db"] >= float(thresholds["local_iso_gate20_db_min"])
        )
        derived_mainlobe = bool(
            row["actual_mainlobe_gain_db"]
            >= float(label["eep_mainlobe_gain_db"])
            - float(thresholds["mainlobe_drop_db_max"])
            and row["actual_target_spread_db"] <= float(thresholds["target_spread_db_max"])
            and float(label["hfss_pointing_error_deg"])
            <= float(thresholds["pointing_error_deg_max"])
        )
        derived_active = bool(
            row["actual_worst_active_rl_db"] >= float(thresholds["active_rl_db_min"])
            and row["actual_worst_total_rl_db"] >= float(thresholds["active_rl_db_min"])
        )
        derived_strict = bool(derived_gate20 and derived_mainlobe and derived_active)
        row["frozen_threshold_label_match"] = int(
            derived_gate15 == bool(row["actual_gate15"])
            and derived_gate20 == bool(row["actual_gate20"])
            and derived_mainlobe == bool(row["actual_mainlobe_gate"])
            and derived_active == bool(row["actual_active_rl_gate"])
            and derived_strict == bool(row["actual_strict_engineering_gate"])
        )
        threshold_label_mismatch_count += 1 - row["frozen_threshold_label_match"]
        candidate_rows.append(row)
    if threshold_label_mismatch_count:
        raise RuntimeError(
            f"HFSS labels disagree with frozen thresholds for {threshold_label_mismatch_count} candidates"
        )
    write_csv(args.out_dir / "prospective_candidate_evaluation.csv", candidate_rows)

    gate_metrics: dict[str, dict[str, float]] = {}
    calibration_output: list[dict[str, Any]] = []
    for name in gate_specs:
        actual = np.asarray([row[f"actual_{name}"] for row in candidate_rows])
        probability = np.asarray([row[f"prob_{name}"] for row in candidate_rows])
        gate_metrics[name] = probability_metrics(actual, probability)
        for row in calibration_rows(actual, probability):
            calibration_output.append({"gate": name, **row})
    write_csv(args.out_dir / "prospective_calibration_bins.csv", calibration_output)
    bootstrap_intervals, bootstrap_rows = grouped_bootstrap_intervals(candidate_rows)
    write_csv(args.out_dir / "prospective_scene_bootstrap_95ci.csv", bootstrap_rows)

    group_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(row["boundary_type"], row["boundary_side"])].append(row)
        grouped[(row["boundary_type"], "all")].append(row)
        grouped[("all", row["boundary_side"])].append(row)
        grouped[("all", "all")].append(row)
    for (boundary_type, boundary_side), rows in sorted(grouped.items()):
        labels15 = np.asarray([row["actual_gate15"] for row in rows])
        probabilities15 = np.asarray([row["prob_gate15"] for row in rows])
        labels20 = np.asarray([row["actual_gate20"] for row in rows])
        probabilities20 = np.asarray([row["prob_gate20"] for row in rows])
        metrics15 = probability_metrics(labels15, probabilities15)
        metrics20 = probability_metrics(labels20, probabilities20)
        group_rows.append(
            {
                "boundary_type": boundary_type,
                "boundary_side": boundary_side,
                "candidate_count": len(rows),
                "scene_count": len({row["sample_index"] for row in rows}),
                **{f"gate15_{key}": value for key, value in metrics15.items()},
                **{f"gate20_{key}": value for key, value in metrics20.items()},
            }
        )
    write_csv(args.out_dir / "prospective_group_metrics.csv", group_rows)

    boundary_pair_metrics: dict[str, dict[str, float]] = {}
    boundary_pair_rows: list[dict[str, Any]] = []
    for boundary_type in ("all", "psll", "nearest", "local"):
        rows = [
            row
            for row in candidate_rows
            if row["boundary_side"] != "control"
            and (boundary_type == "all" or row["boundary_type"] == boundary_type)
        ]
        labels15 = np.asarray([row["actual_gate15"] for row in rows])
        probabilities15 = np.asarray([row["prob_gate15"] for row in rows])
        metrics15 = probability_metrics(labels15, probabilities15)
        pairs: dict[int, dict[str, float]] = defaultdict(dict)
        for row in rows:
            pairs[int(row["sample_index"])][row["boundary_side"]] = row["prob_gate15"]
        margins = np.asarray(
            [pair["inside"] - pair["outside"] for pair in pairs.values()], dtype=float
        )
        pair_order_rate = float(
            np.mean(margins > 0.0) + 0.5 * np.mean(margins == 0.0)
        )
        summary_row = {
            "boundary_type": boundary_type,
            "scene_count": len(pairs),
            "candidate_count_without_control": len(rows),
            "gate15_auroc_without_control": metrics15["auroc"],
            "gate15_auprc_without_control": metrics15["auprc"],
            "inside_above_outside_pair_rate": pair_order_rate,
            "inside_minus_outside_probability_mean": float(np.mean(margins)),
            "inside_minus_outside_probability_min": float(np.min(margins)),
            "inside_minus_outside_probability_max": float(np.max(margins)),
        }
        boundary_pair_rows.append(summary_row)
        boundary_pair_metrics[boundary_type] = {
            key: value for key, value in summary_row.items() if key != "boundary_type"
        }
    write_csv(
        args.out_dir / "prospective_boundary_pair_metrics.csv", boundary_pair_rows
    )

    by_scene: dict[int, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_index = {row["candidate_index"]: row for row in candidate_rows}
    for row in candidate_rows:
        by_scene[row["sample_index"]].append(row)
    selection_rows: list[dict[str, Any]] = []
    selection_summary: dict[str, dict[str, float]] = {}
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for selection in selections:
        candidate = int(selection["candidate_index"])
        row = candidate_by_index[candidate]
        if int(selection["sample_index"]) != row["sample_index"]:
            raise ValueError(f"Frozen selection scene mismatch for candidate {candidate}")
        selected = {
            "sample_index": row["sample_index"],
            "method": selection["method"],
            "candidate_index": candidate,
            "boundary_type": row["boundary_type"],
            "boundary_side": row["boundary_side"],
            "actual_gate15": row["actual_gate15"],
            "actual_gate20": row["actual_gate20"],
            "actual_mainlobe_gate": row["actual_mainlobe_gate"],
            "actual_strict_engineering_gate": row["actual_strict_engineering_gate"],
        }
        selection_rows.append(selected)
        by_method[selected["method"]].append(selected)
    write_csv(args.out_dir / "prospective_scene_selections.csv", selection_rows)
    for method, rows in sorted(by_method.items()):
        selection_summary[method] = {
            "scene_count": len(rows),
            "gate15_top1_rate": float(np.mean([row["actual_gate15"] for row in rows])),
            "gate20_top1_rate": float(np.mean([row["actual_gate20"] for row in rows])),
            "strict_engineering_top1_rate": float(
                np.mean([row["actual_strict_engineering_gate"] for row in rows])
            ),
        }
    oracle = {
        "scene_count": len(by_scene),
        "gate15_oracle_rate": float(
            np.mean([any(row["actual_gate15"] for row in rows) for rows in by_scene.values()])
        ),
        "gate20_oracle_rate": float(
            np.mean([any(row["actual_gate20"] for row in rows) for rows in by_scene.values()])
        ),
        "strict_engineering_oracle_rate": float(
            np.mean(
                [
                    any(row["actual_strict_engineering_gate"] for row in rows)
                    for rows in by_scene.values()
                ]
            )
        ),
    }
    selection_metric_rows = [
        {"method": method, **metrics}
        for method, metrics in sorted(selection_summary.items())
    ]
    selection_metric_rows.append({"method": "best_of_three_oracle", **oracle})
    write_csv(
        args.out_dir / "prospective_scene_selection_metrics.csv",
        selection_metric_rows,
    )

    actual_gate15 = np.asarray([row["actual_gate15"] for row in candidate_rows])
    actual_strict = np.asarray(
        [row["actual_strict_engineering_gate"] for row in candidate_rows]
    )
    admissions = {
        "pattern15": decision_metrics(
            actual_gate15,
            np.asarray([row["pattern15_admit"] for row in candidate_rows]),
        ),
        "strict": decision_metrics(
            actual_strict,
            np.asarray([row["strict_admit"] for row in candidate_rows]),
        ),
    }

    failures = [
        row
        for row in candidate_rows
        if (row["pattern15_admit"] != row["actual_gate15"])
        or (row["strict_admit"] != row["actual_strict_engineering_gate"])
    ]
    write_csv(args.out_dir / "prospective_failures.csv", failures)

    residual_metrics: dict[str, dict[str, float]] = {}
    for metric in (
        "psll_db",
        "nearest_iso_db",
        "local_iso_db",
        "mainlobe_gain_db",
        "target_spread_db",
    ):
        errors = np.asarray([row[f"error_{metric}"] for row in candidate_rows])
        residual_metrics[metric] = {
            "bias": float(np.mean(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "max_abs_error": float(np.max(np.abs(errors))),
        }

    acceptance = {
        "all_hashes_match": bool(all(hash_checks.values())),
        "pre_hfss_freeze_proven": bool(manifest["hfss_results_read"] is False),
        "retraining_disabled": bool(manifest["retraining_allowed_after_hfss"] is False),
        "target_direction_overlap_zero": int(manifest["training_target_overlap_count"]) == 0,
        "recomputed_target_direction_overlap_zero": len(recomputed_target_overlap) == 0,
        "frozen_threshold_labels_consistent": threshold_label_mismatch_count == 0,
        "scene_support_pass": len(by_scene) >= int(args.minimum_scenes),
        "all_hfss_cases_complete": int(analysis["complete_case_count"])
        == int(analysis["expected_case_count"]),
        "all_no_scale_reconstruction_pass": bool(
            analysis["all_no_scale_reconstruction_pass"]
        ),
        "boundary_audit_pass": bool(audit["boundary_dataset_pass"]),
        "gate15_auroc_pass": gate_metrics["gate15"]["auroc"] >= float(args.auroc_min),
        "gate20_auroc_pass": gate_metrics["gate20"]["auroc"] >= float(args.auroc_min),
        "gate15_ece_pass": gate_metrics["gate15"]["ece"] <= float(args.ece_max),
        "gate20_ece_pass": gate_metrics["gate20"]["ece"] <= float(args.ece_max),
    }
    primary_keys = (
        "all_hashes_match",
        "pre_hfss_freeze_proven",
        "retraining_disabled",
        "target_direction_overlap_zero",
        "recomputed_target_direction_overlap_zero",
        "frozen_threshold_labels_consistent",
        "scene_support_pass",
        "all_hfss_cases_complete",
        "all_no_scale_reconstruction_pass",
        "boundary_audit_pass",
        "gate15_auroc_pass",
        "gate20_auroc_pass",
        "gate15_ece_pass",
        "gate20_ece_pass",
    )
    prospective_pass = bool(all(acceptance[key] for key in primary_keys))
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prospective": True,
        "post_hfss_retraining_performed": False,
        "threshold_or_calibrator_adjustment_performed": False,
        "mainlobe_failure_expansion_performed": False,
        "candidate_count": len(candidate_rows),
        "independent_scene_count": len(by_scene),
        "unique_target_hash_count": len({row["target_hash"] for row in candidate_rows}),
        "recomputed_training_target_overlap_count": len(recomputed_target_overlap),
        "freeze_manifest_sha256": sha256(manifest_path),
        "hash_checks": hash_checks,
        "frozen_thresholds": manifest["thresholds_frozen"],
        "hfss": {
            "expected_case_count": int(analysis["expected_case_count"]),
            "complete_case_count": int(analysis["complete_case_count"]),
            "complex_nmse_max": float(analysis["complex_nmse_max"]),
            "magnitude_rmse_db_max": float(analysis["magnitude_rmse_db_max"]),
            "actual_mainlobe_failure_count": int(
                sum(1 - row["actual_mainlobe_gate"] for row in candidate_rows)
            ),
            "labels_sha256": sha256(labels_path),
            "analysis_sha256": sha256(analysis_path),
        },
        "gate_metrics": gate_metrics,
        "scene_grouped_bootstrap_95ci": bootstrap_intervals,
        "admission_metrics": admissions,
        "boundary_pair_metrics": boundary_pair_metrics,
        "scene_selection_metrics": selection_summary,
        "scene_oracle_metrics": oracle,
        "residual_metrics": residual_metrics,
        "acceptance_thresholds": {
            "auroc_min": float(args.auroc_min),
            "ece_max": float(args.ece_max),
            "minimum_scenes": int(args.minimum_scenes),
            "source": "v0.7 retrospective stage-one promotion protocol",
        },
        "acceptance": acceptance,
        "prospective_validation_pass": prospective_pass,
        "automatic_hfss_admission_allowed": prospective_pass,
        "failure_reasons": [
            key for key in primary_keys if not acceptance[key]
        ],
        "next_action": (
            "Keep checkpoint, pooled calibrator, and thresholds frozen; enable guarded candidate "
            "admission only within the validated array and target-distribution scope."
            if prospective_pass
            else "Do not tune on this prospective set. Keep automatic admission disabled and "
            "report the failed pre-registered gate as out-of-sample evidence."
        ),
    }
    summary = finite_json(summary)
    (args.out_dir / "prospective_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
