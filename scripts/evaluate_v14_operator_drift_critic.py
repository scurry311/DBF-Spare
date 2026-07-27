#!/usr/bin/env python3
"""Audit the v1.4 operator-drift residual critic and its evidence scope."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "v14_operator_drift_dataset_20260727_run03"
DEFAULT_CRITIC = ROOT / "hfss_outputs" / "v14_operator_drift_residual_critic_20260727_run01"
DEFAULT_ABLATION = (
    ROOT / "hfss_outputs" / "v14_operator_drift_residual_critic_ablation_no_drift_20260727_run01"
)
DEFAULT_DRIFT = ROOT / "hfss_outputs" / "v14_operator_drift_4x4_smoke_20260727_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v14_operator_drift_critic_evaluation_20260727_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
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


def group_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_names = np.asarray(("train", "val", "test"))
    dimensions = {
        "split": split_names[np.asarray(data["split_id"], dtype=int)],
        "k": np.asarray(data["k_values"], dtype=int).astype(str),
        "ratio": np.char.mod("%.1f", np.asarray(data["active_ratios_requested"], dtype=float)),
        "drift_intensity": np.char.mod("%.2f", np.asarray(data["drift_intensity"], dtype=float)),
        "operator_profile": np.asarray(data["operator_profile"]).astype(str),
    }
    strict = np.asarray(data["strict_gate20"], dtype=bool)
    hard_negative = np.asarray(data["hard_negative"], dtype=bool)
    margins = np.asarray(data["actual_margins"], dtype=float)
    sample_index = np.asarray(data["sample_index"], dtype=np.int64)
    for dimension, values in dimensions.items():
        for value in np.unique(values):
            member = values == value
            scenes = np.unique(sample_index[member])
            oracle = [np.any(strict[member & (sample_index == scene)]) for scene in scenes]
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "candidate_count": int(np.sum(member)),
                    "scene_count": int(scenes.size),
                    "strict_positive_count": int(np.sum(strict[member])),
                    "strict_positive_rate": float(np.mean(strict[member])),
                    "hard_negative_count": int(np.sum(hard_negative[member])),
                    "scene_oracle_rate": float(np.mean(oracle)),
                    "worst_margin_mean_db": float(np.mean(np.min(margins[member], axis=1))),
                }
            )
    return rows


def residual_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    residual = np.asarray(data["margin_residuals"], dtype=float)
    names = np.asarray(data["margin_names"]).astype(str)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        values = residual[:, index]
        rows.append(
            {
                "margin": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q01": float(np.quantile(values, 0.01)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
                "q99": float(np.quantile(values, 0.99)),
                "max_abs": float(np.max(np.abs(values))),
            }
        )
    return rows


def ensemble_predictions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["candidate_index"])].append(row)
    output: list[dict[str, Any]] = []
    for candidate, members in grouped.items():
        output.append(
            {
                "candidate_index": candidate,
                "sample_index": int(members[0]["sample_index"]),
                "k_value": int(members[0]["k_value"]),
                "ratio": float(members[0]["ratio"]),
                "strict_gate20": int(members[0]["strict_gate20"]),
                "active_rl_gate": int(float(members[0]["actual_margin_active_rl_db"]) >= 0.0),
                "prob_strict_gate20": float(
                    np.mean([float(row["prob_strict_gate20"]) for row in members])
                ),
                "prob_active_rl_gate": float(
                    np.mean([float(row["prob_active_rl_gate"]) for row in members])
                ),
                "ranking_score": float(np.mean([float(row["ranking_score"]) for row in members])),
            }
        )
    return output


def calibration_rows(
    predictions: list[dict[str, Any]], probability_key: str, label_key: str, bins: int = 10
) -> list[dict[str, Any]]:
    probability = np.asarray([row[probability_key] for row in predictions], dtype=float)
    label = np.asarray([row[label_key] for row in predictions], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        member = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if not np.any(member):
            continue
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(np.sum(member)),
                "mean_probability": float(np.mean(probability[member])),
                "observed_rate": float(np.mean(label[member])),
            }
        )
    return rows


def plot_calibration(path: Path, curves: dict[str, list[dict[str, Any]]]) -> None:
    width, height = 960, 780
    left, top, right, bottom = 120, 70, 900, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
    for tick in range(6):
        value = tick / 5.0
        x = left + value * (right - left)
        y = bottom - value * (bottom - top)
        draw.line((x, top, x, bottom), fill="#dddddd", width=1)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((x - 12, bottom + 12), f"{value:.1f}", fill="#333333")
        draw.text((left - 45, y - 7), f"{value:.1f}", fill="#333333")
    draw.line((left, bottom, right, top), fill="#777777", width=2)
    colors = {"strict_gate20": "#167d6d", "active_rl_gate": "#c04b32"}
    for name, rows in curves.items():
        points = [
            (
                left + row["mean_probability"] * (right - left),
                bottom - row["observed_rate"] * (bottom - top),
            )
            for row in rows
        ]
        if len(points) > 1:
            draw.line(points, fill=colors[name], width=4)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[name])
    draw.text((360, 22), "Operator-drift critic calibration", fill="#222222")
    draw.text((390, 735), "Mean predicted probability", fill="#222222")
    draw.text((20, 360), "Observed pass rate", fill="#222222")
    legend_x = 640
    for index, name in enumerate(curves):
        y = 92 + index * 28
        draw.line((legend_x, y, legend_x + 38, y), fill=colors[name], width=4)
        draw.text((legend_x + 48, y - 7), name, fill="#222222")
    image.save(path)


def ranking_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k_value in (2, 4, 6):
        selected = [row for row in predictions if row["k_value"] == k_value]
        by_scene: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_scene[row["sample_index"]].append(row)
        top_pass: list[int] = []
        oracle: list[int] = []
        ratios: list[float] = []
        for members in by_scene.values():
            best = max(members, key=lambda row: row["ranking_score"])
            top_pass.append(best["strict_gate20"])
            oracle.append(int(any(row["strict_gate20"] for row in members)))
            ratios.append(best["ratio"])
        rows.append(
            {
                "k_value": k_value,
                "scene_count": len(by_scene),
                "top1_strict_rate": float(np.mean(top_pass)),
                "oracle_strict_rate": float(np.mean(oracle)),
                "top1_mean_ratio": float(np.mean(ratios)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    with np.load(args.dataset_dir / "dataset_arrays.npz", allow_pickle=True) as package:
        data = {key: package[key] for key in package.files}
    training = json.loads((args.critic_dir / "training_summary.json").read_text(encoding="utf-8"))
    ablation = json.loads(
        (args.ablation_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    drift = json.loads((args.drift_dir / "drift_calibration_summary.json").read_text(encoding="utf-8"))
    predictions = ensemble_predictions(read_csv(args.critic_dir / "test_predictions.csv"))
    group_stats = group_rows(data)
    residual_stats = residual_rows(data)
    ranking = ranking_rows(predictions)
    curves = {
        "strict_gate20": calibration_rows(predictions, "prob_strict_gate20", "strict_gate20"),
        "active_rl_gate": calibration_rows(
            predictions, "prob_active_rl_gate", "active_rl_gate"
        ),
    }
    write_csv(args.out_dir / "dataset_group_stats.csv", group_stats)
    write_csv(args.out_dir / "margin_residual_stats.csv", residual_stats)
    write_csv(args.out_dir / "ensemble_test_predictions.csv", predictions)
    write_csv(args.out_dir / "ranking_by_k.csv", ranking)
    write_csv(
        args.out_dir / "calibration_curve.csv",
        [dict(gate=name, **row) for name, values in curves.items() for row in values],
    )
    plot_calibration(args.out_dir / "calibration_curve.png", curves)

    comparison_metrics = (
        "test_strict_gate20_auroc",
        "test_strict_gate20_auprc",
        "test_strict_gate20_brier",
        "test_strict_gate20_ece",
        "test_strict_gate20_precision",
        "test_strict_gate20_recall",
        "test_active_rl_gate_auroc",
        "test_active_rl_gate_auprc",
        "test_active_rl_gate_brier",
        "test_active_rl_gate_ece",
        "test_active_rl_gate_precision",
        "test_active_rl_gate_recall",
        "top1_strict_rate",
    )
    ablation_rows = []
    for metric in comparison_metrics:
        main_value = training["five_seed_aggregate"][metric]["mean"]
        ablation_value = ablation["five_seed_aggregate"][metric]["mean"]
        ablation_rows.append(
            {
                "metric": metric,
                "with_drift_features": main_value,
                "without_drift_features": ablation_value,
                "delta_with_minus_without": main_value - ablation_value,
            }
        )
    write_csv(args.out_dir / "drift_feature_ablation.csv", ablation_rows)

    aggregate = training["five_seed_aggregate"]
    strict_auroc = aggregate["test_strict_gate20_auroc"]["mean"]
    strict_ece = aggregate["test_strict_gate20_ece"]["mean"]
    strict_precision = aggregate["test_strict_gate20_precision"]["mean"]
    top1 = aggregate["top1_strict_rate"]["mean"]
    fixed = aggregate["fixed_strategy_rate"]["mean"]
    target_hashes = np.asarray(data["target_hashes"]).astype(str)
    split_id = np.asarray(data["split_id"], dtype=int)
    hash_sets = [set(target_hashes[split_id == value]) for value in range(3)]
    no_leakage = not (hash_sets[0] & hash_sets[1] or hash_sets[0] & hash_sets[2] or hash_sets[1] & hash_sets[2])
    acceptance = {
        "evidence_scope": "4x4-HFSS-calibrated 16x16 EEP/S256 proxy; not 16x16 HFSS",
        "operator_calibration_gate_pass": bool(drift["operator_drift_calibration_gate_pass"]),
        "candidate_count": int(len(data["candidate_index"])),
        "base_scene_count": int(np.unique(data["base_sample_index"]).size),
        "test_base_scene_count": int(np.unique(data["base_sample_index"][split_id == 2]).size),
        "target_hash_leakage_free": bool(no_leakage),
        "strict_auroc": float(strict_auroc),
        "strict_ece": float(strict_ece),
        "strict_precision": float(strict_precision),
        "top1_strict_rate": float(top1),
        "fixed_ratio_0_7_rate": float(fixed),
        "proxy_discrimination_gate_pass": bool(strict_auroc >= 0.88),
        "proxy_calibration_gate_pass": bool(strict_ece <= 0.08),
        "proxy_acceptance_precision_gate_pass": bool(strict_precision >= 0.90),
        "ranking_improves_fixed_strategy": bool(top1 > fixed),
        "drift_feature_strict_auroc_delta": float(
            strict_auroc
            - ablation["five_seed_aggregate"]["test_strict_gate20_auroc"]["mean"]
        ),
        "drift_feature_strict_auprc_delta": float(
            aggregate["test_strict_gate20_auprc"]["mean"]
            - ablation["five_seed_aggregate"]["test_strict_gate20_auprc"]["mean"]
        ),
        "drift_feature_strict_recall_delta": float(
            aggregate["test_strict_gate20_recall"]["mean"]
            - ablation["five_seed_aggregate"]["test_strict_gate20_recall"]["mean"]
        ),
        "drift_feature_top1_delta": float(
            top1 - ablation["five_seed_aggregate"]["top1_strict_rate"]["mean"]
        ),
        "top1_target_0_80_pass": bool(top1 >= 0.80),
        "has_independent_16x16_hfss_operator_drift_validation": False,
        "engineering_critic_promoted": False,
        "allowed_use": "candidate screening, proxy ablation, and pretraining only",
        "next_gate": "freeze this critic and run independent 16x16 frequency/geometry/hardware-drift HFSS smoke",
    }
    (args.out_dir / "acceptance_summary.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8"
    )
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
