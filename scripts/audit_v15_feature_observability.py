#!/usr/bin/env python3
"""Prepare and analyze the pre-registered v1.5 feature observability audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score

from train_v09_physical_margin_critic import (
    feature_observability_class,
    scalar_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v15_feature_observability_preregistered.json"
DEFAULT_MODEL_A = ROOT / "hfss_outputs" / "v15_observability_model_a_deployable_20260727_run01"
DEFAULT_MODEL_B = ROOT / "hfss_outputs" / "v15_observability_model_b_measurement_20260727_run01"
DEFAULT_MODEL_C = ROOT / "hfss_outputs" / "v14_operator_drift_residual_critic_20260727_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v15_feature_observability_audit_20260727_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "analyze", "status", "all"), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--model-a-dir", type=Path, default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b-dir", type=Path, default=DEFAULT_MODEL_B)
    parser.add_argument("--model-c-dir", type=Path, default=DEFAULT_MODEL_C)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_data(protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    path = ROOT / protocol["dataset_dir"] / "dataset_arrays.npz"
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def feature_metadata(name: str, tier: str) -> dict[str, str | int]:
    base = {
        "physical_meaning": "Derived nominal command or EEP/S256 scene descriptor",
        "training_source": "Frozen v1.4 proxy dataset",
        "inference_source": "Commanded scene and nominal EEP/S256 evaluation",
        "online_measurable": 1,
        "requires_vna_or_calibrator": 0,
        "depends_on_hfss_perturbed_operator": 0,
        "label_leakage_risk": "low",
        "recommendation": "retain",
    }
    if tier == "B":
        base.update(
            {
                "physical_meaning": "Measured or calibrated implementation state",
                "inference_source": "VNA, built-in calibration, telemetry, or fault monitor",
                "online_measurable": 0,
                "requires_vna_or_calibrator": 1,
                "label_leakage_risk": "medium if measurement is unavailable at inference",
                "recommendation": "retain only for measurement-assisted deployment",
            }
        )
    if tier == "C":
        base.update(
            {
                "physical_meaning": "Latent manufacturing or simulation-truth perturbation",
                "inference_source": "Unavailable without metrology or perturbed full-wave truth",
                "online_measurable": 0,
                "requires_vna_or_calibrator": 0,
                "depends_on_hfss_perturbed_operator": 1,
                "label_leakage_risk": "high",
                "recommendation": "oracle ablation only; prohibit deployment",
            }
        )
    overrides: dict[str, dict[str, str | int]] = {
        "frequency_offset_ghz_scaled": {
            "physical_meaning": "Known operating-frequency offset from 10 GHz",
            "inference_source": "RF synthesizer setpoint",
        },
        "s_drift_relative_fro_scaled": {
            "physical_meaning": "Relative complex S-matrix drift",
            "inference_source": "VNA or embedded multiport reflection calibration",
            "depends_on_hfss_perturbed_operator": 0,
        },
        "s_drift_max_abs_scaled": {
            "physical_meaning": "Maximum absolute complex S-parameter drift",
            "inference_source": "VNA or embedded multiport reflection calibration",
            "depends_on_hfss_perturbed_operator": 0,
        },
        "s_projection_scale_scaled": {
            "physical_meaning": "Passivity projection severity of measured S matrix",
            "inference_source": "Derived from measured complex S matrix",
            "depends_on_hfss_perturbed_operator": 0,
        },
        "temperature_offset_c_scaled": {
            "physical_meaning": "Array or RF-chain temperature offset",
            "inference_source": "Temperature telemetry",
            "requires_vna_or_calibrator": 0,
            "online_measurable": 1,
        },
        "patch_length_offset_mm_scaled": {
            "physical_meaning": "True manufactured patch-length offset",
            "training_source": "HFSS geometry profile truth",
        },
        "relative_permittivity_offset_scaled": {
            "physical_meaning": "True substrate permittivity offset",
            "training_source": "HFSS material profile truth",
        },
        "drift_intensity_scaled": {
            "physical_meaning": "Synthetic interpolation strength of the perturbed operator",
            "training_source": "Proxy generator control variable",
        },
    }
    base.update(overrides.get(name, {}))
    return base


def feature_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    _values, names = scalar_features(data, include_drift_features=True, feature_tier="oracle")
    rows: list[dict[str, Any]] = []
    fixed_inputs = (
        ("mask_16x16", "A", "spatial", "Commanded sparse element-selection mask"),
        ("task_weights_real_imag", "A", "spatial", "Commanded task-level complex weights"),
        ("target_set", "A", "set", "Commanded unordered target directions and validity mask"),
    )
    for name, tier, group, meaning in fixed_inputs:
        rows.append(
            {
                "feature_name": name,
                "is_current_input": 1,
                "model_input_group": group,
                "observability_class": tier,
                "physical_meaning": meaning,
                "training_source": "Frozen v1.4 proxy dataset",
                "inference_source": "Beam-command controller",
                "online_measurable": 1,
                "requires_vna_or_calibrator": 0,
                "depends_on_hfss_perturbed_operator": 0,
                "label_leakage_risk": "low",
                "recommendation": "retain",
                "model_a_included": 1,
                "model_b_included": 1,
                "model_c_included": 1,
            }
        )
    for name in names:
        tier = feature_observability_class(name)
        metadata = feature_metadata(name, tier)
        rows.append(
            {
                "feature_name": name,
                "is_current_input": 1,
                "model_input_group": "scalar",
                "observability_class": tier,
                **metadata,
                "model_a_included": int(tier == "A"),
                "model_b_included": int(tier in ("A", "B")),
                "model_c_included": 1,
            }
        )
    forbidden = (
        "actual_hfss_psll_isolation_mainlobe",
        "actual_margins",
        "strict_gate_label",
        "test_derived_drift_feature",
        "future_fullwave_label",
    )
    for name in forbidden:
        rows.append(
            {
                "feature_name": name,
                "is_current_input": 0,
                "model_input_group": "forbidden",
                "observability_class": "C",
                "physical_meaning": "Ground-truth or future evaluation label",
                "training_source": "Evaluation label only",
                "inference_source": "Unavailable before prediction",
                "online_measurable": 0,
                "requires_vna_or_calibrator": 0,
                "depends_on_hfss_perturbed_operator": 1,
                "label_leakage_risk": "prohibited",
                "recommendation": "never use as critic input",
                "model_a_included": 0,
                "model_b_included": 0,
                "model_c_included": 0,
            }
        )
    return rows


def split_audit(data: dict[str, np.ndarray]) -> dict[str, Any]:
    split = np.asarray(data["split_id"], dtype=int)
    hashes = np.asarray(data["target_hashes"]).astype(str)
    hash_sets = [set(hashes[split == value]) for value in range(3)]
    intersections = {
        "train_val": len(hash_sets[0] & hash_sets[1]),
        "train_test": len(hash_sets[0] & hash_sets[2]),
        "val_test": len(hash_sets[1] & hash_sets[2]),
    }
    return {
        "candidate_count": int(split.size),
        "base_scene_count": int(np.unique(data["base_sample_index"]).size),
        "split_candidate_counts": {
            name: int(np.sum(split == value))
            for value, name in enumerate(("train", "val", "test"))
        },
        "target_hash_intersections": intersections,
        "target_hash_leakage_free": not any(intersections.values()),
    }


def prepare(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite observability audit: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    data = load_data(protocol)
    rows = feature_rows(data)
    write_csv(args.out_dir / "feature_observability_matrix.csv", rows)
    audit = split_audit(data)
    tier_counts = {
        tier: sum(
            1
            for row in rows
            if row["is_current_input"] and row["observability_class"] == tier
        )
        for tier in ("A", "B", "C")
    }
    summary = {
        "protocol": protocol,
        "split_audit": audit,
        "current_input_feature_counts_by_class": tier_counts,
        "model_a_scalar_tier": "A",
        "model_b_scalar_tier": "A+B",
        "model_c_scalar_tier": "A+B+C",
        "label_scope": protocol["evidence_scope"],
        "training_allowed": bool(audit["target_hash_leakage_free"]),
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def binary_metrics(probability: np.ndarray, label: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    label = np.asarray(label, dtype=int)
    predicted = probability >= 0.5
    true_positive = int(np.sum(predicted & (label == 1)))
    false_positive = int(np.sum(predicted & (label == 0)))
    false_negative = int(np.sum(~predicted & (label == 1)))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for index in range(10):
        member = (probability >= bins[index]) & (
            probability <= bins[index + 1] if index == 9 else probability < bins[index + 1]
        )
        if np.any(member):
            ece += float(np.mean(member)) * abs(
                float(np.mean(probability[member])) - float(np.mean(label[member]))
            )
    both = np.unique(label).size == 2
    return {
        "auroc": float(roc_auc_score(label, probability)) if both else float("nan"),
        "auprc": float(average_precision_score(label, probability)) if np.any(label) else float("nan"),
        "brier": float(np.mean((probability - label) ** 2)),
        "ece": ece,
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "positive_rate": float(np.mean(label)),
    }


def summarize_models(
    model_dirs: dict[str, Path], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for model, path in model_dirs.items():
        summary = json.loads((path / "training_summary.json").read_text(encoding="utf-8"))
        summaries[model] = summary
        aggregate = summary["five_seed_aggregate"]
        row = {
            "model": model,
            "feature_tier": {"model_a": "A", "model_b": "A+B", "model_c": "A+B+C"}[model],
            "strict_auroc": aggregate["test_strict_gate20_auroc"]["mean"],
            "strict_auprc": aggregate["test_strict_gate20_auprc"]["mean"],
            "strict_brier": aggregate["test_strict_gate20_brier"]["mean"],
            "strict_ece": aggregate["test_strict_gate20_ece"]["mean"],
            "strict_precision": aggregate["test_strict_gate20_precision"]["mean"],
            "strict_recall": aggregate["test_strict_gate20_recall"]["mean"],
            "top1_strict_rate": aggregate["top1_strict_rate"]["mean"],
            "oracle_strict_rate": aggregate["oracle_strict_rate"]["mean"],
            "fixed_strategy_rate": aggregate["fixed_strategy_rate"]["mean"],
            "seed_count": len(protocol["seeds"]),
        }
        gates = protocol["stage_a_gates"]
        row["auroc_gate_pass"] = int(row["strict_auroc"] >= gates["strict_auroc_min"])
        row["ece_gate_pass"] = int(row["strict_ece"] <= gates["strict_ece_max"])
        row["precision_gate_pass"] = int(
            row["strict_precision"] >= gates["strict_precision_min"]
        )
        row["stage_a_model_gate_pass"] = int(
            row["auroc_gate_pass"] and row["ece_gate_pass"] and row["precision_gate_pass"]
        )
        rows.append(row)
    return rows, summaries


def prediction_groups(
    model: str, rows: list[dict[str, str]], data: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_index = np.asarray([int(row["candidate_index"]) for row in rows], dtype=int)
    dimensions = {
        "k": np.asarray(data["k_values"])[candidate_index].astype(str),
        "ratio": np.char.mod("%.1f", np.asarray(data["active_ratios_requested"])[candidate_index]),
        "drift_intensity": np.char.mod("%.2f", np.asarray(data["drift_intensity"])[candidate_index]),
    }
    seeds = np.asarray([int(row["seed"]) for row in rows], dtype=int)
    probability = np.asarray([float(row["prob_strict_gate20"]) for row in rows], dtype=float)
    label = np.asarray([int(row["strict_gate20"]) for row in rows], dtype=int)
    score = np.asarray([float(row["ranking_score"]) for row in rows], dtype=float)
    scene = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    metrics_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for dimension, values in dimensions.items():
        for value in np.unique(values):
            seed_metrics = []
            seed_top = []
            seed_oracle = []
            for seed in np.unique(seeds):
                member = (values == value) & (seeds == seed)
                seed_metrics.append(binary_metrics(probability[member], label[member]))
                selected = []
                oracle = []
                for scene_id in np.unique(scene[member]):
                    group = np.flatnonzero(member & (scene == scene_id))
                    selected.append(int(group[np.argmax(score[group])]))
                    oracle.append(int(np.any(label[group])))
                seed_top.append(float(np.mean(label[selected])) if selected else float("nan"))
                seed_oracle.append(float(np.mean(oracle)) if oracle else float("nan"))
            row: dict[str, Any] = {"model": model, "dimension": dimension, "value": value}
            for key in seed_metrics[0]:
                values_for_key = np.asarray([item[key] for item in seed_metrics], dtype=float)
                row[key] = (
                    float(np.mean(values_for_key[np.isfinite(values_for_key)]))
                    if np.any(np.isfinite(values_for_key))
                    else float("nan")
                )
            metrics_rows.append(row)
            top_rows.append(
                {
                    "model": model,
                    "dimension": dimension,
                    "value": value,
                    "top1_strict_rate": float(np.nanmean(seed_top)),
                    "oracle_strict_rate": float(np.nanmean(seed_oracle)),
                }
            )
    return metrics_rows, top_rows


def calibration_curve(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["candidate_index"])].append(row)
    probability = np.asarray(
        [np.mean([float(row["prob_strict_gate20"]) for row in members]) for members in grouped.values()]
    )
    label = np.asarray([int(members[0]["strict_gate20"]) for members in grouped.values()])
    result = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        member = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 9 else probability < edges[index + 1]
        )
        if np.any(member):
            result.append(
                {
                    "bin": index,
                    "count": int(np.sum(member)),
                    "mean_probability": float(np.mean(probability[member])),
                    "observed_rate": float(np.mean(label[member])),
                }
            )
    return result


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
    colors = {"model_a": "#167d6d", "model_b": "#c04b32", "model_c": "#315caa"}
    for model, rows in curves.items():
        points = [
            (
                left + row["mean_probability"] * (right - left),
                bottom - row["observed_rate"] * (bottom - top),
            )
            for row in rows
        ]
        if len(points) > 1:
            draw.line(points, fill=colors[model], width=4)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[model])
    draw.text((350, 22), "Feature observability calibration", fill="#222222")
    draw.text((390, 735), "Mean predicted probability", fill="#222222")
    draw.text((20, 360), "Observed pass rate", fill="#222222")
    for index, model in enumerate(curves):
        y = 92 + index * 28
        draw.line((650, y, 688, y), fill=colors[model], width=4)
        draw.text((698, y - 7), model, fill="#222222")
    image.save(path)


def analyze(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    data = load_data(protocol)
    model_dirs = {
        "model_a": args.model_a_dir,
        "model_b": args.model_b_dir,
        "model_c": args.model_c_dir,
    }
    missing = [str(path) for path in model_dirs.values() if not (path / "training_summary.json").exists()]
    if missing:
        raise FileNotFoundError(f"Missing model summaries: {missing}")
    comparison, summaries = summarize_models(model_dirs, protocol)
    write_csv(args.out_dir / "observability_ablation.csv", comparison)
    group_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, Any]]] = {}
    for model, path in model_dirs.items():
        predictions = read_csv(path / "test_predictions.csv")
        grouped, top = prediction_groups(model, predictions, data)
        group_rows.extend(grouped)
        top_rows.extend(top)
        curves[model] = calibration_curve(predictions)
    write_csv(args.out_dir / "observability_group_metrics.csv", group_rows)
    write_csv(args.out_dir / "observability_group_top1.csv", top_rows)
    write_csv(
        args.out_dir / "observability_calibration_curve.csv",
        [dict(model=model, **row) for model, values in curves.items() for row in values],
    )
    plot_calibration(args.out_dir / "observability_calibration_curve.png", curves)
    audit = split_audit(data)
    passed = {
        row["model"]: bool(row["stage_a_model_gate_pass"])
        for row in comparison
    }
    e1_precision = {
        row["model"]: row["precision"]
        for row in group_rows
        if row["dimension"] == "drift_intensity" and row["value"] == "0.05"
    }
    e2_precision = {
        row["model"]: row["precision"]
        for row in group_rows
        if row["dimension"] == "drift_intensity" and row["value"] == "0.20"
    }
    precision_target = protocol["stage_a_gates"]["strict_precision_min"]
    e2_precision_gate = {
        model: bool(value >= precision_target) for model, value in e2_precision.items()
    }
    deployable_or_measurement = passed["model_a"] or passed["model_b"]
    summary = {
        "protocol": protocol["protocol"],
        "label_scope": protocol["evidence_scope"],
        "split_audit": audit,
        "model_gates": passed,
        "model_a_or_b_pass": deployable_or_measurement,
        "only_model_c_pass": bool(passed["model_c"] and not deployable_or_measurement),
        "e1_strict_precision_by_model": e1_precision,
        "e2_strict_precision_by_model": e2_precision,
        "e2_precision_gate_by_model": e2_precision_gate,
        "e2_critic_auto_acceptance_allowed": bool(
            e2_precision_gate.get("model_a", False)
            or e2_precision_gate.get("model_b", False)
        ),
        "stage_a_gate_pass": bool(audit["target_hash_leakage_free"] and deployable_or_measurement),
        "stage_b_allowed": bool(audit["target_hash_leakage_free"] and deployable_or_measurement),
        "engineering_critic_promoted": False,
        "automatic_hfss_admission_allowed": False,
        "decision": (
            "allow_stage_b_drift_envelope_preregistration"
            if audit["target_hash_leakage_free"] and deployable_or_measurement
            else "block_engineering_acceptance_due_to_non_deployable_or_failed_features"
        ),
    }
    (args.out_dir / "stage_a_acceptance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def status(args: argparse.Namespace) -> None:
    result = {
        "prepared": (args.out_dir / "prepare_summary.json").exists(),
        "model_a_complete": (args.model_a_dir / "training_summary.json").exists(),
        "model_b_complete": (args.model_b_dir / "training_summary.json").exists(),
        "model_c_complete": (args.model_c_dir / "training_summary.json").exists(),
        "analyzed": (args.out_dir / "stage_a_acceptance_summary.json").exists(),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode in ("prepare", "all"):
        prepare(args)
    if args.mode in ("analyze", "all"):
        analyze(args)
    if args.mode == "status":
        status(args)


if __name__ == "__main__":
    main()
