#!/usr/bin/env python3
"""Run the no-control v0.9 adaptive-ratio loop on the full EEP candidate pool."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_v09_physical_margin_critic import (
    MARGIN_SCALE,
    PhysicalMarginCritic,
    apply_temperature,
    derived_probabilities,
    scalar_features,
    spatial_features,
    target_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "hfss_outputs" / "v09_eep_development_candidates_20260726_run01"
DEFAULT_CRITIC = ROOT / "hfss_outputs" / "v09_physical_margin_critic_20260726_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_adaptive_ratio_eep_loop_20260726_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
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


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite adaptive-ratio result: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.pool_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if np.any(np.isclose(data["active_ratios_requested"], 1.0)):
        raise RuntimeError("Adaptive sparse pool contains ratio=1.0")
    spatial = spatial_features(data["masks"], data["nominal_external_task_weights_real_imag"])
    targets = target_features(data["targets_deg"], data["task_valid"])
    scalars_raw, scalar_names = scalar_features(data)
    nominal_margin = np.asarray(data["nominal_margins"], dtype=np.float32)
    seed_dirs = sorted(path for path in args.critic_dir.glob("seed_*") if path.is_dir())
    if len(seed_dirs) < 3:
        raise RuntimeError("At least three critic checkpoints are required")

    all_margin: list[np.ndarray] = []
    all_lcb: list[np.ndarray] = []
    all_probability: list[np.ndarray] = []
    for seed_dir in seed_dirs:
        checkpoint = torch.load(seed_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        if list(checkpoint["scalar_names"]) != scalar_names:
            raise RuntimeError(f"Scalar schema mismatch for {seed_dir.name}")
        scalars = (
            (scalars_raw - np.asarray(checkpoint["scalar_mean"]))
            / np.asarray(checkpoint["scalar_std"])
        ).astype(np.float32)
        model = PhysicalMarginCritic(scalars.shape[1])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        mean_parts: list[np.ndarray] = []
        sigma_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, spatial.shape[0], int(args.batch_size)):
                stop = min(spatial.shape[0], start + int(args.batch_size))
                mean_std, logvar = model(
                    torch.from_numpy(spatial[start:stop]),
                    torch.from_numpy(targets[start:stop]),
                    torch.from_numpy(scalars[start:stop]),
                )
                mean_parts.append(mean_std.numpy())
                sigma_parts.append(np.exp(0.5 * logvar.numpy()))
        residual_mean = np.asarray(checkpoint["residual_mean"])
        residual_std = np.asarray(checkpoint["residual_std"])
        residual = np.concatenate(mean_parts) * residual_std + residual_mean
        sigma = np.concatenate(sigma_parts) * residual_std
        predicted_margin = nominal_margin + residual
        probability = derived_probabilities(predicted_margin, sigma)["strict_gate20"]
        probability = apply_temperature(
            probability, float(checkpoint["temperatures"]["strict_gate20"])
        )
        kappa = float(checkpoint["uncertainty_kappa"])
        all_margin.append(predicted_margin.astype(np.float32))
        all_lcb.append((predicted_margin - kappa * sigma).astype(np.float32))
        all_probability.append(probability.astype(np.float32))

    predicted_margin = np.mean(np.stack(all_margin), axis=0)
    conservative_margin = np.mean(np.stack(all_lcb), axis=0)
    probability = np.mean(np.stack(all_probability), axis=0)
    probability_std = np.std(np.stack(all_probability), axis=0)
    calibrator: dict[str, Any] | None = None
    if args.calibrator is not None:
        calibrator = json.loads(args.calibrator.read_text(encoding="utf-8"))
        clipped = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
        logit = np.log(clipped / (1.0 - clipped))
        probability = 1.0 / (
            1.0
            + np.exp(
                -(
                    float(calibrator["slope"]) * logit
                    + float(calibrator["intercept"])
                )
            )
        )
    ratios = np.asarray(data["active_ratios_requested"], dtype=np.float32)
    score = np.min(conservative_margin / MARGIN_SCALE[None, :], axis=1) - 0.15 * ratios
    actual_strict = np.asarray(data["strict_gate20"], dtype=bool)
    scenes = np.asarray(data["sample_index"], dtype=np.int64)

    candidate_rows: list[dict[str, Any]] = []
    for index in range(scenes.size):
        row: dict[str, Any] = {
            "candidate_index": index,
            "sample_index": int(scenes[index]),
            "split_id": int(data["split_id"][index]),
            "k_value": int(data["k_values"][index]),
            "ratio": float(ratios[index]),
            "mask_family": str(data["variant_kind"][index]),
            "strict_probability_mean": float(probability[index]),
            "strict_probability_seed_std": float(probability_std[index]),
            "conservative_ranking_score": float(score[index]),
            "predicted_admissible": int(
                probability[index] >= float(args.probability_threshold)
                and np.all(conservative_margin[index] >= 0.0)
            ),
            "eep_actual_strict_gate": int(actual_strict[index]),
        }
        for margin_index, name in enumerate(data["margin_names"].tolist()):
            row[f"predicted_margin_{name}_db"] = float(predicted_margin[index, margin_index])
            row[f"conservative_margin_{name}_db"] = float(conservative_margin[index, margin_index])
            row[f"actual_margin_{name}_db"] = float(data["actual_margins"][index, margin_index])
        candidate_rows.append(row)
    write_csv(args.out_dir / "candidate_predictions.csv", candidate_rows)

    scene_rows: list[dict[str, Any]] = []
    for scene_id in np.unique(scenes):
        group = np.flatnonzero(scenes == scene_id)
        oracle_ratio: float | None = None
        selected: int | None = None
        fallback: int | None = None
        for ratio in sorted(np.unique(ratios[group])):
            ratio_group = group[np.isclose(ratios[group], ratio)]
            best = int(ratio_group[np.argmax(score[ratio_group])])
            fallback = best
            if oracle_ratio is None and np.any(actual_strict[ratio_group]):
                oracle_ratio = float(ratio)
            if (
                selected is None
                and probability[best] >= float(args.probability_threshold)
                and np.all(conservative_margin[best] >= 0.0)
            ):
                selected = best
                break
        output = selected if selected is not None else int(fallback)
        scene_rows.append(
            {
                "sample_index": int(scene_id),
                "split_id": int(data["split_id"][output]),
                "k_value": int(data["k_values"][output]),
                "selected_candidate_index": output,
                "selected_ratio": float(ratios[output]),
                "admitted": int(selected is not None),
                "selected_eep_actual_strict_gate": int(actual_strict[output]),
                "oracle_has_strict_candidate": int(oracle_ratio is not None),
                "oracle_minimum_ratio": oracle_ratio if oracle_ratio is not None else float("nan"),
                "strict_probability": float(probability[output]),
                "ranking_score": float(score[output]),
                "activation_reduction_vs_ratio1": 1.0 - float(ratios[output]),
                "fallback_used": int(selected is None),
            }
        )
    write_csv(args.out_dir / "adaptive_scene_selections.csv", scene_rows)

    split_names = {0: "train", 1: "val", 2: "test"}
    split_summary: dict[str, Any] = {}
    for split, name in split_names.items():
        rows = [row for row in scene_rows if int(row["split_id"]) == split]
        admitted = [row for row in rows if int(row["admitted"]) == 1]
        split_summary[name] = {
            "scene_count": len(rows),
            "admission_rate": len(admitted) / max(len(rows), 1),
            "admitted_strict_precision": float(
                np.mean([row["selected_eep_actual_strict_gate"] for row in admitted])
            ) if admitted else float("nan"),
            "all_scene_selected_strict_rate": float(
                np.mean([row["selected_eep_actual_strict_gate"] for row in rows])
            ),
            "oracle_scene_rate": float(
                np.mean([row["oracle_has_strict_candidate"] for row in rows])
            ),
            "mean_selected_ratio": float(np.mean([row["selected_ratio"] for row in rows])),
            "admitted_ratio_counts": dict(Counter(f"{row['selected_ratio']:.1f}" for row in admitted)),
        }
    summary = {
        "stage": "v0.9-adaptive-ratio-eep-loop",
        "candidate_count": int(scenes.size),
        "scene_count": len(scene_rows),
        "checkpoint_count": len(seed_dirs),
        "probability_threshold": float(args.probability_threshold),
        "calibrator": calibrator,
        "contains_ratio_1": False,
        "contains_nominal_control": False,
        "metric_scope": "EEP/S256 only; HFSS confirmation pending",
        "split_summary": split_summary,
    }
    (args.out_dir / "adaptive_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
