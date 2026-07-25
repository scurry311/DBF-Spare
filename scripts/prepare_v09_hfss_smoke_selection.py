#!/usr/bin/env python3
"""Select 15-20 no-control v0.9 candidates for prospective HFSS smoke."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "v09_margin_development_dataset_20260726_run01"
DEFAULT_CRITIC = ROOT / "hfss_outputs" / "v09_physical_margin_critic_20260726_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_hfss_smoke_dataset_20260726_run02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate-count", type=int, default=18)
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


def main() -> None:
    args = parse_args()
    if not 15 <= int(args.candidate_count) <= 20:
        raise ValueError("candidate-count must be in [15, 20]")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite HFSS smoke selection: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    predictions = read_csv(args.critic_dir / "test_predictions.csv")
    training_summary = json.loads((args.critic_dir / "training_summary.json").read_text(encoding="utf-8"))
    by_candidate: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        by_candidate[int(row["candidate_index"])].append(row)
    if not by_candidate:
        raise RuntimeError("No test predictions available")

    aggregates: dict[int, dict[str, float]] = {}
    for candidate, rows in by_candidate.items():
        aggregates[candidate] = {
            "mean_ranking_score": float(np.mean([float(row["ranking_score"]) for row in rows])),
            "mean_strict_probability": float(np.mean([float(row["prob_strict_gate20"]) for row in rows])),
            "mean_gate15_probability": float(np.mean([float(row["prob_gate15"]) for row in rows])),
            "mean_margin_sigma": float(
                np.mean(
                    [
                        float(value)
                        for row in rows
                        for key, value in row.items()
                        if key.startswith("sigma_margin_")
                    ]
                )
            ),
        }
    test_indices = np.asarray(sorted(by_candidate), dtype=np.int64)
    test_scenes = np.unique(data["sample_index"][test_indices])
    strict_gate = np.asarray(data["strict_gate20"], dtype=bool)
    if int(np.sum(strict_gate[test_indices])) < 1:
        raise RuntimeError("No EEP/S256 strict positive in the held-out test scenes")

    selected: list[int] = []
    reason: dict[int, str] = {}
    # One model top-1 candidate per unseen test scene is the primary prospective set.
    for scene in test_scenes:
        group = test_indices[data["sample_index"][test_indices] == scene]
        best = max(group.tolist(), key=lambda index: aggregates[index]["mean_ranking_score"])
        selected.append(best)
        reason[best] = "scene_top1"

    remaining = [index for index in test_indices.tolist() if index not in selected]
    # Add EEP hard positives, then boundary/high-uncertainty probes with K=6 emphasis.
    priority = sorted(
        remaining,
        key=lambda index: (
            int(strict_gate[index]),
            int(data["k_values"][index] == 6),
            int(
                np.isclose(float(data["active_ratios_requested"][index]), 0.6)
                or np.isclose(float(data["active_ratios_requested"][index]), 0.7)
            ),
            int(data["near_boundary"][index]),
            aggregates[index]["mean_margin_sigma"],
            aggregates[index]["mean_strict_probability"],
        ),
        reverse=True,
    )
    for index in priority:
        if len(selected) >= int(args.candidate_count):
            break
        selected.append(index)
        reason[index] = (
            "eep_hard_positive" if strict_gate[index] else "boundary_uncertainty_probe"
        )
    if len(selected) != int(args.candidate_count):
        raise RuntimeError("Insufficient held-out test candidates for HFSS smoke")

    selected_array = np.asarray(selected, dtype=np.int64)
    source_count = int(data["candidate_indices"].size)
    subset = {
        key: (
            value[selected_array]
            if value.ndim >= 1 and value.shape[0] == source_count
            else value
        )
        for key, value in data.items()
    }
    subset["candidate_index"] = np.arange(selected_array.size, dtype=np.int64)
    subset["candidate_indices"] = np.arange(selected_array.size, dtype=np.int64)
    subset["development_candidate_indices"] = selected_array
    subset["critic_selection_reason"] = np.asarray([reason[index] for index in selected])
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **subset)

    rows: list[dict[str, Any]] = []
    for local, index in enumerate(selected):
        rows.append(
            {
                "candidate_index": local,
                "development_candidate_index": index,
                "source_pool_candidate_index": int(data["source_candidate_indices"][index]),
                "sample_index": int(data["sample_index"][index]),
                "k_value": int(data["k_values"][index]),
                "ratio": float(data["active_ratios_requested"][index]),
                "selection_reason": reason[index],
                "eep_strict_gate20": int(strict_gate[index]),
                **aggregates[index],
            }
        )
    write_csv(args.out_dir / "smoke_selection.csv", rows)
    summary = {
        "candidate_count": len(selected),
        "independent_test_scene_count": int(np.unique(subset["sample_index"]).size),
        "expected_hfss_case_count": int(np.sum(1 + subset["k_values"])),
        "eep_strict_positive_count": int(np.sum(subset["strict_gate20"])),
        "k6_count": int(np.sum(subset["k_values"] == 6)),
        "ratio_06_07_count": int(
            np.sum(
                np.isclose(subset["active_ratios_requested"], 0.6)
                | np.isclose(subset["active_ratios_requested"], 0.7)
            )
        ),
        "contains_ratio_1": False,
        "contains_nominal_control": False,
        "critic_acceptance": training_summary["acceptance"],
        "selection_scope": "held-out v0.9 development test scenes",
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
