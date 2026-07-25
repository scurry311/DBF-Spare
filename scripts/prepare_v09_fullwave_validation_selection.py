#!/usr/bin/env python3
"""Prepare the 72-candidate v0.9 full-wave set after HFSS smoke acceptance."""

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
DEFAULT_SMOKE_DATASET = ROOT / "hfss_outputs" / "v09_hfss_smoke_dataset_20260726_run02"
DEFAULT_SMOKE_HFSS = ROOT / "hfss_outputs" / "v09_hfss_smoke_20260726_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_fullwave_validation_dataset_20260726_run02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--smoke-dataset-dir", type=Path, default=DEFAULT_SMOKE_DATASET)
    parser.add_argument("--smoke-hfss-dir", type=Path, default=DEFAULT_SMOKE_HFSS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-scene", type=int, default=6)
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
    acceptance = json.loads(
        (args.smoke_hfss_dir / "v09_smoke_acceptance.json").read_text(encoding="utf-8")
    )
    if not bool(acceptance["acceptance"]["open_50_100_fullwave"]):
        raise RuntimeError("HFSS smoke did not authorize the 50-100 candidate stage")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite full-wave selection: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    with np.load(args.smoke_dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        smoke = {key: source[key] for key in source.files}
    predictions = read_csv(args.critic_dir / "test_predictions.csv")
    by_candidate: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        by_candidate[int(row["candidate_index"])].append(row)
    score = {
        candidate: float(np.mean([float(row["ranking_score"]) for row in rows]))
        for candidate, rows in by_candidate.items()
    }
    probability = {
        candidate: float(np.mean([float(row["prob_strict_gate20"]) for row in rows]))
        for candidate, rows in by_candidate.items()
    }
    test_indices = np.asarray(sorted(by_candidate), dtype=np.int64)
    smoke_indices = set(np.asarray(smoke["development_candidate_indices"], dtype=int).tolist())
    selected = [int(index) for index in test_indices if int(index) not in smoke_indices]
    reasons: dict[int, list[str]] = defaultdict(list)
    for candidate in selected:
        reasons[candidate].append("heldout_non_smoke")

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
    subset["fullwave_selection_reason"] = np.asarray(
        ["+".join(reasons[index]) for index in selected]
    )
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **subset)
    rows = [
        {
            "candidate_index": local,
            "development_candidate_index": index,
            "sample_index": int(data["sample_index"][index]),
            "k_value": int(data["k_values"][index]),
            "ratio": float(data["active_ratios_requested"][index]),
            "mean_ranking_score": score[index],
            "mean_strict_probability": probability[index],
            "selection_reason": "+".join(reasons[index]),
            "eep_strict_gate20": int(data["strict_gate20"][index]),
        }
        for local, index in enumerate(selected)
    ]
    write_csv(args.out_dir / "fullwave_selection.csv", rows)
    summary = {
        "candidate_count": int(selected_array.size),
        "independent_scene_count": int(np.unique(subset["sample_index"]).size),
        "candidates_per_scene_min": int(
            min(np.sum(subset["sample_index"] == scene) for scene in np.unique(subset["sample_index"]))
        ),
        "candidates_per_scene_max": int(
            max(np.sum(subset["sample_index"] == scene) for scene in np.unique(subset["sample_index"]))
        ),
        "expected_hfss_case_count": int(np.sum(1 + subset["k_values"])),
        "smoke_replicate_count": int(sum(index in smoke_indices for index in selected)),
        "eep_strict_positive_count": int(np.sum(subset["strict_gate20"])),
        "contains_ratio_1": False,
        "contains_nominal_control": False,
        "smoke_acceptance": acceptance["acceptance"],
    }
    if not 50 <= int(selected_array.size) <= 100:
        raise RuntimeError("Full-wave selection is outside the required 50-100 range")
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
