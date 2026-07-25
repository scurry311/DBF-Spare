#!/usr/bin/env python3
"""Select one probability-spanning candidate per v0.9 validation scene."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "v09_margin_development_dataset_20260726_run01"
DEFAULT_PREDICTIONS = ROOT / "hfss_outputs" / "v09_adaptive_ratio_eep_loop_20260726_run01" / "candidate_predictions.csv"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_hfss_calibration_dataset_20260726_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
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
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite HFSS calibration selection: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    pool_probability = {
        int(row["candidate_index"]): float(row["strict_probability_mean"])
        for row in read_csv(args.predictions)
    }
    val_indices = np.flatnonzero(np.asarray(data["split_id"], dtype=int) == 1)
    scenes = np.unique(data["sample_index"][val_indices])
    selected: list[int] = []
    rows: list[dict[str, Any]] = []
    modes = ("near_05", "high", "low")
    for scene_position, scene in enumerate(scenes):
        group = val_indices[data["sample_index"][val_indices] == scene]
        probability = np.asarray(
            [pool_probability[int(data["source_candidate_indices"][index])] for index in group]
        )
        mode = modes[scene_position % len(modes)]
        if mode == "near_05":
            local = int(np.argmin(np.abs(probability - 0.5)))
        elif mode == "high":
            local = int(np.argmax(probability))
        else:
            local = int(np.argmin(probability))
        candidate = int(group[local])
        selected.append(candidate)
        rows.append(
            {
                "candidate_index": scene_position,
                "development_candidate_index": candidate,
                "source_pool_candidate_index": int(data["source_candidate_indices"][candidate]),
                "sample_index": int(scene),
                "k_value": int(data["k_values"][candidate]),
                "ratio": float(data["active_ratios_requested"][candidate]),
                "selection_mode": mode,
                "uncalibrated_strict_probability": float(probability[local]),
                "eep_strict_gate": int(data["strict_gate20"][candidate]),
            }
        )
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
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **subset)
    write_csv(args.out_dir / "calibration_selection.csv", rows)
    summary = {
        "candidate_count": int(selected_array.size),
        "independent_validation_scene_count": int(scenes.size),
        "expected_hfss_case_count": int(np.sum(1 + subset["k_values"])),
        "selection_mode_counts": {
            mode: sum(row["selection_mode"] == mode for row in rows) for mode in modes
        },
        "probability_min": float(min(row["uncalibrated_strict_probability"] for row in rows)),
        "probability_max": float(max(row["uncalibrated_strict_probability"] for row in rows)),
        "test_labels_used_for_selection": False,
        "contains_ratio_1": False,
        "contains_nominal_control": False,
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
