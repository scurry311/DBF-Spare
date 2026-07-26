#!/usr/bin/env python3
"""Evaluate scene-level EEP/S256 best-of-N feasibility for a targeted pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--additional-dataset", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--oracle-target", type=float, default=0.90)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


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


def target_hashes(data: dict[str, np.ndarray]) -> set[str]:
    if "target_hashes" in data:
        return set(np.asarray(data["target_hashes"]).astype(str).tolist())
    return set()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite oracle evaluation: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load(args.dataset_dir / "dataset_arrays.npz")
    concatenated_keys = (
        "sample_index",
        "strict_gate20",
        "actual_margins",
        "active_ratios_requested",
        "k_values",
        "target_hashes",
        "min_target_separation_deg",
        "max_target_theta_deg",
        "large_scan",
        "small_target_gap",
    )
    for additional_dir in args.additional_dataset:
        additional = load(additional_dir / "dataset_arrays.npz")
        left_count = int(data["sample_index"].size)
        right_count = int(additional["sample_index"].size)
        for key in concatenated_keys:
            if key not in data and key not in additional:
                continue
            left = data.get(key, np.zeros(left_count, dtype=np.int8))
            right = additional.get(
                key, np.zeros(right_count, dtype=np.int8)
            )
            data[key] = np.concatenate((left, right), axis=0)
    sample_index = np.asarray(data["sample_index"], dtype=np.int64)
    strict = np.asarray(data["strict_gate20"], dtype=bool)
    margins = np.asarray(data["actual_margins"], dtype=np.float64)
    violation = np.maximum(-margins, 0.0).sum(axis=1)
    ratios = np.asarray(data["active_ratios_requested"], dtype=np.float64)
    k_values = np.asarray(data["k_values"], dtype=int)
    hashes = np.asarray(data["target_hashes"]).astype(str)
    excluded: set[str] = set()
    for source_path in args.exclude:
        path = source_path / "dataset_arrays.npz" if source_path.is_dir() else source_path
        if path.exists():
            excluded.update(target_hashes(load(path)))
    current_hashes = set(hashes.tolist())
    overlap = sorted(current_hashes & excluded)
    if overlap:
        raise RuntimeError(f"Target leakage detected for {len(overlap)} hashes")

    rows: list[dict[str, Any]] = []
    for scene in np.unique(sample_index):
        members = np.flatnonzero(sample_index == scene)
        feasible = members[strict[members]]
        if feasible.size:
            best = int(feasible[np.argmin(ratios[feasible])])
            minimum_ratio = float(np.min(ratios[feasible]))
        else:
            best = int(members[np.argmin(violation[members])])
            minimum_ratio = float("nan")
        row = {
            "sample_index": int(scene),
            "target_hash": str(hashes[best]),
            "k_value": int(k_values[best]),
            "candidate_count": int(members.size),
            "strict_candidate_count": int(feasible.size),
            "oracle_strict": int(feasible.size > 0),
            "oracle_minimum_ratio": minimum_ratio,
            "best_candidate_index": best,
            "best_candidate_ratio": float(ratios[best]),
            "best_strict_violation_db": float(violation[best]),
            "min_target_separation_deg": float(data["min_target_separation_deg"][best]),
            "max_target_theta_deg": float(data["max_target_theta_deg"][best]),
            "large_scan": int(data["large_scan"][best]),
            "small_target_gap": int(data.get("small_target_gap", np.zeros_like(sample_index))[best]),
        }
        for margin_index, name in enumerate(np.asarray(data["margin_names"]).astype(str)):
            row[f"best_margin_{name}_db"] = float(margins[best, margin_index])
        rows.append(row)
    write_csv(args.out_dir / "scene_oracle.csv", rows)

    group_rows: list[dict[str, Any]] = []
    for field, values in (
        ("all", ["all"]),
        ("k_value", sorted({row["k_value"] for row in rows})),
        ("large_scan", [0, 1]),
        ("small_target_gap", [0, 1]),
    ):
        for value in values:
            members = rows if field == "all" else [row for row in rows if row[field] == value]
            if not members:
                continue
            group_rows.append(
                {
                    "group_field": field,
                    "group_value": value,
                    "scene_count": len(members),
                    "oracle_scene_count": sum(row["oracle_strict"] for row in members),
                    "oracle_rate": float(np.mean([row["oracle_strict"] for row in members])),
                    "mean_strict_candidate_count": float(
                        np.mean([row["strict_candidate_count"] for row in members])
                    ),
                }
            )
    write_csv(args.out_dir / "oracle_groups.csv", group_rows)
    overall = next(row for row in group_rows if row["group_field"] == "all")
    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "additional_dataset_count": len(args.additional_dataset),
        "candidate_count": int(sample_index.size),
        "independent_scene_count": len(rows),
        "target_hash_count": len(current_hashes),
        "excluded_target_hash_count": len(excluded),
        "target_hash_overlap_count": 0,
        "oracle_scene_count": int(overall["oracle_scene_count"]),
        "oracle_rate": float(overall["oracle_rate"]),
        "oracle_target": float(args.oracle_target),
        "candidate_space_sufficient": bool(
            float(overall["oracle_rate"]) >= float(args.oracle_target)
        ),
        "critic_retraining_allowed": bool(
            float(overall["oracle_rate"]) >= float(args.oracle_target)
        ),
    }
    (args.out_dir / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
