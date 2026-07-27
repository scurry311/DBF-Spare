#!/usr/bin/env python3
"""Build scene-grouped v1.3 HFSS physical-margin residual critic labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--hfss-dir", type=Path, required=True)
    parser.add_argument("--active-audit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_group_split(
    scene: np.ndarray, k_values: np.ndarray, strict: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    unique_scenes = np.unique(scene)
    scene_k = {int(value): int(k_values[np.flatnonzero(scene == value)[0]]) for value in unique_scenes}
    rng = np.random.default_rng(seed)
    best: tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    for _ in range(20000):
        shuffled = rng.permutation(unique_scenes)
        train = shuffled[:29]
        val = shuffled[29:35]
        test = shuffled[35:]
        groups = (train, val, test)
        valid = True
        score = 0.0
        overall_rate = float(np.mean(strict))
        for values in groups:
            indices = np.flatnonzero(np.isin(scene, values))
            labels = strict[indices]
            if len(np.unique(labels)) < 2:
                valid = False
                break
            present_k = {scene_k[int(value)] for value in values}
            if present_k != {2, 4, 6}:
                valid = False
                break
            score += abs(float(np.mean(labels)) - overall_rate)
        if valid and (best is None or score < best[0]):
            best = (score, groups)
    if best is None:
        raise RuntimeError("Unable to build class-balanced scene-grouped split")
    split_id = np.full(scene.size, -1, dtype=np.int8)
    split_rows: list[dict[str, Any]] = []
    for group_id, (name, values) in enumerate(zip(("train", "val", "test"), best[1])):
        indices = np.flatnonzero(np.isin(scene, values))
        split_id[indices] = group_id
        split_rows.append(
            {
                "split": name,
                "scene_count": int(values.size),
                "candidate_count": int(indices.size),
                "strict_positive_count": int(np.sum(strict[indices])),
                "strict_negative_count": int(indices.size - np.sum(strict[indices])),
                "k_scene_counts": {
                    str(k): int(sum(scene_k[int(value)] == k for value in values))
                    for k in (2, 4, 6)
                },
                "sample_indices": [int(value) for value in sorted(values.tolist())],
            }
        )
    if np.any(split_id < 0):
        raise RuntimeError("Incomplete split assignment")
    return split_id, {"seed": seed, "splits": split_rows, "scene_leakage_free": True}


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.3 critic dataset: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.candidate_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    labels = {int(row["candidate_index"]): row for row in read_csv(args.hfss_dir / "candidate_residual_labels.csv")}
    active = {int(row["candidate_index"]): row for row in read_csv(args.active_audit_dir / "candidate_active_rl_semantics.csv")}
    count = int(data["candidate_indices"].size)
    if set(labels) != set(range(count)) or set(active) != set(range(count)):
        raise RuntimeError("Candidate index mismatch across frozen, HFSS, and active-RL packages")

    actual_margins = np.zeros((count, 5), dtype=np.float32)
    label_rows: list[dict[str, Any]] = []
    for candidate in range(count):
        row = labels[candidate]
        active_row = active[candidate]
        mainlobe_margin = min(
            float(row["hfss_mainlobe_gain_db"])
            - (float(row["eep_mainlobe_gain_db"]) - 0.5),
            3.0 - float(row["hfss_target_spread_db"]),
            1.5 - float(row["hfss_pointing_error_deg"]),
        )
        active_margin = min(
            float(active_row["combined_worst_active_rl_db"]) - 10.0,
            float(active_row["combined_total_rl_db"]) - 10.0,
            float(active_row["worst_significant_case_active_rl_db"]) - 10.0,
        )
        values = np.asarray(
            [
                -float(row["hfss_psll_db"]),
                float(row["hfss_nearest_iso_db"]) - 25.0,
                float(row["hfss_local_iso_db"]) - 20.0,
                mainlobe_margin,
                active_margin,
            ],
            dtype=np.float32,
        )
        actual_margins[candidate] = values
        label_rows.append(
            {
                "candidate_index": candidate,
                "sample_index": int(row["sample_index"]),
                "k_value": int(row["k"]),
                "ratio": float(row["ratio"]),
                "selection_role": row["selection_role"],
                "margin_psll_db": float(values[0]),
                "margin_nearest_iso_db": float(values[1]),
                "margin_local_iso_db": float(values[2]),
                "margin_mainlobe_db": float(values[3]),
                "margin_active_rl_db": float(values[4]),
                "strict_engineering_gate": int(np.all(values >= 0.0)),
            }
        )
    nominal_margins = np.asarray(data["nominal_margins"], dtype=np.float32)
    residual = actual_margins - nominal_margins
    gate15 = (
        (actual_margins[:, 0] >= 0.0)
        & (actual_margins[:, 1] >= 0.0)
        & (actual_margins[:, 2] >= -5.0)
    ).astype(np.int8)
    strict = np.all(actual_margins >= 0.0, axis=1).astype(np.int8)
    mainlobe = (actual_margins[:, 3] >= 0.0).astype(np.int8)
    active_gate = (actual_margins[:, 4] >= 0.0).astype(np.int8)
    hard_negative = (np.all(nominal_margins >= 0.0, axis=1) & (strict == 0)).astype(np.int8)
    hard_positive = strict.copy()
    scaled = np.abs(actual_margins) / np.asarray([3.0, 5.0, 5.0, 0.5, 2.0])[None, :]
    near_boundary = (np.min(scaled, axis=1) <= 0.5).astype(np.int8)
    scene = np.asarray(data["sample_index"], dtype=np.int64)
    k_values = np.asarray(data["k_values"], dtype=np.int8)
    split_id, split_payload = choose_group_split(scene, k_values, strict, int(args.seed))

    payload = dict(data)
    payload["split_id"] = split_id
    payload["nominal_margins"] = nominal_margins
    payload["actual_margins"] = actual_margins
    payload["margin_residuals"] = residual.astype(np.float32)
    payload["gate15"] = gate15
    payload["strict_gate20"] = strict
    payload["mainlobe_gate"] = mainlobe
    payload["active_rl_gate"] = active_gate
    payload["hard_negative"] = hard_negative
    payload["hard_positive"] = hard_positive
    payload["near_boundary"] = near_boundary
    payload["hfss_margin_semantics"] = np.asarray(
        ["psll", "nearest_iso", "local20_iso", "mainlobe", "combined_plus_significant_task_active_rl"]
    )
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **payload)
    for row, split in zip(label_rows, split_id):
        row["split"] = ("train", "val", "test")[int(split)]
    write_csv(args.out_dir / "fullwave_margin_labels.csv", label_rows)
    (args.out_dir / "grouped_split_manifest.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    summary = {
        "candidate_count": count,
        "independent_scene_count": int(np.unique(scene).size),
        "strict_engineering_positive_count": int(np.sum(strict)),
        "strict_engineering_negative_count": int(count - np.sum(strict)),
        "hard_negative_count": int(np.sum(hard_negative)),
        "hard_positive_count": int(np.sum(hard_positive)),
        "near_boundary_count": int(np.sum(near_boundary)),
        "split": split_payload,
        "scene_leakage_free": True,
        "smoke_labels_included": False,
        "old_labels_included": False,
        "active_rl_semantics": "combined all active ports plus -20 dB significant task ports",
        "thresholds_changed": False,
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
