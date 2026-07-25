#!/usr/bin/env python3
"""Select the scene-paired v0.9 physical-margin development subset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "hfss_outputs" / "v09_eep_development_candidates_20260726_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_margin_development_dataset_20260726_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-scene", type=int, default=7)
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


def add_candidate(
    selected: list[int],
    reasons: dict[int, list[str]],
    candidate: int,
    reason: str,
    limit: int,
) -> None:
    if candidate not in selected and len(selected) < limit:
        selected.append(candidate)
    if candidate in selected and reason not in reasons[candidate]:
        reasons[candidate].append(reason)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v0.9 development subset: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.source_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    source_rows = read_csv(args.source_dir / "candidate_manifest.csv")
    count = int(data["candidate_indices"].size)
    if count != len(source_rows):
        raise RuntimeError("Candidate manifest and array package have different lengths")
    if np.any(np.isclose(data["active_ratios_requested"], 1.0)):
        raise RuntimeError("v0.9 development source unexpectedly contains ratio=1.0")

    by_scene: dict[int, list[int]] = defaultdict(list)
    for index, scene in enumerate(np.asarray(data["sample_index"], dtype=int)):
        by_scene[int(scene)].append(index)
    selected: list[int] = []
    reasons: dict[int, list[str]] = defaultdict(list)
    per_scene = max(4, int(args.per_scene))
    margins = np.asarray(data["actual_margins"], dtype=float)
    violations = np.asarray(data["strict_violation"], dtype=float)
    ratios = np.asarray(data["active_ratios_requested"], dtype=float)
    hard_positive = np.asarray(data["hard_positive"], dtype=bool)
    hard_negative = np.asarray(data["hard_negative"], dtype=bool)
    near_boundary = np.asarray(data["near_boundary"], dtype=bool)
    delta_norm = np.asarray(data["implementation_delta_norm"], dtype=float)

    for scene in sorted(by_scene):
        indices = np.asarray(by_scene[scene], dtype=np.int64)
        scene_selected: list[int] = []
        scene_reasons: dict[int, list[str]] = defaultdict(list)

        # Preserve one paired counterfactual at every sparse ratio.
        for ratio in sorted(np.unique(ratios[indices])):
            candidates = indices[np.isclose(ratios[indices], ratio)]
            key = violations[candidates] - 0.15 * hard_positive[candidates].astype(float)
            best = int(candidates[np.argmin(key)])
            add_candidate(scene_selected, scene_reasons, best, f"paired_ratio_{ratio:.1f}", per_scene)

        positives = indices[hard_positive[indices]]
        if positives.size:
            order = positives[
                np.lexsort((-np.min(margins[positives], axis=1), ratios[positives]))
            ]
            for candidate in order[:2]:
                add_candidate(scene_selected, scene_reasons, int(candidate), "hard_positive", per_scene)

        negatives = indices[hard_negative[indices]]
        if negatives.size:
            order = negatives[np.argsort(violations[negatives], kind="stable")]
            for candidate in order[:2]:
                add_candidate(scene_selected, scene_reasons, int(candidate), "hard_negative", per_scene)

        boundary = indices[near_boundary[indices]]
        if boundary.size:
            boundary_distance = np.min(np.abs(margins[boundary]), axis=1)
            order = boundary[np.argsort(boundary_distance, kind="stable")]
            for candidate in order:
                add_candidate(scene_selected, scene_reasons, int(candidate), "near_boundary", per_scene)
                if len(scene_selected) >= per_scene:
                    break

        if len(scene_selected) < per_scene:
            # Fill with diverse, moderate-error candidates rather than remote failures.
            score = violations[indices] + 0.10 * np.abs(delta_norm[indices] - np.median(delta_norm[indices]))
            for candidate in indices[np.argsort(score, kind="stable")]:
                add_candidate(scene_selected, scene_reasons, int(candidate), "feasibility_fill", per_scene)
                if len(scene_selected) >= per_scene:
                    break
        if len(scene_selected) != per_scene:
            raise RuntimeError(f"Scene {scene} produced {len(scene_selected)}/{per_scene} candidates")
        selected.extend(scene_selected)
        for candidate, values in scene_reasons.items():
            reasons[candidate].extend(values)

    selected_array = np.asarray(selected, dtype=np.int64)
    if np.unique(selected_array).size != selected_array.size:
        raise RuntimeError("Duplicate candidate selected across scenes")
    subset = {
        key: (
            value[selected_array]
            if value.ndim >= 1 and value.shape[0] == count
            else value
        )
        for key, value in data.items()
    }
    local = np.arange(selected_array.size, dtype=np.int64)
    subset["candidate_index"] = local
    subset["candidate_indices"] = local
    subset["source_candidate_indices"] = selected_array
    subset["selection_reason"] = np.asarray(
        ["+".join(reasons[int(index)]) for index in selected_array]
    )
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **subset)

    selected_rows: list[dict[str, Any]] = []
    for local_index, source_index in enumerate(selected_array.tolist()):
        row: dict[str, Any] = dict(source_rows[source_index])
        row["source_candidate_index"] = source_index
        row["candidate_index"] = local_index
        row["selection_reason"] = "+".join(reasons[source_index])
        selected_rows.append(row)
    write_csv(args.out_dir / "candidate_manifest.csv", selected_rows)

    distribution_rows: list[dict[str, Any]] = []
    split_names = {0: "train", 1: "val", 2: "test"}
    for split in (0, 1, 2):
        split_mask = np.asarray(subset["split_id"], dtype=int) == split
        for k_value in (2, 4, 6):
            for ratio in sorted(np.unique(subset["active_ratios_requested"])):
                group = split_mask & (subset["k_values"] == k_value) & np.isclose(
                    subset["active_ratios_requested"], ratio
                )
                if not np.any(group):
                    continue
                distribution_rows.append(
                    {
                        "split": split_names[split],
                        "k_value": k_value,
                        "ratio": float(ratio),
                        "count": int(np.sum(group)),
                        "scene_count": int(np.unique(subset["sample_index"][group]).size),
                        "gate15_rate": float(np.mean(subset["gate15"][group])),
                        "strict_gate20_rate": float(np.mean(subset["strict_gate20"][group])),
                        "hard_negative_count": int(np.sum(subset["hard_negative"][group])),
                        "hard_positive_count": int(np.sum(subset["hard_positive"][group])),
                        "near_boundary_count": int(np.sum(subset["near_boundary"][group])),
                    }
                )
    write_csv(args.out_dir / "distribution_by_split_k_ratio.csv", distribution_rows)

    summary = {
        "dataset_version": "v0.9-physical-margin-development",
        "source_dir": str(args.source_dir.resolve()),
        "source_candidate_count": count,
        "selected_candidate_count": int(selected_array.size),
        "scene_count": len(by_scene),
        "candidates_per_scene": per_scene,
        "split_scene_counts": {
            split_names[split]: int(
                np.unique(subset["sample_index"][subset["split_id"] == split]).size
            )
            for split in (0, 1, 2)
        },
        "gate15_count": int(np.sum(subset["gate15"])),
        "strict_gate20_count": int(np.sum(subset["strict_gate20"])),
        "hard_positive_count": int(np.sum(subset["hard_positive"])),
        "hard_negative_count": int(np.sum(subset["hard_negative"])),
        "near_boundary_count": int(np.sum(subset["near_boundary"])),
        "selection_reason_counts": dict(Counter(subset["selection_reason"].tolist())),
        "contains_ratio_1": False,
        "contains_nominal_control": False,
        "v08_used_for_tuning": False,
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
