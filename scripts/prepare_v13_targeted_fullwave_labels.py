#!/usr/bin/env python3
"""Freeze 50 targeted K=2/4/6 full-wave residual-label candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    ROOT / "hfss_outputs" / "v12_k2_operating_envelope_validation_candidates_20260727_run01",
    ROOT / "hfss_outputs" / "v13_k4_operating_envelope_validation_candidates_20260727_run01",
    ROOT / "hfss_outputs" / "v11_operating_envelope_validation_candidates_20260727_run01",
)
DEFAULT_SUPPORT = ROOT / "hfss_outputs" / "v13_k246_operating_envelope_validation_20260727_run01" / "supported_scene_list.csv"
DEFAULT_SMOKE = ROOT / "hfss_outputs" / "v13_frozen_k246_hfss_smoke_dataset_20260727_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v13_targeted_fullwave_label_dataset_20260727_run01"
PRIMARY_COUNTS = {
    2: {"near_boundary": 6, "hard_positive": 4, "hard_negative": 4},
    4: {"near_boundary": 6, "hard_positive": 4, "hard_negative": 4},
    6: {"near_boundary": 8, "hard_positive": 5, "hard_negative": 4},
}
PAIR_COUNTS = {2: 2, 4: 1, 6: 2}
BOUNDARY_METRICS = ("psll", "nearest_iso", "local_iso", "active_rl")
MARGIN_INDEX = {"psll": 0, "nearest_iso": 1, "local_iso": 2, "mainlobe": 3, "active_rl": 4}
RATIOS = (0.5, 0.6, 0.7, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, action="append", default=[])
    parser.add_argument("--supported-scenes", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--smoke-dataset", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


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


def sha256(*values: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite targeted full-wave labels: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = tuple(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASETS
    packages = [(path, load_npz(path / "dataset_arrays.npz")) for path in dataset_dirs]
    with args.supported_scenes.open(newline="", encoding="utf-8-sig") as handle:
        inside_scenes = {
            int(row["sample_index"])
            for row in csv.DictReader(handle)
            if int(row["inside_envelope"]) == 1
        }
    smoke = load_npz(args.smoke_dataset / "dataset_arrays.npz")
    excluded_scenes = set(np.asarray(smoke["sample_index"], dtype=int).tolist())

    refs: list[dict[str, Any]] = []
    for path, data in packages:
        for index in range(int(data["candidate_indices"].size)):
            sample = int(data["sample_index"][index])
            if sample not in inside_scenes or sample in excluded_scenes:
                continue
            refs.append({"path": path, "data": data, "index": index})

    selected: list[dict[str, Any]] = []
    used_candidates: set[tuple[str, int]] = set()
    scene_usage: Counter[int] = Counter()
    ratio_cursor: defaultdict[tuple[int, str], int] = defaultdict(int)

    def candidate_key(ref: dict[str, Any]) -> tuple[str, int]:
        return (str(ref["path"].resolve()), int(ref["index"]))

    def choose(k_value: int, category: str, boundary_metric: str | None = None) -> dict[str, Any]:
        target_ratio = RATIOS[ratio_cursor[(k_value, category)] % len(RATIOS)]
        ratio_cursor[(k_value, category)] += 1
        eligible: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for ref in refs:
            data, index = ref["data"], int(ref["index"])
            if int(data["k_values"][index]) != k_value or candidate_key(ref) in used_candidates:
                continue
            if scene_usage[int(data["sample_index"][index])] >= 2:
                continue
            hard_positive = bool(data["hard_positive"][index])
            hard_negative = bool(data["hard_negative"][index])
            if category == "hard_positive" and not hard_positive:
                continue
            if category == "hard_negative" and not hard_negative:
                continue
            if category == "near_boundary" and (hard_positive or hard_negative):
                continue
            margins = np.asarray(data["actual_margins"][index], dtype=float)
            ratio = float(data["active_ratios_requested"][index])
            sample = int(data["sample_index"][index])
            diversity = scene_usage[sample]
            ratio_distance = abs(ratio - target_ratio)
            if category == "hard_positive":
                category_score = -float(np.min(margins))
            elif category == "hard_negative":
                category_score = float(np.maximum(-margins, 0.0).sum())
            else:
                assert boundary_metric is not None
                category_score = abs(float(margins[MARGIN_INDEX[boundary_metric]]))
            eligible.append(((diversity, ratio_distance, category_score, sample, index), ref))
        if not eligible:
            raise RuntimeError(f"No candidate for K={k_value} category={category}")
        chosen = min(eligible, key=lambda item: item[0])[1].copy()
        chosen["selection_category"] = category
        chosen["boundary_metric"] = boundary_metric or ""
        chosen["pair_group"] = ""
        used_candidates.add(candidate_key(chosen))
        scene_usage[int(chosen["data"]["sample_index"][chosen["index"]])] += 1
        selected.append(chosen)
        return chosen

    for k_value in (2, 4, 6):
        for position in range(PRIMARY_COUNTS[k_value]["near_boundary"]):
            choose(
                k_value,
                "near_boundary",
                BOUNDARY_METRICS[position % len(BOUNDARY_METRICS)],
            )
        for _ in range(PRIMARY_COUNTS[k_value]["hard_positive"]):
            choose(k_value, "hard_positive")
        for _ in range(PRIMARY_COUNTS[k_value]["hard_negative"]):
            choose(k_value, "hard_negative")

    for k_value in (2, 4, 6):
        bases = [
            ref
            for ref in selected
            if int(ref["data"]["k_values"][ref["index"]]) == k_value
            and not ref["pair_group"]
            and scene_usage[int(ref["data"]["sample_index"][ref["index"]])] == 1
        ]
        bases.sort(
            key=lambda ref: (
                float(ref["data"]["active_ratios_requested"][ref["index"]]),
                int(ref["data"]["sample_index"][ref["index"]]),
            )
        )
        for pair_position in range(PAIR_COUNTS[k_value]):
            if not bases:
                raise RuntimeError(f"No single-use K={k_value} base scene for ratio pair")
            base = bases.pop(0)
            data, base_index = base["data"], int(base["index"])
            sample = int(data["sample_index"][base_index])
            base_ratio = float(data["active_ratios_requested"][base_index])
            candidates = []
            for ref in refs:
                other_data, index = ref["data"], int(ref["index"])
                if other_data is not data or candidate_key(ref) in used_candidates:
                    continue
                if int(other_data["sample_index"][index]) != sample:
                    continue
                ratio = float(other_data["active_ratios_requested"][index])
                if np.isclose(ratio, base_ratio):
                    continue
                violation = float(np.maximum(-other_data["actual_margins"][index], 0.0).sum())
                candidates.append(((abs(ratio - base_ratio), violation, index), ref))
            if not candidates:
                raise RuntimeError(f"No paired-ratio candidate for scene {sample}")
            paired = min(candidates, key=lambda item: item[0])[1].copy()
            pair_group = f"k{k_value}_pair_{pair_position:02d}_s{sample}"
            base["pair_group"] = pair_group
            paired["selection_category"] = "paired_ratio"
            paired["boundary_metric"] = ""
            paired["pair_group"] = pair_group
            used_candidates.add(candidate_key(paired))
            scene_usage[sample] += 1
            selected.append(paired)

    if len(selected) != 50:
        raise RuntimeError(f"Expected 50 selected candidates, got {len(selected)}")

    first_data = selected[0]["data"]
    payload: dict[str, np.ndarray] = {}
    for key in first_data:
        candidate_level = all(
            key in ref["data"]
            and ref["data"][key].ndim >= 1
            and ref["data"][key].shape[0] == ref["data"]["candidate_indices"].size
            for ref in selected
        )
        payload[key] = (
            np.stack([ref["data"][key][ref["index"]] for ref in selected])
            if candidate_level
            else np.asarray(first_data[key])
        )
    count = len(selected)
    payload["candidate_index"] = np.arange(count, dtype=np.int64)
    payload["candidate_indices"] = np.arange(count, dtype=np.int64)
    payload["source_candidate_indices"] = np.asarray([ref["index"] for ref in selected], dtype=np.int64)
    payload["source_dataset_dirs"] = np.asarray([str(ref["path"].resolve()) for ref in selected])
    payload["selection_roles"] = np.asarray([ref["selection_category"] for ref in selected])
    payload["boundary_metrics"] = np.asarray([ref["boundary_metric"] for ref in selected])
    payload["pair_group_ids"] = np.asarray([ref["pair_group"] for ref in selected])

    manifest: list[dict[str, Any]] = []
    for local, ref in enumerate(selected):
        data, index = ref["data"], int(ref["index"])
        mask_hash = sha256(np.asarray(data["masks"][index], dtype=np.int8))
        nominal_hash = sha256(
            np.asarray(data["task_weights_real_imag"][index], dtype=np.float32),
            np.asarray(data["combined_weights_real_imag"][index], dtype=np.float32),
        )
        actual_hash = sha256(
            np.asarray(data["hfss_actual_task_weights_real_imag"][index], dtype=np.float32),
            np.asarray(data["hfss_actual_combined_weights_real_imag"][index], dtype=np.float32),
        )
        margins = np.asarray(data["actual_margins"][index], dtype=float)
        manifest.append(
            {
                "candidate_index": local,
                "sample_index": int(data["sample_index"][index]),
                "target_hash": str(data["target_hashes"][index]),
                "k_value": int(data["k_values"][index]),
                "ratio": float(data["active_ratios_requested"][index]),
                "selection_category": ref["selection_category"],
                "boundary_metric": ref["boundary_metric"],
                "pair_group": ref["pair_group"],
                "source_dataset": str(ref["path"].resolve()),
                "source_candidate_index": index,
                "eep_actual_strict_gate20": int(data["strict_gate20"][index]),
                "eep_nominal_strict_gate20": int(np.all(data["nominal_margins"][index] >= 0.0)),
                "margin_psll_db": float(margins[0]),
                "margin_nearest_iso_db": float(margins[1]),
                "margin_local_iso_db": float(margins[2]),
                "margin_mainlobe_db": float(margins[3]),
                "margin_active_rl_db": float(margins[4]),
                "mask_sha256": mask_hash,
                "nominal_weights_sha256": nominal_hash,
                "actual_weights_sha256": actual_hash,
            }
        )
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **payload)
    write_csv(args.out_dir / "frozen_selection_manifest.csv", manifest)
    categories = Counter(row["selection_category"] for row in manifest)
    k_counts = Counter(int(row["k_value"]) for row in manifest)
    summary = {
        "candidate_count": count,
        "independent_scene_count": len(scene_usage),
        "k_counts": {str(k): int(k_counts[k]) for k in (2, 4, 6)},
        "selection_category_counts": dict(sorted(categories.items())),
        "paired_ratio_candidate_count": int(categories["paired_ratio"]),
        "paired_scene_count": len({row["pair_group"] for row in manifest if row["pair_group"]}),
        "expected_hfss_case_count": int(sum(1 + int(row["k_value"]) for row in manifest)),
        "smoke_scene_overlap_count": int(len(set(scene_usage) & excluded_scenes)),
        "contains_ratio_1": False,
        "weights_frozen": True,
        "thresholds_frozen": True,
        "implementation_mismatch_enabled": True,
        "old_labels_included": False,
        "label_scope": "EEP/S256 targeted shortlist; not HFSS full-wave until exported",
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
