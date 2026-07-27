#!/usr/bin/env python3
"""Freeze a scene-independent 7/6/7 K=2/4/6 HFSS smoke shortlist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    ROOT / "hfss_outputs" / "v12_k2_operating_envelope_validation_candidates_20260727_run01",
    ROOT / "hfss_outputs" / "v13_k4_operating_envelope_validation_candidates_20260727_run01",
    ROOT / "hfss_outputs" / "v11_operating_envelope_validation_candidates_20260727_run01",
)
DEFAULT_SUPPORT = (
    ROOT
    / "hfss_outputs"
    / "v13_k246_operating_envelope_validation_20260727_run01"
    / "supported_scene_list.csv"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "v13_frozen_k246_hfss_smoke_dataset_20260727_run01"
SCHEDULE = {
    2: (
        ("strict_positive", 0.5),
        ("strict_positive", 0.5),
        ("strict_positive", 0.6),
        ("nearest_boundary", 0.6),
        ("nearest_boundary", 0.7),
        ("active_rl_boundary", 0.5),
        ("active_rl_boundary", 0.8),
    ),
    4: (
        ("strict_positive", 0.5),
        ("strict_positive", 0.6),
        ("psll_boundary", 0.5),
        ("nearest_boundary", 0.7),
        ("local_boundary", 0.8),
        ("active_rl_boundary", 0.5),
    ),
    6: (
        ("strict_positive", 0.5),
        ("strict_positive", 0.6),
        ("strict_positive", 0.7),
        ("psll_boundary", 0.5),
        ("nearest_boundary", 0.6),
        ("local_boundary", 0.8),
        ("active_rl_boundary", 0.5),
    ),
}
MARGIN_INDEX = {"psll": 0, "nearest": 1, "local": 2, "mainlobe": 3, "active_rl": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, action="append", default=[])
    parser.add_argument("--supported-scenes", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def read_supported(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            int(row["sample_index"])
            for row in csv.DictReader(handle)
            if int(row["inside_envelope"]) == 1 and int(row["oracle_strict"]) == 1
        }


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


def digest(*values: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def candidate_order(
    margins: np.ndarray, role: str, indices: np.ndarray
) -> np.ndarray:
    if role == "strict_positive":
        reserve = np.min(margins[indices], axis=1)
        secondary = np.sum(np.clip(margins[indices], 0.0, 6.0), axis=1)
        return indices[np.lexsort((-secondary, -reserve))]
    margin_name = role.removesuffix("_boundary")
    target = MARGIN_INDEX[margin_name]
    other = np.delete(margins[indices], target, axis=1)
    other_floor = np.min(other, axis=1)
    # Prefer the closest inside-boundary candidate, then the largest reserve on
    # every non-target gate so the probe remains physically interpretable.
    return indices[np.lexsort((-other_floor, margins[indices, target]))]


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen HFSS smoke: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = tuple(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASETS
    packages = [(path, load_npz(path / "dataset_arrays.npz")) for path in dataset_dirs]
    supported = read_supported(args.supported_scenes)

    selected: list[tuple[Path, dict[str, np.ndarray], int, str]] = []
    used_scenes: set[int] = set()
    for k_value, slots in SCHEDULE.items():
        matching = [(path, data) for path, data in packages if np.any(data["k_values"] == k_value)]
        if not matching:
            raise RuntimeError(f"No source package for K={k_value}")
        purity = [float(np.mean(data["k_values"] == k_value)) for _path, data in matching]
        best_purity = max(purity)
        best_sources = [pair for pair, value in zip(matching, purity) if value == best_purity]
        if len(best_sources) != 1:
            raise RuntimeError(f"Ambiguous highest-purity source package for K={k_value}")
        path, data = best_sources[0]
        sample = np.asarray(data["sample_index"], dtype=np.int64)
        k_values = np.asarray(data["k_values"], dtype=int)
        ratios = np.asarray(data["active_ratios_requested"], dtype=float)
        strict = np.asarray(data["strict_gate20"], dtype=bool)
        margins = np.asarray(data["actual_margins"], dtype=float)
        for role, ratio in slots:
            eligible = np.flatnonzero(
                (k_values == k_value)
                & np.isclose(ratios, ratio, atol=1.0e-5)
                & strict
                & np.isin(sample, np.asarray(sorted(supported), dtype=np.int64))
            )
            ordered = candidate_order(margins, role, eligible)
            chosen = next(
                (int(index) for index in ordered if int(sample[index]) not in used_scenes),
                None,
            )
            if chosen is None:
                raise RuntimeError(f"No independent K={k_value} {role} ratio={ratio} candidate")
            used_scenes.add(int(sample[chosen]))
            selected.append((path, data, chosen, role))

    first_data = selected[0][1]
    payload: dict[str, np.ndarray] = {}
    for key in first_data:
        candidate_level = all(
            key in data
            and data[key].ndim >= 1
            and data[key].shape[0] == data["candidate_indices"].size
            for _path, data, _index, _role in selected
        )
        if candidate_level:
            payload[key] = np.stack([data[key][index] for _path, data, index, _role in selected])
        else:
            payload[key] = np.asarray(first_data[key])
    count = len(selected)
    payload["candidate_index"] = np.arange(count, dtype=np.int64)
    payload["candidate_indices"] = np.arange(count, dtype=np.int64)
    payload["source_candidate_indices"] = np.asarray(
        [index for _path, _data, index, _role in selected], dtype=np.int64
    )
    payload["source_dataset_dirs"] = np.asarray(
        [str(path.resolve()) for path, _data, _index, _role in selected]
    )
    payload["selection_roles"] = np.asarray([role for _path, _data, _index, role in selected])

    mask_hashes: list[str] = []
    nominal_hashes: list[str] = []
    actual_hashes: list[str] = []
    manifest: list[dict[str, Any]] = []
    for local, (path, data, index, role) in enumerate(selected):
        mask_hash = digest(np.asarray(data["masks"][index], dtype=np.int8))
        nominal_hash = digest(
            np.asarray(data["task_weights_real_imag"][index], dtype=np.float32),
            np.asarray(data["combined_weights_real_imag"][index], dtype=np.float32),
        )
        actual_hash = digest(
            np.asarray(data["hfss_actual_task_weights_real_imag"][index], dtype=np.float32),
            np.asarray(data["hfss_actual_combined_weights_real_imag"][index], dtype=np.float32),
        )
        mask_hashes.append(mask_hash)
        nominal_hashes.append(nominal_hash)
        actual_hashes.append(actual_hash)
        margins = np.asarray(data["actual_margins"][index], dtype=float)
        metrics = np.asarray(data["actual_metrics"][index], dtype=float)
        manifest.append(
            {
                "candidate_index": local,
                "sample_index": int(data["sample_index"][index]),
                "target_hash": str(data["target_hashes"][index]),
                "k_value": int(data["k_values"][index]),
                "ratio": float(data["active_ratios_requested"][index]),
                "selection_role": role,
                "source_dataset": str(path.resolve()),
                "source_candidate_index": index,
                "eep_actual_strict_gate20": int(data["strict_gate20"][index]),
                "eep_actual_psll_db": float(metrics[0]),
                "eep_actual_nearest_iso_db": float(metrics[3]),
                "eep_actual_local_iso_db": float(metrics[4]),
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
    payload["frozen_mask_hashes"] = np.asarray(mask_hashes)
    payload["frozen_nominal_weight_hashes"] = np.asarray(nominal_hashes)
    payload["frozen_actual_weight_hashes"] = np.asarray(actual_hashes)
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **payload)
    write_csv(args.out_dir / "frozen_selection_manifest.csv", manifest)
    summary = {
        "candidate_count": count,
        "independent_scene_count": len(used_scenes),
        "k_counts": {str(k): sum(int(row["k_value"]) == k for row in manifest) for k in (2, 4, 6)},
        "ratio_counts": {
            str(ratio): int(
                sum(bool(np.isclose(float(row["ratio"]), ratio)) for row in manifest)
            )
            for ratio in (0.5, 0.6, 0.7, 0.8)
        },
        "role_counts": {
            role: sum(row["selection_role"] == role for row in manifest)
            for role in sorted({str(row["selection_role"]) for row in manifest})
        },
        "eep_strict_count": int(sum(row["eep_actual_strict_gate20"] for row in manifest)),
        "expected_hfss_case_count": int(sum(1 + int(row["k_value"]) for row in manifest)),
        "weights_frozen": True,
        "thresholds_frozen": True,
        "contains_ratio_1": False,
        "k1_included": False,
        "label_scope": "EEP/S256 shortlist; not HFSS full-wave",
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
