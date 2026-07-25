#!/usr/bin/env python3
"""Freeze adaptive selections from the second independent v0.9 EEP pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "hfss_outputs" / "v09_second_prospective_eep_candidates_20260726_run01"
DEFAULT_ADAPTIVE = ROOT / "hfss_outputs" / "v09_second_prospective_adaptive_20260726_run01"
DEFAULT_CRITIC = ROOT / "hfss_outputs" / "v09_physical_margin_critic_20260726_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_second_prospective_hfss_dataset_20260726_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--adaptive-dir", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite second prospective set: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.pool_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    selections = read_csv(args.adaptive_dir / "adaptive_scene_selections.csv")
    selected = np.asarray([int(row["selected_candidate_index"]) for row in selections], dtype=np.int64)
    if selected.size < 9 or np.unique(selected).size != selected.size:
        raise RuntimeError("Prospective selection must contain at least nine unique scenes")
    source_count = int(data["candidate_indices"].size)
    subset = {
        key: (
            value[selected]
            if value.ndim >= 1 and value.shape[0] == source_count
            else value
        )
        for key, value in data.items()
    }
    subset["candidate_index"] = np.arange(selected.size, dtype=np.int64)
    subset["candidate_indices"] = np.arange(selected.size, dtype=np.int64)
    subset["prospective_pool_candidate_indices"] = selected
    subset["adaptive_admitted"] = np.asarray(
        [int(row["admitted"]) for row in selections], dtype=np.int8
    )
    subset["adaptive_fallback_used"] = np.asarray(
        [int(row["fallback_used"]) for row in selections], dtype=np.int8
    )
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **subset)
    rows: list[dict[str, Any]] = []
    for local, (source_index, selection) in enumerate(zip(selected.tolist(), selections)):
        rows.append(
            {
                "candidate_index": local,
                "prospective_pool_candidate_index": source_index,
                "sample_index": int(data["sample_index"][source_index]),
                "target_hash": str(data["target_hashes"][source_index]),
                "k_value": int(data["k_values"][source_index]),
                "ratio": float(data["active_ratios_requested"][source_index]),
                "admitted": int(selection["admitted"]),
                "fallback_used": int(selection["fallback_used"]),
                "strict_probability": float(selection["strict_probability"]),
                "eep_actual_strict_gate": int(selection["selected_eep_actual_strict_gate"]),
                "oracle_has_strict_candidate": int(selection["oracle_has_strict_candidate"]),
                "oracle_minimum_ratio": selection["oracle_minimum_ratio"],
            }
        )
    write_csv(args.out_dir / "prospective_frozen_selections.csv", rows)
    checkpoints = sorted(args.critic_dir.glob("seed_*/best_checkpoint.pt"))
    freeze = {
        "stage": "v0.9-second-independent-prospective-hfss",
        "pool_dir": str(args.pool_dir.resolve()),
        "adaptive_dir": str(args.adaptive_dir.resolve()),
        "critic_dir": str(args.critic_dir.resolve()),
        "probability_threshold": 0.5,
        "candidate_count": int(selected.size),
        "independent_scene_count": int(np.unique(subset["sample_index"]).size),
        "admitted_count": int(np.sum(subset["adaptive_admitted"])),
        "target_hash_count": int(np.unique(subset["target_hashes"]).size),
        "contains_ratio_1": False,
        "contains_nominal_control": False,
        "post_hfss_tuning_allowed": False,
        "checkpoint_sha256": {path.parent.name: sha256(path) for path in checkpoints},
    }
    (args.out_dir / "prospective_freeze_manifest.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )
    summary = {
        **freeze,
        "expected_hfss_case_count": int(np.sum(1 + subset["k_values"])),
        "eep_strict_positive_count": int(np.sum(subset["strict_gate20"])),
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
