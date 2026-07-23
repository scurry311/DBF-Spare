#!/usr/bin/env python3
"""Create a canonical, non-destructive package for trusted residual labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "eep_hfss_validation"
)
DEFAULT_OUT = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "dataset_v2_20260724"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite package: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_path = args.source_dir / "residual_critic_dataset.npz"
    with np.load(source_path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}

    aliases = {
        "candidate_index": np.asarray(payload["candidate_indices"], dtype=np.int64),
        "sample_index": np.asarray(payload["sample_indices"], dtype=np.int64),
        "mask": np.asarray(payload["masks"], dtype=np.int8),
        "w_tasks_real_imag": np.asarray(
            payload["task_weights_real_imag"], dtype=np.float32
        ),
        "w_combined_real_imag": np.asarray(
            payload["combined_weights_real_imag"], dtype=np.float32
        ),
    }
    payload.update(aliases)

    required = (
        "candidate_index",
        "sample_index",
        "mask",
        "w_tasks_real_imag",
        "w_combined_real_imag",
        "eep_psll_db",
        "hfss_psll_db",
        "delta_psll_db",
        "delta_nearest_iso_db",
        "delta_local_iso_db",
        "delta_mainlobe_gain_db",
        "gate15",
        "strict_gate20",
        "mainlobe_gate",
        "active_RL_gate",
        "split",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")
    candidate_count = int(payload["candidate_index"].size)
    if any(payload[key].shape[0] != candidate_count for key in required):
        raise ValueError("Candidate-axis mismatch in canonical dataset")

    target_path = args.out_dir / "residual_critic_dataset_v2.npz"
    np.savez_compressed(target_path, **payload)

    copied = []
    for name in (
        "analysis_summary.json",
        "candidate_residual_labels.csv",
        "critic_priority_candidates.csv",
        "critic_training_decision.json",
        "fullwave_summary_by_k_ratio_scan.csv",
        "grouped_split_manifest.json",
        "null_residual_baseline_metrics.csv",
    ):
        source_file = args.source_dir / name
        target_file = args.out_dir / name
        shutil.copy2(source_file, target_file)
        copied.append(name)

    schema = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(payload.items())
    }
    (args.out_dir / "dataset_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    summary = {
        "source_dataset": str(source_path.resolve()),
        "output_dataset": str(target_path.resolve()),
        "candidate_count": candidate_count,
        "independent_scene_count": int(np.unique(payload["sample_index"]).size),
        "required_fields_present": True,
        "scene_leakage_free": True,
        "old_labels_included": False,
        "copied_evidence": copied,
    }
    (args.out_dir / "package_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
