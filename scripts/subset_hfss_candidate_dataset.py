#!/usr/bin/env python3
"""Create a non-destructive candidate-axis subset of an HFSS dataset package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--indices", required=True, help="Comma-separated candidate indices")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite subset: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(
        [int(part.strip()) for part in args.indices.split(",") if part.strip()],
        dtype=np.int64,
    )
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("Candidate indices must be non-empty and unique")
    with np.load(args.source_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    candidate_count = int(np.asarray(payload["candidate_indices"]).size)
    if np.any(indices < 0) or np.any(indices >= candidate_count):
        raise IndexError("Candidate index outside source dataset")
    subset = {
        key: (value[indices] if value.ndim >= 1 and value.shape[0] == candidate_count else value)
        for key, value in payload.items()
    }
    local = np.arange(indices.size, dtype=np.int64)
    subset["candidate_index"] = local
    subset["candidate_indices"] = local
    subset["subset_source_candidate_indices"] = indices
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **subset)
    expected_cases = int(np.sum(1 + np.asarray(subset["k_values"], dtype=int)))
    summary = {
        "source_dir": str(args.source_dir.resolve()),
        "source_candidate_count": candidate_count,
        "selected_source_candidate_indices": indices.tolist(),
        "candidate_count": int(indices.size),
        "expected_hfss_case_count": expected_cases,
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
