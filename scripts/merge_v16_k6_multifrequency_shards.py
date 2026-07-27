#!/usr/bin/env python3
"""Merge deterministic scene shards for the v1.6 K=6 rescue run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "hfss_outputs" / "v16_k6_multifrequency_rescue_20260727_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
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


def main() -> None:
    args = parse_args()
    out = args.out_dir / "candidates"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite candidates: {out}")
    out.mkdir(parents=True, exist_ok=True)

    protocols = []
    summaries = []
    payloads: list[dict[str, np.ndarray]] = []
    rows: list[dict[str, Any]] = []
    samples_seen: set[int] = set()
    for shard_index, shard_dir in enumerate(args.shard_dir):
        source = shard_dir / "candidates"
        protocol = json.loads((source / "preregistered_protocol.json").read_text(encoding="utf-8"))
        summary = json.loads((source / "generate_summary.json").read_text(encoding="utf-8"))
        with np.load(source / "candidate_commands.npz", allow_pickle=False) as data:
            payload = {name: data[name].copy() for name in data.files}
        shard_samples = set(int(value) for value in np.unique(payload["sample_index"]))
        overlap = samples_seen & shard_samples
        if overlap:
            raise ValueError(f"Scene overlap across shards: {sorted(overlap)}")
        samples_seen |= shard_samples
        protocols.append(protocol)
        summaries.append(summary)
        payloads.append(payload)
        for row in read_csv(source / "candidate_manifest.csv"):
            row["shard_index"] = shard_index
            row["shard_source"] = str(shard_dir.resolve())
            rows.append(row)

    canonical = json.dumps(protocols[0], sort_keys=True)
    if any(json.dumps(protocol, sort_keys=True) != canonical for protocol in protocols[1:]):
        raise ValueError("Shard preregistration protocols differ")
    keys = set(payloads[0])
    if any(set(payload) != keys for payload in payloads[1:]):
        raise ValueError("Shard candidate schemas differ")
    merged = {name: np.concatenate([payload[name] for payload in payloads], axis=0) for name in keys}
    count = len(merged["sample_index"])
    merged["candidate_index"] = np.arange(count, dtype=np.int64)
    for index, row in enumerate(rows):
        row["candidate_index"] = index

    expected = int(protocols[0]["target_scene_policy"]["candidate_count_expected"])
    expected_scenes = set(
        int(value) for value in protocols[0]["target_scene_policy"]["sample_indices"]
    )
    if count != expected or samples_seen != expected_scenes or len(rows) != count:
        raise ValueError(
            f"Incomplete merge: candidates={count}/{expected}, "
            f"scenes={len(samples_seen)}/{len(expected_scenes)}, rows={len(rows)}"
        )

    np.savez_compressed(out / "candidate_commands.npz", **merged)
    write_csv(out / "candidate_manifest.csv", rows)
    (out / "preregistered_protocol.json").write_text(
        json.dumps(protocols[0], indent=2), encoding="utf-8"
    )
    result = {
        "candidate_count": count,
        "target_scene_count": len(samples_seen),
        "target_sample_indices": sorted(samples_seen),
        "shard_count": len(payloads),
        "shard_summaries": summaries,
        "deterministic_scene_disjoint_merge": True,
        "E2_or_gates_changed": False,
    }
    (out / "generate_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
