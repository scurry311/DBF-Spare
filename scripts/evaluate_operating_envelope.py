#!/usr/bin/env python3
"""Evaluate a pre-registered K-specific scan/separation operating envelope."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-oracle", type=Path, required=True)
    parser.add_argument(
        "--additional-scene-oracle", type=Path, action="append", default=[]
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k2-max-scan", type=float, default=50.0)
    parser.add_argument("--k2-min-separation", type=float, default=16.0)
    parser.add_argument("--k4-max-scan", type=float, default=48.0)
    parser.add_argument("--k4-min-separation", type=float, default=16.0)
    parser.add_argument("--k6-max-scan", type=float, default=58.0)
    parser.add_argument("--k6-min-separation", type=float, default=13.0)
    parser.add_argument(
        "--required-k-values",
        default="2,6",
        help="Comma-separated K groups required to pass; defaults preserve v1.2.",
    )
    parser.add_argument("--oracle-target", type=float, default=0.90)
    parser.add_argument("--minimum-scenes-per-k", type=int, default=10)
    parser.add_argument("--validation-mode", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite envelope audit: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    limits = {
        2: (float(args.k2_max_scan), float(args.k2_min_separation)),
        4: (float(args.k4_max_scan), float(args.k4_min_separation)),
        6: (float(args.k6_max_scan), float(args.k6_min_separation)),
    }
    required_k_values = tuple(
        int(value.strip())
        for value in str(args.required_k_values).split(",")
        if value.strip()
    )
    if not required_k_values or any(value not in limits for value in required_k_values):
        raise ValueError("required-k-values must be a non-empty subset of 2,4,6")
    annotated: list[dict[str, Any]] = []
    seen_samples: set[int] = set()
    seen_hashes: set[str] = set()
    oracle_paths = [args.scene_oracle, *args.additional_scene_oracle]
    for oracle_path in oracle_paths:
        for source in read_csv(oracle_path):
            sample_index = int(source["sample_index"])
            target_digest = str(source["target_hash"])
            if sample_index in seen_samples or target_digest in seen_hashes:
                raise RuntimeError(
                    f"Duplicate scene in envelope inputs: {sample_index}/{target_digest}"
                )
            seen_samples.add(sample_index)
            seen_hashes.add(target_digest)
            k_value = int(source["k_value"])
            max_scan, min_separation = limits.get(k_value, (-np.inf, np.inf))
            inside = bool(
                float(source["max_target_theta_deg"]) <= max_scan
                and float(source["min_target_separation_deg"]) >= min_separation
            )
            annotated.append(
                {
                    **source,
                    "oracle_source": str(oracle_path.resolve()),
                    "envelope_max_scan_deg": max_scan,
                    "envelope_min_separation_deg": min_separation,
                    "inside_envelope": int(inside),
                }
            )
    write_csv(args.out_dir / "envelope_scene_membership.csv", annotated)
    write_csv(
        args.out_dir / "supported_scene_list.csv",
        [row for row in annotated if int(row["inside_envelope"]) == 1],
    )
    write_csv(
        args.out_dir / "out_of_envelope_stress_scene_list.csv",
        [row for row in annotated if int(row["inside_envelope"]) == 0],
    )
    groups: list[dict[str, Any]] = []
    for k_value in required_k_values:
        members = [
            row for row in annotated
            if int(row["k_value"]) == k_value and int(row["inside_envelope"]) == 1
        ]
        passed = sum(int(row["oracle_strict"]) for row in members)
        rate = float(passed / len(members)) if members else float("nan")
        groups.append(
            {
                "k_value": k_value,
                "max_scan_deg": limits[k_value][0],
                "min_separation_deg": limits[k_value][1],
                "inside_scene_count": len(members),
                "oracle_scene_count": passed,
                "oracle_rate": rate,
                "minimum_scene_count_pass": int(
                    len(members) >= int(args.minimum_scenes_per_k)
                ),
                "oracle_target_pass": int(rate >= float(args.oracle_target)),
            }
        )
    write_csv(args.out_dir / "envelope_groups.csv", groups)
    source_groups: list[dict[str, Any]] = []
    for oracle_source in sorted({str(row["oracle_source"]) for row in annotated}):
        for k_value in required_k_values:
            members = [
                row
                for row in annotated
                if str(row["oracle_source"]) == oracle_source
                and int(row["k_value"]) == k_value
                and int(row["inside_envelope"]) == 1
            ]
            if not members:
                continue
            passed = sum(int(row["oracle_strict"]) for row in members)
            source_groups.append(
                {
                    "oracle_source": oracle_source,
                    "k_value": k_value,
                    "inside_scene_count": len(members),
                    "oracle_scene_count": passed,
                    "oracle_rate": float(passed / len(members)),
                }
            )
    write_csv(args.out_dir / "envelope_source_groups.csv", source_groups)
    all_groups_pass = bool(
        all(
            row["minimum_scene_count_pass"] == 1
            and row["oracle_target_pass"] == 1
            for row in groups
        )
    )
    summary = {
        "scene_oracles": [str(path.resolve()) for path in oracle_paths],
        "validation_mode": bool(args.validation_mode),
        "pre_registered_limits": {
            "K2": {"max_scan_deg": limits[2][0], "min_separation_deg": limits[2][1]},
            "K4": {"max_scan_deg": limits[4][0], "min_separation_deg": limits[4][1]},
            "K6": {"max_scan_deg": limits[6][0], "min_separation_deg": limits[6][1]},
        },
        "required_k_values": list(required_k_values),
        "oracle_target": float(args.oracle_target),
        "minimum_scenes_per_k": int(args.minimum_scenes_per_k),
        "groups": groups,
        "operating_envelope_pass": all_groups_pass,
        "status": (
            "validated" if args.validation_mode and all_groups_pass
            else "validation_failed" if args.validation_mode
            else "provisional_development_pass" if all_groups_pass
            else "development_insufficient"
        ),
    }
    (args.out_dir / "envelope_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
