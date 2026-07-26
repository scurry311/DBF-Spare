#!/usr/bin/env python3
"""Summarize active-RL-guided rescue candidates and scene-oracle changes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-scene-oracle", type=Path, required=True)
    parser.add_argument("--joint-scene-oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
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


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite rescue summary: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(args.candidate_manifest)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        family = (
            "active_rl_guided"
            if str(row["mask_family"]).startswith("active_rl_joint_")
            else "other_local"
        )
        grouped[family].append(row)
    family_rows: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        count = len(rows)
        strict = sum(int(row["actual_strict_gate20"]) for row in rows)
        active = sum(int(row["actual_active_rl_gate"]) for row in rows)
        family_rows.append(
            {
                "family_group": family,
                "candidate_count": count,
                "strict_count": strict,
                "strict_rate": strict / count,
                "active_rl_count": active,
                "active_rl_rate": active / count,
                "best_strict_violation_db": min(
                    float(row["strict_violation"]) for row in rows
                ),
            }
        )
    write_csv(args.out_dir / "candidate_family_summary.csv", family_rows)

    baseline = {
        int(row["sample_index"]): row for row in read_csv(args.baseline_scene_oracle)
    }
    joint = {int(row["sample_index"]): row for row in read_csv(args.joint_scene_oracle)}
    scene_rows: list[dict[str, Any]] = []
    for sample_index in sorted(baseline):
        before = baseline[sample_index]
        after = joint[sample_index]
        scene_rows.append(
            {
                "sample_index": sample_index,
                "k_value": int(after["k_value"]),
                "baseline_oracle": int(before["oracle_strict"]),
                "joint_oracle": int(after["oracle_strict"]),
                "rescued": int(
                    int(before["oracle_strict"]) == 0
                    and int(after["oracle_strict"]) == 1
                ),
                "joint_minimum_ratio": after["oracle_minimum_ratio"],
                "baseline_best_violation_db": float(
                    before["best_strict_violation_db"]
                ),
                "joint_best_violation_db": float(after["best_strict_violation_db"]),
            }
        )
    write_csv(args.out_dir / "scene_oracle_delta.csv", scene_rows)
    summary = {
        "candidate_count": len(candidates),
        "candidate_family_groups": family_rows,
        "scene_count": len(scene_rows),
        "baseline_oracle_count": sum(row["baseline_oracle"] for row in scene_rows),
        "joint_oracle_count": sum(row["joint_oracle"] for row in scene_rows),
        "rescued_scene_count": sum(row["rescued"] for row in scene_rows),
        "label_scope": "EEP/S256; not HFSS full-wave",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
