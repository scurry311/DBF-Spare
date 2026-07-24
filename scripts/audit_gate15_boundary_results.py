#!/usr/bin/env python3
"""Audit dedicated PSLL/nearest/local gate15 boundary HFSS results."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "gate15_boundary_scenes_20260725_run01"
DEFAULT_HFSS = ROOT / "hfss_outputs" / "gate15_boundary_scenes_hfss_20260725_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hfss-dir", type=Path, default=DEFAULT_HFSS)
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


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def isolated_failure(boundary_type: str, row: dict[str, str]) -> bool:
    psll = f(row, "hfss_psll_db")
    nearest = f(row, "hfss_nearest_iso_db")
    local = f(row, "hfss_local_iso_db")
    if boundary_type == "psll":
        return 0.0 < psll <= 1.0 and nearest >= 25.0 and local >= 15.0
    if boundary_type == "nearest":
        return psll <= 0.0 and 23.5 <= nearest < 25.0 and local >= 15.0
    if boundary_type == "local":
        return psll <= 0.0 and nearest >= 25.0 and 13.0 <= local < 15.0
    raise ValueError(f"Unknown boundary type: {boundary_type}")


def boundary_margin(boundary_type: str, row: dict[str, str]) -> float:
    if boundary_type == "psll":
        return -f(row, "hfss_psll_db")
    if boundary_type == "nearest":
        return f(row, "hfss_nearest_iso_db") - 25.0
    if boundary_type == "local":
        return f(row, "hfss_local_iso_db") - 15.0
    raise ValueError(f"Unknown boundary type: {boundary_type}")


def main() -> None:
    args = parse_args()
    labels = read_csv(args.hfss_dir / "candidate_residual_labels.csv")
    with np.load(args.dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        dataset = {key: source[key] for key in source.files}
    candidate_count = int(dataset["candidate_indices"].size)
    if candidate_count != len(labels):
        raise ValueError("Dataset and HFSS label counts do not match")
    labels_by_candidate = {int(row["candidate_index"]): row for row in labels}
    audit_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in range(candidate_count):
        label = labels_by_candidate[candidate]
        boundary_type = str(dataset["boundary_type"][candidate])
        side = str(dataset["boundary_side"][candidate])
        row = {
            "candidate_index": candidate,
            "sample_index": int(dataset["sample_indices"][candidate]),
            "scene_id": str(dataset["scene_ids"][candidate]),
            "boundary_type": boundary_type,
            "boundary_side": side,
            "k": int(dataset["k_values"][candidate]),
            "active_ratio": float(dataset["active_ratios_actual"][candidate]),
            "large_scan": int(dataset["large_scan"][candidate]),
            "hfss_psll_db": f(label, "hfss_psll_db"),
            "hfss_nearest_iso_db": f(label, "hfss_nearest_iso_db"),
            "hfss_local_iso_db": f(label, "hfss_local_iso_db"),
            "hfss_mainlobe_gain_db": f(label, "hfss_mainlobe_gain_db"),
            "hfss_pointing_error_deg": f(label, "hfss_pointing_error_deg"),
            "gate15": int(label["gate15"]),
            "strict_gate20": int(label["strict_gate20"]),
            "mainlobe_gate": int(label["mainlobe_gate"]),
            "active_RL_gate": int(label["active_RL_gate"]),
            "boundary_margin_db": boundary_margin(boundary_type, label),
            "isolated_target_failure": int(
                side == "outside" and isolated_failure(boundary_type, label)
            ),
            "fullwave_complete": int(label["fullwave_complete"]),
        }
        audit_rows.append(row)
        grouped[(boundary_type, side)].append(row)
    write_csv(args.hfss_dir / "gate15_boundary_hfss_audit.csv", audit_rows)

    group_rows: list[dict[str, Any]] = []
    for boundary_type in ("psll", "nearest", "local"):
        for side in ("control", "inside", "outside"):
            rows = grouped[(boundary_type, side)]
            if not rows:
                continue
            margins = np.asarray([row["boundary_margin_db"] for row in rows], dtype=float)
            group_rows.append(
                {
                    "boundary_type": boundary_type,
                    "boundary_side": side,
                    "candidate_count": len(rows),
                    "independent_scene_count": len({row["sample_index"] for row in rows}),
                    "gate15_pass_count": sum(row["gate15"] for row in rows),
                    "gate15_pass_rate": float(np.mean([row["gate15"] for row in rows])),
                    "mainlobe_pass_count": sum(row["mainlobe_gate"] for row in rows),
                    "isolated_target_failure_count": sum(
                        row["isolated_target_failure"] for row in rows
                    ),
                    "boundary_margin_mean_db": float(np.mean(margins)),
                    "boundary_margin_min_db": float(np.min(margins)),
                    "boundary_margin_max_db": float(np.max(margins)),
                    "psll_mean_db": float(np.mean([row["hfss_psll_db"] for row in rows])),
                    "nearest_iso_mean_db": float(
                        np.mean([row["hfss_nearest_iso_db"] for row in rows])
                    ),
                    "local_iso_mean_db": float(
                        np.mean([row["hfss_local_iso_db"] for row in rows])
                    ),
                }
            )
    write_csv(args.hfss_dir / "gate15_boundary_hfss_summary_by_type_side.csv", group_rows)

    inside = [row for row in audit_rows if row["boundary_side"] == "inside"]
    outside = [row for row in audit_rows if row["boundary_side"] == "outside"]
    analysis = json.loads((args.hfss_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(audit_rows),
        "independent_scene_count": len({row["sample_index"] for row in audit_rows}),
        "boundary_scene_counts": {
            name: len(
                {
                    row["sample_index"]
                    for row in audit_rows
                    if row["boundary_type"] == name
                }
            )
            for name in ("psll", "nearest", "local")
        },
        "inside_gate15_pass_count": sum(row["gate15"] for row in inside),
        "outside_gate15_fail_count": sum(1 - row["gate15"] for row in outside),
        "outside_isolated_target_failure_count": sum(
            row["isolated_target_failure"] for row in outside
        ),
        "mainlobe_failure_count": sum(1 - row["mainlobe_gate"] for row in audit_rows),
        "fullwave_incomplete_count": sum(1 - row["fullwave_complete"] for row in audit_rows),
        "all_no_scale_reconstruction_pass": bool(
            analysis["all_no_scale_reconstruction_pass"]
        ),
        "complex_nmse_max": float(analysis["complex_nmse_max"]),
        "magnitude_rmse_db_max": float(analysis["magnitude_rmse_db_max"]),
    }
    summary["boundary_dataset_pass"] = bool(
        summary["candidate_count"] == 90
        and summary["independent_scene_count"] == 30
        and all(value == 10 for value in summary["boundary_scene_counts"].values())
        and summary["inside_gate15_pass_count"] == 30
        and summary["outside_gate15_fail_count"] == 30
        and summary["outside_isolated_target_failure_count"] == 30
        and summary["mainlobe_failure_count"] == 0
        and summary["fullwave_incomplete_count"] == 0
        and summary["all_no_scale_reconstruction_pass"]
    )
    (args.hfss_dir / "gate15_boundary_hfss_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
