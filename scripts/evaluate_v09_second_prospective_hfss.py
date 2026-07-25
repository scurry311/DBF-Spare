#!/usr/bin/env python3
"""Evaluate the frozen second independent v0.9 prospective HFSS run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "v09_second_prospective_hfss_dataset_20260726_run01"
DEFAULT_HFSS = ROOT / "hfss_outputs" / "v09_second_prospective_hfss_20260726_run01"


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


def main() -> None:
    args = parse_args()
    frozen = read_csv(args.dataset_dir / "prospective_frozen_selections.csv")
    labels = read_csv(args.hfss_dir / "candidate_residual_labels.csv")
    analysis = json.loads((args.hfss_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    if len(frozen) != len(labels):
        raise RuntimeError("Frozen selections and HFSS labels differ in length")
    rows: list[dict[str, Any]] = []
    for selection, label in zip(frozen, labels):
        rows.append(
            {
                **selection,
                "hfss_gate15": int(label["gate15"]),
                "hfss_strict_gate20": int(label["strict_gate20"]),
                "hfss_mainlobe_gate": int(label["mainlobe_gate"]),
                "hfss_active_rl_gate": int(label["active_RL_gate"]),
                "hfss_strict_engineering_gate": int(label["strict_engineering_gate"]),
                "hfss_psll_db": float(label["hfss_psll_db"]),
                "hfss_nearest_iso_db": float(label["hfss_nearest_iso_db"]),
                "hfss_local_iso_db": float(label["hfss_local_iso_db"]),
                "hfss_worst_active_rl_db": float(label["all_case_worst_active_rl_db"]),
            }
        )
    write_csv(args.hfss_dir / "prospective_selection_evaluation.csv", rows)
    admitted = [row for row in rows if int(row["admitted"]) == 1]
    acceptance = {
        "fullwave_complete": int(analysis["complete_case_count"]) == int(analysis["expected_case_count"]),
        "all_no_scale_reconstruction_pass": bool(analysis["all_no_scale_reconstruction_pass"]),
        "admitted_hfss_precision_ge_0_80": float(
            np.mean([row["hfss_strict_engineering_gate"] for row in admitted])
        ) >= 0.80 if admitted else False,
        "k6_admitted_positive_exists": any(
            int(row["admitted"]) == 1
            and int(row["k_value"]) == 6
            and int(row["hfss_strict_engineering_gate"]) == 1
            for row in rows
        ),
    }
    acceptance["prospective_pass"] = bool(all(acceptance.values()))
    summary = {
        "candidate_count": len(rows),
        "independent_scene_count": len({int(row["sample_index"]) for row in rows}),
        "admission_rate": len(admitted) / max(len(rows), 1),
        "admitted_hfss_precision": float(
            np.mean([row["hfss_strict_engineering_gate"] for row in admitted])
        ) if admitted else float("nan"),
        "all_scene_hfss_strict_rate": float(
            np.mean([row["hfss_strict_engineering_gate"] for row in rows])
        ),
        "mean_selected_ratio": float(np.mean([float(row["ratio"]) for row in rows])),
        "complex_nmse_max": analysis["complex_nmse_max"],
        "magnitude_rmse_db_max": analysis["magnitude_rmse_db_max"],
        "acceptance": acceptance,
        "post_hfss_tuning_performed": False,
    }
    (args.hfss_dir / "second_prospective_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
