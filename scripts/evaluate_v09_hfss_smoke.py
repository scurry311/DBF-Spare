#!/usr/bin/env python3
"""Evaluate the v0.9 HFSS smoke before opening the 50-100 case stage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "v09_hfss_smoke_dataset_20260726_run02"
DEFAULT_HFSS = ROOT / "hfss_outputs" / "v09_hfss_smoke_20260726_run02"


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
    analysis = json.loads((args.hfss_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    labels = read_csv(args.hfss_dir / "candidate_residual_labels.csv")
    selections = read_csv(args.dataset_dir / "smoke_selection.csv")
    selection_by_local = {int(row["candidate_index"]): row for row in selections}
    rows: list[dict[str, Any]] = []
    for label in labels:
        local = int(label["candidate_index"])
        selection = selection_by_local[local]
        hfss_strict = int(label["strict_engineering_gate"])
        predicted_positive = float(selection["mean_strict_probability"]) >= 0.5
        rows.append(
            {
                **selection,
                "hfss_gate15": int(label["gate15"]),
                "hfss_strict_gate20": int(label["strict_gate20"]),
                "hfss_mainlobe_gate": int(label["mainlobe_gate"]),
                "hfss_active_rl_gate": int(label["active_RL_gate"]),
                "hfss_strict_engineering_gate": hfss_strict,
                "hfss_psll_db": float(label["hfss_psll_db"]),
                "hfss_nearest_iso_db": float(label["hfss_nearest_iso_db"]),
                "hfss_local_iso_db": float(label["hfss_local_iso_db"]),
                "hfss_worst_active_rl_db": float(label["all_case_worst_active_rl_db"]),
                "predicted_positive": int(predicted_positive),
                "correct_strict_sign": int(predicted_positive == bool(hfss_strict)),
            }
        )
    write_csv(args.hfss_dir / "v09_smoke_decisions.csv", rows)
    hfss_positive = np.asarray([row["hfss_strict_engineering_gate"] for row in rows], dtype=bool)
    predicted = np.asarray([row["predicted_positive"] for row in rows], dtype=bool)
    k_value = np.asarray([int(row["k_value"]) for row in rows], dtype=int)
    tp = int(np.sum(hfss_positive & predicted))
    fp = int(np.sum(~hfss_positive & predicted))
    eep_positive = np.asarray([int(row["eep_strict_gate20"]) for row in rows], dtype=bool)
    acceptance = {
        "fullwave_complete": len(rows) == len(selections),
        "all_no_scale_reconstruction_pass": bool(analysis["all_no_scale_reconstruction_pass"]),
        "hfss_strict_positive_ge_5": int(np.sum(hfss_positive)) >= 5,
        "hfss_k6_strict_positive_ge_1": int(np.sum(hfss_positive & (k_value == 6))) >= 1,
        "eep_hfss_strict_agreement_ge_0_80": float(np.mean(eep_positive == hfss_positive)) >= 0.80,
        "predicted_positive_precision_ge_0_70": tp / max(tp + fp, 1) >= 0.70,
    }
    acceptance["open_50_100_fullwave"] = bool(all(acceptance.values()))
    summary = {
        "candidate_count": len(rows),
        "hfss_strict_positive_count": int(np.sum(hfss_positive)),
        "hfss_k6_strict_positive_count": int(np.sum(hfss_positive & (k_value == 6))),
        "eep_hfss_strict_agreement": float(np.mean(eep_positive == hfss_positive)),
        "predicted_positive_precision": tp / max(tp + fp, 1),
        "predicted_positive_recall": tp / max(int(np.sum(hfss_positive)), 1),
        "complex_nmse_max": analysis["complex_nmse_max"],
        "magnitude_rmse_db_max": analysis["magnitude_rmse_db_max"],
        "acceptance": acceptance,
    }
    (args.hfss_dir / "v09_smoke_acceptance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
