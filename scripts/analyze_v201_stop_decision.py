#!/usr/bin/env python3
"""Audit the v20.1 K=2/K=4 gate and emit the preregistered stop decision."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V20 = ROOT / "hfss_outputs" / "v20_three_frequency_joint_search_20260729_run02"
DEFAULT_ALPHA = ROOT / "hfss_outputs" / "v201_dense_alpha_eep_20260729_run01"
DEFAULT_RESCUE = ROOT / "hfss_outputs" / "v201_k24_progressive_rescue_20260729_run01"
DEFAULT_OUT = DEFAULT_RESCUE

STATES = (
    "nominal_identity",
    "frequency_low_identity",
    "frequency_low_E2_source",
    "frequency_high_identity",
    "frequency_high_E2_source",
)
MARGINS = (
    "psll_margin_db",
    "nearest_iso_margin_db",
    "local20_iso_margin_db",
    "mainlobe_margin_db",
    "active_rl_margin_db",
    "hardware_margin_db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v20-dir", type=Path, default=DEFAULT_V20)
    parser.add_argument("--alpha-dir", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--rescue-dir", type=Path, default=DEFAULT_RESCUE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path, source: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["result_source"] = source
    return rows


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


def limiting_constraint(row: dict[str, Any]) -> tuple[str, str, float]:
    values: list[tuple[float, str, str]] = []
    for state in STATES:
        for margin in MARGINS:
            key = f"{state}_{margin}"
            if key in row and str(row[key]).strip() != "":
                values.append((float(row[key]), state, margin.removesuffix("_margin_db")))
    value, state, metric = min(values)
    return state, metric, value


def best_by_scene(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample = int(row["sample_index"])
        if sample not in result or float(row["robust_worst_margin_db"]) > float(
            result[sample]["robust_worst_margin_db"]
        ):
            result[sample] = row
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    v20_rows = read_csv(args.v20_dir / "full_refined_candidate_metrics.csv", "v20")
    alpha_rows = read_csv(args.alpha_dir / "dense_alpha_metrics.csv", "dense_alpha")
    rescue_rows = read_csv(args.rescue_dir / "progressive_path_metrics.csv", "pareto_rescue")
    baseline = best_by_scene(v20_rows)
    combined = best_by_scene([*v20_rows, *alpha_rows, *rescue_rows])
    scene_rows: list[dict[str, Any]] = []
    for sample in sorted(combined):
        row = combined[sample]
        k_value = int(float(row["k"]))
        if k_value not in (2, 4):
            continue
        state, metric, value = limiting_constraint(row)
        baseline_margin = float(baseline[sample]["robust_worst_margin_db"])
        scene_rows.append(
            {
                "sample_index": sample,
                "k": k_value,
                "strict_pass": int(row["all_corner_strict_pass"]),
                "reserve11_pass": int(
                    row.get("design_reserve11_pass", row.get("design_reserve_pass", 0))
                ),
                "best_ratio": float(row["ratio"]),
                "result_source": row["result_source"],
                "best_worst_margin_db": float(row["robust_worst_margin_db"]),
                "v20_best_worst_margin_db": baseline_margin,
                "v201_improvement_db": float(row["robust_worst_margin_db"]) - baseline_margin,
                "limiting_state": state,
                "limiting_metric": metric,
                "limiting_margin_db": value,
                "robust_pattern_margin_db": float(row["robust_pattern_margin_db"]),
                "robust_active_rl_margin_db": float(row["robust_active_rl_margin_db"]),
                "robust_hardware_margin_db": float(row["robust_hardware_margin_db"]),
                "generated_index": row.get("generated_index", ""),
                "selection_role": row.get("selection_role", ""),
                "alpha": row.get("alpha", row.get("weight_path_alpha", "")),
            }
        )
    write_csv(args.out_dir / "k24_stop_decision_audit.csv", scene_rows)

    strict_counts = {
        k: sum(row["strict_pass"] for row in scene_rows if row["k"] == k) for k in (2, 4)
    }
    total_counts = {k: sum(row["k"] == k for row in scene_rows) for k in (2, 4)}
    reserve_counts = {
        k: sum(row["reserve11_pass"] for row in scene_rows if row["k"] == k)
        for k in (2, 4)
    }
    failed = [row for row in scene_rows if not row["strict_pass"]]
    failure_metrics = Counter(str(row["limiting_metric"]) for row in failed)
    failure_states = Counter(str(row["limiting_state"]) for row in failed)
    gate = strict_counts[2] >= 6 and strict_counts[4] >= 5
    reserve_near_zero = sum(reserve_counts.values()) == 0
    decision = {
        "protocol": "v20.1-preregistered-stop-decision",
        "strict_counts_by_k": {str(k): strict_counts[k] for k in (2, 4)},
        "total_counts_by_k": {str(k): total_counts[k] for k in (2, 4)},
        "strict_rates_by_k": {
            str(k): strict_counts[k] / total_counts[k] for k in (2, 4)
        },
        "reserve11_counts_by_k": {str(k): reserve_counts[k] for k in (2, 4)},
        "failed_scene_count": len(failed),
        "failed_limiting_metric_counts": dict(failure_metrics),
        "failed_limiting_state_counts": dict(failure_states),
        "k24_stage_gate_pass": gate,
        "k6_execution_allowed": gate,
        "hfss_smoke_allowed": False,
        "critic_training_allowed": False,
        "thresholds_changed": False,
        "stop_algorithm_expansion": not gate,
        "next_action": (
            "run_k6_pareto_rescue" if gate else "broaden_10p04GHz_matching_bandwidth"
        ),
        "reason": (
            "K2/K4 preregistered gate passed"
            if gate
            else "K2/K4 Pareto reranking remained below 80% and 11 dB reserve set is empty"
        ),
        "reserve_set_empty": reserve_near_zero,
    }
    (args.out_dir / "stop_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
