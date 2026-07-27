#!/usr/bin/env python3
"""Summarize the preregistered v1.6 robust-oracle Stage-B experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "hfss_outputs" / "v16_robust_drift_oracle_20260727_run01"
MARGINS = ("psll", "nearest_iso", "local20_iso", "mainlobe", "active_rl", "hardware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
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
    out = args.run_dir / "diagnostics"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite diagnostics: {out}")
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.run_dir / "pool" / "preregistered_protocol.json").read_text())
    initial = json.loads((args.run_dir / "initial" / "stage_b_summary.json").read_text())
    final = json.loads((args.run_dir / "final" / "stage_b_summary.json").read_text())
    post = json.loads((args.run_dir / "post_rescue" / "stage_b_summary.json").read_text())
    stages = (("initial", initial), ("common_weight_projection", final), ("mask_rescue", post))
    metrics = (
        "E1_new_scene_robust_oracle",
        "E2_overall_robust_oracle",
        "E2_k2_robust_oracle",
        "E2_k4_robust_oracle",
        "E2_k6_robust_oracle",
        "E3_stress_oracle",
    )
    improvement_rows = [
        {"stage": stage, **{metric: float(values[metric]) for metric in metrics}}
        for stage, values in stages
    ]
    write_csv(out / "oracle_improvement.csv", improvement_rows)

    gates = protocol["stage_b_acceptance"]
    gate_rows = [
        {
            "criterion": "E1 new scenes",
            "observed": post["E1_new_scene_robust_oracle"],
            "required": gates["E1_new_scene_robust_oracle_min"],
            "passed": int(post["E1_new_scene_robust_oracle"] >= gates["E1_new_scene_robust_oracle_min"]),
        },
        *[
            {
                "criterion": f"E2 K={k_value}",
                "observed": post[f"E2_k{k_value}_robust_oracle"],
                "required": gates[f"E2_k{k_value}_robust_oracle_min"],
                "passed": int(post[f"E2_k{k_value}_robust_oracle"] >= gates[f"E2_k{k_value}_robust_oracle_min"]),
            }
            for k_value in (2, 4, 6)
        ],
        {
            "criterion": "E2 overall",
            "observed": post["E2_overall_robust_oracle"],
            "required": gates["E2_overall_robust_oracle_min"],
            "passed": int(post["E2_overall_robust_oracle"] >= gates["E2_overall_robust_oracle_min"]),
        },
        {
            "criterion": "K6 positive ratio <= 0.7",
            "observed": int(post["E2_has_k6_positive_ratio_le_0_7"]),
            "required": 1,
            "passed": int(post["E2_has_k6_positive_ratio_le_0_7"]),
        },
    ]
    write_csv(out / "stage_b_gate_table.csv", gate_rows)

    scene_manifest = {int(row["sample_index"]): row for row in read_csv(args.run_dir / "pool" / "scene_manifest.csv")}
    scene_rows = read_csv(args.run_dir / "post_rescue" / "post_rescue_scene_oracle.csv")
    final_candidates = read_csv(args.run_dir / "final" / "robust_candidate_metrics.csv")
    rescue_candidates = read_csv(args.run_dir / "post_rescue" / "rescue_robust_candidate_metrics.csv")
    candidates = [*final_candidates, *rescue_candidates]
    failed_rows: list[dict[str, Any]] = []
    for scene in scene_rows:
        if scene["envelope"] != "E2" or int(scene["robust_oracle_pass"]) == 1:
            continue
        best = next(
            row
            for row in candidates
            if int(row["sample_index"]) == int(scene["sample_index"])
            and str(row["evaluation_index"]) == str(scene["best_evaluation_index"])
        )
        values = {name: float(best[f"E2_{name}_margin_db"]) for name in MARGINS}
        cause = min(values, key=values.get)
        metadata = scene_manifest[int(scene["sample_index"])]
        failed_rows.append(
            {
                "sample_index": int(scene["sample_index"]),
                "scene_origin": scene["scene_origin"],
                "k_value": int(scene["k_value"]),
                "max_target_theta_deg": float(metadata["max_target_theta_deg"]),
                "min_target_separation_deg": float(metadata["min_target_separation_deg"]),
                "best_ratio": float(best["ratio"]),
                "best_candidate_source": best["candidate_origin"],
                "root_cause": cause,
                "worst_margin_db": values[cause],
                **{f"{name}_margin_db": value for name, value in values.items()},
                **{f"{name}_worst_corner": best[f"E2_{name}_worst_corner"] for name in MARGINS},
            }
        )
    write_csv(out / "failed_scene_diagnostics.csv", failed_rows)

    ratio_rows = [
        row for row in scene_rows if row["envelope"] == "E2" and int(row["robust_oracle_pass"]) == 1
    ]
    ratio_counts = Counter(round(float(row["minimum_feasible_ratio"]), 1) for row in ratio_rows)
    write_csv(
        out / "minimum_ratio_distribution.csv",
        [
            {"minimum_ratio": ratio, "scene_count": count, "fraction_of_75": count / 75.0}
            for ratio, count in sorted(ratio_counts.items())
        ],
    )

    with np.load(args.run_dir / "pool" / "candidate_pool.npz", allow_pickle=False) as source:
        pool_sample = source["sample_index"]
        pool_ratio = source["ratio"]
        pool_hash = source["mask_hash"]
    with np.load(args.run_dir / "rescue" / "rescue_candidates.npz", allow_pickle=False) as source:
        rescue_sample = source["sample_index"]
        rescue_ratio = source["ratio"]
        rescue_hash = source["mask_hash"]
    budget_rows: list[dict[str, Any]] = []
    for sample in sorted(set(pool_sample.tolist())):
        for ratio in (0.5, 0.6, 0.7, 0.8):
            initial_hashes = set(pool_hash[(pool_sample == sample) & np.isclose(pool_ratio, ratio)].tolist())
            added_hashes = set(rescue_hash[(rescue_sample == sample) & np.isclose(rescue_ratio, ratio)].tolist())
            budget_rows.append(
                {
                    "sample_index": sample,
                    "ratio": ratio,
                    "initial_unique_masks": len(initial_hashes),
                    "rescue_unique_masks": len(added_hashes),
                    "total_unique_masks": len(initial_hashes | added_hashes),
                }
            )
    write_csv(out / "candidate_budget.csv", budget_rows)

    cause_counts = Counter(row["root_cause"] for row in failed_rows)
    summary = {
        "stage_b_gate_pass": bool(post["stage_b_gate_pass"]),
        "E2_oracle_initial": initial["E2_overall_robust_oracle"],
        "E2_oracle_after_common_weight_projection": final["E2_overall_robust_oracle"],
        "E2_oracle_after_mask_rescue": post["E2_overall_robust_oracle"],
        "E2_k2_post_rescue": post["E2_k2_robust_oracle"],
        "E2_k4_post_rescue": post["E2_k4_robust_oracle"],
        "E2_k6_post_rescue": post["E2_k6_robust_oracle"],
        "E1_new_post_rescue": post["E1_new_scene_robust_oracle"],
        "E3_post_rescue": post["E3_stress_oracle"],
        "failed_scene_count": len(failed_rows),
        "failed_scene_root_causes": dict(cause_counts),
        "candidate_budget_min": min(row["total_unique_masks"] for row in budget_rows),
        "candidate_budget_max": max(row["total_unique_masks"] for row in budget_rows),
        "critic_retraining_allowed": False,
        "hfss_smoke_allowed": False,
        "next_action": (
            "Keep E2 fixed. Repair K6 frequency-corner mainlobe robustness and active-RL "
            "with explicit multi-frequency combined-beam constraints before another oracle run."
        ),
    }
    (out / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "STAGE_B_REPORT.md").write_text(
        "\n".join(
            (
                "# v1.6 Stage-B Robust Oracle",
                "",
                "Evidence remains a 4x4-HFSS-calibrated 16x16 EEP/S256 proxy, not perturbed 16x16 HFSS.",
                "",
                "| Stage | E2 overall | K2 | K4 | K6 |",
                "|---|---:|---:|---:|---:|",
                f"| Initial | {initial['E2_overall_robust_oracle']:.2%} | {initial['E2_k2_robust_oracle']:.2%} | {initial['E2_k4_robust_oracle']:.2%} | {initial['E2_k6_robust_oracle']:.2%} |",
                f"| Common-weight projection | {final['E2_overall_robust_oracle']:.2%} | {final['E2_k2_robust_oracle']:.2%} | {final['E2_k4_robust_oracle']:.2%} | {final['E2_k6_robust_oracle']:.2%} |",
                f"| Mask rescue | {post['E2_overall_robust_oracle']:.2%} | {post['E2_k2_robust_oracle']:.2%} | {post['E2_k4_robust_oracle']:.2%} | {post['E2_k6_robust_oracle']:.2%} |",
                "",
                f"E1 new-scene oracle is {post['E1_new_scene_robust_oracle']:.2%}; E3 stress oracle is {post['E3_stress_oracle']:.2%}.",
                f"Stage B does not pass. Remaining best-candidate causes: {dict(cause_counts)}.",
                "Critic retraining and 16x16 HFSS smoke remain disabled.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
