#!/usr/bin/env python3
"""Audit the preregistered v1.6 K=6 multifrequency rescue experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "hfss_outputs" / "v16_k6_multifrequency_rescue_20260727_run01"
DEFAULT_PARENT = ROOT / "hfss_outputs" / "v16_robust_drift_oracle_20260727_run01"
MARGINS = ("psll", "nearest_iso", "local20_iso", "mainlobe", "active_rl", "hardware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--additional-parent-dir", type=Path, action="append", default=[])
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


def root_cause(row: dict[str, str]) -> tuple[str, float]:
    values = {name: float(row[f"E2_{name}_margin_db"]) for name in MARGINS}
    cause = min(values, key=values.get)
    return cause, values[cause]


def main() -> None:
    args = parse_args()
    out = args.run_dir / "diagnostics"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite diagnostics: {out}")
    out.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(
        (args.run_dir / "candidates" / "preregistered_protocol.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (args.run_dir / "evaluation" / "stage_b_summary.json").read_text(encoding="utf-8")
    )
    new_rows = read_csv(args.run_dir / "evaluation" / "candidate_metrics.csv")
    combined_scenes = read_csv(args.run_dir / "evaluation" / "combined_scene_oracle.csv")
    parent_scenes = read_csv(args.parent_dir / "post_rescue" / "post_rescue_scene_oracle.csv")
    parent_rows = [
        *read_csv(args.parent_dir / "final" / "robust_candidate_metrics.csv"),
        *read_csv(args.parent_dir / "post_rescue" / "rescue_robust_candidate_metrics.csv"),
    ]
    additional_parent_rows = [
        row
        for additional_dir in args.additional_parent_dir
        for row in read_csv(additional_dir / "evaluation" / "candidate_metrics.csv")
    ]
    all_rows = [*parent_rows, *additional_parent_rows, *new_rows]
    row_by_evaluation = {str(row["evaluation_index"]): row for row in all_rows}

    variant_rows: list[dict[str, Any]] = []
    for variant in sorted({row["variant"] for row in new_rows}):
        members = [row for row in new_rows if row["variant"] == variant]
        passed = [row for row in members if int(row["E2_strict_pass"]) == 1]
        analog = np.asarray([float(row["analog_E2_active_rl_floor_db"]) for row in members])
        implementation = np.asarray(
            [
                float(row.get("frozen_implementation_E2_active_rl_floor_db", row["analog_E2_active_rl_floor_db"]))
                for row in members
            ]
        )
        evaluated = np.asarray([10.0 + float(row["E2_active_rl_margin_db"]) for row in members])
        design = np.asarray([float(row["active_rl_design_min_db"]) for row in members])
        variant_rows.append(
            {
                "variant": variant,
                "candidate_count": len(members),
                "E2_pass_count": len(passed),
                "E2_candidate_pass_rate": len(passed) / len(members),
                "E2_rescued_scene_count": len({int(row["sample_index"]) for row in passed}),
                "mean_E2_worst_margin_db": float(
                    np.mean([float(row["E2_worst_margin_db"]) for row in members])
                ),
                "min_analog_active_rl_db": float(np.min(analog)),
                "min_frozen_implementation_active_rl_db": float(np.min(implementation)),
                "min_evaluated_active_rl_db": float(np.min(evaluated)),
                "mean_quantized_active_rl_delta_db": float(np.mean(evaluated - analog)),
                "p05_quantized_active_rl_delta_db": float(np.quantile(evaluated - analog, 0.05)),
                "mean_evaluation_minus_frozen_design_db": float(np.mean(evaluated - implementation)),
                "frozen_design_reserve_pass_rate": float(np.mean(implementation >= design)),
            }
        )
    write_csv(out / "variant_performance.csv", variant_rows)

    targets = set(int(value) for value in protocol["target_scene_policy"]["sample_indices"])
    parent_e2 = {
        int(row["sample_index"]): row
        for row in parent_scenes
        if row["envelope"] == "E2" and int(row["sample_index"]) in targets
    }
    combined_e2 = {
        int(row["sample_index"]): row
        for row in combined_scenes
        if row["envelope"] == "E2" and int(row["sample_index"]) in targets
    }
    scene_rows: list[dict[str, Any]] = []
    for sample in sorted(targets):
        before_scene = parent_e2[sample]
        after_scene = combined_e2[sample]
        before = row_by_evaluation[str(before_scene["best_evaluation_index"])]
        after = row_by_evaluation[str(after_scene["best_evaluation_index"])]
        before_cause, before_worst = root_cause(before)
        after_cause, after_worst = root_cause(after)
        scene_rows.append(
            {
                "sample_index": sample,
                "old_E2_pass": int(before_scene["robust_oracle_pass"]),
                "new_E2_pass": int(after_scene["robust_oracle_pass"]),
                "rescued": int(
                    int(before_scene["robust_oracle_pass"]) == 0
                    and int(after_scene["robust_oracle_pass"]) == 1
                ),
                "old_best_worst_margin_db": before_worst,
                "new_best_worst_margin_db": after_worst,
                "best_margin_change_db": after_worst - before_worst,
                "old_root_cause": before_cause,
                "new_root_cause": after_cause,
                "new_best_source": after["candidate_origin"],
                "new_best_ratio": float(after["ratio"]),
                "new_best_variant": after.get("variant", "parent"),
                **{f"new_{name}_margin_db": float(after[f"E2_{name}_margin_db"]) for name in MARGINS},
            }
        )
    write_csv(out / "target_scene_before_after.csv", scene_rows)

    remaining = [row for row in scene_rows if int(row["new_E2_pass"]) == 0]
    causes = Counter(row["new_root_cause"] for row in remaining)
    report = {
        **summary,
        "target_scene_rescue_count_verified": sum(int(row["rescued"]) for row in scene_rows),
        "remaining_target_scene_count": len(remaining),
        "remaining_target_root_causes": dict(causes),
        "best_variant_by_rescued_scenes": (
            max(variant_rows, key=lambda row: (row["E2_rescued_scene_count"], row["mean_E2_worst_margin_db"]))[
                "variant"
            ]
            if variant_rows
            else None
        ),
        "E2_and_thresholds_unchanged": True,
        "critic_or_HFSS_started": False,
    }
    (out / "diagnostic_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out / "K6_MULTIFREQUENCY_RESCUE_REPORT.md").write_text(
        "\n".join(
            (
                "# v1.6 K=6 Multifrequency Rescue",
                "",
                "E2 intensity, calibration realization, strict thresholds, masks, and matching network are frozen.",
                "Evidence remains a 4x4-HFSS-calibrated 16x16 EEP/S256 proxy, not perturbed 16x16 HFSS.",
                "",
                f"- New candidates: {summary['candidate_count']}",
                f"- Target K=6 scenes rescued: {report['target_scene_rescue_count_verified']}/12",
                f"- E2 K=6 robust oracle: {summary['E2_k6_robust_oracle']:.2%}",
                f"- E2 overall robust oracle: {summary['E2_overall_robust_oracle']:.2%}",
                f"- E1 new-scene robust oracle: {summary['E1_new_scene_robust_oracle']:.2%}",
                f"- Stage-B gate pass: {summary['stage_b_gate_pass']}",
                f"- Remaining target root causes: {dict(causes)}",
                "",
                "Critic retraining and HFSS smoke are governed only by the preregistered Stage-B gate.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
