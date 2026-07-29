#!/usr/bin/env python3
"""Summarize v20 three-frequency search, failure intersections, and stage locks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH = ROOT / "hfss_outputs" / "v20_three_frequency_joint_search_20260729_run02"
DEFAULT_AUDIT = ROOT / "hfss_outputs" / "v20_three_frequency_active_rl_audit_20260729_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v20_three_frequency_stage_summary_20260729_run01"
CORNERS = (
    "nominal_identity",
    "frequency_low_identity",
    "frequency_low_E2_source",
    "frequency_high_identity",
    "frequency_high_E2_source",
)
MARGINS = ("psll", "nearest_iso", "local20_iso", "mainlobe", "active_rl", "hardware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
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
        raise FileExistsError(f"Refusing to overwrite v20 summary: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage = json.loads((args.search_dir / "stage_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((args.audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(args.search_dir / "full_refined_candidate_metrics.csv")
    scenes = read_csv(args.search_dir / "scene_oracle.csv")
    by_evaluation = {int(row["evaluation_index"]): row for row in rows}
    by_scene: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_scene[int(row["sample_index"])].append(row)

    failure_rows: list[dict[str, Any]] = []
    root_counts: Counter[str] = Counter()
    bridge_counts_by_k: Counter[int] = Counter()
    minimum_ratio_counts: Counter[str] = Counter()
    for scene in scenes:
        sample = int(scene["sample_index"])
        k_value = int(scene["k"])
        if scene["three_frequency_oracle_pass"] == "1":
            minimum_ratio_counts[f"{float(scene['minimum_strict_ratio']):.1f}"] += 1
            continue
        best = by_evaluation[int(scene["best_evaluation_index"])]
        values = [
            (float(best[f"{corner}_{margin}_margin_db"]), margin, corner)
            for corner in CORNERS
            for margin in MARGINS
        ]
        worst_value, root, root_corner = min(values)
        root_counts[root] += 1
        members = by_scene[sample]
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in members:
            if row["round"] != "v19_frozen":
                groups[row["mask_hash"]].append(row)
        bridge_masks = 0
        for group in groups.values():
            pattern = any(int(row["all_corner_pattern_pass"]) for row in group)
            active_rl = any(int(row["all_corner_active_rl_pass"]) for row in group)
            joint = any(
                int(row["all_corner_pattern_pass"])
                and int(row["all_corner_active_rl_pass"])
                and float(row["robust_hardware_margin_db"]) >= 0.0
                for row in group
            )
            bridge_masks += int(pattern and active_rl and not joint)
        bridge_counts_by_k[k_value] += bridge_masks
        failure_rows.append(
            {
                "sample_index": sample,
                "k": k_value,
                "best_ratio": float(scene["best_ratio"]),
                "best_worst_margin_db": float(scene["best_worst_margin_db"]),
                "root_constraint": root,
                "root_corner": root_corner,
                "root_margin_db": worst_value,
                "best_active_rl_floor_db": float(scene["best_active_rl_floor_db"]),
                "best_high_E2_active_rl_floor_db": float(scene["best_high_E2_active_rl_floor_db"]),
                "pattern_feasible_candidate_count": sum(
                    int(row["all_corner_pattern_pass"]) for row in members
                ),
                "active_rl_feasible_candidate_count": sum(
                    int(row["all_corner_active_rl_pass"]) for row in members
                ),
                "same_mask_pattern_rl_bridge_count": bridge_masks,
            }
        )
    write_csv(args.out_dir / "failed_scene_root_causes.csv", failure_rows)

    decision = {
        "protocol": "v20-three-frequency-stage-decision",
        "frozen_v19_strict_count": 2,
        "v20_strict_oracle_count": int(stage["three_frequency_strict_oracle_count"]),
        "v20_strict_oracle_rate": float(stage["three_frequency_strict_oracle_rate"]),
        "v20_reserve11_count": int(stage["three_frequency_reserve11_oracle_count"]),
        "strict_improvement_count": int(stage["three_frequency_strict_oracle_count"]) - 2,
        "minimum_strict_ratio_counts": dict(minimum_ratio_counts),
        "failed_scene_root_counts": dict(root_counts),
        "same_mask_bridge_counts_by_k": {str(k): bridge_counts_by_k[k] for k in (2, 4, 6)},
        "high_best_of_n_ge_10p5_rate_by_k": {
            k: float(stage["per_k"][k]["high_best_of_n_ge_10p5_rate"])
            for k in ("2", "4", "6")
        },
        "hardware_redesign_stop_condition_triggered": bool(
            stage["hardware_redesign_trigger_groups"]
        ),
        "stage_acceptance_pass": bool(stage["stage_acceptance_pass"]),
        "hfss_smoke_allowed": False,
        "critic_training_allowed": False,
        "next_action": (
            "Run a targeted same-mask joint-feasibility rescue on the recorded bridge masks: "
            "fine Pareto continuation plus augmented active-RL projection constrained by the "
            "three-frequency target equalities. Reassess K-stratified oracle and the 11 dB reserve "
            "before considering matching-network bandwidth changes."
        ),
    }
    (args.out_dir / "stage_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lock = {
        "hfss_smoke_allowed": False,
        "critic_training_allowed": False,
        "reason": "v20 strict oracle is below 18/20 and the 11 dB reserve oracle is 0/20",
        "thresholds_changed": False,
        "candidate_or_weight_reoptimization_after_HFSS_prohibited": True,
    }
    (args.out_dir / "downstream_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    report = [
        "# v20 Three-Frequency Stage Decision",
        "",
        "## Confirmed results",
        "",
        f"- Strict oracle improved from 2/20 to {decision['v20_strict_oracle_count']}/20.",
        f"- The 11 dB active-RL reserve oracle remains {decision['v20_reserve11_count']}/20.",
        f"- Minimum feasible ratio counts are {dict(minimum_ratio_counts)}; ratio 1.0 was not optimized.",
        f"- Failed-scene root constraints are {dict(root_counts)}.",
        f"- Same-mask pattern/RL bridge counts by K are {dict(bridge_counts_by_k)}.",
        f"- Below-11-dB worst-port classes in the audit are {audit['worst_port_class_counts_below_11db']}.",
        "",
        "## Decision",
        "",
        "- The 18/20 acceptance gate failed, so no HFSS smoke list is frozen and critic training remains locked.",
        "- The preregistered matching-redesign stop condition did not trigger: K=2 and K=4 each reached a 100% high-corner best-of-N rate at or above 10.5 dB.",
        "- The next step is a targeted same-mask joint-feasibility rescue, not more random masks, HFSS labels, or critic training.",
        "- If that rescue cannot create 11 dB reserve candidates, the 10.04 GHz matching bandwidth must then be broadened before HFSS.",
    ]
    (args.out_dir / "STAGE_DECISION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
