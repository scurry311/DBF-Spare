#!/usr/bin/env python3
"""Analyze ratio-paired and close-target task-optimization results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPARSE_RATIOS = (0.5, 0.6, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def finite_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(values.mean())


def summarize_group(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "case_count": int(len(group)),
            "scene_count": int(group["base_scene_id"].nunique()),
            "joint_pass_count": int(group["joint_gate_pass"].sum()),
            "joint_pass_rate": float(group["joint_gate_pass"].mean()),
            "af_pass_rate": float(group["af_gate_pass"].mean()),
            "rf_pass_rate": float(group["proxy_rf_gate_pass"].mean()),
            "mean_final_psll_db": finite_mean(group["final_psll_db"]),
            "mean_psll_delta_db": finite_mean(group["psll_delta_db"]),
            "mean_nearest_iso_db": finite_mean(group["final_nearest_iso_db"]),
            "mean_local_iso_db": finite_mean(group["final_local_iso_db"]),
            "mean_worst_active_rl_db": finite_mean(group["worst_active_rl_db"]),
            "mean_total_rl_db": finite_mean(group["total_rl_db"]),
            "mean_mainlobe_loss_db": finite_mean(group["mainlobe_loss_db"]),
            "mean_energy_proxy": finite_mean(group["final_energy_proxy"]),
            "null_overlap_risk_rate": float(group["regional_null_overlap_risk"].mean()),
        }
    )


def adaptive_scene_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scene_id, group in frame.groupby("base_scene_id", sort=True):
        group = group.sort_values("ratio_requested")
        sparse = group[group["ratio_requested"].isin(SPARSE_RATIOS)]
        feasible = sparse[sparse["joint_gate_pass"] == 1]
        control = group[np.isclose(group["ratio_requested"], 1.0)]
        selected = feasible.iloc[0] if not feasible.empty else None
        first = group.iloc[0]
        rows.append(
            {
                "base_scene_id": scene_id,
                "scene_type": first["scene_type"],
                "k": int(first["k"]),
                "min_target_separation_deg": float(first["min_target_separation_deg"]),
                "max_target_theta_deg": float(first["max_target_theta_deg"]),
                "large_scan": int(first["large_scan"]),
                "regional_null_overlap_risk": int(first["regional_null_overlap_risk"]),
                "sparse_feasible": int(selected is not None),
                "minimum_feasible_ratio": (
                    float(selected["ratio_requested"]) if selected is not None else np.nan
                ),
                "minimum_feasible_active_count": (
                    int(selected["active_count"]) if selected is not None else np.nan
                ),
                "selected_final_psll_db": (
                    float(selected["final_psll_db"]) if selected is not None else np.nan
                ),
                "selected_nearest_iso_db": (
                    float(selected["final_nearest_iso_db"]) if selected is not None else np.nan
                ),
                "selected_local_iso_db": (
                    float(selected["final_local_iso_db"]) if selected is not None else np.nan
                ),
                "selected_worst_active_rl_db": (
                    float(selected["worst_active_rl_db"]) if selected is not None else np.nan
                ),
                "ratio1_control_pass": int(
                    not control.empty and int(control.iloc[0]["joint_gate_pass"]) == 1
                ),
                "ratio1_control_psll_db": (
                    float(control.iloc[0]["final_psll_db"]) if not control.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    metrics = pd.read_csv(args.metrics)
    if manifest["sample_index"].duplicated().any() or metrics["sample_index"].duplicated().any():
        raise ValueError("sample_index must be unique in both inputs")

    frame = manifest.merge(
        metrics,
        on="sample_index",
        how="inner",
        validate="one_to_one",
        suffixes=("_manifest", "_metric"),
    )
    if len(frame) != len(manifest) or len(frame) != len(metrics):
        raise ValueError(
            f"paired merge incomplete: manifest={len(manifest)}, metrics={len(metrics)}, merged={len(frame)}"
        )

    for name in ("k", "ratio_requested", "active_count", "max_target_theta_deg", "large_scan"):
        metric_name = f"{name}_metric"
        manifest_name = f"{name}_manifest"
        if manifest_name in frame and metric_name in frame:
            frame[name] = frame[manifest_name]
        elif manifest_name in frame:
            frame[name] = frame[manifest_name]
        elif metric_name in frame:
            frame[name] = frame[metric_name]

    for name in ("joint_gate_pass", "af_gate_pass", "proxy_rf_gate_pass"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0).astype(int)
    frame["regional_null_overlap_risk"] = (
        (frame["k"] > 1) & (frame["min_target_separation_deg"] < 10.0)
    ).astype(int)
    frame["separation_eval_bin"] = pd.cut(
        frame["min_target_separation_deg"],
        bins=[-np.inf, 5.0, 7.5, 10.0, 15.0, 30.0, np.inf],
        labels=["<5", "5-7.5", "7.5-10", "10-15", "15-30", ">=30"],
        right=False,
    ).astype(str)

    group_keys = ["scene_type", "k", "ratio_requested", "large_scan", "separation_eval_bin"]
    grouped = frame.groupby(group_keys, dropna=False, observed=True).apply(
        summarize_group, include_groups=False
    ).reset_index()
    adaptive = adaptive_scene_rows(frame)

    adaptive_summary = adaptive.groupby(
        ["scene_type", "k", "large_scan"], dropna=False
    ).agg(
        scene_count=("base_scene_id", "count"),
        sparse_feasible_count=("sparse_feasible", "sum"),
        sparse_feasible_rate=("sparse_feasible", "mean"),
        mean_minimum_feasible_ratio=("minimum_feasible_ratio", "mean"),
        ratio1_control_pass_rate=("ratio1_control_pass", "mean"),
        null_overlap_risk_rate=("regional_null_overlap_risk", "mean"),
    ).reset_index()

    close_frame = frame[frame["scene_type"].str.contains("close", case=False, na=False)]
    existing_frame = frame[~frame["scene_type"].str.contains("close", case=False, na=False)]
    summary = {
        "case_count": int(len(frame)),
        "scene_count": int(frame["base_scene_id"].nunique()),
        "ratios": sorted(float(value) for value in frame["ratio_requested"].unique()),
        "overall_joint_pass_rate": float(frame["joint_gate_pass"].mean()),
        "existing_joint_pass_rate": finite_mean(existing_frame["joint_gate_pass"]),
        "close_5_10deg_joint_pass_rate": finite_mean(close_frame["joint_gate_pass"]),
        "close_5_10deg_scene_count": int(close_frame["base_scene_id"].nunique()),
        "regional_null_overlap_risk_case_count": int(frame["regional_null_overlap_risk"].sum()),
        "adaptive_sparse_feasible_scene_rate": float(adaptive["sparse_feasible"].mean()),
        "adaptive_ratio1_control_pass_rate": float(adaptive["ratio1_control_pass"].mean()),
        "minimum_feasible_ratio_distribution": {
            str(ratio): int((adaptive["minimum_feasible_ratio"] == ratio).sum())
            for ratio in SPARSE_RATIOS
        },
        "label_scope": "AF task-level regional LCMV/SOCP plus S256 local-kernel active-RL proxy; not HFSS full-wave",
        "hfss_training_labels_generated": False,
    }

    frame.to_csv(args.out_dir / "paired_case_metrics_enriched.csv", index=False)
    grouped.to_csv(args.out_dir / "paired_group_summary.csv", index=False)
    adaptive.to_csv(args.out_dir / "adaptive_min_ratio_by_scene.csv", index=False)
    adaptive_summary.to_csv(args.out_dir / "adaptive_min_ratio_summary.csv", index=False)
    (args.out_dir / "paired_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    report = [
        "# Paired Ratio and Close-Target Analysis",
        "",
        f"- Cases/scenes: {summary['case_count']}/{summary['scene_count']}",
        f"- Overall joint proxy-gate pass rate: {summary['overall_joint_pass_rate']:.2%}",
        f"- Existing-scene pass rate: {summary['existing_joint_pass_rate']:.2%}",
        f"- 5-10 deg close-target pass rate: {summary['close_5_10deg_joint_pass_rate']:.2%}",
        f"- Adaptive sparse feasible scene rate: {summary['adaptive_sparse_feasible_scene_rate']:.2%}",
        f"- Ratio=1 control pass rate: {summary['adaptive_ratio1_control_pass_rate']:.2%}",
        "",
        "These are AF/active-RL-proxy results, not HFSS full-wave labels.",
        "For target separations below 10 deg, fixed +/-5 deg regional-null windows overlap",
        "the desired-target neighborhood and should be replaced by separation-aware windows.",
    ]
    (args.out_dir / "PAIRED_SCENE_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
