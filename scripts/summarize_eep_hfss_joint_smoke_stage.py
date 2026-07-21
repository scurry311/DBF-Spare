#!/usr/bin/env python3
"""Summarize the grounded-patch EEP/HFSS resource-smoke stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-manifest", required=True, type=Path)
    parser.add_argument("--hfss-dir", required=True, type=Path)
    parser.add_argument("--array8-summary", required=True, type=Path)
    parser.add_argument("--array16-summary", required=True, type=Path)
    parser.add_argument("--eep-summary", required=True, type=Path)
    parser.add_argument("--superposition-summary", required=True, type=Path)
    parser.add_argument("--paired-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite stage summary: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    manifest = pd.read_csv(args.smoke_manifest)
    metrics = pd.read_csv(args.hfss_dir / "hfss_task_fullwave_metrics.csv")
    gates = pd.read_csv(args.hfss_dir / "hfss_fullwave_gate_results.csv")
    frame = manifest.merge(metrics, on="sample_index", suffixes=("_af", "_hfss"), validate="one_to_one")
    frame = frame.merge(
        gates[
            [
                "sample_index",
                "fullwave_gate_pass",
                "fullwave_gate_pass_strict_local",
                "fail_reasons",
                "strict_fail_reasons",
            ]
        ],
        on="sample_index",
        validate="one_to_one",
    )
    frame["delta_psll_hfss_minus_af_db"] = (
        frame["combined_psll_to_weakest_peak_db"] - frame["af_psll_db"]
    )
    frame["delta_nearest_iso_hfss_minus_af_db"] = (
        frame["isolation_worst_nearest_db"] - frame["af_nearest_iso_db"]
    )
    frame["delta_local_iso_hfss_minus_af_db"] = (
        frame["isolation_worst_local_db"] - frame["af_local_iso_db"]
    )

    def group_row(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "case_count": int(len(group)),
                "base_gate_pass_count": int(group["fullwave_gate_pass"].sum()),
                "base_gate_pass_rate": float(group["fullwave_gate_pass"].mean()),
                "strict_gate_pass_count": int(group["fullwave_gate_pass_strict_local"].sum()),
                "strict_gate_pass_rate": float(group["fullwave_gate_pass_strict_local"].mean()),
                "hfss_smoke_psll_mean_db": float(group["combined_psll_to_weakest_peak_db"].mean()),
                "hfss_smoke_target_spread_mean_db": float(group["combined_target_spread_db"].mean()),
                "hfss_smoke_nearest_iso_mean_db": float(group["isolation_worst_nearest_db"].mean()),
                "hfss_smoke_local_iso_mean_db": float(group["isolation_worst_local_db"].mean()),
                "delta_psll_hfss_minus_af_mean_db": float(
                    group["delta_psll_hfss_minus_af_db"].mean()
                ),
                "delta_nearest_iso_hfss_minus_af_mean_db": float(
                    group["delta_nearest_iso_hfss_minus_af_db"].mean()
                ),
                "delta_local_iso_hfss_minus_af_mean_db": float(
                    group["delta_local_iso_hfss_minus_af_db"].mean()
                ),
            }
        )

    k_summary = frame.groupby("k_af", dropna=False).apply(
        group_row, include_groups=False
    ).reset_index().rename(columns={"k_af": "k"})
    ratio_summary = frame.groupby("ratio_requested", dropna=False).apply(
        group_row, include_groups=False
    ).reset_index()
    k_summary.insert(0, "group_type", "k")
    ratio_summary.insert(0, "group_type", "ratio")
    k_summary["group_value"] = k_summary["k"]
    ratio_summary["group_value"] = ratio_summary["ratio_requested"]
    group_columns = ["group_type", "group_value"] + [
        column
        for column in k_summary.columns
        if column not in {"group_type", "group_value", "k"}
    ]
    group_summary = pd.concat(
        [k_summary[group_columns], ratio_summary[group_columns]], ignore_index=True
    )

    array8 = load_json(args.array8_summary)["metric"]
    array16 = load_json(args.array16_summary)["metric"]
    eep = load_json(args.eep_summary)
    superposition = load_json(args.superposition_summary)
    paired = load_json(args.paired_summary)
    fullwave_complete = bool(
        len(frame) == 26
        and int(frame["combined_complete"].sum()) == len(frame)
        and int(frame["isolation_complete"].sum()) == len(frame)
    )
    pipeline_smoke_pass = bool(
        array8.get("gate_pass") == 1
        and eep.get("field_complete") is True
        and eep.get("structural_gate_pass") is True
        and superposition.get("all_cases_passed") is True
        and fullwave_complete
    )
    engineering_label_allowed = bool(
        pipeline_smoke_pass
        and array16.get("converged") is True
        and float(array16.get("final_delta_s", np.inf)) <= 0.05
        and float(array16.get("matched_passive_rl_min_db", -np.inf)) >= 10.0
    )
    decision = {
        "pipeline_smoke_pass": pipeline_smoke_pass,
        "engineering_label_allowed": engineering_label_allowed,
        "critic_retraining_allowed": engineering_label_allowed,
        "array8_strict_gate_pass": bool(array8.get("gate_pass") == 1),
        "array8_final_delta_s": array8.get("final_delta_s"),
        "array8_matched_passive_rl_min_db": array8.get("matched_passive_rl_min_db"),
        "array16_resource_smoke_converged": bool(array16.get("converged")),
        "array16_resource_smoke_final_delta_s": array16.get("final_delta_s"),
        "array16_resource_smoke_matched_passive_rl_min_db": array16.get(
            "matched_passive_rl_min_db"
        ),
        "eep_port_count": eep.get("port_count"),
        "eep_grid_point_count": eep.get("grid_point_count"),
        "eep_structural_gate_pass": eep.get("structural_gate_pass"),
        "superposition_case_count": superposition.get("complete_case_count"),
        "superposition_complex_nmse_max": superposition.get("complex_nmse_max"),
        "superposition_magnitude_rmse_db_max": superposition.get("magnitude_rmse_db_max"),
        "hfss_smoke_sample_count": int(len(frame)),
        "hfss_smoke_base_gate_pass_count": int(frame["fullwave_gate_pass"].sum()),
        "hfss_smoke_strict_gate_pass_count": int(
            frame["fullwave_gate_pass_strict_local"].sum()
        ),
        "hfss_smoke_combined_complete_count": int(frame["combined_complete"].sum()),
        "hfss_smoke_isolation_complete_count": int(frame["isolation_complete"].sum()),
        "af_to_hfss_smoke_psll_residual_mean_db": float(
            frame["delta_psll_hfss_minus_af_db"].mean()
        ),
        "paired_scene_count": paired.get("scene_count"),
        "paired_variant_count": paired.get("case_count"),
        "paired_close_5_10deg_joint_pass_rate": paired.get("close_5_10deg_joint_pass_rate"),
        "paired_adaptive_sparse_feasible_scene_rate": paired.get(
            "adaptive_sparse_feasible_scene_rate"
        ),
        "label_block_reason": (
            "16x16 resource smoke used only two adaptive passes and is not converged; "
            "its S256, EEP and pattern metrics validate the pipeline only."
        ),
        "next_action": (
            "Continue the 16x16 solution in staged convergence checkpoints, stopping if memory "
            "exceeds the resource ceiling; rerun EEP/HFSS metrics only after Delta S <= 0.05, "
            "two consecutive converged passes and matched passive RL >= 10 dB."
        ),
    }

    frame.to_csv(args.out_dir / "joint_smoke_case_comparison.csv", index=False)
    group_summary.to_csv(args.out_dir / "joint_smoke_group_summary.csv", index=False)
    (args.out_dir / "stage_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    (args.out_dir / "hfss_training_label_decision.json").write_text(
        json.dumps(
            {
                "allow": False,
                "accepted_sample_indices": [],
                "reason": decision["label_block_reason"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    k6 = group_summary[
        (group_summary["group_type"] == "k") & (group_summary["group_value"] == 6)
    ].iloc[0]
    report = [
        "# Grounded-Patch EEP/HFSS Joint Smoke Stage",
        "",
        f"- Pipeline smoke pass: {pipeline_smoke_pass}",
        f"- Engineering label allowed: {engineering_label_allowed}",
        f"- 8x8 Delta S / matched RL: {array8['final_delta_s']:.5f} / {array8['matched_passive_rl_min_db']:.2f} dB",
        f"- 16x16 resource-smoke Delta S / matched RL: {array16['final_delta_s']:.5f} / {array16['matched_passive_rl_min_db']:.2f} dB",
        f"- EEP operator: {eep['port_count']} ports, {eep['grid_point_count']} angular samples",
        f"- Direct-superposition max complex NMSE: {superposition['complex_nmse_max']:.3e}",
        f"- HFSS pattern completeness: {int(frame['combined_complete'].sum())}/{len(frame)} combined, {int(frame['isolation_complete'].sum())}/{len(frame)} isolation",
        f"- Resource-smoke gates: {int(frame['fullwave_gate_pass'].sum())}/{len(frame)} base, {int(frame['fullwave_gate_pass_strict_local'].sum())}/{len(frame)} strict",
        f"- K=6 resource-smoke gate / nearest / local: {int(k6['base_gate_pass_count'])}/{int(k6['case_count'])}, {k6['hfss_smoke_nearest_iso_mean_db']:.2f} dB, {k6['hfss_smoke_local_iso_mean_db']:.2f} dB",
        f"- 5-10 deg paired-scene joint proxy-gate rate: {paired['close_5_10deg_joint_pass_rate']:.2%}",
        "",
        "The 16x16 results are pipeline diagnostics from a deliberately unconverged two-pass solution.",
        "They must not be appended to the HFSS training dataset or used as engineering performance claims.",
    ]
    (args.out_dir / "STAGE_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
