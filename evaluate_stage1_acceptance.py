"""Create a hard Stage-1 acceptance report and HFSS launch guard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_RESIDUAL_DIR = DATASET_DIR / "stage1_fullwave_residual_dataset_v2_20260714"
DEFAULT_TRAIN_DIR = DATASET_DIR / "training_runs" / "fullwave_residual_critic_v2_20260714"
DEFAULT_OUT_DIR = DATASET_DIR / "stage1_acceptance_v2_20260714"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-dir", type=Path, default=DEFAULT_RESIDUAL_DIR)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


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


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(1.96 * array.std(ddof=1) / math.sqrt(array.size))


def bin_scan(value: float) -> str:
    if value <= 30.0:
        return "0_30"
    if value <= 50.0:
        return "30_50"
    return "50_90"


def bin_separation(value: float) -> str:
    if value < 15.0:
        return "lt15"
    if value < 25.0:
        return "15_25"
    return "ge25"


def grouped_fullwave_stats(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    scalar_names = [str(name) for name in data["scalar_names"]]
    max_scan_col = scalar_names.index("max_scan_theta_deg_norm")
    min_sep_col = scalar_names.index("min_target_separation_deg_norm")
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    split_names = ("train", "val", "test")
    for row in range(data["sample_index"].shape[0]):
        max_scan = float(data["scalar_features"][row, max_scan_col]) * 90.0
        min_sep = float(data["scalar_features"][row, min_sep_col]) * 180.0
        key = (
            split_names[int(data["split_id"][row])],
            int(data["k_values"][row]),
            round(float(data["active_ratios"][row]), 2),
            bin_scan(max_scan),
            bin_separation(min_sep),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        rows = np.asarray(values, dtype=np.int64)
        output.append(
            {
                "split": key[0],
                "k": key[1],
                "active_ratio": key[2],
                "scan_angle_bin_deg": key[3],
                "target_separation_bin_deg": key[4],
                "variant_count": int(rows.size),
                "unique_scene_count": int(np.unique(data["sample_index"][rows]).size),
                "gate15_rate": float(data["gates"][rows, 0].mean()),
                "gate20_rate": float(data["gates"][rows, 1].mean()),
                "mainlobe_gate_rate": float(data["gates"][rows, 2].mean()),
                "strict_engineering_rate": float(data["gates"][rows, 3].mean()),
                "psll_mean_db": float(data["hfss_metrics"][rows, 0].mean()),
                "psll_worst_db": float(data["hfss_metrics"][rows, 0].max()),
                "nearest_iso_mean_db": float(data["hfss_metrics"][rows, 1].mean()),
                "nearest_iso_worst_db": float(data["hfss_metrics"][rows, 1].min()),
                "local_iso_mean_db": float(data["hfss_metrics"][rows, 2].mean()),
                "local_iso_worst_db": float(data["hfss_metrics"][rows, 2].min()),
                "mainlobe_drop_mean_db": float(
                    np.mean(
                        data["scalar_features"][rows, scalar_names.index("reference_hfss_peak_min_db")]
                        - data["hfss_metrics"][rows, 3]
                    )
                ),
            }
        )
    return output


def best_fixed_strategy(
    data: dict[str, np.ndarray], gate_col: int, min_variants: int = 20
) -> dict[str, Any]:
    test = np.flatnonzero(data["split_id"] == 2)
    best: dict[str, Any] | None = None
    for strategy in np.unique(data["strategy"][test]).tolist():
        rows = test[data["strategy"][test] == strategy]
        if rows.size < int(min_variants):
            continue
        value = float(data["gates"][rows, gate_col].mean())
        item = {"strategy": str(strategy), "rate": value, "variant_count": int(rows.size)}
        if best is None or value > best["rate"]:
            best = item
    assert best is not None
    return best


def aggregate_seed_metrics(train_dir: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    summaries: dict[int, dict[str, Any]] = {}
    for path in sorted(train_dir.glob("seed_*/run_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries[int(payload["seed"])] = payload
    rows: list[dict[str, Any]] = []
    metrics = {
        "gate15_auroc": "gate15_cal_auroc",
        "gate15_auprc": "gate15_cal_auprc",
        "gate15_ece": "gate15_cal_ece",
        "gate20_auroc": "gate20_cal_auroc",
        "gate20_auprc": "gate20_cal_auprc",
        "gate20_brier": "gate20_cal_brier",
        "gate20_ece": "gate20_cal_ece",
        "mainlobe_auroc": "mainlobe_gate_cal_auroc",
        "strict_auroc": "strict_engineering_gate_cal_auroc",
        "rank_gate20_top1": "rank_gate20_rate",
        "rank_strict_top1": "rank_strict_engineering_gate_rate",
        "conservative_gate20_top1": "conservative_gate20_rate",
        "oracle_gate20": "oracle_gate20_rate",
        "oracle_strict": "oracle_strict_engineering_rate",
    }
    for label, key in metrics.items():
        values = [float(summary["test"][key]) for summary in summaries.values()]
        mean, ci = mean_ci95(values)
        rows.append({"metric": label, "mean": mean, "ci95_half_width": ci, "seed_values": json.dumps(values)})
    return rows, summaries


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.residual_dir / "fullwave_residual_dataset_v2.npz")
    aggregate_rows, summaries = aggregate_seed_metrics(args.train_dir)
    aggregate = {row["metric"]: row for row in aggregate_rows}
    fixed_gate15 = best_fixed_strategy(data, 0)
    fixed_gate20 = best_fixed_strategy(data, 1)
    fixed_strict = best_fixed_strategy(data, 3)
    phase1_checks = [
        {
            "criterion": "scene_test_gate20_auroc_ge_0.88",
            "value": aggregate["gate20_auroc"]["mean"],
            "threshold": 0.88,
            "passed": aggregate["gate20_auroc"]["mean"] >= 0.88,
        },
        {
            "criterion": "scene_test_gate20_ece_le_0.08",
            "value": aggregate["gate20_ece"]["mean"],
            "threshold": 0.08,
            "passed": aggregate["gate20_ece"]["mean"] <= 0.08,
        },
        {
            "criterion": "top1_gate20_above_best_fixed",
            "value": aggregate["rank_gate20_top1"]["mean"],
            "threshold": fixed_gate20["rate"],
            "passed": aggregate["rank_gate20_top1"]["mean"] > fixed_gate20["rate"],
        },
        {
            "criterion": "strict_top1_ge_0.35",
            "value": aggregate["rank_strict_top1"]["mean"],
            "threshold": 0.35,
            "passed": aggregate["rank_strict_top1"]["mean"] >= 0.35,
        },
    ]
    phase2_checks = [
        {
            "criterion": "best_of_n_oracle_gate20_ge_0.75",
            "value": aggregate["oracle_gate20"]["mean"],
            "threshold": 0.75,
            "passed": aggregate["oracle_gate20"]["mean"] >= 0.75,
        },
        {
            "criterion": "critic_top1_gate20_ge_0.60",
            "value": aggregate["rank_gate20_top1"]["mean"],
            "threshold": 0.60,
            "passed": aggregate["rank_gate20_top1"]["mean"] >= 0.60,
        },
        {
            "criterion": "strict_top1_ge_0.50",
            "value": aggregate["rank_strict_top1"]["mean"],
            "threshold": 0.50,
            "passed": aggregate["rank_strict_top1"]["mean"] >= 0.50,
        },
    ]
    phase1_passed = all(bool(row["passed"]) for row in phase1_checks)
    phase2_passed = all(bool(row["passed"]) for row in phase2_checks)
    launch_guard = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage1_passed": phase1_passed,
        "stage2_passed": phase2_passed,
        "small_hfss_smoke_allowed": True,
        "large_hfss_allowed": bool(phase1_passed and phase2_passed),
        "eep_16_port_smoke_allowed": True,
        "eep_256_port_allowed": False,
        "reason": (
            "Candidate oracle and strict top-1 remain below acceptance thresholds; "
            "only reviewed near-boundary shortlists and a 16-port EEP smoke may run."
        ),
    }
    grouped_rows = grouped_fullwave_stats(data)
    write_csv(args.out_dir / "five_seed_metrics.csv", aggregate_rows)
    write_csv(args.out_dir / "fullwave_breakdown_k_ratio_scan_separation.csv", grouped_rows)
    write_csv(args.out_dir / "phase1_checks.csv", phase1_checks)
    write_csv(args.out_dir / "phase2_checks.csv", phase2_checks)
    ablation_rows = [
        {"method": "best_fixed_gate20_strategy", "gate20_top1_rate": fixed_gate20["rate"], "strict_top1_rate": fixed_strict["rate"]},
        {"method": "critic_rank_mean_5seed", "gate20_top1_rate": aggregate["rank_gate20_top1"]["mean"], "strict_top1_rate": aggregate["rank_strict_top1"]["mean"]},
        {"method": "critic_conservative_mean_5seed", "gate20_top1_rate": aggregate["conservative_gate20_top1"]["mean"], "strict_top1_rate": float("nan")},
        {"method": "observed_best_of_n_oracle", "gate20_top1_rate": aggregate["oracle_gate20"]["mean"], "strict_top1_rate": aggregate["oracle_strict"]["mean"]},
    ]
    write_csv(args.out_dir / "stage1_ablation_table.csv", ablation_rows)
    (args.out_dir / "launch_guard.json").write_text(json.dumps(launch_guard, indent=2), encoding="utf-8")
    failed_phase1 = [row["criterion"] for row in phase1_checks if not row["passed"]]
    failed_phase2 = [row["criterion"] for row in phase2_checks if not row["passed"]]
    report_lines = [
        "# Stage-1 Full-Wave Residual Critic Report",
        "",
        f"- Variants/scenes: {data['sample_index'].shape[0]}/{np.unique(data['sample_index']).size}",
        f"- Five-seed gate20 AUROC: {aggregate['gate20_auroc']['mean']:.4f} +/- {aggregate['gate20_auroc']['ci95_half_width']:.4f}",
        f"- Five-seed gate20 AUPRC: {aggregate['gate20_auprc']['mean']:.4f} +/- {aggregate['gate20_auprc']['ci95_half_width']:.4f}",
        f"- Five-seed gate20 ECE: {aggregate['gate20_ece']['mean']:.4f} +/- {aggregate['gate20_ece']['ci95_half_width']:.4f}",
        f"- Rank top-1 gate20: {aggregate['rank_gate20_top1']['mean']:.4f} +/- {aggregate['rank_gate20_top1']['ci95_half_width']:.4f}",
        f"- Observed test oracle gate20: {aggregate['oracle_gate20']['mean']:.4f}",
        f"- Best fixed gate20 strategy: {fixed_gate20['strategy']} ({fixed_gate20['rate']:.4f})",
        f"- Strict engineering top-1: {aggregate['rank_strict_top1']['mean']:.4f}",
        "",
        f"## Stage 1: {'PASS' if phase1_passed else 'FAIL'}",
        "Failed checks: " + (", ".join(failed_phase1) if failed_phase1 else "none"),
        "",
        f"## Stage 2: {'PASS' if phase2_passed else 'FAIL'}",
        "Failed checks: " + (", ".join(failed_phase2) if failed_phase2 else "none"),
        "",
        "## Engineering Decision",
        "Do not start a large HFSS batch. Expand the candidate oracle with regional inequality projection, run only critic-near-boundary smoke cases, and validate a 16-port EEP export before scaling.",
    ]
    (args.out_dir / "STAGE1_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary = {
        "phase1_passed": phase1_passed,
        "phase2_passed": phase2_passed,
        "failed_phase1": failed_phase1,
        "failed_phase2": failed_phase2,
        "best_fixed": {"gate15": fixed_gate15, "gate20": fixed_gate20, "strict": fixed_strict},
        "five_seed_metrics": {row["metric"]: {"mean": row["mean"], "ci95_half_width": row["ci95_half_width"]} for row in aggregate_rows},
        "launch_guard": launch_guard,
        "outputs": {
            "report": str(args.out_dir / "STAGE1_REPORT.md"),
            "breakdown": str(args.out_dir / "fullwave_breakdown_k_ratio_scan_separation.csv"),
            "ablation": str(args.out_dir / "stage1_ablation_table.csv"),
            "launch_guard": str(args.out_dir / "launch_guard.json"),
        },
    }
    (args.out_dir / "stage1_acceptance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
