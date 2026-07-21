"""Merge non-overlapping task-level optimization chunks and add diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from reconstruct_task_lcmv_socp_psll import summarize, write_csv


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_INPUTS = [
    ROOT / "hfss_outputs" / f"grounded_patch_task_lcmv_psll_20260717_run01_k{k}"
    for k in (1, 2, 4, 6)
]
DEFAULT_OUT = ROOT / "hfss_outputs" / "grounded_patch_task_lcmv_psll_20260717_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--input-dirs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def target_separation_deg(targets: np.ndarray) -> float:
    if targets.shape[0] <= 1:
        return 180.0
    theta = np.deg2rad(targets[:, 0])
    phi = np.deg2rad(targets[:, 1])
    directions = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1
    )
    dots = np.clip(directions @ directions.T, -1.0, 1.0)
    upper = np.triu_indices(targets.shape[0], 1)
    return float(np.min(np.rad2deg(np.arccos(dots[upper]))))


def separation_bin(value: float) -> str:
    if value < 10.0:
        return "<10"
    if value < 20.0:
        return "10-20"
    if value < 30.0:
        return "20-30"
    return ">=30"


def scan_bin(value: float) -> str:
    if value < 30.0:
        return "<30"
    if value < 45.0:
        return "30-45"
    if value < 60.0:
        return "45-60"
    return ">=60"


def diagnostic_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        keys = [
            ("all", "sparse" if float(row["ratio_requested"]) < 1.0 else "control", "all", "all"),
            (str(row["k"]), str(row["ratio_requested"]), str(row["scan_bin"]), str(row["separation_bin"])),
            (str(row["k"]), str(row["ratio_requested"]), "all", "all"),
        ]
        for key in keys:
            groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (k_value, ratio, scan, separation), members in sorted(groups.items()):
        output.append(
            {
                "k": k_value,
                "ratio": ratio,
                "scan_bin_deg": scan,
                "min_separation_bin_deg": separation,
                "case_count": len(members),
                "effective_ratio_mean": float(np.mean([float(row["ratio_effective"]) for row in members])),
                "joint_gate_rate": float(np.mean([int(row["joint_gate_pass"]) for row in members])),
                "af_gate_rate": float(np.mean([int(row["af_gate_pass"]) for row in members])),
                "rf_gate_rate": float(np.mean([int(row["proxy_rf_gate_pass"]) for row in members])),
                "mainlobe_gate_rate": float(np.mean([int(row["mainlobe_gate_pass"]) for row in members])),
                "final_psll_mean_db": float(np.mean([float(row["final_psll_db"]) for row in members])),
                "psll_delta_mean_db": float(np.mean([float(row["psll_delta_db"]) for row in members])),
                "psll_le_m3_rate": float(np.mean([float(row["final_psll_db"]) <= -3.0 for row in members])),
                "psll_le_m6_rate": float(np.mean([float(row["final_psll_db"]) <= -6.0 for row in members])),
                "worst_active_rl_mean_db": float(np.mean([float(row["worst_active_rl_db"]) for row in members])),
                "total_rl_mean_db": float(np.mean([float(row["total_rl_db"]) for row in members])),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    dataset = np.load(args.dataset, allow_pickle=False)
    rows_by_index: dict[int, dict[str, Any]] = {}
    payloads: list[dict[str, np.ndarray]] = []
    for input_dir in args.input_dirs:
        for row in read_csv(input_dir / "task_lcmv_psll_case_metrics.csv"):
            rows_by_index[int(row["sample_index"])] = row
        source = np.load(input_dir / "task_level_weights.npz", allow_pickle=False)
        payloads.append({key: source[key] for key in source.files})
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(sample_indices, np.arange(len(dataset["k_values"]), dtype=np.int64)):
        raise ValueError("Merged chunks do not cover exactly all dataset samples")
    for row in rows:
        index = int(row["sample_index"])
        k_value = int(dataset["k_values"][index])
        targets = np.asarray(dataset["targets_deg"][index, :k_value], dtype=np.float64)
        separation = target_separation_deg(targets)
        max_theta = float(np.max(targets[:, 0]))
        row["min_target_separation_deg"] = separation
        row["separation_bin"] = separation_bin(separation)
        row["scan_bin"] = scan_bin(max_theta)
        row["ratio_role"] = "control_only" if float(row["ratio_requested"]) >= 0.999 else "sparse"
        row["target_set_hash"] = hashlib.sha1(np.round(targets, 4).tobytes()).hexdigest()
    write_csv(args.out_dir / "task_lcmv_psll_case_metrics.csv", rows)
    write_csv(args.out_dir / "task_lcmv_psll_group_summary.csv", summarize(rows))
    diagnostic_rows = diagnostic_summary(rows)
    write_csv(args.out_dir / "task_lcmv_psll_diagnostic_summary.csv", diagnostic_rows)

    case_count = len(dataset["k_values"])
    weight_shape = payloads[0]["task_weights_real_imag"].shape[1:]
    combined_shape = payloads[0]["combined_weights_real_imag"].shape[1:]
    mask_shape = payloads[0]["masks"].shape[1:]
    weights = np.zeros((case_count, *weight_shape), dtype=np.float32)
    combined = np.zeros((case_count, *combined_shape), dtype=np.float32)
    masks = np.zeros((case_count, *mask_shape), dtype=np.int8)
    covered = np.zeros(case_count, dtype=bool)
    for payload in payloads:
        payload_indices = np.asarray(payload["sample_indices"], dtype=np.int64)
        weights[payload_indices] = payload["task_weights_real_imag"]
        combined[payload_indices] = payload["combined_weights_real_imag"]
        masks[payload_indices] = payload["masks"]
        covered[payload_indices] = True
    if not np.all(covered):
        raise ValueError("Weight payloads do not cover all dataset samples")
    np.savez_compressed(
        args.out_dir / "task_level_weights.npz",
        sample_indices=sample_indices,
        sample_ids=np.asarray(dataset["sample_ids"]),
        task_weights_real_imag=weights,
        combined_weights_real_imag=combined,
        masks=masks,
        k_values=np.asarray(dataset["k_values"]),
        active_ratios_requested=np.asarray(dataset["active_ratios_requested"]),
        active_ratios_effective=np.mean(masks, axis=1).astype(np.float32),
        targets_deg=np.asarray(dataset["targets_deg"]),
        task_valid=np.asarray(dataset["task_valid"]),
        positions_lambda=np.asarray(dataset["positions_lambda"]),
        model_scope=np.asarray("AF regional-SOCP plus local-kernel S256 proxy RF; not HFSS full-wave"),
    )

    sparse = [row for row in rows if row["ratio_role"] == "sparse"]
    control = [row for row in rows if row["ratio_role"] == "control_only"]
    unique_target_sets = len({str(row["target_set_hash"]) for row in rows})
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(rows),
        "method_scope": "AF task metrics plus local-kernel S256 proxy RF; not HFSS full-wave",
        "joint_gate_count": int(sum(int(row["joint_gate_pass"]) for row in rows)),
        "joint_gate_rate": float(np.mean([int(row["joint_gate_pass"]) for row in rows])),
        "sparse_case_count": len(sparse),
        "sparse_joint_gate_rate": float(np.mean([int(row["joint_gate_pass"]) for row in sparse])),
        "control_ratio1_case_count": len(control),
        "control_ratio1_joint_gate_rate": float(np.mean([int(row["joint_gate_pass"]) for row in control])),
        "unique_target_set_count": unique_target_sets,
        "paired_ratio_counterfactual_available": bool(unique_target_sets < len(rows)),
        "adaptive_minimum_ratio_claim_allowed": False,
        "adaptive_minimum_ratio_block_reason": (
            "All 2400 target direction sets are unique; ratios are not paired within the same scene."
        ),
        "hfss_label_generation_allowed": False,
        "hfss_next_step": "Validate only joint-pass and near-boundary candidates with a validated 256-port EEP or HFSS operator.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "fullwave_label_gate_decision.json").write_text(
        json.dumps(
            {
                "allowed": False,
                "reason": "Current RF results use a local-kernel S256 proxy and current pattern results use AF.",
                "proxy_joint_pass_count": summary["joint_gate_count"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
