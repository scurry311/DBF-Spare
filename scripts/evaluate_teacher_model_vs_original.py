"""Compare optimized teachers and trained steering-mask models with original labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_hfss_grid_mask import GridMaskNet, collect_grid_logits
from train_hfss_steering_mask import MaskNet, analytic_weights_from_gate, collect_logits
from train_hfss_surrogate import (
    DEFAULT_DATASET_DIR,
    HfssArrayDataset,
    K_VALUES,
    build_features,
    compute_af_metrics,
    hard_gate_from_logits,
    load_split_manifest,
    make_grid,
    safe_mean,
    safe_percentile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--teacher-dataset-dir", type=Path, required=True)
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_arrays(dataset_dir: Path) -> dict[str, np.ndarray]:
    arrays_npz = np.load(dataset_dir / "dataset_arrays.npz", allow_pickle=False)
    return {key: arrays_npz[key] for key in arrays_npz.files}


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


def summarize_group(
    split: str,
    k_label: int | str,
    active_label: str,
    idx: np.ndarray,
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    ref_psll = reference["psll_to_weakest_peak_db"][idx]
    cand_psll = candidate["psll_to_weakest_peak_db"][idx]
    ref_weak = reference["target_peak_min_db"][idx]
    cand_weak = candidate["target_peak_min_db"][idx]
    return {
        "split": split,
        "k": k_label,
        "active_ratio": active_label,
        "n": int(idx.size),
        "reference_psll_weak_mean_db": safe_mean(ref_psll),
        "candidate_psll_weak_mean_db": safe_mean(cand_psll),
        "delta_psll_weak_mean_db": safe_mean(cand_psll - ref_psll),
        "reference_psll_weak_p95_db": safe_percentile(ref_psll, 95),
        "candidate_psll_weak_p95_db": safe_percentile(cand_psll, 95),
        "reference_weak_peak_mean_db": safe_mean(ref_weak),
        "candidate_weak_peak_mean_db": safe_mean(cand_weak),
        "delta_weak_peak_mean_db": safe_mean(cand_weak - ref_weak),
        "reference_weak_peak_p05_db": safe_percentile(ref_weak, 5),
        "candidate_weak_peak_p05_db": safe_percentile(cand_weak, 5),
        "reference_isolation_worst_p05_db": safe_percentile(reference["isolation_worst_db"][idx], 5),
        "candidate_isolation_worst_p05_db": safe_percentile(candidate["isolation_worst_db"][idx], 5),
    }


def summarize_all(
    arrays: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for split, idx in splits.items():
        rows.append(summarize_group(split, "all", "all", idx, reference, candidate))
        for k in K_VALUES:
            k_idx = idx[arrays["k_values"][idx] == k]
            rows.append(summarize_group(split, k, "all", k_idx, reference, candidate))
            for active_ratio in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                sub = k_idx[np.isclose(arrays["active_ratios_requested"][k_idx], active_ratio)]
                rows.append(summarize_group(split, k, f"{active_ratio:.1f}", sub, reference, candidate))

    test_idx = splits["test"]
    k46 = test_idx[np.isin(arrays["k_values"][test_idx], [4, 6])]
    headline = {
        "test_delta_psll_weak_mean_db": safe_mean(
            candidate["psll_to_weakest_peak_db"][test_idx] - reference["psll_to_weakest_peak_db"][test_idx]
        ),
        "test_delta_weak_peak_mean_db": safe_mean(
            candidate["target_peak_min_db"][test_idx] - reference["target_peak_min_db"][test_idx]
        ),
        "test_k46_delta_psll_weak_mean_db": safe_mean(
            candidate["psll_to_weakest_peak_db"][k46] - reference["psll_to_weakest_peak_db"][k46]
        ),
        "test_k46_delta_weak_peak_mean_db": safe_mean(
            candidate["target_peak_min_db"][k46] - reference["target_peak_min_db"][k46]
        ),
        "test_candidate_psll_weak_p95_db": safe_percentile(candidate["psll_to_weakest_peak_db"][test_idx], 95),
        "test_k46_candidate_psll_weak_p95_db": safe_percentile(candidate["psll_to_weakest_peak_db"][k46], 95),
    }
    return rows, headline


def main() -> None:
    args = parse_args()
    original = load_arrays(args.original_dataset_dir)
    teacher = load_arrays(args.teacher_dataset_dir)
    splits = load_split_manifest(args.original_dataset_dir / "training_split_manifest.json")
    _, _, eval_grid = make_grid(theta_step=2.0, phi_step=5.0)

    original_weights = original["task_weights_real_imag"][..., 0] + 1j * original["task_weights_real_imag"][..., 1]
    teacher_weights = teacher["task_weights_real_imag"][..., 0] + 1j * teacher["task_weights_real_imag"][..., 1]
    original_metrics = compute_af_metrics(
        original_weights.astype(np.complex64),
        original["targets_deg"].astype(np.float32),
        original["task_valid"].astype(np.float32),
        original["positions_lambda"].astype(np.float32),
        eval_grid,
    )
    teacher_metrics = compute_af_metrics(
        teacher_weights.astype(np.complex64),
        teacher["targets_deg"].astype(np.float32),
        teacher["task_valid"].astype(np.float32),
        teacher["positions_lambda"].astype(np.float32),
        eval_grid,
    )

    features = build_features(
        teacher["k_values"],
        teacher["active_ratios_requested"],
        teacher["targets_deg"],
        teacher["task_valid"],
    )
    all_idx = np.arange(teacher["k_values"].shape[0], dtype=np.int64)
    dataset = HfssArrayDataset(teacher, features, all_idx)
    checkpoint_path = args.model_run_dir / "steering_mask_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.model_run_dir / "grid_mask_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if config.get("model_type") == "grid_mask":
        model_cfg = config["model"]
        model = GridMaskNet(
            feature_dim=features.shape[1],
            element_ixiy=teacher["element_ixiy"],
            positions_lambda=teacher["positions_lambda"],
            channels=int(model_cfg["channels"]),
            condition_dim=int(model_cfg["condition_dim"]),
            attention_layers=int(model_cfg["attention_layers"]),
        )
        model.load_state_dict(checkpoint["model_state"])
        _idx, logits = collect_grid_logits(model, dataset, torch.device("cpu"), args.batch_size)
    else:
        hidden = int(config["args"]["hidden"])
        model = MaskNet(features.shape[1], hidden)
        model.load_state_dict(checkpoint["model_state"])
        _idx, logits = collect_logits(model, dataset, torch.device("cpu"), args.batch_size)
    hard_gate = hard_gate_from_logits(logits, teacher["num_active"].astype(np.float32))
    model_weights = analytic_weights_from_gate(hard_gate, teacher)
    model_metrics = compute_af_metrics(
        model_weights.astype(np.complex64),
        teacher["targets_deg"].astype(np.float32),
        teacher["task_valid"].astype(np.float32),
        teacher["positions_lambda"].astype(np.float32),
        eval_grid,
    )

    teacher_rows, teacher_headline = summarize_all(teacher, splits, original_metrics, teacher_metrics)
    model_rows, model_headline = summarize_all(teacher, splits, original_metrics, model_metrics)
    teacher_summary_path = args.teacher_dataset_dir / "teacher_vs_original_fine_summary.csv"
    teacher_json_path = args.teacher_dataset_dir / "teacher_vs_original_fine_summary.json"
    model_summary_path = args.model_run_dir / "eval_summary_vs_original_random.csv"
    model_json_path = args.model_run_dir / "eval_summary_vs_original_random.json"
    write_csv(teacher_summary_path, teacher_rows)
    write_csv(model_summary_path, model_rows)
    teacher_json_path.write_text(json.dumps({"headline": teacher_headline}, ensure_ascii=False, indent=2), encoding="utf-8")
    model_json_path.write_text(json.dumps({"headline": model_headline}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "teacher_vs_original": teacher_headline,
                "model_vs_original": model_headline,
                "outputs": {
                    "teacher_summary": str(teacher_summary_path),
                    "teacher_json": str(teacher_json_path),
                    "model_summary": str(model_summary_path),
                    "model_json": str(model_json_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
