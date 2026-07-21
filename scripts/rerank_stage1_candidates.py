"""Rerank teacher candidates with the stage-1 metric critic and merge top-1.

This script takes several optimized teacher directories, predicts HFSS-aware
metrics for their selected samples, and writes a dataset_arrays-compatible
teacher directory containing the best candidate per sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_stage1_metric_critic import MetricCritic, build_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_CRITIC = DEFAULT_DATASET_DIR / "training_runs" / "stage1_metric_critic_hfiloop_k6_lr_60ep_20260701" / "metric_critic.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--critic-path", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--gate-psll-max-db", type=float, default=0.0)
    parser.add_argument("--gate-nearest-iso-min-db", type=float, default=25.0)
    parser.add_argument("--gate-local-iso-min-db", type=float, default=15.0)
    parser.add_argument(
        "--min-gate-prob",
        type=float,
        default=0.0,
        help=(
            "Optional full-wave feasibility critic hard gate. Candidates with "
            "predicted gate probability below this value are ranked behind "
            "nearby feasible candidates even if AF-derived metrics look good."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    return {key: arrays[key] for key in arrays.files}


def gate_penalties(psll: float, nearest: float, local: float, args: argparse.Namespace) -> dict[str, float]:
    psll_pen = max(0.0, psll - float(args.gate_psll_max_db))
    nearest_pen = max(0.0, float(args.gate_nearest_iso_min_db) - nearest)
    local_pen = max(0.0, float(args.gate_local_iso_min_db) - local)
    return {
        "psll_penalty_db": psll_pen,
        "nearest_penalty_db": nearest_pen,
        "local_penalty_db": local_pen,
        "iso_worst_penalty_db": max(nearest_pen, local_pen),
        "iso_sum_penalty_db": nearest_pen + local_pen,
    }


def candidate_score(
    psll: float,
    nearest: float,
    local: float,
    energy: float,
    gate_prob: float,
    args: argparse.Namespace,
) -> float:
    penalties = gate_penalties(psll, nearest, local, args)
    psll_pen = penalties["psll_penalty_db"]
    nearest_pen = penalties["nearest_penalty_db"]
    local_pen = penalties["local_penalty_db"]
    gate_prob_pen = max(0.0, float(args.min_gate_prob) - float(gate_prob))
    return (
        150.0 * gate_prob_pen
        + 100.0 * psll_pen
        + 60.0 * nearest_pen
        + 35.0 * local_pen
        + 1.0 * psll
        - 0.15 * nearest
        - 0.10 * local
        + 5.0 * energy
        - 0.5 * gate_prob
    )


def hard_gate_key(
    psll: float,
    nearest: float,
    local: float,
    energy: float,
    gate_prob: float,
    args: argparse.Namespace,
) -> tuple[float, ...]:
    penalties = gate_penalties(psll, nearest, local, args)
    psll_pen = penalties["psll_penalty_db"]
    iso_worst = penalties["iso_worst_penalty_db"]
    iso_sum = penalties["iso_sum_penalty_db"]
    critic_fail = 1.0 if float(gate_prob) < float(args.min_gate_prob) else 0.0
    isolation_fail = 1.0 if iso_sum > 0.0 else 0.0
    psll_fail = 1.0 if psll_pen > 0.0 else 0.0
    # Lexicographic priority: isolation feasibility first, PSLL second.
    return (
        critic_fail,
        max(0.0, float(args.min_gate_prob) - float(gate_prob)),
        isolation_fail,
        iso_worst,
        iso_sum,
        psll_fail,
        psll_pen,
        psll,
        energy,
        -nearest,
        -local,
        -gate_prob,
    )


def make_metric_payload(base: dict[str, np.ndarray], teacher: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "sample_index": indices.astype(np.int64),
        "masks": teacher["masks"][indices].astype(np.int8),
        "weights_real_imag": teacher["task_weights_real_imag"][indices].astype(np.float32),
        "k_values": teacher.get("k_values", base["k_values"])[indices],
        "active_ratios_requested": teacher.get("active_ratios_requested", base["active_ratios_requested"])[indices],
        "num_active": teacher.get("num_active", base["num_active"])[indices],
        "targets_deg": teacher.get("targets_deg", base["targets_deg"])[indices],
        "task_valid": teacher.get("task_valid", base["task_valid"])[indices],
    }


def predict_candidates(
    *,
    model: MetricCritic,
    checkpoint: dict[str, Any],
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x = build_features(payload)
    x = ((x - checkpoint["x_mean"]) / checkpoint["x_std"]).astype(np.float32)
    device = torch.device("cpu")
    model.eval()
    preds: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 128):
            xb = torch.from_numpy(x[start : start + 128]).to(device)
            metric_std, gate_logit = model(xb)
            metric = metric_std.cpu().numpy() * checkpoint["y_std"] + checkpoint["y_mean"]
            preds.append(metric.astype(np.float32))
            gates.append(gate_logit.cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0), np.concatenate(gates, axis=0)


def read_candidate_metrics(path: Path) -> dict[int, dict[str, str]]:
    metrics_path = path / "iso_lcmv_metrics.csv"
    out: dict[int, dict[str, str]] = {}
    if not metrics_path.exists():
        return out
    with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            out[int(row["sample_index"])] = row
    return out


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
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = load_npz(args.dataset_dir / "dataset_arrays.npz")
    checkpoint = torch.load(args.critic_path, map_location="cpu", weights_only=False)
    model = MetricCritic(int(checkpoint["input_dim"]), int(checkpoint["hidden"]), len(checkpoint["metric_names"]))
    model.load_state_dict(checkpoint["model_state"])

    compatible = dict(base)
    masks = base["masks"].astype(np.int8).copy()
    weights_ri = base["task_weights_real_imag"].astype(np.float32).copy()
    rows: list[dict[str, Any]] = []
    best_by_sample: dict[int, dict[str, Any]] = {}

    for candidate_dir in args.candidate_dirs:
        teacher = load_npz(candidate_dir / "dataset_arrays.npz")
        aux_path = candidate_dir / "iso_lcmv_teacher_arrays.npz"
        if aux_path.exists():
            aux = load_npz(aux_path)
            indices = aux.get("selected_indices", np.asarray([], dtype=np.int64))
        else:
            indices = np.arange(base["k_values"].shape[0], dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        payload = make_metric_payload(base, teacher, indices)
        pred, gate_logits = predict_candidates(model=model, checkpoint=checkpoint, payload=payload)
        af_metrics = read_candidate_metrics(candidate_dir)
        for row_idx, sample_index in enumerate(indices):
            psll, nearest, local, energy = [float(v) for v in pred[row_idx]]
            gate_prob = float(1.0 / (1.0 + np.exp(-float(gate_logits[row_idx]))))
            score = candidate_score(psll, nearest, local, energy, gate_prob, args)
            penalties = gate_penalties(psll, nearest, local, args)
            selection_key = hard_gate_key(psll, nearest, local, energy, gate_prob, args)
            source_metrics = af_metrics.get(int(sample_index), {})
            row = {
                "sample_index": int(sample_index),
                "sample_id": str(base["sample_ids"][sample_index]),
                "candidate_dir": str(candidate_dir),
                "candidate_name": candidate_dir.name,
                "pred_psll_db": psll,
                "pred_nearest_iso_db": nearest,
                "pred_local_iso_db": local,
                "pred_energy_proxy": energy,
                "pred_gate_prob": gate_prob,
                "pred_critic_gate_pass": gate_prob >= float(args.min_gate_prob),
                "pred_iso_gate_pass": penalties["iso_sum_penalty_db"] == 0.0,
                "pred_full_gate_pass": penalties["iso_sum_penalty_db"] == 0.0 and penalties["psll_penalty_db"] == 0.0,
                "pred_psll_penalty_db": penalties["psll_penalty_db"],
                "pred_nearest_penalty_db": penalties["nearest_penalty_db"],
                "pred_local_penalty_db": penalties["local_penalty_db"],
                "pred_iso_worst_penalty_db": penalties["iso_worst_penalty_db"],
                "pred_iso_sum_penalty_db": penalties["iso_sum_penalty_db"],
                "rerank_score": score,
                "hard_gate_key": "|".join(f"{part:.6g}" for part in selection_key),
                "af_psll_db": source_metrics.get("chosen_psll_to_weakest_peak_db", ""),
                "af_local_iso_db": source_metrics.get("chosen_local_isolation_min_db", ""),
                "af_point_iso_db": source_metrics.get("chosen_isolation_min_db", ""),
            }
            rows.append(row)
            old = best_by_sample.get(int(sample_index))
            if old is None or selection_key < old["_selection_key"]:
                chosen_row = dict(row)
                chosen_row["_selection_key"] = selection_key
                best_by_sample[int(sample_index)] = chosen_row
                masks[sample_index] = teacher["masks"][sample_index].astype(np.int8)
                weights_ri[sample_index] = teacher["task_weights_real_imag"][sample_index].astype(np.float32)

    selected_indices = np.asarray(sorted(best_by_sample), dtype=np.int64)
    compatible["masks"] = masks
    compatible["task_weights_real_imag"] = weights_ri
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **compatible)
    np.savez_compressed(
        args.out_dir / "iso_lcmv_teacher_arrays.npz",
        masks=masks,
        task_weights_real_imag=weights_ri,
        selected_indices=selected_indices,
        sample_ids=base["sample_ids"],
        k_values=base["k_values"],
        active_ratios_requested=base["active_ratios_requested"],
        num_active=base["num_active"],
        targets_deg=base["targets_deg"],
        task_valid=base["task_valid"],
        positions_lambda=base["positions_lambda"],
        element_ixiy=base["element_ixiy"],
    )
    write_csv(args.out_dir / "candidate_scores.csv", rows)
    selected_rows = []
    for sample_index in selected_indices:
        row = dict(best_by_sample[int(sample_index)])
        row.pop("_selection_key", None)
        selected_rows.append(row)
    write_csv(args.out_dir / "selected_candidate_scores.csv", selected_rows)
    summary = {
        "out_dir": str(args.out_dir),
        "critic_path": str(args.critic_path),
        "candidate_dirs": [str(path) for path in args.candidate_dirs],
        "candidate_count": len(rows),
        "selected_count": int(selected_indices.shape[0]),
        "selected_indices": [int(i) for i in selected_indices],
        "outputs": {
            "dataset_arrays": str(args.out_dir / "dataset_arrays.npz"),
            "candidate_scores": str(args.out_dir / "candidate_scores.csv"),
            "selected_scores": str(args.out_dir / "selected_candidate_scores.csv"),
        },
    }
    (args.out_dir / "rerank_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
