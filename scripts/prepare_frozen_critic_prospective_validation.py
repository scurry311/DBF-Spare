#!/usr/bin/env python3
"""Pre-register frozen-critic predictions before prospective HFSS execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build_dense_implementation_residual_dataset import (
    KMAX,
    NUM_ELEMENTS,
    amplitude_stats,
    min_separation,
    nominal_active_rl,
)
from generate_dense_boundary_hard_negatives import active_metrics
from generate_expanded_independent_residual_scenes import target_hash
from optimize_trusted_eep_s256_joint_weights import pattern_metrics
from train_fullwave_residual_critic_v2 import (
    FullwaveResidualCritic,
    spatial_features,
    target_features,
)
from validate_trusted_eep_hfss_residuals import series_network_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "prospective_gate15_scenes_20260725_run01"
DEFAULT_CHECKPOINT = (
    ROOT
    / "baselines"
    / "2026-07-25-gate15-boundary"
    / "snapshots"
    / "best_checkpoint.pt"
)
DEFAULT_CALIBRATOR = (
    ROOT
    / "baselines"
    / "2026-07-25-gate15-boundary"
    / "snapshots"
    / "pooled_calibration.json"
)
DEFAULT_TRAINING_DATASET = (
    ROOT
    / "baselines"
    / "2026-07-25-gate15-boundary"
    / "snapshots"
    / "critic_dataset.npz"
)
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_EXCITATIONS = (
    ROOT
    / "hfss_outputs"
    / "trusted_dense_joint_hfss_smoke_20260724_run01"
    / "case_excitations.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "prospective_frozen_critic_20260725_run01"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--training-dataset", type=Path, default=DEFAULT_TRAINING_DATASET)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--excitations", type=Path, default=DEFAULT_EXCITATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--uncertainty-kappa", type=float, default=1.0)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float64)
    return np.where(
        data >= 0.0,
        1.0 / (1.0 + np.exp(-data)),
        np.exp(data) / (1.0 + np.exp(data)),
    )


def apply_calibrator(raw_probability: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    isotonic = np.interp(
        raw_probability,
        np.asarray(payload["x_thresholds"], dtype=float),
        np.asarray(payload["y_thresholds"], dtype=float),
    )
    alpha = float(payload["alpha"])
    return np.clip(
        (1.0 - alpha) * isotonic + alpha * raw_probability,
        1.0e-6,
        1.0 - 1.0e-6,
    )


def exact_metrics(
    tasks_external: np.ndarray,
    targets: np.ndarray,
    operator: dict[str, np.ndarray],
    antenna_map: np.ndarray,
) -> dict[str, float]:
    return pattern_metrics(
        tasks_external,
        targets,
        operator["theta_deg"],
        operator["phi_deg"],
        operator["etheta"],
        operator["ephi"],
        antenna_map,
    )


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prospective freeze: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset_dir / "dataset_arrays.npz"
    dataset = load_npz(dataset_path)
    training = load_npz(args.training_dataset)
    operator = load_npz(args.operator)
    excitations = load_npz(args.excitations)
    antenna_map = np.asarray(excitations["antenna_wave_map"], dtype=np.complex64)
    s_matrix, expected_map, _series_z = series_network_map(
        np.asarray(operator["s_raw"], dtype=np.complex128), 1.0e10
    )
    if np.max(np.abs(expected_map - antenna_map)) > 1.0e-7:
        raise RuntimeError("Antenna-wave map mismatch")

    training_hashes = set()
    for index in range(int(training["sample_index"].size)):
        valid = np.asarray(training["task_valid"][index], dtype=bool)
        training_hashes.add(target_hash(training["targets_deg"][index][valid]))
    prospective_hashes = []
    for index in range(int(dataset["candidate_indices"].size)):
        valid = np.asarray(dataset["task_valid"][index], dtype=bool)
        prospective_hashes.append(target_hash(dataset["targets_deg"][index][valid]))
    overlap = sorted(set(prospective_hashes) & training_hashes)
    if overlap:
        raise RuntimeError(f"Prospective target leakage detected: {overlap[:5]}")

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    calibrator_document = json.loads(args.calibrator.read_text(encoding="utf-8"))
    calibrators = calibrator_document["calibrators"]
    scalar_names = [str(value) for value in training["scalar_names"]]
    scalar_col = {name: index for index, name in enumerate(scalar_names)}
    if int(checkpoint["scalar_dim"]) != len(scalar_names):
        raise ValueError("Checkpoint and scalar schema dimensions do not match")

    candidate_count = int(dataset["candidate_indices"].size)
    scalar = np.zeros((candidate_count, len(scalar_names)), dtype=np.float32)
    eep_metrics = np.zeros((candidate_count, 5), dtype=np.float32)
    nominal_tasks_ri = np.asarray(dataset["task_weights_real_imag"], dtype=np.float32)
    cache: dict[int, tuple[dict[str, float], float, float]] = {}
    pre_hfss_active_gate = np.zeros(candidate_count, dtype=np.int8)
    pre_hfss_worst_active_rl = np.zeros(candidate_count, dtype=np.float32)
    pre_hfss_total_rl = np.zeros(candidate_count, dtype=np.float32)
    for candidate in range(candidate_count):
        sample = int(dataset["sample_indices"][candidate])
        k_value = int(dataset["k_values"][candidate])
        mask = np.asarray(dataset["masks"][candidate], dtype=bool)
        valid = np.asarray(dataset["task_valid"][candidate], dtype=bool)
        targets = np.asarray(dataset["targets_deg"][candidate][valid], dtype=np.float64)
        tasks_internal = (
            nominal_tasks_ri[candidate, :, :k_value, 0]
            + 1j * nominal_tasks_ri[candidate, :, :k_value, 1]
        )
        if sample not in cache:
            nominal = exact_metrics(
                np.conjugate(tasks_internal), targets, operator, antenna_map
            )
            nominal_rl, nominal_total_rl = nominal_active_rl(
                tasks_internal, mask, s_matrix
            )
            cache[sample] = (nominal, nominal_rl, nominal_total_rl)
        nominal, nominal_rl, nominal_total_rl = cache[sample]
        energy, weight_l2, max_amp, dynamic_db = amplitude_stats(tasks_internal)
        phase_rms = float(dataset.get("phase_error_rms_deg", np.zeros(candidate_count))[candidate])
        gain_rms = float(dataset.get("gain_error_rms_db", np.zeros(candidate_count))[candidate])
        dropout = int(dataset.get("dropout_count", np.zeros(candidate_count))[candidate])
        phase_bits = int(dataset.get("phase_bits", np.full(candidate_count, 16))[candidate])
        amplitude_bits = int(dataset.get("amplitude_bits", np.full(candidate_count, 16))[candidate])
        ratio_delta = float(dataset.get("ratio_delta", np.zeros(candidate_count))[candidate])
        phase_ramp = float(dataset.get("phase_ramp_deg", np.zeros(candidate_count))[candidate])
        boundary_code = int(dataset.get("boundary_metric_code", np.zeros(candidate_count))[candidate])
        values = {
            "k_norm": k_value / KMAX,
            "active_ratio": float(np.mean(mask)),
            "num_active_norm": float(np.sum(mask)) / NUM_ELEMENTS,
            "eep_psll_db": nominal["psll_db"],
            "eep_iso_nearest_db": nominal["nearest_iso_db"],
            "eep_iso_local_db": nominal["local_iso_db"],
            "eep_peak_min_db": nominal["weakest_target_gain_db"],
            "eep_peak_spread_db": nominal["target_spread_db"],
            "energy_proxy": energy,
            # Frozen pre-HFSS substitution; no full-wave result is read here.
            "energy_normalized_hfss": energy
            / max(10.0 ** (nominal["weakest_target_gain_db"] / 10.0), 1.0e-20),
            "weight_l2": weight_l2,
            "max_channel_amplitude": max_amp,
            "amplitude_dynamic_range_db": dynamic_db,
            "condition_log10": 0.0,
            "null_constraint_count_norm": 0.0,
            "min_target_separation_deg_norm": min_separation(targets) / 180.0,
            "max_scan_theta_deg_norm": float(np.max(targets[:, 0])) / 90.0,
            "mean_scan_theta_deg_norm": float(np.mean(targets[:, 0])) / 90.0,
            "reference_hfss_peak_min_db": nominal["weakest_target_gain_db"],
            "reference_hfss_peak_spread_db": nominal["target_spread_db"],
            "nominal_worst_active_rl_db": nominal_rl,
            "nominal_total_rl_db": nominal_total_rl,
            "phase_error_rms_deg_norm": phase_rms / 35.0,
            "gain_error_rms_db_norm": gain_rms / 1.75,
            "dropout_fraction": dropout / max(float(np.sum(mask)), 1.0),
            "phase_bits_norm": phase_bits / 16.0,
            "amplitude_bits_norm": amplitude_bits / 16.0,
            "ratio_delta": ratio_delta,
            "phase_ramp_deg_norm": phase_ramp / 5.0,
            "implementation_delta_norm": float(dataset["implementation_delta_norm"][candidate]),
            "implementation_delta_max": float(dataset["implementation_delta_max"][candidate]),
            "task_mixing_amplitude": float(dataset["task_mixing_amplitude"][candidate]),
            "coherent_mode_amplitude": float(dataset["coherent_mode_amplitude"][candidate]),
            "boundary_psll_mode": float(boundary_code == 1),
            "boundary_nearest_mode": float(boundary_code == 2),
            "boundary_local_mode": float(boundary_code == 3),
        }
        scalar[candidate] = np.asarray([values[name] for name in scalar_names], dtype=np.float32)
        eep_metrics[candidate] = np.asarray(
            [
                nominal["psll_db"],
                nominal["nearest_iso_db"],
                nominal["local_iso_db"],
                nominal["weakest_target_gain_db"],
                nominal["target_spread_db"],
            ],
            dtype=np.float32,
        )
        actual_ri = np.asarray(dataset["hfss_actual_task_weights_real_imag"][candidate])
        actual_internal = actual_ri[:, :k_value, 0] + 1j * actual_ri[:, :k_value, 1]
        active = active_metrics(np.conjugate(actual_internal), mask, s_matrix)
        pre_hfss_active_gate[candidate] = int(active["robust_active_gate"])
        pre_hfss_worst_active_rl[candidate] = min(
            float(active["combined_worst_active_rl_db"]),
            float(active["task_significant_worst_active_rl_db"]),
        )
        pre_hfss_total_rl[candidate] = float(active["combined_total_rl_db"])

    spatial = spatial_features(dataset["masks"], nominal_tasks_ri, dataset["num_active"])
    spatial[:, 1:] = (
        spatial[:, 1:] - np.asarray(checkpoint["spatial_mean"], dtype=np.float32)
    ) / np.asarray(checkpoint["spatial_scale"], dtype=np.float32)
    targets_feature = target_features(dataset["targets_deg"], dataset["task_valid"])
    scalar_standard = (
        scalar - np.asarray(checkpoint["scalar_mean"], dtype=np.float32)
    ) / np.asarray(checkpoint["scalar_scale"], dtype=np.float32)
    model = FullwaveResidualCritic(
        int(checkpoint["scalar_dim"]),
        int(checkpoint["residual_dim"]),
        int(checkpoint["gate_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        mean, logvar, gate_logits, rank_score = model(
            torch.from_numpy(spatial),
            torch.from_numpy(targets_feature),
            torch.from_numpy(scalar_standard.astype(np.float32)),
        )
    residual_mean = (
        mean.numpy() * np.asarray(checkpoint["residual_scale"], dtype=np.float32)
        + np.asarray(checkpoint["residual_mean"], dtype=np.float32)
    )
    residual_sigma = np.exp(0.5 * logvar.numpy()) * np.asarray(
        checkpoint["residual_scale"], dtype=np.float32
    )
    logits = gate_logits.numpy()
    raw_probability = sigmoid(logits)
    probability = sigmoid(
        logits / np.asarray(checkpoint["temperatures"], dtype=np.float32)[None, :]
    )
    probability[:, 0] = apply_calibrator(raw_probability[:, 0], calibrators["gate15"])
    probability[:, 1] = apply_calibrator(raw_probability[:, 1], calibrators["gate20"])
    predicted = eep_metrics + residual_mean
    kappa = float(args.uncertainty_kappa)
    psll_ucb = predicted[:, 0] + kappa * residual_sigma[:, 0]
    nearest_lcb = predicted[:, 1] - kappa * residual_sigma[:, 1]
    local_lcb = predicted[:, 2] - kappa * residual_sigma[:, 2]
    main_drop = eep_metrics[:, 3] - predicted[:, 3]
    pattern15_admit = (
        (probability[:, 0] >= 0.5)
        & (psll_ucb <= 0.0)
        & (nearest_lcb >= 25.0)
        & (local_lcb >= 15.0)
    )
    strict_admit = (
        (probability[:, 1] >= 0.5)
        & (probability[:, 2] >= 0.5)
        & (probability[:, 3] >= 0.5)
        & (psll_ucb <= 0.0)
        & (nearest_lcb >= 25.0)
        & (local_lcb >= 20.0)
        & (main_drop <= 0.5)
        & (predicted[:, 4] <= 3.0)
        & (pre_hfss_active_gate == 1)
    )

    prediction_rows: list[dict[str, Any]] = []
    for candidate in range(candidate_count):
        prediction_rows.append(
            {
                "candidate_index": candidate,
                "sample_index": int(dataset["sample_indices"][candidate]),
                "scene_id": str(dataset["scene_ids"][candidate]),
                "target_hash": prospective_hashes[candidate],
                "boundary_type": str(dataset["boundary_type"][candidate]),
                "boundary_side_audit_only": str(dataset["boundary_side"][candidate]),
                "variant_kind": str(dataset["variant_kind"][candidate]),
                "k": int(dataset["k_values"][candidate]),
                "active_ratio": float(dataset["active_ratios_actual"][candidate]),
                "prob_gate15": float(probability[candidate, 0]),
                "prob_gate20": float(probability[candidate, 1]),
                "prob_mainlobe_gate": float(probability[candidate, 2]),
                "prob_strict_engineering_gate": float(probability[candidate, 3]),
                "pred_psll_db": float(predicted[candidate, 0]),
                "pred_psll_ucb_db": float(psll_ucb[candidate]),
                "pred_nearest_iso_db": float(predicted[candidate, 1]),
                "pred_nearest_iso_lcb_db": float(nearest_lcb[candidate]),
                "pred_local_iso_db": float(predicted[candidate, 2]),
                "pred_local_iso_lcb_db": float(local_lcb[candidate]),
                "pred_mainlobe_gain_db": float(predicted[candidate, 3]),
                "pred_mainlobe_drop_db": float(main_drop[candidate]),
                "pred_target_spread_db": float(predicted[candidate, 4]),
                "pre_hfss_worst_active_rl_db": float(pre_hfss_worst_active_rl[candidate]),
                "pre_hfss_total_rl_db": float(pre_hfss_total_rl[candidate]),
                "pre_hfss_active_gate": int(pre_hfss_active_gate[candidate]),
                "pattern15_admit": int(pattern15_admit[candidate]),
                "strict_admit": int(strict_admit[candidate]),
                "rank_score": float(rank_score.numpy()[candidate]),
            }
        )
    predictions_path = args.out_dir / "frozen_predictions_before_hfss.csv"
    write_csv(predictions_path, prediction_rows)

    by_scene: dict[int, list[int]] = defaultdict(list)
    for candidate, sample in enumerate(dataset["sample_indices"].tolist()):
        by_scene[int(sample)].append(candidate)
    selection_rows: list[dict[str, Any]] = []
    for sample, indices in sorted(by_scene.items()):
        scene = np.asarray(indices, dtype=np.int64)
        conservative_utility = (
            2.0 * probability[scene, 1]
            + probability[scene, 2]
            + probability[scene, 3]
            - np.maximum(psll_ucb[scene], 0.0)
            - 0.5 * np.maximum(25.0 - nearest_lcb[scene], 0.0)
            - 0.35 * np.maximum(20.0 - local_lcb[scene], 0.0)
            - 0.25 * np.maximum(main_drop[scene] - 0.5, 0.0)
            - 0.25 * (1 - pre_hfss_active_gate[scene])
        )
        choices = {
            "gate15_probability": int(scene[np.argmax(probability[scene, 0])]),
            "gate20_probability": int(scene[np.argmax(probability[scene, 1])]),
            "rank": int(scene[np.argmax(rank_score.numpy()[scene])]),
            "conservative": int(scene[np.argmax(conservative_utility)]),
        }
        for method, candidate in choices.items():
            selection_rows.append(
                {
                    "sample_index": sample,
                    "method": method,
                    "candidate_index": candidate,
                    "variant_kind": str(dataset["variant_kind"][candidate]),
                    "boundary_side_audit_only": str(dataset["boundary_side"][candidate]),
                    "prob_gate15": float(probability[candidate, 0]),
                    "prob_gate20": float(probability[candidate, 1]),
                    "strict_admit": int(strict_admit[candidate]),
                }
            )
    selections_path = args.out_dir / "frozen_scene_selections_before_hfss.csv"
    write_csv(selections_path, selection_rows)

    freeze = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prospective": True,
        "hfss_results_read": False,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_seed": int(checkpoint["seed"]),
        "calibrator_path": str(args.calibrator.resolve()),
        "calibrator_sha256": sha256(args.calibrator),
        "training_dataset_path": str(args.training_dataset.resolve()),
        "training_dataset_sha256": sha256(args.training_dataset),
        "prospective_dataset_path": str(dataset_path.resolve()),
        "prospective_dataset_sha256": sha256(dataset_path),
        "prediction_sha256": sha256(predictions_path),
        "selection_sha256": sha256(selections_path),
        "candidate_count": candidate_count,
        "independent_scene_count": len(by_scene),
        "unique_target_hash_count": len(set(prospective_hashes)),
        "training_target_overlap_count": len(overlap),
        "input_substitutions": {
            "energy_normalized_hfss": "energy / 10**(eep_peak_min_db/10)",
            "reference_hfss_peak_min_db": "same-scene nominal EEP peak",
            "reference_hfss_peak_spread_db": "same-scene nominal EEP spread",
        },
        "thresholds_frozen": {
            "psll_db_max": 0.0,
            "nearest_iso_db_min": 25.0,
            "local_iso_gate15_db_min": 15.0,
            "local_iso_gate20_db_min": 20.0,
            "mainlobe_drop_db_max": 0.5,
            "target_spread_db_max": 3.0,
            "pointing_error_deg_max": 1.5,
            "active_rl_db_min": 10.0,
            "probability_threshold": 0.5,
            "uncertainty_kappa": kappa,
        },
        "pattern15_admit_count": int(np.sum(pattern15_admit)),
        "strict_admit_count": int(np.sum(strict_admit)),
        "mainlobe_failures_added_for_training": 0,
        "retraining_allowed_after_hfss": False,
        "freeze_pass": bool(
            len(overlap) == 0
            and len(set(prospective_hashes)) == len(by_scene)
            and candidate_count == 3 * len(by_scene)
        ),
    }
    (args.out_dir / "prospective_freeze_manifest.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
