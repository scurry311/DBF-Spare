"""Compare original and active-RL-projected combined array-factor patterns."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_PROJECTED = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_active_rl_projection_20260717_run04_prune12_strong"
    / "projected_source_weights.npz"
)
DEFAULT_ACTIVE_METRICS = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_active_rl_projection_20260717_run04_prune12_strong"
    / "projected_active_return_case_metrics.csv"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "grounded_patch_projected_af_20260717_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--projected", type=Path, default=DEFAULT_PROJECTED)
    parser.add_argument("--active-metrics", type=Path, default=DEFAULT_ACTIVE_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--theta-step", type=float, default=2.0)
    parser.add_argument("--phi-step", type=float, default=5.0)
    parser.add_argument("--sidelobe-exclusion-deg", type=float, default=8.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unit_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
    ).astype(np.float32)


def make_grid(theta_step: float, phi_step: float) -> np.ndarray:
    theta_values = np.arange(0.0, 90.0 + 0.1, theta_step, dtype=np.float32)
    phi_values = np.arange(0.0, 360.0, phi_step, dtype=np.float32)
    theta, phi = np.meshgrid(theta_values, phi_values, indexing="ij")
    return unit_vectors(theta.reshape(-1), phi.reshape(-1))


def steering(positions: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return np.exp(-1j * 2.0 * np.pi * (directions @ positions.T)).astype(np.complex64)


def pattern_metrics(
    weights: np.ndarray,
    positions: np.ndarray,
    targets: np.ndarray,
    grid_dirs: np.ndarray,
    grid_steering: np.ndarray,
    exclusion_deg: float,
) -> dict[str, float]:
    target_dirs = unit_vectors(targets[:, 0].astype(np.float32), targets[:, 1].astype(np.float32))
    target_response = steering(positions, target_dirs) @ weights
    target_db = 10.0 * np.log10(np.maximum(np.abs(target_response) ** 2, 1.0e-12))
    grid_response = grid_steering @ weights
    grid_db = 10.0 * np.log10(np.maximum(np.abs(grid_response) ** 2, 1.0e-12))
    angular_distance = np.rad2deg(
        np.arccos(np.clip(grid_dirs @ target_dirs.T, -1.0, 1.0))
    )
    sidelobe = angular_distance.min(axis=1) > exclusion_deg
    side_max = float(np.max(grid_db[sidelobe])) if np.any(sidelobe) else float("nan")
    return {
        "psll_to_weakest_peak_db": side_max - float(np.min(target_db)),
        "weakest_target_db": float(np.min(target_db)),
        "target_spread_db": float(np.max(target_db) - np.min(target_db)),
        "weight_norm_sq": float(np.vdot(weights, weights).real),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for key in (("all", "all"), (str(row["k"]), "all"), (str(row["k"]), str(row["rf_gate_pass"]))):
            groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (k_value, rf_gate), members in sorted(groups.items()):
        delta_psll = np.asarray([float(item["delta_psll_db"]) for item in members])
        peak_loss = np.asarray([float(item["weakest_target_loss_db"]) for item in members])
        output.append(
            {
                "k": k_value,
                "rf_gate_pass": rf_gate,
                "case_count": len(members),
                "original_psll_mean_db": float(np.mean([float(item["original_psll_db"]) for item in members])),
                "projected_psll_mean_db": float(np.mean([float(item["projected_psll_db"]) for item in members])),
                "delta_psll_mean_db": float(np.mean(delta_psll)),
                "delta_psll_p95_db": float(np.quantile(delta_psll, 0.95)),
                "psll_not_worse_0p5db_rate": float(np.mean(delta_psll <= 0.5)),
                "weakest_target_loss_mean_db": float(np.mean(peak_loss)),
                "weakest_target_loss_p95_db": float(np.quantile(peak_loss, 0.95)),
                "mainlobe_loss_le_0p5db_rate": float(np.mean(peak_loss <= 0.5)),
                "projected_spread_le_3db_rate": float(
                    np.mean([float(item["projected_target_spread_db"]) <= 3.0 for item in members])
                ),
                "combined_af_screen_pass_rate": float(
                    np.mean([int(item["combined_af_screen_pass"]) for item in members])
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    dataset = np.load(args.dataset, allow_pickle=False)
    projected = np.load(args.projected, allow_pickle=False)
    original_ri = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float32)
    original = original_ri[..., 0] + 1j * original_ri[..., 1]
    original /= np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1.0e-15)
    projected_ri = np.asarray(projected["weights_real_imag"], dtype=np.float32)
    projected_weights = projected_ri[..., 0] + 1j * projected_ri[..., 1]
    if original.shape != projected_weights.shape:
        raise ValueError("Original and projected combined weights do not align")
    positions = np.asarray(dataset["positions_lambda"], dtype=np.float32)
    grid_dirs = make_grid(float(args.theta_step), float(args.phi_step))
    grid_steering = steering(positions, grid_dirs)
    with args.active_metrics.open("r", newline="", encoding="utf-8-sig") as handle:
        active_rows = {int(row["sample_index"]): row for row in csv.DictReader(handle)}
    rows: list[dict[str, Any]] = []
    for index in range(original.shape[0]):
        k_value = int(dataset["k_values"][index])
        targets = np.asarray(dataset["targets_deg"][index, :k_value], dtype=np.float32)
        base = pattern_metrics(
            original[index], positions, targets, grid_dirs, grid_steering, float(args.sidelobe_exclusion_deg)
        )
        candidate = pattern_metrics(
            projected_weights[index], positions, targets, grid_dirs, grid_steering, float(args.sidelobe_exclusion_deg)
        )
        delta_psll = candidate["psll_to_weakest_peak_db"] - base["psll_to_weakest_peak_db"]
        weakest_loss = base["weakest_target_db"] - candidate["weakest_target_db"]
        rf_pass = int(active_rows[index]["engineering_10db_gate_pass"])
        rows.append(
            {
                "sample_index": index,
                "sample_id": str(dataset["sample_ids"][index]),
                "k": k_value,
                "projected_ratio": float(np.mean(np.asarray(projected["masks"][index], dtype=bool))),
                "rf_gate_pass": rf_pass,
                "original_psll_db": base["psll_to_weakest_peak_db"],
                "projected_psll_db": candidate["psll_to_weakest_peak_db"],
                "delta_psll_db": delta_psll,
                "original_weakest_target_db": base["weakest_target_db"],
                "projected_weakest_target_db": candidate["weakest_target_db"],
                "weakest_target_loss_db": weakest_loss,
                "original_target_spread_db": base["target_spread_db"],
                "projected_target_spread_db": candidate["target_spread_db"],
                "original_weight_norm_sq": base["weight_norm_sq"],
                "projected_weight_norm_sq": candidate["weight_norm_sq"],
                "combined_af_screen_pass": int(
                    candidate["psll_to_weakest_peak_db"] <= 0.0
                    and weakest_loss <= 0.5
                    and candidate["target_spread_db"] <= 3.0
                ),
            }
        )
    summary_rows = summarize(rows)
    write_csv(args.out_dir / "paired_combined_af_cases.csv", rows)
    write_csv(args.out_dir / "paired_combined_af_summary.csv", summary_rows)
    overall = next(item for item in summary_rows if item["k"] == "all")
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "combined array-factor paired check only",
        "case_count": len(rows),
        "grid": {
            "theta_step_deg": float(args.theta_step),
            "phi_step_deg": float(args.phi_step),
            "sidelobe_exclusion_deg": float(args.sidelobe_exclusion_deg),
        },
        "overall": overall,
        "joint_proxy_rf_and_combined_af_pass_count": int(
            sum(int(row["rf_gate_pass"]) and int(row["combined_af_screen_pass"]) for row in rows)
        ),
        "limitations": [
            "This is an AF calculation, not HFSS full-wave pattern validation.",
            "Projected weights are combined source weights; task-level isolation is not available.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
