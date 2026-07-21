"""Build and evaluate a 16x16 local-full-wave coupling proxy from converged S16.

The output is a passive reciprocal circuit proxy derived from a converged 4x4
HFSS model.  It is suitable for pretraining and candidate screening, but is
never labeled as a full 16x16 HFSS result.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_4x4_20260717_run01"
    / "grounded_patch_4x4"
    / "matched_s_10ghz.npz"
)
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_OUT = ROOT / "hfss_outputs" / "grounded_patch_s256_proxy_20260717_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-side", type=int, default=4)
    parser.add_argument("--target-side", type=int, default=16)
    parser.add_argument("--max-offset", type=int, default=3)
    parser.add_argument("--return-loss-db", type=float, default=10.0)
    parser.add_argument("--significant-power-relative-db", type=float, default=-30.0)
    parser.add_argument("--large-scan-theta-deg", type=float, default=45.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def port_class(ix: int, iy: int, side: int) -> str:
    x_edge = ix in (0, side - 1)
    y_edge = iy in (0, side - 1)
    return "corner" if x_edge and y_edge else ("edge" if x_edge or y_edge else "interior")


def displacement_kernel(z: np.ndarray, side: int, max_offset: int) -> dict[tuple[int, int], complex]:
    values: dict[tuple[int, int], list[complex]] = {}
    for first in range(side * side):
        x1, y1 = divmod(first, side)
        for second in range(side * side):
            x2, y2 = divmod(second, side)
            dx, dy = x1 - x2, y1 - y2
            if first != second and abs(dx) <= max_offset and abs(dy) <= max_offset:
                values.setdefault((dx, dy), []).append(complex(z[first, second]))
    kernel = {offset: complex(np.mean(samples)) for offset, samples in values.items()}
    for dx, dy in list(kernel):
        reverse = (-dx, -dy)
        if reverse in kernel:
            average = 0.5 * (kernel[(dx, dy)] + kernel[reverse])
            kernel[(dx, dy)] = average
            kernel[reverse] = average
    return kernel


def synthesize_z(z_source: np.ndarray, source_side: int, target_side: int, max_offset: int) -> tuple[np.ndarray, dict[str, Any]]:
    kernel = displacement_kernel(z_source, source_side, max_offset)
    source_diagonal: dict[str, complex] = {}
    for category in ("corner", "edge", "interior"):
        indices = [
            index
            for index in range(source_side * source_side)
            if port_class(*divmod(index, source_side), source_side) == category
        ]
        source_diagonal[category] = complex(np.mean(np.diag(z_source)[indices]))

    count = target_side * target_side
    z_target = np.zeros((count, count), dtype=np.complex128)
    for first in range(count):
        x1, y1 = divmod(first, target_side)
        z_target[first, first] = source_diagonal[port_class(x1, y1, target_side)]
        for second in range(first):
            x2, y2 = divmod(second, target_side)
            coupling = kernel.get((x1 - x2, y1 - y2), 0.0j)
            z_target[first, second] = coupling
            z_target[second, first] = coupling

    z_target = 0.5 * (z_target + z_target.T)
    resistance = 0.5 * (z_target + z_target.conj().T)
    reactance = (z_target - z_target.conj().T) / (2.0j)
    eigenvalues, eigenvectors = np.linalg.eigh(resistance)
    scale = max(float(np.max(eigenvalues)), 1.0)
    floor = 1.0e-8 * scale
    clipped = np.maximum(eigenvalues, floor)
    resistance_psd = (eigenvectors * clipped) @ eigenvectors.conj().T
    z_passive = resistance_psd + 1j * reactance
    z_passive = 0.5 * (z_passive + z_passive.T)
    diagnostics = {
        "kernel_term_count": len(kernel),
        "kernel_max_offset": max_offset,
        "resistance_min_eigenvalue_before": float(np.min(eigenvalues)),
        "resistance_min_eigenvalue_after": float(np.min(clipped)),
        "passivity_projection_relative_norm": float(
            np.linalg.norm(z_passive - z_target) / max(np.linalg.norm(z_target), 1.0e-15)
        ),
        "source_diagonal_ohm": {
            name: {"real": float(value.real), "imag": float(value.imag)}
            for name, value in source_diagonal.items()
        },
    }
    return z_passive, diagnostics


def z_to_s(z: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(z.shape[0], dtype=np.complex128)
    return (z - z0 * identity) @ np.linalg.inv(z + z0 * identity)


def evaluate_cases(
    s: np.ndarray,
    weights: np.ndarray,
    masks: np.ndarray,
    dataset: np.lib.npyio.NpzFile,
    threshold_db: float,
    significant_relative_db: float,
    large_scan_theta_deg: float,
) -> list[dict[str, Any]]:
    reflected = weights @ s.T
    incident_power = np.sum(np.abs(weights) ** 2, axis=1)
    reflected_power = np.sum(np.abs(reflected) ** 2, axis=1)
    total_rl = -10.0 * np.log10(np.maximum(reflected_power / incident_power, 1.0e-30))
    significant_ratio = 10.0 ** (significant_relative_db / 10.0)
    rows: list[dict[str, Any]] = []
    for index in range(weights.shape[0]):
        amplitudes = np.abs(weights[index])
        active = masks[index] & (amplitudes > 1.0e-10)
        maximum_power = max(float(np.max(amplitudes**2)), 1.0e-30)
        significant = active & ((amplitudes**2) >= significant_ratio * maximum_power)
        gamma = np.full(256, np.nan, dtype=np.float64)
        gamma[active] = np.abs(reflected[index, active] / weights[index, active])
        active_rl = -20.0 * np.log10(np.maximum(gamma[active], 1.0e-15))
        significant_rl = -20.0 * np.log10(np.maximum(gamma[significant], 1.0e-15))
        k_value = int(dataset["k_values"][index])
        target_theta = np.asarray(dataset["targets_deg"][index, :k_value, 0], dtype=float)
        worst_active = float(np.min(active_rl)) if active_rl.size else float("nan")
        rows.append(
            {
                "sample_index": index,
                "sample_id": str(dataset["sample_ids"][index]),
                "k": k_value,
                "ratio_requested": float(dataset["active_ratios_requested"][index]),
                "ratio_actual": float(dataset["active_ratios_actual"][index]),
                "num_active": int(dataset["num_active"][index]),
                "max_target_theta_deg": float(np.max(target_theta)),
                "large_scan": int(float(np.max(target_theta)) >= large_scan_theta_deg),
                "worst_active_rl_db": worst_active,
                "worst_significant_rl_db": float(np.min(significant_rl)) if significant_rl.size else float("nan"),
                "total_return_loss_db": float(total_rl[index]),
                "accepted_power_fraction": float(1.0 - reflected_power[index] / incident_power[index]),
                "active_port_pass_fraction": float(np.mean(active_rl >= threshold_db)),
                "all_active_ports_pass_10db": int(bool(active_rl.size and np.all(active_rl >= threshold_db))),
                "all_significant_ports_pass_10db": int(
                    bool(significant_rl.size and np.all(significant_rl >= threshold_db))
                ),
                "total_return_pass_10db": int(total_rl[index] >= threshold_db),
                "proxy_engineering_gate": int(
                    bool(active_rl.size and np.all(active_rl >= threshold_db) and total_rl[index] >= threshold_db)
                ),
                "near_boundary": int(7.0 <= worst_active <= 13.0),
            }
        )
    return rows


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        keys = [
            ("all", "all", "all"),
            (str(row["k"]), "all", "all"),
            (str(row["k"]), f"{float(row['ratio_requested']):.1f}", "all"),
            (str(row["k"]), f"{float(row['ratio_requested']):.1f}", str(row["large_scan"])),
        ]
        for key in keys:
            groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (k_value, ratio, large_scan), members in sorted(groups.items()):
        worst = np.asarray([float(item["worst_active_rl_db"]) for item in members])
        total = np.asarray([float(item["total_return_loss_db"]) for item in members])
        summaries.append(
            {
                "k": k_value,
                "ratio_requested": ratio,
                "large_scan": large_scan,
                "case_count": len(members),
                "worst_active_rl_min_db": float(np.min(worst)),
                "worst_active_rl_p05_db": float(np.quantile(worst, 0.05)),
                "worst_active_rl_mean_db": float(np.mean(worst)),
                "total_rl_min_db": float(np.min(total)),
                "total_rl_mean_db": float(np.mean(total)),
                "all_active_10db_pass_rate": float(
                    np.mean([int(item["all_active_ports_pass_10db"]) for item in members])
                ),
                "total_10db_pass_rate": float(
                    np.mean([int(item["total_return_pass_10db"]) for item in members])
                ),
                "proxy_engineering_gate_pass_rate": float(
                    np.mean([int(item["proxy_engineering_gate"]) for item in members])
                ),
                "near_boundary_count": int(sum(int(item["near_boundary"]) for item in members)),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    source = np.load(args.source, allow_pickle=False)
    dataset = np.load(args.dataset, allow_pickle=False)
    source_side = int(args.source_side)
    target_side = int(args.target_side)
    if source["z_matched"].shape != (source_side**2, source_side**2):
        raise ValueError("Source Z matrix does not match source-side")
    if target_side != 16 or dataset["masks"].shape[1] != target_side**2:
        raise ValueError("This evaluator expects the existing 16x16 dataset")

    z_proxy, diagnostics = synthesize_z(
        np.asarray(source["z_matched"], dtype=np.complex128),
        source_side,
        target_side,
        int(args.max_offset),
    )
    z0 = 50.0
    s_proxy = z_to_s(z_proxy, z0)
    reciprocity = float(np.max(np.abs(s_proxy - s_proxy.T)))
    sigma_max = float(np.max(np.linalg.svd(s_proxy, compute_uv=False)))

    weights_ri = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float64)
    weights = weights_ri[:, :, 0] + 1j * weights_ri[:, :, 1]
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1.0e-15)
    masks = np.asarray(dataset["masks"], dtype=bool)
    rows = evaluate_cases(
        s_proxy,
        weights,
        masks,
        dataset,
        float(args.return_loss_db),
        float(args.significant_power_relative_db),
        float(args.large_scan_theta_deg),
    )
    summaries = group_summary(rows)
    overall = next(item for item in summaries if item["k"] == "all")
    write_csv(args.out_dir / "active_return_cases_2400.csv", rows)
    write_csv(args.out_dir / "active_return_group_summary.csv", summaries)
    np.savez_compressed(
        args.out_dir / "grounded_patch_s256_local_fullwave_proxy.npz",
        s_parameters=s_proxy.astype(np.complex64),
        z_parameters=z_proxy.astype(np.complex64),
        antenna_incident_wave_map=np.eye(target_side**2, dtype=np.complex64),
        frequency_hz=np.asarray(10.0e9),
        reference_impedance_ohm=np.asarray(z0),
        port_names=np.asarray([f"P{index:03d}" for index in range(target_side**2)]),
        source_s16_path=np.asarray(str(args.source)),
        model_label=np.asarray("local_fullwave_kernel_proxy_not_full_16x16_hfss"),
    )
    np.savez_compressed(
        args.out_dir / "active_return_proxy_pretraining_dataset.npz",
        sample_index=np.arange(weights.shape[0], dtype=np.int64),
        masks=masks.astype(np.int8),
        weights_real_imag=np.stack((weights.real, weights.imag), axis=-1).astype(np.float32),
        k_values=np.asarray(dataset["k_values"], dtype=np.int64),
        ratios=np.asarray(dataset["active_ratios_requested"], dtype=np.float32),
        targets_deg=np.asarray(dataset["targets_deg"], dtype=np.float32),
        worst_active_rl_db=np.asarray([row["worst_active_rl_db"] for row in rows], dtype=np.float32),
        total_return_loss_db=np.asarray([row["total_return_loss_db"] for row in rows], dtype=np.float32),
        proxy_gate=np.asarray([row["proxy_engineering_gate"] for row in rows], dtype=np.int8),
        near_boundary=np.asarray([row["near_boundary"] for row in rows], dtype=np.int8),
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_label": "local_fullwave_kernel_proxy_not_full_16x16_hfss",
        "source": str(args.source),
        "source_gate": "converged 4x4 HFSS direct-port matched model",
        "target_shape": [target_side, target_side],
        "reciprocity_max_abs": reciprocity,
        "passivity_sigma_max": sigma_max,
        **diagnostics,
        "case_count": len(rows),
        "overall": overall,
        "proxy_gate_pass_count": int(sum(int(row["proxy_engineering_gate"]) for row in rows)),
        "near_boundary_count": int(sum(int(row["near_boundary"]) for row in rows)),
        "training_decision": (
            "allow_proxy_pretraining_only"
            if int(sum(int(row["proxy_engineering_gate"]) for row in rows)) > 0
            else "block_proxy_pretraining_and_revisit_matching"
        ),
        "limitations": [
            "The S256 matrix is synthesized from a converged local S16 impedance kernel.",
            "It is not a full 16x16 HFSS matrix and cannot certify final engineering active return loss.",
            "Final labels still require DDM/HPC full-array HFSS or measured hardware validation.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
