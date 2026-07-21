"""Project multi-beam source weights toward full-S active-return feasibility."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_NETWORK = (
    DATASET_ROOT
    / "full_s256p_matched_v2_20260714"
    / "port_class_matching_20260714"
    / "port_class_matched_s256.npz"
)
DEFAULT_DATASET = DATASET_ROOT / "dataset_arrays.npz"
DEFAULT_OUT = (
    DATASET_ROOT
    / "full_s256p_matched_v2_20260714"
    / "active_return_projection_20260714"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--active-penalty", type=float, default=5.0)
    parser.add_argument("--total-penalty", type=float, default=2.0)
    parser.add_argument("--proximity-penalty", type=float, default=0.02)
    parser.add_argument("--return-loss-min-db", type=float, default=10.0)
    parser.add_argument("--large-scan-theta-deg", type=float, default=45.0)
    parser.add_argument("--significant-power-relative-db", type=float, default=-30.0)
    parser.add_argument(
        "--prune-amplitude-relative-db",
        type=float,
        help="Turn mask-on channels below this level relative to the case maximum fully off before projection.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_effective_target_rows(
    targets_deg: np.ndarray,
    k_values: np.ndarray,
    positions_lambda: np.ndarray,
    antenna_wave_map: np.ndarray,
) -> np.ndarray:
    case_count = targets_deg.shape[0]
    max_tasks = targets_deg.shape[1]
    rows = np.zeros((case_count, max_tasks, positions_lambda.shape[0]), dtype=np.complex128)
    for case_index in range(case_count):
        for task_index in range(int(k_values[case_index])):
            theta_deg, phi_deg = targets_deg[case_index, task_index]
            theta = math.radians(float(theta_deg))
            phi = math.radians(float(phi_deg))
            direction = np.asarray(
                [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
                dtype=np.float64,
            )
            antenna_row = np.exp(1j * 2.0 * math.pi * (positions_lambda @ direction))
            rows[case_index, task_index] = antenna_row @ antenna_wave_map
    return rows


def restore_target_responses(
    weights: np.ndarray,
    target_rows: np.ndarray,
    desired: np.ndarray,
    k_values: np.ndarray,
    masks: np.ndarray,
) -> None:
    for case_index in range(weights.shape[0]):
        active = np.flatnonzero(masks[case_index])
        task_count = int(k_values[case_index])
        matrix = target_rows[case_index, :task_count][:, active]
        residual = desired[case_index, :task_count] - matrix @ weights[case_index, active]
        gram = matrix @ matrix.conj().T
        regularization = 1.0e-8 * max(float(np.trace(gram).real / max(task_count, 1)), 1.0)
        correction = matrix.conj().T @ np.linalg.solve(
            gram + regularization * np.eye(task_count, dtype=np.complex128), residual
        )
        weights[case_index, active] += correction
        weights[case_index, ~masks[case_index]] = 0.0


def evaluate_iteration(
    weights: np.ndarray,
    s_matrix: np.ndarray,
    command_masks: np.ndarray,
    rho: float,
    significant_ratio: float,
) -> dict[str, np.ndarray | float]:
    reflected = weights @ s_matrix.T
    amplitude = np.abs(weights)
    source_active = command_masks & (amplitude > 1.0e-10)
    gamma = np.full(weights.shape, np.nan, dtype=np.float64)
    gamma[source_active] = np.abs(reflected[source_active] / weights[source_active])
    max_power = np.max(amplitude**2, axis=1, keepdims=True)
    significant = source_active & ((amplitude**2) >= significant_ratio * np.maximum(max_power, 1.0e-30))
    worst = np.full(weights.shape[0], np.nan, dtype=np.float64)
    worst_significant = np.full(weights.shape[0], np.nan, dtype=np.float64)
    active_pass = np.zeros(weights.shape[0], dtype=bool)
    significant_pass = np.zeros(weights.shape[0], dtype=bool)
    for case_index in range(weights.shape[0]):
        values = gamma[case_index, source_active[case_index]]
        significant_values = gamma[case_index, significant[case_index]]
        if values.size:
            worst[case_index] = -20.0 * math.log10(max(float(values.max()), 1.0e-15))
            active_pass[case_index] = bool(np.all(values <= rho))
        if significant_values.size:
            worst_significant[case_index] = -20.0 * math.log10(
                max(float(significant_values.max()), 1.0e-15)
            )
            significant_pass[case_index] = bool(np.all(significant_values <= rho))
    incident_power = np.sum(np.abs(weights) ** 2, axis=1)
    reflected_power = np.sum(np.abs(reflected) ** 2, axis=1)
    total_rl = -10.0 * np.log10(np.maximum(reflected_power / incident_power, 1.0e-30))
    score = np.nan_to_num(worst_significant, nan=-300.0) + 0.15 * total_rl
    return {
        "reflected": reflected,
        "worst": worst,
        "worst_significant": worst_significant,
        "active_pass": active_pass,
        "significant_pass": significant_pass,
        "total_rl": total_rl,
        "score": score,
        "all_active_pass_rate": float(np.mean(active_pass)),
        "all_significant_pass_rate": float(np.mean(significant_pass)),
        "total_pass_rate": float(np.mean(total_rl >= -20.0 * math.log10(rho))),
    }


def project_single_source_weights(
    initial: np.ndarray,
    command_mask: np.ndarray,
    s_matrix: np.ndarray,
    target_rows: np.ndarray,
    *,
    iterations: int = 50,
    step_size: float = 0.04,
    active_penalty: float = 12.0,
    total_penalty: float = 1.0,
    proximity_penalty: float = 0.01,
    return_loss_min_db: float = 10.0,
    hard_gate_priority: bool = True,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    rho = 10.0 ** (-float(return_loss_min_db) / 20.0)
    mask = np.asarray(command_mask, dtype=bool)
    start = np.asarray(initial, dtype=np.complex128).copy()
    start[~mask] = 0.0
    start /= max(float(np.linalg.norm(start)), 1.0e-15)
    desired = target_rows @ start
    weights = start.copy()
    best = start.copy()
    best_score: tuple[int, float, float] | float = (
        (-1, -float("inf"), -float("inf")) if hard_gate_priority else -float("inf")
    )
    amplitude_floor = 0.15 * np.abs(start)
    for _iteration in range(int(iterations) + 1):
        reflected = s_matrix @ weights
        source_active = mask & (np.abs(weights) > 1.0e-10)
        gamma = np.abs(reflected[source_active] / weights[source_active])
        worst_rl = -20.0 * math.log10(max(float(np.max(gamma)), 1.0e-15))
        total_rl = -10.0 * math.log10(
            max(float(np.vdot(reflected, reflected).real / np.vdot(weights, weights).real), 1.0e-30)
        )
        score = (
            (
                int(worst_rl >= float(return_loss_min_db) and total_rl >= float(return_loss_min_db)),
                min(worst_rl, total_rl),
                worst_rl + 0.15 * total_rl,
            )
            if hard_gate_priority
            else worst_rl + 0.15 * total_rl
        )
        if score > best_score:
            best_score = score
            best = weights.copy()
        if _iteration == int(iterations):
            break
        amplitude = np.abs(weights)
        reflected_amplitude = np.abs(reflected)
        phase_reflected = reflected / np.maximum(reflected_amplitude, 1.0e-15)
        phase_weight = weights / np.maximum(amplitude, 1.0e-15)
        violation = np.maximum(reflected_amplitude - rho * amplitude, 0.0) * mask
        gradient_active = s_matrix.conj().T @ (violation * phase_reflected)
        gradient_active -= rho * violation * phase_weight
        weight_norm = max(float(np.linalg.norm(weights)), 1.0e-15)
        reflected_norm = max(float(np.linalg.norm(reflected)), 1.0e-15)
        total_violation = max(reflected_norm - rho * weight_norm, 0.0)
        gradient_total = total_violation * (
            s_matrix.conj().T @ (reflected / reflected_norm) - rho * weights / weight_norm
        )
        gradient = (
            float(active_penalty) * gradient_active
            + float(total_penalty) * gradient_total
            + float(proximity_penalty) * (weights - start)
        )
        gradient[~mask] = 0.0
        weights -= float(step_size) * weight_norm * gradient / max(float(np.linalg.norm(gradient)), 1.0e-15)
        low = mask & (np.abs(weights) < amplitude_floor)
        weights[low] = amplitude_floor[low] * phase_weight[low]
        weights[~mask] = 0.0
        active = np.flatnonzero(mask)
        matrix = target_rows[:, active]
        residual = desired - matrix @ weights[active]
        gram = matrix @ matrix.conj().T
        regularization = 1.0e-8 * max(float(np.trace(gram).real / max(matrix.shape[0], 1)), 1.0)
        weights[active] += matrix.conj().T @ np.linalg.solve(
            gram + regularization * np.eye(matrix.shape[0], dtype=np.complex128), residual
        )
    weights = best / max(float(np.linalg.norm(best)), 1.0e-15)
    desired_scaled = desired / max(float(np.linalg.norm(best)), 1.0e-15)
    reflected = s_matrix @ weights
    source_active = mask & (np.abs(weights) > 1.0e-10)
    gamma = np.abs(reflected[source_active] / weights[source_active])
    worst_rl = -20.0 * math.log10(max(float(np.max(gamma)), 1.0e-15))
    total_rl = -10.0 * math.log10(
        max(float(np.vdot(reflected, reflected).real / np.vdot(weights, weights).real), 1.0e-30)
    )
    target_error = float(
        np.max(np.abs(target_rows @ weights - desired_scaled) / np.maximum(np.abs(desired_scaled), 1.0e-9))
    )
    all_active_pass = bool(np.all(gamma <= rho))
    total_pass = bool(total_rl >= float(return_loss_min_db))
    return weights.astype(np.complex64), {
        "worst_active_return_loss_db": worst_rl,
        "total_return_loss_db": total_rl,
        "all_active_10db_pass": all_active_pass,
        "total_10db_pass": total_pass,
        "engineering_10db_gate_pass": bool(all_active_pass and total_pass),
        "target_response_error_max": target_error,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for key in (
            ("all", "all", "all"),
            (str(row["k"]), "all", "all"),
            (str(row["k"]), "all", str(row["large_scan"])),
            (str(row["k"]), f"{row['active_ratio']:.1f}", str(row["large_scan"])),
        ):
            groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (k_value, ratio, large_scan), group in sorted(groups.items()):
        worst = np.asarray([float(row["worst_active_return_loss_db"]) for row in group])
        significant = np.asarray([float(row["worst_significant_return_loss_db"]) for row in group])
        total = np.asarray([float(row["total_return_loss_db"]) for row in group])
        output.append(
            {
                "k": k_value,
                "active_ratio": ratio,
                "large_scan": large_scan,
                "case_count": len(group),
                "worst_active_rl_min_db": float(np.nanmin(worst)),
                "worst_active_rl_p05_db": float(np.nanquantile(worst, 0.05)),
                "worst_active_rl_mean_db": float(np.nanmean(worst)),
                "worst_significant_rl_min_db": float(np.nanmin(significant)),
                "worst_significant_rl_mean_db": float(np.nanmean(significant)),
                "total_rl_min_db": float(np.nanmin(total)),
                "total_rl_mean_db": float(np.nanmean(total)),
                "all_active_10db_pass_rate": float(np.mean([int(row["all_active_10db_pass"]) for row in group])),
                "all_significant_10db_pass_rate": float(
                    np.mean([int(row["all_significant_10db_pass"]) for row in group])
                ),
                "total_10db_pass_rate": float(np.mean([int(row["total_10db_pass"]) for row in group])),
                "engineering_10db_gate_pass_rate": float(
                    np.mean([int(row["engineering_10db_gate_pass"]) for row in group])
                ),
                "target_response_error_max": float(
                    np.max([float(row["target_response_error_max"]) for row in group])
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    network = np.load(args.network, allow_pickle=False)
    s_matrix = np.asarray(network["s_parameters"], dtype=np.complex128)
    antenna_wave_map = np.asarray(network["antenna_incident_wave_map"], dtype=np.complex128)
    network_ports = [str(port) for port in network["port_names"]]
    dataset = np.load(args.dataset, allow_pickle=False)
    dataset_ports = [str(port) for port in dataset["port_names"]]
    reorder = [dataset_ports.index(port) for port in network_ports]
    weights_ri = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float64)
    initial = (weights_ri[:, :, 0] + 1j * weights_ri[:, :, 1])[:, reorder]
    initial /= np.maximum(np.linalg.norm(initial, axis=1, keepdims=True), 1.0e-15)
    command_masks = np.asarray(dataset["masks"], dtype=bool)[:, reorder]
    initial[~command_masks] = 0.0
    k_values = np.asarray(dataset["k_values"], dtype=int)
    ratios = np.asarray(dataset["active_ratios_actual"], dtype=float)
    targets = np.asarray(dataset["targets_deg"], dtype=float)
    positions = np.asarray(dataset["positions_lambda"], dtype=np.float64)[reorder]
    sample_ids = [str(value) for value in dataset["sample_ids"]]
    target_rows = build_effective_target_rows(targets, k_values, positions, antenna_wave_map)
    desired = np.einsum("nkp,np->nk", target_rows, initial)
    original_command_masks = command_masks.copy()
    original_active_counts = np.sum(original_command_masks, axis=1)
    if args.prune_amplitude_relative_db is not None:
        relative_floor = 10.0 ** (float(args.prune_amplitude_relative_db) / 20.0)
        case_maximum = np.max(np.abs(initial), axis=1, keepdims=True)
        command_masks &= np.abs(initial) >= relative_floor * np.maximum(case_maximum, 1.0e-15)
        for case_index in range(command_masks.shape[0]):
            minimum_keep = max(2 * int(k_values[case_index]), 8)
            if int(np.sum(command_masks[case_index])) < minimum_keep:
                available = np.flatnonzero(original_command_masks[case_index])
                strongest = available[np.argsort(np.abs(initial[case_index, available]))[-minimum_keep:]]
                command_masks[case_index, strongest] = True
        initial[~command_masks] = 0.0
        restore_target_responses(initial, target_rows, desired, k_values, command_masks)
    effective_active_counts = np.sum(command_masks, axis=1)
    ratios = effective_active_counts.astype(np.float64) / command_masks.shape[1]

    rho = 10.0 ** (-float(args.return_loss_min_db) / 20.0)
    significant_ratio = 10.0 ** (float(args.significant_power_relative_db) / 10.0)
    weights = initial.copy()
    best_weights = initial.copy()
    initial_metrics = evaluate_iteration(weights, s_matrix, command_masks, rho, significant_ratio)
    best_score = np.asarray(initial_metrics["score"], dtype=np.float64).copy()
    history_rows: list[dict[str, Any]] = []
    initial_amplitude = np.abs(initial)

    for iteration in range(int(args.iterations) + 1):
        metrics = evaluate_iteration(weights, s_matrix, command_masks, rho, significant_ratio)
        current_score = np.asarray(metrics["score"], dtype=np.float64)
        improved = current_score > best_score
        best_weights[improved] = weights[improved]
        best_score[improved] = current_score[improved]
        target_response = np.einsum("nkp,np->nk", target_rows, weights)
        valid_task = np.arange(target_rows.shape[1])[None, :] < k_values[:, None]
        target_error = np.abs(target_response - desired) / np.maximum(np.abs(desired), 1.0e-9)
        history_rows.append(
            {
                "iteration": iteration,
                "all_active_10db_pass_rate": metrics["all_active_pass_rate"],
                "all_significant_10db_pass_rate": metrics["all_significant_pass_rate"],
                "total_10db_pass_rate": metrics["total_pass_rate"],
                "worst_active_rl_min_db": float(np.nanmin(metrics["worst"])),
                "worst_significant_rl_min_db": float(np.nanmin(metrics["worst_significant"])),
                "total_rl_mean_db": float(np.mean(metrics["total_rl"])),
                "target_response_error_max": float(np.max(target_error[valid_task])),
            }
        )
        if iteration == int(args.iterations):
            break

        reflected = np.asarray(metrics["reflected"], dtype=np.complex128)
        weight_norm = np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1.0e-15)
        reflected_norm = np.maximum(np.linalg.norm(reflected, axis=1, keepdims=True), 1.0e-15)
        amplitude = np.abs(weights)
        reflected_amplitude = np.abs(reflected)
        phase_reflected = reflected / np.maximum(reflected_amplitude, 1.0e-15)
        phase_weight = weights / np.maximum(amplitude, 1.0e-15)
        active_violation = np.maximum(reflected_amplitude - rho * amplitude, 0.0) * command_masks
        gradient_active = (active_violation * phase_reflected) @ s_matrix.conj()
        gradient_active -= rho * active_violation * phase_weight
        total_violation = np.maximum(reflected_norm - rho * weight_norm, 0.0)
        gradient_total = total_violation * (
            (reflected / reflected_norm) @ s_matrix.conj() - rho * weights / weight_norm
        )
        gradient = (
            float(args.active_penalty) * gradient_active
            + float(args.total_penalty) * gradient_total
            + float(args.proximity_penalty) * (weights - initial)
        )
        gradient[~command_masks] = 0.0
        gradient_norm = np.maximum(np.linalg.norm(gradient, axis=1, keepdims=True), 1.0e-15)
        weights -= float(args.step_size) * weight_norm * gradient / gradient_norm
        amplitude_floor = 0.15 * initial_amplitude
        low = command_masks & (np.abs(weights) < amplitude_floor)
        weights[low] = amplitude_floor[low] * phase_weight[low]
        weights[~command_masks] = 0.0
        restore_target_responses(weights, target_rows, desired, k_values, command_masks)

    selected_norm = np.maximum(np.linalg.norm(best_weights, axis=1, keepdims=True), 1.0e-15)
    weights = best_weights / selected_norm
    final_metrics = evaluate_iteration(weights, s_matrix, command_masks, rho, significant_ratio)
    final_target = np.einsum("nkp,np->nk", target_rows, weights)
    desired_normalized = desired / selected_norm
    valid_task = np.arange(target_rows.shape[1])[None, :] < k_values[:, None]
    final_target_error = np.abs(final_target - desired_normalized) / np.maximum(np.abs(desired_normalized), 1.0e-9)

    rows: list[dict[str, Any]] = []
    for case_index in range(weights.shape[0]):
        max_theta = float(np.nanmax(targets[case_index, : k_values[case_index], 0]))
        rows.append(
            {
                "sample_index": case_index,
                "sample_id": sample_ids[case_index],
                "k": int(k_values[case_index]),
                "active_ratio": float(ratios[case_index]),
                "max_target_theta_deg": max_theta,
                "large_scan": int(max_theta >= float(args.large_scan_theta_deg)),
                "worst_active_return_loss_db": float(final_metrics["worst"][case_index]),
                "worst_significant_return_loss_db": float(final_metrics["worst_significant"][case_index]),
                "total_return_loss_db": float(final_metrics["total_rl"][case_index]),
                "all_active_10db_pass": int(final_metrics["active_pass"][case_index]),
                "all_significant_10db_pass": int(final_metrics["significant_pass"][case_index]),
                "total_10db_pass": int(float(final_metrics["total_rl"][case_index]) >= float(args.return_loss_min_db)),
                "engineering_10db_gate_pass": int(
                    bool(final_metrics["active_pass"][case_index])
                    and float(final_metrics["total_rl"][case_index]) >= float(args.return_loss_min_db)
                ),
                "target_response_error_max": float(
                    np.max(final_target_error[case_index, : k_values[case_index]])
                ),
            }
        )
    write_csv(args.out_dir / "projected_active_return_case_metrics.csv", rows)
    group_rows = summarize_rows(rows)
    write_csv(args.out_dir / "projected_active_return_group_summary.csv", group_rows)
    write_csv(args.out_dir / "projection_iteration_history.csv", history_rows)
    eligible = [row for row in rows if row["engineering_10db_gate_pass"]]
    eligible_path = args.out_dir / "projected_active_return_eligible.csv"
    if eligible:
        write_csv(eligible_path, eligible)
    else:
        eligible_path.write_text(
            "sample_index,sample_id,k,active_ratio,engineering_10db_gate_pass\n",
            encoding="utf-8-sig",
        )
    failures = [row for row in rows if not row["engineering_10db_gate_pass"]]
    failures.sort(key=lambda row: float(row["worst_active_return_loss_db"]))
    write_csv(args.out_dir / "projected_active_return_failures.csv", failures)
    np.savez_compressed(
        args.out_dir / "projected_source_weights.npz",
        sample_ids=np.asarray(sample_ids),
        weights_real_imag=np.stack((weights.real, weights.imag), axis=-1).astype(np.float32),
        masks=command_masks.astype(np.int8),
        k_values=k_values,
        active_ratios_actual=ratios,
        targets_deg=targets,
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "target-response-preserving reflected-power LCMV gradient plus sequential SOC-violation projection",
        "case_count": int(weights.shape[0]),
        "iterations": int(args.iterations),
        "return_loss_requirement_db": float(args.return_loss_min_db),
        "mask_pruning": {
            "amplitude_relative_db": args.prune_amplitude_relative_db,
            "mean_active_count_before": float(np.mean(original_active_counts)),
            "mean_active_count_after": float(np.mean(effective_active_counts)),
            "mean_active_count_reduction": float(np.mean(original_active_counts - effective_active_counts)),
            "mean_ratio_before": float(np.mean(original_active_counts / command_masks.shape[1])),
            "mean_ratio_after": float(np.mean(effective_active_counts / command_masks.shape[1])),
        },
        "initial": {
            "all_active_10db_pass_rate": initial_metrics["all_active_pass_rate"],
            "all_significant_10db_pass_rate": initial_metrics["all_significant_pass_rate"],
            "total_10db_pass_rate": initial_metrics["total_pass_rate"],
            "total_rl_mean_db": float(np.mean(initial_metrics["total_rl"])),
        },
        "projected": {
            "all_active_10db_pass_count": int(np.sum(final_metrics["active_pass"])),
            "all_active_10db_pass_rate": final_metrics["all_active_pass_rate"],
            "all_significant_10db_pass_count": int(np.sum(final_metrics["significant_pass"])),
            "all_significant_10db_pass_rate": final_metrics["all_significant_pass_rate"],
            "total_10db_pass_rate": final_metrics["total_pass_rate"],
            "engineering_10db_gate_pass_count": len(eligible),
            "engineering_10db_gate_pass_rate": float(len(eligible) / max(len(rows), 1)),
            "worst_active_return_loss_min_db": float(np.nanmin(final_metrics["worst"])),
            "worst_significant_return_loss_min_db": float(np.nanmin(final_metrics["worst_significant"])),
            "total_rl_mean_db": float(np.mean(final_metrics["total_rl"])),
            "target_response_error_max": float(np.max(final_target_error[valid_task])),
        },
        "hard_gate_passed": bool(len(eligible) == len(rows)),
        "new_fullwave_labels_allowed": bool(len(eligible) == len(rows)),
        "outputs": {
            "weights": str(args.out_dir / "projected_source_weights.npz"),
            "case_metrics": str(args.out_dir / "projected_active_return_case_metrics.csv"),
            "group_summary": str(args.out_dir / "projected_active_return_group_summary.csv"),
            "failures": str(args.out_dir / "projected_active_return_failures.csv"),
            "eligible": str(args.out_dir / "projected_active_return_eligible.csv"),
        },
    }
    (args.out_dir / "active_return_projection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gate_decision = {
        "created_at": summary["created_at"],
        "gate": "per-active-port return loss >= 10 dB AND total reflected-power return loss >= 10 dB",
        "case_count": int(weights.shape[0]),
        "eligible_case_count": len(eligible),
        "new_fullwave_labels_allowed": bool(summary["new_fullwave_labels_allowed"]),
        "decision": "block_fullwave_label_generation" if not summary["new_fullwave_labels_allowed"] else "allow",
        "reason": (
            "No case satisfies both active-port and total-reflected-power gates."
            if not eligible
            else "Only a subset is eligible; the stage-wide hard gate still fails."
        ),
    }
    (args.out_dir / "fullwave_label_gate_decision.json").write_text(
        json.dumps(gate_decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
