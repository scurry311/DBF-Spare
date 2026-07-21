"""Reconstruct task-level weights with regional nulls and proxy RF gating.

The optimized variable is W=[w_1,...,w_K].  The simultaneous source vector is
always recomputed as sum_k w_k; no legacy combined-weight field is used.  The
regional leakage constraints are complex-disk (second-order cone) constraints
enforced by cyclic projections.  S256 is a local-full-wave-kernel proxy, so the
outputs are candidates for later HFSS validation rather than full-wave labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from generate_iso_lcmv_teacher import (
    evaluate_weights,
    make_grid,
    make_local_null_dirs_by_target,
    side_mask_for_targets,
    steering_rx,
    target_dirs_for_sample,
    valid_targets_deg_for_sample,
)
from project_active_return_weights import project_single_source_weights


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_NETWORK = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_s256_proxy_20260717_run02"
    / "grounded_patch_s256_local_fullwave_proxy.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "grounded_patch_task_lcmv_psll_20260717_run01"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-cases", type=int, default=0, help="0 processes all cases")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--ratio-filter",
        type=float,
        default=None,
        help="Optionally process only one requested ratio (for control-group reruns).",
    )
    parser.add_argument("--theta-step", type=float, default=4.0)
    parser.add_argument("--phi-step", type=float, default=8.0)
    parser.add_argument("--regional-offsets-deg", default="2,5")
    parser.add_argument("--nearest-isolation-db", type=float, default=25.0)
    parser.add_argument("--local-isolation-db", type=float, default=20.0)
    parser.add_argument("--projection-margin-db", type=float, default=1.0)
    parser.add_argument("--regional-sweeps", type=int, default=24)
    parser.add_argument("--joint-outer-iterations", type=int, default=24)
    parser.add_argument("--joint-task-sweeps", type=int, default=1)
    parser.add_argument("--sum-rel-tolerance", type=float, default=0.02)
    parser.add_argument("--prune-relative-db", type=float, default=-12.0)
    parser.add_argument("--return-loss-db", type=float, default=10.0)
    parser.add_argument(
        "--rf-design-margin-db",
        type=float,
        default=0.25,
        help="Projection target margin above the 10 dB reporting gate.",
    )
    parser.add_argument("--rf-iterations", type=int, default=45)
    parser.add_argument("--rf-step-size", type=float, default=0.04)
    parser.add_argument("--rf-active-penalty", type=float, default=20.0)
    parser.add_argument("--rf-total-penalty", type=float, default=3.0)
    parser.add_argument("--psll-steps", type=int, default=8)
    parser.add_argument("--psll-topk", type=int, default=16)
    parser.add_argument("--psll-step-size", type=float, default=0.055)
    parser.add_argument("--psll-min-improvement-db", type=float, default=0.02)
    parser.add_argument("--max-mainlobe-loss-db", type=float, default=0.5)
    parser.add_argument("--large-scan-theta-deg", type=float, default=45.0)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def complex_from_ri(values: np.ndarray) -> np.ndarray:
    return np.asarray(values[..., 0] + 1j * values[..., 1], dtype=np.complex64)


def prune_mask_from_true_sum(
    original_mask: np.ndarray,
    combined: np.ndarray,
    k_value: int,
    relative_db: float,
) -> np.ndarray:
    mask = np.asarray(original_mask, dtype=bool).copy()
    active = np.flatnonzero(mask)
    if active.size == 0:
        return mask
    threshold = 10.0 ** (float(relative_db) / 20.0) * float(np.max(np.abs(combined[active])))
    mask &= np.abs(combined) >= threshold
    minimum_keep = min(active.size, max(2 * int(k_value), 8))
    if int(np.sum(mask)) < minimum_keep:
        strongest = active[np.argsort(np.abs(combined[active]))[-minimum_keep:]]
        mask[strongest] = True
    return mask


def project_disk(weight: np.ndarray, row: np.ndarray, radius: float) -> None:
    response = complex(row @ weight)
    magnitude = abs(response)
    if magnitude <= radius:
        return
    denominator = max(float(np.vdot(row, row).real), EPS)
    target = response * (float(radius) / max(magnitude, math.sqrt(EPS)))
    weight += row.conj() * ((target - response) / denominator)


def project_affine(weight: np.ndarray, row: np.ndarray, desired: complex) -> None:
    denominator = max(float(np.vdot(row, row).real), EPS)
    weight += row.conj() * ((desired - complex(row @ weight)) / denominator)


def build_task_constraints(
    positions_active: np.ndarray,
    target_dirs: np.ndarray,
    local_dirs: list[np.ndarray],
    desired: np.ndarray,
    nearest_db: float,
    local_db: float,
    margin_db: float,
) -> list[tuple[np.ndarray, complex, np.ndarray, np.ndarray]]:
    target_rows = steering_rx(positions_active, target_dirs).astype(np.complex128)
    constraints: list[tuple[np.ndarray, complex, np.ndarray, np.ndarray]] = []
    nearest_ratio = 10.0 ** (-(float(nearest_db) + float(margin_db)) / 20.0)
    local_ratio = 10.0 ** (-(float(local_db) + float(margin_db)) / 20.0)
    local_rows = [steering_rx(positions_active, dirs).astype(np.complex128) for dirs in local_dirs]
    for task_index in range(target_dirs.shape[0]):
        rows: list[np.ndarray] = []
        bounds: list[float] = []
        desired_scale = max(abs(complex(desired[task_index])), math.sqrt(EPS))
        for other_index in range(target_dirs.shape[0]):
            if other_index == task_index:
                continue
            rows.append(target_rows[other_index])
            bounds.append(desired_scale * nearest_ratio)
            for row in local_rows[other_index]:
                rows.append(row)
                bounds.append(desired_scale * local_ratio)
        leakage_rows = np.asarray(rows, dtype=np.complex128)
        leakage_bounds = np.asarray(bounds, dtype=np.float64)
        constraints.append(
            (target_rows[task_index], complex(desired[task_index]), leakage_rows, leakage_bounds)
        )
    return constraints


def project_task(
    weight: np.ndarray,
    payload: tuple[np.ndarray, complex, np.ndarray, np.ndarray],
    sweeps: int,
) -> np.ndarray:
    desired_row, desired, leakage_rows, leakage_bounds = payload
    output = np.asarray(weight, dtype=np.complex128).copy()
    for _ in range(max(1, int(sweeps))):
        project_affine(output, desired_row, desired)
        for row, bound in zip(leakage_rows, leakage_bounds):
            project_disk(output, row, float(bound))
    project_affine(output, desired_row, desired)
    return output


def task_constraint_errors(
    weights_active: np.ndarray,
    constraints: list[tuple[np.ndarray, complex, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    desired_error = 0.0
    leakage_violation = 0.0
    for task_index, payload in enumerate(constraints):
        desired_row, desired, leakage_rows, leakage_bounds = payload
        desired_error = max(
            desired_error,
            abs(complex(desired_row @ weights_active[:, task_index]) - desired)
            / max(abs(desired), math.sqrt(EPS)),
        )
        if leakage_rows.size:
            ratios = np.abs(leakage_rows @ weights_active[:, task_index]) / np.maximum(
                leakage_bounds, math.sqrt(EPS)
            )
            leakage_violation = max(leakage_violation, float(np.max(ratios) - 1.0))
    return float(desired_error), float(max(leakage_violation, 0.0))


def regional_seed(
    initial_active: np.ndarray,
    constraints: list[tuple[np.ndarray, complex, np.ndarray, np.ndarray]],
    sweeps: int,
) -> np.ndarray:
    output = np.asarray(initial_active, dtype=np.complex128).copy()
    for task_index, payload in enumerate(constraints):
        output[:, task_index] = project_task(output[:, task_index], payload, sweeps)
    return output


def match_combined_by_alternating_projection(
    seed_active: np.ndarray,
    combined_reference: np.ndarray,
    constraints: list[tuple[np.ndarray, complex, np.ndarray, np.ndarray]],
    outer_iterations: int,
    task_sweeps: int,
) -> tuple[np.ndarray, dict[str, float]]:
    output = np.asarray(seed_active, dtype=np.complex128).copy()
    task_count = output.shape[1]
    best = output.copy()
    best_score = float("inf")
    best_diag: dict[str, float] = {}
    reference_norm = max(float(np.linalg.norm(combined_reference)), math.sqrt(EPS))
    for _ in range(max(1, int(outer_iterations))):
        residual = combined_reference - np.sum(output, axis=1)
        output += residual[:, None] / float(max(task_count, 1))
        for task_index, payload in enumerate(constraints):
            output[:, task_index] = project_task(output[:, task_index], payload, task_sweeps)
        sum_error = float(np.linalg.norm(np.sum(output, axis=1) - combined_reference) / reference_norm)
        desired_error, leakage_violation = task_constraint_errors(output, constraints)
        score = sum_error + 2.0 * desired_error + 0.25 * leakage_violation
        if score < best_score:
            best_score = score
            best = output.copy()
            best_diag = {
                "sum_relative_error": sum_error,
                "desired_relative_error": desired_error,
                "leakage_bound_relative_violation": leakage_violation,
            }
    return best, best_diag


def active_return_metrics(
    combined: np.ndarray,
    command_mask: np.ndarray,
    s_matrix: np.ndarray,
    return_loss_db: float,
) -> dict[str, float | bool]:
    weights = np.asarray(combined, dtype=np.complex128)
    norm = max(float(np.linalg.norm(weights)), EPS)
    weights = weights / norm
    reflected = s_matrix @ weights
    active = np.asarray(command_mask, dtype=bool) & (np.abs(weights) > 1.0e-10)
    gamma = np.abs(reflected[active] / weights[active])
    worst = -20.0 * math.log10(max(float(np.max(gamma)) if gamma.size else 1.0e15, 1.0e-15))
    total = -10.0 * math.log10(
        max(float(np.vdot(reflected, reflected).real / np.vdot(weights, weights).real), 1.0e-30)
    )
    all_active = bool(gamma.size and np.all(gamma <= 10.0 ** (-float(return_loss_db) / 20.0)))
    total_pass = bool(total >= float(return_loss_db))
    return {
        "worst_active_rl_db": worst,
        "total_rl_db": total,
        "all_active_rl_pass": all_active,
        "total_rl_pass": total_pass,
        "rf_gate_pass": bool(all_active and total_pass),
    }


def af_gate(metrics: dict[str, float], k_value: int) -> bool:
    if float(metrics["target_spread_db"]) > 3.0:
        return False
    if int(k_value) <= 1:
        return True
    return bool(
        np.isfinite(metrics["isolation_min_db"])
        and float(metrics["isolation_min_db"]) >= 25.0
        and np.isfinite(metrics["local_isolation_min_db"])
        and float(metrics["local_isolation_min_db"]) >= 20.0
    )


def expand_task_weights(active_weights: np.ndarray, active: np.ndarray, element_count: int, kmax: int) -> np.ndarray:
    output = np.zeros((element_count, kmax), dtype=np.complex64)
    output[active, : active_weights.shape[1]] = active_weights.astype(np.complex64)
    return output


def optimize_psll(
    weights: np.ndarray,
    mask: np.ndarray,
    constraints: list[tuple[np.ndarray, complex, np.ndarray, np.ndarray]],
    target_rows_full: np.ndarray,
    s_matrix: np.ndarray,
    positions: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    target_dirs: np.ndarray,
    local_dirs: list[np.ndarray],
    grid_dirs: np.ndarray,
    grid_steer: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, float | int | bool]]:
    active = np.flatnonzero(mask)
    valid_indices = np.flatnonzero(task_valid)
    current = np.asarray(weights, dtype=np.complex64).copy()
    best = current.copy()
    best_metrics = evaluate_weights(
        weights=best,
        targets_deg=targets_deg,
        task_valid=task_valid,
        positions=positions,
        grid_dirs=grid_dirs,
        grid_steer=grid_steer,
        local_null_dirs_by_target=local_dirs,
    )
    baseline_weak = float(best_metrics["weak_peak_db"])
    best_rf = active_return_metrics(np.sum(best, axis=1), mask, s_matrix, args.return_loss_db)
    accepted = 0
    side_indices = np.flatnonzero(side_mask_for_targets(grid_dirs, target_dirs))
    if side_indices.size == 0 or int(args.psll_steps) <= 0:
        return best, best_metrics, {**best_rf, "psll_steps_accepted": accepted}

    for step_index in range(int(args.psll_steps)):
        combined = np.sum(current, axis=1)
        side_response = grid_steer[side_indices] @ combined
        top_count = min(max(1, int(args.psll_topk)), side_response.size)
        top_local = np.argpartition(np.abs(side_response) ** 2, -top_count)[-top_count:]
        rows = grid_steer[side_indices[top_local]][:, active]
        gradient = rows.conj().T @ side_response[top_local] / float(top_count)
        gradient_norm = float(np.linalg.norm(gradient))
        if not np.isfinite(gradient_norm) or gradient_norm <= EPS:
            break
        active_tasks = current[active[:, None], valid_indices[None, :]].astype(np.complex128)
        step_size = float(args.psll_step_size) / math.sqrt(float(step_index + 1))
        active_tasks -= (
            step_size * gradient[:, None] / gradient_norm / math.sqrt(float(max(valid_indices.size, 1)))
        )
        for task_index, payload in enumerate(constraints):
            active_tasks[:, task_index] = project_task(
                active_tasks[:, task_index], payload, max(2, int(args.joint_task_sweeps))
            )
        source_trial = np.zeros(current.shape[0], dtype=np.complex128)
        source_trial[active] = np.sum(active_tasks, axis=1)
        source_projected, _ = project_single_source_weights(
            source_trial,
            mask,
            s_matrix,
            target_rows_full,
            iterations=max(12, int(args.rf_iterations) // 2),
            step_size=float(args.rf_step_size),
            active_penalty=float(args.rf_active_penalty),
            total_penalty=float(args.rf_total_penalty),
            proximity_penalty=0.02,
            return_loss_min_db=float(args.return_loss_db) + float(args.rf_design_margin_db),
        )
        active_tasks, joint_diag = match_combined_by_alternating_projection(
            active_tasks,
            source_projected[active],
            constraints,
            max(6, int(args.joint_outer_iterations) // 2),
            int(args.joint_task_sweeps),
        )
        trial = expand_task_weights(active_tasks, active, current.shape[0], current.shape[1])
        metrics = evaluate_weights(
            weights=trial,
            targets_deg=targets_deg,
            task_valid=task_valid,
            positions=positions,
            grid_dirs=grid_dirs,
            grid_steer=grid_steer,
            local_null_dirs_by_target=local_dirs,
        )
        rf_metrics = active_return_metrics(np.sum(trial, axis=1), mask, s_matrix, args.return_loss_db)
        feasible = bool(
            af_gate(metrics, valid_indices.size)
            and rf_metrics["rf_gate_pass"]
            and float(joint_diag.get("sum_relative_error", 1.0)) <= float(args.sum_rel_tolerance)
            and baseline_weak - float(metrics["weak_peak_db"]) <= float(args.max_mainlobe_loss_db)
        )
        improved = (
            feasible
            and float(metrics["psll_to_weakest_peak_db"])
            <= float(best_metrics["psll_to_weakest_peak_db"]) - float(args.psll_min_improvement_db)
        )
        if improved:
            best = trial
            best_metrics = metrics
            best_rf = rf_metrics
            current = trial
            accepted += 1
        elif feasible:
            current = trial
        else:
            break
    return best, best_metrics, {**best_rf, "psll_steps_accepted": accepted}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def finite_mean(values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(np.mean(finite)) if finite.size else float("nan")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        keys = [
            ("all", "all", "all"),
            (str(row["k"]), "all", "all"),
            (str(row["k"]), f"{float(row['ratio_requested']):.1f}", "all"),
            (str(row["k"]), f"{float(row['ratio_requested']):.1f}", str(row["large_scan"])),
        ]
        for key in keys:
            groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (k_value, ratio, large_scan), members in sorted(groups.items()):
        output.append(
            {
                "k": k_value,
                "ratio_requested": ratio,
                "large_scan": large_scan,
                "case_count": len(members),
                "ratio_effective_mean": float(np.mean([float(row["ratio_effective"]) for row in members])),
                "original_psll_mean_db": float(np.mean([float(row["original_psll_db"]) for row in members])),
                "final_psll_mean_db": float(np.mean([float(row["final_psll_db"]) for row in members])),
                "psll_delta_mean_db": float(np.mean([float(row["psll_delta_db"]) for row in members])),
                "final_psll_le_0_rate": float(np.mean([float(row["final_psll_db"]) <= 0.0 for row in members])),
                "final_psll_le_m3_rate": float(np.mean([float(row["final_psll_db"]) <= -3.0 for row in members])),
                "final_psll_le_m6_rate": float(np.mean([float(row["final_psll_db"]) <= -6.0 for row in members])),
                "final_nearest_iso_mean_db": finite_mean([float(row["final_nearest_iso_db"]) for row in members]),
                "final_local_iso_mean_db": finite_mean([float(row["final_local_iso_db"]) for row in members]),
                "mainlobe_loss_mean_db": float(np.mean([float(row["mainlobe_loss_db"]) for row in members])),
                "mainlobe_gate_rate": float(np.mean([int(row["mainlobe_gate_pass"]) for row in members])),
                "af_gate_rate": float(np.mean([int(row["af_gate_pass"]) for row in members])),
                "proxy_rf_gate_rate": float(np.mean([int(row["proxy_rf_gate_pass"]) for row in members])),
                "joint_gate_rate": float(np.mean([int(row["joint_gate_pass"]) for row in members])),
                "psll_refined_rate": float(np.mean([int(row["psll_steps_accepted"]) > 0 for row in members])),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    dataset = np.load(args.dataset, allow_pickle=False)
    network = np.load(args.network, allow_pickle=False)
    model_label = str(network["model_label"])
    if "proxy" not in model_label:
        raise ValueError("This script expects an explicitly labeled proxy S256 network")
    s_matrix = np.asarray(network["s_parameters"], dtype=np.complex128)
    positions = np.asarray(dataset["positions_lambda"], dtype=np.float32)
    original_weights_all = complex_from_ri(dataset["task_weights_real_imag"])
    original_masks = np.asarray(dataset["masks"], dtype=bool)
    k_values = np.asarray(dataset["k_values"], dtype=int)
    start = max(0, int(args.start_index))
    indices = np.arange(start, len(k_values), dtype=int)
    if args.ratio_filter is not None:
        ratios_requested = np.asarray(dataset["active_ratios_requested"], dtype=np.float64)
        indices = indices[np.isclose(ratios_requested[indices], float(args.ratio_filter), atol=1.0e-6)]
    if int(args.max_cases) > 0:
        indices = indices[: int(args.max_cases)]
    _, _, grid_dirs = make_grid(float(args.theta_step), float(args.phi_step))
    grid_steer = steering_rx(positions, grid_dirs)
    offsets = parse_float_list(args.regional_offsets_deg)
    output_weights = np.zeros((indices.size, positions.shape[0], original_weights_all.shape[2]), dtype=np.complex64)
    output_masks = np.zeros((indices.size, positions.shape[0]), dtype=np.int8)
    rows: list[dict[str, Any]] = []
    started_at = time.time()

    for output_index, sample_index in enumerate(indices):
        k_value = int(k_values[sample_index])
        valid = np.asarray(dataset["task_valid"][sample_index], dtype=bool)
        valid_indices = np.flatnonzero(valid)
        targets_deg = np.asarray(dataset["targets_deg"][sample_index], dtype=np.float32)
        target_values = valid_targets_deg_for_sample(targets_deg, valid)
        target_dirs = target_dirs_for_sample(targets_deg, valid)
        local_dirs = make_local_null_dirs_by_target(target_values, offsets) if k_value > 1 else [np.zeros((0, 3), dtype=np.float32)]
        original_weights = original_weights_all[sample_index]
        original_combined = np.sum(original_weights[:, valid_indices], axis=1)
        original_source_norm = max(float(np.linalg.norm(original_combined)), math.sqrt(EPS))
        original_weights_eirp = original_weights / original_source_norm
        original_metrics = evaluate_weights(
            weights=original_weights_eirp,
            targets_deg=targets_deg,
            task_valid=valid,
            positions=positions,
            grid_dirs=grid_dirs,
            grid_steer=grid_steer,
            local_null_dirs_by_target=local_dirs,
        )
        ratio_requested = float(dataset["active_ratios_requested"][sample_index])
        mask = (
            original_masks[sample_index].copy()
            if ratio_requested >= 0.999
            else prune_mask_from_true_sum(
                original_masks[sample_index], original_combined, k_value, float(args.prune_relative_db)
            )
        )
        active = np.flatnonzero(mask)
        positions_active = positions[active]
        target_rows_active = steering_rx(positions_active, target_dirs).astype(np.complex128)
        target_rows_full = steering_rx(positions, target_dirs).astype(np.complex128)
        original_active = original_weights[active[:, None], valid_indices[None, :]].astype(np.complex128)
        desired = np.asarray(
            [target_rows_active[task_index] @ original_active[:, task_index] for task_index in range(k_value)],
            dtype=np.complex128,
        )
        desired = np.exp(1j * np.angle(desired))
        constraints = build_task_constraints(
            positions_active,
            target_dirs,
            local_dirs,
            desired,
            float(args.nearest_isolation_db),
            float(args.local_isolation_db),
            float(args.projection_margin_db),
        )
        seed_active = regional_seed(original_active, constraints, int(args.regional_sweeps))
        seed_combined = np.sum(seed_active, axis=1)
        seed_norm = max(float(np.linalg.norm(seed_combined)), math.sqrt(EPS))
        seed_active /= seed_norm
        desired /= seed_norm
        constraints = build_task_constraints(
            positions_active,
            target_dirs,
            local_dirs,
            desired,
            float(args.nearest_isolation_db),
            float(args.local_isolation_db),
            float(args.projection_margin_db),
        )
        source_seed = np.zeros(positions.shape[0], dtype=np.complex128)
        source_seed[active] = np.sum(seed_active, axis=1)
        source_rf, source_rf_diag = project_single_source_weights(
            source_seed,
            mask,
            s_matrix,
            target_rows_full,
            iterations=int(args.rf_iterations),
            step_size=float(args.rf_step_size),
            active_penalty=float(args.rf_active_penalty),
            total_penalty=float(args.rf_total_penalty),
            proximity_penalty=0.01,
            return_loss_min_db=float(args.return_loss_db) + float(args.rf_design_margin_db),
        )
        joint_active, joint_diag = match_combined_by_alternating_projection(
            seed_active,
            source_rf[active],
            constraints,
            int(args.joint_outer_iterations),
            int(args.joint_task_sweeps),
        )
        stage1_weights = expand_task_weights(
            joint_active, active, positions.shape[0], original_weights_all.shape[2]
        )
        stage1_metrics = evaluate_weights(
            weights=stage1_weights,
            targets_deg=targets_deg,
            task_valid=valid,
            positions=positions,
            grid_dirs=grid_dirs,
            grid_steer=grid_steer,
            local_null_dirs_by_target=local_dirs,
        )
        stage1_rf = active_return_metrics(
            np.sum(stage1_weights, axis=1), mask, s_matrix, float(args.return_loss_db)
        )
        can_refine = bool(
            af_gate(stage1_metrics, k_value)
            and stage1_rf["rf_gate_pass"]
            and float(joint_diag.get("sum_relative_error", 1.0)) <= float(args.sum_rel_tolerance)
        )
        if can_refine:
            final_weights, final_metrics, final_rf = optimize_psll(
                stage1_weights,
                mask,
                constraints,
                target_rows_full,
                s_matrix,
                positions,
                targets_deg,
                valid,
                target_dirs,
                local_dirs,
                grid_dirs,
                grid_steer,
                args,
            )
        else:
            final_weights = stage1_weights
            final_metrics = stage1_metrics
            final_rf = {**stage1_rf, "psll_steps_accepted": 0}
        final_source_norm = max(float(np.linalg.norm(np.sum(final_weights, axis=1))), math.sqrt(EPS))
        final_weights = final_weights / final_source_norm
        final_metrics = evaluate_weights(
            weights=final_weights,
            targets_deg=targets_deg,
            task_valid=valid,
            positions=positions,
            grid_dirs=grid_dirs,
            grid_steer=grid_steer,
            local_null_dirs_by_target=local_dirs,
        )
        final_rf = {
            **active_return_metrics(
                np.sum(final_weights, axis=1), mask, s_matrix, float(args.return_loss_db)
            ),
            "psll_steps_accepted": int(final_rf["psll_steps_accepted"]),
        }
        final_af_pass = af_gate(final_metrics, k_value)
        mainlobe_loss = float(original_metrics["weak_peak_db"] - final_metrics["weak_peak_db"])
        mainlobe_gate = bool(mainlobe_loss <= float(args.max_mainlobe_loss_db))
        final_joint = bool(
            final_af_pass
            and final_rf["rf_gate_pass"]
            and mainlobe_gate
            and float(final_metrics["psll_to_weakest_peak_db"]) <= 0.0
        )
        output_weights[output_index] = final_weights
        output_masks[output_index] = mask.astype(np.int8)
        max_theta = float(np.max(target_values[:, 0])) if target_values.size else float("nan")
        rows.append(
            {
                "sample_index": int(sample_index),
                "sample_id": str(dataset["sample_ids"][sample_index]),
                "k": k_value,
                "ratio_requested": ratio_requested,
                "ratio_effective": float(np.mean(mask)),
                "active_count": int(np.sum(mask)),
                "max_target_theta_deg": max_theta,
                "large_scan": int(max_theta >= float(args.large_scan_theta_deg)),
                "original_psll_db": float(original_metrics["psll_to_weakest_peak_db"]),
                "stage1_psll_db": float(stage1_metrics["psll_to_weakest_peak_db"]),
                "final_psll_db": float(final_metrics["psll_to_weakest_peak_db"]),
                "psll_delta_db": float(final_metrics["psll_to_weakest_peak_db"] - original_metrics["psll_to_weakest_peak_db"]),
                "final_weak_peak_db": float(final_metrics["weak_peak_db"]),
                "original_weak_peak_db_same_incident_power": float(original_metrics["weak_peak_db"]),
                "mainlobe_loss_db": mainlobe_loss,
                "mainlobe_gate_pass": int(mainlobe_gate),
                "final_target_spread_db": float(final_metrics["target_spread_db"]),
                "final_nearest_iso_db": float(final_metrics["isolation_min_db"]),
                "final_local_iso_db": float(final_metrics["local_isolation_min_db"]),
                "final_energy_proxy": float(final_metrics["energy_proxy"]),
                "rf_source_projection_gate": int(bool(source_rf_diag["engineering_10db_gate_pass"])),
                "joint_sum_relative_error": float(joint_diag.get("sum_relative_error", float("nan"))),
                "joint_desired_relative_error": float(joint_diag.get("desired_relative_error", float("nan"))),
                "joint_leakage_bound_violation": float(joint_diag.get("leakage_bound_relative_violation", float("nan"))),
                "worst_active_rl_db": float(final_rf["worst_active_rl_db"]),
                "total_rl_db": float(final_rf["total_rl_db"]),
                "af_gate_pass": int(final_af_pass),
                "proxy_rf_gate_pass": int(bool(final_rf["rf_gate_pass"])),
                "joint_gate_pass": int(final_joint),
                "psll_steps_accepted": int(final_rf["psll_steps_accepted"]),
            }
        )
        if (output_index + 1) % 25 == 0 or output_index + 1 == indices.size:
            elapsed = time.time() - started_at
            rate = (output_index + 1) / max(elapsed, EPS)
            remaining = (indices.size - output_index - 1) / max(rate, EPS)
            print(
                f"processed={output_index + 1}/{indices.size} rate={rate:.2f}/s "
                f"eta_s={remaining:.1f} joint={sum(int(row['joint_gate_pass']) for row in rows)}",
                flush=True,
            )

    summary_rows = summarize(rows)
    write_csv(args.out_dir / "task_lcmv_psll_case_metrics.csv", rows)
    write_csv(args.out_dir / "task_lcmv_psll_group_summary.csv", summary_rows)
    np.savez_compressed(
        args.out_dir / "task_level_weights.npz",
        sample_indices=indices,
        sample_ids=np.asarray(dataset["sample_ids"])[indices],
        task_weights_real_imag=np.stack((output_weights.real, output_weights.imag), axis=-1).astype(np.float32),
        combined_weights_real_imag=np.stack((output_weights.sum(axis=2).real, output_weights.sum(axis=2).imag), axis=-1).astype(np.float32),
        masks=output_masks,
        k_values=k_values[indices],
        active_ratios_requested=np.asarray(dataset["active_ratios_requested"])[indices],
        active_ratios_effective=np.mean(output_masks, axis=1).astype(np.float32),
        targets_deg=np.asarray(dataset["targets_deg"])[indices],
        task_valid=np.asarray(dataset["task_valid"])[indices],
        positions_lambda=positions,
        model_scope=np.asarray("AF regional-SOCP plus local-kernel S256 proxy RF; not HFSS full-wave"),
    )
    overall = next(row for row in summary_rows if row["k"] == "all")
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "task-level LCMV initialization + regional complex-disk SOCP projections + proxy active-return projection + gated PSLL descent",
        "scope": "array-factor task metrics and local-full-wave-kernel S256 proxy RF metrics; not 16x16 HFSS full-wave",
        "case_count": len(rows),
        "source_indices": [int(indices[0]), int(indices[-1])] if indices.size else [],
        "network_model_label": model_label,
        "constraints": {
            "nearest_isolation_db": float(args.nearest_isolation_db),
            "local_isolation_db": float(args.local_isolation_db),
            "regional_offsets_deg": offsets,
            "return_loss_db": float(args.return_loss_db),
            "rf_design_margin_db": float(args.rf_design_margin_db),
            "psll_stage_targets_db": [-3.0, -6.0],
        },
        "overall": overall,
        "hfss_label_generation_allowed": False,
        "decision": "select smoke-test candidates only; require EEP/HFSS validation before labels or engineering claims",
        "outputs": {
            "weights": str(args.out_dir / "task_level_weights.npz"),
            "case_metrics": str(args.out_dir / "task_lcmv_psll_case_metrics.csv"),
            "group_summary": str(args.out_dir / "task_lcmv_psll_group_summary.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "fullwave_label_gate_decision.json").write_text(
        json.dumps(
            {
                "allowed": False,
                "reason": "Metrics use AF plus a local-kernel S256 proxy, not a validated full 16x16 HFSS/EEP operator.",
                "eligible_proxy_joint_count": int(sum(int(row["joint_gate_pass"]) for row in rows)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
