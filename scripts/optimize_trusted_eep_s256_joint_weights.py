#!/usr/bin/env python3
"""Jointly project task weights with trusted EEP and matched-S256 constraints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hfss_task_fullwave_validate import (
    combined_metrics,
    isolation_metrics,
    pattern_grid_dirs,
    unit_vector,
)
from validate_trusted_eep_hfss_residuals import field_pattern, pointing_error


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "dataset_v2_20260724"
    / "residual_critic_dataset_v2.npz"
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
    / "trusted_eep_residual_20260723_run02"
    / "eep_hfss_validation"
    / "case_excitations.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_eep_s256_joint_optimization_20260724_run03"
EPS = 1.0e-12


@dataclass(frozen=True)
class Config:
    name: str
    iterations: int
    step_size: float
    combined_penalty: float
    task_penalty: float
    total_penalty: float
    proximity: float
    amplitude_floor_db: float
    pocs_passes: int


CONFIGS = (
    Config("balanced30", 70, 0.025, 16.0, 3.0, 2.0, 0.005, -30.0, 1),
    Config("combined20", 90, 0.030, 28.0, 1.5, 2.0, 0.003, -20.0, 1),
    Config("robust20", 110, 0.018, 22.0, 6.0, 3.0, 0.008, -20.0, 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--excitations", type=Path, default=DEFAULT_EXCITATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--nearest-isolation-db", type=float, default=25.0)
    parser.add_argument("--local-isolation-db", type=float, default=20.0)
    parser.add_argument("--return-loss-min-db", type=float, default=10.0)
    parser.add_argument("--task-significant-relative-db", type=float, default=-20.0)
    return parser.parse_args()


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


def normalize(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), EPS)


def active_return(
    s_matrix: np.ndarray,
    excitation: np.ndarray,
    mask: np.ndarray,
    *,
    relative_db: float | None,
    threshold_db: float,
) -> dict[str, float | int]:
    value = normalize(np.asarray(excitation, dtype=np.complex128))
    reflected = s_matrix @ value
    amplitude = np.abs(value)
    maximum = max(float(np.max(amplitude)), EPS)
    if relative_db is None:
        driven = mask.copy()
        if np.any(amplitude[driven] < maximum * 1.0e-8):
            return {
                "worst_active_rl_db": -300.0,
                "total_rl_db": float(
                    -10.0 * np.log10(max(float(np.vdot(reflected, reflected).real), 1.0e-30))
                ),
                "gate_pass": 0,
                "evaluated_port_count": int(np.sum(driven)),
                "dynamic_range_db": float("inf"),
            }
    else:
        driven = mask & (amplitude >= maximum * 10.0 ** (float(relative_db) / 20.0))
    indices = np.flatnonzero(driven)
    gamma = np.abs(reflected[indices] / value[indices])
    worst_rl = float(-20.0 * np.log10(max(float(np.max(gamma)), 1.0e-30)))
    total_rl = float(-10.0 * np.log10(max(float(np.vdot(reflected, reflected).real), 1.0e-30)))
    active_amplitude = amplitude[mask]
    dynamic = float(20.0 * np.log10(maximum / max(float(np.min(active_amplitude)), 1.0e-30)))
    return {
        "worst_active_rl_db": worst_rl,
        "total_rl_db": total_rl,
        "gate_pass": int(worst_rl >= threshold_db and total_rl >= threshold_db),
        "evaluated_port_count": int(indices.size),
        "dynamic_range_db": dynamic,
    }


def reflection_gradient(
    s_matrix: np.ndarray,
    excitation: np.ndarray,
    considered: np.ndarray,
    rho: float,
    total_penalty: float,
) -> np.ndarray:
    value = np.asarray(excitation, dtype=np.complex128)
    reflected = s_matrix @ value
    amplitude = np.abs(value)
    reflected_amplitude = np.abs(reflected)
    phase_reflected = reflected / np.maximum(reflected_amplitude, EPS)
    phase_value = value / np.maximum(amplitude, EPS)
    violation = np.maximum(reflected_amplitude - rho * amplitude, 0.0) * considered
    gradient = s_matrix.conj().T @ (violation * phase_reflected)
    gradient -= rho * violation * phase_value
    value_norm = max(float(np.linalg.norm(value)), EPS)
    reflected_norm = max(float(np.linalg.norm(reflected)), EPS)
    total_violation = max(reflected_norm - rho * value_norm, 0.0)
    if total_violation > 0.0:
        gradient += float(total_penalty) * total_violation * (
            s_matrix.conj().T @ (reflected / reflected_norm) - rho * value / value_norm
        )
    return gradient


def nearest_grid_index(grid_dirs: np.ndarray, theta_deg: float, phi_deg: float) -> int:
    target = unit_vector(theta_deg, phi_deg)
    return int(np.argmax(grid_dirs @ target))


def local_grid_indices(
    grid_dirs: np.ndarray, theta_deg: float, phi_deg: float
) -> list[int]:
    offsets = ((-5.0, 0.0), (-2.0, 0.0), (2.0, 0.0), (5.0, 0.0),
               (0.0, -5.0), (0.0, -2.0), (0.0, 2.0), (0.0, 5.0))
    indices: list[int] = []
    for dtheta, dphi in offsets:
        theta = min(90.0, max(0.0, theta_deg + dtheta))
        phi = (phi_deg + dphi) % 360.0
        index = nearest_grid_index(grid_dirs, theta, phi)
        if index not in indices:
            indices.append(index)
    return indices


class ExternalOperator:
    def __init__(
        self,
        etheta: np.ndarray,
        ephi: np.ndarray,
        antenna_map: np.ndarray,
    ) -> None:
        self.etheta = np.asarray(etheta, dtype=np.complex128)
        self.ephi = np.asarray(ephi, dtype=np.complex128)
        self.map_t = np.asarray(antenna_map, dtype=np.complex128).T
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def rows(self, grid_index: int) -> tuple[np.ndarray, np.ndarray]:
        if grid_index not in self._cache:
            self._cache[grid_index] = (
                self.map_t @ self.etheta[:, grid_index],
                self.map_t @ self.ephi[:, grid_index],
            )
        return self._cache[grid_index]


@dataclass
class TaskConstraint:
    equality_row: np.ndarray
    desired: complex
    leakage: list[tuple[np.ndarray, float, str]]


def build_constraints(
    initial_tasks: np.ndarray,
    targets: np.ndarray,
    grid_dirs: np.ndarray,
    operator: ExternalOperator,
    nearest_db: float,
    local_db: float,
) -> list[TaskConstraint]:
    constraints: list[TaskConstraint] = []
    nearest_ratio = 10.0 ** (-nearest_db / 20.0)
    local_ratio = 10.0 ** (-local_db / 20.0)
    for task_index, (theta, phi) in enumerate(targets):
        own_index = nearest_grid_index(grid_dirs, float(theta), float(phi))
        own_theta, own_phi = operator.rows(own_index)
        initial = initial_tasks[:, task_index]
        field = np.asarray([initial @ own_theta, initial @ own_phi])
        field_norm = max(float(np.linalg.norm(field)), EPS)
        polarization = field / field_norm
        equality_row = np.conjugate(polarization[0]) * own_theta + np.conjugate(polarization[1]) * own_phi
        desired = complex(initial @ equality_row)
        leakage: list[tuple[np.ndarray, float, str]] = []
        for other_index, (other_theta, other_phi) in enumerate(targets):
            if other_index == task_index:
                continue
            center = nearest_grid_index(grid_dirs, float(other_theta), float(other_phi))
            rows = operator.rows(center)
            component_bound = field_norm * nearest_ratio / math.sqrt(2.0)
            leakage.extend((row, component_bound, "nearest") for row in rows)
            local_bound = field_norm * local_ratio / math.sqrt(2.0)
            for grid_index in local_grid_indices(grid_dirs, float(other_theta), float(other_phi)):
                leakage.extend((row, local_bound, "local") for row in operator.rows(grid_index))
        constraints.append(TaskConstraint(equality_row, desired, leakage))
    return constraints


def project_task(
    value: np.ndarray,
    mask: np.ndarray,
    constraint: TaskConstraint,
    passes: int,
) -> np.ndarray:
    active = np.flatnonzero(mask)
    out = np.asarray(value, dtype=np.complex128).copy()
    row = constraint.equality_row[active]
    denom = max(float(np.vdot(row, row).real), EPS)
    for _ in range(max(1, passes)):
        response = complex(row @ out[active])
        out[active] += np.conjugate(row) * ((constraint.desired - response) / denom)
        for leakage_row, bound, _kind in constraint.leakage:
            local_row = leakage_row[active]
            response = complex(local_row @ out[active])
            magnitude = abs(response)
            if magnitude > bound:
                target = response * (bound / magnitude)
                local_denom = max(float(np.vdot(local_row, local_row).real), EPS)
                out[active] += np.conjugate(local_row) * ((target - response) / local_denom)
        response = complex(row @ out[active])
        out[active] += np.conjugate(row) * ((constraint.desired - response) / denom)
    out[~mask] = 0.0
    return out


def constraint_metrics(
    tasks: np.ndarray,
    constraints: list[TaskConstraint],
) -> dict[str, float | int]:
    target_error = 0.0
    nearest_ratio = 0.0
    local_ratio = 0.0
    for task_index, constraint in enumerate(constraints):
        value = tasks[:, task_index]
        response = complex(constraint.equality_row @ value)
        target_error = max(target_error, abs(response - constraint.desired) / max(abs(constraint.desired), EPS))
        for row, bound, kind in constraint.leakage:
            ratio = abs(complex(row @ value)) / max(bound, EPS)
            if kind == "nearest":
                nearest_ratio = max(nearest_ratio, ratio)
            else:
                local_ratio = max(local_ratio, ratio)
    return {
        "target_response_error_max": float(target_error),
        "nearest_bound_ratio_max": float(nearest_ratio),
        "local_bound_ratio_max": float(local_ratio),
        "constraint_pass": int(target_error <= 0.02 and nearest_ratio <= 1.05 and local_ratio <= 1.05),
    }


def apply_combined_floor(
    tasks: np.ndarray,
    initial_combined: np.ndarray,
    mask: np.ndarray,
    floor_db: float,
) -> None:
    combined = np.sum(tasks, axis=1)
    maximum = max(float(np.max(np.abs(combined))), EPS)
    floor = maximum * 10.0 ** (floor_db / 20.0)
    low = mask & (np.abs(combined) < floor)
    if not np.any(low):
        return
    reference_phase = initial_combined / np.maximum(np.abs(initial_combined), EPS)
    fallback_phase = combined / np.maximum(np.abs(combined), EPS)
    phase = np.where(np.abs(initial_combined) > EPS, reference_phase, fallback_phase)
    correction = floor * phase[low] - combined[low]
    tasks[low, :] += correction[:, None] / tasks.shape[1]


def optimize_one(
    initial_tasks: np.ndarray,
    mask: np.ndarray,
    constraints: list[TaskConstraint],
    s_matrix: np.ndarray,
    config: Config,
    rl_min_db: float,
    task_relative_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    tasks = np.asarray(initial_tasks, dtype=np.complex128).copy()
    initial = tasks.copy()
    initial_combined = np.sum(initial, axis=1)
    rho = 10.0 ** (-rl_min_db / 20.0)
    best = tasks.copy()
    best_key: tuple[int, int, int, float, float] = (-1, -1, -1, -1.0e9, -1.0e9)
    best_iteration = 0
    for iteration in range(config.iterations + 1):
        combined = np.sum(tasks, axis=1)
        combined_metrics_rl = active_return(
            s_matrix, combined, mask, relative_db=None, threshold_db=rl_min_db
        )
        task_metrics = [
            active_return(
                s_matrix,
                tasks[:, task_index],
                mask,
                relative_db=task_relative_db,
                threshold_db=rl_min_db,
            )
            for task_index in range(tasks.shape[1])
        ]
        constraints_now = constraint_metrics(tasks, constraints)
        task_gate = int(all(int(item["gate_pass"]) for item in task_metrics))
        minimum_rl = min(
            float(combined_metrics_rl["worst_active_rl_db"]),
            float(combined_metrics_rl["total_rl_db"]),
            *(float(item["worst_active_rl_db"]) for item in task_metrics),
            *(float(item["total_rl_db"]) for item in task_metrics),
        )
        key = (
            int(int(combined_metrics_rl["gate_pass"]) and int(constraints_now["constraint_pass"])),
            int(combined_metrics_rl["gate_pass"]),
            int(constraints_now["constraint_pass"]),
            minimum_rl,
            -float(constraints_now["target_response_error_max"]),
        )
        if key > best_key:
            best_key = key
            best = tasks.copy()
            best_iteration = iteration
        if iteration == config.iterations:
            break

        combined_gradient = reflection_gradient(
            s_matrix, combined, mask, rho, config.total_penalty
        )
        combined_gradient *= config.combined_penalty
        combined_norm = max(float(np.linalg.norm(combined)), EPS)
        combined_gradient_norm = max(float(np.linalg.norm(combined_gradient)), EPS)
        combined_step = config.step_size * combined_norm * combined_gradient / combined_gradient_norm
        for task_index in range(tasks.shape[1]):
            value = tasks[:, task_index]
            amplitude = np.abs(value)
            maximum = max(float(np.max(amplitude)), EPS)
            significant = mask & (amplitude >= maximum * 10.0 ** (task_relative_db / 20.0))
            gradient = reflection_gradient(s_matrix, value, significant, rho, config.total_penalty)
            gradient *= config.task_penalty
            gradient_norm = max(float(np.linalg.norm(gradient)), EPS)
            task_norm = max(float(np.linalg.norm(value)), EPS)
            tasks[:, task_index] -= combined_step / tasks.shape[1]
            tasks[:, task_index] -= config.step_size * task_norm * gradient / gradient_norm
            tasks[:, task_index] -= config.step_size * config.proximity * (
                tasks[:, task_index] - initial[:, task_index]
            )
            tasks[~mask, task_index] = 0.0
        apply_combined_floor(tasks, initial_combined, mask, config.amplitude_floor_db)
        for task_index, constraint in enumerate(constraints):
            tasks[:, task_index] = project_task(
                tasks[:, task_index], mask, constraint, config.pocs_passes
            )

    tasks = best
    combined = np.sum(tasks, axis=1)
    combined_rl = active_return(s_matrix, combined, mask, relative_db=None, threshold_db=rl_min_db)
    task_rl = [
        active_return(
            s_matrix,
            tasks[:, task_index],
            mask,
            relative_db=task_relative_db,
            threshold_db=rl_min_db,
        )
        for task_index in range(tasks.shape[1])
    ]
    task_strict = [
        active_return(s_matrix, tasks[:, task_index], mask, relative_db=None, threshold_db=rl_min_db)
        for task_index in range(tasks.shape[1])
    ]
    return tasks.astype(np.complex64), {
        "config": config.name,
        "best_iteration": best_iteration,
        "combined_worst_active_rl_db": combined_rl["worst_active_rl_db"],
        "combined_total_rl_db": combined_rl["total_rl_db"],
        "combined_active_gate": combined_rl["gate_pass"],
        "combined_dynamic_range_db": combined_rl["dynamic_range_db"],
        "all_tasks_significant_worst_active_rl_db": min(float(item["worst_active_rl_db"]) for item in task_rl),
        "all_tasks_significant_worst_total_rl_db": min(float(item["total_rl_db"]) for item in task_rl),
        "all_tasks_significant_gate": int(all(int(item["gate_pass"]) for item in task_rl)),
        "all_tasks_strict_worst_active_rl_db": min(float(item["worst_active_rl_db"]) for item in task_strict),
        "all_tasks_strict_gate": int(all(int(item["gate_pass"]) for item in task_strict)),
        **constraint_metrics(tasks, constraints),
    }


def pattern_metrics(
    tasks: np.ndarray,
    targets: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    etheta: np.ndarray,
    ephi: np.ndarray,
    antenna_map: np.ndarray,
) -> dict[str, float]:
    external_cases = [normalize(tasks[:, task_index]) for task_index in range(tasks.shape[1])]
    external_cases.append(normalize(np.sum(tasks, axis=1)))
    external = np.stack(external_cases)
    antenna = external @ antenna_map.T
    fields_theta = antenna @ etheta
    fields_phi = antenna @ ephi
    task_patterns = {
        task_index: field_pattern(theta, phi, fields_theta[task_index], fields_phi[task_index])
        for task_index in range(tasks.shape[1])
    }
    combined_pattern = field_pattern(theta, phi, fields_theta[-1], fields_phi[-1])
    target_list = [[float(value[0]), float(value[1])] for value in targets]
    combined = combined_metrics(
        combined_pattern, target_list, target_radius_deg=5.0, sidelobe_exclusion_deg=8.0
    )
    isolation = isolation_metrics(task_patterns, target_list, target_radius_deg=5.0)
    return {
        "psll_db": float(combined["combined_psll_to_weakest_peak_db"]),
        "weakest_target_gain_db": float(combined["combined_target_peak_min_db"]),
        "target_spread_db": float(combined["combined_target_spread_db"]),
        "nearest_iso_db": float(isolation["isolation_worst_nearest_db"]),
        "local_iso_db": float(isolation["isolation_worst_local_db"]),
        "pointing_error_deg": float(pointing_error(combined_pattern, target_list)),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite joint optimization: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with np.load(args.dataset, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    with np.load(args.operator, allow_pickle=False) as source:
        theta = np.asarray(source["theta_deg"], dtype=np.float64)
        phi = np.asarray(source["phi_deg"], dtype=np.float64)
        etheta = np.asarray(source["etheta"], dtype=np.complex64)
        ephi = np.asarray(source["ephi"], dtype=np.complex64)
    with np.load(args.excitations, allow_pickle=False) as source:
        antenna_map = np.asarray(source["antenna_wave_map"], dtype=np.complex64)
        s_matrix = np.asarray(source["matched_s"], dtype=np.complex128)

    operator = ExternalOperator(etheta, ephi, antenna_map)
    grid_dirs = pattern_grid_dirs(theta, phi)
    task_weights = data["w_tasks_real_imag"][..., 0] + 1j * data["w_tasks_real_imag"][..., 1]
    total_candidate_count = int(data["candidate_index"].size)
    start_index = max(0, int(args.start_index))
    stop_index = total_candidate_count
    if int(args.max_candidates) > 0:
        stop_index = min(stop_index, start_index + int(args.max_candidates))
    candidate_indices = list(range(start_index, stop_index))
    candidate_count = len(candidate_indices)
    if not candidate_indices:
        raise ValueError("Candidate selection is empty")
    selected_tasks = np.zeros((candidate_count, 256, 6), dtype=np.complex64)
    candidate_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    for output_pos, candidate in enumerate(candidate_indices):
        k_value = int(data["k_values"][candidate])
        mask = np.asarray(data["mask"][candidate], dtype=bool)
        initial_tasks = np.conjugate(np.asarray(task_weights[candidate, :, :k_value], dtype=np.complex128))
        initial_tasks[~mask, :] = 0.0
        targets = np.asarray(data["targets_deg"][candidate, :k_value], dtype=np.float64)
        constraints = build_constraints(
            initial_tasks,
            targets,
            grid_dirs,
            operator,
            float(args.nearest_isolation_db),
            float(args.local_isolation_db),
        )
        baseline_rl = active_return(
            s_matrix, np.sum(initial_tasks, axis=1), mask, relative_db=None,
            threshold_db=float(args.return_loss_min_db)
        )
        baseline_pattern = pattern_metrics(
            initial_tasks, targets, theta, phi, etheta, ephi, antenna_map
        )
        trials: list[tuple[np.ndarray, dict[str, Any]]] = []
        for config in CONFIGS:
            optimized, metrics = optimize_one(
                initial_tasks,
                mask,
                constraints,
                s_matrix,
                config,
                float(args.return_loss_min_db),
                float(args.task_significant_relative_db),
            )
            trial_rows.append(
                {
                    "candidate_index": candidate,
                    "sample_index": int(data["sample_index"][candidate]),
                    "k": k_value,
                    "ratio": float(data["active_ratios_requested"][candidate]),
                    **metrics,
                }
            )
            trials.append((optimized, metrics))
        trials.sort(
            key=lambda item: (
                int(item[1]["combined_active_gate"] and item[1]["constraint_pass"]),
                int(item[1]["combined_active_gate"]),
                int(item[1]["constraint_pass"]),
                min(
                    float(item[1]["combined_worst_active_rl_db"]),
                    float(item[1]["combined_total_rl_db"]),
                ),
                float(item[1]["all_tasks_significant_worst_active_rl_db"]),
            ),
            reverse=True,
        )
        optimized_tasks, selected = trials[0]
        optimized_pattern = pattern_metrics(
            optimized_tasks, targets, theta, phi, etheta, ephi, antenna_map
        )
        mainlobe_gate = bool(
            optimized_pattern["weakest_target_gain_db"] >= baseline_pattern["weakest_target_gain_db"] - 0.5
            and optimized_pattern["target_spread_db"] <= 3.0
            and optimized_pattern["pointing_error_deg"] <= 1.5
        )
        gate15 = bool(
            optimized_pattern["psll_db"] <= 0.0
            and optimized_pattern["nearest_iso_db"] >= 25.0
            and optimized_pattern["local_iso_db"] >= 15.0
        )
        gate20 = bool(
            optimized_pattern["psll_db"] <= 0.0
            and optimized_pattern["nearest_iso_db"] >= 25.0
            and optimized_pattern["local_iso_db"] >= 20.0
        )
        robust_active = bool(
            selected["combined_active_gate"] and selected["all_tasks_significant_gate"]
        )
        strict_engineering = bool(gate20 and mainlobe_gate and robust_active)
        power_ratio_equal_weakest = float(
            10.0 ** (
                (baseline_pattern["weakest_target_gain_db"] - optimized_pattern["weakest_target_gain_db"])
                / 10.0
            )
        )
        selected_tasks[output_pos, :, :k_value] = optimized_tasks
        candidate_rows.append(
            {
                "candidate_index": candidate,
                "sample_index": int(data["sample_index"][candidate]),
                "sample_id": str(data["sample_ids"][candidate]),
                "k": k_value,
                "ratio": float(data["active_ratios_requested"][candidate]),
                "large_scan": int(data["large_scan"][candidate]),
                "min_target_separation_deg": float(data["min_target_separation_deg"][candidate]),
                "selected_config": selected["config"],
                "baseline_combined_worst_active_rl_db": baseline_rl["worst_active_rl_db"],
                "baseline_combined_total_rl_db": baseline_rl["total_rl_db"],
                "optimized_combined_worst_active_rl_db": selected["combined_worst_active_rl_db"],
                "optimized_combined_total_rl_db": selected["combined_total_rl_db"],
                "optimized_combined_active_gate": selected["combined_active_gate"],
                "optimized_all_tasks_significant_worst_active_rl_db": selected["all_tasks_significant_worst_active_rl_db"],
                "optimized_all_tasks_significant_gate": selected["all_tasks_significant_gate"],
                "optimized_all_tasks_strict_worst_active_rl_db": selected["all_tasks_strict_worst_active_rl_db"],
                "optimized_all_tasks_strict_gate": selected["all_tasks_strict_gate"],
                "target_response_error_max": selected["target_response_error_max"],
                "nearest_bound_ratio_max": selected["nearest_bound_ratio_max"],
                "local_bound_ratio_max": selected["local_bound_ratio_max"],
                "constraint_pass": selected["constraint_pass"],
                "baseline_psll_db": baseline_pattern["psll_db"],
                "optimized_psll_db": optimized_pattern["psll_db"],
                "baseline_nearest_iso_db": baseline_pattern["nearest_iso_db"],
                "optimized_nearest_iso_db": optimized_pattern["nearest_iso_db"],
                "baseline_local_iso_db": baseline_pattern["local_iso_db"],
                "optimized_local_iso_db": optimized_pattern["local_iso_db"],
                "baseline_weakest_target_gain_db": baseline_pattern["weakest_target_gain_db"],
                "optimized_weakest_target_gain_db": optimized_pattern["weakest_target_gain_db"],
                "optimized_target_spread_db": optimized_pattern["target_spread_db"],
                "optimized_pointing_error_deg": optimized_pattern["pointing_error_deg"],
                "equal_weakest_gain_power_ratio": power_ratio_equal_weakest,
                "gate15": int(gate15),
                "strict_gate20": int(gate20),
                "mainlobe_gate": int(mainlobe_gate),
                "robust_active_RL_gate": int(robust_active),
                "strict_engineering_gate": int(strict_engineering),
            }
        )
        print(
            f"candidate {output_pos + 1}/{candidate_count} (index {candidate}): "
            f"RL {float(baseline_rl['worst_active_rl_db']):.2f} -> "
            f"{float(selected['combined_worst_active_rl_db']):.2f} dB, "
            f"joint={int(strict_engineering)}"
        )

    write_csv(args.out_dir / "optimization_trials.csv", trial_rows)
    write_csv(args.out_dir / "optimization_candidate_metrics.csv", candidate_rows)
    np.savez_compressed(
        args.out_dir / "optimized_task_weights.npz",
        candidate_index=np.asarray(data["candidate_index"][candidate_indices]),
        sample_index=np.asarray(data["sample_index"][candidate_indices]),
        mask=np.asarray(data["mask"][candidate_indices]),
        task_valid=np.asarray(data["task_valid"][candidate_indices]),
        targets_deg=np.asarray(data["targets_deg"][candidate_indices]),
        optimized_external_task_weights=selected_tasks,
        optimized_w_tasks_real_imag=np.stack(
            [np.conjugate(selected_tasks).real, np.conjugate(selected_tasks).imag], axis=-1
        ).astype(np.float32),
        optimized_external_combined_weights=np.sum(selected_tasks, axis=2),
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(candidate_rows),
        "runtime_seconds": time.time() - started,
        "configuration_names": [config.name for config in CONFIGS],
        "baseline_combined_active_gate_count": int(
            sum(
                float(row["baseline_combined_worst_active_rl_db"]) >= float(args.return_loss_min_db)
                and float(row["baseline_combined_total_rl_db"]) >= float(args.return_loss_min_db)
                for row in candidate_rows
            )
        ),
        "optimized_combined_active_gate_count": int(
            sum(int(row["optimized_combined_active_gate"]) for row in candidate_rows)
        ),
        "optimized_robust_active_gate_count": int(
            sum(int(row["robust_active_RL_gate"]) for row in candidate_rows)
        ),
        "optimized_gate15_count": int(sum(int(row["gate15"]) for row in candidate_rows)),
        "optimized_strict_gate20_count": int(sum(int(row["strict_gate20"]) for row in candidate_rows)),
        "optimized_mainlobe_gate_count": int(sum(int(row["mainlobe_gate"]) for row in candidate_rows)),
        "optimized_strict_engineering_gate_count": int(
            sum(int(row["strict_engineering_gate"]) for row in candidate_rows)
        ),
        "mean_combined_worst_active_rl_improvement_db": float(
            np.mean(
                [
                    float(row["optimized_combined_worst_active_rl_db"])
                    - float(row["baseline_combined_worst_active_rl_db"])
                    for row in candidate_rows
                ]
            )
        ),
        "max_optimized_combined_worst_active_rl_db": float(
            max(float(row["optimized_combined_worst_active_rl_db"]) for row in candidate_rows)
        ),
        "hfss_labels_allowed": bool(
            any(int(row["strict_engineering_gate"]) for row in candidate_rows)
        ),
    }
    (args.out_dir / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
