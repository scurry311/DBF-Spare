#!/usr/bin/env python3
"""Refine trusted task weights with a dense local-5deg EEP operator."""

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

from hfss_task_fullwave_validate import pattern_grid_dirs, unit_vector
from optimize_trusted_eep_s256_joint_weights import (
    active_return,
    apply_combined_floor,
    normalize,
    pattern_metrics,
    reflection_gradient,
    write_csv,
)


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
DEFAULT_WARM_START = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_s256_joint_optimization_20260724_run03"
    / "optimized_task_weights.npz"
)
DEFAULT_PRIOR_METRICS = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_s256_joint_optimization_20260724_run03"
    / "optimization_candidate_metrics.csv"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_dense_local_eep_joint_20260724_run02"
EPS = 1.0e-12


@dataclass(frozen=True)
class DenseConfig:
    name: str
    equalize_combined_targets: bool
    cycles: int
    active_steps: int
    step_size: float
    combined_penalty: float
    task_penalty: float
    total_penalty: float
    amplitude_floor_db: float
    dense_passes: int
    dense_top_count: int
    projection_margin_db: float


CONFIGS = (
    DenseConfig("dense_preserve30", False, 28, 2, 0.018, 24.0, 4.0, 2.0, -30.0, 2, 64, 2.0),
    DenseConfig("dense_equal20", True, 36, 2, 0.014, 28.0, 5.0, 2.5, -20.0, 3, 96, 3.0),
)


@dataclass
class DenseTaskConstraint:
    active: np.ndarray
    equality_row: np.ndarray
    desired: complex
    leakage_rows: np.ndarray
    leakage_bounds: np.ndarray
    leakage_kind: np.ndarray
    leakage_row_norm_sq: np.ndarray


@dataclass
class CombinedConstraint:
    active: np.ndarray
    rows: np.ndarray
    preserve_desired: np.ndarray
    equal_desired: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--excitations", type=Path, default=DEFAULT_EXCITATIONS)
    parser.add_argument("--warm-start", type=Path, default=DEFAULT_WARM_START)
    parser.add_argument("--prior-metrics", type=Path, default=DEFAULT_PRIOR_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--local-radius-deg", type=float, default=5.0)
    parser.add_argument("--nearest-isolation-db", type=float, default=25.0)
    parser.add_argument("--local-isolation-db", type=float, default=20.0)
    parser.add_argument("--return-loss-min-db", type=float, default=10.0)
    parser.add_argument("--task-significant-relative-db", type=float, default=-20.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class DenseExternalEEP:
    def __init__(
        self,
        etheta: np.ndarray,
        ephi: np.ndarray,
        antenna_map: np.ndarray,
    ) -> None:
        started = time.time()
        map_t = np.asarray(antenna_map, dtype=np.complex64).T
        self.etheta = map_t @ np.asarray(etheta, dtype=np.complex64)
        self.ephi = map_t @ np.asarray(ephi, dtype=np.complex64)
        self.build_seconds = time.time() - started

    def component_rows(self, indices: np.ndarray, active: np.ndarray) -> np.ndarray:
        theta_rows = self.etheta[np.ix_(active, indices)].T
        phi_rows = self.ephi[np.ix_(active, indices)].T
        return np.concatenate((theta_rows, phi_rows), axis=0).astype(np.complex64, copy=False)

    def point_rows(self, index: int, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.etheta[active, index], dtype=np.complex128),
            np.asarray(self.ephi[active, index], dtype=np.complex128),
        )


def nearest_grid_index(grid_dirs: np.ndarray, theta_deg: float, phi_deg: float) -> int:
    return int(np.argmax(grid_dirs @ unit_vector(theta_deg, phi_deg)))


def dense_local_indices(
    grid_dirs: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    radius_deg: float,
) -> np.ndarray:
    target = unit_vector(theta_deg, phi_deg)
    distance = np.rad2deg(np.arccos(np.clip(grid_dirs @ target, -1.0, 1.0)))
    candidates = np.flatnonzero(distance <= radius_deg + 1.0e-7)
    rounded = np.round(grid_dirs[candidates], decimals=7)
    _unique, positions = np.unique(rounded, axis=0, return_index=True)
    return candidates[np.sort(positions)]


def build_constraints(
    original_tasks: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    grid_dirs: np.ndarray,
    operator: DenseExternalEEP,
    *,
    local_radius_deg: float,
    nearest_isolation_db: float,
    local_isolation_db: float,
) -> tuple[list[DenseTaskConstraint], CombinedConstraint, dict[str, Any]]:
    active = np.flatnonzero(mask)
    nearest_ratio = 10.0 ** (-nearest_isolation_db / 20.0)
    local_ratio = 10.0 ** (-local_isolation_db / 20.0)
    target_centers = [
        nearest_grid_index(grid_dirs, float(theta), float(phi)) for theta, phi in targets
    ]
    target_regions = [
        dense_local_indices(
            grid_dirs, float(theta), float(phi), float(local_radius_deg)
        )
        for theta, phi in targets
    ]
    constraints: list[DenseTaskConstraint] = []
    total_dense_points = 0
    original_active = original_tasks[active]
    for task_index, center in enumerate(target_centers):
        own_theta, own_phi = operator.point_rows(center, active)
        initial = original_active[:, task_index]
        own_field = np.asarray([initial @ own_theta, initial @ own_phi])
        own_norm = max(float(np.linalg.norm(own_field)), EPS)
        polarization = own_field / own_norm
        equality_row = (
            np.conjugate(polarization[0]) * own_theta
            + np.conjugate(polarization[1]) * own_phi
        )
        desired = complex(initial @ equality_row)
        rows: list[np.ndarray] = []
        bounds: list[np.ndarray] = []
        kinds: list[np.ndarray] = []
        for other_index, other_center in enumerate(target_centers):
            if other_index == task_index:
                continue
            center_rows = operator.component_rows(
                np.asarray([other_center], dtype=int), active
            )
            rows.append(center_rows)
            bounds.append(
                np.full(center_rows.shape[0], own_norm * nearest_ratio / math.sqrt(2.0))
            )
            kinds.append(np.zeros(center_rows.shape[0], dtype=np.int8))
            region = target_regions[other_index]
            region = region[region != other_center]
            local_rows = operator.component_rows(region, active)
            rows.append(local_rows)
            bounds.append(
                np.full(local_rows.shape[0], own_norm * local_ratio / math.sqrt(2.0))
            )
            kinds.append(np.ones(local_rows.shape[0], dtype=np.int8))
            total_dense_points += int(region.size)
        leakage_rows = np.concatenate(rows, axis=0) if rows else np.zeros((0, active.size), np.complex64)
        leakage_bounds = np.concatenate(bounds) if bounds else np.zeros(0, dtype=np.float64)
        leakage_kind = np.concatenate(kinds) if kinds else np.zeros(0, dtype=np.int8)
        constraints.append(
            DenseTaskConstraint(
                active=active,
                equality_row=equality_row,
                desired=desired,
                leakage_rows=leakage_rows,
                leakage_bounds=leakage_bounds,
                leakage_kind=leakage_kind,
                leakage_row_norm_sq=np.maximum(
                    np.sum(np.abs(leakage_rows) ** 2, axis=1), EPS
                ),
            )
        )

    original_combined = np.sum(original_active, axis=1)
    combined_rows: list[np.ndarray] = []
    combined_desired: list[complex] = []
    for center in target_centers:
        row_theta, row_phi = operator.point_rows(center, active)
        field = np.asarray(
            [original_combined @ row_theta, original_combined @ row_phi]
        )
        norm = max(float(np.linalg.norm(field)), EPS)
        polarization = field / norm
        row = (
            np.conjugate(polarization[0]) * row_theta
            + np.conjugate(polarization[1]) * row_phi
        )
        combined_rows.append(row)
        combined_desired.append(complex(original_combined @ row))
    preserve = np.asarray(combined_desired, dtype=np.complex128)
    common = max(float(np.min(np.abs(preserve))), EPS)
    equal = np.full(preserve.shape, common + 0.0j, dtype=np.complex128)
    combined = CombinedConstraint(
        active=active,
        rows=np.stack(combined_rows),
        preserve_desired=preserve,
        equal_desired=equal,
    )
    diagnostics = {
        "unique_local_points_min": int(min(region.size for region in target_regions)),
        "unique_local_points_mean": float(np.mean([region.size for region in target_regions])),
        "unique_local_points_max": int(max(region.size for region in target_regions)),
        "dense_task_region_point_uses": total_dense_points,
    }
    return constraints, combined, diagnostics


def project_equality(value: np.ndarray, row: np.ndarray, desired: complex) -> None:
    denom = max(float(np.vdot(row, row).real), EPS)
    value += np.conjugate(row) * ((desired - complex(row @ value)) / denom)


def project_dense_task(
    value: np.ndarray,
    constraint: DenseTaskConstraint,
    *,
    passes: int,
    top_count: int,
    margin_db: float,
) -> np.ndarray:
    out = np.asarray(value, dtype=np.complex128).copy()
    active_value = out[constraint.active].copy()
    project_equality(active_value, constraint.equality_row, constraint.desired)
    if constraint.leakage_rows.shape[0]:
        bounds = constraint.leakage_bounds * 10.0 ** (-margin_db / 20.0)
        for _ in range(max(1, passes)):
            responses = constraint.leakage_rows @ active_value
            ratios = np.abs(responses) / np.maximum(bounds, EPS)
            count = min(int(top_count), ratios.size)
            if count <= 0 or float(np.max(ratios)) <= 1.0:
                break
            selected = np.argpartition(ratios, -count)[-count:]
            selected = selected[np.argsort(ratios[selected])[::-1]]
            for index in selected:
                row = constraint.leakage_rows[index]
                response = complex(row @ active_value)
                magnitude = abs(response)
                bound = float(bounds[index])
                if magnitude <= bound:
                    continue
                target = response * (bound / magnitude)
                active_value += np.conjugate(row) * (
                    (target - response) / float(constraint.leakage_row_norm_sq[index])
                )
            project_equality(active_value, constraint.equality_row, constraint.desired)
    out[constraint.active] = active_value
    active_mask = np.zeros(out.size, dtype=bool)
    active_mask[constraint.active] = True
    out[~active_mask] = 0.0
    return out


def project_combined_targets(
    tasks: np.ndarray,
    constraint: CombinedConstraint,
    desired: np.ndarray,
) -> None:
    combined = np.sum(tasks[constraint.active], axis=1)
    residual = desired - constraint.rows @ combined
    gram = constraint.rows @ constraint.rows.conj().T
    regularization = 1.0e-9 * max(
        float(np.trace(gram).real / max(gram.shape[0], 1)), 1.0
    )
    correction = constraint.rows.conj().T @ np.linalg.solve(
        gram + regularization * np.eye(gram.shape[0], dtype=np.complex128),
        residual,
    )
    tasks[constraint.active] += correction[:, None] / tasks.shape[1]


def dense_constraint_metrics(
    tasks: np.ndarray,
    constraints: list[DenseTaskConstraint],
    combined: CombinedConstraint,
    desired: np.ndarray,
) -> dict[str, float | int]:
    target_error = 0.0
    nearest_ratio = 0.0
    local_ratio = 0.0
    for task_index, constraint in enumerate(constraints):
        active_value = tasks[constraint.active, task_index]
        response = complex(constraint.equality_row @ active_value)
        target_error = max(
            target_error,
            abs(response - constraint.desired) / max(abs(constraint.desired), EPS),
        )
        if constraint.leakage_rows.shape[0]:
            ratios = np.abs(constraint.leakage_rows @ active_value) / np.maximum(
                constraint.leakage_bounds, EPS
            )
            nearest = ratios[constraint.leakage_kind == 0]
            local = ratios[constraint.leakage_kind == 1]
            if nearest.size:
                nearest_ratio = max(nearest_ratio, float(np.max(nearest)))
            if local.size:
                local_ratio = max(local_ratio, float(np.max(local)))
    combined_value = np.sum(tasks[combined.active], axis=1)
    combined_response = combined.rows @ combined_value
    combined_error = float(
        np.max(np.abs(combined_response - desired) / np.maximum(np.abs(desired), EPS))
    )
    return {
        "task_target_error_max": float(target_error),
        "dense_nearest_bound_ratio_max": float(nearest_ratio),
        "dense_local_bound_ratio_max": float(local_ratio),
        "combined_target_error_max": combined_error,
        "dense_constraint_pass": int(
            target_error <= 0.02
            and nearest_ratio <= 1.05
            and local_ratio <= 1.05
            and combined_error <= 0.02
        ),
    }


def refine_one(
    warm_tasks: np.ndarray,
    original_tasks: np.ndarray,
    mask: np.ndarray,
    constraints: list[DenseTaskConstraint],
    combined_constraint: CombinedConstraint,
    s_matrix: np.ndarray,
    config: DenseConfig,
    *,
    rl_min_db: float,
    task_relative_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    tasks = np.asarray(warm_tasks, dtype=np.complex128).copy()
    tasks[~mask] = 0.0
    original_combined = np.sum(original_tasks, axis=1)
    desired = (
        combined_constraint.equal_desired
        if config.equalize_combined_targets
        else combined_constraint.preserve_desired
    )
    rho = 10.0 ** (-rl_min_db / 20.0)
    best = tasks.copy()
    best_key: tuple[int, int, int, float, float] = (-1, -1, -1, -1.0e9, -1.0e9)
    best_cycle = 0
    for cycle in range(config.cycles + 1):
        combined_value = np.sum(tasks, axis=1)
        combined_rl = active_return(
            s_matrix, combined_value, mask, relative_db=None, threshold_db=rl_min_db
        )
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
        dense = dense_constraint_metrics(
            tasks, constraints, combined_constraint, desired
        )
        task_gate = int(all(int(item["gate_pass"]) for item in task_rl))
        minimum_rl = min(
            float(combined_rl["worst_active_rl_db"]),
            float(combined_rl["total_rl_db"]),
            *(float(item["worst_active_rl_db"]) for item in task_rl),
            *(float(item["total_rl_db"]) for item in task_rl),
        )
        key = (
            int(combined_rl["gate_pass"] and task_gate and dense["dense_constraint_pass"]),
            int(combined_rl["gate_pass"] and task_gate),
            int(dense["dense_constraint_pass"]),
            minimum_rl,
            -max(
                float(dense["dense_nearest_bound_ratio_max"]),
                float(dense["dense_local_bound_ratio_max"]),
            ),
        )
        if key > best_key:
            best_key = key
            best = tasks.copy()
            best_cycle = cycle
        if cycle == config.cycles:
            break

        for _ in range(config.active_steps):
            combined_value = np.sum(tasks, axis=1)
            gradient_combined = reflection_gradient(
                s_matrix, combined_value, mask, rho, config.total_penalty
            )
            gradient_combined *= config.combined_penalty
            combined_step = (
                config.step_size
                * max(float(np.linalg.norm(combined_value)), EPS)
                * gradient_combined
                / max(float(np.linalg.norm(gradient_combined)), EPS)
            )
            for task_index in range(tasks.shape[1]):
                value = tasks[:, task_index]
                amplitude = np.abs(value)
                maximum = max(float(np.max(amplitude)), EPS)
                significant = mask & (
                    amplitude >= maximum * 10.0 ** (task_relative_db / 20.0)
                )
                gradient_task = reflection_gradient(
                    s_matrix, value, significant, rho, config.total_penalty
                )
                gradient_task *= config.task_penalty
                tasks[:, task_index] -= combined_step / tasks.shape[1]
                tasks[:, task_index] -= (
                    config.step_size
                    * max(float(np.linalg.norm(value)), EPS)
                    * gradient_task
                    / max(float(np.linalg.norm(gradient_task)), EPS)
                )
                tasks[~mask, task_index] = 0.0
            apply_combined_floor(
                tasks, original_combined, mask, config.amplitude_floor_db
            )

        for task_index, constraint in enumerate(constraints):
            tasks[:, task_index] = project_dense_task(
                tasks[:, task_index],
                constraint,
                passes=config.dense_passes,
                top_count=config.dense_top_count,
                margin_db=config.projection_margin_db,
            )
        project_combined_targets(tasks, combined_constraint, desired)
        tasks[~mask] = 0.0

    tasks = best
    combined_rl = active_return(
        s_matrix, np.sum(tasks, axis=1), mask, relative_db=None, threshold_db=rl_min_db
    )
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
    strict_task_rl = [
        active_return(
            s_matrix, tasks[:, task_index], mask, relative_db=None, threshold_db=rl_min_db
        )
        for task_index in range(tasks.shape[1])
    ]
    return tasks.astype(np.complex64), {
        "config": config.name,
        "best_cycle": best_cycle,
        "combined_worst_active_rl_db": combined_rl["worst_active_rl_db"],
        "combined_total_rl_db": combined_rl["total_rl_db"],
        "combined_active_gate": combined_rl["gate_pass"],
        "combined_dynamic_range_db": combined_rl["dynamic_range_db"],
        "all_tasks_significant_worst_active_rl_db": min(
            float(item["worst_active_rl_db"]) for item in task_rl
        ),
        "all_tasks_significant_worst_total_rl_db": min(
            float(item["total_rl_db"]) for item in task_rl
        ),
        "all_tasks_significant_gate": int(
            all(int(item["gate_pass"]) for item in task_rl)
        ),
        "all_tasks_strict_worst_active_rl_db": min(
            float(item["worst_active_rl_db"]) for item in strict_task_rl
        ),
        "all_tasks_strict_gate": int(
            all(int(item["gate_pass"]) for item in strict_task_rl)
        ),
        **dense_constraint_metrics(tasks, constraints, combined_constraint, desired),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dense refinement: {args.out_dir}")
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
    with np.load(args.warm_start, allow_pickle=False) as source:
        warm_candidate_indices = np.asarray(source["candidate_index"], dtype=int)
        warm_all = np.asarray(source["optimized_external_task_weights"], dtype=np.complex64)
    warm_by_candidate = {
        int(candidate): warm_all[position]
        for position, candidate in enumerate(warm_candidate_indices)
    }
    prior_by_candidate = {
        int(float(row["candidate_index"])): row for row in read_csv(args.prior_metrics)
    }

    dense_operator = DenseExternalEEP(etheta, ephi, antenna_map)
    grid_dirs = pattern_grid_dirs(theta, phi)
    task_weights = data["w_tasks_real_imag"][..., 0] + 1j * data["w_tasks_real_imag"][..., 1]
    total_count = int(data["candidate_index"].size)
    start_index = max(0, int(args.start_index))
    stop_index = total_count
    if int(args.max_candidates) > 0:
        stop_index = min(stop_index, start_index + int(args.max_candidates))
    candidate_indices = list(range(start_index, stop_index))
    if not candidate_indices:
        raise ValueError("Candidate selection is empty")

    selected_tasks = np.zeros((len(candidate_indices), 256, 6), dtype=np.complex64)
    trial_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    dense_point_stats: list[dict[str, Any]] = []
    for output_pos, candidate in enumerate(candidate_indices):
        k_value = int(data["k_values"][candidate])
        mask = np.asarray(data["mask"][candidate], dtype=bool)
        original_tasks = np.conjugate(
            np.asarray(task_weights[candidate, :, :k_value], dtype=np.complex128)
        )
        original_tasks[~mask] = 0.0
        warm_tasks = np.asarray(
            warm_by_candidate[candidate][:, :k_value], dtype=np.complex128
        )
        targets = np.asarray(data["targets_deg"][candidate, :k_value], dtype=np.float64)
        constraints, combined_constraint, point_stats = build_constraints(
            original_tasks,
            mask,
            targets,
            grid_dirs,
            dense_operator,
            local_radius_deg=float(args.local_radius_deg),
            nearest_isolation_db=float(args.nearest_isolation_db),
            local_isolation_db=float(args.local_isolation_db),
        )
        dense_point_stats.append(point_stats)
        baseline_pattern = pattern_metrics(
            original_tasks, targets, theta, phi, etheta, ephi, antenna_map
        )
        trials: list[tuple[np.ndarray, dict[str, Any]]] = []
        for config in CONFIGS:
            refined, metrics = refine_one(
                warm_tasks,
                original_tasks,
                mask,
                constraints,
                combined_constraint,
                s_matrix,
                config,
                rl_min_db=float(args.return_loss_min_db),
                task_relative_db=float(args.task_significant_relative_db),
            )
            refined_pattern = pattern_metrics(
                refined, targets, theta, phi, etheta, ephi, antenna_map
            )
            mainlobe_gate = bool(
                refined_pattern["weakest_target_gain_db"]
                >= baseline_pattern["weakest_target_gain_db"] - 0.5
                and refined_pattern["target_spread_db"] <= 3.0
                and refined_pattern["pointing_error_deg"] <= 1.5
            )
            gate15 = bool(
                refined_pattern["psll_db"] <= 0.0
                and refined_pattern["nearest_iso_db"] >= 25.0
                and refined_pattern["local_iso_db"] >= 15.0
            )
            gate20 = bool(
                refined_pattern["psll_db"] <= 0.0
                and refined_pattern["nearest_iso_db"] >= 25.0
                and refined_pattern["local_iso_db"] >= 20.0
            )
            robust_active = bool(
                metrics["combined_active_gate"]
                and metrics["all_tasks_significant_gate"]
            )
            strict_engineering = bool(gate20 and mainlobe_gate and robust_active)
            trial = {
                **metrics,
                **{f"pattern_{key}": value for key, value in refined_pattern.items()},
                "gate15": int(gate15),
                "strict_gate20": int(gate20),
                "mainlobe_gate": int(mainlobe_gate),
                "robust_active_RL_gate": int(robust_active),
                "strict_engineering_gate": int(strict_engineering),
            }
            trial_rows.append(
                {
                    "candidate_index": candidate,
                    "sample_index": int(data["sample_index"][candidate]),
                    "k": k_value,
                    "ratio": float(data["active_ratios_requested"][candidate]),
                    **trial,
                }
            )
            trials.append((refined, trial))
        trials.sort(
            key=lambda item: (
                int(item[1]["strict_engineering_gate"]),
                int(item[1]["robust_active_RL_gate"] and item[1]["strict_gate20"] and item[1]["mainlobe_gate"]),
                int(item[1]["robust_active_RL_gate"]),
                int(item[1]["strict_gate20"]),
                int(item[1]["mainlobe_gate"]),
                float(item[1]["pattern_local_iso_db"]),
                -float(item[1]["pattern_psll_db"]),
            ),
            reverse=True,
        )
        refined_tasks, selected = trials[0]
        selected_tasks[output_pos, :, :k_value] = refined_tasks
        prior = prior_by_candidate[candidate]
        power_ratio = float(
            10.0 ** (
                (
                    baseline_pattern["weakest_target_gain_db"]
                    - float(selected["pattern_weakest_target_gain_db"])
                )
                / 10.0
            )
        )
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
                "prior_combined_active_rl_db": float(prior["optimized_combined_worst_active_rl_db"]),
                "refined_combined_active_rl_db": selected["combined_worst_active_rl_db"],
                "refined_combined_total_rl_db": selected["combined_total_rl_db"],
                "refined_task_significant_active_rl_db": selected["all_tasks_significant_worst_active_rl_db"],
                "refined_task_strict_active_rl_db": selected["all_tasks_strict_worst_active_rl_db"],
                "dense_nearest_bound_ratio_max": selected["dense_nearest_bound_ratio_max"],
                "dense_local_bound_ratio_max": selected["dense_local_bound_ratio_max"],
                "combined_target_error_max": selected["combined_target_error_max"],
                "baseline_psll_db": baseline_pattern["psll_db"],
                "refined_psll_db": selected["pattern_psll_db"],
                "baseline_nearest_iso_db": baseline_pattern["nearest_iso_db"],
                "refined_nearest_iso_db": selected["pattern_nearest_iso_db"],
                "baseline_local_iso_db": baseline_pattern["local_iso_db"],
                "refined_local_iso_db": selected["pattern_local_iso_db"],
                "baseline_weakest_target_gain_db": baseline_pattern["weakest_target_gain_db"],
                "refined_weakest_target_gain_db": selected["pattern_weakest_target_gain_db"],
                "refined_target_spread_db": selected["pattern_target_spread_db"],
                "refined_pointing_error_deg": selected["pattern_pointing_error_deg"],
                "equal_weakest_gain_power_ratio": power_ratio,
                "gate15": selected["gate15"],
                "strict_gate20": selected["strict_gate20"],
                "mainlobe_gate": selected["mainlobe_gate"],
                "combined_active_gate": selected["combined_active_gate"],
                "robust_active_RL_gate": selected["robust_active_RL_gate"],
                "strict_engineering_gate": selected["strict_engineering_gate"],
                **point_stats,
            }
        )
        print(
            f"candidate {output_pos + 1}/{len(candidate_indices)} (index {candidate}, K={k_value}): "
            f"local={float(selected['pattern_local_iso_db']):.2f} dB, "
            f"RL={float(selected['combined_worst_active_rl_db']):.2f} dB, "
            f"joint={int(selected['strict_engineering_gate'])}",
            flush=True,
        )

    write_csv(args.out_dir / "dense_refinement_trials.csv", trial_rows)
    write_csv(args.out_dir / "dense_refinement_candidate_metrics.csv", candidate_rows)
    np.savez_compressed(
        args.out_dir / "dense_refined_task_weights.npz",
        candidate_index=np.asarray(data["candidate_index"][candidate_indices]),
        sample_index=np.asarray(data["sample_index"][candidate_indices]),
        mask=np.asarray(data["mask"][candidate_indices]),
        task_valid=np.asarray(data["task_valid"][candidate_indices]),
        targets_deg=np.asarray(data["targets_deg"][candidate_indices]),
        refined_external_task_weights=selected_tasks,
        refined_external_combined_weights=np.sum(selected_tasks, axis=2),
        refined_w_tasks_real_imag=np.stack(
            [np.conjugate(selected_tasks).real, np.conjugate(selected_tasks).imag], axis=-1
        ).astype(np.float32),
    )
    sparse_multibeam = [
        row
        for row in candidate_rows
        if float(row["ratio"]) < 0.999
        and int(row["k"]) in (2, 4, 6)
        and int(row["strict_engineering_gate"])
    ]
    sparse_k6 = [row for row in sparse_multibeam if int(row["k"]) == 6]
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(candidate_rows),
        "runtime_seconds": time.time() - started,
        "dense_operator_build_seconds": dense_operator.build_seconds,
        "local_radius_deg": float(args.local_radius_deg),
        "unique_local_points_min": min(int(row["unique_local_points_min"]) for row in dense_point_stats),
        "unique_local_points_mean": float(np.mean([row["unique_local_points_mean"] for row in dense_point_stats])),
        "unique_local_points_max": max(int(row["unique_local_points_max"]) for row in dense_point_stats),
        "combined_active_gate_count": int(sum(int(row["combined_active_gate"]) for row in candidate_rows)),
        "robust_active_gate_count": int(sum(int(row["robust_active_RL_gate"]) for row in candidate_rows)),
        "gate15_count": int(sum(int(row["gate15"]) for row in candidate_rows)),
        "strict_gate20_count": int(sum(int(row["strict_gate20"]) for row in candidate_rows)),
        "mainlobe_gate_count": int(sum(int(row["mainlobe_gate"]) for row in candidate_rows)),
        "strict_engineering_gate_count": int(sum(int(row["strict_engineering_gate"]) for row in candidate_rows)),
        "sparse_multibeam_strict_positive_count": len(sparse_multibeam),
        "sparse_k6_strict_positive_count": len(sparse_k6),
        "hfss_shortlist_gate_pass": bool(len(sparse_k6) >= 1 and len(sparse_multibeam) >= 5),
    }
    (args.out_dir / "dense_refinement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
