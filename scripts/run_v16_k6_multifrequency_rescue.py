#!/usr/bin/env python3
"""Run the preregistered K=6 multi-frequency mainlobe/active-RL rescue."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import metric_at
from generate_v09_eep_development_candidates import (
    METRIC_NAMES,
    full_active_metrics,
    metric_vector,
    physical_margins,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from optimize_trusted_eep_s256_joint_weights import reflection_gradient
from refine_trusted_dense_local_eep_joint import (
    DenseConfig,
    build_constraints,
    dense_local_indices,
    nearest_grid_index,
    refine_one,
)
from run_v16_robust_drift_oracle import (
    ROBUST_MARGIN_NAMES,
    apply_analog_state,
    apply_calibration,
    build_corners,
    complex_to_ri,
    hardware_margin,
    load_nominal_operator,
    load_npz,
    read_csv,
    ri_to_complex,
    scene_calibration_states,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "hfss_outputs" / "v16_robust_drift_oracle_20260727_run01"
DEFAULT_PROTOCOL = ROOT / "configs" / "v16_k6_multifrequency_rescue_preregistered.json"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v16_k6_multifrequency_rescue_20260727_run01"
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_DRIFT = ROOT / "hfss_outputs" / "v14_operator_drift_4x4_smoke_20260727_run01"
EPS = 1.0e-12
KMAX = 6


@dataclass(frozen=True)
class Variant:
    name: str
    target_amplitude_mode: str
    active_rl_design_min_db: float
    local_radius_deg: float
    regional_ceiling_db: float
    joint_projection_passes: int
    corner_sweeps: int


@dataclass
class JointCombinedConstraint:
    active: np.ndarray
    equality_rows: np.ndarray
    equality_desired: np.ndarray
    equality_gram: np.ndarray
    regional_rows: np.ndarray
    regional_bounds: np.ndarray
    regional_row_norm_sq: np.ndarray
    corner_names: np.ndarray


OPTIMIZER_CONFIG = DenseConfig(
    "v16_k6_multifrequency_reserve",
    True,
    6,
    3,
    0.010,
    50.0,
    15.0,
    6.0,
    -16.0,
    2,
    96,
    3.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "evaluate", "all"))
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--additional-parent-dir", type=Path, action="append", default=[])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--max-parents-per-ratio", type=int, default=0)
    parser.add_argument("--max-variants", type=int, default=0)
    return parser.parse_args()


def load_protocol(path: Path) -> tuple[dict[str, Any], list[Variant]]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if float(protocol["frozen_E2"]["intensity"]) != 0.20:
        raise RuntimeError("The frozen E2 intensity changed")
    variants = [Variant(**row) for row in protocol["design_variants"]]
    return protocol, variants


def project_equalities(value: np.ndarray, constraint: JointCombinedConstraint) -> np.ndarray:
    residual = constraint.equality_desired - constraint.equality_rows @ value
    scale = max(float(np.trace(constraint.equality_gram).real / len(residual)), 1.0)
    correction = constraint.equality_rows.conj().T @ np.linalg.solve(
        constraint.equality_gram
        + (1.0e-7 * scale) * np.eye(len(residual), dtype=np.complex128),
        residual,
    )
    return value + correction


def build_joint_combined_constraint(
    command: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    grid_dirs: np.ndarray,
    *,
    local_radius_deg: float,
    regional_ceiling_db: float,
    target_amplitude_mode: str,
) -> JointCombinedConstraint:
    active = np.flatnonzero(mask)
    combined = np.sum(command[active], axis=1).astype(np.complex128)
    equality_rows: list[np.ndarray] = []
    equality_desired: list[complex] = []
    regional_rows: list[np.ndarray] = []
    regional_bounds: list[float] = []
    corner_labels: list[str] = []
    ceiling = 10.0 ** (float(regional_ceiling_db) / 20.0)
    for corner_name, corner in corners.items():
        factor = np.asarray(states[corner_name]["factor"], dtype=np.complex128)[active]
        center_rows: list[np.ndarray] = []
        center_responses: list[complex] = []
        centers: list[int] = []
        for theta, phi in targets:
            center = nearest_grid_index(grid_dirs, float(theta), float(phi))
            centers.append(center)
            row_theta = factor * np.asarray(corner["effective"].etheta[active, center], dtype=np.complex128)
            row_phi = factor * np.asarray(corner["effective"].ephi[active, center], dtype=np.complex128)
            field = np.asarray([combined @ row_theta, combined @ row_phi])
            field_norm = max(float(np.linalg.norm(field)), EPS)
            polarization = field / field_norm
            row = np.conjugate(polarization[0]) * row_theta + np.conjugate(polarization[1]) * row_phi
            center_rows.append(row)
            center_responses.append(complex(combined @ row))
        center_amplitudes = np.asarray([abs(value) for value in center_responses], dtype=float)
        if target_amplitude_mode == "weakest":
            common_amplitude = max(float(np.min(center_amplitudes)), EPS)
        elif target_amplitude_mode == "median":
            common_amplitude = max(float(np.median(center_amplitudes)), EPS)
        elif target_amplitude_mode == "strongest":
            common_amplitude = max(float(np.max(center_amplitudes)), EPS)
        else:
            raise ValueError(f"Unknown target amplitude mode: {target_amplitude_mode}")
        for target_index, ((theta, phi), center, row, response) in enumerate(
            zip(targets, centers, center_rows, center_responses)
        ):
            equality_rows.append(row)
            equality_desired.append(common_amplitude * np.exp(1j * np.angle(response)))
            corner_labels.append(corner_name)
            local = dense_local_indices(
                grid_dirs, float(theta), float(phi), float(local_radius_deg)
            )
            local = local[local != center]
            for grid_index in local:
                local_theta = factor * np.asarray(
                    corner["effective"].etheta[active, grid_index], dtype=np.complex128
                )
                local_phi = factor * np.asarray(
                    corner["effective"].ephi[active, grid_index], dtype=np.complex128
                )
                local_field = np.asarray([combined @ local_theta, combined @ local_phi])
                local_norm = max(float(np.linalg.norm(local_field)), EPS)
                local_polarization = local_field / local_norm
                regional_rows.append(
                    np.conjugate(local_polarization[0]) * local_theta
                    + np.conjugate(local_polarization[1]) * local_phi
                )
                regional_bounds.append(common_amplitude * ceiling)
    equality = np.stack(equality_rows)
    regional = np.stack(regional_rows) if regional_rows else np.zeros((0, active.size), np.complex128)
    bounds = np.asarray(regional_bounds, dtype=np.float64)
    return JointCombinedConstraint(
        active=active,
        equality_rows=equality,
        equality_desired=np.asarray(equality_desired, dtype=np.complex128),
        equality_gram=equality @ equality.conj().T,
        regional_rows=regional,
        regional_bounds=bounds,
        regional_row_norm_sq=np.maximum(np.sum(np.abs(regional) ** 2, axis=1), EPS),
        corner_names=np.asarray(corner_labels),
    )


def project_joint_combined(
    tasks: np.ndarray,
    mask: np.ndarray,
    constraint: JointCombinedConstraint,
    *,
    passes: int,
    top_count: int = 192,
) -> np.ndarray:
    output = np.asarray(tasks, dtype=np.complex128).copy()
    before = np.sum(output[constraint.active], axis=1)
    combined = project_equalities(before, constraint)
    for _pass in range(max(1, passes)):
        if not len(constraint.regional_bounds):
            break
        responses = constraint.regional_rows @ combined
        ratios = np.abs(responses) / np.maximum(constraint.regional_bounds, EPS)
        if float(np.max(ratios)) <= 1.0:
            break
        count = min(int(top_count), len(ratios))
        selected = np.argpartition(ratios, -count)[-count:]
        selected = selected[np.argsort(ratios[selected])[::-1]]
        for index in selected:
            row = constraint.regional_rows[index]
            response = complex(row @ combined)
            magnitude = abs(response)
            bound = float(constraint.regional_bounds[index])
            if magnitude <= bound:
                continue
            desired = response * (bound / magnitude)
            combined += np.conjugate(row) * (
                (desired - response) / float(constraint.regional_row_norm_sq[index])
            )
        combined = project_equalities(combined, constraint)
    correction = combined - before
    output[constraint.active] += correction[:, None] / output.shape[1]
    output[~mask] = 0.0
    return output.astype(np.complex64)


def joint_constraint_metrics(
    command: np.ndarray,
    constraint: JointCombinedConstraint,
) -> dict[str, float]:
    combined = np.sum(command[constraint.active], axis=1)
    equality = constraint.equality_rows @ combined
    equality_error = float(
        np.max(
            np.abs(equality - constraint.equality_desired)
            / np.maximum(np.abs(constraint.equality_desired), EPS)
        )
    )
    regional_ratio = 0.0
    if len(constraint.regional_bounds):
        regional_ratio = float(
            np.max(
                np.abs(constraint.regional_rows @ combined)
                / np.maximum(constraint.regional_bounds, EPS)
            )
        )
    return {
        "joint_equality_error_max": equality_error,
        "joint_regional_ratio_max": regional_ratio,
        "joint_equality_condition_number": float(np.linalg.cond(constraint.equality_gram)),
    }


def proxy_active_rl_floor(
    command: np.ndarray,
    mask: np.ndarray,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> float:
    values = []
    for name, corner in corners.items():
        actual = apply_analog_state(command, mask, states[name])
        values.append(float(full_active_metrics(actual, mask, corner["s"])["active_rl_floor_db"]))
    return min(values)


def frozen_implementation_active_rl_floor(
    command: np.ndarray,
    mask: np.ndarray,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> float:
    values = []
    for name, corner in corners.items():
        actual = apply_calibration(command, mask, states[name])
        values.append(float(full_active_metrics(actual, mask, corner["s"])["active_rl_floor_db"]))
    return min(values)


def equality_nullspace(
    value: np.ndarray,
    constraint: JointCombinedConstraint,
) -> np.ndarray:
    active_value = np.asarray(value[constraint.active], dtype=np.complex128)
    residual = constraint.equality_rows @ active_value
    scale = max(
        float(np.trace(constraint.equality_gram).real / constraint.equality_gram.shape[0]),
        1.0,
    )
    projected = active_value - constraint.equality_rows.conj().T @ np.linalg.solve(
        constraint.equality_gram
        + (1.0e-7 * scale)
        * np.eye(constraint.equality_gram.shape[0], dtype=np.complex128),
        residual,
    )
    output = np.zeros_like(value, dtype=np.complex128)
    output[constraint.active] = projected
    return output


def repair_active_rl_in_equality_nullspace(
    command: np.ndarray,
    mask: np.ndarray,
    constraint: JointCombinedConstraint,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    *,
    rl_design_min_db: float,
    steps: int,
    step_size: float,
    quantization_aware_selection: bool,
) -> np.ndarray:
    current = np.asarray(command, dtype=np.complex128).copy()
    rho = 10.0 ** (-float(rl_design_min_db) / 20.0)
    best = current.copy()
    best_key = (-1.0e9, -1.0e9, -1.0e9)
    for step in range(max(1, int(steps)) + 1):
        metrics = joint_constraint_metrics(current, constraint)
        active_floor = (
            frozen_implementation_active_rl_floor(current, mask, corners, states)
            if quantization_aware_selection
            else proxy_active_rl_floor(current, mask, corners, states)
        )
        key = (
            active_floor,
            -metrics["joint_equality_error_max"],
            -metrics["joint_regional_ratio_max"],
        )
        if key > best_key:
            best_key = key
            best = current.copy()
        if step == int(steps):
            break
        combined_gradient = np.zeros(256, dtype=np.complex128)
        task_gradient = np.zeros_like(current, dtype=np.complex128)
        for name, corner in corners.items():
            factor = np.asarray(states[name]["factor"], dtype=np.complex128)
            actual = apply_analog_state(current, mask, states[name]).astype(np.complex128)
            combined_actual = np.sum(actual, axis=1)
            gradient = reflection_gradient(
                corner["s"], combined_actual, mask, rho, 6.0
            )
            gradient *= np.conjugate(factor)
            combined_gradient += gradient / max(float(np.linalg.norm(gradient)), EPS)
            for task_index in range(current.shape[1]):
                value = actual[:, task_index]
                amplitude = np.abs(value)
                maximum = max(float(np.max(amplitude)), EPS)
                significant = mask & (amplitude >= maximum * 0.1)
                gradient_task = reflection_gradient(
                    corner["s"], value, significant, rho, 5.0
                )
                gradient_task *= np.conjugate(factor)
                task_gradient[:, task_index] += gradient_task / max(
                    float(np.linalg.norm(gradient_task)), EPS
                )
        combined_gradient = equality_nullspace(combined_gradient, constraint)
        task_gradient[~mask] = 0.0
        task_gradient -= np.mean(task_gradient, axis=1, keepdims=True)
        combined_norm = max(float(np.linalg.norm(np.sum(current, axis=1))), EPS)
        combined_grad_norm = max(float(np.linalg.norm(combined_gradient)), EPS)
        current -= (
            float(step_size)
            * combined_norm
            * combined_gradient[:, None]
            / (combined_grad_norm * current.shape[1])
        )
        task_norm = max(float(np.linalg.norm(current)), EPS)
        task_grad_norm = max(float(np.linalg.norm(task_gradient)), EPS)
        current -= float(step_size) * task_norm * task_gradient / task_grad_norm
        current[~mask] = 0.0
    return best.astype(np.complex64)


def optimize_common_command(
    initial: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    grid_dirs: np.ndarray,
    variant: Variant,
    nullspace_steps: int,
    nullspace_step_size: float,
    quantization_aware_selection: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    joint = build_joint_combined_constraint(
        initial,
        mask,
        targets,
        corners,
        states,
        grid_dirs,
        local_radius_deg=variant.local_radius_deg,
        regional_ceiling_db=variant.regional_ceiling_db,
        target_amplitude_mode=variant.target_amplitude_mode,
    )
    current = np.asarray(initial, dtype=np.complex64).copy()
    candidates: list[np.ndarray] = []
    for _sweep in range(variant.corner_sweeps):
        for name, corner in corners.items():
            state = states[name]
            factor = np.asarray(state["factor"], dtype=np.complex128)
            actual = apply_analog_state(current, mask, state)
            constraints, combined_constraint, _stats = build_constraints(
                actual,
                mask,
                targets,
                grid_dirs,
                corner["effective"],
                local_radius_deg=5.0,
                nearest_isolation_db=25.0,
                local_isolation_db=20.0,
            )
            refined_actual, _diagnostics = refine_one(
                actual,
                actual,
                mask,
                constraints,
                combined_constraint,
                corner["s"],
                OPTIMIZER_CONFIG,
                rl_min_db=variant.active_rl_design_min_db,
                task_relative_db=-20.0,
            )
            proposed = np.zeros_like(refined_actual)
            proposed[mask] = refined_actual[mask] / factor[mask, None]
            current = (0.75 * proposed + 0.25 * current).astype(np.complex64)
            current[~mask] = 0.0
        current = project_joint_combined(
            current,
            mask,
            joint,
            passes=variant.joint_projection_passes,
        )
        current = repair_active_rl_in_equality_nullspace(
            current,
            mask,
            joint,
            corners,
            states,
            rl_design_min_db=variant.active_rl_design_min_db,
            steps=nullspace_steps,
            step_size=nullspace_step_size,
            quantization_aware_selection=quantization_aware_selection,
        )
        candidates.append(current.copy())

    scored: list[
        tuple[tuple[int, int, int, float, float, float], np.ndarray, dict[str, float]]
    ] = []
    for sweep_index, candidate in enumerate(candidates, start=1):
        joint_metrics = joint_constraint_metrics(candidate, joint)
        analog_floor = proxy_active_rl_floor(candidate, mask, corners, states)
        implementation_floor = frozen_implementation_active_rl_floor(
            candidate, mask, corners, states
        )
        active_floor = implementation_floor if quantization_aware_selection else analog_floor
        key = (
            int(active_floor >= variant.active_rl_design_min_db),
            int(active_floor >= 10.0),
            int(joint_metrics["joint_equality_error_max"] <= 0.05),
            min(active_floor - 10.0, 3.0),
            -joint_metrics["joint_equality_error_max"],
            -joint_metrics["joint_regional_ratio_max"],
        )
        scored.append(
            (
                key,
                candidate,
                {
                    **joint_metrics,
                    "analog_E2_active_rl_floor_db": analog_floor,
                    "frozen_implementation_E2_active_rl_floor_db": implementation_floor,
                    "selected_sweep_index": float(sweep_index),
                },
            )
        )
    _key, selected, diagnostics = max(scored, key=lambda item: item[0])
    return selected, diagnostics


def parent_commands(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    pool = load_npz(args.parent_dir / "pool" / "candidate_pool.npz")
    final_arrays = load_npz(args.parent_dir / "final" / "robust_arrays.npz")
    rescue_arrays = load_npz(args.parent_dir / "rescue" / "rescue_candidates.npz")
    final_rows = read_csv(args.parent_dir / "final" / "robust_candidate_metrics.csv")
    rescue_rows = read_csv(args.parent_dir / "post_rescue" / "rescue_robust_candidate_metrics.csv")
    final_tasks = ri_to_complex(final_arrays["tasks_real_imag"])
    rescue_tasks = ri_to_complex(rescue_arrays["tasks_real_imag"])
    records: list[dict[str, Any]] = []
    for row in final_rows:
        evaluation = int(row["evaluation_index"])
        candidate = int(row["candidate_index"])
        records.append(
            {
                "row": row,
                "source": "final",
                "command": final_tasks[evaluation],
                "mask": np.asarray(pool["masks"][candidate], dtype=bool),
                "targets": np.asarray(pool["targets"][candidate], dtype=np.float32),
            }
        )
    for row in rescue_rows:
        rescue_index = int(row["rescue_index"])
        records.append(
            {
                "row": row,
                "source": "rescue",
                "command": rescue_tasks[rescue_index],
                "mask": np.asarray(rescue_arrays["masks"][rescue_index], dtype=bool),
                "targets": np.asarray(rescue_arrays["targets"][rescue_index], dtype=np.float32),
            }
        )
    for additional_dir in args.additional_parent_dir:
        additional = load_npz(additional_dir / "candidates" / "candidate_commands.npz")
        additional_tasks = ri_to_complex(additional["tasks_real_imag"])
        additional_rows = read_csv(additional_dir / "evaluation" / "candidate_metrics.csv")
        for row in additional_rows:
            candidate = int(row["candidate_index"])
            records.append(
                {
                    "row": row,
                    "source": f"additional:{additional_dir.name}",
                    "command": additional_tasks[candidate],
                    "mask": np.asarray(additional["masks"][candidate], dtype=bool),
                    "targets": np.asarray(additional["targets"][candidate], dtype=np.float32),
                }
            )
    return pool, records


def generate(args: argparse.Namespace) -> None:
    out = args.out_dir / "candidates"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite candidates: {out}")
    out.mkdir(parents=True, exist_ok=True)
    protocol, variants = load_protocol(args.protocol)
    if args.max_variants > 0:
        variants = variants[: int(args.max_variants)]
    pool, parents = parent_commands(args)
    base, _nominal_effective, _fast, _s = load_nominal_operator(args.operator)
    _base, e2_corners = build_corners(args, levels=(0.20,))
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    target_scenes = [int(value) for value in protocol["target_scene_policy"]["sample_indices"]]
    if args.scene_offset < 0 or args.scene_offset >= len(target_scenes):
        raise ValueError(f"scene-offset must be in [0, {len(target_scenes) - 1}]")
    target_scenes = target_scenes[int(args.scene_offset):]
    if args.max_scenes > 0:
        target_scenes = target_scenes[: int(args.max_scenes)]
    parent_count = int(protocol["target_scene_policy"]["parents_per_ratio"])
    if args.max_parents_per_ratio > 0:
        parent_count = min(parent_count, int(args.max_parents_per_ratio))
    arrays: dict[str, list[Any]] = {
        key: []
        for key in (
            "sample_index", "ratio", "targets", "masks", "tasks", "parent_source",
            "parent_evaluation", "mask_hash", "variant", "rl_design", "radius", "ceiling",
            "equality_error", "regional_ratio", "equality_condition", "analog_rl",
            "implementation_rl",
        )
    }
    rows: list[dict[str, Any]] = []
    started = time.time()
    for scene_position, sample in enumerate(target_scenes):
        pool_scene = np.flatnonzero(pool_samples == sample)
        targets = np.asarray(pool["targets"][pool_scene[0], :KMAX], dtype=float)
        states = scene_calibration_states(
            pool,
            pool_scene,
            e2_corners,
            base["element_ixiy"],
            int(protocol["seed"]),
        )
        for ratio in protocol["target_scene_policy"]["ratios"]:
            matching = [
                record
                for record in parents
                if int(record["row"]["sample_index"]) == sample
                and np.isclose(float(record["row"]["ratio"]), float(ratio), atol=1.0e-5)
            ]
            matching.sort(key=lambda record: float(record["row"]["E2_worst_margin_db"]), reverse=True)
            selected_parents: list[dict[str, Any]] = []
            seen_masks: set[str] = set()
            for record in matching:
                digest = str(record["row"]["mask_hash"])
                if digest in seen_masks:
                    continue
                seen_masks.add(digest)
                selected_parents.append(record)
                if len(selected_parents) == parent_count:
                    break
            if len(selected_parents) != parent_count:
                raise RuntimeError(f"Not enough unique parents for scene={sample}, ratio={ratio}")
            for parent_rank, parent in enumerate(selected_parents):
                initial = np.asarray(parent["command"][:, :KMAX], dtype=np.complex64)
                mask = np.asarray(parent["mask"], dtype=bool)
                for variant in variants:
                    optimized, diagnostics = optimize_common_command(
                        initial,
                        mask,
                        targets,
                        e2_corners,
                        states,
                        grid_dirs,
                        variant,
                        int(protocol["multifrequency_combined_mainlobe"]["active_rl_nullspace_repair_steps"]),
                        float(protocol["multifrequency_combined_mainlobe"]["active_rl_nullspace_step_size"]),
                        bool(
                            protocol["multifrequency_combined_mainlobe"].get(
                                "quantization_aware_active_rl_selection", False
                            )
                        ),
                    )
                    padded_targets = np.asarray(parent["targets"], dtype=np.float32)
                    arrays["sample_index"].append(sample)
                    arrays["ratio"].append(float(ratio))
                    arrays["targets"].append(padded_targets)
                    arrays["masks"].append(mask.astype(np.int8))
                    arrays["tasks"].append(optimized)
                    arrays["parent_source"].append(str(parent["source"]))
                    arrays["parent_evaluation"].append(str(parent["row"]["evaluation_index"]))
                    arrays["mask_hash"].append(str(parent["row"]["mask_hash"]))
                    arrays["variant"].append(variant.name)
                    arrays["rl_design"].append(variant.active_rl_design_min_db)
                    arrays["radius"].append(variant.local_radius_deg)
                    arrays["ceiling"].append(variant.regional_ceiling_db)
                    arrays["equality_error"].append(diagnostics["joint_equality_error_max"])
                    arrays["regional_ratio"].append(diagnostics["joint_regional_ratio_max"])
                    arrays["equality_condition"].append(diagnostics["joint_equality_condition_number"])
                    arrays["analog_rl"].append(diagnostics["analog_E2_active_rl_floor_db"])
                    arrays["implementation_rl"].append(
                        diagnostics["frozen_implementation_E2_active_rl_floor_db"]
                    )
                    rows.append(
                        {
                            "candidate_index": len(rows),
                            "sample_index": sample,
                            "k_value": 6,
                            "ratio": ratio,
                            "parent_rank": parent_rank,
                            "parent_source": parent["source"],
                            "parent_evaluation_index": parent["row"]["evaluation_index"],
                            "parent_E2_worst_margin_db": parent["row"]["E2_worst_margin_db"],
                            "mask_hash": parent["row"]["mask_hash"],
                            "variant": variant.name,
                            "active_rl_design_min_db": variant.active_rl_design_min_db,
                            "target_amplitude_mode": variant.target_amplitude_mode,
                            "local_radius_deg": variant.local_radius_deg,
                            "regional_ceiling_db": variant.regional_ceiling_db,
                            **diagnostics,
                        }
                    )
        print(
            f"K6 multifrequency scene {scene_position + 1:02d}/{len(target_scenes):02d} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    np.savez_compressed(
        out / "candidate_commands.npz",
        candidate_index=np.arange(len(rows), dtype=np.int64),
        sample_index=np.asarray(arrays["sample_index"], dtype=np.int64),
        k_values=np.full(len(rows), 6, dtype=np.int8),
        ratio=np.asarray(arrays["ratio"], dtype=np.float32),
        targets=np.stack(arrays["targets"]),
        masks=np.stack(arrays["masks"]),
        tasks_real_imag=complex_to_ri(np.stack(arrays["tasks"])),
        parent_source=np.asarray(arrays["parent_source"]),
        parent_evaluation=np.asarray(arrays["parent_evaluation"]),
        mask_hash=np.asarray(arrays["mask_hash"]),
        variant=np.asarray(arrays["variant"]),
        active_rl_design_min_db=np.asarray(arrays["rl_design"], dtype=np.float32),
        local_radius_deg=np.asarray(arrays["radius"], dtype=np.float32),
        regional_ceiling_db=np.asarray(arrays["ceiling"], dtype=np.float32),
        joint_equality_error=np.asarray(arrays["equality_error"], dtype=np.float32),
        joint_regional_ratio=np.asarray(arrays["regional_ratio"], dtype=np.float32),
        joint_equality_condition=np.asarray(arrays["equality_condition"], dtype=np.float64),
        analog_E2_active_rl_floor_db=np.asarray(arrays["analog_rl"], dtype=np.float32),
        frozen_implementation_E2_active_rl_floor_db=np.asarray(
            arrays["implementation_rl"], dtype=np.float32
        ),
    )
    write_csv(out / "candidate_manifest.csv", rows)
    (out / "preregistered_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    summary = {
        "candidate_count": len(rows),
        "target_scene_count": len(target_scenes),
        "target_sample_indices": target_scenes,
        "scene_offset": int(args.scene_offset),
        "k_value": 6,
        "parents_per_ratio": parent_count,
        "variants_per_parent": len(variants),
        "common_command_across_E2_corners": True,
        "profile_specific_weights_saved": False,
        "E2_or_gates_changed": False,
        "elapsed_seconds": time.time() - started,
        "evidence_scope": protocol["evidence_scope"],
    }
    (out / "generate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    out = args.out_dir / "evaluation"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation: {out}")
    out.mkdir(parents=True, exist_ok=True)
    protocol, _variants = load_protocol(args.protocol)
    gates = protocol["strict_corner_gate_unchanged"]
    candidates = load_npz(args.out_dir / "candidates" / "candidate_commands.npz")
    pool = load_npz(args.parent_dir / "pool" / "candidate_pool.npz")
    base, _nominal_effective, nominal_fast, _nominal_s = load_nominal_operator(args.operator)
    _base, corners = build_corners(args)
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    commands_full = ri_to_complex(candidates["tasks_real_imag"])
    samples = np.asarray(candidates["sample_index"], dtype=np.int64)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    count = len(samples)
    robust = {
        name: np.full((count, len(ROBUST_MARGIN_NAMES)), np.inf, dtype=np.float32)
        for name in ("E1", "E2", "E3")
    }
    worst_corner = {
        name: np.full((count, len(ROBUST_MARGIN_NAMES)), "", dtype="<U40")
        for name in ("E1", "E2", "E3")
    }
    main_components = {
        name: np.full((count, 3), np.inf, dtype=np.float32) for name in ("E1", "E2", "E3")
    }
    reference_metrics = np.zeros((count, len(METRIC_NAMES)), dtype=np.float32)
    started = time.time()
    for scene_position, sample in enumerate(sorted(set(samples.tolist()))):
        members = np.flatnonzero(samples == sample)
        pool_scene = np.flatnonzero(pool_samples == sample)
        targets = np.asarray(candidates["targets"][members[0], :KMAX], dtype=float)
        commands = np.asarray(commands_full[members, :, :KMAX], dtype=np.complex64)
        nominal_batch = nominal_fast.evaluate(commands, targets)
        for local, candidate in enumerate(members):
            reference_metrics[candidate] = metric_vector(metric_at(nominal_batch, local))
        states = scene_calibration_states(
            pool,
            pool_scene,
            corners,
            base["element_ixiy"],
            int(protocol["seed"]),
        )
        for corner_name, corner in corners.items():
            state = states[corner_name]
            actual = np.stack(
                [
                    apply_calibration(
                        commands[local],
                        np.asarray(candidates["masks"][candidate], dtype=bool),
                        state,
                    )
                    for local, candidate in enumerate(members)
                ]
            )
            batch = corner["fast"].evaluate(actual, targets)
            envelope = str(corner["envelope"])
            for local, candidate in enumerate(members):
                metrics = metric_at(batch, local)
                reference = {
                    str(name): float(reference_metrics[candidate, index])
                    for index, name in enumerate(METRIC_NAMES)
                }
                mask = np.asarray(candidates["masks"][candidate], dtype=bool)
                active = full_active_metrics(actual[local], mask, corner["s"])
                physical = physical_margins(metrics, reference, active)
                realizability, _hardware = hardware_margin(
                    commands[local],
                    actual[local],
                    mask,
                    targets,
                    corner["effective"],
                    grid_dirs,
                    gates,
                )
                margins = np.concatenate((physical, np.asarray([realizability], dtype=np.float32)))
                improve = margins < robust[envelope][candidate]
                robust[envelope][candidate, improve] = margins[improve]
                worst_corner[envelope][candidate, improve] = corner_name
                components = np.asarray(
                    [
                        float(metrics["weakest_target_gain_db"])
                        - (float(reference["weakest_target_gain_db"]) - 0.5),
                        3.0 - float(metrics["target_spread_db"]),
                        1.5 - float(metrics["pointing_error_deg"]),
                    ],
                    dtype=np.float32,
                )
                main_components[envelope][candidate] = np.minimum(
                    main_components[envelope][candidate], components
                )
        print(
            f"evaluate K6 scene {scene_position + 1:02d}/{len(set(samples.tolist())):02d} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    evaluation_prefix = str(protocol.get("evaluation_prefix", "mf"))
    for candidate in range(count):
        row: dict[str, Any] = {
            "evaluation_index": f"{evaluation_prefix}_{candidate}",
            "candidate_index": candidate,
            "sample_index": int(samples[candidate]),
            "scene_origin": str(
                pool["scene_origin"][np.flatnonzero(pool_samples == samples[candidate])[0]]
            ),
            "k_value": 6,
            "ratio": float(candidates["ratio"][candidate]),
            "mask_hash": str(candidates["mask_hash"][candidate]),
            "candidate_origin": "K6_multifrequency_combined_mainlobe_reserve",
            "variant": str(candidates["variant"][candidate]),
            "active_rl_design_min_db": float(candidates["active_rl_design_min_db"][candidate]),
            "joint_equality_error": float(candidates["joint_equality_error"][candidate]),
            "joint_regional_ratio": float(candidates["joint_regional_ratio"][candidate]),
            "analog_E2_active_rl_floor_db": float(candidates["analog_E2_active_rl_floor_db"][candidate]),
            "frozen_implementation_E2_active_rl_floor_db": float(
                candidates["frozen_implementation_E2_active_rl_floor_db"][candidate]
            ),
        }
        for envelope in ("E1", "E2", "E3"):
            row[f"{envelope}_strict_pass"] = int(np.all(robust[envelope][candidate] >= 0.0))
            row[f"{envelope}_worst_margin_db"] = float(np.min(robust[envelope][candidate]))
            for index, margin_name in enumerate(ROBUST_MARGIN_NAMES):
                row[f"{envelope}_{margin_name}_margin_db"] = float(robust[envelope][candidate, index])
                row[f"{envelope}_{margin_name}_worst_corner"] = str(worst_corner[envelope][candidate, index])
            row[f"{envelope}_gain_margin_db"] = float(main_components[envelope][candidate, 0])
            row[f"{envelope}_spread_margin_db"] = float(main_components[envelope][candidate, 1])
            row[f"{envelope}_pointing_margin_deg"] = float(main_components[envelope][candidate, 2])
        rows.append(row)
    write_csv(out / "candidate_metrics.csv", rows)
    np.savez_compressed(
        out / "evaluation_arrays.npz",
        candidate_index=np.arange(count, dtype=np.int64),
        reference_metrics=reference_metrics,
        E1_robust_margins=robust["E1"],
        E2_robust_margins=robust["E2"],
        E3_robust_margins=robust["E3"],
        E1_main_components=main_components["E1"],
        E2_main_components=main_components["E2"],
        E3_main_components=main_components["E3"],
        robust_margin_names=ROBUST_MARGIN_NAMES,
    )

    parent_final = read_csv(args.parent_dir / "final" / "robust_candidate_metrics.csv")
    parent_rescue = read_csv(args.parent_dir / "post_rescue" / "rescue_robust_candidate_metrics.csv")
    additional_parent_rows = [
        row
        for additional_dir in args.additional_parent_dir
        for row in read_csv(additional_dir / "evaluation" / "candidate_metrics.csv")
    ]
    scene_origin = {
        int(sample): str(pool["scene_origin"][np.flatnonzero(pool_samples == sample)[0]])
        for sample in set(pool_samples.tolist())
    }
    for row in [*parent_final, *parent_rescue, *additional_parent_rows]:
        row["scene_origin"] = scene_origin[int(row["sample_index"])]
    all_rows = [*parent_final, *parent_rescue, *additional_parent_rows, *rows]
    scene_rows: list[dict[str, Any]] = []
    for sample in sorted(set(pool_samples.tolist())):
        members = [row for row in all_rows if int(row["sample_index"]) == sample]
        k_value = int(members[0]["k_value"])
        for envelope in ("E1", "E2", "E3"):
            passed = [row for row in members if int(row[f"{envelope}_strict_pass"]) == 1]
            best = max(members, key=lambda row: float(row[f"{envelope}_worst_margin_db"]))
            scene_rows.append(
                {
                    "sample_index": sample,
                    "scene_origin": scene_origin[sample],
                    "k_value": k_value,
                    "envelope": envelope,
                    "robust_oracle_pass": int(bool(passed)),
                    "minimum_feasible_ratio": (
                        min(float(row["ratio"]) for row in passed) if passed else float("nan")
                    ),
                    "best_candidate_source": best["candidate_origin"],
                    "best_evaluation_index": best["evaluation_index"],
                    "best_worst_margin_db": float(best[f"{envelope}_worst_margin_db"]),
                }
            )
    write_csv(out / "combined_scene_oracle.csv", scene_rows)

    group_rows: list[dict[str, Any]] = []
    for envelope in ("E1", "E2", "E3"):
        for origin in ("all", "existing45", "new30"):
            for k_value in (0, 2, 4, 6):
                members = [
                    row
                    for row in scene_rows
                    if row["envelope"] == envelope
                    and (origin == "all" or row["scene_origin"] == origin)
                    and (k_value == 0 or int(row["k_value"]) == k_value)
                ]
                if members:
                    group_rows.append(
                        {
                            "envelope": envelope,
                            "scene_origin": origin,
                            "k_value": "all" if k_value == 0 else k_value,
                            "scene_count": len(members),
                            "oracle_pass_count": sum(int(row["robust_oracle_pass"]) for row in members),
                            "robust_oracle_rate": float(np.mean([row["robust_oracle_pass"] for row in members])),
                        }
                    )
    write_csv(out / "combined_oracle_groups.csv", group_rows)

    def rate(envelope: str, origin: str = "all", k_value: int = 0) -> float:
        desired = "all" if k_value == 0 else k_value
        return float(
            next(
                row["robust_oracle_rate"]
                for row in group_rows
                if row["envelope"] == envelope
                and row["scene_origin"] == origin
                and row["k_value"] == desired
            )
        )

    k6_low = any(
        row["envelope"] == "E2"
        and int(row["k_value"]) == 6
        and int(row["robust_oracle_pass"]) == 1
        and float(row["minimum_feasible_ratio"]) <= 0.70 + 1.0e-6
        for row in scene_rows
    )
    acceptance = protocol["acceptance_unchanged"]
    values = {
        "E1_new_scene_robust_oracle": rate("E1", "new30"),
        "E2_overall_robust_oracle": rate("E2"),
        "E2_k2_robust_oracle": rate("E2", "all", 2),
        "E2_k4_robust_oracle": rate("E2", "all", 4),
        "E2_k6_robust_oracle": rate("E2", "all", 6),
        "E2_has_k6_positive_ratio_le_0_7": k6_low,
    }
    stage_pass = bool(
        values["E1_new_scene_robust_oracle"] >= acceptance["E1_new_scene_robust_oracle_min"]
        and values["E2_overall_robust_oracle"] >= acceptance["E2_overall_robust_oracle_min"]
        and values["E2_k2_robust_oracle"] >= acceptance["E2_k2_robust_oracle_min"]
        and values["E2_k4_robust_oracle"] >= acceptance["E2_k4_robust_oracle_min"]
        and values["E2_k6_robust_oracle"] >= acceptance["E2_k6_robust_oracle_min"]
        and k6_low
    )
    failed_causes = Counter()
    for scene in scene_rows:
        if scene["envelope"] != "E2" or int(scene["robust_oracle_pass"]) == 1:
            continue
        best = next(
            row
            for row in all_rows
            if int(row["sample_index"]) == int(scene["sample_index"])
            and str(row["evaluation_index"]) == str(scene["best_evaluation_index"])
        )
        margin_values = [float(best[f"E2_{name}_margin_db"]) for name in ROBUST_MARGIN_NAMES]
        failed_causes[str(ROBUST_MARGIN_NAMES[int(np.argmin(margin_values))])] += 1
    targeted_pass = len(
        {
            int(row["sample_index"])
            for row in rows
            if int(row["E2_strict_pass"]) == 1
        }
    )
    summary = {
        "candidate_count": count,
        "target_K6_failed_scene_count": len(set(samples.tolist())),
        "target_K6_scenes_rescued_by_new_candidates": targeted_pass,
        **values,
        "E3_stress_oracle": rate("E3"),
        "stage_b_gate_pass": stage_pass,
        "remaining_failed_scene_root_causes": dict(failed_causes),
        "critic_retraining_allowed": stage_pass,
        "hfss_smoke_allowed": stage_pass,
        "automatic_large_HFSS_allowed": False,
        "E2_or_gates_changed": False,
        "elapsed_seconds": time.time() - started,
        "evidence_scope": protocol["evidence_scope"],
    }
    (out / "stage_b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("generate", "all"):
        generate(args)
    if args.mode in ("evaluate", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
