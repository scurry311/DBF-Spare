#!/usr/bin/env python3
"""Run the preregistered frozen-S4 task-weight active-RL feasibility oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "cvxpy is required; install cvxpy==1.6.5 into .python_deps"
    ) from exc

from generate_gate15_boundary_scenes import metric_at
from generate_v09_eep_development_candidates import (
    METRIC_NAMES,
    full_active_metrics,
    physical_margins,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from refine_trusted_dense_local_eep_joint import (
    DenseTaskConstraint,
    build_constraints,
    dense_constraint_metrics,
)
from run_v16_robust_drift_oracle import (
    hardware_margin,
    ri_to_complex,
)
from run_v20_three_frequency_mask_weight_joint import operator_bundle


DEFAULT_CONFIG = ROOT / "configs" / "v140_frozen_s4_task_weight_projection_preregistered.json"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v140_frozen_s4_task_weight_projection_20260808_run01"
EPS = 1.0e-12
KMAX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_input(config_path: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else config_path.parents[1] / path


def tiled_s256(s4: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    if s4.shape != (4, 4):
        raise ValueError(f"Expected S4, got {s4.shape}")
    result = np.zeros((256, 256), dtype=np.complex128)
    blocks: list[list[int]] = []
    for ix in range(0, 16, 2):
        for iy in range(0, 16, 2):
            indices = [
                ix * 16 + iy,
                (ix + 1) * 16 + iy,
                ix * 16 + (iy + 1),
                (ix + 1) * 16 + (iy + 1),
            ]
            result[np.ix_(indices, indices)] = s4
            blocks.append(indices)
    if sorted(index for block in blocks for index in block) != list(range(256)):
        raise RuntimeError("The 2x2 S4 tiling does not cover every 16x16 port exactly once")
    return result, blocks


def reference_dict(data: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {
        str(name): float(data["reference_metrics"][index, metric_index])
        for metric_index, name in enumerate(data["reference_metric_names"])
    }


def evaluate(
    tasks: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    reference: dict[str, float],
    s256: np.ndarray,
    corner: dict[str, Any],
    grid_dirs: np.ndarray,
    gates: dict[str, Any],
    constraints: list[Any],
    combined: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    metrics = metric_at(corner["fast"].evaluate(tasks, targets), 0)
    active = full_active_metrics(tasks, mask, s256)
    margins = physical_margins(metrics, reference, active)
    hardware_value, hardware = hardware_margin(
        original,
        tasks,
        mask,
        targets,
        corner["effective"],
        grid_dirs,
        {
            "combined_channel_dynamic_range_max_db": gates["combined_channel_dynamic_range_max_db"],
            "combined_peak_to_rms_max_db": gates["combined_peak_to_rms_max_db"],
            "corner_power_increase_max_db": gates["power_increase_max_db"],
            "normalized_wng_min_db": gates["normalized_wng_min_db"],
        },
    )
    dense = dense_constraint_metrics(
        tasks,
        constraints,
        combined,
        combined.preserve_desired,
    )
    pattern_margin = float(np.min(margins[:4]))
    active_floor = float(active["active_rl_floor_db"])
    engineering_pass = int(
        pattern_margin >= 0.0
        and active_floor >= float(gates["engineering_active_rl_min_db"])
        and hardware_value >= 0.0
        and int(dense["dense_constraint_pass"]) == 1
    )
    reserve_pass = int(
        engineering_pass == 1
        and active_floor >= float(gates["design_active_rl_min_db"])
    )
    row: dict[str, Any] = {
        **{name: float(metrics[name]) for name in METRIC_NAMES},
        **{name: value for name, value in active.items()},
        **hardware,
        **dense,
        "pattern_margin_db": pattern_margin,
        "active_rl_engineering_margin_db": active_floor
        - float(gates["engineering_active_rl_min_db"]),
        "active_rl_design_margin_db": active_floor
        - float(gates["design_active_rl_min_db"]),
        "hardware_margin_db": float(hardware_value),
        "engineering_strict_pass": engineering_pass,
        "reserve11_strict_pass": reserve_pass,
        "relative_task_weight_change": float(
            np.linalg.norm(tasks - original) / max(float(np.linalg.norm(original)), EPS)
        ),
    }
    return row, margins


def make_socp(
    original_active: np.ndarray,
    s_active: np.ndarray,
    constraints: list[Any],
    combined: Any,
    gates: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[cp.Problem, cp.Variable, dict[str, Any]]:
    active_count, k_value = original_active.shape
    weights = cp.Variable((active_count, k_value), complex=True, name="task_weights")
    combined_weights = cp.sum(weights, axis=1)
    rho = cp.Parameter(nonneg=True, name="rho")
    anchor = cp.Parameter((active_count, k_value), complex=True, name="anchor")
    combined_phase = cp.Parameter(active_count, complex=True, name="combined_phase")
    combined_anchor = cp.Parameter(active_count, complex=True, name="combined_anchor")
    task_phase = [cp.Parameter(active_count, complex=True, name=f"task_phase_{j}") for j in range(k_value)]
    task_anchor = [cp.Parameter(active_count, complex=True, name=f"task_anchor_{j}") for j in range(k_value)]
    significant = []
    for task_index in range(k_value):
        amplitudes = np.abs(original_active[:, task_index])
        significant.append(
            amplitudes
            >= max(float(np.max(amplitudes)), EPS)
            * 10.0 ** (float(settings["task_significant_relative_db"]) / 20.0)
        )

    conic: list[Any] = []
    norm_growth = 10.0 ** (float(settings["task_norm_growth_max_db"]) / 20.0)
    for task_index, task_constraint in enumerate(constraints):
        value = weights[:, task_index]
        conic.append(task_constraint.equality_row @ value == task_constraint.desired)
        if task_constraint.leakage_rows.shape[0]:
            conic.append(
                cp.abs(task_constraint.leakage_rows @ value)
                <= task_constraint.leakage_bounds
            )
        conic.append(
            cp.norm(value, 2)
            <= float(np.linalg.norm(original_active[:, task_index])) * norm_growth
        )
    conic.extend(
        [
            combined.rows @ combined_weights == combined.preserve_desired,
            cp.norm(combined_weights, 2)
            <= float(np.linalg.norm(np.sum(original_active, axis=1)))
            * 10.0 ** (float(settings["combined_norm_growth_max_db"]) / 20.0),
        ]
    )

    combined_alignment = cp.real(
        cp.multiply(cp.conj(combined_phase), combined_weights)
    )
    conic.extend(
        [
            combined_alignment >= 0.0,
            cp.abs(s_active @ combined_weights) <= rho * combined_alignment,
        ]
    )
    combined_total_lower = cp.real(
        cp.sum(cp.multiply(cp.conj(combined_anchor), combined_weights))
    )
    conic.extend(
        [
            combined_total_lower >= 0.0,
            cp.norm(s_active @ combined_weights, 2)
            <= rho * combined_total_lower,
        ]
    )
    for task_index in range(k_value):
        value = weights[:, task_index]
        selected = significant[task_index]
        task_alignment = cp.real(
            cp.multiply(cp.conj(task_phase[task_index][selected]), value[selected])
        )
        conic.extend(
            [
                task_alignment >= 0.0,
                cp.abs(s_active[selected] @ value) <= rho * task_alignment,
            ]
        )
        task_total_lower = cp.real(
            cp.sum(cp.multiply(cp.conj(task_anchor[task_index]), value))
        )
        conic.extend(
            [
                task_total_lower >= 0.0,
                cp.norm(s_active @ value, 2) <= rho * task_total_lower,
            ]
        )

    objective = cp.Minimize(
        cp.sum_squares(cp.abs(weights - anchor))
        + float(settings["proximal_to_initial_weight"])
        * cp.sum_squares(cp.abs(weights - original_active))
    )
    problem = cp.Problem(objective, conic)
    parameters = {
        "rho": rho,
        "anchor": anchor,
        "combined_phase": combined_phase,
        "combined_anchor": combined_anchor,
        "task_phase": task_phase,
        "task_anchor": task_anchor,
        "significant": significant,
    }
    return problem, weights, parameters


def solver_active_set(
    constraints: list[DenseTaskConstraint],
    original_active: np.ndarray,
    limit: int,
) -> list[DenseTaskConstraint]:
    """Keep the initially tightest rows in the solver; final gating remains fully dense."""
    if limit <= 0:
        return constraints
    selected_constraints: list[DenseTaskConstraint] = []
    for task_index, constraint in enumerate(constraints):
        count = constraint.leakage_rows.shape[0]
        if count <= limit:
            selected_constraints.append(constraint)
            continue
        ratios = np.abs(constraint.leakage_rows @ original_active[:, task_index]) / np.maximum(
            constraint.leakage_bounds, EPS
        )
        nearest = np.flatnonzero(constraint.leakage_kind == 0)
        remaining = max(0, limit - nearest.size)
        local = np.flatnonzero(constraint.leakage_kind == 1)
        local_order = local[np.argsort(ratios[local], kind="stable")[::-1][:remaining]]
        selected = np.unique(np.concatenate((nearest, local_order)))
        if selected.size > limit:
            selected = selected[np.argsort(ratios[selected], kind="stable")[::-1][:limit]]
        selected_constraints.append(
            DenseTaskConstraint(
                active=constraint.active,
                equality_row=constraint.equality_row,
                desired=constraint.desired,
                leakage_rows=constraint.leakage_rows[selected],
                leakage_bounds=constraint.leakage_bounds[selected],
                leakage_kind=constraint.leakage_kind[selected],
                leakage_row_norm_sq=constraint.leakage_row_norm_sq[selected],
            )
        )
    return selected_constraints


def update_parameters(parameters: dict[str, Any], current: np.ndarray, target_db: float) -> None:
    parameters["rho"].value = 10.0 ** (-float(target_db) / 20.0)
    parameters["anchor"].value = current
    combined = np.sum(current, axis=1)
    parameters["combined_phase"].value = np.where(
        np.abs(combined) > 1.0e-10, combined / np.maximum(np.abs(combined), EPS), 1.0 + 0.0j
    )
    parameters["combined_anchor"].value = combined / max(float(np.linalg.norm(combined)), EPS)
    for task_index in range(current.shape[1]):
        value = current[:, task_index]
        parameters["task_phase"][task_index].value = np.where(
            np.abs(value) > 1.0e-10, value / np.maximum(np.abs(value), EPS), 1.0 + 0.0j
        )
        parameters["task_anchor"][task_index].value = value / max(
            float(np.linalg.norm(value)), EPS
        )


def solve_scene(
    scene_index: int,
    data: dict[str, np.ndarray],
    all_tasks: np.ndarray,
    s256: np.ndarray,
    corner: dict[str, Any],
    grid_dirs: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    sample = int(data["sample_index"][scene_index])
    k_value = int(data["k_values"][scene_index])
    ratio = float(data["ratio"][scene_index])
    mask = np.asarray(data["masks"][scene_index], dtype=bool)
    targets = np.asarray(data["targets"][scene_index, :k_value], dtype=np.float64)
    original = np.asarray(all_tasks[scene_index, :, :k_value], dtype=np.complex128)
    active = np.flatnonzero(mask)
    original_active = original[active]
    reference = reference_dict(data, scene_index)
    constraints, combined, dense_stats = build_constraints(
        original,
        mask,
        targets,
        grid_dirs,
        corner["effective"],
        local_radius_deg=float(config["sequential_socp"]["local_region_radius_deg"]),
        nearest_isolation_db=float(config["gates"]["nearest_isolation_min_db"]),
        local_isolation_db=float(config["gates"]["local_5deg_isolation_min_db"]),
    )
    solver_constraints = solver_active_set(
        constraints,
        original_active,
        int(config["sequential_socp"]["max_dense_rows_per_task_in_solver"]),
    )
    dense_stats = {
        **dense_stats,
        "full_dense_row_count": int(
            sum(value.leakage_rows.shape[0] for value in constraints)
        ),
        "solver_dense_row_count": int(
            sum(value.leakage_rows.shape[0] for value in solver_constraints)
        ),
    }
    rows: list[dict[str, Any]] = []
    candidates: list[np.ndarray] = []

    def append_candidate(
        tasks: np.ndarray,
        stage: str,
        target_db: float,
        outer: int,
        status: str,
        solve_seconds: float,
    ) -> None:
        metrics, _margins = evaluate(
            tasks,
            original,
            mask,
            targets,
            reference,
            s256,
            corner,
            grid_dirs,
            config["gates"],
            constraints,
            combined,
        )
        rows.append(
            {
                "scene_index": scene_index,
                "sample_index": sample,
                "k_value": k_value,
                "ratio": ratio,
                "active_count": int(np.sum(mask)),
                "candidate_index_in_scene": len(candidates),
                "stage": stage,
                "socp_target_active_rl_db": target_db,
                "outer_iteration": outer,
                "solver_status": status,
                "solve_seconds": solve_seconds,
                **dense_stats,
                **metrics,
            }
        )
        padded = np.zeros((256, KMAX), dtype=np.complex64)
        padded[:, :k_value] = tasks
        candidates.append(padded)

    append_candidate(original, "baseline", float("nan"), 0, "not_run", 0.0)
    problem, variable, parameters = make_socp(
        original_active,
        s256[np.ix_(active, active)],
        solver_constraints,
        combined,
        config["gates"],
        config["sequential_socp"],
    )
    current = original_active.copy()
    stop = False
    for target_db in config["sequential_socp"]["active_rl_stages_db"]:
        for outer in range(int(config["sequential_socp"]["outer_iterations_per_stage"])):
            update_parameters(parameters, current, float(target_db))
            started = time.time()
            try:
                problem.solve(
                    solver=str(config["sequential_socp"]["solver"]),
                    # CLARABEL's matrix update path is unreliable for this non-DPP
                    # complex parameterization; rebuild while retaining the SCP anchor.
                    warm_start=False,
                    max_iter=int(config["sequential_socp"]["solver_max_iterations"]),
                    time_limit=float(config["sequential_socp"]["solver_time_limit_seconds"]),
                    tol_gap_abs=float(config["sequential_socp"]["solver_tolerance"]),
                    tol_feas=float(config["sequential_socp"]["solver_tolerance"]),
                    verbose=False,
                )
                status = str(problem.status)
            except cp.error.SolverError:
                status = "solver_error"
            elapsed = time.time() - started
            if variable.value is None or status not in {"optimal", "optimal_inaccurate"}:
                rows.append(
                    {
                        "scene_index": scene_index,
                        "sample_index": sample,
                        "k_value": k_value,
                        "ratio": ratio,
                        "active_count": int(np.sum(mask)),
                        "candidate_index_in_scene": -1,
                        "stage": "failed_stage",
                        "socp_target_active_rl_db": float(target_db),
                        "outer_iteration": outer + 1,
                        "solver_status": status,
                        "solve_seconds": elapsed,
                    }
                )
                stop = True
                break
            current = np.asarray(variable.value, dtype=np.complex128)
            full = np.zeros_like(original)
            full[active] = current
            append_candidate(
                full,
                "sequential_socp",
                float(target_db),
                outer + 1,
                status,
                elapsed,
            )
        if stop:
            break
    return rows, np.stack(candidates)


def candidate_key(row: dict[str, Any]) -> tuple[int, int, int, float, float, float]:
    return (
        int(row["reserve11_strict_pass"]),
        int(row["engineering_strict_pass"]),
        int(float(row["pattern_margin_db"]) >= 0.0),
        min(float(row["active_rl_design_margin_db"]), 3.0),
        min(float(row["pattern_margin_db"]), 3.0),
        -float(row["relative_task_weight_change"]),
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite v1.40 output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    scene_dir = out / "scenes"
    scene_dir.mkdir(exist_ok=True)
    inputs = {
        name: resolve_input(config_path, relative).resolve()
        for name, relative in config["inputs"].items()
        if name in {"physical_s4", "frozen_scenes", "pattern_operator"}
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    preregistration = {
        **config,
        "config_sha256": sha256(config_path),
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "cvxpy_runtime_version": cp.__version__,
    }
    prereg_path = out / "preregistration.json"
    if prereg_path.exists():
        previous = json.loads(prereg_path.read_text(encoding="utf-8"))
        if previous["config_sha256"] != preregistration["config_sha256"]:
            raise RuntimeError("Resume refused because the preregistered config changed")
    else:
        write_json(prereg_path, preregistration)

    with np.load(inputs["physical_s4"], allow_pickle=False) as source:
        s4 = ri_to_complex(source["s_real_imag"])
        port_order = [str(value) for value in source["port_order"]]
    if port_order != config["inputs"]["physical_s4_port_order"]:
        raise RuntimeError(f"Physical S4 port order changed: {port_order}")
    s256, blocks = tiled_s256(s4)
    with np.load(inputs["frozen_scenes"], allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    all_tasks = ri_to_complex(data["tasks_real_imag"])
    base, corner = operator_bundle(inputs["pattern_operator"], 10.0)
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    scene_count = len(data["sample_index"])
    if args.max_scenes > 0:
        scene_count = min(scene_count, int(args.max_scenes))
    started = time.time()
    for scene_index in range(scene_count):
        scene_npz = scene_dir / f"scene_{scene_index:02d}.npz"
        scene_csv = scene_dir / f"scene_{scene_index:02d}.csv"
        if args.resume and scene_npz.exists() and scene_csv.exists():
            continue
        rows, candidates = solve_scene(
            scene_index,
            data,
            all_tasks,
            s256,
            corner,
            grid_dirs,
            config,
        )
        write_csv(scene_csv, rows)
        np.savez_compressed(
            scene_npz,
            tasks_real_imag=np.stack((candidates.real, candidates.imag), axis=-1).astype(np.float32),
        )
        print(
            json.dumps(
                {
                    "scene": scene_index + 1,
                    "of": scene_count,
                    "sample_index": int(data["sample_index"][scene_index]),
                    "best_active_rl_db": max(
                        float(row["active_rl_floor_db"])
                        for row in rows
                        if "active_rl_floor_db" in row
                    ),
                    "reserve11_pass": max(
                        int(row["reserve11_strict_pass"])
                        for row in rows
                        if "reserve11_strict_pass" in row
                    ),
                }
            ),
            flush=True,
        )

    all_rows: list[dict[str, Any]] = []
    for path in sorted(scene_dir.glob("scene_*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            all_rows.extend(csv.DictReader(handle))
    candidate_rows = [row for row in all_rows if row.get("candidate_index_in_scene") != "-1"]
    write_csv(out / "candidate_metrics.csv", candidate_rows)
    scene_rows: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[int(row["sample_index"])].append(row)
    for sample, rows in sorted(grouped.items()):
        best = max(rows, key=candidate_key)
        scene_rows.append(
            {
                "sample_index": sample,
                "k_value": int(best["k_value"]),
                "ratio": float(best["ratio"]),
                "best_stage": best["stage"],
                "best_socp_target_active_rl_db": best["socp_target_active_rl_db"],
                "best_active_rl_floor_db": float(best["active_rl_floor_db"]),
                "best_pattern_margin_db": float(best["pattern_margin_db"]),
                "best_hardware_margin_db": float(best["hardware_margin_db"]),
                "engineering_strict_pass": int(best["engineering_strict_pass"]),
                "reserve11_strict_pass": int(best["reserve11_strict_pass"]),
                "best_candidate_index_in_scene": int(best["candidate_index_in_scene"]),
            }
        )
    write_csv(out / "scene_oracle.csv", scene_rows)
    per_k = []
    for k_value in sorted({int(row["k_value"]) for row in scene_rows}):
        rows = [row for row in scene_rows if int(row["k_value"]) == k_value]
        per_k.append(
            {
                "k_value": k_value,
                "scene_count": len(rows),
                "engineering_pass_count": sum(int(row["engineering_strict_pass"]) for row in rows),
                "reserve11_pass_count": sum(int(row["reserve11_strict_pass"]) for row in rows),
                "reserve11_pass_rate": float(np.mean([int(row["reserve11_strict_pass"]) for row in rows])),
                "best_active_rl_floor_db": max(float(row["best_active_rl_floor_db"]) for row in rows),
            }
        )
    write_csv(out / "per_k_oracle.csv", per_k)
    reserve_count = sum(int(row["reserve11_strict_pass"]) for row in scene_rows)
    summary = {
        "scope": config["evidence_scope"],
        "scene_count": len(scene_rows),
        "k_distribution": {
            str(k): sum(int(row["k_value"]) == k for row in scene_rows) for k in (2, 4, 6)
        },
        "s4_block_count": len(blocks),
        "engineering_strict_oracle_count": sum(
            int(row["engineering_strict_pass"]) for row in scene_rows
        ),
        "reserve11_strict_oracle_count": reserve_count,
        "reserve11_strict_oracle_nonempty": bool(reserve_count),
        "best_active_rl_floor_db": max(float(row["best_active_rl_floor_db"]) for row in scene_rows),
        "elapsed_seconds": time.time() - started,
        "allow_hfss": False,
        "allow_critic_training": False,
        "next_action": (
            config["decision"]["nonempty_11db_oracle_action"]
            if reserve_count
            else config["decision"]["empty_11db_oracle_action"]
        ),
        "limitations": [
            "The S256 matching operator is block diagonal and omits coupling between 2x2 cells.",
            "The pattern operator is the frozen nominal EEP used by the scene package, not an EEP exported from the v1.39 differential geometry.",
            "A solver_error is a failed search stage, not a mathematical proof of global infeasibility.",
        ],
    }
    write_json(out / "stage_summary.json", summary)
    write_json(
        out / "stage_decision.json",
        {
            "task_weight_projection_gate_pass": bool(reserve_count),
            "authorize_single_xy_modal_correction": not bool(reserve_count),
            "authorize_hfss": False,
            "authorize_larger_array": False,
            "authorize_labels_or_critic": False,
            "next_action": summary["next_action"],
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
