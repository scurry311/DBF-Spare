#!/usr/bin/env python3
"""Optimize frozen task weights on the v1.41 corrected S4 and mapped EEP."""

from __future__ import annotations

import argparse
import csv
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
    raise SystemExit("cvxpy is required in .python_deps") from exc

from generate_gate15_boundary_scenes import FastPatternEvaluator
from hfss_task_fullwave_validate import pattern_grid_dirs
from refine_trusted_dense_local_eep_joint import DenseExternalEEP, build_constraints
from run_v16_robust_drift_oracle import load_nominal_operator, ri_to_complex
from run_v140_frozen_s4_task_weight_projection import (
    KMAX,
    candidate_key,
    evaluate,
    reference_dict,
    sha256,
    solver_active_set,
    tiled_s256,
    update_parameters,
    write_csv,
    write_json,
)


DEFAULT_CONFIG = ROOT / "configs" / "v142_corrected_s4_mapped_eep_joint_preregistered.json"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v142_corrected_s4_mapped_eep_joint_20260809_run01"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parents[1] / path


def tile_map(map4: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    output = np.zeros((256, 256), dtype=np.complex128)
    for indices in blocks:
        output[np.ix_(indices, indices)] = map4
    return output


def load_reference_commands(projection_dir: Path) -> list[np.ndarray]:
    with (projection_dir / "scene_oracle.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    commands: list[np.ndarray] = []
    for scene_index, row in enumerate(rows):
        with np.load(projection_dir / "scenes" / f"scene_{scene_index:02d}.npz") as source:
            values = ri_to_complex(source["tasks_real_imag"])
        commands.append(
            np.asarray(
                values[int(row["best_candidate_index_in_scene"]), :, : int(row["k_value"])],
                dtype=np.complex128,
            )
        )
    return commands


def seed_commands(
    reference: np.ndarray,
    mask: np.ndarray,
    map4: np.ndarray,
    blocks: list[list[int]],
    rcond: float,
) -> dict[str, np.ndarray]:
    direct = np.asarray(reference, dtype=np.complex128).copy()
    direct[~mask] = 0.0
    block_ls = np.zeros_like(direct)
    for indices in blocks:
        block = np.asarray(indices, dtype=int)
        local_active = np.flatnonzero(mask[block])
        if not local_active.size:
            continue
        source = map4[:, local_active]
        solution = np.linalg.lstsq(source, reference[block], rcond=rcond)[0]
        block_ls[block[local_active]] = solution
    seeds = {"direct": direct, "block_ls": block_ls}
    for alpha in (0.25, 0.50, 0.75):
        value = (1.0 - alpha) * direct + alpha * block_ls
        value[~mask] = 0.0
        seeds[f"blend_ls_{alpha:.2f}"] = value
    return seeds


def make_soft_socp(
    original_active: np.ndarray,
    s_active: np.ndarray,
    constraints: list[Any],
    combined: Any,
    settings: dict[str, Any],
) -> tuple[cp.Problem, cp.Variable, dict[str, Any]]:
    """Use solver-only leakage slacks; final dense gating remains unrelaxed."""
    active_count, k_value = original_active.shape
    weights = cp.Variable((active_count, k_value), complex=True, name="task_weights")
    combined_weights = cp.sum(weights, axis=1)
    rho = cp.Parameter(nonneg=True, name="rho")
    anchor = cp.Parameter((active_count, k_value), complex=True, name="anchor")
    combined_phase = cp.Parameter(active_count, complex=True, name="combined_phase")
    combined_anchor = cp.Parameter(active_count, complex=True, name="combined_anchor")
    task_phase = [cp.Parameter(active_count, complex=True, name=f"task_phase_{j}") for j in range(k_value)]
    task_anchor = [cp.Parameter(active_count, complex=True, name=f"task_anchor_{j}") for j in range(k_value)]
    significant: list[np.ndarray] = []
    conic: list[Any] = []
    leakage_slacks: list[Any] = []
    norm_growth = 10.0 ** (float(settings["task_norm_growth_max_db"]) / 20.0)
    for task_index, task_constraint in enumerate(constraints):
        value = weights[:, task_index]
        amplitudes = np.abs(original_active[:, task_index])
        significant.append(
            amplitudes
            >= max(float(np.max(amplitudes)), EPS)
            * 10.0 ** (float(settings["task_significant_relative_db"]) / 20.0)
        )
        conic.append(task_constraint.equality_row @ value == task_constraint.desired)
        if task_constraint.leakage_rows.shape[0]:
            slack = cp.Variable(task_constraint.leakage_rows.shape[0], nonneg=True)
            conic.append(
                cp.abs(task_constraint.leakage_rows @ value)
                <= cp.multiply(task_constraint.leakage_bounds, 1.0 + slack)
            )
            leakage_slacks.append(slack)
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
    combined_alignment = cp.real(cp.multiply(cp.conj(combined_phase), combined_weights))
    combined_total_lower = cp.real(
        cp.sum(cp.multiply(cp.conj(combined_anchor), combined_weights))
    )
    conic.extend(
        [
            combined_alignment >= 0.0,
            cp.abs(s_active @ combined_weights) <= rho * combined_alignment,
            combined_total_lower >= 0.0,
            cp.norm(s_active @ combined_weights, 2) <= rho * combined_total_lower,
        ]
    )
    for task_index in range(k_value):
        value = weights[:, task_index]
        selected = significant[task_index]
        alignment = cp.real(cp.multiply(cp.conj(task_phase[task_index][selected]), value[selected]))
        total_lower = cp.real(cp.sum(cp.multiply(cp.conj(task_anchor[task_index]), value)))
        conic.extend(
            [
                alignment >= 0.0,
                cp.abs(s_active[selected] @ value) <= rho * alignment,
                total_lower >= 0.0,
                cp.norm(s_active @ value, 2) <= rho * total_lower,
            ]
        )
    leakage_penalty = 0.0
    if leakage_slacks:
        leakage_penalty = sum(cp.sum(slack) for slack in leakage_slacks) / float(
            sum(int(slack.shape[0]) for slack in leakage_slacks)
        )
    objective = cp.Minimize(
        cp.sum_squares(cp.abs(weights - anchor))
        + float(settings["proximal_to_initial_weight"])
        * cp.sum_squares(cp.abs(weights - original_active))
        + float(settings["solver_leakage_slack_penalty"]) * leakage_penalty
    )
    problem = cp.Problem(objective, conic)
    return problem, weights, {
        "rho": rho,
        "anchor": anchor,
        "combined_phase": combined_phase,
        "combined_anchor": combined_anchor,
        "task_phase": task_phase,
        "task_anchor": task_anchor,
        "significant": significant,
        "leakage_slacks": leakage_slacks,
    }


def build_seed_problem(
    seed: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    grid_dirs: np.ndarray,
    corner: dict[str, Any],
    s256: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    active = np.flatnonzero(mask)
    constraints, combined, stats = build_constraints(
        seed,
        mask,
        targets,
        grid_dirs,
        corner["effective"],
        local_radius_deg=float(config["sequential_socp"]["local_region_radius_deg"]),
        nearest_isolation_db=float(config["gates"]["nearest_isolation_min_db"]),
        local_isolation_db=float(config["gates"]["local_5deg_isolation_min_db"]),
    )
    selected = solver_active_set(
        constraints,
        seed[active],
        int(config["sequential_socp"]["max_dense_rows_per_task_in_solver"]),
    )
    problem, variable, parameters = make_soft_socp(
        seed[active],
        s256[np.ix_(active, active)],
        selected,
        combined,
        config["sequential_socp"],
    )
    return {
        "active": active,
        "constraints": constraints,
        "combined": combined,
        "problem": problem,
        "variable": variable,
        "parameters": parameters,
        "full_dense_row_count": int(sum(item.leakage_rows.shape[0] for item in constraints)),
        "solver_dense_row_count": int(sum(item.leakage_rows.shape[0] for item in selected)),
        **stats,
    }


def seed_selection_key(row: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(float(row["pattern_margin_db"]) >= 0.0),
        float(row["pattern_margin_db"]),
        -float(row["relative_task_weight_change"]),
    )


def balanced_selection_key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    pattern = float(row["pattern_margin_db"])
    active = float(row["active_rl_design_margin_db"])
    return (
        int(pattern >= 0.0 and active >= 0.0),
        min(pattern, active),
        pattern + active,
        -float(row["relative_task_weight_change"]),
    )


def solve_scene(
    scene_index: int,
    data: dict[str, np.ndarray],
    reference: np.ndarray,
    map4: np.ndarray,
    blocks: list[list[int]],
    s256: np.ndarray,
    corner: dict[str, Any],
    grid_dirs: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    sample = int(data["sample_index"][scene_index])
    k_value = int(data["k_values"][scene_index])
    ratio = float(data["ratio"][scene_index])
    mask = np.asarray(data["masks"][scene_index], dtype=bool)
    targets = np.asarray(data["targets"][scene_index, :k_value], dtype=float)
    reference = np.asarray(reference[:, :k_value], dtype=np.complex128)
    reference_metrics = reference_dict(data, scene_index)
    seed_values = seed_commands(
        reference,
        mask,
        map4,
        blocks,
        float(config["seed_policy"]["block_ls_rcond"]),
    )
    rows: list[dict[str, Any]] = []
    candidates: list[np.ndarray] = []
    candidate_seeds: list[str] = []
    seed_contexts: dict[str, dict[str, Any]] = {}

    def append_candidate(
        tasks: np.ndarray,
        seed_name: str,
        context: dict[str, Any],
        stage: str,
        target_db: float,
        status: str,
        solve_seconds: float,
    ) -> None:
        metrics, _ = evaluate(
            tasks,
            reference,
            mask,
            targets,
            reference_metrics,
            s256,
            corner,
            grid_dirs,
            config["gates"],
            context["constraints"],
            context["combined"],
        )
        rows.append(
            {
                "scene_index": scene_index,
                "sample_index": sample,
                "k_value": k_value,
                "ratio": ratio,
                "active_count": int(np.sum(mask)),
                "candidate_index_in_scene": len(candidates),
                "seed_name": seed_name,
                "stage": stage,
                "socp_target_active_rl_db": target_db,
                "solver_status": status,
                "solve_seconds": solve_seconds,
                "full_dense_row_count": context["full_dense_row_count"],
                "solver_dense_row_count": context["solver_dense_row_count"],
                **metrics,
            }
        )
        padded = np.zeros((256, KMAX), dtype=np.complex64)
        padded[:, :k_value] = tasks
        candidates.append(padded)
        candidate_seeds.append(seed_name)

    seed_rows: dict[str, dict[str, Any]] = {}
    for seed_name, seed in seed_values.items():
        context = build_seed_problem(seed, mask, targets, grid_dirs, corner, s256, config)
        seed_contexts[seed_name] = context
        append_candidate(seed, seed_name, context, "seed", float("nan"), "not_run", 0.0)
        seed_rows[seed_name] = rows[-1]

    best_pattern = max(seed_rows, key=lambda name: seed_selection_key(seed_rows[name]))
    best_balanced = max(seed_rows, key=lambda name: balanced_selection_key(seed_rows[name]))
    selected_names = [best_pattern]
    if best_balanced not in selected_names:
        selected_names.append(best_balanced)
    for fallback in sorted(seed_rows, key=lambda name: balanced_selection_key(seed_rows[name]), reverse=True):
        if len(selected_names) >= int(config["seed_policy"]["maximum_optimized_seeds_per_scene"]):
            break
        if fallback not in selected_names:
            selected_names.append(fallback)

    for seed_name in selected_names:
        context = seed_contexts[seed_name]
        current = seed_values[seed_name][context["active"]].copy()
        for target_db in config["sequential_socp"]["active_rl_stages_db"]:
            update_parameters(context["parameters"], current, float(target_db))
            started = time.time()
            try:
                context["problem"].solve(
                    solver=str(config["sequential_socp"]["solver"]),
                    warm_start=False,
                    max_iter=int(config["sequential_socp"]["solver_max_iterations"]),
                    time_limit=float(config["sequential_socp"]["solver_time_limit_seconds"]),
                    tol_gap_abs=float(config["sequential_socp"]["solver_tolerance"]),
                    tol_feas=float(config["sequential_socp"]["solver_tolerance"]),
                    verbose=False,
                )
                status = str(context["problem"].status)
            except cp.error.SolverError:
                status = "solver_error"
            elapsed = time.time() - started
            if context["variable"].value is None or status not in {"optimal", "optimal_inaccurate"}:
                rows.append(
                    {
                        "scene_index": scene_index,
                        "sample_index": sample,
                        "k_value": k_value,
                        "ratio": ratio,
                        "candidate_index_in_scene": -1,
                        "seed_name": seed_name,
                        "stage": "failed_stage",
                        "socp_target_active_rl_db": float(target_db),
                        "solver_status": status,
                        "solve_seconds": elapsed,
                    }
                )
                break
            current = np.asarray(context["variable"].value, dtype=np.complex128)
            full = np.zeros_like(reference)
            full[context["active"]] = current
            append_candidate(
                full,
                seed_name,
                context,
                "sequential_socp",
                float(target_db),
                status,
                elapsed,
            )
    return rows, np.stack(candidates), candidate_seeds


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite v1.42 output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    scene_dir = out / "scenes"
    scene_dir.mkdir(exist_ok=True)
    input_names = ("corrected_operator", "projection_run", "frozen_scenes", "pattern_operator")
    inputs = {name: resolve(config_path, config["inputs"][name]).resolve() for name in input_names}
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    file_inputs = {name: path for name, path in inputs.items() if path.is_file()}
    preregistration = {
        **config,
        "config_sha256": sha256(config_path),
        "input_sha256": {name: sha256(path) for name, path in file_inputs.items()},
        "cvxpy_runtime_version": cp.__version__,
    }
    prereg_path = out / "preregistration.json"
    if prereg_path.exists():
        previous = json.loads(prereg_path.read_text(encoding="utf-8"))
        if previous["config_sha256"] != preregistration["config_sha256"]:
            raise RuntimeError("Resume refused because the preregistered config changed")
    else:
        write_json(prereg_path, preregistration)

    with np.load(inputs["corrected_operator"], allow_pickle=False) as source:
        corrected_s4 = ri_to_complex(source["external_s4_real_imag"]).astype(np.complex128)
        map4 = ri_to_complex(source["antenna_incident_map4_real_imag"]).astype(np.complex128)
        parameters = np.asarray(source["parameters"], dtype=float)
    s256, blocks = tiled_s256(corrected_s4)
    map256 = tile_map(map4, blocks)
    base, nominal_effective, _nominal_fast, _nominal_s = load_nominal_operator(inputs["pattern_operator"])
    mapped_effective = DenseExternalEEP(nominal_effective.etheta, nominal_effective.ephi, map256)
    corner = {
        "effective": mapped_effective,
        "fast": FastPatternEvaluator(mapped_effective, base["theta_deg"], base["phi_deg"]),
    }
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    with np.load(inputs["frozen_scenes"], allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    references = load_reference_commands(inputs["projection_run"])
    if len(references) != len(data["sample_index"]):
        raise RuntimeError("Projection commands and frozen scenes do not align")
    scene_count = len(references)
    if args.max_scenes > 0:
        scene_count = min(scene_count, int(args.max_scenes))

    started = time.time()
    for scene_index in range(scene_count):
        scene_csv = scene_dir / f"scene_{scene_index:02d}.csv"
        scene_npz = scene_dir / f"scene_{scene_index:02d}.npz"
        if args.resume and scene_csv.exists() and scene_npz.exists():
            continue
        rows, candidates, seed_names = solve_scene(
            scene_index,
            data,
            references[scene_index],
            map4,
            blocks,
            s256,
            corner,
            grid_dirs,
            config,
        )
        write_csv(scene_csv, rows)
        np.savez_compressed(
            scene_npz,
            tasks_real_imag=np.stack((candidates.real, candidates.imag), axis=-1).astype(np.float32),
            seed_names=np.asarray(seed_names),
        )
        valid = [row for row in rows if "reserve11_strict_pass" in row]
        print(
            json.dumps(
                {
                    "scene": scene_index + 1,
                    "of": scene_count,
                    "sample_index": int(data["sample_index"][scene_index]),
                    "k": int(data["k_values"][scene_index]),
                    "best_pattern_margin_db": max(float(row["pattern_margin_db"]) for row in valid),
                    "best_active_rl_db": max(float(row["active_rl_floor_db"]) for row in valid),
                    "reserve11_pass": max(int(row["reserve11_strict_pass"]) for row in valid),
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
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[int(row["sample_index"])].append(row)
    scene_rows: list[dict[str, Any]] = []
    for sample, rows in sorted(grouped.items()):
        best = max(rows, key=candidate_key)
        scene_rows.append(
            {
                "sample_index": sample,
                "k_value": int(best["k_value"]),
                "ratio": float(best["ratio"]),
                "best_seed": best["seed_name"],
                "best_stage": best["stage"],
                "best_active_rl_floor_db": float(best["active_rl_floor_db"]),
                "best_pattern_margin_db": float(best["pattern_margin_db"]),
                "best_hardware_margin_db": float(best["hardware_margin_db"]),
                "engineering_strict_pass": int(best["engineering_strict_pass"]),
                "reserve11_strict_pass": int(best["reserve11_strict_pass"]),
                "best_candidate_index_in_scene": int(best["candidate_index_in_scene"]),
            }
        )
    write_csv(out / "scene_oracle.csv", scene_rows)
    per_k: list[dict[str, Any]] = []
    for k_value in (2, 4, 6):
        rows = [row for row in scene_rows if int(row["k_value"]) == k_value]
        if not rows:
            continue
        per_k.append(
            {
                "k_value": k_value,
                "scene_count": len(rows),
                "engineering_pass_count": sum(int(row["engineering_strict_pass"]) for row in rows),
                "reserve11_pass_count": sum(int(row["reserve11_strict_pass"]) for row in rows),
                "engineering_pass_rate": float(np.mean([int(row["engineering_strict_pass"]) for row in rows])),
                "reserve11_pass_rate": float(np.mean([int(row["reserve11_strict_pass"]) for row in rows])),
                "best_active_rl_floor_db": max(float(row["best_active_rl_floor_db"]) for row in rows),
                "best_pattern_margin_db": max(float(row["best_pattern_margin_db"]) for row in rows),
            }
        )
    write_csv(out / "per_k_oracle.csv", per_k)
    required = {int(value) for value in config["decision"]["required_k_values"]}
    counts = {int(row["k_value"]): int(row["reserve11_pass_count"]) for row in per_k}
    joint_gate = all(
        counts.get(k_value, 0)
        >= int(config["decision"]["minimum_reserve11_strict_count_per_required_k"])
        for k_value in required
    )
    summary = {
        "scope": config["evidence_scope"],
        "scene_count": len(scene_rows),
        "network_parameters": parameters.tolist(),
        "mapped_eep_build_seconds": mapped_effective.build_seconds,
        "engineering_strict_oracle_count": sum(int(row["engineering_strict_pass"]) for row in scene_rows),
        "reserve11_strict_oracle_count": sum(int(row["reserve11_strict_pass"]) for row in scene_rows),
        "reserve11_counts_by_k": {str(k): counts.get(k, 0) for k in (2, 4, 6)},
        "required_k_joint_gate_pass": joint_gate,
        "stop_v141_network_topology": not joint_gate,
        "authorize_element_input_impedance_redesign": not joint_gate,
        "authorize_hfss": False,
        "authorize_labels_or_critic": False,
        "elapsed_seconds": time.time() - started,
        "next_action": config["decision"]["pass_action"] if joint_gate else config["decision"]["fail_action"],
        "limitations": [
            "The corrected S256 is 64 block-diagonal copies of a finite-Q circuit S4 and omits coupling between 2x2 cells.",
            "The EEP is a mapped nominal operator, not an EEP export of an integrated corrected 16x16 structure.",
            "Failure is a stop decision for this network topology under the preregistered search, not a proof that no unconstrained global solution exists."
        ]
    }
    write_json(out / "stage_summary.json", summary)
    write_json(
        out / "stage_decision.json",
        {
            "required_k_joint_gate_pass": joint_gate,
            "stop_v141_network_topology": not joint_gate,
            "authorize_element_input_impedance_redesign": not joint_gate,
            "authorize_hfss": False,
            "authorize_larger_array": False,
            "authorize_labels_or_critic": False,
            "next_action": summary["next_action"],
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
