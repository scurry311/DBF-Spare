#!/usr/bin/env python3
"""Progressively rescue Pareto-ranked K=2/K=4 masks across three frequencies."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import metric_at
from hfss_task_fullwave_validate import pattern_grid_dirs
from run_v16_k6_multifrequency_rescue import Variant, optimize_common_command
from run_v16_robust_drift_oracle import complex_to_ri, load_npz, ri_to_complex
from run_v19_nominal_9p96_joint_projection import evaluate_command
from run_v20_three_frequency_mask_weight_joint import (
    DEFAULT_HIGH,
    DEFAULT_LOW,
    DEFAULT_NOMINAL,
    DEFAULT_POOL,
    DEFAULT_PROJECTED,
    DEFAULT_SOURCE,
    operator_bundle,
    scene_states,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v201_pareto_joint_feasibility_rescue.json"
DEFAULT_SCREEN = ROOT / "hfss_outputs" / "v201_warm_mask_pareto_screen_20260729_run01"
DEFAULT_V20 = ROOT / "hfss_outputs" / "v20_three_frequency_joint_search_20260729_run02"
DEFAULT_ALPHA = ROOT / "hfss_outputs" / "v201_dense_alpha_eep_20260729_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v201_k24_progressive_rescue_20260729_run01"
KMAX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--v20-dir", type=Path, default=DEFAULT_V20)
    parser.add_argument("--alpha-dir", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--projected", type=Path, default=DEFAULT_PROJECTED)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--low-operator", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--high-operator", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-candidates", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def candidate_key(row: dict[str, Any]) -> tuple[int, int, float, float, float]:
    return (
        int(row["design_reserve11_pass"]),
        int(row["all_corner_strict_pass"]),
        min(float(row["robust_worst_margin_db"]), 3.0),
        min(float(row["robust_pattern_margin_db"]), 3.0),
        min(float(row["robust_active_rl_margin_db"]), 3.0),
    )


def evaluate_value(
    command: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    reference: dict[str, float],
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    grid_dirs: np.ndarray,
    gates: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    values, margins = evaluate_command(
        command, mask, targets, reference, corners, states, grid_dirs, gates
    )
    values.update(metadata)
    values.update(
        {
            "robust_psll_margin_db": float(np.min(margins[:, 0])),
            "design_reserve11_pass": int(
                values["all_corner_pattern_pass"]
                and float(values["robust_active_rl_margin_db"]) >= 1.0
                and float(values["robust_hardware_margin_db"]) >= 0.0
            ),
        }
    )
    return values, margins


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite K2/K4 rescue: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    v20_protocol = json.loads(
        (ROOT / "configs" / "v20_three_frequency_mask_weight_joint.json").read_text(
            encoding="utf-8"
        )
    )
    gates = v20_protocol["gates"]
    rescue = protocol["k24_rescue"]
    selection_rows = read_csv(args.screen_dir / "pareto_selection.csv")
    selection_arrays = load_npz(args.screen_dir / "pareto_selected_candidates.npz")
    selection_commands = ri_to_complex(selection_arrays["tasks_real_imag"])
    failed = {int(value) for value in rescue["failed_sample_indices"]}
    selected_indices = [
        index
        for index, row in enumerate(selection_rows)
        if int(row["sample_index"]) in failed
    ]
    if args.max_candidates > 0:
        selected_indices = selected_indices[: args.max_candidates]
    projected = load_npz(args.projected)
    source = load_npz(args.source)
    pool = load_npz(args.pool)
    nominal_base, nominal = operator_bundle(args.nominal_operator, 10.0)
    low_base, low = operator_bundle(args.low_operator, 9.96)
    high_base, high = operator_bundle(args.high_operator, 10.04)
    corners = {
        "nominal_identity": nominal,
        "frequency_low_identity": low,
        "frequency_low_E2_source": low,
        "frequency_high_identity": high,
        "frequency_high_E2_source": high,
    }
    grid_dirs = pattern_grid_dirs(nominal_base["theta_deg"], nominal_base["phi_deg"])
    element_ixiy = np.asarray(nominal_base["element_ixiy"], dtype=np.int64)
    projected_samples = np.asarray(projected["sample_index"], dtype=np.int64)
    projected_lookup = {int(sample): index for index, sample in enumerate(projected_samples)}
    source_indices = np.asarray(projected["source_candidate_index"], dtype=np.int64)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    context: dict[int, dict[str, Any]] = {}
    for sample in failed:
        projected_index = projected_lookup[sample]
        k_value = int(projected["k_values"][projected_index])
        targets = np.asarray(projected["targets_deg"][projected_index, :k_value], dtype=float)
        states, _ = scene_states(
            pool,
            np.flatnonzero(pool_samples == sample),
            element_ixiy,
            int(protocol["seed"]),
        )
        original = ri_to_complex(
            source[
                "nominal_external_task_weights_real_imag"
            ][int(source_indices[projected_index]), :, :k_value]
        )
        context[sample] = {
            "k": k_value,
            "targets": targets,
            "states": states,
            "reference": metric_at(nominal["fast"].evaluate(original, targets), 0),
        }

    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_commands_out: list[np.ndarray] = []
    selected_masks_out: list[np.ndarray] = []
    selected_margins_out: list[np.ndarray] = []
    started = time.time()
    for position, selection_index in enumerate(selected_indices, start=1):
        selection = selection_rows[selection_index]
        sample = int(selection["sample_index"])
        ratio = float(selection["ratio"])
        generated_index = int(selection["generated_index"])
        role = str(selection["selection_role"])
        k_value = int(context[sample]["k"])
        targets = context[sample]["targets"]
        states = context[sample]["states"]
        reference = context[sample]["reference"]
        mask = np.asarray(selection_arrays["masks"][selection_index], dtype=bool)
        current = np.asarray(selection_commands[selection_index, :, :k_value], dtype=np.complex64)
        path_candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
        initial_values, initial_margins = evaluate_value(
            current,
            mask,
            targets,
            reference,
            corners,
            states,
            grid_dirs,
            gates,
            {
                "selection_index": selection_index,
                "generated_index": generated_index,
                "sample_index": sample,
                "k": k_value,
                "ratio": ratio,
                "selection_role": role,
                "target_stage_db": "warm",
                "backtracking_alpha": 0.0,
            },
        )
        all_rows.append(initial_values)
        path_candidates.append((initial_values, current.copy(), initial_margins))
        for target_db in rescue["active_rl_target_schedule_db"]:
            variant = Variant(
                name=f"v201_progressive_rl{float(target_db):g}",
                target_amplitude_mode="median",
                active_rl_design_min_db=float(target_db),
                local_radius_deg=float(rescue["regional_radius_deg"]),
                regional_ceiling_db=float(rescue["regional_ceiling_db"]),
                joint_projection_passes=int(rescue["joint_projection_passes"]),
                corner_sweeps=int(rescue["corner_sweeps_per_target"]),
            )
            proposed, diagnostics = optimize_common_command(
                current,
                mask,
                targets,
                corners,
                states,
                grid_dirs,
                variant,
                int(rescue["active_rl_nullspace_steps_per_target"]),
                float(rescue["active_rl_nullspace_step_size"]),
                True,
            )
            stage_candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
            for alpha in rescue["backtracking_alphas"]:
                command = ((1.0 - float(alpha)) * current + float(alpha) * proposed).astype(
                    np.complex64
                )
                command[~mask] = 0.0
                values, margins = evaluate_value(
                    command,
                    mask,
                    targets,
                    reference,
                    corners,
                    states,
                    grid_dirs,
                    gates,
                    {
                        "selection_index": selection_index,
                        "generated_index": generated_index,
                        "sample_index": sample,
                        "k": k_value,
                        "ratio": ratio,
                        "selection_role": role,
                        "target_stage_db": float(target_db),
                        "backtracking_alpha": float(alpha),
                        **{f"optimizer_{key}": float(value) for key, value in diagnostics.items()},
                    },
                )
                all_rows.append(values)
                stage_candidates.append((values, command, margins))
                path_candidates.append((values, command, margins))
            _stage_values, current, _stage_margins = max(
                stage_candidates, key=lambda item: candidate_key(item[0])
            )
        best_values, best_command, best_margins = max(
            path_candidates, key=lambda item: candidate_key(item[0])
        )
        best_values = dict(best_values)
        best_values.update(
            {
                "path_index": len(selected_rows),
                "selected_best": 1,
                "path_candidate_count": len(path_candidates),
            }
        )
        selected_rows.append(best_values)
        padded = np.zeros((256, KMAX), dtype=np.complex64)
        padded[:, :k_value] = best_command
        selected_commands_out.append(padded)
        selected_masks_out.append(mask.astype(np.int8))
        selected_margins_out.append(best_margins)
        if position % 4 == 0 or position == len(selected_indices):
            print(
                f"v20.1 K24 rescue {position:03d}/{len(selected_indices):03d} "
                f"sample={sample} K={k_value} ratio={ratio:.1f} role={role} "
                f"strict={best_values['all_corner_strict_pass']} "
                f"reserve11={best_values['design_reserve11_pass']} "
                f"margin={float(best_values['robust_worst_margin_db']):.3f} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
            write_csv(args.out_dir / "checkpoint_path_metrics.csv", all_rows)
            write_csv(args.out_dir / "checkpoint_selected_paths.csv", selected_rows)
            (args.out_dir / "checkpoint_progress.json").write_text(
                json.dumps(
                    {
                        "completed_path_count": position,
                        "total_path_count": len(selected_indices),
                        "sample_index": sample,
                        "ratio": ratio,
                        "elapsed_seconds": time.time() - started,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    write_csv(args.out_dir / "progressive_path_metrics.csv", all_rows)
    write_csv(args.out_dir / "selected_path_metrics.csv", selected_rows)
    np.savez_compressed(
        args.out_dir / "selected_path_candidates.npz",
        path_index=np.arange(len(selected_rows), dtype=np.int64),
        sample_index=np.asarray([int(row["sample_index"]) for row in selected_rows], np.int64),
        k_values=np.asarray([int(row["k"]) for row in selected_rows], np.int8),
        ratio=np.asarray([float(row["ratio"]) for row in selected_rows], np.float32),
        generated_index=np.asarray([int(row["generated_index"]) for row in selected_rows], np.int64),
        masks=np.stack(selected_masks_out),
        tasks_real_imag=complex_to_ri(np.stack(selected_commands_out)),
        robust_margins=np.stack(selected_margins_out),
        corner_names=np.asarray(list(corners)),
        selection_roles=np.asarray([str(row["selection_role"]) for row in selected_rows]),
    )

    combined_rows = [
        *read_csv(args.v20_dir / "full_refined_candidate_metrics.csv"),
        *read_csv(args.alpha_dir / "dense_alpha_metrics.csv"),
        *[{key: str(value) for key, value in row.items()} for row in all_rows],
    ]
    scene_rows: list[dict[str, Any]] = []
    for sample in sorted(set(int(value) for value in projected_samples)):
        projected_index = projected_lookup[sample]
        k_value = int(projected["k_values"][projected_index])
        members = [row for row in combined_rows if int(row["sample_index"]) == sample]
        strict = [row for row in members if int(row["all_corner_strict_pass"]) == 1]
        reserve = [
            row
            for row in members
            if int(row.get("design_reserve11_pass", row.get("design_reserve_pass", 0))) == 1
        ]
        best = max(members, key=lambda row: float(row["robust_worst_margin_db"]))
        scene_rows.append(
            {
                "sample_index": sample,
                "k": k_value,
                "strict_oracle_pass": int(bool(strict)),
                "reserve11_oracle_pass": int(bool(reserve)),
                "minimum_strict_ratio": min(float(row["ratio"]) for row in strict) if strict else float("nan"),
                "best_worst_margin_db": float(best["robust_worst_margin_db"]),
                "candidate_count": len(members),
            }
        )
    write_csv(args.out_dir / "k24_scene_oracle.csv", scene_rows)
    counts = {
        k: sum(int(row["strict_oracle_pass"]) for row in scene_rows if int(row["k"]) == k)
        for k in (2, 4, 6)
    }
    reserve_counts = {
        k: sum(int(row["reserve11_oracle_pass"]) for row in scene_rows if int(row["k"]) == k)
        for k in (2, 4, 6)
    }
    gate = bool(
        counts[2] >= int(protocol["stage_gate"]["k2_strict_count_min"])
        and counts[4] >= int(protocol["stage_gate"]["k4_strict_count_min"])
    )
    summary = {
        "protocol": protocol["protocol"],
        "selected_path_count": len(selected_rows),
        "evaluated_command_count": len(all_rows),
        "strict_counts_by_k": {str(key): value for key, value in counts.items()},
        "reserve11_counts_by_k": {str(key): value for key, value in reserve_counts.items()},
        "k24_stage_gate_pass": gate,
        "k6_execution_allowed": gate,
        "hfss_smoke_allowed": False,
        "critic_training_allowed": False,
        "thresholds_changed": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
