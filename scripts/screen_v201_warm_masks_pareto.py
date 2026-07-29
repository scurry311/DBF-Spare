#!/usr/bin/env python3
"""Batch-evaluate all v20 warm masks and select four physical Pareto roles."""

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
from generate_v09_eep_development_candidates import full_active_metrics, physical_margins
from hfss_task_fullwave_validate import pattern_grid_dirs
from run_v16_robust_drift_oracle import (
    apply_calibration,
    complex_to_ri,
    hardware_margin,
    load_npz,
    ri_to_complex,
)
from run_v20_three_frequency_mask_weight_joint import (
    DEFAULT_HIGH,
    DEFAULT_LOW,
    DEFAULT_NOMINAL,
    DEFAULT_POOL,
    DEFAULT_PROJECTED,
    DEFAULT_SOURCE,
    adapt_command,
    load_parent_records,
    operator_bundle,
    scene_states,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v201_pareto_joint_feasibility_rescue.json"
DEFAULT_V20 = ROOT / "hfss_outputs" / "v20_three_frequency_joint_search_20260729_run02"
DEFAULT_PARENT = ROOT / "hfss_outputs" / "v16_robust_drift_oracle_20260727_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v201_warm_mask_pareto_screen_20260729_run01"
ROBUST_MARGIN_NAMES = ("psll", "nearest_iso", "local20_iso", "mainlobe", "active_rl", "hardware")
EPS = 1.0e-12
KMAX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--v20-dir", type=Path, default=DEFAULT_V20)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--projected", type=Path, default=DEFAULT_PROJECTED)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--low-operator", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--high-operator", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
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


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - float(np.min(values))) / max(float(np.ptp(values)), EPS)


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if float(row["robust_hardware_margin_db"]) >= 0.0]
    if not eligible:
        eligible = rows
    output: list[dict[str, Any]] = []
    for candidate in eligible:
        pattern = float(candidate["robust_pattern_margin_db"])
        active_rl = float(candidate["robust_active_rl_margin_db"])
        dominated = any(
            other is not candidate
            and float(other["robust_pattern_margin_db"]) >= pattern
            and float(other["robust_active_rl_margin_db"]) >= active_rl
            and (
                float(other["robust_pattern_margin_db"]) > pattern
                or float(other["robust_active_rl_margin_db"]) > active_rl
            )
            for other in eligible
        )
        if not dominated:
            output.append(candidate)
    return output


def role_rankings(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pattern = sorted(
        rows,
        key=lambda row: (
            float(row["robust_pattern_margin_db"]),
            float(row["robust_active_rl_margin_db"]),
            float(row["robust_worst_margin_db"]),
        ),
        reverse=True,
    )
    active_rl = sorted(
        rows,
        key=lambda row: (
            float(row["robust_active_rl_margin_db"]),
            float(row["robust_pattern_margin_db"]),
            float(row["robust_worst_margin_db"]),
        ),
        reverse=True,
    )
    max_min = sorted(
        rows,
        key=lambda row: (
            float(row["robust_worst_margin_db"]),
            float(row["robust_pattern_margin_db"]),
            float(row["robust_active_rl_margin_db"]),
        ),
        reverse=True,
    )
    front = pareto_front(rows)
    pattern_values = normalize01(
        np.asarray([float(row["robust_pattern_margin_db"]) for row in front])
    )
    active_values = normalize01(
        np.asarray([float(row["robust_active_rl_margin_db"]) for row in front])
    )
    hardware_values = normalize01(
        np.asarray([float(row["robust_hardware_margin_db"]) for row in front])
    )
    knee_scores = np.minimum(pattern_values, active_values) + 0.10 * hardware_values
    knee_order = np.argsort(knee_scores, kind="stable")[::-1]
    knee = [front[int(index)] for index in knee_order]
    remaining = [row for row in max_min if row not in knee]
    return {
        "pattern_best": pattern,
        "active_rl_best": active_rl,
        "max_min_best": max_min,
        "pareto_knee": [*knee, *remaining],
    }


def choose_roles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rankings = role_rankings(rows)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for role in ("pattern_best", "active_rl_best", "max_min_best", "pareto_knee"):
        candidate = next(row for row in rankings[role] if str(row["mask_hash"]) not in used)
        used.add(str(candidate["mask_hash"]))
        selected.append({**candidate, "selection_role": role})
    return selected


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Pareto screen: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    v20_protocol = json.loads(
        (ROOT / "configs" / "v20_three_frequency_mask_weight_joint.json").read_text(
            encoding="utf-8"
        )
    )
    gates = v20_protocol["gates"]
    manifest = read_csv(args.v20_dir / "generated_mask_manifest.csv")
    generated = load_npz(args.v20_dir / "generated_mask_pool.npz")
    v20_rows = read_csv(args.v20_dir / "full_refined_candidate_metrics.csv")
    v20_arrays = load_npz(args.v20_dir / "full_refined_candidates.npz")
    v20_commands = ri_to_complex(v20_arrays["tasks_real_imag"])
    projected = load_npz(args.projected)
    source = load_npz(args.source)
    pool = load_npz(args.pool)
    parent_records = load_parent_records(args.parent_dir, pool)
    nominal_base, nominal = operator_bundle(args.nominal_operator, 10.0)
    low_base, low = operator_bundle(args.low_operator, 9.96)
    high_base, high = operator_bundle(args.high_operator, 10.04)
    for other in (low_base, high_base):
        if not (
            np.array_equal(nominal_base["element_ixiy"], other["element_ixiy"])
            and np.array_equal(nominal_base["theta_deg"], other["theta_deg"])
            and np.array_equal(nominal_base["phi_deg"], other["phi_deg"])
        ):
            raise RuntimeError("Three-frequency operator ordering/grid differs")
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
    v20_row_lookup = {int(row["evaluation_index"]): row for row in v20_rows}
    parent_lookup: dict[tuple[int, float, str, int], dict[str, Any]] = {}
    for record in parent_records:
        parent_lookup[
            (
                int(record["sample_index"]),
                round(float(record["ratio"]), 3),
                str(record["origin"]),
                int(record["source_index"]),
            )
        ] = record
    masks = np.asarray(generated["masks"], dtype=bool)
    if len(manifest) != int(protocol["warm_mask_screen"]["expected_mask_count"]):
        raise RuntimeError("Generated mask count changed")
    if masks.shape != (len(manifest), 256):
        raise RuntimeError("Generated mask array does not match manifest")
    warm_commands = np.zeros((len(manifest), 256, KMAX), dtype=np.complex64)
    warm_rows: list[dict[str, Any]] = []
    robust_margins = np.zeros((len(manifest), len(corners), len(ROBUST_MARGIN_NAMES)), np.float32)
    grouped_indices: dict[tuple[int, float], list[int]] = defaultdict(list)
    for index, row in enumerate(manifest):
        grouped_indices[(int(row["sample_index"]), round(float(row["ratio"]), 3))].append(index)

    started = time.time()
    for group_position, ((sample, ratio), indices) in enumerate(sorted(grouped_indices.items()), start=1):
        projected_index = projected_lookup[sample]
        k_value = int(projected["k_values"][projected_index])
        targets = np.asarray(projected["targets_deg"][projected_index, :k_value], dtype=float)
        source_index = int(source_indices[projected_index])
        states, _ = scene_states(
            pool,
            np.flatnonzero(pool_samples == sample),
            element_ixiy,
            int(protocol["seed"]),
        )
        original = ri_to_complex(
            source["nominal_external_task_weights_real_imag"][source_index, :, :k_value]
        )
        reference = metric_at(nominal["fast"].evaluate(original, targets), 0)
        commands: list[np.ndarray] = []
        for generated_index in indices:
            row = manifest[generated_index]
            if row["round"] == "initial":
                key = (
                    sample,
                    round(ratio, 3),
                    str(row["parent_origin"]),
                    int(row["parent_source_index"]),
                )
                if key not in parent_lookup:
                    raise RuntimeError(f"Missing parent record for generated index {generated_index}")
                parent = np.asarray(parent_lookup[key]["command"], dtype=np.complex64)
            elif row["round"] == "alternating":
                evaluation = int(row["parent_source_index"])
                if evaluation not in v20_row_lookup:
                    raise RuntimeError(f"Missing v20 parent evaluation {evaluation}")
                parent = np.asarray(v20_commands[evaluation, :, :k_value], dtype=np.complex64)
            else:
                raise RuntimeError(f"Unknown generated round: {row['round']}")
            command = adapt_command(parent, masks[generated_index], targets, nominal["effective"], grid_dirs)
            commands.append(command)
            warm_commands[generated_index, :, :k_value] = command
        command_batch = np.stack(commands)
        group_margins = np.zeros((len(indices), len(corners), len(ROBUST_MARGIN_NAMES)), np.float32)
        corner_values: list[dict[str, Any]] = [dict() for _ in indices]
        for corner_index, (corner_name, corner) in enumerate(corners.items()):
            actual = np.stack(
                [
                    apply_calibration(command_batch[local], masks[generated_index], states[corner_name])
                    for local, generated_index in enumerate(indices)
                ]
            )
            pattern_batch = corner["fast"].evaluate(actual, targets)
            for local, generated_index in enumerate(indices):
                metrics = metric_at(pattern_batch, local)
                active = full_active_metrics(actual[local], masks[generated_index], corner["s"])
                physical = physical_margins(metrics, reference, active)
                hardware_value, hardware = hardware_margin(
                    command_batch[local],
                    actual[local],
                    masks[generated_index],
                    targets,
                    corner["effective"],
                    grid_dirs,
                    gates,
                )
                margins = np.concatenate((physical, np.asarray([hardware_value], np.float32)))
                group_margins[local, corner_index] = margins
                corner_values[local].update(
                    {
                        f"{corner_name}_psll_db": float(metrics["psll_db"]),
                        f"{corner_name}_nearest_iso_db": float(metrics["nearest_iso_db"]),
                        f"{corner_name}_local_iso_db": float(metrics["local_iso_db"]),
                        f"{corner_name}_weakest_target_gain_db": float(metrics["weakest_target_gain_db"]),
                        f"{corner_name}_active_rl_floor_db": float(active["active_rl_floor_db"]),
                        f"{corner_name}_hardware_margin_db": float(hardware_value),
                    }
                )
        for local, generated_index in enumerate(indices):
            robust_margins[generated_index] = group_margins[local]
            minimum = np.min(group_margins[local], axis=0)
            base = manifest[generated_index]
            warm_rows.append(
                {
                    "generated_index": generated_index,
                    "sample_index": sample,
                    "k": k_value,
                    "ratio": ratio,
                    "round": base["round"],
                    "mask_family": base["mask_family"],
                    "mask_hash": base["mask_hash"],
                    "robust_worst_margin_db": float(np.min(minimum)),
                    "robust_pattern_margin_db": float(np.min(minimum[:4])),
                    "robust_active_rl_margin_db": float(minimum[4]),
                    "robust_hardware_margin_db": float(minimum[5]),
                    "all_corner_strict_pass": int(float(np.min(minimum)) >= 0.0),
                    "all_corner_pattern_pass": int(float(np.min(minimum[:4])) >= 0.0),
                    "all_corner_active_rl_pass": int(float(minimum[4]) >= 0.0),
                    "design_reserve11_pass": int(
                        float(np.min(minimum[:4])) >= 0.0
                        and float(minimum[4]) >= 1.0
                        and float(minimum[5]) >= 0.0
                    ),
                    **{
                        f"robust_{name}_margin_db": float(value)
                        for name, value in zip(ROBUST_MARGIN_NAMES, minimum)
                    },
                    **corner_values[local],
                }
            )
        if group_position % 8 == 0 or group_position == len(grouped_indices):
            print(
                f"warm EEP screen {group_position:02d}/{len(grouped_indices):02d} "
                f"candidates={group_position * len(indices)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    warm_rows.sort(key=lambda row: int(row["generated_index"]))
    write_csv(args.out_dir / "warm_mask_metrics.csv", warm_rows)
    rows_by_group: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in warm_rows:
        rows_by_group[(int(row["sample_index"]), round(float(row["ratio"]), 3))].append(row)
    selected_rows: list[dict[str, Any]] = []
    selected_commands: list[np.ndarray] = []
    selected_masks: list[np.ndarray] = []
    selected_margins: list[np.ndarray] = []
    for (sample, ratio), rows in sorted(rows_by_group.items()):
        selected = choose_roles(rows)
        if len(selected) != int(protocol["warm_mask_screen"]["selected_masks_per_scene_ratio"]):
            raise RuntimeError("Pareto role selection count changed")
        for row in selected:
            generated_index = int(row["generated_index"])
            selected_rows.append({**row, "selection_index": len(selected_rows)})
            selected_commands.append(warm_commands[generated_index])
            selected_masks.append(masks[generated_index].astype(np.int8))
            selected_margins.append(robust_margins[generated_index])
    write_csv(args.out_dir / "pareto_selection.csv", selected_rows)
    np.savez_compressed(
        args.out_dir / "pareto_selected_candidates.npz",
        selection_index=np.arange(len(selected_rows), dtype=np.int64),
        generated_index=np.asarray([int(row["generated_index"]) for row in selected_rows], np.int64),
        sample_index=np.asarray([int(row["sample_index"]) for row in selected_rows], np.int64),
        k_values=np.asarray([int(row["k"]) for row in selected_rows], np.int8),
        ratio=np.asarray([float(row["ratio"]) for row in selected_rows], np.float32),
        masks=np.stack(selected_masks),
        tasks_real_imag=complex_to_ri(np.stack(selected_commands)),
        robust_margins=np.stack(selected_margins),
        corner_names=np.asarray(list(corners)),
        robust_margin_names=np.asarray(ROBUST_MARGIN_NAMES),
        selection_roles=np.asarray([str(row["selection_role"]) for row in selected_rows]),
    )
    summary = {
        "protocol": protocol["protocol"],
        "warm_candidate_count": len(warm_rows),
        "scene_ratio_group_count": len(rows_by_group),
        "selected_candidate_count": len(selected_rows),
        "selected_per_group": int(protocol["warm_mask_screen"]["selected_masks_per_scene_ratio"]),
        "warm_strict_count": sum(int(row["all_corner_strict_pass"]) for row in warm_rows),
        "warm_reserve11_count": sum(int(row["design_reserve11_pass"]) for row in warm_rows),
        "role_counts": {
            role: sum(str(row["selection_role"]) == role for row in selected_rows)
            for role in protocol["warm_mask_screen"]["selection_roles"]
        },
        "thresholds_changed": False,
        "hfss_smoke_allowed": False,
        "critic_training_allowed": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
