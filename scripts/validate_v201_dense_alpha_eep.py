#!/usr/bin/env python3
"""Validate predicted same-mask Pareto intersections on physical EEP/S256 operators."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import metric_at
from hfss_task_fullwave_validate import pattern_grid_dirs
from run_v16_robust_drift_oracle import load_npz, ri_to_complex
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
DEFAULT_PARENT = ROOT / "hfss_outputs" / "v20_three_frequency_joint_search_20260729_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v201_dense_alpha_eep_20260729_run01"
MARGIN_NAMES = ("psll", "nearest_iso", "local20_iso", "mainlobe", "active_rl", "hardware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interpolated_path_score(rows: list[dict[str, str]]) -> tuple[float, float]:
    ordered = sorted(rows, key=lambda row: float(row["weight_path_alpha"]))
    alpha = np.asarray([float(row["weight_path_alpha"]) for row in ordered])
    pattern = np.asarray([float(row["robust_pattern_margin_db"]) for row in ordered])
    active_rl = np.asarray([float(row["robust_active_rl_margin_db"]) for row in ordered])
    hardware = np.asarray([float(row["robust_hardware_margin_db"]) for row in ordered])
    dense = np.linspace(0.0, 1.0, 1001)
    margin = np.minimum.reduce(
        [
            np.interp(dense, alpha, pattern),
            np.interp(dense, alpha, active_rl),
            np.interp(dense, alpha, hardware),
        ]
    )
    best = int(np.argmax(margin))
    return float(dense[best]), float(margin[best])


def select_path(
    rows: list[dict[str, str]],
    sample: int,
    ratio: float,
    predicted_alpha: float,
) -> tuple[int, list[dict[str, str]], float, float]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row["round"] != "v19_frozen"
            and int(row["sample_index"]) == sample
            and np.isclose(float(row["ratio"]), ratio, atol=1.0e-5)
            and row.get("generated_index", "") != ""
        ):
            groups[int(row["generated_index"])].append(row)
    candidates: list[tuple[tuple[int, float, float], int, list[dict[str, str]], float, float]] = []
    for generated_index, members in groups.items():
        pattern = any(int(row["all_corner_pattern_pass"]) for row in members)
        active_rl = any(int(row["all_corner_active_rl_pass"]) for row in members)
        strict = any(int(row["all_corner_strict_pass"]) for row in members)
        alpha, margin = interpolated_path_score(members)
        key = (int(pattern and active_rl and not strict), margin, -abs(alpha - predicted_alpha))
        candidates.append((key, generated_index, members, alpha, margin))
    if not candidates:
        raise RuntimeError(f"No weight path for sample={sample}, ratio={ratio}")
    _key, generated_index, members, alpha, margin = max(candidates, key=lambda item: item[0])
    return generated_index, members, alpha, margin


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dense-alpha validation: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    parent_rows = read_csv(args.parent_dir / "full_refined_candidate_metrics.csv")
    parent_arrays = load_npz(args.parent_dir / "full_refined_candidates.npz")
    parent_commands = ri_to_complex(parent_arrays["tasks_real_imag"])
    projected = load_npz(args.projected)
    source = load_npz(args.source)
    pool = load_npz(args.pool)
    nominal_base, nominal = operator_bundle(args.nominal_operator, 10.0)
    low_base, low = operator_bundle(args.low_operator, 9.96)
    high_base, high = operator_bundle(args.high_operator, 10.04)
    for other in (low_base, high_base):
        if not np.array_equal(nominal_base["element_ixiy"], other["element_ixiy"]):
            raise RuntimeError("Three-frequency port order differs")
    corners = {
        "nominal_identity": nominal,
        "frequency_low_identity": low,
        "frequency_low_E2_source": low,
        "frequency_high_identity": high,
        "frequency_high_E2_source": high,
    }
    grid_dirs = pattern_grid_dirs(nominal_base["theta_deg"], nominal_base["phi_deg"])
    samples = np.asarray(projected["sample_index"], dtype=np.int64)
    projected_lookup = {int(sample): index for index, sample in enumerate(samples)}
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    source_indices = np.asarray(projected["source_candidate_index"], dtype=np.int64)
    gates = json.loads(
        (ROOT / "configs" / "v20_three_frequency_mask_weight_joint.json").read_text(
            encoding="utf-8"
        )
    )["gates"]
    output_rows: list[dict[str, Any]] = []
    output_commands: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []

    for request in protocol["dense_alpha_validation"]:
        sample = int(request["sample_index"])
        ratio = float(request["ratio"])
        predicted = float(request["predicted_alpha"])
        generated_index, members, interpolated_alpha, interpolated_margin = select_path(
            parent_rows, sample, ratio, predicted
        )
        endpoint_rows = {
            float(row["weight_path_alpha"]): row for row in members
        }
        low_row = endpoint_rows[min(endpoint_rows)]
        high_row = endpoint_rows[max(endpoint_rows)]
        command_low = np.asarray(
            parent_commands[int(low_row["evaluation_index"]), :, : int(low_row["k"])],
            dtype=np.complex64,
        )
        command_high = np.asarray(
            parent_commands[int(high_row["evaluation_index"]), :, : int(high_row["k"])],
            dtype=np.complex64,
        )
        mask = np.asarray(parent_arrays["masks"][int(low_row["evaluation_index"])], dtype=bool)
        projected_index = projected_lookup[sample]
        source_index = int(source_indices[projected_index])
        k_value = int(projected["k_values"][projected_index])
        targets = np.asarray(projected["targets_deg"][projected_index, :k_value], dtype=float)
        states, _ = scene_states(
            pool,
            np.flatnonzero(pool_samples == sample),
            np.asarray(nominal_base["element_ixiy"], dtype=np.int64),
            int(protocol["seed"]),
        )
        original = ri_to_complex(
            source["nominal_external_task_weights_real_imag"][source_index, :, :k_value]
        )
        reference = metric_at(nominal["fast"].evaluate(original, targets), 0)
        half_width = float(protocol["dense_alpha_half_width"])
        step = float(protocol["dense_alpha_step"])
        start = max(0.0, predicted - half_width)
        stop = min(1.0, predicted + half_width)
        alphas = np.arange(start, stop + step * 0.5, step)
        request_rows: list[dict[str, Any]] = []
        for alpha in alphas:
            command = ((1.0 - alpha) * command_low + alpha * command_high).astype(np.complex64)
            values, margins = evaluate_command(
                command, mask, targets, reference, corners, states, grid_dirs, gates
            )
            values.update(
                {
                    "sample_index": sample,
                    "k": k_value,
                    "ratio": ratio,
                    "generated_index": generated_index,
                    "alpha": float(alpha),
                    "interpolated_prediction_alpha": interpolated_alpha,
                    "interpolated_prediction_margin_db": interpolated_margin,
                    "robust_psll_margin_db": float(np.min(margins[:, 0])),
                    "design_reserve11_pass": int(
                        values["all_corner_pattern_pass"]
                        and float(values["robust_active_rl_margin_db"]) >= 1.0
                        and float(values["robust_hardware_margin_db"]) >= 0.0
                    ),
                }
            )
            request_rows.append(values)
            output_rows.append(values)
            padded = np.zeros((256, 6), dtype=np.complex64)
            padded[:, :k_value] = command
            output_commands.append(padded)
        best = max(
            request_rows,
            key=lambda row: (
                int(row["design_reserve11_pass"]),
                int(row["all_corner_strict_pass"]),
                float(row["robust_worst_margin_db"]),
            ),
        )
        summaries.append(
            {
                "sample_index": sample,
                "k": k_value,
                "ratio": ratio,
                "generated_index": generated_index,
                "evaluated_alpha_count": len(alphas),
                "best_alpha": float(best["alpha"]),
                "best_worst_margin_db": float(best["robust_worst_margin_db"]),
                "strict_pass": int(best["all_corner_strict_pass"]),
                "design_reserve11_pass": int(best["design_reserve11_pass"]),
                "best_active_rl_floor_db": 10.0 + float(best["robust_active_rl_margin_db"]),
                "best_pattern_margin_db": float(best["robust_pattern_margin_db"]),
            }
        )

    write_csv(args.out_dir / "dense_alpha_metrics.csv", output_rows)
    write_csv(args.out_dir / "dense_alpha_summary.csv", summaries)
    np.savez_compressed(
        args.out_dir / "dense_alpha_commands.npz",
        commands_real_imag=np.stack(
            [np.stack((value.real, value.imag), axis=-1) for value in output_commands]
        ).astype(np.float32),
    )
    summary = {
        "protocol": protocol["protocol"],
        "request_count": len(summaries),
        "strict_pass_count": sum(int(row["strict_pass"]) for row in summaries),
        "reserve11_pass_count": sum(int(row["design_reserve11_pass"]) for row in summaries),
        "results": summaries,
        "hfss_smoke_allowed": False,
        "critic_training_allowed": False,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
