#!/usr/bin/env python3
"""Compare v1.20 physical S8 runs and audit sparse-graph residual support."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from design_v119_multiport_post_decoupler import reordered_network
from design_v120_joint_feed_fanout_sparse_graph import sparse_pi_s8, unpack
from run_v1191_multiconductor_post_block import phase_align


ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS_CONFIG = ROOT / "configs" / "v120_joint_feed_fanout_sparse_graph.json"
SYNTHESIS_SUMMARY = ROOT / "baselines" / "2026-07-31-v120-joint-feed-fanout-sparse-graph" / "snapshots" / "circuit_synthesis.json"
RUNS = {
    "run01_nonuniform_width": ROOT / "hfss_outputs" / "v120_sparse_graph_physical_front_gate_20260731_run01" / "physical_s8_direct01",
    "run02_uniform_50ohm_width": ROOT / "hfss_outputs" / "v120_sparse_graph_physical_front_gate_20260731_run02" / "physical_s8_direct01",
}
OUT = ROOT / "hfss_outputs" / "v120_joint_feed_fanout_sparse_graph_20260731_summary"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def support_masks(graph: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    masks = {name: np.zeros((8, 8), dtype=bool) for name in ("reflection_diagonal", "same_side_graph", "through_diagonal", "through_graph")}
    for side in (0, 4):
        for port in range(4):
            masks["reflection_diagonal"][side + port, side + port] = True
        for first, second in graph:
            masks["same_side_graph"][side + first, side + second] = True
            masks["same_side_graph"][side + second, side + first] = True
    for port in range(4):
        masks["through_diagonal"][port, 4 + port] = True
        masks["through_diagonal"][4 + port, port] = True
    for first, second in graph:
        for row, column in ((first, 4 + second), (second, 4 + first), (4 + first, second), (4 + second, first)):
            masks["through_graph"][row, column] = True
    return masks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = read_json(SYNTHESIS_CONFIG)
    synthesis = read_json(SYNTHESIS_SUMMARY)
    graph = [tuple(int(value) for value in pair) for pair in config["manufacturable_graph_pairs"]]
    _, series_ground, series_pair, input_ground, input_pair, output_ground, output_pair = unpack(np.asarray(synthesis["optimized_parameters"], dtype=float))
    names = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    masks = support_masks(graph)
    supported = np.logical_or.reduce(list(masks.values()))
    rows: list[dict[str, Any]] = []
    run_summaries: dict[str, Any] = {}
    for run_name, case in RUNS.items():
        touchstone = case / "v120_sparse_graph_physical_front_direct01.s8p"
        frequencies, physical = reordered_network(touchstone, names, 8)
        target = np.stack([
            sparse_pi_s8(float(frequency), series_ground, series_pair, input_ground, input_pair, output_ground, output_pair, graph, config)
            for frequency in frequencies
        ])
        run_rows = []
        for index, frequency in enumerate(frequencies):
            aligned, _, _ = phase_align(physical[index], target[index])
            error = aligned - target[index]
            total_energy = float(np.sum(np.abs(error) ** 2))
            row: dict[str, Any] = {
                "run": run_name,
                "frequency_ghz": float(frequency),
                "residual_frobenius_norm": float(np.linalg.norm(error)),
                "supported_residual_energy_fraction": float(np.sum(np.abs(error[supported]) ** 2) / total_energy),
                "unsupported_max_abs_residual": float(np.max(np.abs(error[~supported]))),
            }
            for category, mask in masks.items():
                row[f"{category}_energy_fraction"] = float(np.sum(np.abs(error[mask]) ** 2) / total_energy)
                row[f"{category}_max_abs_residual"] = float(np.max(np.abs(error[mask])))
            rows.append(row)
            run_rows.append(row)
        analysis = read_json(case / "analysis.json")
        run_summaries[run_name] = {
            "analysis": analysis,
            "minimum_supported_residual_energy_fraction": min(item["supported_residual_energy_fraction"] for item in run_rows),
            "maximum_unsupported_abs_residual": max(item["unsupported_max_abs_residual"] for item in run_rows),
        }
    with (OUT / "residual_support_by_frequency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first = run_summaries["run01_nonuniform_width"]["analysis"]
    second = run_summaries["run02_uniform_50ohm_width"]["analysis"]
    summary = {
        "protocol": "v1.20-sparse-graph-physical-residual-audit",
        "graph_pairs": [list(pair) for pair in graph],
        "runs": run_summaries,
        "paired_change_run02_minus_run01": {
            key: float(second[key] - first[key])
            for key in ("active_rl_min_db", "total_rl_min_db", "network_efficiency_min", "physical_vs_target_s8_max_abs_delta", "corrected_vs_target_max_abs_delta_s")
        },
        "decision": {
            "same_sparse_graph_contains_residual": all(item["minimum_supported_residual_energy_fraction"] >= 0.95 for item in run_summaries.values()),
            "width_reference_plane_remap_successful": False,
            "allow_more_decoupling_stages": False,
            "allow_integrated_2x2": False,
            "allow_4x4_or_16x16": False,
            "allow_training_labels": False,
            "next_action": "parameterize and jointly optimize the radiator feed-point/launch and the existing single-stage POST transition against physical S8 sensitivities; do not add another decoupling stage",
        },
    }
    write_json(OUT / "residual_support_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
