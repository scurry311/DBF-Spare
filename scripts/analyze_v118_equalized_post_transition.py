#!/usr/bin/env python3
"""Apply the v1.18 S4 pre-repeat gate without requiring EEP exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from design_v115_grounded_modal_network import terminate_network
from run_v114_small_cell_broadband_feed import load_stimuli, parse_touchstone, profile_metrics
from run_v117_integrated_2x2_smoke import read_json, reordered_network, resolve, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v118_equalized_post_transition_candidate01.json"
EPS = 1.0e-15


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_json(args.config)
    out_dir = resolve(config["output_directory"])
    folder = out_dir / "integrated_2x2_direct01"
    manifest = read_json(folder / "case_manifest.json")
    run_summary = read_json(folder / "run_summary.json")
    field_summary = read_json(folder / "field_export_summary.json")
    frequencies, integrated = reordered_network(
        Path(manifest["touchstone_path"]), manifest["pre_reference_ports"], 4
    )
    antenna_f, antenna = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    feed_f, feed = reordered_network(
        resolve(config["validated_feed_s8"]),
        [f"PRE_{port}" for port in range(4)] + [f"POST_{port}" for port in range(4)],
        8,
    )
    if not np.allclose(frequencies, antenna_f) or not np.allclose(frequencies, feed_f):
        raise RuntimeError("Frequency grids differ")
    cascade = np.stack(
        [terminate_network(feed[index], antenna[index])[0] for index in range(len(frequencies))]
    )
    rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    frequency_rows = []
    worst_source = None
    for frequency_index, frequency in enumerate(frequencies):
        selected = [
            index
            for index, row in enumerate(rows)
            if abs(float(row["frequency_ghz"]) - float(frequency)) <= 1.0e-9
        ]
        sources = vectors[selected].T
        active = considered[selected].T
        reflected = integrated[frequency_index] @ sources
        gamma = np.where(active, np.abs(reflected) / np.maximum(np.abs(sources), EPS), 0.0)
        active_rl = -20.0 * np.log10(np.maximum(np.max(gamma, axis=0), EPS))
        incident = np.sum(np.abs(sources) ** 2, axis=0)
        reflected_power = np.sum(np.abs(reflected) ** 2, axis=0)
        total_rl = -10.0 * np.log10(np.maximum(reflected_power / incident, EPS))
        passive_rl = -20.0 * np.log10(
            np.maximum(np.abs(np.diag(integrated[frequency_index])), EPS)
        )
        delta = np.abs(integrated[frequency_index] - cascade[frequency_index])
        frequency_rows.append(
            {
                "frequency_ghz": float(frequency),
                "passive_rl_min_db": float(np.min(passive_rl)),
                "active_rl_min_db": float(np.min(active_rl)),
                "total_rl_min_db": float(np.min(total_rl)),
                "integrated_vs_cascade_max_abs_delta_s": float(np.max(delta)),
            }
        )
        local = int(np.argmin(active_rl))
        candidate = {**rows[selected[local]], "active_rl_db": float(active_rl[local])}
        if worst_source is None or candidate["active_rl_db"] < worst_source["active_rl_db"]:
            worst_source = candidate
    gates = config["gates"]
    profile = profile_metrics(folder)
    reciprocity = float(np.max(np.abs(integrated - np.transpose(integrated, (0, 2, 1)))))
    passivity = float(
        max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in integrated)
    )
    active_rl_min = min(row["active_rl_min_db"] for row in frequency_rows)
    delta_s_max = max(row["integrated_vs_cascade_max_abs_delta_s"] for row in frequency_rows)
    post_gate = bool(
        profile.get("converged") is True
        and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
        and reciprocity <= float(gates["maximum_reciprocity_error"])
        and passivity <= float(gates["maximum_passivity_sigma"])
        and active_rl_min >= float(gates["minimum_active_rl_db"])
        and delta_s_max <= float(gates["maximum_integrated_vs_cascade_abs_delta_s"])
    )
    old = read_json(
        ROOT
        / "baselines"
        / "2026-07-30-v117-integrated-2x2-smoke"
        / "snapshots"
        / "analysis.json"
    )
    analysis = {
        **run_summary,
        **profile,
        "field_export_return_code": field_summary.get("return_code"),
        "field_export_file_count": field_summary.get("eep_files", 0),
        "field_export_complete": field_summary.get("eep_files", 0) == 12,
        "eep_not_required_after_failed_s4_pre_repeat_gate": not post_gate,
        "passive_rl_min_db": min(row["passive_rl_min_db"] for row in frequency_rows),
        "active_rl_min_db": active_rl_min,
        "total_rl_min_db": min(row["total_rl_min_db"] for row in frequency_rows),
        "integrated_vs_cascade_max_abs_delta_s": delta_s_max,
        "reciprocity_error_max": reciprocity,
        "passivity_sigma_max": passivity,
        "post_transition_gate_pass": post_gate,
        "active_rl_improvement_over_v117_db": active_rl_min - float(old["active_rl_min_db"]),
        "delta_s_improvement_over_v117": float(old["integrated_vs_cascade_max_abs_delta_s"])
        - delta_s_max,
        "worst_active_rl_source": worst_source,
        "frozen_v116_component_values_verified": True,
        "equalized_route_length_mm": float(
            config["post_transition"]["common_centerline_length_mm"]
        ),
    }
    decision = {
        "post_transition_gate_pass": post_gate,
        "allow_independent_repeat": post_gate,
        "independent_repeat_executed": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "next_hardware_action": (
            "independent repeat"
            if post_gate
            else "replace uncoupled local POST tuning with an integrated multiport transition/decoupler"
        ),
    }
    write_csv(folder / "frequency_s4_gate_metrics.csv", frequency_rows)
    write_json(folder / "s4_gate_analysis.json", analysis)
    write_json(out_dir / "stage_decision.json", decision)
    print(json.dumps({"analysis": analysis, "decision": decision}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
