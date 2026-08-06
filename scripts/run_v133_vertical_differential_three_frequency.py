#!/usr/bin/env python3
"""Verify one frozen vertical differential 1x1 candidate at three frequencies."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import memory_available_gb, parse_touchstone, profile_metrics
from run_v121_parametric_feed_post import aedt_processes
from run_v125_feedpoint_input_impedance import topology_warning_count, write_csv, write_json
from run_v130_fixed_reference_cps_transformer import read_json, resolve, run, status
from run_v132_vertical_differential_launch import prepare_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v133_vertical_differential_three_frequency_preregistered.json"
EPS = 1.0e-15


def frequency_id(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.33 output: {out}")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    source = read_json(config["inputs"]["v132_summary"])
    if not source.get("allow_three_frequency") or source.get("selected_candidate") != config["inputs"]["required_selected_candidate"]:
        raise RuntimeError("v1.32 did not authorize the frozen three-frequency candidate")
    base = config["frozen_candidate"]
    cases = []
    for frequency in config["frequencies_ghz"]:
        value = float(frequency)
        geometry = {**base, "candidate_id": f"via_v0_f{frequency_id(value)}"}
        cases.append(prepare_case(out, geometry, value))
    write_json(out / "case_manifest.json", {"cases": cases})
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {"head_commit": head, "tag_commit": tag, "free_memory_gib": memory_available_gb(), "aedt_processes": aedt_processes()},
            "evidence_rules": {
                "one_geometry_for_all_frequencies": True,
                "independent_project_per_frequency": True,
                "thresholds_unchanged": True,
                "two_by_two_and_learning_locked": True
            },
        },
    )
    decision = {
        "stage": "A_three_frequency_preregistered",
        "allow_run": True,
        "allow_efficiency_audit": False,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "case_count": len(cases), "decision": decision}


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "case_manifest.json")["cases"]
    gates = config["gates"]
    rows: list[dict[str, Any]] = []
    for case in cases:
        folder = Path(case["touchstone_path"]).parent
        frequencies, matrices = parse_touchstone(Path(case["touchstone_path"]), 1)
        target = float(case["frequency_ghz"])
        index = int(np.argmin(np.abs(frequencies - target)))
        s11 = matrices[index, 0, 0]
        impedance = 50.0 * (1.0 + s11) / (1.0 - s11)
        passive_rl = float(-20.0 * np.log10(max(float(abs(s11)), EPS)))
        profile = profile_metrics(folder)
        passed = bool(
            profile.get("converged") is True
            and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
            and topology_warning_count(folder) <= int(gates["maximum_port_topology_warning_count"])
            and passive_rl >= float(gates["minimum_three_frequency_passive_rl_db"])
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "frequency_ghz": float(frequencies[index]),
                **profile,
                "s11_real": float(s11.real),
                "s11_imag": float(s11.imag),
                "s11_magnitude": float(abs(s11)),
                "input_resistance_ohm": float(impedance.real),
                "input_reactance_ohm": float(impedance.imag),
                "passive_rl_db": passive_rl,
                "topology_warning_count": topology_warning_count(folder),
                "frequency_gate_pass": passed,
            }
        )
    rows.sort(key=lambda item: float(item["frequency_ghz"]))
    write_csv(out / "three_frequency_metrics.csv", rows)
    gate = bool(len(rows) == 3 and all(row["frequency_gate_pass"] for row in rows))
    minimum_rl = min(float(row["passive_rl_db"]) for row in rows)
    summary = {
        "frequency_count": len(rows),
        "frequency_gate_pass_count": sum(bool(row["frequency_gate_pass"]) for row in rows),
        "minimum_passive_rl_db": minimum_rl,
        "preferred_15db_reserve": minimum_rl >= float(gates["preferred_three_frequency_passive_rl_db"]),
        "three_frequency_gate_pass": gate,
        "allow_efficiency_audit": gate,
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "B_three_frequency_complete",
        "allow_run": True,
        "allow_efficiency_audit": gate,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "The frozen vertical differential candidate passed the three-frequency 10 dB gate; radiation efficiency and independent repeat are next."
            if gate
            else "The frozen vertical differential candidate failed the three-frequency gate; all downstream work remains locked."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": rows, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=str)
    parser.add_argument("--mode", choices=("preregister", "run", "analyze", "status"), default="status")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.output_directory:
        config["output_directory"] = args.output_directory
    actions = {"preregister": preregister, "run": run, "analyze": analyze, "status": status}
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
