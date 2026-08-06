#!/usr/bin/env python3
"""Recover the input match by varying only the dual-resonant load behind a frozen CPS."""

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
from run_v130_fixed_reference_cps_transformer import prepare_case, read_json, resolve, run, status


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v131_dual_resonant_recovery_preregistered.json"
EPS = 1.0e-15


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.31 output: {out}")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    transformer_summary = read_json(config["inputs"]["fixed_cps_summary"])
    if transformer_summary.get("best_passive_rl_db", 0.0) <= 0.0:
        raise RuntimeError("Missing completed v1.30 physical CPS evidence")
    direct_metrics = resolve(config["inputs"]["direct_gap_metrics"])
    if not direct_metrics.exists():
        raise RuntimeError("Missing v1.29 direct-gap physical impedance evidence")
    base = config["frozen_geometry"]
    cases = [prepare_case(out, {**base, **candidate}, float(config["frequency_ghz"])) for candidate in config["candidates"]]
    write_json(out / "case_manifest.json", {"cases": cases})
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {"head_commit": head, "tag_commit": tag, "free_memory_gib": memory_available_gb(), "aedt_processes": aedt_processes()},
            "evidence_rules": {
                "cps_width_length_port_and_total_path_frozen": True,
                "only_dual_resonant_load_variables_change": True,
                "all_seven_candidates_fixed_before_solve": True,
                "engineering_thresholds_unchanged": True,
                "array_and_learning_stages_locked": True
            },
        },
    )
    decision = {
        "stage": "A_dual_resonant_recovery_preregistered",
        "allow_run": True,
        "allow_three_frequency": False,
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
        index = int(np.argmin(np.abs(frequencies - float(case["frequency_ghz"]))))
        s11 = matrices[index, 0, 0]
        impedance = 50.0 * (1.0 + s11) / (1.0 - s11)
        passive_rl = float(-20.0 * np.log10(max(float(abs(s11)), EPS)))
        profile = profile_metrics(folder)
        passed = bool(
            profile.get("converged") is True
            and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
            and topology_warning_count(folder) <= int(gates["maximum_port_topology_warning_count"])
            and passive_rl >= float(gates["minimum_screen_passive_rl_db"])
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "primary_arm_length_mm": case["geometry"]["primary_arm_length_mm"],
                "secondary_arm_length_mm": case["geometry"]["secondary_arm_length_mm"],
                "secondary_arm_offset_y_mm": case["geometry"]["secondary_arm_offset_y_mm"],
                "frozen_transformer_width_mm": case["geometry"]["transformer_width_mm"],
                "frozen_transformer_length_mm": case["geometry"]["transformer_length_mm"],
                **profile,
                "frequency_ghz": float(frequencies[index]),
                "s11_real": float(s11.real),
                "s11_imag": float(s11.imag),
                "s11_magnitude": float(abs(s11)),
                "s11_phase_deg": float(np.angle(s11, deg=True)),
                "input_resistance_ohm": float(impedance.real),
                "input_reactance_ohm": float(impedance.imag),
                "passive_rl_db": passive_rl,
                "topology_warning_count": topology_warning_count(folder),
                "screen_gate_pass": passed,
            }
        )
    rows.sort(key=lambda item: (not item["screen_gate_pass"], -float(item["passive_rl_db"])))
    write_csv(out / "candidate_metrics.csv", rows)
    passing = [row for row in rows if row["screen_gate_pass"]]
    preferred = [row for row in rows if row["passive_rl_db"] >= float(gates["preferred_passive_rl_db"])]
    summary = {
        "candidate_count": len(rows),
        "screen_gate_pass_count": len(passing),
        "preferred_15db_count": len(preferred),
        "selected_candidate": rows[0]["case_id"] if passing else None,
        "best_passive_rl_db": rows[0]["passive_rl_db"],
        "allow_three_frequency": bool(passing),
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "B_dual_resonant_recovery_complete",
        "allow_run": True,
        "allow_three_frequency": bool(passing),
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "The dual-resonant load recovered 10 dB matching behind the independently frozen CPS; only three-frequency 1x1 verification is authorized."
            if passing
            else "The dual-resonant load recovery failed behind the frozen CPS; array, EEP, labels, and critic remain locked."
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
