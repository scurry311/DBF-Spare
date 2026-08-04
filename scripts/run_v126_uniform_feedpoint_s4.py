#!/usr/bin/env python3
"""Run one preregistered uniform-feed physical 2x2 S4 smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import active_metrics, parse_touchstone, solve_text as v114_solve_text
from run_v121_parametric_feed_post import aedt_processes
from run_v125_feedpoint_input_impedance import (
    EPS,
    analyze_case,
    patterned_builder_text,
    read_json,
    resolve,
    run_manifest,
    sha256_file,
    trusted_inputs,
    verify_inputs,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v126_uniform_2p10_feedpoint_s4_preregistered.json"


def prepare_case(out: Path, candidate: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    case_id = str(candidate["candidate_id"])
    folder = out / "2x2" / case_id
    folder.mkdir(parents=True)
    project = folder / f"v126_2x2_{case_id}.aedt"
    touchstone = folder / f"v126_2x2_{case_id}.s4p"
    efficiency = folder / "radiation_efficiency.csv"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    builder.write_text(patterned_builder_text(project, 2, candidate, protocol), encoding="ascii")
    solver.write_text(v114_solve_text(project, touchstone, efficiency), encoding="ascii")
    manifest = {
        "case_id": case_id,
        "side": 2,
        "port_count": 4,
        "candidate": candidate,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "efficiency_csv_path": str(efficiency.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "evidence_scope": "one physical 2x2 HFSS S4 with uniform 2.10 mm feed inset; no bridge or matching network",
    }
    write_json(folder / "case_manifest.json", manifest)
    return manifest


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.26 output: {out}")
    out.mkdir(parents=True)
    inputs = verify_inputs(config)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    tag_commit = subprocess.run(
        ["git", "rev-list", "-n", "1", config["parent_tag"]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != config["parent_commit"] or tag_commit != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag_commit}")
    protocol, geometry = trusted_inputs(config)
    candidate = {**geometry, **config["candidate"]}
    case = prepare_case(out, candidate, protocol)
    write_json(out / "two_by_two_manifest.json", {"cases": [case]})
    write_csv(out / "frozen_input_manifest.csv", inputs)
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {
                "head_commit": head,
                "tag_commit": tag_commit,
                "aedt_processes": aedt_processes(),
            },
            "evidence_rules": {
                "single_candidate_only": True,
                "same_geometry_all_frequencies": True,
                "same_285_frozen_stimuli": True,
                "no_post_result_retuning": True,
                "no_bridge_component_tuning": True,
            },
        },
    )
    decision = {
        "stage": "A_uniform_2p10_s4_preregistered",
        "allow_run": True,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "case": case, "decision": decision}


def selected_s4(path: Path, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, matrices = parse_touchstone(path, 4)
    indices = [int(np.argmin(np.abs(frequencies - value))) for value in targets]
    return frequencies[indices], matrices[indices]


def matrix_rows(label: str, frequencies: np.ndarray, matrices: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frequency, matrix in zip(frequencies, matrices):
        diagonal = np.diag(matrix)
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        eigenvalues = np.linalg.eigvals(matrix)
        row: dict[str, Any] = {
            "source": label,
            "frequency_ghz": float(frequency),
            "passive_rl_min_db": float(np.min(-20.0 * np.log10(np.maximum(np.abs(diagonal), EPS)))),
            "maximum_off_diagonal_magnitude": float(np.max(np.abs(off_diagonal))),
            "maximum_off_diagonal_db": float(20.0 * np.log10(max(float(np.max(np.abs(off_diagonal))), EPS))),
            "maximum_modal_reflection_magnitude": float(np.max(np.abs(eigenvalues))),
            "minimum_modal_rl_db": float(-20.0 * np.log10(max(float(np.max(np.abs(eigenvalues))), EPS))),
        }
        for index, value in enumerate(diagonal, start=1):
            row[f"s{index}{index}_magnitude"] = float(abs(value))
            row[f"s{index}{index}_rl_db"] = float(-20.0 * np.log10(max(float(abs(value)), EPS)))
        rows.append(row)
    return rows


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    case = read_json(out / "two_by_two_manifest.json")["cases"][0]
    physical = analyze_case(config, case)
    targets = np.asarray(config["frequencies_ghz"], dtype=float)
    physical_f, physical_s = selected_s4(Path(case["touchstone_path"]), targets)
    trusted_f, trusted_s = selected_s4(resolve(config["inputs"]["trusted_s4"]), targets)
    trusted_replay, trusted_active, trusted_total = active_metrics(
        resolve(config["inputs"]["stimulus_root"]), 2, trusted_f, trusted_s
    )
    matrix_audit = matrix_rows("uniform_feed_2p10", physical_f, physical_s)
    matrix_audit.extend(matrix_rows("trusted_feed_2p30", trusted_f, trusted_s))
    write_csv(out / "three_frequency_s4_modal_audit.csv", matrix_audit)
    write_csv(out / "trusted_control_active_rl_replay.csv", trusted_replay)
    control = {
        "case_id": "trusted_v1143_feed_2p30_control",
        "passive_rl_min_db": float(
            min(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matrix)), EPS))) for matrix in trusted_s)
        ),
        "representative_active_rl_min_db": float(trusted_active),
        "representative_total_rl_min_db": float(trusted_total),
        "representative_source_count": len(trusted_replay),
        "evidence_scope": "trusted control; not rerun in v1.26",
    }
    comparison = {
        "physical_candidate": physical,
        "trusted_control": control,
        "physical_minus_control": {
            "passive_rl_db": float(physical["passive_rl_min_db"]) - control["passive_rl_min_db"],
            "active_rl_db": float(physical["representative_active_rl_min_db"])
            - control["representative_active_rl_min_db"],
            "total_rl_db": float(physical["representative_total_rl_min_db"])
            - control["representative_total_rl_min_db"],
            "max_abs_s4_change": float(np.max(np.abs(physical_s - trusted_s))),
        },
    }
    write_json(out / "paired_control_comparison.json", comparison)
    passed = bool(physical["gate_pass"])
    summary = {
        "candidate_count": 1,
        "gate_pass_count": int(passed),
        "candidate": physical,
        "trusted_control": control,
        "feedpoint_only_active_rl_feasible": passed,
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "B_uniform_2p10_s4_gate_complete",
        "feedpoint_only_active_rl_feasible": passed,
        "allow_independent_repeat": passed,
        "transition_to_radiator_input_geometry": not passed,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "The uniform feed-point candidate passed the physical 2x2 three-frequency active-RL gate."
            if passed
            else "Feed-point relocation alone failed the physical active-RL gate; modify the radiator/input impedance and do not resume bridge tuning."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "comparison": comparison, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest = read_json(out / "two_by_two_manifest.json") if (out / "two_by_two_manifest.json").exists() else {"cases": []}
    return {
        "output_directory": str(out),
        "case_count": len(manifest["cases"]),
        "touchstone_count": sum(Path(case["touchstone_path"]).exists() for case in manifest["cases"]),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("preregister", "run", "analyze", "status"), default="status")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    actions = {
        "preregister": preregister,
        "run": lambda item: run_manifest(item, "two_by_two_manifest.json", "allow_run"),
        "analyze": analyze,
        "status": status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
