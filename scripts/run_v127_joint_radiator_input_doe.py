#!/usr/bin/env python3
"""Run the preregistered v1.27 joint radiator/input 1x1 DOE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import memory_available_gb, parse_touchstone, profile_metrics
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v125_feedpoint_input_impedance import topology_warning_count, write_csv, write_json
from run_v127_aperture_coupled_radiator import (
    ROOT,
    builder_text,
    read_json,
    resolve,
    sha256_file,
    solver_text,
)


DEFAULT_CONFIG = ROOT / "configs" / "v127_joint_radiator_input_doe_preregistered.json"
EPS = 1.0e-15


def prepare_case(out: Path, geometry: dict[str, Any], frequency_ghz: float) -> dict[str, Any]:
    case_id = str(geometry["candidate_id"])
    folder = out / "1x1_doe" / case_id
    folder.mkdir(parents=True)
    project = folder / f"v127_doe_{case_id}.aedt"
    touchstone = folder / f"v127_doe_{case_id}.s1p"
    efficiency = folder / "radiation_efficiency.csv"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    builder.write_text(builder_text(project, 1, geometry, frequency_ghz), encoding="ascii")
    solver.write_text(solver_text(project, touchstone, efficiency, frequency_ghz), encoding="ascii")
    case = {
        "case_id": case_id,
        "frequency_ghz": frequency_ghz,
        "geometry": geometry,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "efficiency_csv_path": str(efficiency.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
    }
    write_json(folder / "case_manifest.json", case)
    return case


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.27 DOE: {out}")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    source = resolve(config["inputs"]["numerically_valid_center_s1"])
    if sha256_file(source) != config["inputs"]["numerically_valid_center_s1_sha256"]:
        raise RuntimeError("Center S1 hash mismatch")
    base = config["frozen_geometry"]
    cases = [
        prepare_case(out, {**base, **candidate}, float(config["frequency_ghz"]))
        for candidate in config["candidates"]
    ]
    write_json(out / "case_manifest.json", {"cases": cases})
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {
                "head_commit": head,
                "tag_commit": tag,
                "free_memory_gib": memory_available_gb(),
                "aedt_processes": aedt_processes(),
            },
            "evidence_rules": {
                "all_candidates_fixed_before_solve": True,
                "engineering_thresholds_unchanged": True,
                "one_by_one_only": True,
                "two_by_two_locked": True,
            },
        },
    )
    decision = {
        "stage": "A_joint_1x1_doe_preregistered",
        "allow_run": True,
        "allow_three_frequency_1x1": False,
        "allow_2x2_jacobian": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "case_count": len(cases), "decision": decision}


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def wait_for_memory(config: dict[str, Any]) -> float:
    required = float(config["resources"]["minimum_free_memory_before_1x1_gib"])
    deadline = time.time() + float(config["resources"]["memory_recovery_wait_seconds"])
    while True:
        require_no_aedt()
        free = memory_available_gb()
        if free >= required:
            return free
        if time.time() >= deadline:
            raise MemoryError(f"Memory did not recover to {required:.2f} GiB; current {free:.2f} GiB")
        time.sleep(float(config["resources"]["poll_interval_seconds"]))


def run(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_run"):
        raise RuntimeError("DOE run is not authorized")
    cases = read_json(out / "case_manifest.json")["cases"]
    progress = out / "run_progress.csv"
    rows = list(csv.DictReader(progress.open(encoding="utf-8"))) if progress.exists() else []
    completed = {row["case_id"] for row in rows if row.get("touchstone_exists", "").lower() == "true"}
    executable = str(resolve(config["ansys_executable"]))
    poll = float(config["resources"]["poll_interval_seconds"])
    abort = float(config["resources"]["abort_free_memory_during_solve_gib"])
    for case in cases:
        if case["case_id"] in completed:
            continue
        free = wait_for_memory(config)
        folder = Path(case["project_path"]).parent
        with (folder / "build.log").open("w", encoding="utf-8") as handle:
            build = subprocess.run(
                [executable, "-RunScriptAndExit", case["builder_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if build.returncode != 0 or topology_warning_count(folder) > 0 or not Path(case["project_path"]).exists():
            raise RuntimeError(f"Build failed for {case['case_id']}")
        code, aborted, minimum_free = run_process_with_memory_guard(
            [executable, "-ng", "-RunScriptAndExit", case["solver_path"]],
            folder / "solve_export.log",
            abort,
            poll,
        )
        touchstone = Path(case["touchstone_path"])
        row = {
            "case_id": case["case_id"],
            "build_return_code": build.returncode,
            "solve_return_code": code,
            "memory_aborted": aborted,
            "free_memory_gib_before": free,
            "minimum_free_memory_gib": minimum_free,
            "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
            "topology_warning_count": topology_warning_count(folder),
        }
        rows.append(row)
        write_csv(progress, rows)
        if code != 0 or aborted or not row["touchstone_exists"] or row["topology_warning_count"] > 0:
            raise RuntimeError(f"Solve failed for {case['case_id']}: {row}")
    return {"completed_count": len(rows), "rows": rows}


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "case_manifest.json")["cases"]
    rows = []
    gates = config["gates"]
    for case in cases:
        folder = Path(case["touchstone_path"]).parent
        frequencies, matrices = parse_touchstone(Path(case["touchstone_path"]), 1)
        index = int(np.argmin(np.abs(frequencies - float(case["frequency_ghz"]))))
        s11 = matrices[index, 0, 0]
        impedance = 50.0 * (1.0 + s11) / (1.0 - s11)
        profile = profile_metrics(folder)
        passive_rl = float(-20.0 * np.log10(max(float(abs(s11)), EPS)))
        passed = bool(
            profile.get("converged") is True
            and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
            and passive_rl >= float(gates["minimum_screen_passive_rl_db"])
            and topology_warning_count(folder) <= int(gates["maximum_port_topology_warning_count"])
        )
        rows.append(
            {
                "case_id": case["case_id"],
                **case["geometry"],
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
    rows.sort(key=lambda item: float(item["passive_rl_db"]), reverse=True)
    write_csv(out / "candidate_metrics.csv", rows)
    passing = [row for row in rows if row["screen_gate_pass"]]
    summary = {
        "candidate_count": len(rows),
        "screen_gate_pass_count": len(passing),
        "best_candidate": rows[0]["case_id"],
        "best_passive_rl_db": rows[0]["passive_rl_db"],
        "three_frequency_verification_authorized": bool(passing),
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "B_joint_1x1_doe_complete",
        "allow_three_frequency_1x1": bool(passing),
        "allow_2x2_jacobian": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "At least one joint radiator/input candidate passed the 10 GHz screen; only three-frequency 1x1 verification is authorized."
            if passing
            else "No joint candidate reached 10 dB at 10 GHz; 2x2 and downstream work remain locked."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": rows, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "case_manifest.json")["cases"] if (out / "case_manifest.json").exists() else []
    return {
        "output_directory": str(out),
        "touchstone_count": sum(Path(case["touchstone_path"]).exists() for case in cases),
        "case_count": len(cases),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=str)
    parser.add_argument("--mode", choices=("preregister", "run", "analyze", "status"), default="status")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    if args.output_directory:
        config["output_directory"] = args.output_directory
    actions = {"preregister": preregister, "run": run, "analyze": analyze, "status": status}
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
