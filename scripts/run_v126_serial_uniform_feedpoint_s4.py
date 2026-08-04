#!/usr/bin/env python3
"""Run the v1.26 uniform-feed 2x2 S4 at three frequencies serially."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import (
    active_metrics,
    efficiency_from_csv,
    memory_available_gb,
    parse_touchstone,
    profile_metrics,
)
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v125_feedpoint_input_impedance import (
    EPS,
    patterned_builder_text,
    read_json,
    resolve,
    topology_warning_count,
    trusted_inputs,
    verify_inputs,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v126_uniform_2p10_feedpoint_s4_serial_run02_preregistered.json"


def token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def single_frequency_builder(
    project: Path,
    frequency_ghz: float,
    candidate: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    text = patterned_builder_text(project, 2, candidate, protocol)
    text = text.replace('"Frequency:=", "10GHz"', f'"Frequency:=", "{frequency_ghz:g}GHz"')
    text = re.sub(r'^oAnalysis\.InsertFrequencySweep .*$', '', text, flags=re.MULTILINE)
    return text


def single_frequency_solver(project: Path, touchstone: Path, efficiency: Path, frequency_ghz: float) -> str:
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, reportName
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("SmallCell_DualSlot_10GHz")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
Set oReport = oDesign.GetModule("ReportSetup")
reportName = "V126_RadiationEfficiency"
On Error Resume Next
oReport.DeleteReports Array(reportName)
Err.Clear
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V114"), Array("Freq:=", Array("{frequency_ghz:g}GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(efficiency)}"
On Error GoTo 0
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def prepare_case(
    out: Path,
    frequency_ghz: float,
    candidate: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    name = f"f_{token(frequency_ghz)}ghz"
    folder = out / "2x2_serial" / name
    folder.mkdir(parents=True)
    project = folder / f"v126_2x2_uniform_2p10_{name}.aedt"
    touchstone = folder / f"v126_2x2_uniform_2p10_{name}.s4p"
    efficiency = folder / "radiation_efficiency.csv"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    builder.write_text(single_frequency_builder(project, frequency_ghz, candidate, protocol), encoding="ascii")
    solver.write_text(single_frequency_solver(project, touchstone, efficiency, frequency_ghz), encoding="ascii")
    case = {
        "case_id": name,
        "frequency_ghz": frequency_ghz,
        "side": 2,
        "port_count": 4,
        "candidate": candidate,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "efficiency_csv_path": str(efficiency.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "solution": "Setup_10GHz:LastAdaptive",
    }
    write_json(folder / "case_manifest.json", case)
    return case


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.26 run02: {out}")
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
    cases = [prepare_case(out, float(value), candidate, protocol) for value in config["frequencies_ghz"]]
    write_json(out / "serial_case_manifest.json", {"cases": cases})
    write_csv(out / "frozen_input_manifest.csv", inputs)
    write_json(
        out / "preregistration.json",
        {
            **config,
            "runtime_audit": {
                "head_commit": head,
                "tag_commit": tag_commit,
                "free_memory_gib": memory_available_gb(),
                "aedt_processes": aedt_processes(),
            },
            "run01_resource_failure": {
                "path": "hfss_outputs/v126_uniform_2p10_feedpoint_s4_20260805_run01",
                "minimum_free_memory_gib": 0.8420295715332031,
                "touchstone_generated": False,
                "physics_conclusion_allowed": False,
            },
            "evidence_rules": {
                "geometry_identical_across_frequencies": True,
                "single_frequency_cases_strictly_serial": True,
                "same_285_frozen_stimuli_used_after_s4_assembly": True,
                "no_post_result_retuning": True,
            },
        },
    )
    decision = {
        "stage": "A_serial_uniform_2p10_s4_preregistered",
        "allow_run": True,
        "allow_4x4": False,
        "allow_16x16": False,
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
    required = float(config["resources"]["minimum_free_memory_before_2x2_gib"])
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
        raise RuntimeError("Serial solve is not authorized")
    cases = read_json(out / "serial_case_manifest.json")["cases"]
    progress = out / "run_serial_manifest.csv"
    rows = list(csv.DictReader(progress.open(encoding="utf-8"))) if progress.exists() else []
    complete = {row["case_id"] for row in rows if row.get("touchstone_exists", "").lower() == "true"}
    executable = str(resolve(config["ansys_executable"]))
    poll = float(config["resources"]["poll_interval_seconds"])
    abort = float(config["resources"]["abort_free_memory_during_solve_gib"])
    for case in cases:
        if case["case_id"] in complete:
            continue
        free = wait_for_memory(config)
        folder = Path(case["project_path"]).parent
        build = subprocess.run(
            [executable, "-RunScriptAndExit", case["builder_path"]],
            cwd=ROOT,
            stdout=(folder / "build.log").open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            check=False,
        )
        if build.returncode != 0 or topology_warning_count(folder) > 0 or not Path(case["project_path"]).exists():
            raise RuntimeError(f"Build gate failed for {case['case_id']}")
        code, aborted, minimum_free = run_process_with_memory_guard(
            [executable, "-ng", "-RunScriptAndExit", case["solver_path"]],
            folder / "solve_export.log",
            abort,
            poll,
        )
        touchstone = Path(case["touchstone_path"])
        row = {
            "case_id": case["case_id"],
            "frequency_ghz": case["frequency_ghz"],
            "build_return_code": int(build.returncode),
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
            raise RuntimeError(f"Single-frequency solve failed for {case['case_id']}: {row}")
    return {"completed_count": len(rows), "rows": rows}


def frequency_audit(source: str, frequency: float, matrix: np.ndarray) -> dict[str, Any]:
    diagonal = np.diag(matrix)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    eigenvalues = np.linalg.eigvals(matrix)
    row: dict[str, Any] = {
        "source": source,
        "frequency_ghz": frequency,
        "passive_rl_min_db": float(np.min(-20.0 * np.log10(np.maximum(np.abs(diagonal), EPS)))),
        "maximum_off_diagonal_magnitude": float(np.max(np.abs(off_diagonal))),
        "maximum_off_diagonal_db": float(20.0 * np.log10(max(float(np.max(np.abs(off_diagonal))), EPS))),
        "minimum_modal_rl_db": float(-20.0 * np.log10(max(float(np.max(np.abs(eigenvalues))), EPS))),
    }
    for index, value in enumerate(diagonal, start=1):
        row[f"s{index}{index}_rl_db"] = float(-20.0 * np.log10(max(float(abs(value)), EPS)))
    return row


def grouped_active_rows(
    replay: list[dict[str, Any]], active_gate_db: float, total_gate_db: float
) -> list[dict[str, Any]]:
    groups: list[tuple[str, tuple[str, ...]]] = [
        ("overall", ()),
        ("frequency", ("frequency_ghz",)),
        ("k", ("k_value",)),
        ("k_frequency", ("k_value", "frequency_ghz")),
    ]
    rows: list[dict[str, Any]] = []
    for scope, fields in groups:
        buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for item in replay:
            key = tuple(str(item[field]) for field in fields)
            buckets.setdefault(key, []).append(item)
        for key, items in sorted(buckets.items()):
            active = np.asarray([float(item["active_rl_db"]) for item in items], dtype=float)
            total = np.asarray([float(item["total_rl_db"]) for item in items], dtype=float)
            active_pass = active >= active_gate_db
            total_pass = total >= total_gate_db
            row: dict[str, Any] = {
                "scope": scope,
                "count": len(items),
                "active_rl_min_db": float(np.min(active)),
                "active_rl_q05_db": float(np.quantile(active, 0.05)),
                "active_rl_median_db": float(np.median(active)),
                "total_rl_min_db": float(np.min(total)),
                "active_gate_pass_count": int(np.sum(active_pass)),
                "active_gate_pass_rate": float(np.mean(active_pass)),
                "total_gate_pass_count": int(np.sum(total_pass)),
                "total_gate_pass_rate": float(np.mean(total_pass)),
                "joint_gate_pass_count": int(np.sum(active_pass & total_pass)),
                "joint_gate_pass_rate": float(np.mean(active_pass & total_pass)),
            }
            row.update(dict(zip(fields, key)))
            rows.append(row)
    return rows


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "serial_case_manifest.json")["cases"]
    matrices = []
    frequencies = []
    numerical_rows = []
    efficiencies = []
    for case in cases:
        path = Path(case["touchstone_path"])
        parsed_f, parsed_s = parse_touchstone(path, 4)
        target = float(case["frequency_ghz"])
        index = int(np.argmin(np.abs(parsed_f - target)))
        frequencies.append(float(parsed_f[index]))
        matrices.append(parsed_s[index])
        folder = path.parent
        profile = profile_metrics(folder)
        efficiency = efficiency_from_csv(Path(case["efficiency_csv_path"]))
        efficiencies.append(float(efficiency) if efficiency is not None else math.nan)
        numerical_rows.append(
            {
                "case_id": case["case_id"],
                "target_frequency_ghz": target,
                "actual_frequency_ghz": float(parsed_f[index]),
                **profile,
                "radiation_efficiency": efficiency,
                "topology_warning_count": topology_warning_count(folder),
            }
        )
    frequency_array = np.asarray(frequencies, dtype=float)
    matrix_array = np.asarray(matrices, dtype=complex)
    replay, active_rl, total_rl = active_metrics(
        resolve(config["inputs"]["stimulus_root"]), 2, frequency_array, matrix_array
    )
    write_csv(out / "representative_active_rl_replay.csv", replay)
    active_gate = float(config["gates"]["minimum_2x2_active_rl_db"])
    total_gate = float(config["gates"]["minimum_2x2_total_rl_db"])
    write_csv(out / "active_rl_grouped_stats.csv", grouped_active_rows(replay, active_gate, total_gate))
    worst_active = sorted(replay, key=lambda item: float(item["active_rl_db"]))[:20]
    write_csv(out / "worst_active_rl_stimuli.csv", worst_active)
    trusted_f, trusted_s = parse_touchstone(resolve(config["inputs"]["trusted_s4"]), 4)
    targets = np.asarray(config["frequencies_ghz"], dtype=float)
    trusted_indices = [int(np.argmin(np.abs(trusted_f - target))) for target in targets]
    trusted_f = trusted_f[trusted_indices]
    trusted_s = trusted_s[trusted_indices]
    trusted_replay, trusted_active, trusted_total = active_metrics(
        resolve(config["inputs"]["stimulus_root"]), 2, trusted_f, trusted_s
    )
    write_csv(out / "trusted_control_active_rl_replay.csv", trusted_replay)
    audit_rows = [frequency_audit("uniform_feed_2p10", f, s) for f, s in zip(frequency_array, matrix_array)]
    audit_rows.extend(frequency_audit("trusted_feed_2p30", f, s) for f, s in zip(trusted_f, trusted_s))
    write_csv(out / "three_frequency_s4_modal_audit.csv", audit_rows)
    write_csv(out / "serial_numerical_metrics.csv", numerical_rows)
    passive_rl = float(
        min(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matrix)), EPS))) for matrix in matrix_array)
    )
    reciprocity = float(np.max(np.abs(matrix_array - np.transpose(matrix_array, (0, 2, 1)))))
    passivity = float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in matrix_array))
    final_delta = max(float(row.get("final_delta_s") or math.inf) for row in numerical_rows)
    minimum_efficiency = min(efficiencies)
    topology_warnings = sum(int(row["topology_warning_count"]) for row in numerical_rows)
    frequency_error = float(np.max(np.abs(frequency_array - targets)))
    gates = config["gates"]
    passed = bool(
        all(row.get("converged") is True for row in numerical_rows)
        and final_delta <= float(gates["maximum_final_delta_s"])
        and reciprocity <= float(gates["maximum_reciprocity_error"])
        and passivity <= float(gates["maximum_passivity_sigma"])
        and passive_rl >= float(gates["minimum_2x2_passive_rl_db"])
        and active_rl >= active_gate
        and total_rl >= total_gate
        and minimum_efficiency >= float(gates["minimum_2x2_radiation_efficiency"])
        and topology_warnings <= int(gates["maximum_port_topology_warning_count"])
    )
    trusted_passive = float(
        min(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matrix)), EPS))) for matrix in trusted_s)
    )
    summary = {
        "execution_mode": "three_independent_single_frequency_adaptive_solves",
        "frequency_max_error_ghz": frequency_error,
        "maximum_final_delta_s": final_delta,
        "reciprocity_error": reciprocity,
        "passivity_sigma": passivity,
        "minimum_radiation_efficiency": minimum_efficiency,
        "passive_rl_min_db": passive_rl,
        "representative_active_rl_min_db": float(active_rl),
        "representative_total_rl_min_db": float(total_rl),
        "representative_source_count": len(replay),
        "gate_pass": passed,
        "trusted_control": {
            "passive_rl_min_db": trusted_passive,
            "representative_active_rl_min_db": float(trusted_active),
            "representative_total_rl_min_db": float(trusted_total),
        },
        "physical_minus_control": {
            "passive_rl_db": passive_rl - trusted_passive,
            "active_rl_db": float(active_rl - trusted_active),
            "total_rl_db": float(total_rl - trusted_total),
            "max_abs_s4_change": float(np.max(np.abs(matrix_array - trusted_s))),
        },
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "B_serial_uniform_2p10_s4_gate_complete",
        "feedpoint_only_active_rl_feasible": passed,
        "allow_independent_repeat": passed,
        "transition_to_radiator_input_geometry": not passed,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "Uniform 2.10 mm feed relocation passed the physical three-frequency active-RL gate."
            if passed
            else "Feed relocation alone failed the physical active-RL gate; change radiator/input geometry and do not tune the bridge again."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "serial_case_manifest.json")["cases"] if (out / "serial_case_manifest.json").exists() else []
    return {
        "output_directory": str(out),
        "completed_touchstones": sum(Path(case["touchstone_path"]).exists() for case in cases),
        "case_count": len(cases),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("preregister", "run", "analyze", "status"), default="status")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    actions = {"preregister": preregister, "run": run, "analyze": analyze, "status": status}
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
