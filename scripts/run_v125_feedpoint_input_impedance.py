#!/usr/bin/env python3
"""Screen physical feed-point input-impedance patterns at 1x1 and 2x2."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
    builder_text as v114_builder_text,
    efficiency_from_csv,
    memory_available_gb,
    parse_touchstone,
    profile_metrics,
    solve_text as v114_solve_text,
)
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v125_feedpoint_input_impedance_preregistered.json"
EPS = 1.0e-15


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in (
        "builder_source",
        "trusted_protocol",
        "trusted_geometry",
        "trusted_s4",
        "stimulus_csv",
        "stimulus_npz",
    ):
        path = resolve(config["inputs"][key])
        observed = sha256_file(path)
        expected = str(config["inputs"][f"{key}_sha256"]).lower()
        if observed != expected:
            raise RuntimeError(f"Hash mismatch for {key}: {observed} != {expected}")
        rows.append({"role": key, "path": str(path), "size_bytes": path.stat().st_size, "sha256": observed})
    return rows


def trusted_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(resolve(config["inputs"]["trusted_protocol"]))
    geometry = read_json(resolve(config["inputs"]["trusted_geometry"]))["candidate"]
    return protocol, geometry


def patterned_builder_text(
    project: Path,
    side: int,
    candidate: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    base = copy.deepcopy(candidate)
    pattern = str(base.pop("feed_pattern", "uniform"))
    delta = float(base.pop("feed_delta_mm", 0.0))
    center = float(base.pop("feed_center_mm", base["feed_inset_from_edge_mm"]))
    if pattern == "uniform":
        base["feed_inset_from_edge_mm"] = center
        return v114_builder_text(project, side, base, protocol)
    if pattern not in {"checkerboard", "x_stripe", "y_stripe"}:
        raise ValueError(f"Unsupported feed pattern: {pattern}")
    base["feed_inset_from_edge_mm"] = center
    text = v114_builder_text(project, side, base, protocol)
    old_dim = "Dim ix, iy, idx, xc, yc, patchBottom, feedY, slotOffset, nameBase"
    new_dim = "Dim ix, iy, idx, xc, yc, patchBottom, feedY, feedInsetLocal, slotOffset, nameBase"
    if text.count(old_dim) != 1:
        raise RuntimeError("Cannot locate the v1.14 feed-variable declaration")
    text = text.replace(old_dim, new_dim)
    old_line = f"        feedY = patchBottom + {center:.6f}"
    selector = {
        "checkerboard": "((ix + iy) Mod 2)",
        "x_stripe": "(ix Mod 2)",
        "y_stripe": "(iy Mod 2)",
    }[pattern]
    replacement = (
        f"        feedInsetLocal = {center + delta:.6f}\n"
        f"        If {selector} = 1 Then feedInsetLocal = {center - delta:.6f}\n"
        "        feedY = patchBottom + feedInsetLocal"
    )
    if text.count(old_line) != 1:
        raise RuntimeError("Cannot locate the v1.14 feed-inset assignment")
    return text.replace(old_line, replacement)


def prepare_case(
    out: Path,
    side: int,
    candidate: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(candidate["candidate_id"])
    folder = out / f"{side}x{side}" / case_id
    if folder.exists():
        return read_json(folder / "case_manifest.json")
    folder.mkdir(parents=True)
    project = folder / f"v125_{side}x{side}_{case_id}.aedt"
    touchstone = folder / f"v125_{side}x{side}_{case_id}.s{side*side}p"
    efficiency = folder / "radiation_efficiency.csv"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    builder.write_text(patterned_builder_text(project, side, candidate, protocol), encoding="ascii")
    solver.write_text(v114_solve_text(project, touchstone, efficiency), encoding="ascii")
    manifest = {
        "case_id": case_id,
        "side": side,
        "port_count": side * side,
        "candidate": candidate,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "efficiency_csv_path": str(efficiency.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "evidence_scope": "physical patch/coax HFSS with modified feed point; no external matching or bridge network",
    }
    write_json(folder / "case_manifest.json", manifest)
    return manifest


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.25 output: {out}")
    out.mkdir(parents=True)
    input_rows = verify_inputs(config)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
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
    cases = []
    for inset in config["one_by_one_feed_insets_mm"]:
        candidate = {
            **geometry,
            "candidate_id": f"uniform_{float(inset):.2f}".replace(".", "p"),
            "feed_pattern": "uniform",
            "feed_center_mm": float(inset),
            "feed_delta_mm": 0.0,
        }
        cases.append(prepare_case(out, 1, candidate, protocol))
    write_json(out / "one_by_one_manifest.json", {"cases": cases})
    write_csv(out / "frozen_input_manifest.csv", input_rows)
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
            "evidence_rules": {
                "feedpoint_geometry_only": True,
                "no_external_matching_or_bridge_network": True,
                "one_by_one_gate_before_two_by_two": True,
                "same_three_frequencies_and_stimuli": True,
                "no_training_or_array_expansion": True,
            },
        },
    )
    decision = {
        "stage": "A_feedpoint_preregistered",
        "allow_1x1_run": True,
        "allow_2x2_prepare": False,
        "allow_2x2_run": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "one_by_one_case_count": len(cases), "decision": decision}


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def wait_for_memory(config: dict[str, Any], side: int) -> float:
    required = float(
        config["resources"]["minimum_free_memory_before_1x1_gib" if side == 1 else "minimum_free_memory_before_2x2_gib"]
    )
    deadline = time.time() + float(config["resources"]["memory_recovery_wait_seconds"])
    while True:
        require_no_aedt()
        free = memory_available_gb()
        if free >= required:
            return free
        if time.time() >= deadline:
            raise MemoryError(f"Memory did not recover to {required:.2f} GiB; current {free:.2f} GiB")
        time.sleep(float(config["resources"]["poll_interval_seconds"]))


def topology_warning_count(folder: Path) -> int:
    count = 0
    for log in (folder / "build.log", folder / "solve_export.log"):
        if not log.exists():
            continue
        text = log.read_text(encoding="utf-8", errors="ignore")
        for pattern in (
            "Too many conductors touch lumped port",
            "'0' conductors touch lumped port",
            "'1' conductors touch lumped port",
            "Both endpoints of port lines must lie on port",
            "script error",
            "invalid geometry",
        ):
            count += text.lower().count(pattern.lower())
    return count


def run_manifest(config: dict[str, Any], manifest_name: str, authorization: str) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get(authorization):
        raise RuntimeError(f"Run not authorized by {authorization}")
    cases = read_json(out / manifest_name)["cases"]
    progress_path = out / f"run_{Path(manifest_name).stem}.csv"
    rows = list(csv.DictReader(progress_path.open(encoding="utf-8"))) if progress_path.exists() else []
    completed = {str(row["case_id"]) for row in rows if str(row.get("touchstone_exists", "")).lower() == "true"}
    for case in cases:
        if case["case_id"] in completed:
            continue
        side = int(case["side"])
        free = wait_for_memory(config, side)
        folder = Path(case["project_path"]).parent
        with (folder / "build.log").open("w", encoding="utf-8") as handle:
            build = subprocess.run(
                [str(resolve(config["ansys_executable"])), "-RunScriptAndExit", case["builder_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if build.returncode != 0 or topology_warning_count(folder) > 0 or not Path(case["project_path"]).exists():
            raise RuntimeError(f"Build gate failed for {case['case_id']}")
        code, aborted, minimum_free = run_process_with_memory_guard(
            [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", case["solver_path"]],
            folder / "solve_export.log",
            float(config["resources"]["abort_free_memory_during_solve_gib"]),
            float(config["resources"]["poll_interval_seconds"]),
        )
        touchstone = Path(case["touchstone_path"])
        row = {
            "case_id": case["case_id"],
            "side": side,
            "build_return_code": int(build.returncode),
            "solve_return_code": code,
            "memory_aborted": aborted,
            "free_memory_gib_before": free,
            "minimum_free_memory_gib": minimum_free,
            "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
            "topology_warning_count": topology_warning_count(folder),
        }
        rows.append(row)
        write_csv(progress_path, rows)
        if code != 0 or aborted or not row["touchstone_exists"] or row["topology_warning_count"] > 0:
            raise RuntimeError(f"Solve gate failed for {case['case_id']}: {row}")
    return {"completed_count": len(rows), "rows": rows}


def analyze_case(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    folder = Path(case["touchstone_path"]).parent
    touchstone = Path(case["touchstone_path"])
    side = int(case["side"])
    frequencies, matrices = parse_touchstone(touchstone, side * side)
    target_f = np.asarray(config["frequencies_ghz"], dtype=float)
    indices = [int(np.argmin(np.abs(frequencies - value))) for value in target_f]
    selected_f = frequencies[indices]
    selected = matrices[indices]
    passive_rl = float(
        min(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matrix)), EPS))) for matrix in selected)
    )
    reciprocity = float(np.max(np.abs(selected - np.transpose(selected, (0, 2, 1)))))
    passivity = float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in selected))
    efficiency = efficiency_from_csv(Path(case["efficiency_csv_path"]))
    result = {
        "case_id": case["case_id"],
        "side": side,
        "feed_pattern": case["candidate"].get("feed_pattern"),
        "feed_center_mm": case["candidate"].get("feed_center_mm"),
        "feed_delta_mm": case["candidate"].get("feed_delta_mm"),
        **profile_metrics(folder),
        "frequency_max_error_ghz": float(np.max(np.abs(selected_f - target_f))),
        "passive_rl_min_db": passive_rl,
        "reciprocity_error": reciprocity,
        "passivity_sigma": passivity,
        "minimum_radiation_efficiency": efficiency,
        "topology_warning_count": topology_warning_count(folder),
    }
    if side == 2:
        replay, active_rl, total_rl = active_metrics(resolve(config["inputs"]["stimulus_root"]), 2, selected_f, selected)
        write_csv(folder / "representative_active_rl_replay.csv", replay)
        result["representative_active_rl_min_db"] = active_rl
        result["representative_total_rl_min_db"] = total_rl
        result["representative_source_count"] = len(replay)
    gates = config["gates"]
    common = bool(
        result.get("converged") is True
        and float(result.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
        and reciprocity <= float(gates["maximum_reciprocity_error"])
        and passivity <= float(gates["maximum_passivity_sigma"])
        and efficiency is not None
        and result["topology_warning_count"] <= int(gates["maximum_port_topology_warning_count"])
    )
    if side == 1:
        result["gate_pass"] = bool(
            common
            and passive_rl >= float(gates["minimum_1x1_passive_rl_db"])
            and float(efficiency) >= float(gates["minimum_1x1_radiation_efficiency"])
        )
    else:
        result["gate_pass"] = bool(
            common
            and passive_rl >= float(gates["minimum_2x2_passive_rl_db"])
            and float(result["representative_active_rl_min_db"]) >= float(gates["minimum_2x2_active_rl_db"])
            and float(result["representative_total_rl_min_db"]) >= float(gates["minimum_2x2_total_rl_db"])
            and float(efficiency) >= float(gates["minimum_2x2_radiation_efficiency"])
        )
    write_json(folder / "analysis.json", result)
    return result


def analyze_1x1(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "one_by_one_manifest.json")["cases"]
    rows = [analyze_case(config, case) for case in cases]
    write_csv(out / "one_by_one_metrics.csv", rows)
    by_inset = {float(row["feed_center_mm"]): row for row in rows}
    selected_range = None
    for item in sorted(config["candidate_ranges"], key=lambda value: int(value["priority"])):
        low = by_inset[float(item["low_mm"])]
        high = by_inset[float(item["high_mm"])]
        if low["gate_pass"] and high["gate_pass"]:
            selected_range = item
            break
    passing = [row for row in rows if row["gate_pass"]]
    uniform_best = max(
        passing,
        key=lambda row: (float(row["passive_rl_min_db"]), float(row["minimum_radiation_efficiency"])),
        default=None,
    )
    summary = {
        "case_count": len(rows),
        "gate_pass_count": sum(bool(row["gate_pass"]) for row in rows),
        "selected_periodic_range": selected_range,
        "uniform_best_feed_inset_mm": float(uniform_best["feed_center_mm"]) if uniform_best else None,
        "range_gate_pass": selected_range is not None and uniform_best is not None,
    }
    write_json(out / "one_by_one_summary.json", summary)
    decision = {
        "stage": "B_1x1_feedpoint_gate_complete",
        "one_by_one_range_gate_pass": summary["range_gate_pass"],
        "allow_2x2_prepare": summary["range_gate_pass"],
        "allow_2x2_run": False,
        "transition_to_radiator_input_geometry": not summary["range_gate_pass"],
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": rows, "decision": decision}


def prepare_2x2(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_2x2_prepare"):
        raise RuntimeError("2x2 preparation is not authorized")
    summary = read_json(out / "one_by_one_summary.json")
    selected = summary["selected_periodic_range"]
    center = 0.5 * (float(selected["low_mm"]) + float(selected["high_mm"]))
    delta = float(selected["delta_mm"])
    protocol, geometry = trusted_inputs(config)
    candidates = []
    for pattern in ("checkerboard", "x_stripe", "y_stripe"):
        candidates.append(
            {
                **geometry,
                "candidate_id": f"{pattern}_d{delta:.2f}".replace(".", "p"),
                "feed_pattern": pattern,
                "feed_center_mm": center,
                "feed_delta_mm": delta,
            }
        )
    best = float(summary["uniform_best_feed_inset_mm"])
    candidates.append(
        {
            **geometry,
            "candidate_id": f"uniform_best_{best:.2f}".replace(".", "p"),
            "feed_pattern": "uniform",
            "feed_center_mm": best,
            "feed_delta_mm": 0.0,
        }
    )
    cases = [prepare_case(out, 2, candidate, protocol) for candidate in candidates]
    write_json(out / "two_by_two_manifest.json", {"cases": cases})
    decision = {
        "stage": "C_2x2_feedpoint_cases_prepared",
        "allow_2x2_run": True,
        "allow_independent_repeat": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"case_count": len(cases), "candidates": candidates, "decision": decision}


def analyze_2x2(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "two_by_two_manifest.json")["cases"]
    rows = [analyze_case(config, case) for case in cases]
    trusted_f, trusted_s = parse_touchstone(resolve(config["inputs"]["trusted_s4"]), 4)
    trusted_replay, trusted_active, trusted_total = active_metrics(
        resolve(config["inputs"]["stimulus_root"]), 2, trusted_f, trusted_s
    )
    rows.append(
        {
            "case_id": "trusted_v1143_control",
            "side": 2,
            "feed_pattern": "uniform",
            "feed_center_mm": 2.3,
            "feed_delta_mm": 0.0,
            "passive_rl_min_db": float(
                min(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matrix)), EPS))) for matrix in trusted_s)
            ),
            "representative_active_rl_min_db": trusted_active,
            "representative_total_rl_min_db": trusted_total,
            "representative_source_count": len(trusted_replay),
            "gate_pass": False,
            "evidence_scope": "trusted control; not rerun in v1.25",
        }
    )
    write_csv(out / "two_by_two_metrics.csv", rows)
    candidates = [row for row in rows if row["case_id"] != "trusted_v1143_control"]
    passing = [row for row in candidates if row["gate_pass"]]
    best = max(
        candidates,
        key=lambda row: (
            float(row["representative_active_rl_min_db"]),
            float(row["representative_total_rl_min_db"]),
        ),
    )
    summary = {
        "candidate_count": len(candidates),
        "gate_pass_count": len(passing),
        "best_candidate": best["case_id"],
        "best_active_rl_db": best["representative_active_rl_min_db"],
        "best_total_rl_db": best["representative_total_rl_min_db"],
        "trusted_control_active_rl_db": trusted_active,
        "trusted_control_total_rl_db": trusted_total,
        "feedpoint_feasible_set_nonempty": bool(passing),
    }
    write_json(out / "stage_summary.json", summary)
    decision = {
        "stage": "D_2x2_feedpoint_gate_complete",
        "feedpoint_feasible_set_nonempty": bool(passing),
        "allow_one_independent_2x2_repeat": bool(passing),
        "transition_to_radiator_input_geometry": not bool(passing),
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "A physical 2x2 feedpoint candidate passed; one independent repeat is authorized."
            if passing
            else "Periodic feed-inset patterns did not form an 11 dB active-RL feasible set; modify the radiator/input geometry next."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": rows, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    one = read_json(out / "one_by_one_manifest.json")["cases"] if (out / "one_by_one_manifest.json").exists() else []
    two = read_json(out / "two_by_two_manifest.json")["cases"] if (out / "two_by_two_manifest.json").exists() else []
    return {
        "output_directory": str(out),
        "one_by_one_touchstones": sum(Path(case["touchstone_path"]).exists() for case in one),
        "one_by_one_case_count": len(one),
        "two_by_two_touchstones": sum(Path(case["touchstone_path"]).exists() for case in two),
        "two_by_two_case_count": len(two),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=(
            "preregister",
            "run-1x1",
            "analyze-1x1",
            "prepare-2x2",
            "run-2x2",
            "analyze-2x2",
            "status",
        ),
        default="status",
    )
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    actions = {
        "preregister": preregister,
        "run-1x1": lambda item: run_manifest(item, "one_by_one_manifest.json", "allow_1x1_run"),
        "analyze-1x1": analyze_1x1,
        "prepare-2x2": prepare_2x2,
        "run-2x2": lambda item: run_manifest(item, "two_by_two_manifest.json", "allow_2x2_run"),
        "analyze-2x2": analyze_2x2,
        "status": status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
