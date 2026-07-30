#!/usr/bin/env python3
"""Run exact HFSS-variable physical S8 optimization without circuit extrapolation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import memory_available_gb, parse_touchstone, profile_metrics
from run_v115_physical_modal_feed_fixture import (
    builder_text,
    evaluate_physical_s8,
    solver_text,
    touchstone_port_names,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v116_physical_s8_hfss_optimization.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="status")
    parser.add_argument("--stage", choices=("coarse", "refine", "repeat"), default="coarse")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def context(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    config = read_json(config_path)
    protocol_path = resolve(config["base_protocol"])
    design_path = resolve(config["base_physical_design"])
    return config, read_json(protocol_path), read_json(design_path), resolve(config["output_directory"]), protocol_path


def component_values(base: dict[str, Any], factors: dict[str, float]) -> dict[str, float]:
    values = base["physical_component_values"]
    return {
        "series_inductor_nh": float(values["series_inductor_nh"]) * float(factors["series_l"]),
        "ground_capacitor_pf": float(values["ground_capacitor_pf"]) * float(factors["ground_c"]),
        "bridge_inductor_nh": float(values["bridge_inductor_nh"]) * float(factors["bridge_l"]),
    }


def candidate_folder(out_dir: Path, candidate_id: str) -> Path:
    return out_dir / "candidates" / candidate_id


def wait_for_stable_file(path: Path, timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    previous_size = -1
    stable_checks = 0
    while time.monotonic() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 100 and size == previous_size:
                stable_checks += 1
                if stable_checks >= 2:
                    return True
            else:
                stable_checks = 0
            previous_size = size
        time.sleep(1.0)
    return False


def make_candidate(
    out_dir: Path,
    protocol: dict[str, Any],
    base: dict[str, Any],
    stage: str,
    candidate_id: str,
    factors: dict[str, float],
) -> dict[str, Any]:
    folder = candidate_folder(out_dir, candidate_id)
    folder.mkdir(parents=True, exist_ok=False)
    selected = {
        **base,
        "variant": "hfss_variable_grounded_lowpass_modal",
        "hfss_parameterize_components": True,
        "physical_component_values": component_values(base, factors),
    }
    project = folder / f"{candidate_id}.aedt"
    touchstone = folder / f"{candidate_id}.s8p"
    build_path = folder / "build.vbs"
    solve_path = folder / "solve_export.vbs"
    build_path.write_text(builder_text(project, protocol, selected), encoding="ascii")
    solve_path.write_text(solver_text(project, touchstone), encoding="ascii")
    write_json(folder / "selected_network.json", selected)
    manifest = {
        "candidate_id": candidate_id,
        "stage": stage,
        "factors": factors,
        "component_values": selected["physical_component_values"],
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(build_path.resolve()),
        "solver_path": str(solve_path.resolve()),
        "pre_reference_ports": [f"PRE_{index}" for index in range(4)],
        "post_reference_ports": [f"POST_{index}" for index in range(4)],
    }
    write_json(folder / "candidate_manifest.json", manifest)
    return manifest


def metric_score(row: dict[str, Any], config: dict[str, Any], protocol: dict[str, Any]) -> float:
    gates = protocol["gates"]
    active_target = float(config["search"]["design_active_rl_db"])
    margins = (
        float(row["passive_rl_min_db"]) - float(gates["minimum_passive_rl_db"]),
        float(row["active_rl_min_db"]) - active_target,
        float(row["total_rl_min_db"]) - float(gates["minimum_representative_total_rl_db"]),
        100.0 * (float(row["actual_load_insertion_efficiency_min"]) - float(gates["minimum_actual_load_insertion_efficiency"])),
        100.0 * (float(row["matched_load_network_efficiency_min"]) - float(gates["minimum_matched_load_network_efficiency"])),
    )
    return min(margins)


def analyzed_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out_dir / "candidates").glob("*/analysis.json")):
        rows.append(read_json(path))
    return rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    config, protocol, base, out_dir, protocol_path = context(args.config)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "config_snapshot.json").exists():
        shutil.copy2(args.config, out_dir / "config_snapshot.json")
        shutil.copy2(protocol_path, out_dir / "base_protocol_snapshot.json")
        write_json(out_dir / "base_physical_design_snapshot.json", base)
    manifests = []
    if args.stage == "coarse":
        definitions = config["search"]["coarse_factors"]
        for index, definition in enumerate(definitions):
            candidate_id = f"coarse_{index:02d}_{definition['name']}"
            folder = candidate_folder(out_dir, candidate_id)
            if not folder.exists():
                factors = {key: float(definition[key]) for key in ("series_l", "ground_c", "bridge_l")}
                manifests.append(make_candidate(out_dir, protocol, base, "coarse", candidate_id, factors))
    elif args.stage == "refine":
        rows = [row for row in analyzed_rows(out_dir) if row.get("stage") in ("coarse", "refine")]
        if not rows:
            raise RuntimeError("Analyze the coarse HFSS candidates before preparing refinement")
        best = max(rows, key=lambda row: metric_score(row, config, protocol))
        center = best["factors"]
        steps = config["search"]["refine_step_fraction"]
        definitions = []
        for key in ("series_l", "ground_c", "bridge_l"):
            for sign in (-1.0, 1.0):
                factors = {name: float(center[name]) for name in center}
                factors[key] *= 1.0 + sign * float(steps[key])
                definitions.append((key, sign, factors))
        for index, (key, sign, factors) in enumerate(definitions):
            direction = "minus" if sign < 0 else "plus"
            candidate_id = f"refine_{index:02d}_{key}_{direction}"
            if not candidate_folder(out_dir, candidate_id).exists():
                manifests.append(make_candidate(out_dir, protocol, base, "refine", candidate_id, factors))
        write_json(out_dir / "refine_center.json", best)
    else:
        rows = analyzed_rows(out_dir)
        qualified = [row for row in rows if row.get("design_gate_pass")]
        if not qualified:
            raise RuntimeError("No exact-HFSS candidate passed the 11.5 dB design gate")
        best = max(qualified, key=lambda row: metric_score(row, config, protocol))
        candidate_id = "repeat_00_independent"
        if not candidate_folder(out_dir, candidate_id).exists():
            manifests.append(make_candidate(out_dir, protocol, base, "repeat", candidate_id, best["factors"]))
        write_json(out_dir / "repeat_source.json", best)
    return {"stage": args.stage, "prepared": len(manifests), "candidates": manifests}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, _, _, out_dir, _ = context(args.config)
    ansys = resolve(config["ansys_executable"])
    minimum_memory = float(config["search"]["minimum_free_memory_gb"])
    results = []
    paths = sorted((out_dir / "candidates").glob(f"{args.stage}_*/candidate_manifest.json"))
    for manifest_path in paths:
        folder = manifest_path.parent
        manifest = read_json(manifest_path)
        touchstone = Path(manifest["touchstone_path"])
        if touchstone.exists() and touchstone.stat().st_size > 100:
            results.append({"candidate_id": manifest["candidate_id"], "status": "already_complete"})
            continue
        free_memory = memory_available_gb()
        if math.isfinite(free_memory) and free_memory < minimum_memory:
            raise MemoryError(f"Only {free_memory:.2f} GiB free; {minimum_memory:.2f} GiB required")
        with (folder / "build.log").open("w", encoding="utf-8") as handle:
            build = subprocess.run(
                [str(ansys), "-RunScriptAndExit", manifest["builder_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        solve_code = None
        project_ready = wait_for_stable_file(Path(manifest["project_path"]))
        if build.returncode == 0 and project_ready:
            with (folder / "solve_export.log").open("w", encoding="utf-8") as handle:
                solve = subprocess.run(
                    [str(ansys), "-ng", "-RunScriptAndExit", manifest["solver_path"]],
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            solve_code = int(solve.returncode)
        result = {
            "candidate_id": manifest["candidate_id"],
            "build_return_code": int(build.returncode),
            "solve_return_code": solve_code,
            "project_ready_before_solve": project_ready,
            "free_memory_gb_before": free_memory,
        }
        write_json(folder / "run_summary.json", result)
        results.append(result)
    return {"stage": args.stage, "run_results": results}


def analyze_candidate(
    folder: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    manifest = read_json(folder / "candidate_manifest.json")
    run_summary = read_json(folder / "run_summary.json") if (folder / "run_summary.json").exists() else {}
    result = {**manifest, **run_summary, **profile_metrics(folder)}
    touchstone = Path(manifest["touchstone_path"])
    if not touchstone.exists() or touchstone.stat().st_size < 100:
        result.update({"engineering_gate_pass": False, "design_gate_pass": False})
        write_json(folder / "analysis.json", result)
        return result
    frequencies, network_s8 = parse_touchstone(touchstone, 8)
    names = touchstone_port_names(touchstone)
    desired = manifest["pre_reference_ports"] + manifest["post_reference_ports"]
    if set(names) != set(desired):
        raise RuntimeError(f"Unexpected port order for {manifest['candidate_id']}: {names}")
    permutation = [names.index(name) for name in desired]
    network_s8 = network_s8[:, permutation][:, :, permutation]
    if not np.allclose(frequencies, protocol["frequencies_ghz"], atol=1.0e-9):
        raise RuntimeError("HFSS frequency grid mismatch")
    reciprocity = float(np.max(np.abs(network_s8 - np.transpose(network_s8, (0, 2, 1)))))
    passivity = float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in network_s8))
    metrics, sources = evaluate_physical_s8(network_s8, protocol, folder)
    frequency_rows = metrics.pop("frequency_rows")
    write_csv(folder / "physical_frequency_metrics.csv", frequency_rows)
    write_csv(folder / "physical_stimulus_metrics.csv", sources)
    result.update(metrics)
    result.update({"reciprocity_error_max": reciprocity, "passivity_sigma_max": passivity})
    gates = protocol["gates"]
    common = bool(
        result.get("converged") is True
        and float(result.get("final_delta_s") or math.inf) <= float(gates["maximum_physical_final_delta_s"])
        and reciprocity <= float(gates["maximum_physical_reciprocity_error"])
        and passivity <= float(gates["maximum_physical_passivity_sigma"])
        and result["passive_rl_min_db"] >= float(gates["minimum_passive_rl_db"])
        and result["total_rl_min_db"] >= float(gates["minimum_representative_total_rl_db"])
        and result["actual_load_insertion_efficiency_min"] >= float(gates["minimum_actual_load_insertion_efficiency"])
        and result["matched_load_network_efficiency_min"] >= float(gates["minimum_matched_load_network_efficiency"])
    )
    result["engineering_gate_pass"] = bool(
        common and result["active_rl_min_db"] >= float(config["search"]["engineering_active_rl_db"])
    )
    result["design_gate_pass"] = bool(
        common and result["active_rl_min_db"] >= float(config["search"]["design_active_rl_db"])
    )
    result["optimization_score"] = metric_score(result, config, protocol)
    result["frequency_rows"] = frequency_rows
    write_json(folder / "analysis.json", result)
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config, protocol, _, out_dir, _ = context(args.config)
    rows = []
    for manifest_path in sorted((out_dir / "candidates").glob("*/candidate_manifest.json")):
        rows.append(analyze_candidate(manifest_path.parent, config, protocol))
    write_csv(out_dir / "candidate_metrics.csv", rows)
    complete = [row for row in rows if "active_rl_min_db" in row]
    best = max(complete, key=lambda row: metric_score(row, config, protocol)) if complete else None
    nominal = next((row for row in complete if row["candidate_id"] == "coarse_00_nominal"), None)
    repeat = next((row for row in complete if row["stage"] == "repeat"), None)
    repeat_source_path = out_dir / "repeat_source.json"
    repeat_delta = None
    if repeat is not None and repeat_source_path.exists():
        source = read_json(repeat_source_path)
        _, first_s = parse_touchstone(Path(source["touchstone_path"]), 8)
        _, second_s = parse_touchstone(Path(repeat["touchstone_path"]), 8)
        repeat_delta = float(np.max(np.abs(first_s - second_s)))
    repeat_pass = bool(
        repeat is not None
        and repeat.get("design_gate_pass")
        and repeat_delta is not None
        and repeat_delta <= float(protocol["gates"]["maximum_independent_repeat_delta_s"])
    )
    decision = {
        "best_candidate_id": best["candidate_id"] if best else None,
        "best_active_rl_db": best.get("active_rl_min_db") if best else None,
        "best_active_rl_improvement_db": (
            float(best["active_rl_min_db"]) - float(nominal["active_rl_min_db"])
            if best and nominal
            else None
        ),
        "engineering_active_rl_target_db": float(config["search"]["engineering_active_rl_db"]),
        "design_active_rl_target_db": float(config["search"]["design_active_rl_db"]),
        "best_design_gate_pass": bool(best and best.get("design_gate_pass")),
        "engineering_pass_count": sum(bool(row.get("engineering_gate_pass")) for row in complete),
        "design_pass_count": sum(bool(row.get("design_gate_pass")) for row in complete),
        "independent_repeat_max_abs_delta_s": repeat_delta,
        "independent_repeat_pass": repeat_pass,
        "topology_decision": (
            "retain_grounded_x_modal_network_and_run_one_integrated_physical_2x2_smoke"
            if repeat_pass
            else "keep_distributed_even_odd_hybrid_as_fallback"
        ),
        "allow_integrated_physical_2x2_smoke": repeat_pass,
        "allow_distributed_topology_switch": not repeat_pass,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out_dir / "stage_decision.json", decision)
    if best:
        write_json(out_dir / "best_candidate.json", best)
    return {"candidate_count": len(rows), "best": best, "decision": decision}


def status(args: argparse.Namespace) -> dict[str, Any]:
    _, _, _, out_dir, _ = context(args.config)
    manifests = sorted((out_dir / "candidates").glob("*/candidate_manifest.json")) if out_dir.exists() else []
    return {
        "prepared": len(manifests),
        "solved": sum(Path(read_json(path)["touchstone_path"]).exists() for path in manifests),
        "analyzed": sum((path.parent / "analysis.json").exists() for path in manifests),
        "decision": read_json(out_dir / "stage_decision.json") if (out_dir / "stage_decision.json").exists() else None,
        "free_memory_gb": memory_available_gb(),
    }


def main() -> None:
    args = parse_args()
    result = {"prepare": prepare, "run": run, "analyze": analyze, "status": status}[args.mode](args)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
