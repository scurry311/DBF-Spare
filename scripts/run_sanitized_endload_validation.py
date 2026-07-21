"""Gated 4x4 -> 8x8 validation of the sanitized L=10.8 mm end load.

This is a geometry-quality smoke only.  It neither writes HFSS training labels
nor starts the 2,400 active-return scenarios unless a separate later command is
issued after the 8x8 passive gate has passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_full_s256p_active_return import parse_touchstone
from design_modal_subarray_network import passive_metrics
from evaluate_embedded8x8_hierarchical_network import convergence
from run_embedded8x8_geometry_smoke import DEFAULT_ANSYS, ROOT, write_builder
from run_geometry_feed_smoke import write_solve_export


DEFAULT_OUT = ROOT / "hfss_outputs" / "sanitized_endload_validation_20260716_run04"
BASE: dict[str, Any] = {
    "kind": "planar_t_sheet",
    "length": 10.8,
    "radius": 0.35,
    "bar_length": 2.4,
    "bar_radius": 0.24,
    "cap_radius": 0.28,
    "blend_radius": 0.52,
    "blend_sleeve_length": 0.60,
    "spacing_x": 16.0,
    "spacing_y": 15.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="status")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mesh_warning_count(project: Path) -> int:
    return sum(
        text.lower().count("small mesh segment")
        for path in project.with_suffix(".aedtresults").rglob("*.g3derr")
        for text in [path.read_text(encoding="utf-8", errors="replace")]
    )


def evaluate(project: Path, touchstone: Path) -> dict[str, Any]:
    result = convergence(project)
    result["touchstone_exists"] = touchstone.exists()
    result["small_mesh_segment_count"] = mesh_warning_count(project)
    if result["converged"] and touchstone.exists() and touchstone.stat().st_size > 1000:
        s = np.asarray(parse_touchstone(touchstone)["s_parameters"][0], dtype=np.complex128)
        result.update(passive_metrics(s))
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for side in (4, 8):
        name = f"sanitized_l10p8_bar2p4_cap028_blend052_{side}x{side}"
        candidate = {**BASE, "name": name, "side": side}
        folder = args.out_dir / name
        folder.mkdir()
        project = folder / f"{name}.aedt"
        touchstone = folder / f"{name}.s{side * side}p"
        builder = folder / f"build_{name}.vbs"
        solver = folder / f"solve_export_{name}.vbs"
        write_builder(builder, project, candidate)
        write_solve_export(solver, project, touchstone)
        rows.append({**candidate, "project_path": str(project), "touchstone_path": str(touchstone), "builder_path": str(builder), "solver_path": str(solver)})
    write_csv(args.out_dir / "candidate_manifest.csv", rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "single-piece planar PEC T end-load 4x4 then gated 8x8 passive validation",
        "fixed_parameters_mm": {key: BASE[key] for key in ("length", "bar_length", "cap_radius", "blend_radius")},
        "4x4_gate": "converged; S16 exists; no small-mesh warnings; min passive RL >= 10 dB",
        "8x8_gate": "converged; S64 exists; no small-mesh warnings; min passive RL >= 10 dB",
        "active_2400_started": False,
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_one(row: dict[str, str], args: argparse.Namespace) -> int:
    folder = Path(row["project_path"]).parent
    with (folder / "build.log").open("w", encoding="utf-8", errors="replace") as handle:
        build = subprocess.run([str(args.ansys_exe), "-RunScriptAndExit", row["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if build.returncode:
        return int(build.returncode)
    with (folder / "solve_export.log").open("w", encoding="utf-8", errors="replace") as handle:
        solve = subprocess.run([str(args.ansys_exe), "-ng", "-RunScriptAndExit", row["solver_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(solve.returncode)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = list(csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")))
    four, eight = rows
    results: list[dict[str, Any]] = []
    started = time.time()
    code = run_one(four, args)
    four_metrics = evaluate(Path(four["project_path"]), Path(four["touchstone_path"]))
    four_gate = bool(code == 0 and four_metrics["converged"] and four_metrics["small_mesh_segment_count"] == 0 and four_metrics.get("passive_rl_min_db", -999.0) >= 10.0)
    results.append({"stage": "4x4", "return_code": code, "gate_pass": int(four_gate), **four_metrics})
    if four_gate:
        code = run_one(eight, args)
        eight_metrics = evaluate(Path(eight["project_path"]), Path(eight["touchstone_path"]))
        eight_gate = bool(code == 0 and eight_metrics["converged"] and eight_metrics["small_mesh_segment_count"] == 0 and eight_metrics.get("passive_rl_min_db", -999.0) >= 10.0)
        results.append({"stage": "8x8", "return_code": code, "gate_pass": int(eight_gate), **eight_metrics})
    write_csv(args.out_dir / "validation_metrics.csv", results)
    decision = "allow_active_2400_in_separate_command" if len(results) == 2 and results[-1]["gate_pass"] else "block_active_2400_and_model_balanced_feed_or_matching"
    summary = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_seconds": time.time() - started, "stages": results, "active_2400_started": False, "decision": decision}
    (args.out_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "run":
        result = run(args)
    elif args.mode == "analyze":
        rows = list(csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")))
        result = {row["name"]: evaluate(Path(row["project_path"]), Path(row["touchstone_path"])) for row in rows}
    else:
        result = {"output_exists": args.out_dir.exists(), "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
