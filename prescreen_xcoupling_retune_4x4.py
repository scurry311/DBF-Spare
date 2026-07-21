"""Strict 4x4 HFSS pre-screen for dx=16 mm x-coupling retuning candidates."""

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


DEFAULT_OUT = ROOT / "hfss_outputs" / "xcoupling_retune_4x4_20260716_run01"
CANDIDATES: tuple[dict[str, Any], ...] = (
    {"name": "load_bar2p4_cap028_blend052", "kind": "smooth_blended", "length": 10.8, "radius": 0.35, "bar_length": 2.4, "bar_radius": 0.24, "blend_radius": 0.52, "cap_radius": 0.28, "spacing_x": 16.0, "spacing_y": 15.0, "side": 4},
    {"name": "load_bar3p2_cap028_blend052", "kind": "smooth_blended", "length": 10.8, "radius": 0.35, "bar_length": 3.2, "bar_radius": 0.24, "blend_radius": 0.52, "cap_radius": 0.28, "spacing_x": 16.0, "spacing_y": 15.0, "side": 4},
    {"name": "load_bar2p8_cap022_blend052", "kind": "smooth_blended", "length": 10.8, "radius": 0.35, "bar_length": 2.8, "bar_radius": 0.24, "blend_radius": 0.52, "cap_radius": 0.22, "spacing_x": 16.0, "spacing_y": 15.0, "side": 4},
    {"name": "load_bar2p8_cap034_blend052", "kind": "smooth_blended", "length": 10.8, "radius": 0.35, "bar_length": 2.8, "bar_radius": 0.24, "blend_radius": 0.52, "cap_radius": 0.34, "spacing_x": 16.0, "spacing_y": 15.0, "side": 4},
    {"name": "load_bar2p8_cap028_blend044", "kind": "smooth_blended", "length": 10.8, "radius": 0.35, "bar_length": 2.8, "bar_radius": 0.24, "blend_radius": 0.44, "cap_radius": 0.28, "spacing_x": 16.0, "spacing_y": 15.0, "side": 4},
    {"name": "load_bar2p8_cap028_blend060", "kind": "smooth_blended", "length": 10.8, "radius": 0.35, "bar_length": 2.8, "bar_radius": 0.24, "blend_radius": 0.60, "cap_radius": 0.28, "spacing_x": 16.0, "spacing_y": 15.0, "side": 4},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="status")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        folder = args.out_dir / str(candidate["name"])
        folder.mkdir()
        project = folder / f"{candidate['name']}.aedt"
        touchstone = folder / f"{candidate['name']}.s16p"
        builder = folder / f"build_{candidate['name']}.vbs"
        solver = folder / f"solve_export_{candidate['name']}.vbs"
        write_builder(builder, project, candidate)
        write_solve_export(solver, project, touchstone)
        rows.append({**candidate, "project_path": str(project), "touchstone_path": str(touchstone), "builder_path": str(builder), "solver_path": str(solver)})
    write_csv(args.out_dir / "candidate_manifest.csv", rows)
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "scope": "strict 4x4 L=10.8 mm independent smooth-end-load pre-screen", "delta_s_gate": 0.05, "promotion_gate": "converged; min passive RL >= 10 dB; nearest-x coupling <= -14.5 dB"}
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = list(csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")))
    status: list[dict[str, Any]] = []
    for row in rows:
        folder = Path(row["project_path"]).parent
        with (folder / "build.log").open("w", encoding="utf-8", errors="replace") as handle:
            build = subprocess.run([str(args.ansys_exe), "-RunScriptAndExit", row["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        code = int(build.returncode)
        if code == 0:
            with (folder / "solve_export.log").open("w", encoding="utf-8", errors="replace") as handle:
                solve = subprocess.run([str(args.ansys_exe), "-ng", "-RunScriptAndExit", row["solver_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            code = int(solve.returncode)
        status.append({"name": row["name"], "return_code": code})
    write_csv(args.out_dir / "run_status.csv", status)
    return {"runs": status, "all_zero_exit": bool(status and all(row["return_code"] == 0 for row in status))}


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in csv.DictReader((args.out_dir / "candidate_manifest.csv").open(encoding="utf-8-sig")):
        project, touchstone = Path(item["project_path"]), Path(item["touchstone_path"])
        gate = convergence(project)
        row: dict[str, Any] = {"name": item["name"], "length_mm": float(item["length"]), "bar_length_mm": float(item["bar_length"]), "final_delta_s": gate["final_delta_s"], "pass_count": gate["pass_count"], "converged": int(gate["converged"]), "touchstone_exists": touchstone.exists()}
        if gate["converged"] and touchstone.exists() and touchstone.stat().st_size > 1000:
            s = np.asarray(parse_touchstone(touchstone)["s_parameters"][0], dtype=np.complex128)
            if s.shape == (16, 16):
                row.update(passive_metrics(s))
                row["promoted"] = int(row["passive_rl_min_db"] >= 10.0 and row["nearest_x_worst_db"] <= -14.5)
            else:
                row["status"] = f"unexpected_shape_{s.shape}"
        rows.append(row)
    rows.sort(key=lambda row: (int(row.get("promoted", 0)), float(row.get("passive_rl_min_db", -999.0)), -float(row.get("nearest_x_worst_db", 0.0))), reverse=True)
    write_csv(args.out_dir / "prescreen_metrics.csv", rows)
    promoted = [row["name"] for row in rows if int(row.get("promoted", 0))]
    summary = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "promoted_candidates": promoted, "allow_one_strict_8x8_smoke": bool(promoted), "decision": "promote_best_candidate_to_strict_8x8" if promoted else "block_8x8_until_4x4_passive_gate"}
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "run":
        result = run(args)
    elif args.mode == "analyze":
        result = analyze(args)
    else:
        result = {"output_exists": args.out_dir.exists(), "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
