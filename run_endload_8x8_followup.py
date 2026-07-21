"""Promote one strict 4x4 end-load winner to a single strict 8x8 HFSS smoke.

This script is deliberately gated: it only runs after the 4x4 sweep has finished,
requires the existing passive promotion rule, and creates a fresh output directory.
It never launches a 16x16 solve or writes any training/full-wave labels.
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


DEFAULT_SOURCE = ROOT / "hfss_outputs" / "xcoupling_load_tune_4x4_20260716_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "xcoupling_load_tune_8x8_20260716_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def winner(source: Path) -> dict[str, str]:
    metrics = source / "prescreen_metrics.csv"
    manifest = source / "candidate_manifest.csv"
    if not metrics.exists():
        raise FileNotFoundError(f"4x4 analysis is missing: {metrics}")
    metric_rows = list(csv.DictReader(metrics.open(encoding="utf-8-sig")))
    promoted = [row for row in metric_rows if int(row.get("promoted", "0")) == 1]
    if not promoted:
        raise RuntimeError("No 4x4 candidate passed the passive promotion gate; 8x8 is blocked.")
    best = promoted[0]
    candidates = {row["name"]: row for row in csv.DictReader(manifest.open(encoding="utf-8-sig"))}
    return candidates[best["name"]]


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    selected = winner(args.source_dir)
    candidate: dict[str, Any] = {
        "name": f"strict8x8_{selected['name']}",
        "kind": selected["kind"],
        "length": float(selected["length"]),
        "radius": float(selected["radius"]),
        "bar_length": float(selected["bar_length"]),
        "bar_radius": float(selected["bar_radius"]),
        "blend_radius": float(selected["blend_radius"]),
        "cap_radius": float(selected["cap_radius"]),
        "spacing_x": float(selected["spacing_x"]),
        "spacing_y": float(selected["spacing_y"]),
        "side": 8,
    }
    args.out_dir.mkdir(parents=True)
    folder = args.out_dir / str(candidate["name"])
    folder.mkdir()
    project = folder / f"{candidate['name']}.aedt"
    touchstone = folder / f"{candidate['name']}.s64p"
    builder = folder / f"build_{candidate['name']}.vbs"
    solver = folder / f"solve_export_{candidate['name']}.vbs"
    write_builder(builder, project, candidate)
    write_solve_export(solver, project, touchstone)
    write_csv(args.out_dir / "candidate_manifest.csv", [{**candidate, "source_4x4": selected["name"], "project_path": str(project), "touchstone_path": str(touchstone)}])
    results: dict[str, Any] = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source_4x4": selected["name"], "scope": "one gated strict 8x8 embedded HFSS smoke; not a 16x16 training label"}
    with (folder / "build.log").open("w", encoding="utf-8", errors="replace") as handle:
        results["build_return_code"] = subprocess.run([str(args.ansys_exe), "-RunScriptAndExit", str(builder)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
    if results["build_return_code"] == 0:
        with (folder / "solve_export.log").open("w", encoding="utf-8", errors="replace") as handle:
            results["solve_return_code"] = subprocess.run([str(args.ansys_exe), "-ng", "-RunScriptAndExit", str(solver)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
    gate = convergence(project)
    results.update(gate)
    results["touchstone_exists"] = touchstone.exists()
    if gate["converged"] and touchstone.exists() and touchstone.stat().st_size > 1000:
        s = np.asarray(parse_touchstone(touchstone)["s_parameters"][0], dtype=np.complex128)
        results.update(passive_metrics(s))
        results["passive_10db_gate"] = bool(results["passive_rl_min_db"] >= 10.0)
    results["next_decision"] = "allow_paired_active_2400_evaluation" if results.get("passive_10db_gate") else "block_active_2400_and_model_balanced_feed_or_matching"
    (args.out_dir / "stage_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
