#!/usr/bin/env python3
"""Audit small-mesh-segment bodies across the v1.32-v1.37 differential inputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from run_v114_small_cell_broadband_feed import parse_touchstone, profile_metrics
from run_v125_feedpoint_input_impedance import write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v137_vertical_mesh_audit.json"
BODY_RE = re.compile(r"Small mesh segment detected on body\s*:\s*(\S+)", re.IGNORECASE)
LENGTH_RE = re.compile(r"Segment length\(s\)\s*:\s*([0-9.]+)mm", re.IGNORECASE)
EPS = 1.0e-15


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def segment_audit(folder: Path) -> tuple[Counter[str], list[float]]:
    bodies: Counter[str] = Counter()
    lengths: list[float] = []
    for error_file in folder.rglob("*.g3derr"):
        text = error_file.read_text(encoding="utf-8", errors="ignore")
        bodies.update(match.group(1) for match in BODY_RE.finditer(text))
        lengths.extend(float(match.group(1)) for match in LENGTH_RE.finditer(text))
    return bodies, lengths


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["output_directory"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.37 mesh audit: {output}")
    output.mkdir(parents=True)
    conductive = set(config["conductive_body_names"])
    gates = config["gates"]
    rows: list[dict[str, Any]] = []
    for case in config["cases"]:
        folder = resolve(case["folder"])
        touchstones = list(folder.glob("*.s1p"))
        if len(touchstones) != 1:
            raise RuntimeError(f"Expected one S1P in {folder}, found {len(touchstones)}")
        frequencies, matrices = parse_touchstone(touchstones[0], 1)
        index = int(np.argmin(np.abs(frequencies - 10.0)))
        s11 = matrices[index, 0, 0]
        passive_rl = float(-20.0 * np.log10(max(float(abs(s11)), EPS)))
        profile = profile_metrics(folder)
        bodies, lengths = segment_audit(folder)
        conductor_count = sum(count for body, count in bodies.items() if body in conductive)
        total_count = sum(bodies.values())
        rows.append(
            {
                "case_id": case["case_id"],
                "passive_rl_db": passive_rl,
                "final_delta_s": profile.get("final_delta_s"),
                "converged": profile.get("converged"),
                "total_small_segment_message_count": total_count,
                "conductor_small_segment_message_count": conductor_count,
                "dielectric_air_small_segment_message_count": total_count - conductor_count,
                "minimum_segment_length_mm": min(lengths) if lengths else None,
                "maximum_segment_length_mm": max(lengths) if lengths else None,
                "body_counts_json": json.dumps(dict(sorted(bodies.items())), separators=(",", ":")),
                "strict_total_gate_pass": total_count <= int(gates["maximum_total_small_mesh_segment_count"]),
                "conductor_geometry_gate_pass": conductor_count <= int(gates["maximum_conductor_small_mesh_segment_count"]),
                "matching_numerical_gate_pass": bool(
                    profile.get("converged") is True
                    and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
                    and passive_rl >= float(gates["minimum_passive_rl_db"])
                ),
            }
        )
    write_csv(output / "mesh_segment_audit.csv", rows)
    selected = rows[-1]
    decision = {
        "stage": "v137_mesh_segment_audit_complete",
        "selected_case": selected["case_id"],
        "strict_total_small_segment_gate_pass": selected["strict_total_gate_pass"],
        "conductor_geometry_gate_pass": selected["conductor_geometry_gate_pass"],
        "matching_numerical_gate_pass": selected["matching_numerical_gate_pass"],
        "allow_three_frequency_rerun": bool(selected["strict_total_gate_pass"] and selected["matching_numerical_gate_pass"]),
        "allow_2x2": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": "Conductive-sheet slivers are eliminated, but the preregistered zero-total-warning gate still fails on dielectric/air mesh messages; downstream stages remain locked."
    }
    write_json(output / "stage_decision.json", decision)
    write_json(output / "audit_config_snapshot.json", config)
    return {"rows": rows, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(analyze(read_json(args.config)), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
