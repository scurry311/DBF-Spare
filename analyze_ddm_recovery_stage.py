#!/usr/bin/env python3
"""Audit a staged DDM recovery S-matrix without opening training labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from run_staged_16x16_convergence import (
    NPORTS,
    PASSIVITY_SIGMA_LIMIT,
    RECIPROCITY_LIMIT,
    parse_touchstone,
    s_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--touchstone", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-csv", type=Path)
    parser.add_argument("--stage", required=True)
    return parser.parse_args()


def profile_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    current_pass: int | None = None
    deltas: dict[str, float] = {}
    matrix_sizes: dict[str, int] = {}
    tetrahedra: dict[str, int] = {}
    for line in text.splitlines():
        match = re.search(r"Name='Adaptive Pass\s+(\d+)'", line)
        if match:
            current_pass = int(match.group(1))
            continue
        if current_pass is None:
            continue
        match = re.search(r"Max Mag\. Delta S.*?,\s*([0-9.+\-Ee]+)\s*,", line)
        if match:
            deltas[str(current_pass)] = float(match.group(1))
        if "ProfileItem('Domain Partitioning'" in line:
            match = re.search(r"Tetrahedra.*?,\s*(\d+)", line)
            if match:
                tetrahedra[str(current_pass)] = int(match.group(1))
        if "ProfileItem('Iterations'" in line:
            match = re.search(r"Total matrix size.*?([0-9]{4,})", line)
            if match:
                matrix_sizes[str(current_pass)] = int(match.group(1))
    final_pass = max(
        [int(key) for key in set(deltas) | set(matrix_sizes) | set(tetrahedra)],
        default=0,
    )
    consecutive = 0
    for pass_number in range(final_pass, 1, -1):
        value = deltas.get(str(pass_number))
        if value is None or value > 0.05:
            break
        consecutive += 1
    return {
        "profile": str(path.resolve()),
        "normal_completion": "Normal Completion" in text,
        "hfss_converged": (
            "Adaptive Passes converged" in text and "did not converge" not in text
        ),
        "final_pass": final_pass,
        "delta_s_by_pass": deltas,
        "final_delta_s": deltas.get(str(final_pass)),
        "consecutive_delta_s_pass_count": consecutive,
        "matrix_size_by_pass": matrix_sizes,
        "tetrahedra_by_pass": tetrahedra,
    }


def comparison(current: Path, reference: Path) -> dict[str, object]:
    s_current = parse_touchstone(current, NPORTS)
    s_reference = parse_touchstone(reference, NPORTS)
    delta = np.abs(s_current - s_reference)
    diagonal = np.diag(delta)
    off_diagonal = delta.copy()
    np.fill_diagonal(off_diagonal, np.nan)
    max_index = np.unravel_index(np.argmax(delta), delta.shape)
    off_index = np.unravel_index(np.nanargmax(off_diagonal), off_diagonal.shape)
    return {
        "reference": str(reference.resolve()),
        "max_abs_delta_s": float(np.max(delta)),
        "max_abs_delta_s_ports_1based": [int(value) + 1 for value in max_index],
        "max_diagonal_delta_s": float(np.max(diagonal)),
        "max_diagonal_delta_s_port_1based": int(np.argmax(diagonal)) + 1,
        "max_offdiagonal_delta_s": float(np.nanmax(off_diagonal)),
        "max_offdiagonal_delta_s_ports_1based": [int(value) + 1 for value in off_index],
        "rms_abs_delta_s": float(np.sqrt(np.mean(delta**2))),
    }


def write_port_csv(path: Path, current: Path, reference: Path | None) -> None:
    s_current = parse_touchstone(current, NPORTS)
    s_reference = parse_touchstone(reference, NPORTS) if reference is not None else None
    rows: list[dict[str, object]] = []
    for port in range(NPORTS):
        row, column = divmod(port, 16)
        if row in (0, 15) and column in (0, 15):
            port_class = "corner"
        elif row in (0, 15) or column in (0, 15):
            port_class = "edge"
        else:
            port_class = "interior"
        item: dict[str, object] = {
            "port_1based": port + 1,
            "row_0based": row,
            "column_0based": column,
            "port_class": port_class,
            "raw_rl_db": float(-20.0 * np.log10(max(abs(s_current[port, port]), 1.0e-15))),
        }
        if s_reference is not None:
            item["diagonal_delta_s"] = float(
                abs(s_current[port, port] - s_reference[port, port])
            )
        rows.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result: dict[str, object] = {
        "stage": args.stage,
        **profile_metrics(args.profile),
        **s_metrics(args.touchstone),
    }
    result["numerical_smatrix_valid"] = bool(
        result["touchstone_complete"]
        and float(result["s_reciprocity_max_abs"]) <= RECIPROCITY_LIMIT
        and float(result["s_passivity_sigma_max"]) <= PASSIVITY_SIGMA_LIMIT
    )
    if args.reference is not None:
        result["comparison"] = comparison(args.touchstone, args.reference)
    if args.port_csv is not None:
        write_port_csv(args.port_csv, args.touchstone, args.reference)
    result["delta_s_limit"] = 0.05
    result["matched_rl_limit_db"] = 10.0
    result["delta_s_gate_pass"] = bool(
        result["final_delta_s"] is not None and float(result["final_delta_s"]) <= 0.05
    )
    result["consecutive_delta_s_gate_pass"] = bool(
        int(result["consecutive_delta_s_pass_count"]) >= 2
    )
    result["matched_rl_gate_pass"] = bool(
        float(result["matched_passive_rl_min_db"]) >= 10.0
    )
    result["strict_benchmark_gate_pass"] = bool(
        result["numerical_smatrix_valid"]
        and result["delta_s_gate_pass"]
        and result["consecutive_delta_s_gate_pass"]
        and result["matched_rl_gate_pass"]
    )
    result["training_labels_locked"] = True
    if not result["numerical_smatrix_valid"]:
        result["decision"] = "stop_invalid_smatrix"
    elif not result["delta_s_gate_pass"] or not result["consecutive_delta_s_gate_pass"]:
        result["decision"] = "continue_staged_ddm_convergence"
    elif not result["matched_rl_gate_pass"]:
        result["decision"] = "stop_converged_but_matching_gate_failed"
    else:
        result["training_labels_locked"] = False
        result["decision"] = "strict_gate_pass_allow_new_labels"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
