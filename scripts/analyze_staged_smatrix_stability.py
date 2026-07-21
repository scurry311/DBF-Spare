#!/usr/bin/env python3
"""Locate S-matrix convergence hot spots in staged HFSS checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from run_staged_16x16_convergence import parse_touchstone


NPORTS = 256
SIDE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--first-pass", type=int, default=12)
    parser.add_argument("--last-pass", type=int)
    return parser.parse_args()


def port_class(index: int) -> str:
    row, column = divmod(index, SIDE)
    x_edge = row in (0, SIDE - 1)
    y_edge = column in (0, SIDE - 1)
    if x_edge and y_edge:
        return "corner"
    if x_edge or y_edge:
        return "edge"
    return "interior"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def successful_stage_by_pass(stage_root: Path) -> dict[int, Path]:
    selected: dict[int, tuple[int, Path]] = {}
    for path in stage_root.glob("pass*"):
        match = re.fullmatch(r"pass(\d+)(?:_retry(\d+))?", path.name)
        if match is None:
            continue
        pass_number = int(match.group(1))
        attempt = int(match.group(2) or 0)
        touchstone = path / "grounded_patch_16x16.s256p"
        metrics_path = path / "stage_metrics.json"
        if not touchstone.exists() or touchstone.stat().st_size <= 1000 or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        valid = bool(
            metrics.get("touchstone_complete") is True
            and int(metrics.get("final_pass", -1)) == pass_number
            and metrics.get("solve_return_code") in (None, 0)
        )
        if valid and (pass_number not in selected or attempt > selected[pass_number][0]):
            selected[pass_number] = (attempt, path)
    return {pass_number: item[1] for pass_number, item in selected.items()}


def main() -> None:
    args = parse_args()
    stage_root = args.out_dir / "stages"
    stage_by_pass = successful_stage_by_pass(stage_root)
    available = sorted(stage_by_pass)
    last_pass = args.last_pass or max(available)
    selected = [value for value in available if args.first_pass <= value <= last_pass]
    matrices = {
        value: parse_touchstone(
            stage_by_pass[value] / "grounded_patch_16x16.s256p", NPORTS
        )
        for value in selected
    }

    stage_quality = {}
    for value in selected:
        matrix = matrices[value]
        stage_quality[value] = {
            "stage_dir": stage_by_pass[value].name,
            "reciprocity_max_abs": float(np.max(np.abs(matrix - matrix.T))),
            "passivity_sigma_max": float(np.linalg.svd(matrix, compute_uv=False)[0]),
        }

    rows: list[dict[str, object]] = []
    dominant_ports: Counter[int] = Counter()
    for previous, current in zip(selected, selected[1:]):
        delta = np.abs(matrices[current] - matrices[previous])
        i, j = np.unravel_index(int(np.argmax(delta)), delta.shape)
        diagonal = np.abs(np.diag(matrices[current] - matrices[previous]))
        diagonal_port = int(np.argmax(diagonal))
        off_diagonal = delta.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        oi, oj = np.unravel_index(int(np.argmax(off_diagonal)), off_diagonal.shape)
        dominant_ports[diagonal_port] += 1
        rows.append(
            {
                "previous_pass": previous,
                "current_pass": current,
                "previous_stage_dir": stage_quality[previous]["stage_dir"],
                "current_stage_dir": stage_quality[current]["stage_dir"],
                "previous_reciprocity_max_abs": stage_quality[previous]["reciprocity_max_abs"],
                "current_reciprocity_max_abs": stage_quality[current]["reciprocity_max_abs"],
                "previous_passivity_sigma_max": stage_quality[previous]["passivity_sigma_max"],
                "current_passivity_sigma_max": stage_quality[current]["passivity_sigma_max"],
                "max_abs_delta": float(delta[i, j]),
                "max_i_port_1based": i + 1,
                "max_j_port_1based": j + 1,
                "max_is_diagonal": int(i == j),
                "max_diagonal_delta": float(diagonal[diagonal_port]),
                "max_diagonal_port_1based": diagonal_port + 1,
                "max_diagonal_row": diagonal_port // SIDE,
                "max_diagonal_column": diagonal_port % SIDE,
                "max_diagonal_port_class": port_class(diagonal_port),
                "max_offdiagonal_delta": float(off_diagonal[oi, oj]),
                "max_offdiag_i_port_1based": oi + 1,
                "max_offdiag_j_port_1based": oj + 1,
            }
        )

    csv_path = args.out_dir / "smatrix_stability_by_pass.csv"
    write_csv(csv_path, rows)
    summary = {
        "first_pass": selected[0],
        "last_pass": selected[-1],
        "pair_count": len(rows),
        "selected_stages": {str(key): value["stage_dir"] for key, value in stage_quality.items()},
        "stage_quality": {str(key): value for key, value in stage_quality.items()},
        "all_pairwise_maxima_are_diagonal": all(row["max_is_diagonal"] == 1 for row in rows),
        "latest_max_abs_delta": rows[-1]["max_abs_delta"],
        "latest_max_diagonal_port_1based": rows[-1]["max_diagonal_port_1based"],
        "latest_max_offdiagonal_delta": rows[-1]["max_offdiagonal_delta"],
        "dominant_diagonal_ports": [
            {
                "port_1based": port + 1,
                "row": port // SIDE,
                "column": port % SIDE,
                "port_class": port_class(port),
                "pair_count": count,
            }
            for port, count in dominant_ports.most_common(12)
        ],
        "diagnosis": (
            "Pairwise convergence is dominated by changing self-reflection terms. "
            "Apply deterministic local mesh controls to all feed/port regions before "
            "another full-array adaptive sequence."
        ),
        "training_labels_unlocked": False,
    }
    summary_path = args.out_dir / "smatrix_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
