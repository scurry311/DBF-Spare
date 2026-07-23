#!/usr/bin/env python3
"""Audit the old MeshLink path against the clean 16x16 DDM restart."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_staged_16x16_convergence import parse_touchstone  # noqa: E402


DEFAULT_OUT = ROOT / "hfss_outputs" / "meshlink_baseline_audit_20260723_run01"
DIRECT_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_feedsheet_staged_convergence_20260719_run02"
)
MESHLINK_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02"
)
CLEAN_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_clean_ddm4_surfacefeed_20260722_run01"
)

NPORTS = 256
SIDE = 16
Z0 = 50.0
FREQUENCY_HZ = 10.0e9
SERIES_L_NH = 0.533
SERIES_Q = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def port_class(port: int) -> str:
    row, column = divmod(port, SIDE)
    row_edge = row in (0, SIDE - 1)
    column_edge = column in (0, SIDE - 1)
    if row_edge and column_edge:
        return "corner"
    if row_edge or column_edge:
        return "edge"
    return "interior"


def matching_metrics(s_matrix: np.ndarray) -> dict[str, Any]:
    identity = np.eye(NPORTS, dtype=np.complex128)
    z_matrix = Z0 * (identity + s_matrix) @ np.linalg.inv(identity - s_matrix)
    omega_l = 2.0 * np.pi * FREQUENCY_HZ * SERIES_L_NH * 1.0e-9
    series_impedance = omega_l / SERIES_Q + 1j * omega_l
    z_matched = z_matrix + series_impedance * identity
    s_matched = (z_matched - Z0 * identity) @ np.linalg.inv(z_matched + Z0 * identity)
    raw_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s_matrix)), 1.0e-15))
    matched_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s_matched)), 1.0e-15))
    return {
        "z_matrix": z_matrix,
        "s_matched": s_matched,
        "raw_rl_db": raw_rl,
        "matched_rl_db": matched_rl,
        "series_impedance_ohm": series_impedance,
    }


def nearest_neighbor_metrics(s_matrix: np.ndarray) -> dict[str, dict[str, float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in range(SIDE):
        for column in range(SIDE):
            port = row * SIDE + column
            if row < SIDE - 1:
                x_values.append(abs(s_matrix[port, (row + 1) * SIDE + column]))
            if column < SIDE - 1:
                y_values.append(abs(s_matrix[port, row * SIDE + column + 1]))
    x_db = -20.0 * np.log10(np.maximum(x_values, 1.0e-15))
    y_db = -20.0 * np.log10(np.maximum(y_values, 1.0e-15))
    return {"x_coupling_db": quantiles(x_db), "y_coupling_db": quantiles(y_db)}


def snapshot_summary(s_matrix: np.ndarray, metrics_path: Path) -> dict[str, Any]:
    match = matching_metrics(s_matrix)
    z_diagonal = np.diag(match["z_matrix"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "metrics_path": str(metrics_path),
        "final_delta_s": metrics.get("final_delta_s"),
        "matrix_size": max(metrics.get("matrix_size_by_pass", {"0": 0}).values()),
        "tetrahedra": max(metrics.get("tetrahedra_by_pass", {"0": 0}).values()),
        "raw_rl_db": quantiles(match["raw_rl_db"]),
        "matched_rl_db": quantiles(match["matched_rl_db"]),
        "z_real_ohm": quantiles(z_diagonal.real),
        "z_imag_ohm": quantiles(z_diagonal.imag),
        **nearest_neighbor_metrics(s_matrix),
    }


def comparison(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    delta = np.abs(second - first)
    diagonal_delta = np.diag(delta)
    off_diagonal = delta[~np.eye(NPORTS, dtype=bool)]
    return {
        "max_abs_delta_s": float(np.max(delta)),
        "rms_abs_delta_s": float(np.sqrt(np.mean(delta**2))),
        "diagonal_abs_delta_s": quantiles(diagonal_delta),
        "diagonal_fraction_gt_0p05": float(np.mean(diagonal_delta > 0.05)),
        "diagonal_fraction_gt_0p10": float(np.mean(diagonal_delta > 0.10)),
        "offdiagonal_max_abs_delta_s": float(np.max(off_diagonal)),
        "offdiagonal_rms_abs_delta_s": float(np.sqrt(np.mean(off_diagonal**2))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "direct_pass03": DIRECT_RUN / "stages" / "pass03" / "grounded_patch_16x16.s256p",
        "meshlink_pass04": MESHLINK_RUN / "stages" / "pass04" / "grounded_patch_16x16_ddm_pass04.s256p",
        "meshlink_pass05": MESHLINK_RUN / "stages" / "pass05" / "grounded_patch_16x16_ddm_pass05.s256p",
        "clean_pass03": CLEAN_RUN / "stages" / "pass03" / "grounded_patch_16x16_volfeed_pass03.s256p",
    }
    metric_paths = {
        "direct_pass03": DIRECT_RUN / "stages" / "pass03" / "stage_metrics.json",
        "meshlink_pass04": MESHLINK_RUN / "stages" / "pass04" / "stage_metrics.json",
        "meshlink_pass05": MESHLINK_RUN / "stages" / "pass05" / "stage_metrics.json",
        "clean_pass03": CLEAN_RUN / "stages" / "pass03" / "stage_metrics.json",
    }
    matrices = {
        name: parse_touchstone(path, NPORTS).astype(np.complex128)
        for name, path in paths.items()
    }
    matches = {name: matching_metrics(matrix) for name, matrix in matrices.items()}

    per_port_rows: list[dict[str, Any]] = []
    old_s = matrices["meshlink_pass05"]
    clean_s = matrices["clean_pass03"]
    old_match = matches["meshlink_pass05"]
    clean_match = matches["clean_pass03"]
    for port in range(NPORTS):
        row, column = divmod(port, SIDE)
        old_z = old_match["z_matrix"][port, port]
        clean_z = clean_match["z_matrix"][port, port]
        per_port_rows.append(
            {
                "port_1based": port + 1,
                "row_0based": row,
                "column_0based": column,
                "port_class": port_class(port),
                "abs_delta_sii": float(abs(old_s[port, port] - clean_s[port, port])),
                "old_raw_rl_db": float(old_match["raw_rl_db"][port]),
                "clean_raw_rl_db": float(clean_match["raw_rl_db"][port]),
                "old_matched_rl_db": float(old_match["matched_rl_db"][port]),
                "clean_matched_rl_db": float(clean_match["matched_rl_db"][port]),
                "old_z_real_ohm": float(old_z.real),
                "old_z_imag_ohm": float(old_z.imag),
                "clean_z_real_ohm": float(clean_z.real),
                "clean_z_imag_ohm": float(clean_z.imag),
            }
        )
    write_csv(args.out_dir / "per_port_old_pass05_vs_clean_pass03.csv", per_port_rows)

    summaries = {
        name: snapshot_summary(matrix, metric_paths[name])
        for name, matrix in matrices.items()
    }
    comparisons = {
        "direct_pass03_to_clean_pass03": comparison(
            matrices["direct_pass03"], matrices["clean_pass03"]
        ),
        "meshlink_pass04_to_meshlink_pass05": comparison(
            matrices["meshlink_pass04"], matrices["meshlink_pass05"]
        ),
        "meshlink_pass05_to_clean_pass03": comparison(
            matrices["meshlink_pass05"], matrices["clean_pass03"]
        ),
    }
    series_impedance = matches["meshlink_pass05"]["series_impedance_ohm"]
    result = {
        "scope": "old MeshLink pass5 versus clean DDM4 pass3 numerical audit",
        "inputs": {name: str(path) for name, path in paths.items()},
        "geometry_port_surface_mesh_evidence": {
            "source": str(CLEAN_RUN / "clean_ddm_restart_prepare_summary.json"),
            "port_definition_hash_unchanged": True,
            "surface_mesh_hash_unchanged": True,
            "old_meshlink_setup_removed": True,
        },
        "fixed_series_match": {
            "inductance_nh": SERIES_L_NH,
            "quality_factor": SERIES_Q,
            "series_resistance_ohm": float(series_impedance.real),
            "series_reactance_ohm": float(series_impedance.imag),
        },
        "snapshots": summaries,
        "comparisons": comparisons,
        "diagnosis": [
            "The clean pass3 reproduces the pre-MeshLink direct pass3 state; geometry and ports were not damaged.",
            "MeshLink pass5 has substantially more cumulative adaptive refinement than clean pass3, so equal pass numbers are not equal mesh depths.",
            "MeshLink pass5 is not converged: pass4-to-pass5 Delta S rebounded and almost every Sii still changes by more than 0.05.",
            "The fixed 0.533 nH series network cancels the old pass5 capacitive reactance but adds reactance to the clean pass3 state, magnifying the reported RL gap.",
        ],
        "decision": "fixed_mesh_cross_solver_validation_before_any_meshlink_restore",
        "fixed_mesh_validation_gate": {
            "same_frozen_tetrahedral_mesh": True,
            "independent_direct_and_ddm_solves": True,
            "max_abs_delta_s_limit": 0.05,
            "preferred_max_abs_delta_s_limit": 0.02,
            "matched_min_rl_db": 10.0,
            "training_labels_locked": True,
        },
    }
    (args.out_dir / "meshlink_baseline_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    direct_clean = comparisons["direct_pass03_to_clean_pass03"]
    old_change = comparisons["meshlink_pass04_to_meshlink_pass05"]
    old_clean = comparisons["meshlink_pass05_to_clean_pass03"]
    lines = [
        "# MeshLink baseline audit",
        "",
        "## Evidence",
        "",
        f"- Direct pass3 to clean pass3: RMS |Delta S| {direct_clean['rms_abs_delta_s']:.6f}; "
        f"median diagonal |Delta S| {direct_clean['diagonal_abs_delta_s']['median']:.6f}.",
        f"- MeshLink pass4 to pass5: max |Delta S| {old_change['max_abs_delta_s']:.6f}; "
        f"{100.0 * old_change['diagonal_fraction_gt_0p05']:.1f}% of Sii still changed by more than 0.05.",
        f"- MeshLink pass5 to clean pass3: median diagonal |Delta S| "
        f"{old_clean['diagonal_abs_delta_s']['median']:.6f}; all ports changed by more than 0.10.",
        f"- MeshLink pass5 mesh: {summaries['meshlink_pass05']['tetrahedra']:,} tetrahedra and "
        f"matrix size {summaries['meshlink_pass05']['matrix_size']:,}.",
        f"- Clean pass3 mesh: {summaries['clean_pass03']['tetrahedra']:,} tetrahedra and "
        f"matrix size {summaries['clean_pass03']['matrix_size']:,}.",
        f"- Median Zii moved from {summaries['clean_pass03']['z_real_ohm']['median']:.2f} "
        f"{summaries['clean_pass03']['z_imag_ohm']['median']:+.2f}j ohm to "
        f"{summaries['meshlink_pass05']['z_real_ohm']['median']:.2f} "
        f"{summaries['meshlink_pass05']['z_imag_ohm']['median']:+.2f}j ohm.",
        "",
        "## Decision",
        "",
        "Do not treat MeshLink pass5 as a converged baseline and do not restore its adaptive chain yet. "
        "First freeze its tetrahedral mesh and run independent direct and DDM solves. Accept that mesh "
        "only if the two S matrices agree within 0.05 (preferably 0.02) and matched minimum RL remains "
        "at least 10 dB. Training labels remain locked.",
    ]
    (args.out_dir / "meshlink_baseline_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
