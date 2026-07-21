"""Compare converged embedded-8x8 antenna geometries without feed-network cascades."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_full_s256p_active_return import parse_touchstone
from design_modal_subarray_network import aggregate_case_metrics, case_metrics, passive_metrics
from evaluate_embedded8x8_hierarchical_network import convergence


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
BASE_ROOT = ROOT / "hfss_outputs" / "embedded8x8_modal_smoke_20260716_run04" / "smooth_blended_l11p2_bar2p0"
NEW_ROOT = ROOT / "hfss_outputs" / "hardware_xcoupling_20260716_run01" / "smooth_compact_l10p4_bar3p0_dx16p0"
DEFAULT_OUT = ROOT / "hfss_outputs" / "hardware_xcoupling_20260716_run01" / "paired_hardware_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--return-loss-db", type=float, default=10.0)
    parser.add_argument("--significant-power-relative-db", type=float, default=-30.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scenarios(dataset_path: Path) -> dict[str, np.ndarray]:
    payload = np.load(dataset_path, allow_pickle=False)
    points = np.asarray(payload["element_ixiy"], dtype=int)
    chosen = np.flatnonzero(
        (points[:, 0] >= 4) & (points[:, 0] <= 11) & (points[:, 1] >= 4) & (points[:, 1] <= 11)
    )
    chosen = chosen[np.lexsort((points[chosen, 1], points[chosen, 0]))]
    if chosen.size != 64:
        raise RuntimeError(f"Expected 64 central ports, got {chosen.size}")
    weights_ri = np.asarray(payload["hfss_weights_real_imag"], dtype=np.float64)[:, chosen]
    weights = weights_ri[:, :, 0] + 1j * weights_ri[:, :, 1]
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1.0e-15)
    targets = np.asarray(payload["targets_deg"], dtype=np.float64)
    k_values = np.asarray(payload["k_values"], dtype=int)
    max_theta = np.asarray([np.nanmax(targets[index, : k_values[index], 0]) for index in range(len(k_values))])
    return {
        "weights": weights,
        "masks": np.asarray(payload["masks"], dtype=bool)[:, chosen],
        "k": k_values,
        "ratio": np.asarray(payload["active_ratios_actual"], dtype=float),
        "sample_index": np.arange(weights.shape[0], dtype=int),
        "max_theta": max_theta,
        "large_scan": max_theta >= 45.0,
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    data = scenarios(args.dataset)
    candidates = (
        ("baseline_smooth_l11p2_dx15", BASE_ROOT / "smooth_blended_l11p2_bar2p0.aedt", BASE_ROOT / "smooth_blended_l11p2_bar2p0.s64p", 15.0, 11.2),
        ("compact_smooth_l10p4_dx16", NEW_ROOT / "smooth_compact_l10p4_bar3p0_dx16p0.aedt", NEW_ROOT / "smooth_compact_l10p4_bar3p0_dx16p0.s64p", 16.0, 10.4),
    )
    hardware_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for name, project, touchstone, dx_mm, length_mm in candidates:
        gate = convergence(project)
        if not gate["converged"] or not touchstone.exists():
            raise RuntimeError(f"{name} is not a converged S64 candidate: {gate}")
        matrix = np.asarray(parse_touchstone(touchstone)["s_parameters"][0], dtype=np.complex128)
        if matrix.shape != (64, 64):
            raise RuntimeError(f"{name} S shape is {matrix.shape}, expected (64, 64)")
        metrics = passive_metrics(matrix)
        values = case_metrics(matrix, data["weights"], data["masks"], float(args.return_loss_db), float(args.significant_power_relative_db))
        hardware_rows.append({
            "candidate": name,
            "dx_mm": dx_mm,
            "dy_mm": 15.0,
            "dipole_length_mm": length_mm,
            "final_delta_s": gate["final_delta_s"],
            "pass_count": gate["pass_count"],
            **metrics,
            "all_active_10db_pass_count": int(np.count_nonzero(values["all_active_pass"])),
            "all_active_10db_pass_rate": float(np.mean(values["all_active_pass"])),
            "total_10db_pass_count": int(np.count_nonzero(values["total_pass"])),
            "total_10db_pass_rate": float(np.mean(values["total_pass"])),
        })
        cases, groups = aggregate_case_metrics(name, values, data)
        case_rows.extend(cases)
        group_rows.extend(groups)
    baseline, compact = hardware_rows
    comparison = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "same_2400_masks_and_weights": True,
        "same_8x8_port_order": True,
        "raw_hfss_s64_only": True,
        "x_nearest_coupling_change_db": float(compact["nearest_x_worst_db"] - baseline["nearest_x_worst_db"]),
        "all_active_pass_rate_change": float(compact["all_active_10db_pass_rate"] - baseline["all_active_10db_pass_rate"]),
        "total_pass_rate_change": float(compact["total_10db_pass_rate"] - baseline["total_10db_pass_rate"]),
        "engineering_promotion": bool(compact["all_active_10db_pass_rate"] > baseline["all_active_10db_pass_rate"] and compact["nearest_x_worst_db"] <= baseline["nearest_x_worst_db"] - 3.0),
        "interpretation_limit": "This compares raw converged HFSS S64 matrices. It does not evaluate PSLL, task isolation, an EEP, or a physical matching network.",
    }
    write_csv(args.out_dir / "hardware_comparison.csv", hardware_rows)
    write_csv(args.out_dir / "active_return_case_metrics.csv", case_rows)
    write_csv(args.out_dir / "active_return_group_summary.csv", group_rows)
    (args.out_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps({"hardware": hardware_rows, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
