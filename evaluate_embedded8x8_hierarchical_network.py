"""Evaluate a gated 8x8 class-modal and inter-tile even/odd feed network.

The 64 antenna ports remain individually addressable.  This is a circuit-domain
upper-bound calculation cascaded with a converged HFSS S64 matrix, not an HFSS
feed-layout result.  It refuses to evaluate an unconverged antenna export.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_full_s256p_active_return import parse_touchstone
from design_modal_subarray_network import (
    aggregate_case_metrics,
    case_metrics,
    dct_matrix,
    optimize_scalar_mode_matches,
    passive_metrics,
    write_touchstone,
)
from design_port_class_matching import compose_nonuniform_network


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "hfss_outputs" / "embedded8x8_modal_smoke_20260716_run04"
DEFAULT_S64 = RUN_ROOT / "smooth_blended_l11p2_bar2p0" / "smooth_blended_l11p2_bar2p0.s64p"
DEFAULT_PROJECT = RUN_ROOT / "smooth_blended_l11p2_bar2p0" / "smooth_blended_l11p2_bar2p0.aedt"
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_OUT = RUN_ROOT / "hierarchical_modal_network"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s64", type=Path, default=DEFAULT_S64)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260716)
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


def convergence(project: Path) -> dict[str, Any]:
    profiles = list(project.with_suffix(".aedtresults").rglob("*.profile"))
    if not profiles:
        return {"profile": "", "pass_count": 0, "final_delta_s": float("nan"), "converged": False}
    profile = max(profiles, key=lambda item: item.stat().st_mtime)
    text = profile.read_text(encoding="utf-8", errors="replace")
    values = [float(value) for value in re.findall(r"Max Mag\. Delta S\\',\s*([0-9.eE+-]+)", text)]
    return {
        "profile": str(profile),
        "pass_count": len(re.findall(r"Name='Adaptive Pass \d+'", text)),
        "final_delta_s": values[-1] if values else float("nan"),
        "converged": bool(values and values[-1] <= 0.05 and "did not converge" not in text),
    }


def tile_order() -> tuple[list[int], list[str], list[list[int]]]:
    # Tile order makes the secondary 4-port transform operate across tiles.
    tiles = ((0, 0, "corner"), (0, 4, "edge"), (4, 0, "edge"), (4, 4, "interior"))
    order: list[int] = []
    names: list[str] = []
    blocks: list[list[int]] = []
    for x0, y0, name in tiles:
        block = [(x0 + dx) * 8 + y0 + dy for dx in range(4) for dy in range(4)]
        blocks.append(block)
        order.extend(block)
        names.append(name)
    return order, names, blocks


def local_transform() -> np.ndarray:
    local = np.kron(dct_matrix(4), dct_matrix(4))
    transform = np.zeros((64, 64), dtype=np.float64)
    for tile in range(4):
        start = tile * 16
        transform[start : start + 16, start : start + 16] = local
    return transform


def intertile_transform() -> np.ndarray:
    # Each local mode is mixed only with the same local mode in the other tiles.
    h4 = np.kron(dct_matrix(2), dct_matrix(2))
    transform = np.zeros((64, 64), dtype=np.float64)
    for mode in range(16):
        indices = [tile * 16 + mode for tile in range(4)]
        transform[np.ix_(indices, indices)] = h4
    return transform


def load_scenarios(dataset_path: Path, order: list[int]) -> dict[str, np.ndarray]:
    dataset = np.load(dataset_path, allow_pickle=False)
    ixiy = np.asarray(dataset["element_ixiy"], dtype=int)
    central_native = [(ix, iy) for ix in range(4, 12) for iy in range(4, 12)]
    lookup = {tuple(point): index for index, point in enumerate(ixiy.tolist())}
    central_indices = np.asarray([lookup[point] for point in central_native], dtype=int)
    selected = central_indices[np.asarray(order, dtype=int)]
    weights_ri = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float64)[:, selected]
    weights = weights_ri[:, :, 0] + 1j * weights_ri[:, :, 1]
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1.0e-15)
    targets = np.asarray(dataset["targets_deg"], dtype=np.float64)
    k_values = np.asarray(dataset["k_values"], dtype=int)
    max_theta = np.asarray([np.nanmax(targets[i, : k_values[i], 0]) for i in range(len(k_values))])
    return {
        "weights": weights,
        "masks": np.asarray(dataset["masks"], dtype=bool)[:, selected],
        "k": k_values,
        "ratio": np.asarray(dataset["active_ratios_actual"], dtype=np.float64),
        "sample_index": np.arange(weights.shape[0], dtype=int),
        "max_theta": max_theta,
        "large_scan": max_theta >= 45.0,
    }


def class_match_parameters(s_tile: np.ndarray, class_names: list[str], z0: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    local = np.kron(dct_matrix(4), dct_matrix(4))
    parameters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(("corner", "edge", "interior")):
        tiles = [index for index, value in enumerate(class_names) if value == class_name]
        representative = np.mean([s_tile[i * 16 : (i + 1) * 16, i * 16 : (i + 1) * 16] for i in tiles], axis=0)
        modal = local.T @ representative @ local
        x_value, b_value = optimize_scalar_mode_matches(modal, z0, seed + 100 * class_index)
        parameters[class_name] = (x_value, b_value)
        for mode in range(16):
            rows.append({"stage": "class_4x4", "class": class_name, "mode": mode, "series_reactance_ohm": float(x_value[mode]), "shunt_susceptance_siemens": float(b_value[mode])})
    series_x = np.concatenate([parameters[name][0] for name in class_names])
    shunt_b = np.concatenate([parameters[name][1] for name in class_names])
    return series_x, shunt_b, rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    gate = convergence(args.project)
    gate["s64_exists"] = args.s64.exists() and args.s64.stat().st_size > 1000 if args.s64.exists() else False
    if not gate["converged"] or not gate["s64_exists"]:
        summary = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hfss_delta_s_gate": 0.05,
            "convergence": gate,
            "decision": "block_hierarchical_network_evaluation_until_converged_s64",
            "interpretation": "No circuit cascade, active-RL metric, or engineering-value conclusion was computed.",
        }
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    parsed = parse_touchstone(args.s64)
    s_native = np.asarray(parsed["s_parameters"][0], dtype=np.complex128)
    if s_native.shape != (64, 64):
        raise ValueError(f"Expected S64, found {s_native.shape}")
    z0 = float(parsed["reference_impedance_ohm"])
    order, class_names, _ = tile_order()
    inverse = np.argsort(np.asarray(order))
    s_tile = s_native[np.ix_(order, order)]
    h_local = local_transform()
    x_local, b_local, component_rows = class_match_parameters(s_tile, class_names, z0, int(args.seed))
    s_local_modal, _, _ = compose_nonuniform_network(h_local.T @ s_tile @ h_local, x_local, b_local, z0)
    h_intertile = intertile_transform()
    s_second_basis = h_intertile.T @ s_local_modal @ h_intertile
    # One shared secondary set per same-mode, cross-tile even/odd branch.
    x_second, b_second = optimize_scalar_mode_matches(s_second_basis, z0, int(args.seed) + 1000)
    s_second, _, _ = compose_nonuniform_network(s_second_basis, x_second, b_second, z0)
    for mode in range(64):
        component_rows.append({"stage": "intertile_4x4_even_odd", "class": "all_tiles", "mode": mode, "series_reactance_ohm": float(x_second[mode]), "shunt_susceptance_siemens": float(b_second[mode])})
    s_hier_tile = h_local @ h_intertile @ s_second @ h_intertile.T @ h_local.T
    s_hier_native = s_hier_tile[np.ix_(inverse, inverse)]
    scenarios = load_scenarios(args.dataset, order)
    raw_cases = case_metrics(s_tile, scenarios["weights"], scenarios["masks"], float(args.return_loss_db), float(args.significant_power_relative_db))
    hierarchical_cases = case_metrics(s_hier_tile, scenarios["weights"], scenarios["masks"], float(args.return_loss_db), float(args.significant_power_relative_db))
    case_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for name, values in (("raw_converged_hfss_s64", raw_cases), ("hierarchical_ideal_circuit_cascade", hierarchical_cases)):
        cases, groups = aggregate_case_metrics(name, values, scenarios)
        case_rows.extend(cases)
        group_rows.extend(groups)
    raw_passive = passive_metrics(s_native)
    hierarchical_passive = passive_metrics(s_hier_native)
    all_active_rate = float(np.mean(hierarchical_cases["all_active_pass"]))
    total_rate = float(np.mean(hierarchical_cases["total_pass"]))
    nonzero_k = all(bool(np.any(hierarchical_cases["all_active_pass"] & hierarchical_cases["total_pass"] & (scenarios["k"] == value))) for value in sorted(set(scenarios["k"].tolist())))
    value_gate = bool(hierarchical_passive["passive_rl_min_db"] >= float(args.return_loss_db) and all_active_rate > 0.0 and total_rate > 0.0 and nonzero_k)
    write_csv(args.out_dir / "network_mode_components.csv", component_rows)
    write_csv(args.out_dir / "active_return_case_metrics.csv", case_rows)
    write_csv(args.out_dir / "active_return_group_summary.csv", group_rows)
    write_touchstone(args.out_dir / "hierarchical_ideal_circuit_cascade.s64p", s_hier_native, float(parsed["frequency_hz"][0]), z0)
    np.savez_compressed(args.out_dir / "hierarchical_network.npz", raw_s_native=s_native.astype(np.complex64), hierarchical_s_native=s_hier_native.astype(np.complex64), local_transform=h_local, intertile_transform=h_intertile, tile_order=np.asarray(order), class_names=np.asarray(class_names), local_series_x=x_local, local_shunt_b=b_local, secondary_series_x=x_second, secondary_shunt_b=b_second)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hfss_delta_s_gate": 0.05,
        "convergence": gate,
        "source_is_converged_hfss_s64": True,
        "raw_passive_metrics": raw_passive,
        "hierarchical_passive_metrics": hierarchical_passive,
        "hierarchical_all_active_10db_pass_rate": all_active_rate,
        "hierarchical_total_10db_pass_rate": total_rate,
        "nonzero_combined_pass_for_each_k": nonzero_k,
        "allow_physical_feed_layout_smoke": value_gate,
        "decision": "allow_physical_feed_layout_smoke" if value_gate else "block_physical_feed_layout_due_to_active_rl_gate",
        "interpretation_limit": "This is a cascade of converged antenna S64 with ideal lossless modal hybrids and lumped L matches. It is not a full-wave feed-network validation.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
