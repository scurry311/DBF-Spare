#!/usr/bin/env python3
"""Fit the v1.17 POST route phase and select an equal-length v1.18 route."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from design_v115_grounded_modal_network import terminate_network
from run_v114_small_cell_broadband_feed import load_stimuli, parse_touchstone
from run_v115_physical_modal_feed_fixture import touchstone_port_names


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v117_integrated_2x2_smoke.json"
DEFAULT_OUTPUT = ROOT / "hfss_outputs" / "v118_equalized_post_transition_20260730_run01"
EPS = 1.0e-15


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reordered_network(path: Path, desired: list[str], nports: int) -> tuple[np.ndarray, np.ndarray]:
    frequencies, network = parse_touchstone(path, nports)
    names = touchstone_port_names(path)
    order = [names.index(name) for name in desired]
    return frequencies, network[:, order][:, :, order]


def old_route_lengths(config: dict[str, Any], antenna: dict[str, Any]) -> np.ndarray:
    physical = antenna["physical_topology"]
    candidate = antenna["one_by_one_candidates"][0]
    network = config["feed_network"]
    spacing = float(physical["spacing_mm"])
    patch_l = float(physical["patch_length_mm"])
    feed_inset = float(candidate["feed_inset_from_edge_mm"])
    post_x = float(network["x_offset_mm"]) + float(network["post_reference_x_local_mm"])
    y_channels = np.asarray(network["channel_y_mm_by_port"], dtype=float)
    lengths = []
    for port in range(4):
        ix, iy = divmod(port, 2)
        feed_x = (ix - 0.5) * spacing
        feed_y = (iy - 0.5) * spacing - patch_l / 2.0 + feed_inset
        lengths.append(abs(feed_x - post_x) + abs(feed_y - y_channels[port]))
    return np.asarray(lengths)


def routed_load(antenna_s4: np.ndarray, phases: np.ndarray) -> np.ndarray:
    delay = np.diag(np.exp(-1j * phases))
    return delay @ antenna_s4 @ delay


def cascaded(feed_s8: np.ndarray, antenna_s4: np.ndarray, phases: np.ndarray) -> np.ndarray:
    return terminate_network(feed_s8, routed_load(antenna_s4, phases))[0]


def active_rl_min(
    frequencies: np.ndarray,
    s4: np.ndarray,
    stimulus_root: Path,
) -> float:
    rows, vectors, considered = load_stimuli(stimulus_root)
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    minimum = math.inf
    for index, frequency in enumerate(frequencies):
        selected = np.asarray(
            [abs(float(row["frequency_ghz"]) - float(frequency)) <= 1.0e-9 for row in rows]
        )
        sources = vectors[selected].T
        active = considered[selected].T
        reflected = s4[index] @ sources
        gamma = np.where(active, np.abs(reflected) / np.maximum(np.abs(sources), EPS), 0.0)
        rl = -20.0 * np.log10(np.maximum(np.max(gamma, axis=0), EPS))
        minimum = min(minimum, float(np.min(rl)))
    return minimum


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = read_json(args.config)
    antenna = read_json(resolve(config["trusted_antenna_protocol"]))
    args.output.mkdir(parents=True, exist_ok=True)
    frequencies, integrated = reordered_network(
        resolve(config["output_directory"]) / "integrated_2x2_direct01" / "v117_integrated_2x2_direct01.s4p",
        [f"PRE_{port}" for port in range(4)],
        4,
    )
    antenna_f, antenna_s4 = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    feed_f, feed_s8 = reordered_network(
        resolve(config["validated_feed_s8"]),
        [f"PRE_{port}" for port in range(4)] + [f"POST_{port}" for port in range(4)],
        8,
    )
    if not np.allclose(frequencies, antenna_f) or not np.allclose(frequencies, feed_f):
        raise RuntimeError("Frequency grids differ")
    lengths = old_route_lengths(config, antenna)
    fit_rows = []
    best = None
    for effective_er in np.linspace(1.55, 2.15, 61):
        for extra_length in np.linspace(-3.0, 3.0, 121):
            predicted = []
            for index, frequency in enumerate(frequencies):
                beta = 2.0 * math.pi * float(frequency) * math.sqrt(effective_er) / 299.792458
                phases = beta * (lengths + extra_length)
                predicted.append(cascaded(feed_s8[index], antenna_s4[index], phases))
            predicted_array = np.stack(predicted)
            error = np.abs(predicted_array - integrated)
            row = {
                "effective_er": float(effective_er),
                "common_extra_length_mm": float(extra_length),
                "fit_rmse": float(np.sqrt(np.mean(error**2))),
                "fit_max_abs_delta_s": float(np.max(error)),
            }
            if best is None or row["fit_rmse"] < best["fit_rmse"]:
                best = row
    assert best is not None
    lambda_g = 299.792458 / (10.0 * math.sqrt(float(best["effective_er"])))
    fitted_target = lambda_g - float(best["common_extra_length_mm"])
    baseline = np.stack(
        [terminate_network(feed_s8[index], antenna_s4[index])[0] for index in range(3)]
    )
    for target in np.arange(fitted_target - 0.6, fitted_target + 0.6001, 0.05):
        predicted = []
        for index, frequency in enumerate(frequencies):
            beta = 2.0 * math.pi * float(frequency) * math.sqrt(float(best["effective_er"])) / 299.792458
            phase = beta * (float(target) + float(best["common_extra_length_mm"]))
            predicted.append(cascaded(feed_s8[index], antenna_s4[index], np.full(4, phase)))
        predicted_array = np.stack(predicted)
        fit_rows.append(
            {
                "common_centerline_length_mm": float(target),
                "proxy_max_abs_delta_s": float(np.max(np.abs(predicted_array - baseline))),
                "proxy_active_rl_min_db": active_rl_min(
                    frequencies, predicted_array, resolve(config["trusted_stimulus_root"])
                ),
            }
        )
    selected = max(
        fit_rows,
        key=lambda row: (
            min(row["proxy_active_rl_min_db"] - 11.0, 0.05 - row["proxy_max_abs_delta_s"]),
            row["proxy_active_rl_min_db"],
        ),
    )
    summary = {
        "source_integrated_s4": str(
            (resolve(config["output_directory"]) / "integrated_2x2_direct01" / "v117_integrated_2x2_direct01.s4p").resolve()
        ),
        "frozen_v116_components": config["feed_network"],
        "old_route_centerline_lengths_mm": lengths.tolist(),
        "fit": best,
        "guided_wavelength_at_10ghz_mm": lambda_g,
        "fitted_repeating_length_mm": fitted_target,
        "selected_proxy_candidate": selected,
        "proxy_is_not_hfss_evidence": True,
    }
    write_csv(args.output / "proxy_candidate_screen.csv", fit_rows)
    (args.output / "proxy_fit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
