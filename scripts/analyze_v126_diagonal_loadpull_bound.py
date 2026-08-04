#!/usr/bin/env python3
"""Compute optimistic diagonal-only load-pull bounds from the v1.26 physical S4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from run_v114_small_cell_broadband_feed import load_stimuli, parse_touchstone
from run_v125_feedpoint_input_impedance import write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "hfss_outputs" / "v126_uniform_2p10_feedpoint_s4_20260805_run02_serial"
DEFAULT_STIMULI = ROOT / "hfss_outputs" / "v1143_small_cell_surface_mesh_20260730_run07"
EPS = 1.0e-15


def physical_matrices(run: Path) -> tuple[np.ndarray, np.ndarray]:
    items = ((9.96, "9p96"), (10.0, "10p00"), (10.04, "10p04"))
    frequencies = []
    matrices = []
    for target, text in items:
        path = run / "2x2_serial" / f"f_{text}ghz" / f"v126_2x2_uniform_2p10_f_{text}ghz.s4p"
        parsed_f, parsed_s = parse_touchstone(path, 4)
        index = int(np.argmin(np.abs(parsed_f - target)))
        frequencies.append(float(parsed_f[index]))
        matrices.append(parsed_s[index])
    return np.asarray(frequencies, dtype=float), np.asarray(matrices, dtype=complex)


def active_metrics(matrix: np.ndarray, sources: np.ndarray, considered: np.ndarray) -> tuple[float, float]:
    reflected = sources @ matrix.T
    active_gammas = []
    total_gammas = []
    for index in range(len(sources)):
        active = considered[index]
        active_gammas.append(
            float(
                np.max(
                    np.abs(reflected[index, active])
                    / np.maximum(np.abs(sources[index, active]), EPS)
                )
            )
        )
        total_gammas.append(
            float(np.sum(np.abs(reflected[index]) ** 2) / np.sum(np.abs(sources[index]) ** 2))
        )
    return (
        float(-20.0 * np.log10(max(active_gammas))),
        float(-10.0 * np.log10(max(total_gammas))),
    )


def optimize_diagonal(
    matrix: np.ndarray,
    sources: np.ndarray,
    considered: np.ndarray,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    count = 1 if mode == "uniform" else 4

    def construct(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diagonal = values[:count] + 1j * values[count:]
        if count == 1:
            diagonal = np.repeat(diagonal, 4)
        return off_diagonal + np.diag(diagonal), diagonal

    def objective(values: np.ndarray) -> float:
        candidate, diagonal = construct(values)
        reflected = sources @ candidate.T
        gamma = max(
            float(
                np.max(
                    np.abs(reflected[index, considered[index]])
                    / np.maximum(np.abs(sources[index, considered[index]]), EPS)
                )
            )
            for index in range(len(sources))
        )
        sigma = float(np.linalg.svd(candidate, compute_uv=False)[0])
        magnitude_penalty = float(np.sum(np.maximum(0.0, np.abs(diagonal) - 10.0 ** (-10.0 / 20.0)) ** 2))
        passivity_penalty = max(0.0, sigma - 1.001) ** 2
        return gamma + 100.0 * magnitude_penalty + 100.0 * passivity_penalty

    result = differential_evolution(
        objective,
        [(-0.316, 0.316)] * (2 * count),
        seed=seed,
        popsize=20 if count == 4 else 30,
        maxiter=500,
        tol=1.0e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    candidate, diagonal = construct(result.x)
    active_rl, total_rl = active_metrics(candidate, sources, considered)
    output: dict[str, Any] = {
        "mode": mode,
        "active_rl_min_db": active_rl,
        "total_rl_min_db": total_rl,
        "passivity_sigma": float(np.linalg.svd(candidate, compute_uv=False)[0]),
        "maximum_sii_magnitude": float(np.max(np.abs(diagonal))),
        "optimizer_success": bool(result.success),
        "optimizer_evaluations": int(result.nfev),
    }
    for port, value in enumerate(diagonal, start=1):
        output[f"s{port}{port}_real"] = float(value.real)
        output[f"s{port}{port}_imag"] = float(value.imag)
        output[f"s{port}{port}_magnitude"] = float(abs(value))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stimuli", type=Path, default=DEFAULT_STIMULI)
    args = parser.parse_args()
    run = args.run.resolve()
    frequencies, matrices = physical_matrices(run)
    metadata, vectors, considered = load_stimuli(args.stimuli.resolve())
    rows: list[dict[str, Any]] = []
    for frequency, matrix in zip(frequencies, matrices):
        indices = [
            index
            for index, item in enumerate(metadata)
            if int(item["side"]) == 2 and abs(float(item["frequency_ghz"]) - frequency) < 1.0e-6
        ]
        sources = vectors[indices, :4]
        active = considered[indices, :4]
        physical_active, physical_total = active_metrics(matrix, sources, active)
        rows.append(
            {
                "frequency_ghz": float(frequency),
                "mode": "physical_uniform_feed_2p10",
                "active_rl_min_db": physical_active,
                "total_rl_min_db": physical_total,
                "passivity_sigma": float(np.linalg.svd(matrix, compute_uv=False)[0]),
                "maximum_sii_magnitude": float(np.max(np.abs(np.diag(matrix)))),
            }
        )
        for mode, seed in (("uniform_ideal_sii", 20260805), ("per_port_ideal_sii", 20260806)):
            optimized = optimize_diagonal(
                matrix,
                sources,
                active,
                "uniform" if mode.startswith("uniform") else "per_port",
                seed,
            )
            optimized["mode"] = mode
            optimized["frequency_ghz"] = float(frequency)
            rows.append(optimized)
    write_csv(run / "diagonal_loadpull_bound.csv", rows)
    uniform = [row for row in rows if row["mode"] == "uniform_ideal_sii"]
    per_port = [row for row in rows if row["mode"] == "per_port_ideal_sii"]
    summary = {
        "evidence_scope": "optimistic frequency-specific diagonal load-pull on measured physical S4; non-diagonal terms frozen",
        "uniform_sii_worst_frequency_upper_bound_active_rl_db": min(float(row["active_rl_min_db"]) for row in uniform),
        "per_port_sii_worst_frequency_upper_bound_active_rl_db": min(float(row["active_rl_min_db"]) for row in per_port),
        "active_rl_gate_db": 11.0,
        "uniform_diagonal_only_gate_pass": all(float(row["active_rl_min_db"]) >= 11.0 for row in uniform),
        "per_port_diagonal_only_gate_pass": all(float(row["active_rl_min_db"]) >= 11.0 for row in per_port),
        "requires_non_diagonal_physical_change": True,
        "decision": "Change radiator/feed input geometry so both Sii and mutual-coupling terms move; do not tune the existing bridge.",
    }
    write_json(run / "diagonal_loadpull_bound.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
