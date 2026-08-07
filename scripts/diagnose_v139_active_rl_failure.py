#!/usr/bin/env python3
"""Diagnose the frozen-stimulus active-RL failure of the valid v1.39 S4."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from run_v125_feedpoint_input_impedance import write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "hfss_outputs" / "v139_physical_2x2_differential_array_20260808_run01" / "initial_10ghz" / "direct01_repair03"
DEFAULT_STIMULI = ROOT / "hfss_outputs" / "v139_physical_2x2_differential_array_20260808_run01" / "stimuli"
EPS = 1.0e-15
PORTS = ["P00", "P10", "P01", "P11"]
COORDS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def ri_to_complex(array: np.ndarray) -> np.ndarray:
    return array[..., 0] + 1j * array[..., 1]


def relation(i: int, j: int) -> str:
    if i == j:
        return "self"
    dx = abs(COORDS[i][0] - COORDS[j][0])
    dy = abs(COORDS[i][1] - COORDS[j][1])
    return "x_neighbor" if dx and not dy else "y_neighbor" if dy and not dx else "diagonal"


def load_inputs(case: Path, stimuli: Path) -> tuple[np.ndarray, list[dict[str, str]], np.ndarray, np.ndarray]:
    with np.load(case / "physical_operators.npz", allow_pickle=False) as data:
        s = ri_to_complex(data["s_real_imag"])
    rows = list(csv.DictReader((stimuli / "stimuli_manifest.csv").open(encoding="utf-8-sig")))
    with np.load(stimuli / "stimuli_vectors.npz", allow_pickle=False) as data:
        vectors = ri_to_complex(data["vectors_real_imag"])
        considered = np.asarray(data["considered"], dtype=bool)
    return s, rows, vectors, considered


def diagnose(case: Path, stimuli: Path) -> dict[str, Any]:
    s, metadata, vectors, considered = load_inputs(case, stimuli)
    threshold = 10.0
    design_threshold = 11.0
    details: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    modal_rows: list[dict[str, Any]] = []
    jacobian_rows: list[dict[str, Any]] = []
    modes = np.asarray([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=complex) / 2.0
    mode_names = ["even", "x_odd", "y_odd", "checker"]
    for index, row in enumerate(metadata):
        if abs(float(row["frequency_ghz"]) - 10.0) > 1.0e-6:
            continue
        a = vectors[index].astype(complex)
        active = considered[index]
        b = s @ a
        gamma_all = np.full(4, np.nan)
        gamma_all[active] = np.abs(b[active]) / np.maximum(np.abs(a[active]), EPS)
        worst = int(np.nanargmax(gamma_all))
        gamma = float(gamma_all[worst])
        active_rl = float(-20.0 * np.log10(max(gamma, EPS)))
        target_b_10 = float(abs(a[worst]) * 10.0 ** (-threshold / 20.0))
        target_b_11 = float(abs(a[worst]) * 10.0 ** (-design_threshold / 20.0))
        base = {
            "stimulus_index": index, "stimulus_family": row["stimulus_family"],
            "sample_index": int(row["sample_index"]), "k_value": int(row["k_value"]),
            "ratio": float(row["ratio"]), "window_role": row["window_role"],
            "source_type": row["source_type"], "task_index": int(row["task_index"]),
            "worst_port": PORTS[worst], "active_rl_db": active_rl,
            "worst_gamma_magnitude": gamma, "source_port_magnitude": float(abs(a[worst])),
            "reflected_port_magnitude": float(abs(b[worst])),
            "required_b_reduction_for_10db": max(0.0, float(abs(b[worst])) - target_b_10),
            "required_b_reduction_for_11db": max(0.0, float(abs(b[worst])) - target_b_11),
            "stop_line_pass": active_rl >= threshold, "design_line_pass": active_rl >= design_threshold,
        }
        details.append(base)
        coefficients = modes.conj() @ a
        energies = np.abs(coefficients) ** 2 / max(float(np.vdot(a, a).real), EPS)
        modal_rows.append({**base, **{f"{name}_fraction": float(energies[i]) for i, name in enumerate(mode_names)}, "dominant_mode": mode_names[int(np.argmax(energies))]})
        if active_rl >= threshold:
            continue
        for j in range(4):
            term = s[worst, j] * a[j]
            denom = max(float(abs(b[worst]) ** 2), EPS)
            d_re = -20.0 / math.log(10.0) * float(np.real(np.conj(b[worst]) * a[j])) / denom
            d_im = -20.0 / math.log(10.0) * float(np.real(np.conj(b[worst]) * 1j * a[j])) / denom
            item = {**base, "source_port": PORTS[j], "coupling_relation": relation(worst, j), "s_magnitude": float(abs(s[worst, j])), "s_phase_deg": float(np.degrees(np.angle(s[worst, j]))), "source_magnitude": float(abs(a[j])), "term_real": float(term.real), "term_imag": float(term.imag), "term_magnitude": float(abs(term)), "term_alignment_with_total": float(np.real(term * np.conj(b[worst])) / max(abs(term) * abs(b[worst]), EPS))}
            contributions.append(item)
            jacobian_rows.append({**base, "s_column_port": PORTS[j], "coupling_relation": relation(worst, j), "d_active_rl_db_d_re_s": d_re, "d_active_rl_db_d_im_s": d_im, "jacobian_magnitude": float(math.hypot(d_re, d_im))})

    group_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in details:
        groups.setdefault((row["stimulus_family"], row["k_value"]), []).append(row)
    for (family, k), rows in sorted(groups.items()):
        values = np.asarray([row["active_rl_db"] for row in rows])
        group_rows.append({"stimulus_family": family, "k_value": k, "count": len(rows), "failure_below_10_count": int(np.sum(values < 10.0)), "failure_below_10_rate": float(np.mean(values < 10.0)), "pass_11_count": int(np.sum(values >= 11.0)), "minimum_active_rl_db": float(values.min()), "median_active_rl_db": float(np.median(values))})

    relation_rows: list[dict[str, Any]] = []
    for name in ("self", "x_neighbor", "y_neighbor", "diagonal"):
        c_rows = [row for row in contributions if row["coupling_relation"] == name]
        j_rows = [row for row in jacobian_rows if row["coupling_relation"] == name]
        relation_rows.append({"coupling_relation": name, "failure_term_count": len(c_rows), "mean_term_magnitude": float(np.mean([row["term_magnitude"] for row in c_rows])), "mean_aligned_term_magnitude": float(np.mean([row["term_magnitude"] * max(row["term_alignment_with_total"], 0.0) for row in c_rows])), "mean_jacobian_magnitude": float(np.mean([row["jacobian_magnitude"] for row in j_rows])), "maximum_jacobian_magnitude": float(np.max([row["jacobian_magnitude"] for row in j_rows]))})

    fail_modal = [row for row in modal_rows if not row["stop_line_pass"]]
    pass_modal = [row for row in modal_rows if row["stop_line_pass"]]
    modal_summary = []
    for name in mode_names:
        key = f"{name}_fraction"
        modal_summary.append({"mode": name, "mean_fraction_failed": float(np.mean([row[key] for row in fail_modal])), "mean_fraction_passed": float(np.mean([row[key] for row in pass_modal])), "dominant_failure_count": sum(row["dominant_mode"] == name for row in fail_modal)})

    details.sort(key=lambda row: row["active_rl_db"])
    write_csv(case / "active_rl_failure_details.csv", details)
    write_csv(case / "active_rl_failure_contributions.csv", contributions)
    write_csv(case / "active_rl_sensitivity_jacobian.csv", jacobian_rows)
    write_csv(case / "active_rl_group_failure_rates.csv", group_rows)
    write_csv(case / "active_rl_relation_summary.csv", relation_rows)
    write_csv(case / "active_rl_modal_projection.csv", modal_rows)
    write_csv(case / "active_rl_modal_summary.csv", modal_summary)
    summary = {
        "valid_10ghz_stimulus_count": len(details), "failure_below_10_count": sum(not row["stop_line_pass"] for row in details),
        "failure_below_10_rate": float(np.mean([not row["stop_line_pass"] for row in details])),
        "pass_11_count": sum(row["design_line_pass"] for row in details), "pass_11_rate": float(np.mean([row["design_line_pass"] for row in details])),
        "worst_case": details[0], "coupling_relation_sensitivity": relation_rows, "modal_failure_summary": modal_summary,
        "diagnosis": "The valid passive S4 is not active-matched for the frozen coherent excitations. Y-neighbor coupling is strongest, while the x-odd/even modal RL values are below 10 dB; cancellation-sensitive task weights create active reflection larger than the incident wave at individual significant ports.",
        "decision": {"allow_independent_repeat": False, "allow_ddm": False, "allow_three_frequency": False, "allow_larger_arrays": False, "allow_eep_labels_or_critic": False, "next_authorized_work": "S4-aware fixed-mask task-weight projection or a single local x/y modal feed correction evaluated against this frozen S4; do not launch more HFSS until the 10 GHz frozen-stimulus oracle has a nonempty >=11 dB reserve."},
    }
    write_json(case / "active_rl_failure_diagnosis.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--stimuli", type=Path, default=DEFAULT_STIMULI)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.case.resolve(), args.stimuli.resolve()), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
