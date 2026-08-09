#!/usr/bin/env python3
"""Find the nearest symmetric modal S4 satisfying frozen active-RL stimuli."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".python_deps").exists():
    sys.path.insert(0, str(ROOT / ".python_deps"))
sys.path.insert(0, str(ROOT / "scripts"))

import cvxpy as cp

from run_v16_robust_drift_oracle import ri_to_complex


MODES = np.asarray(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
    dtype=float,
).T / 2.0
MODE_NAMES = ("even", "x_odd", "y_odd", "checker")


def complex_ri(value: np.ndarray) -> np.ndarray:
    return np.stack((value.real, value.imag), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-operator",
        type=Path,
        default=ROOT / "hfss_outputs/v139_physical_2x2_differential_array_20260808_run01/initial_10ghz/direct01_repair03/physical_operators.npz",
    )
    parser.add_argument(
        "--stimuli",
        type=Path,
        default=ROOT / "hfss_outputs/v139_physical_2x2_differential_array_20260808_run01/stimuli/stimuli_vectors.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "hfss_outputs/v143_y_balanced_element_input_20260809_run02/target_synthesis",
    )
    parser.add_argument("--design-active-rl-db", type=float, default=11.0)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.physical_operator.resolve(), allow_pickle=False) as source:
        current_s4 = ri_to_complex(source["s_real_imag"]).astype(np.complex128)
    with np.load(args.stimuli.resolve(), allow_pickle=False) as source:
        commands = ri_to_complex(source["vectors_real_imag"]).astype(np.complex128)
        considered = np.asarray(source["considered"], dtype=bool)
    modal = MODES.T @ current_s4 @ MODES
    current_gamma = np.diag(modal)
    gamma = cp.Variable(4, complex=True)
    target_s4_expression = MODES @ cp.diag(gamma) @ MODES.T
    rho = 10.0 ** (-float(args.design_active_rl_db) / 20.0)
    constraints: list[object] = [cp.abs(gamma) <= 0.5]
    for command, selected in zip(commands, considered):
        reflected = target_s4_expression @ command
        indices = np.flatnonzero(selected)
        constraints.extend(
            [
                cp.abs(reflected[indices]) <= rho * np.abs(command[indices]),
                cp.norm(reflected, 2) <= rho * float(np.linalg.norm(command)),
            ]
        )
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(cp.abs(gamma - current_gamma))),
        constraints,
    )
    problem.solve(
        solver="CLARABEL",
        max_iter=500,
        tol_gap_abs=1.0e-8,
        tol_feas=1.0e-8,
        verbose=False,
    )
    if gamma.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Target synthesis failed: {problem.status}")
    target_gamma = np.asarray(gamma.value, dtype=np.complex128)
    target_s4 = MODES @ np.diag(target_gamma) @ MODES.T
    current_impedance = 50.0 * (1.0 + current_gamma) / (1.0 - current_gamma)
    target_impedance = 50.0 * (1.0 + target_gamma) / (1.0 - target_gamma)
    rows = []
    for index, name in enumerate(MODE_NAMES):
        rows.append(
            {
                "mode": name,
                "current_gamma_real": float(current_gamma[index].real),
                "current_gamma_imag": float(current_gamma[index].imag),
                "current_rl_db": float(-20.0 * np.log10(max(abs(current_gamma[index]), 1.0e-15))),
                "current_resistance_ohm": float(current_impedance[index].real),
                "current_reactance_ohm": float(current_impedance[index].imag),
                "target_gamma_real": float(target_gamma[index].real),
                "target_gamma_imag": float(target_gamma[index].imag),
                "target_rl_db": float(-20.0 * np.log10(max(abs(target_gamma[index]), 1.0e-15))),
                "target_resistance_ohm": float(target_impedance[index].real),
                "target_reactance_ohm": float(target_impedance[index].imag),
            }
        )
    import csv

    with (out / "modal_impedance_target.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out / "target_s4.npz",
        current_s4_real_imag=complex_ri(current_s4),
        target_s4_real_imag=complex_ri(target_s4),
        current_modal_gamma_real_imag=complex_ri(current_gamma),
        target_modal_gamma_real_imag=complex_ri(target_gamma),
        mode_names=np.asarray(MODE_NAMES),
    )
    summary = {
        "status": problem.status,
        "stimulus_count": int(commands.shape[0]),
        "design_active_rl_db": float(args.design_active_rl_db),
        "current_modal_offdiagonal_max_abs": float(
            np.max(np.abs(modal - np.diag(current_gamma)))
        ),
        "modal_gamma_change_l2": float(np.linalg.norm(target_gamma - current_gamma)),
        "target_modal_impedance_ohm": {
            name: [float(value.real), float(value.imag)]
            for name, value in zip(MODE_NAMES, target_impedance)
        },
        "interpretation": "The target is a geometry/input-impedance specification. It is not HFSS evidence and does not authorize array scaling.",
    }
    (out / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
