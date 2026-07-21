"""Design a reproducible single-frequency 50-ohm L-match for an EEP operator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EEP_DIR = ROOT / "hfss_outputs" / "multitask_dataset" / "eep_smoke_16port_matched_v2_20260714"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eep-dir", type=Path, default=DEFAULT_EEP_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--return-loss-min-db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def s_to_z(s_matrix: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(s_matrix.shape[0], dtype=np.complex128)
    return z0 * (identity + s_matrix) @ np.linalg.inv(identity - s_matrix)


def z_to_s(z_matrix: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(z_matrix.shape[0], dtype=np.complex128)
    return (z_matrix - z0 * identity) @ np.linalg.inv(z_matrix + z0 * identity)


def transform_impedance(z_antenna: np.ndarray, series_x_ohm: float, shunt_b_siemens: float) -> np.ndarray:
    identity = np.eye(z_antenna.shape[0], dtype=np.complex128)
    shunt_loaded = np.linalg.inv(np.linalg.inv(z_antenna) + 1j * shunt_b_siemens * identity)
    return shunt_loaded + 1j * series_x_ohm * identity


def lmatch_s_parameters(series_x_ohm: float, shunt_b_siemens: float, z0: float) -> tuple[complex, complex, complex, complex]:
    series_z = 1j * series_x_ohm
    shunt_y = 1j * shunt_b_siemens
    a = 1.0 + series_z * shunt_y
    b = series_z
    c = shunt_y
    d = 1.0
    denominator = a + b / z0 + c * z0 + d
    s11 = (a + b / z0 - c * z0 - d) / denominator
    s21 = 2.0 / denominator
    s12 = 2.0 * (a * d - b * c) / denominator
    s22 = (-a + b / z0 - c * z0 + d) / denominator
    return s11, s12, s21, s22


def cascade_matching_network(
    s_antenna: np.ndarray,
    series_x_ohm: float,
    shunt_b_siemens: float,
    z0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = s_antenna.shape[0]
    identity = np.eye(count, dtype=np.complex128)
    s11, s12, s21, s22 = lmatch_s_parameters(series_x_ohm, shunt_b_siemens, z0)
    antenna_wave_map = np.linalg.inv(identity - s22 * s_antenna) @ (s21 * identity)
    composite_s = s11 * identity + s12 * s_antenna @ antenna_wave_map
    network_s = np.asarray([[s11, s12], [s21, s22]], dtype=np.complex128)
    return composite_s, antenna_wave_map, network_s


def db20(values: np.ndarray | float) -> np.ndarray | float:
    result = 20.0 * np.log10(np.maximum(np.abs(values), 1.0e-12))
    return float(result) if np.ndim(result) == 0 else result


def component_description(series_x_ohm: float, shunt_b_siemens: float, frequency_hz: float) -> dict[str, Any]:
    omega = 2.0 * math.pi * frequency_hz
    if series_x_ohm >= 0.0:
        series = {"kind": "inductor", "value_h": series_x_ohm / omega}
    else:
        series = {"kind": "capacitor", "value_f": -1.0 / (omega * series_x_ohm)}
    if shunt_b_siemens >= 0.0:
        shunt = {"kind": "capacitor", "value_f": shunt_b_siemens / omega}
    else:
        shunt = {"kind": "inductor", "value_h": -1.0 / (omega * shunt_b_siemens)}
    return {"topology": "series element followed by shunt element at antenna plane", "series": series, "shunt": shunt}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.eep_dir / "matching_50ohm_lsection")
    out_dir.mkdir(parents=True, exist_ok=True)
    operator_path = args.eep_dir / "eep_operator_16port.npz"
    operator = np.load(operator_path, allow_pickle=False)
    s_antenna = np.asarray(operator["s_parameters"], dtype=np.complex128)
    z0 = float(args.z0_ohm)
    z_antenna = s_to_z(s_antenna, z0)
    identity = np.eye(s_antenna.shape[0], dtype=np.complex128)

    def objective(parameters: np.ndarray) -> float:
        transformed_z = transform_impedance(z_antenna, float(parameters[0]), float(parameters[1]))
        transformed_s = z_to_s(transformed_z, z0)
        reflection = np.abs(np.diag(transformed_s))
        off_diagonal = np.abs(transformed_s - np.diag(np.diag(transformed_s)))
        return float(reflection.max() + 0.15 * reflection.mean() + 0.03 * off_diagonal.max())

    optimization = differential_evolution(
        objective,
        bounds=[(-300.0, 300.0), (-0.05, 0.05)],
        seed=int(args.seed),
        tol=1.0e-9,
        polish=True,
    )
    series_x = float(optimization.x[0])
    shunt_b = float(optimization.x[1])
    transformed_z = transform_impedance(z_antenna, series_x, shunt_b)
    matched_s_impedance = z_to_s(transformed_z, z0)
    matched_s, antenna_wave_map, network_s = cascade_matching_network(s_antenna, series_x, shunt_b, z0)
    cascade_consistency = float(np.max(np.abs(matched_s - matched_s_impedance)))
    passivity_error = float(np.max(np.abs(network_s.conj().T @ network_s - np.eye(2))))

    etheta = np.asarray(operator["etheta"], dtype=np.complex128)
    ephi = np.asarray(operator["ephi"], dtype=np.complex128)
    matched_etheta = antenna_wave_map.T @ etheta
    matched_ephi = antenna_wave_map.T @ ephi
    frequency_ghz = float(operator["frequency_ghz"])
    components = component_description(series_x, shunt_b, frequency_ghz * 1.0e9)

    raw_reflection_db = np.asarray(db20(np.diag(s_antenna)))
    matched_reflection_db = np.asarray(db20(np.diag(matched_s)))
    raw_off = np.abs(s_antenna - np.diag(np.diag(s_antenna)))
    matched_off = np.abs(matched_s - np.diag(np.diag(matched_s)))
    raw_z_diag = np.diag(z_antenna)
    matched_z_diag = np.diag(transformed_z)
    ports = [str(item) for item in operator["port_names"]]
    rows: list[dict[str, Any]] = []
    for index, port in enumerate(ports):
        rows.append(
            {
                "port_name": port,
                "raw_reflection_db": float(raw_reflection_db[index]),
                "matched_reflection_db": float(matched_reflection_db[index]),
                "raw_z_real_ohm": float(raw_z_diag[index].real),
                "raw_z_imag_ohm": float(raw_z_diag[index].imag),
                "matched_z_real_ohm": float(matched_z_diag[index].real),
                "matched_z_imag_ohm": float(matched_z_diag[index].imag),
            }
        )
    metrics_csv = out_dir / "port_matching_metrics.csv"
    write_csv(metrics_csv, rows)

    matched_operator_path = out_dir / "eep_operator_16port_50ohm_lmatch.npz"
    np.savez_compressed(
        matched_operator_path,
        port_names=operator["port_names"],
        element_indices=operator["element_indices"],
        element_ixiy=operator["element_ixiy"],
        theta_deg=operator["theta_deg"],
        phi_deg=operator["phi_deg"],
        etheta=matched_etheta.astype(np.complex64),
        ephi=matched_ephi.astype(np.complex64),
        raw_s_parameters=s_antenna.astype(np.complex64),
        s_parameters=matched_s.astype(np.complex64),
        antenna_incident_wave_map=antenna_wave_map.astype(np.complex64),
        matching_network_s=network_s.astype(np.complex64),
        series_reactance_ohm=np.asarray(series_x),
        shunt_susceptance_siemens=np.asarray(shunt_b),
        reference_impedance_ohm=np.asarray(z0),
        frequency_ghz=operator["frequency_ghz"],
        source_convention=np.asarray("driven_modal_incident_power_watt"),
    )

    matched_return_loss_min = -float(matched_reflection_db.max())
    engineering_smoke_passed = bool(
        matched_return_loss_min >= float(args.return_loss_min_db)
        and cascade_consistency <= 1.0e-8
        and passivity_error <= 1.0e-8
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_operator": str(operator_path),
        "port_count": len(ports),
        "frequency_ghz": frequency_ghz,
        "reference_impedance_ohm": z0,
        "network_scope": "single-frequency identical per-port external lossless L-section smoke model",
        "series_reactance_ohm": series_x,
        "shunt_susceptance_siemens": shunt_b,
        "components": components,
        "raw_return_loss_min_db": -float(raw_reflection_db.max()),
        "matched_return_loss_min_db": matched_return_loss_min,
        "matched_return_loss_median_db": -float(np.median(matched_reflection_db)),
        "return_loss_requirement_db": float(args.return_loss_min_db),
        "raw_mutual_coupling_worst_db": float(db20(raw_off.max())),
        "matched_mutual_coupling_worst_db": float(db20(matched_off.max())),
        "network_passivity_error": passivity_error,
        "cascade_consistency_max_abs": cascade_consistency,
        "engineering_matching_smoke_passed": engineering_smoke_passed,
        "full_array_active_match_validated": False,
        "bandwidth_validated": False,
        "limitations": [
            "Only a representative 16x16 S submatrix is available; omitted ports make this an open subnetwork.",
            "The optimized network is ideal and lossless and has not been embedded in the 3D feed geometry.",
            "Active reflection for 256-port scan and multi-beam excitations is not yet validated.",
        ],
        "outputs": {"metrics_csv": str(metrics_csv), "matched_operator": str(matched_operator_path)},
    }
    summary_path = out_dir / "matching_network_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# 50-ohm L-match smoke assessment",
        "",
        f"- Raw minimum return loss: {summary['raw_return_loss_min_db']:.2f} dB",
        f"- Matched minimum return loss: {summary['matched_return_loss_min_db']:.2f} dB",
        f"- Matched median return loss: {summary['matched_return_loss_median_db']:.2f} dB",
        f"- Raw/matched worst selected-port coupling: {summary['raw_mutual_coupling_worst_db']:.2f} / {summary['matched_mutual_coupling_worst_db']:.2f} dB",
        f"- Series reactance: {series_x:.6f} ohm",
        f"- Shunt susceptance: {shunt_b:.9f} S",
        f"- Components at {frequency_ghz:.3f} GHz: {json.dumps(components, ensure_ascii=True)}",
        f"- Engineering matching smoke passed: {engineering_smoke_passed}",
        "",
        "This result validates only the ideal single-frequency external matching layer. It is not a 256-port active-match or bandwidth sign-off.",
    ]
    (out_dir / "matching_network_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
