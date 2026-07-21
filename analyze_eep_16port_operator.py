"""Build and validate a complex 16-port EEP operator from HFSS exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EEP_DIR = ROOT / "hfss_outputs" / "multitask_dataset" / "eep_smoke_16port_20260714"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eep-dir", type=Path, default=DEFAULT_EEP_DIR)
    parser.add_argument("--reciprocity-max", type=float, default=1.0e-5)
    parser.add_argument("--return-loss-min-db", type=float, default=10.0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_complex_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig") as handle:
        header = [item.strip() for item in handle.readline().strip().split(",")]
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError(f"Unexpected EEP field shape in {path}: {values.shape}")
    lower = [item.lower() for item in header]
    real_imag = any("re(retheta)" in item for item in lower) or any("re(rETheta)" in item for item in header)
    if real_imag:
        etheta = values[:, 2] + 1j * values[:, 3]
        ephi = values[:, 4] + 1j * values[:, 5]
    else:
        etheta = values[:, 2] * np.exp(1j * np.deg2rad(values[:, 3]))
        ephi = values[:, 4] * np.exp(1j * np.deg2rad(values[:, 5]))
    angles = values[:, :2].astype(np.float32)
    return angles, etheta.astype(np.complex64), ephi.astype(np.complex64)


def read_s_submatrix(path: Path, ports: list[str]) -> tuple[float, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        row = next(reader)
    frequency_ghz = float(row[0])
    s_matrix = np.full((len(ports), len(ports)), np.nan + 1j * np.nan, dtype=np.complex128)
    port_col = {port: index for index, port in enumerate(ports)}
    pending: dict[tuple[int, int], dict[str, float]] = {}
    pattern = re.compile(r"(re|im)\(S\(([^,]+),([^\)]+)\)\)", re.IGNORECASE)
    for column, name in enumerate(header[1:], 1):
        match = pattern.search(name)
        if not match:
            continue
        part, left, right = match.groups()
        if left not in port_col or right not in port_col:
            continue
        key = (port_col[left], port_col[right])
        pending.setdefault(key, {})[part.lower()] = float(row[column])
    for key, parts in pending.items():
        if "re" in parts and "im" in parts:
            s_matrix[key] = parts["re"] + 1j * parts["im"]
    return frequency_ghz, s_matrix.astype(np.complex64)


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.eep_dir / "representative_ports.csv")
    ports = [row["port_name"] for row in manifest]
    element_indices = np.asarray([int(row["element_index"]) for row in manifest], dtype=np.int16)
    ixiy = np.asarray([[int(row["ix"]), int(row["iy"])] for row in manifest], dtype=np.int8)
    all_theta: np.ndarray | None = None
    all_phi: np.ndarray | None = None
    etheta_rows: list[np.ndarray] = []
    ephi_rows: list[np.ndarray] = []
    failures: list[str] = []
    for port in ports:
        path = args.eep_dir / f"eep_{port.lower()}_complex.csv"
        if not path.exists():
            failures.append(f"missing:{path.name}")
            continue
        try:
            angles, etheta, ephi = read_complex_field(path)
        except (OSError, ValueError) as error:
            failures.append(f"invalid:{path.name}:{error}")
            continue
        if all_theta is None:
            all_theta = angles[:, 0]
            all_phi = angles[:, 1]
        elif not (np.array_equal(all_theta, angles[:, 0]) and np.array_equal(all_phi, angles[:, 1])):
            failures.append(f"grid_mismatch:{path.name}")
            continue
        etheta_rows.append(etheta)
        ephi_rows.append(ephi)
    field_complete = len(etheta_rows) == len(ports) and not failures
    s_path = args.eep_dir / "s_parameter_submatrix_complex.csv"
    s_complete = s_path.exists()
    frequency_ghz = float("nan")
    s_matrix = np.empty((0, 0), dtype=np.complex64)
    if s_complete:
        try:
            frequency_ghz, s_matrix = read_s_submatrix(s_path, ports)
            s_complete = bool(s_matrix.shape == (len(ports), len(ports)) and np.all(np.isfinite(s_matrix)))
        except (OSError, ValueError, StopIteration) as error:
            failures.append(f"invalid_s:{error}")
            s_complete = False
    if field_complete:
        etheta_operator = np.stack(etheta_rows).astype(np.complex64)
        ephi_operator = np.stack(ephi_rows).astype(np.complex64)
        field_finite = bool(np.all(np.isfinite(etheta_operator)) and np.all(np.isfinite(ephi_operator)))
        field_nonzero = bool(np.max(np.abs(etheta_operator)) > 0.0 or np.max(np.abs(ephi_operator)) > 0.0)
        field_norms = np.sqrt(np.sum(np.abs(etheta_operator) ** 2 + np.abs(ephi_operator) ** 2, axis=1))
        norm_ratio = float(field_norms.max() / max(float(field_norms.min()), 1.0e-12))
        rng = np.random.default_rng(20260714)
        trial_weights = rng.normal(size=len(ports)) + 1j * rng.normal(size=len(ports))
        trial_weights = trial_weights / max(float(np.linalg.norm(trial_weights)), 1.0e-12)
        combined_theta = trial_weights @ etheta_operator
        combined_phi = trial_weights @ ephi_operator
        combination_finite = bool(np.all(np.isfinite(combined_theta)) and np.all(np.isfinite(combined_phi)))
        np.savez_compressed(
            args.eep_dir / "eep_operator_16port.npz",
            port_names=np.asarray(ports),
            element_indices=element_indices,
            element_ixiy=ixiy,
            theta_deg=all_theta,
            phi_deg=all_phi,
            etheta=etheta_operator,
            ephi=ephi_operator,
            s_parameters=s_matrix,
            frequency_ghz=np.asarray(frequency_ghz, dtype=np.float32),
        )
    else:
        field_finite = False
        field_nonzero = False
        norm_ratio = float("nan")
        combination_finite = False
    if s_complete:
        reciprocity = np.abs(s_matrix - s_matrix.T)
        reciprocity_max = float(reciprocity.max())
        reciprocity_rms = float(np.sqrt(np.mean(reciprocity**2)))
        reflection_mag_max = float(np.abs(np.diag(s_matrix)).max())
        reflection_db_worst = 20.0 * math.log10(max(reflection_mag_max, 1.0e-12))
        return_loss_min_db = -reflection_db_worst
        port_matching_passed = bool(return_loss_min_db >= float(args.return_loss_min_db))
        off_diagonal = np.abs(s_matrix.copy())
        np.fill_diagonal(off_diagonal, 0.0)
        coupling_mag_max = float(off_diagonal.max())
        coupling_db_worst = 20.0 * math.log10(max(coupling_mag_max, 1.0e-12))
    else:
        reciprocity_max = reciprocity_rms = reflection_mag_max = reflection_db_worst = float("nan")
        return_loss_min_db = coupling_mag_max = coupling_db_worst = float("nan")
        port_matching_passed = False
    smoke_passed = bool(
        field_complete
        and field_finite
        and field_nonzero
        and combination_finite
        and s_complete
        and reciprocity_max <= float(args.reciprocity_max)
    )
    summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port_count": len(ports),
        "grid_point_count": int(all_theta.size) if all_theta is not None else 0,
        "frequency_ghz": frequency_ghz,
        "field_complete": field_complete,
        "field_finite": field_finite,
        "field_nonzero": field_nonzero,
        "field_norm_max_min_ratio": norm_ratio,
        "linear_combination_finite": combination_finite,
        "s_parameter_complete": s_complete,
        "s_reciprocity_max_abs": reciprocity_max,
        "s_reciprocity_rms_abs": reciprocity_rms,
        "reflection_magnitude_max": reflection_mag_max,
        "reflection_db_worst": reflection_db_worst,
        "return_loss_min_db": return_loss_min_db,
        "return_loss_requirement_db": float(args.return_loss_min_db),
        "engineering_port_matching_passed": port_matching_passed,
        "mutual_coupling_magnitude_max": coupling_mag_max,
        "mutual_coupling_worst_db": coupling_db_worst,
        "smoke_passed": smoke_passed,
        "allow_256_port_export": False,
        "failures": failures,
        "next_gate": (
            "The EEP operator is structurally valid, but raw 50-ohm port matching failed. "
            "Add a physical matching layer and validate full-array active reflection before expansion."
            if smoke_passed and not port_matching_passed
            else (
                "The 16-port operator and passive port match are valid. Validate direct combinations and "
                "full-array active reflection before 256-port expansion."
                if smoke_passed
                else "Fix EEP/S completeness before any 256-port expansion."
            )
        ),
        "outputs": {"operator": str(args.eep_dir / "eep_operator_16port.npz")},
    }
    (args.eep_dir / "operator_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
