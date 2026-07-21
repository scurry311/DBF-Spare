"""Cluster URA16 ports by geometry/K=1 active impedance and design class L-matches."""

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from design_eep_port_match import component_description, lmatch_s_parameters, s_to_z


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_S_NPZ = (
    DATASET_ROOT
    / "full_s256p_matched_v2_20260714"
    / "active_return_analysis_20260714"
    / "full_s_matrix_256.npz"
)
DEFAULT_DATASET = DATASET_ROOT / "dataset_arrays.npz"
DEFAULT_OUT = (
    DATASET_ROOT
    / "full_s256p_matched_v2_20260714"
    / "port_class_matching_20260714"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-npz", type=Path, default=DEFAULT_S_NPZ)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--edge-clusters", type=int, default=2)
    parser.add_argument("--interior-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def db20(value: np.ndarray | float) -> np.ndarray | float:
    result = 20.0 * np.log10(np.maximum(np.abs(value), 1.0e-15))
    return float(result) if np.ndim(result) == 0 else result


def geometry_class(ix: int, iy: int) -> str:
    x_edge = ix in (0, 15)
    y_edge = iy in (0, 15)
    if x_edge and y_edge:
        return "corner"
    if x_edge or y_edge:
        return "edge"
    return "interior"


def compose_nonuniform_network(
    s_antenna: np.ndarray,
    series_x: np.ndarray,
    shunt_b: np.ndarray,
    z0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = s_antenna.shape[0]
    s11 = np.empty(count, dtype=np.complex128)
    s12 = np.empty(count, dtype=np.complex128)
    s21 = np.empty(count, dtype=np.complex128)
    s22 = np.empty(count, dtype=np.complex128)
    for index in range(count):
        s11[index], s12[index], s21[index], s22[index] = lmatch_s_parameters(
            float(series_x[index]), float(shunt_b[index]), z0
        )
    identity = np.eye(count, dtype=np.complex128)
    antenna_wave_map = np.linalg.solve(identity - s22[:, None] * s_antenna, np.diag(s21))
    composite = np.diag(s11) + s12[:, None] * (s_antenna @ antenna_wave_map)
    network_parameters = np.column_stack((s11, s12, s21, s22))
    return composite, antenna_wave_map, network_parameters


def optimize_class_match(active_impedances: np.ndarray, z0: float, seed: int) -> tuple[float, float, dict[str, float]]:
    finite = active_impedances[np.isfinite(active_impedances)]
    if finite.size == 0:
        raise ValueError("No finite active impedances for class")

    def objective(parameters: np.ndarray) -> float:
        series_x = float(parameters[0])
        shunt_b = float(parameters[1])
        transformed = 1.0 / (1.0 / finite + 1j * shunt_b) + 1j * series_x
        reflection = np.abs((transformed - z0) / (transformed + z0))
        return float(
            np.quantile(reflection, 0.99)
            + 0.35 * np.quantile(reflection, 0.95)
            + 0.10 * np.median(reflection)
        )

    result = differential_evolution(
        objective,
        bounds=[(-300.0, 300.0), (-0.05, 0.05)],
        seed=int(seed),
        tol=1.0e-8,
        polish=True,
    )
    x_value = float(result.x[0])
    b_value = float(result.x[1])
    transformed = 1.0 / (1.0 / finite + 1j * b_value) + 1j * x_value
    reflection = np.abs((transformed - z0) / (transformed + z0))
    metrics = {
        "sample_count": int(finite.size),
        "scalar_rl_min_db": -float(np.max(db20(reflection))),
        "scalar_rl_p05_db": -float(np.quantile(db20(reflection), 0.95)),
        "scalar_rl_median_db": -float(np.median(db20(reflection))),
    }
    return x_value, b_value, metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    s_payload = np.load(args.s_npz, allow_pickle=False)
    s_matrix = np.asarray(s_payload["s_parameters"], dtype=np.complex128)
    port_names = [str(port) for port in s_payload["port_names"]]
    z0 = float(s_payload["reference_impedance_ohm"])
    z_passive = s_to_z(s_matrix, z0)
    passive_diag = np.diag(z_passive)

    dataset = np.load(args.dataset, allow_pickle=False)
    dataset_ports = [str(port) for port in dataset["port_names"]]
    reorder = [dataset_ports.index(port) for port in port_names]
    weights_ri = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float64)
    weights = (weights_ri[:, :, 0] + 1j * weights_ri[:, :, 1])[:, reorder]
    masks = np.asarray(dataset["masks"], dtype=bool)[:, reorder]
    k_values = np.asarray(dataset["k_values"], dtype=int)
    k1 = k_values == 1
    weights_k1 = weights[k1]
    weights_k1 /= np.maximum(np.linalg.norm(weights_k1, axis=1, keepdims=True), 1.0e-15)
    masks_k1 = masks[k1]
    reflected_k1 = weights_k1 @ s_matrix.T
    gamma_k1 = np.full(weights_k1.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    source_active = masks_k1 & (np.abs(weights_k1) > 1.0e-10)
    gamma_k1[source_active] = reflected_k1[source_active] / weights_k1[source_active]
    with np.errstate(divide="ignore", invalid="ignore"):
        active_z = z0 * (1.0 + gamma_k1) / (1.0 - gamma_k1)

    feature_rows: list[list[float]] = []
    base_classes: list[str] = []
    active_z_by_port: list[np.ndarray] = []
    for index in range(256):
        ix, iy = divmod(index, 16)
        values = active_z[:, index]
        values = values[np.isfinite(values)]
        active_z_by_port.append(values)
        base_classes.append(geometry_class(ix, iy))
        feature_rows.append(
            [
                float(passive_diag[index].real),
                float(passive_diag[index].imag),
                float(np.median(values.real)),
                float(np.median(values.imag)),
                float(np.quantile(values.real, 0.75) - np.quantile(values.real, 0.25)),
                float(np.quantile(values.imag, 0.75) - np.quantile(values.imag, 0.25)),
            ]
        )
    features = np.asarray(feature_rows, dtype=np.float64)
    cluster_labels = np.full(256, -1, dtype=int)
    class_names: dict[int, str] = {}
    next_label = 0
    for base_class, cluster_count in (
        ("corner", 1),
        ("edge", int(args.edge_clusters)),
        ("interior", int(args.interior_clusters)),
    ):
        indices = np.asarray([index for index, value in enumerate(base_classes) if value == base_class], dtype=int)
        scaled = StandardScaler().fit_transform(features[indices])
        if cluster_count == 1:
            local_labels = np.zeros(indices.size, dtype=int)
        else:
            local_labels = KMeans(
                n_clusters=cluster_count,
                random_state=int(args.seed) + next_label,
                n_init=30,
            ).fit_predict(scaled)
        for local_label in range(cluster_count):
            selected = indices[local_labels == local_label]
            cluster_labels[selected] = next_label
            class_names[next_label] = f"{base_class}_{local_label}"
            next_label += 1
    if np.any(cluster_labels < 0):
        raise RuntimeError("Port clustering left unassigned ports")

    series_x_by_class: dict[int, float] = {}
    shunt_b_by_class: dict[int, float] = {}
    class_rows: list[dict[str, Any]] = []
    for class_id in sorted(class_names):
        indices = np.flatnonzero(cluster_labels == class_id)
        class_samples = np.concatenate([active_z_by_port[index] for index in indices])
        series_x, shunt_b, scalar_metrics = optimize_class_match(
            class_samples, z0, int(args.seed) + class_id
        )
        series_x_by_class[class_id] = series_x
        shunt_b_by_class[class_id] = shunt_b
        components = component_description(series_x, shunt_b, float(s_payload["frequency_hz"][0]))
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "port_count": int(indices.size),
                "ports": ";".join(port_names[index] for index in indices),
                "series_reactance_ohm": series_x,
                "shunt_susceptance_siemens": shunt_b,
                "series_component": json.dumps(components["series"], separators=(",", ":")),
                "shunt_component": json.dumps(components["shunt"], separators=(",", ":")),
                **scalar_metrics,
            }
        )

    series_x_ports = np.asarray([series_x_by_class[int(label)] for label in cluster_labels])
    shunt_b_ports = np.asarray([shunt_b_by_class[int(label)] for label in cluster_labels])
    class_s, antenna_wave_map, network_parameters = compose_nonuniform_network(
        s_matrix, series_x_ports, shunt_b_ports, z0
    )
    reflected_class = weights_k1 @ class_s.T
    gamma_class = np.full(weights_k1.shape, np.nan, dtype=np.float64)
    gamma_class[source_active] = np.abs(reflected_class[source_active] / weights_k1[source_active])
    active_rl = -20.0 * np.log10(np.maximum(gamma_class[source_active], 1.0e-15))
    case_worst: list[float] = []
    for case_index in range(weights_k1.shape[0]):
        values = gamma_class[case_index, source_active[case_index]]
        case_worst.append(float(-20.0 * np.log10(max(float(np.max(values)), 1.0e-15))))
    passive_rl = -np.asarray(db20(np.diag(class_s)))
    singular_max = float(np.linalg.svd(class_s, compute_uv=False).max())

    port_rows: list[dict[str, Any]] = []
    for index, port in enumerate(port_names):
        ix, iy = divmod(index, 16)
        values = active_z_by_port[index]
        port_rows.append(
            {
                "port_index": index,
                "port_name": port,
                "ix": ix,
                "iy": iy,
                "geometry_class": base_classes[index],
                "cluster_id": int(cluster_labels[index]),
                "cluster_name": class_names[int(cluster_labels[index])],
                "passive_z_real_ohm": float(passive_diag[index].real),
                "passive_z_imag_ohm": float(passive_diag[index].imag),
                "k1_active_z_real_median_ohm": float(np.median(values.real)),
                "k1_active_z_imag_median_ohm": float(np.median(values.imag)),
                "k1_active_z_real_iqr_ohm": float(np.quantile(values.real, 0.75) - np.quantile(values.real, 0.25)),
                "k1_active_z_imag_iqr_ohm": float(np.quantile(values.imag, 0.75) - np.quantile(values.imag, 0.25)),
                "series_reactance_ohm": float(series_x_ports[index]),
                "shunt_susceptance_siemens": float(shunt_b_ports[index]),
                "class_network_passive_rl_db": float(passive_rl[index]),
            }
        )
    write_csv(args.out_dir / "port_clusters.csv", port_rows)
    write_csv(args.out_dir / "port_class_matching_networks.csv", class_rows)
    np.savez_compressed(
        args.out_dir / "port_class_matched_s256.npz",
        port_names=np.asarray(port_names),
        frequency_hz=s_payload["frequency_hz"],
        reference_impedance_ohm=np.asarray(z0),
        raw_s_parameters=s_matrix.astype(np.complex64),
        s_parameters=class_s.astype(np.complex64),
        antenna_incident_wave_map=antenna_wave_map.astype(np.complex64),
        network_s_parameters=network_parameters.astype(np.complex64),
        cluster_labels=cluster_labels.astype(np.int16),
        series_reactance_ohm=series_x_ports,
        shunt_susceptance_siemens=shunt_b_ports,
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port_count": 256,
        "class_count": len(class_names),
        "class_names": class_names,
        "k1_case_count": int(weights_k1.shape[0]),
        "raw_passive_return_loss_min_db": -float(np.max(db20(np.diag(s_matrix)))),
        "class_network_passive_return_loss_min_db": float(passive_rl.min()),
        "class_network_passive_return_loss_median_db": float(np.median(passive_rl)),
        "class_network_passivity_sigma_max": singular_max,
        "k1_worst_active_return_loss_min_db": float(np.min(case_worst)),
        "k1_worst_active_return_loss_mean_db": float(np.mean(case_worst)),
        "k1_all_active_10db_case_pass_rate": float(np.mean(np.asarray(case_worst) >= 10.0)),
        "k1_active_port_return_loss_min_db": float(active_rl.min()),
        "k1_active_port_return_loss_median_db": float(np.median(active_rl)),
        "outputs": {
            "clusters": str(args.out_dir / "port_clusters.csv"),
            "class_networks": str(args.out_dir / "port_class_matching_networks.csv"),
            "matched_s": str(args.out_dir / "port_class_matched_s256.npz"),
        },
    }
    (args.out_dir / "port_class_matching_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
