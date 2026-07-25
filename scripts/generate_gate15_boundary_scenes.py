#!/usr/bin/env python3
"""Generate independent PSLL/nearest/local gate15 boundary scene triplets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from generate_dense_boundary_hard_negatives import (
    active_metrics,
    gate15,
    gate20,
    mainlobe_gate,
)
from generate_expanded_independent_residual_scenes import (
    angular_separation_deg,
    optimize_scene,
    phase_migrate,
    strict_metrics,
    target_hash,
)
from hfss_task_fullwave_validate import pattern_grid_dirs, unit_vector
from optimize_trusted_eep_s256_joint_weights import pattern_metrics
from refine_trusted_dense_local_eep_joint import DenseExternalEEP, nearest_grid_index
from validate_trusted_eep_hfss_residuals import series_network_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_dataset_20260724_run01"
DEFAULT_EXPANDED = ROOT / "hfss_outputs" / "expanded_independent_scenes_20260724_run02"
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_EXCITATIONS = (
    ROOT
    / "hfss_outputs"
    / "trusted_dense_joint_hfss_smoke_20260724_run01"
    / "case_excitations.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "gate15_boundary_scenes_20260725_run01"
KMAX = 6
ETA0 = 376.730313668
EPS = 1.0e-12


GLOBAL_SHIFTS = (
    (0, 4),
    (0, -4),
    (1, 2),
    (1, -2),
    (-1, 0),
    (-1, 2),
    (-1, -2),
    (0, 6),
    (0, -6),
    (2, 0),
    (-2, 0),
    (1, 4),
    (1, -4),
    (-1, 4),
    (-1, -4),
    (2, 2),
    (2, -2),
    (-2, 2),
    (-2, -2),
    (0, 8),
    (0, -8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--expanded-dir", type=Path, default=DEFAULT_EXPANDED)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--excitations", type=Path, default=DEFAULT_EXCITATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenes-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--exclude-dataset-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional dataset package whose target hashes must remain unseen.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("global", "prospective"),
        default="global",
        help="Prospective mode also perturbs individual task directions and separations.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shifted_grid_targets(old: np.ndarray, dtheta: int, dphi: int) -> np.ndarray:
    theta = np.clip(np.rint(old[:, 0]) + int(dtheta), 1.0, 75.0)
    phi = (np.rint(old[:, 1] / 2.0) * 2.0 + int(dphi)) % 360.0
    return np.column_stack((theta, phi)).astype(np.float64)


def prospective_grid_targets(
    old: np.ndarray, dtheta: int, dphi: int, pattern: int
) -> np.ndarray:
    task = np.arange(old.shape[0], dtype=np.int64)
    theta_local = ((2 * task + int(pattern)) % 5) - 2
    phi_local = 2 * (((3 * task + 2 * int(pattern)) % 5) - 2)
    theta = np.clip(
        np.rint(old[:, 0]) + int(dtheta) + theta_local.astype(float), 1.0, 75.0
    )
    phi = (
        np.rint(old[:, 1] / 2.0) * 2.0
        + int(dphi)
        + phi_local.astype(float)
    ) % 360.0
    return np.column_stack((theta, phi)).astype(np.float64)


class FastPatternEvaluator:
    """Vectorized EEP metrics for boundary line searches."""

    def __init__(self, operator: DenseExternalEEP, theta: np.ndarray, phi: np.ndarray):
        self.etheta = np.asarray(operator.etheta, dtype=np.complex64)
        self.ephi = np.asarray(operator.ephi, dtype=np.complex64)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.phi = np.asarray(phi, dtype=np.float64)
        self.dirs = pattern_grid_dirs(self.theta, self.phi)

    def evaluate(self, tasks: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(tasks, dtype=np.complex64)
        if values.ndim == 2:
            values = values[None, ...]
        batch, _ports, k_value = values.shape
        task_norm = np.maximum(np.linalg.norm(values, axis=1), EPS)
        task_external = values / task_norm[:, None, :]
        combined = np.sum(values, axis=2)
        combined /= np.maximum(np.linalg.norm(combined, axis=1), EPS)[:, None]
        cases = np.concatenate(
            (task_external.transpose(0, 2, 1), combined[:, None, :]), axis=1
        ).reshape(batch * (k_value + 1), values.shape[1])
        field_theta = (cases @ self.etheta).reshape(batch, k_value + 1, -1)
        field_phi = (cases @ self.ephi).reshape(batch, k_value + 1, -1)
        gain = 10.0 * np.log10(
            np.maximum(
                (2.0 * np.pi / ETA0) * (np.abs(field_theta) ** 2 + np.abs(field_phi) ** 2),
                1.0e-30,
            )
        )
        target_dirs = np.stack([unit_vector(float(t), float(p)) for t, p in targets])
        distance = np.rad2deg(
            np.arccos(np.clip(self.dirs @ target_dirs.T, -1.0, 1.0))
        )
        nearest = np.argmin(distance, axis=0)
        local_masks = [distance[:, target] <= 5.0 for target in range(k_value)]

        combined_gain = gain[:, -1, :]
        peaks = np.stack(
            [np.max(combined_gain[:, mask], axis=1) for mask in local_masks], axis=1
        )
        side_mask = np.min(distance, axis=1) > 8.0
        worst_side = np.max(combined_gain[:, side_mask], axis=1)
        pointing = []
        for target, mask in enumerate(local_masks):
            local_indices = np.flatnonzero(mask)
            best = np.argmax(combined_gain[:, mask], axis=1)
            pointing.append(distance[local_indices[best], target])

        task_gain = gain[:, :k_value, :]
        nearest_matrix = task_gain[:, :, nearest]
        local_matrix = np.stack(
            [np.max(task_gain[:, :, mask], axis=2) for mask in local_masks], axis=2
        )
        nearest_iso = []
        local_iso = []
        for task in range(k_value):
            other = [target for target in range(k_value) if target != task]
            nearest_iso.append(
                nearest_matrix[:, task, task]
                - np.max(nearest_matrix[:, task, other], axis=1)
            )
            local_iso.append(
                local_matrix[:, task, task]
                - np.max(local_matrix[:, task, other], axis=1)
            )
        return {
            "psll_db": worst_side - np.min(peaks, axis=1),
            "weakest_target_gain_db": np.min(peaks, axis=1),
            "target_spread_db": np.max(peaks, axis=1) - np.min(peaks, axis=1),
            "nearest_iso_db": np.min(np.stack(nearest_iso, axis=1), axis=1),
            "local_iso_db": np.min(np.stack(local_iso, axis=1), axis=1),
            "pointing_error_deg": np.max(np.stack(pointing, axis=1), axis=1),
        }


def metric_at(batch: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {key: float(value[index]) for key, value in batch.items()}


def batch_mainlobe(metrics: dict[str, np.ndarray], reference: dict[str, float]) -> np.ndarray:
    return (
        (metrics["weakest_target_gain_db"] >= reference["weakest_target_gain_db"] - 0.5)
        & (metrics["target_spread_db"] <= 3.0)
        & (metrics["pointing_error_deg"] <= 1.5)
    )


def projected_probe_mode(
    effective: DenseExternalEEP,
    active: np.ndarray,
    centers: list[int],
    probe: int,
) -> tuple[float, np.ndarray]:
    constraint_rows: list[np.ndarray] = []
    for center in centers:
        constraint_rows.extend(
            (effective.etheta[active, center], effective.ephi[active, center])
        )
    matrix = np.stack(constraint_rows, axis=1).astype(np.complex128)
    best_norm = -1.0
    best_value = np.zeros(active.size, dtype=np.complex128)
    for probe_row in (effective.etheta[active, probe], effective.ephi[active, probe]):
        row = np.asarray(probe_row, dtype=np.complex128)
        projected = row - matrix @ np.linalg.lstsq(matrix, row, rcond=1.0e-8)[0]
        norm = float(np.linalg.norm(projected))
        if norm > best_norm:
            best_norm = norm
            best_value = np.conjugate(projected / max(norm, EPS))
    return best_norm, best_value


def exact_metrics(
    tasks: np.ndarray,
    targets: np.ndarray,
    raw_operator: dict[str, np.ndarray],
    antenna_map: np.ndarray,
) -> dict[str, float]:
    return pattern_metrics(
        tasks,
        targets,
        raw_operator["theta_deg"],
        raw_operator["phi_deg"],
        raw_operator["etheta"],
        raw_operator["ephi"],
        antenna_map,
    )


def verify_pair(
    boundary_type: str,
    inside: np.ndarray,
    outside: np.ndarray,
    nominal: dict[str, float],
    targets: np.ndarray,
    raw_operator: dict[str, np.ndarray],
    antenna_map: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]] | None:
    inside_metrics = exact_metrics(inside, targets, raw_operator, antenna_map)
    outside_metrics = exact_metrics(outside, targets, raw_operator, antenna_map)
    if not mainlobe_gate(inside_metrics, nominal) or not mainlobe_gate(outside_metrics, nominal):
        return None
    inside_ok = gate15(inside_metrics)
    outside_fail = not gate15(outside_metrics)
    isolated = {
        "psll": bool(
            0.0 < outside_metrics["psll_db"] <= 1.0
            and outside_metrics["nearest_iso_db"] >= 25.0
            and outside_metrics["local_iso_db"] >= 15.0
        ),
        "nearest": bool(
            outside_metrics["psll_db"] <= 0.0
            and 23.5 <= outside_metrics["nearest_iso_db"] < 25.0
            and outside_metrics["local_iso_db"] >= 15.0
        ),
        "local": bool(
            outside_metrics["psll_db"] <= 0.0
            and outside_metrics["nearest_iso_db"] >= 25.0
            and 13.0 <= outside_metrics["local_iso_db"] < 15.0
        ),
    }
    if not (inside_ok and outside_fail and isolated[boundary_type]):
        return None
    return inside_metrics, outside_metrics


def psll_pair(
    tasks: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    nominal: dict[str, float],
    effective: DenseExternalEEP,
    fast: FastPatternEvaluator,
    grid_dirs: np.ndarray,
    raw_operator: dict[str, np.ndarray],
    antenna_map: np.ndarray,
) -> dict[str, Any] | None:
    active = np.flatnonzero(mask)
    centers = [nearest_grid_index(grid_dirs, float(t), float(p)) for t, p in targets]
    combined = np.sum(tasks, axis=1)
    target_amplitudes = []
    for center in centers:
        target_amplitudes.append(
            math.sqrt(
                abs(combined[active] @ effective.etheta[active, center]) ** 2
                + abs(combined[active] @ effective.ephi[active, center]) ** 2
            )
        )
    weakest = min(target_amplitudes)
    probes: list[tuple[float, int, int, int, float, np.ndarray]] = []
    for theta in range(8, 77, 8):
        for phi in range(0, 360, 20):
            direction = unit_vector(theta, phi)
            distance = min(
                np.rad2deg(
                    np.arccos(np.clip(direction @ unit_vector(*target), -1.0, 1.0))
                )
                for target in targets
            )
            if distance <= 12.0:
                continue
            probe = nearest_grid_index(grid_dirs, float(theta), float(phi))
            response, mode = projected_probe_mode(effective, active, centers, probe)
            scale = weakest / max(response, EPS)
            relative = scale / max(float(np.linalg.norm(combined[active])), EPS)
            probes.append((relative, theta, phi, probe, scale, mode))
    for relative, theta, phi, _probe, scale, mode in sorted(probes)[:30]:
        factors = np.linspace(0.70, 1.65, 20)
        trials = np.repeat(tasks[None, ...], factors.size, axis=0)
        delta = np.zeros(256, dtype=np.complex128)
        delta[active] = scale * mode
        trials += factors[:, None, None] * delta[None, :, None] / tasks.shape[1]
        metrics = fast.evaluate(trials, targets)
        main = batch_mainlobe(metrics, nominal)
        inside_indices = np.flatnonzero(
            main
            & (metrics["psll_db"] >= -0.8)
            & (metrics["psll_db"] <= -0.05)
            & (metrics["nearest_iso_db"] >= 25.0)
            & (metrics["local_iso_db"] >= 15.0)
        )
        outside_indices = np.flatnonzero(
            main
            & (metrics["psll_db"] > 0.0)
            & (metrics["psll_db"] <= 1.0)
            & (metrics["nearest_iso_db"] >= 25.0)
            & (metrics["local_iso_db"] >= 15.0)
        )
        if not inside_indices.size or not outside_indices.size:
            continue
        inside_index = int(
            inside_indices[np.argmin(np.abs(metrics["psll_db"][inside_indices] + 0.35))]
        )
        outside_index = int(
            outside_indices[np.argmin(np.abs(metrics["psll_db"][outside_indices] - 0.35))]
        )
        verified = verify_pair(
            "psll",
            trials[inside_index],
            trials[outside_index],
            nominal,
            targets,
            raw_operator,
            antenna_map,
        )
        if verified is None:
            continue
        inside_metrics, outside_metrics = verified
        return {
            "inside": trials[inside_index].astype(np.complex64),
            "outside": trials[outside_index].astype(np.complex64),
            "inside_metrics": inside_metrics,
            "outside_metrics": outside_metrics,
            "inside_amplitude": float(factors[inside_index] * relative),
            "outside_amplitude": float(factors[outside_index] * relative),
            "mode_theta_deg": theta,
            "mode_phi_deg": phi,
            "task_mixing_amplitude": 0.0,
        }
    return None


def nearest_pair(
    tasks: np.ndarray,
    targets: np.ndarray,
    nominal: dict[str, float],
    fast: FastPatternEvaluator,
    raw_operator: dict[str, np.ndarray],
    antenna_map: np.ndarray,
) -> dict[str, Any] | None:
    k_value = tasks.shape[1]
    alphas = np.linspace(0.025, 0.12, 20)
    phases = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    for task in range(k_value):
        for source in range(k_value):
            if source == task:
                continue
            for phase in phases:
                trials = np.repeat(tasks[None, ...], alphas.size, axis=0)
                trials[:, :, task] += (
                    alphas[:, None]
                    * np.exp(1j * phase)
                    * tasks[None, :, source]
                )
                metrics = fast.evaluate(trials, targets)
                main = batch_mainlobe(metrics, nominal)
                common = main & (metrics["psll_db"] <= 0.0) & (metrics["local_iso_db"] >= 15.0)
                inside_indices = np.flatnonzero(
                    common
                    & (metrics["nearest_iso_db"] >= 25.0)
                    & (metrics["nearest_iso_db"] <= 26.0)
                )
                outside_indices = np.flatnonzero(
                    common
                    & (metrics["nearest_iso_db"] >= 23.5)
                    & (metrics["nearest_iso_db"] < 25.0)
                )
                if not inside_indices.size or not outside_indices.size:
                    continue
                inside_index = int(
                    inside_indices[
                        np.argmin(np.abs(metrics["nearest_iso_db"][inside_indices] - 25.4))
                    ]
                )
                outside_index = int(
                    outside_indices[
                        np.argmin(np.abs(metrics["nearest_iso_db"][outside_indices] - 24.4))
                    ]
                )
                verified = verify_pair(
                    "nearest",
                    trials[inside_index],
                    trials[outside_index],
                    nominal,
                    targets,
                    raw_operator,
                    antenna_map,
                )
                if verified is None:
                    continue
                inside_metrics, outside_metrics = verified
                return {
                    "inside": trials[inside_index].astype(np.complex64),
                    "outside": trials[outside_index].astype(np.complex64),
                    "inside_metrics": inside_metrics,
                    "outside_metrics": outside_metrics,
                    "inside_amplitude": float(alphas[inside_index]),
                    "outside_amplitude": float(alphas[outside_index]),
                    "task_mixing_amplitude": float(alphas[outside_index]),
                    "mix_task": task,
                    "mix_source": source,
                    "mix_phase_deg": float(np.rad2deg(phase)),
                }
    return None


def local_pair(
    tasks: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    nominal: dict[str, float],
    effective: DenseExternalEEP,
    fast: FastPatternEvaluator,
    grid_dirs: np.ndarray,
    raw_operator: dict[str, np.ndarray],
    antenna_map: np.ndarray,
) -> dict[str, Any] | None:
    active = np.flatnonzero(mask)
    centers = [nearest_grid_index(grid_dirs, float(t), float(p)) for t, p in targets]
    factors = np.linspace(0.45, 1.55, 18)
    for task in range(tasks.shape[1]):
        own_center = centers[task]
        own_amplitude = math.sqrt(
            abs(tasks[active, task] @ effective.etheta[active, own_center]) ** 2
            + abs(tasks[active, task] @ effective.ephi[active, own_center]) ** 2
        )
        for target in range(tasks.shape[1]):
            if target == task:
                continue
            theta, phi = targets[target]
            offsets = (
                (theta + 4, phi),
                (theta - 4, phi),
                (theta, phi + 4),
                (theta, phi - 4),
                (theta + 5, phi),
                (theta - 5, phi),
                (theta, phi + 5),
                (theta, phi - 5),
            )
            for probe_theta, probe_phi in offsets:
                probe_theta = float(np.clip(probe_theta, 1.0, 80.0))
                probe_phi = float(probe_phi % 360.0)
                probe = nearest_grid_index(grid_dirs, probe_theta, probe_phi)
                response, mode = projected_probe_mode(effective, active, centers, probe)
                scale = own_amplitude * 10.0 ** (-14.25 / 20.0) / max(response, EPS)
                trials = np.repeat(tasks[None, ...], factors.size, axis=0)
                trials[:, active, task] += factors[:, None] * scale * mode[None, :]
                metrics = fast.evaluate(trials, targets)
                main = batch_mainlobe(metrics, nominal)
                common = main & (metrics["psll_db"] <= 0.0) & (metrics["nearest_iso_db"] >= 25.0)
                inside_indices = np.flatnonzero(
                    common
                    & (metrics["local_iso_db"] >= 15.0)
                    & (metrics["local_iso_db"] <= 16.0)
                )
                outside_indices = np.flatnonzero(
                    common
                    & (metrics["local_iso_db"] >= 13.0)
                    & (metrics["local_iso_db"] < 15.0)
                )
                if not inside_indices.size or not outside_indices.size:
                    continue
                inside_index = int(
                    inside_indices[
                        np.argmin(np.abs(metrics["local_iso_db"][inside_indices] - 15.4))
                    ]
                )
                outside_index = int(
                    outside_indices[
                        np.argmin(np.abs(metrics["local_iso_db"][outside_indices] - 14.4))
                    ]
                )
                verified = verify_pair(
                    "local",
                    trials[inside_index],
                    trials[outside_index],
                    nominal,
                    targets,
                    raw_operator,
                    antenna_map,
                )
                if verified is None:
                    continue
                inside_metrics, outside_metrics = verified
                relative_scale = scale / max(float(np.linalg.norm(tasks[active, task])), EPS)
                return {
                    "inside": trials[inside_index].astype(np.complex64),
                    "outside": trials[outside_index].astype(np.complex64),
                    "inside_metrics": inside_metrics,
                    "outside_metrics": outside_metrics,
                    "inside_amplitude": float(factors[inside_index] * relative_scale),
                    "outside_amplitude": float(factors[outside_index] * relative_scale),
                    "task_mixing_amplitude": 0.0,
                    "leakage_task": task,
                    "leakage_target": target,
                    "mode_theta_deg": probe_theta,
                    "mode_phi_deg": probe_phi,
                }
    return None


def implementation_descriptors(
    nominal: np.ndarray,
    actual: np.ndarray,
    *,
    boundary_type: str,
    amplitude: float,
    task_mixing: float,
) -> dict[str, float | int]:
    delta = actual - nominal
    return {
        "implementation_delta_norm": float(
            np.linalg.norm(delta) / max(float(np.linalg.norm(nominal)), EPS)
        ),
        "implementation_delta_max": float(
            np.max(np.abs(delta)) / max(float(np.max(np.abs(nominal))), EPS)
        ),
        "task_mixing_amplitude": float(task_mixing),
        "coherent_mode_amplitude": float(amplitude if boundary_type != "nearest" else 0.0),
        "boundary_metric_code": {"control": 0, "psll": 1, "nearest": 2, "local": 3}[
            boundary_type
        ],
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with np.load(args.base_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        base = {key: source[key] for key in source.files}
    with np.load(args.expanded_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        expanded = {key: source[key] for key in source.files}
    with np.load(args.operator, allow_pickle=False) as source:
        raw_operator = {key: source[key] for key in source.files}
    with np.load(args.excitations, allow_pickle=False) as source:
        antenna_map = np.asarray(source["antenna_wave_map"], dtype=np.complex64)
        s_matrix = np.asarray(source["matched_s"], dtype=np.complex128)
    expected_s, expected_map, _series_z = series_network_map(
        np.asarray(raw_operator["s_raw"], dtype=np.complex128), 1.0e10
    )
    if np.max(np.abs(expected_s - s_matrix)) > 1.0e-7:
        raise RuntimeError("Matched S mismatch")
    if np.max(np.abs(expected_map - antenna_map)) > 1.0e-7:
        raise RuntimeError("Antenna-wave map mismatch")

    theta = np.asarray(raw_operator["theta_deg"], dtype=np.float64)
    phi = np.asarray(raw_operator["phi_deg"], dtype=np.float64)
    grid_dirs = pattern_grid_dirs(theta, phi)
    effective = DenseExternalEEP(raw_operator["etheta"], raw_operator["ephi"], antenna_map)
    fast = FastPatternEvaluator(effective, theta, phi)
    positions = np.asarray(base["positions_lambda"], dtype=np.float64)
    internal = (
        np.asarray(base["task_weights_real_imag"][..., 0], dtype=np.float64)
        + 1j * np.asarray(base["task_weights_real_imag"][..., 1], dtype=np.float64)
    )

    used_hashes = set()
    packages_for_exclusion = [base, expanded]
    for directory in args.exclude_dataset_dir:
        with np.load(directory / "dataset_arrays.npz", allow_pickle=False) as source:
            packages_for_exclusion.append({key: source[key] for key in source.files})
    for package in packages_for_exclusion:
        for candidate in range(int(package["candidate_indices"].size)):
            k_value = int(package["k_values"][candidate])
            used_hashes.add(target_hash(package["targets_deg"][candidate, :k_value]))

    parent_pools = {
        "psll": [index for index, value in enumerate(base["k_values"]) if int(value) == 6],
        "nearest": [index for index, value in enumerate(base["k_values"]) if int(value) in (2, 4)],
        "local": [index for index, value in enumerate(base["k_values"]) if int(value) in (2, 4)],
    }
    scene_records: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for boundary_type in ("psll", "nearest", "local"):
        accepted = 0
        patterns = (0,) if args.target_mode == "global" else (1, 2, 3, 4, 5)
        attempts = [
            (parent, shift, pattern)
            for pattern in patterns
            for shift in GLOBAL_SHIFTS
            for parent in parent_pools[boundary_type]
        ]
        for parent, (dtheta, dphi), pattern in attempts:
            if accepted >= int(args.scenes_per_type):
                break
            k_value = int(base["k_values"][parent])
            old_targets = np.asarray(base["targets_deg"][parent, :k_value], dtype=np.float64)
            targets = (
                shifted_grid_targets(old_targets, dtheta, dphi)
                if args.target_mode == "global"
                else prospective_grid_targets(old_targets, dtheta, dphi, pattern)
            )
            digest = target_hash(targets)
            if digest in used_hashes or angular_separation_deg(targets) < 5.0:
                continue
            mask = np.asarray(base["masks"][parent], dtype=bool)
            original = np.conjugate(internal[parent, :, :k_value]).astype(np.complex64)
            original[~mask] = 0.0
            warm = phase_migrate(original, old_targets, targets, positions)
            warm[~mask] = 0.0
            refined, nominal, active, reference = optimize_scene(
                warm,
                mask,
                targets,
                theta,
                phi,
                raw_operator["etheta"],
                raw_operator["ephi"],
                antenna_map,
                s_matrix,
                grid_dirs,
                effective,
            )
            strict = strict_metrics(nominal, reference, active)
            pair: dict[str, Any] | None = None
            if strict:
                if boundary_type == "psll":
                    pair = psll_pair(
                        refined,
                        mask,
                        targets,
                        nominal,
                        effective,
                        fast,
                        grid_dirs,
                        raw_operator,
                        antenna_map,
                    )
                elif boundary_type == "nearest":
                    pair = nearest_pair(
                        refined,
                        targets,
                        nominal,
                        fast,
                        raw_operator,
                        antenna_map,
                    )
                else:
                    pair = local_pair(
                        refined,
                        mask,
                        targets,
                        nominal,
                        effective,
                        fast,
                        grid_dirs,
                        raw_operator,
                        antenna_map,
                    )
            attempt_rows.append(
                {
                    "boundary_type": boundary_type,
                    "parent_candidate_index": parent,
                    "k": k_value,
                    "dtheta_deg": dtheta,
                    "dphi_deg": dphi,
                    "target_pattern": pattern,
                    "target_hash": digest,
                    "nominal_strict": int(strict),
                    "boundary_pair_found": int(pair is not None),
                    "nominal_psll_db": nominal["psll_db"],
                    "nominal_nearest_iso_db": nominal["nearest_iso_db"],
                    "nominal_local_iso_db": nominal["local_iso_db"],
                }
            )
            if not strict or pair is None:
                continue
            scene_number = len(scene_records)
            scene = {
                "scene_number": scene_number,
                "sample_index": 200000 + scene_number,
                "scene_id": f"gate15_{boundary_type}_{200000 + scene_number}_{digest}",
                "target_hash": digest,
                "boundary_type": boundary_type,
                "parent": parent,
                "k": k_value,
                "mask": mask,
                "targets": targets,
                "tasks": refined,
                "nominal_metrics": nominal,
                "active_metrics": active,
                "pair": pair,
                "min_target_separation_deg": angular_separation_deg(targets),
                "max_target_theta_deg": float(np.max(targets[:, 0])),
                "large_scan": int(np.max(targets[:, 0]) >= 45.0),
            }
            scene_records.append(scene)
            used_hashes.add(digest)
            accepted += 1
            outside = pair["outside_metrics"]
            print(
                f"scene={scene_number + 1:02d} type={boundary_type} K={k_value} "
                f"outside=({outside['psll_db']:.2f},{outside['nearest_iso_db']:.2f},"
                f"{outside['local_iso_db']:.2f}) point={outside['pointing_error_deg']:.2f}",
                flush=True,
            )
        if accepted < int(args.scenes_per_type):
            raise RuntimeError(
                f"Only {accepted}/{args.scenes_per_type} {boundary_type} scenes were generated"
            )

    candidates: list[dict[str, Any]] = []
    for scene in scene_records:
        pair = scene["pair"]
        for side, actual, metrics, amplitude in (
            ("control", scene["tasks"], scene["nominal_metrics"], 0.0),
            ("inside", pair["inside"], pair["inside_metrics"], pair["inside_amplitude"]),
            ("outside", pair["outside"], pair["outside_metrics"], pair["outside_amplitude"]),
        ):
            descriptor = implementation_descriptors(
                scene["tasks"],
                actual,
                boundary_type="control" if side == "control" else scene["boundary_type"],
                amplitude=amplitude,
                task_mixing=0.0 if side == "control" else pair["task_mixing_amplitude"],
            )
            candidates.append(
                {
                    "scene": scene,
                    "side": side,
                    "variant_kind": (
                        "nominal_control"
                        if side == "control"
                        else f"gate15_{scene['boundary_type']}_{side}"
                    ),
                    "command": scene["tasks"],
                    "actual": actual,
                    "actual_metrics": metrics,
                    "amplitude": amplitude,
                    **descriptor,
                }
            )

    n = len(candidates)
    masks = np.zeros((n, 256), dtype=np.int8)
    nominal_tasks = np.zeros((n, 256, KMAX), dtype=np.complex64)
    actual_tasks = np.zeros((n, 256, KMAX), dtype=np.complex64)
    targets_padded = np.zeros((n, KMAX, 2), dtype=np.float32)
    task_valid = np.zeros((n, KMAX), dtype=np.int8)
    manifest: list[dict[str, Any]] = []
    for candidate, record in enumerate(candidates):
        scene = record["scene"]
        k_value = int(scene["k"])
        mask = np.asarray(scene["mask"], dtype=bool)
        nominal_tasks[candidate, :, :k_value] = record["command"]
        actual_tasks[candidate, :, :k_value] = record["actual"]
        masks[candidate] = mask.astype(np.int8)
        targets_padded[candidate, :k_value] = scene["targets"]
        task_valid[candidate, :k_value] = 1
        nominal = scene["nominal_metrics"]
        actual = record["actual_metrics"]
        actual_rl = active_metrics(record["actual"], mask, s_matrix)
        manifest.append(
            {
                "candidate_index": candidate,
                "sample_index": scene["sample_index"],
                "scene_id": scene["scene_id"],
                "target_hash": scene["target_hash"],
                "boundary_type": scene["boundary_type"],
                "boundary_side": record["side"],
                "variant_kind": record["variant_kind"],
                "parent_candidate_index": scene["parent"],
                "k": k_value,
                "active_ratio": float(np.mean(mask)),
                "active_count": int(np.sum(mask)),
                "implementation_delta_norm": record["implementation_delta_norm"],
                "implementation_delta_max": record["implementation_delta_max"],
                "task_mixing_amplitude": record["task_mixing_amplitude"],
                "coherent_mode_amplitude": record["coherent_mode_amplitude"],
                "boundary_metric_code": record["boundary_metric_code"],
                "nominal_eep_gate15": int(gate15(nominal)),
                "actual_basis_gate15": int(gate15(actual)),
                "actual_basis_gate20": int(gate20(actual)),
                "actual_basis_mainlobe_gate": int(mainlobe_gate(actual, nominal)),
                "predicted_hard_negative": int(gate15(nominal) and not gate15(actual)),
                "nominal_psll_db": nominal["psll_db"],
                "actual_basis_psll_db": actual["psll_db"],
                "nominal_nearest_iso_db": nominal["nearest_iso_db"],
                "actual_basis_nearest_iso_db": actual["nearest_iso_db"],
                "nominal_local_iso_db": nominal["local_iso_db"],
                "actual_basis_local_iso_db": actual["local_iso_db"],
                "actual_basis_pointing_error_deg": actual["pointing_error_deg"],
                **actual_rl,
                "expected_hfss_case_count": 1 + k_value,
            }
        )

    nominal_internal = np.conjugate(nominal_tasks)
    actual_internal = np.conjugate(actual_tasks)
    nominal_combined = np.sum(nominal_internal, axis=2)
    actual_combined = np.sum(actual_internal, axis=2)
    scene_index = np.asarray([record["scene"]["sample_index"] for record in candidates])
    parent = np.asarray([record["scene"]["parent"] for record in candidates])
    k_values = np.asarray([record["scene"]["k"] for record in candidates])
    np.savez_compressed(
        args.out_dir / "dataset_arrays.npz",
        candidate_index=np.arange(n, dtype=np.int64),
        candidate_indices=np.arange(n, dtype=np.int64),
        sample_index=scene_index.astype(np.int64),
        sample_indices=scene_index.astype(np.int64),
        sample_ids=np.asarray([f"gate15_c{i:03d}_s{s}" for i, s in enumerate(scene_index)]),
        scene_ids=np.asarray([record["scene"]["scene_id"] for record in candidates]),
        source_dataset=np.full(n, "gate15_boundary_scenes_run01"),
        source_sample_indices=np.asarray(base["source_sample_indices"][parent]),
        selection_roles=np.asarray([record["variant_kind"] for record in candidates]),
        variant_kind=np.asarray([record["variant_kind"] for record in candidates]),
        boundary_type=np.asarray([record["scene"]["boundary_type"] for record in candidates]),
        boundary_side=np.asarray([record["side"] for record in candidates]),
        parent_candidate_index=parent.astype(np.int64),
        parent_ratio=np.asarray([np.mean(record["scene"]["mask"]) for record in candidates], dtype=np.float32),
        ratio_delta=np.zeros(n, dtype=np.float32),
        k_values=k_values.astype(np.int64),
        active_ratios_requested=masks.mean(axis=1).astype(np.float32),
        active_ratios_actual=masks.mean(axis=1).astype(np.float32),
        num_active=np.sum(masks, axis=1).astype(np.int64),
        targets_deg=targets_padded,
        task_valid=task_valid,
        mask=masks,
        masks=masks,
        w_tasks_real_imag=np.stack([nominal_internal.real, nominal_internal.imag], axis=-1).astype(np.float32),
        task_weights_real_imag=np.stack([nominal_internal.real, nominal_internal.imag], axis=-1).astype(np.float32),
        w_combined_real_imag=np.stack([nominal_combined.real, nominal_combined.imag], axis=-1).astype(np.float32),
        combined_weights_real_imag=np.stack([nominal_combined.real, nominal_combined.imag], axis=-1).astype(np.float32),
        hfss_actual_task_weights_real_imag=np.stack([actual_internal.real, actual_internal.imag], axis=-1).astype(np.float32),
        hfss_actual_combined_weights_real_imag=np.stack([actual_combined.real, actual_combined.imag], axis=-1).astype(np.float32),
        hfss_weights_real_imag=np.stack([actual_combined.real, actual_combined.imag], axis=-1).astype(np.float32),
        min_target_separation_deg=np.asarray([record["scene"]["min_target_separation_deg"] for record in candidates], dtype=np.float32),
        max_target_theta_deg=np.asarray([record["scene"]["max_target_theta_deg"] for record in candidates], dtype=np.float32),
        large_scan=np.asarray([record["scene"]["large_scan"] for record in candidates], dtype=np.int8),
        implementation_delta_norm=np.asarray([record["implementation_delta_norm"] for record in candidates], dtype=np.float32),
        implementation_delta_max=np.asarray([record["implementation_delta_max"] for record in candidates], dtype=np.float32),
        task_mixing_amplitude=np.asarray([record["task_mixing_amplitude"] for record in candidates], dtype=np.float32),
        coherent_mode_amplitude=np.asarray([record["coherent_mode_amplitude"] for record in candidates], dtype=np.float32),
        boundary_metric_code=np.asarray([record["boundary_metric_code"] for record in candidates], dtype=np.int8),
        phase_error_rms_deg=np.zeros(n, dtype=np.float32),
        gain_error_rms_db=np.zeros(n, dtype=np.float32),
        dropout_count=np.zeros(n, dtype=np.int16),
        phase_ramp_deg=np.zeros(n, dtype=np.float32),
        phase_bits=np.full(n, 16, dtype=np.int16),
        amplitude_bits=np.full(n, 16, dtype=np.int16),
        perturbation_seed=np.full(n, -1, dtype=np.int64),
        port_names=np.asarray(base["port_names"]),
        element_ixiy=np.asarray(base["element_ixiy"]),
        positions_lambda=positions,
    )
    write_csv(args.out_dir / "scene_generation_attempts.csv", attempt_rows)
    write_csv(args.out_dir / "candidate_manifest.csv", manifest)

    outside_rows = [row for row in manifest if row["boundary_side"] == "outside"]
    inside_rows = [row for row in manifest if row["boundary_side"] == "inside"]
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "independent_scene_count": len(scene_records),
        "candidate_count": n,
        "expected_hfss_case_count": int(sum(int(row["expected_hfss_case_count"]) for row in manifest)),
        "scene_counts_by_boundary_type": {
            kind: int(sum(scene["boundary_type"] == kind for scene in scene_records))
            for kind in ("psll", "nearest", "local")
        },
        "inside_gate15_pass_count": int(sum(int(row["actual_basis_gate15"]) for row in inside_rows)),
        "outside_gate15_failure_count": int(sum(not int(row["actual_basis_gate15"]) for row in outside_rows)),
        "new_mainlobe_failure_count": int(sum(not int(row["actual_basis_mainlobe_gate"]) for row in manifest)),
        "large_scan_scene_count": int(sum(int(scene["large_scan"]) for scene in scene_records)),
        "target_mode": args.target_mode,
        "excluded_dataset_dirs": [str(path.resolve()) for path in args.exclude_dataset_dir],
        "ratio1_included": False,
        "command_actual_mismatch_enabled": True,
        "runtime_seconds": time.time() - started,
        "hfss_gate_pass": bool(
            len(scene_records) >= 3 * int(args.scenes_per_type)
            and len(inside_rows) == len(scene_records)
            and len(outside_rows) == len(scene_records)
            and all(int(row["actual_basis_gate15"]) == 1 for row in inside_rows)
            and all(int(row["actual_basis_gate15"]) == 0 for row in outside_rows)
            and all(int(row["actual_basis_mainlobe_gate"]) == 1 for row in manifest)
        ),
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
