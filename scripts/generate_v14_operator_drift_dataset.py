#!/usr/bin/env python3
"""Build a scene-grouped 16x16 operator-drift critic development dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import FastPatternEvaluator, metric_at
from generate_v09_eep_development_candidates import (
    MARGIN_NAMES,
    METRIC_NAMES,
    full_active_metrics,
    metric_vector,
    physical_margins,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from refine_trusted_dense_local_eep_joint import DenseExternalEEP
from validate_trusted_eep_hfss_residuals import series_network_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    ROOT / "hfss_outputs" / "v12_k2_operating_envelope_validation_candidates_20260727_run01",
    ROOT / "hfss_outputs" / "v13_k4_operating_envelope_validation_candidates_20260727_run01",
    ROOT / "hfss_outputs" / "v11_operating_envelope_validation_candidates_20260727_run01",
)
DEFAULT_SUPPORT = (
    ROOT
    / "hfss_outputs"
    / "v13_k246_operating_envelope_validation_20260727_run01"
    / "supported_scene_list.csv"
)
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_DRIFT = ROOT / "hfss_outputs" / "v14_operator_drift_4x4_smoke_20260727_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v14_operator_drift_dataset_20260727_run01"
RATIOS = (0.5, 0.6, 0.7, 0.8)
PROFILE_NAMES = (
    "frequency_low",
    "frequency_high",
    "patch_length_low",
    "patch_length_high",
    "dielectric_low",
    "dielectric_high",
)
DRIFT_LEVELS = (0.05, 0.20, 0.50, 1.00)
KMAX = 6
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, action="append", default=[])
    parser.add_argument("--supported-scenes", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--operator-path", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenes-per-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def complex_to_ri(value: np.ndarray) -> np.ndarray:
    return np.stack((value.real, value.imag), axis=-1).astype(np.float32)


def digest_targets(targets: np.ndarray) -> str:
    values = np.ascontiguousarray(np.round(targets, 5), dtype=np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def supported_scene_ids(path: Path) -> set[int]:
    return {
        int(row["sample_index"])
        for row in read_csv(path)
        if int(row["inside_envelope"]) == 1 and int(row["oracle_strict"]) == 1
    }


def select_source(
    packages: list[tuple[Path, dict[str, np.ndarray]]], k_value: int
) -> tuple[Path, dict[str, np.ndarray]]:
    matching = [(path, data) for path, data in packages if np.any(data["k_values"] == k_value)]
    if not matching:
        raise RuntimeError(f"No source package contains K={k_value}")
    purity = [float(np.mean(data["k_values"] == k_value)) for _path, data in matching]
    best = max(purity)
    selected = [pair for pair, value in zip(matching, purity) if value == best]
    if len(selected) != 1:
        raise RuntimeError(f"Ambiguous source package for K={k_value}")
    return selected[0]


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) < count:
        raise RuntimeError(f"Need {count} scenes, only {len(values)} are available")
    positions = np.linspace(0, len(values) - 1, count)
    return [values[int(round(position))] for position in positions]


def select_base_candidates(
    packages: list[tuple[Path, dict[str, np.ndarray]]],
    supported: set[int],
    scenes_per_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for k_value in (2, 4, 6):
        path, data = select_source(packages, k_value)
        sample = np.asarray(data["sample_index"], dtype=np.int64)
        k_values = np.asarray(data["k_values"], dtype=int)
        ratios = np.asarray(data["active_ratios_requested"], dtype=float)
        candidates: list[tuple[float, int]] = []
        for scene in sorted(set(sample[(k_values == k_value) & np.isin(sample, list(supported))])):
            if not all(
                np.any(
                    (sample == scene)
                    & (k_values == k_value)
                    & np.isclose(ratios, ratio, atol=1.0e-5)
                )
                for ratio in RATIOS
            ):
                continue
            member = int(np.flatnonzero(sample == scene)[0])
            difficulty = float(data["max_target_theta_deg"][member]) - 0.5 * float(
                data["min_target_separation_deg"][member]
            )
            candidates.append((difficulty, int(scene)))
        chosen_scenes = evenly_spaced(
            [scene for _difficulty, scene in sorted(candidates)], scenes_per_k
        )
        for local_scene, scene in enumerate(chosen_scenes):
            first_index = int(np.flatnonzero(sample == scene)[0])
            targets = np.asarray(data["targets_deg"][first_index], dtype=np.float32)
            scene_rows.append(
                {
                    "base_sample_index": scene,
                    "k_value": k_value,
                    "target_hash": str(data["target_hashes"][first_index]),
                    "max_target_theta_deg": float(data["max_target_theta_deg"][first_index]),
                    "min_target_separation_deg": float(data["min_target_separation_deg"][first_index]),
                    "source_dataset": str(path.resolve()),
                    "computed_target_hash": digest_targets(targets[:k_value]),
                }
            )
            for ratio in RATIOS:
                members = np.flatnonzero(
                    (sample == scene)
                    & (k_values == k_value)
                    & np.isclose(ratios, ratio, atol=1.0e-5)
                )
                margins = np.asarray(data["nominal_margins"][members], dtype=float)
                floor = np.min(margins, axis=1)
                feasible = np.flatnonzero(floor >= 0.0)
                if feasible.size:
                    if local_scene % 3 == 1:
                        local = int(feasible[np.argmin(floor[feasible])])
                    else:
                        local = int(feasible[np.argmax(floor[feasible])])
                else:
                    local = int(np.argmax(floor))
                index = int(members[local])
                selected.append(
                    {
                        "path": path,
                        "data": data,
                        "index": index,
                        "base_sample_index": scene,
                        "k_value": k_value,
                        "ratio": ratio,
                    }
                )
    return selected, scene_rows


def class_regularized_transfer(nominal: np.ndarray, actual: np.ndarray) -> np.ndarray:
    power = np.abs(nominal) ** 2
    regularizer = max(float(np.max(power)) * 1.0e-5, 1.0e-20)
    return (actual * np.conjugate(nominal) + regularizer) / (power + regularizer)


def bilinear_transfer(transfer_4x4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid_count = transfer_4x4.shape[-1]
    output = np.empty((256, grid_count), dtype=np.complex64)
    mapped_positions = np.empty((256, 2), dtype=np.float64)
    for ix in range(16):
        x = ix * 3.0 / 15.0
        x0 = int(math.floor(x))
        x1 = min(x0 + 1, 3)
        tx = x - x0
        for iy in range(16):
            y = iy * 3.0 / 15.0
            y0 = int(math.floor(y))
            y1 = min(y0 + 1, 3)
            ty = y - y0
            value = (
                (1.0 - tx) * (1.0 - ty) * transfer_4x4[x0, y0]
                + tx * (1.0 - ty) * transfer_4x4[x1, y0]
                + (1.0 - tx) * ty * transfer_4x4[x0, y1]
                + tx * ty * transfer_4x4[x1, y1]
            )
            index = ix * 16 + iy
            output[index] = value
            mapped_positions[index] = ((x - 1.5) * 0.5, (y - 1.5) * 0.5)
    return output, mapped_positions


def best_frequency_phase_sign(
    nominal: dict[str, np.ndarray], actual: dict[str, np.ndarray], frequency_ratio: float
) -> int:
    theta = np.deg2rad(np.asarray(nominal["theta_deg"], dtype=float))
    phi = np.deg2rad(np.asarray(nominal["phi_deg"], dtype=float))
    ux = np.sin(theta) * np.cos(phi)
    uy = np.sin(theta) * np.sin(phi)
    positions = np.asarray(
        [[(ix - 1.5) * 0.5, (iy - 1.5) * 0.5] for ix in range(4) for iy in range(4)]
    )
    phase = 2.0 * np.pi * (frequency_ratio - 1.0) * (
        positions[:, :1] * ux[None, :] + positions[:, 1:] * uy[None, :]
    )
    nominal_field = np.concatenate((nominal["etheta"], nominal["ephi"]), axis=1)
    actual_field = np.concatenate((actual["etheta"], actual["ephi"]), axis=1)
    phase_twice = np.concatenate((phase, phase), axis=1)
    errors = {}
    for sign in (-1, 1):
        predicted = nominal_field * np.exp(1j * sign * phase_twice)
        errors[sign] = float(np.linalg.norm(actual_field - predicted))
    return min(errors, key=errors.get)


def local_s_kernel(delta_s: np.ndarray) -> dict[tuple[int, int], complex]:
    values: dict[tuple[int, int], list[complex]] = {}
    for first in range(16):
        ix, iy = divmod(first, 4)
        for second in range(16):
            jx, jy = divmod(second, 4)
            key = (abs(ix - jx), abs(iy - jy))
            values.setdefault(key, []).append(complex(delta_s[first, second]))
    return {key: complex(np.mean(items)) for key, items in values.items()}


def passive_full_s(base_s: np.ndarray, kernel: dict[tuple[int, int], complex]) -> tuple[np.ndarray, float]:
    delta = np.zeros_like(base_s, dtype=np.complex128)
    for first in range(256):
        ix, iy = divmod(first, 16)
        for second in range(256):
            jx, jy = divmod(second, 16)
            key = (abs(ix - jx), abs(iy - jy))
            if key in kernel:
                delta[first, second] = kernel[key]
    delta = 0.5 * (delta + delta.T)
    low, high = 0.0, 1.0
    for _ in range(24):
        middle = 0.5 * (low + high)
        candidate = 0.5 * ((base_s + middle * delta) + (base_s + middle * delta).T)
        sigma = float(np.max(np.linalg.svd(candidate, compute_uv=False)))
        if sigma <= 0.995:
            low = middle
        else:
            high = middle
    output = 0.5 * ((base_s + low * delta) + (base_s + low * delta).T)
    return output, low


def build_drift_operator(
    base: dict[str, np.ndarray],
    nominal_4: dict[str, np.ndarray],
    actual_4: dict[str, np.ndarray],
    drift_level: float,
) -> tuple[DenseExternalEEP, np.ndarray, dict[str, float]]:
    blended_4 = dict(actual_4)
    for key in ("etheta", "ephi", "s_raw"):
        blended_4[key] = np.asarray(nominal_4[key]) + float(drift_level) * (
            np.asarray(actual_4[key]) - np.asarray(nominal_4[key])
        )
    blended_4["frequency_ghz"] = np.asarray(
        float(nominal_4["frequency_ghz"])
        + float(drift_level)
        * (float(actual_4["frequency_ghz"]) - float(nominal_4["frequency_ghz"]))
    )
    theta_transfer = class_regularized_transfer(nominal_4["etheta"], blended_4["etheta"])
    phi_transfer = class_regularized_transfer(nominal_4["ephi"], blended_4["ephi"])
    theta_full, mapped = bilinear_transfer(theta_transfer.reshape(4, 4, -1))
    phi_full, _mapped = bilinear_transfer(phi_transfer.reshape(4, 4, -1))
    frequency = float(blended_4["frequency_ghz"])
    nominal_frequency = float(nominal_4["frequency_ghz"])
    phase_sign = 0
    if not np.isclose(frequency, nominal_frequency):
        ratio = frequency / nominal_frequency
        phase_sign = best_frequency_phase_sign(nominal_4, actual_4, ratio)
        theta = np.deg2rad(np.asarray(base["theta_deg"], dtype=float))
        phi = np.deg2rad(np.asarray(base["phi_deg"], dtype=float))
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        full_positions = np.asarray(
            [[(ix - 7.5) * 0.5, (iy - 7.5) * 0.5] for ix in range(16) for iy in range(16)]
        )
        delta_position = full_positions - mapped
        phase = 2.0 * np.pi * (ratio - 1.0) * (
            delta_position[:, :1] * ux[None, :] + delta_position[:, 1:] * uy[None, :]
        )
        phase_factor = np.exp(1j * phase_sign * phase).astype(np.complex64)
        theta_full *= phase_factor
        phi_full *= phase_factor
    raw_theta = np.asarray(base["etheta"], dtype=np.complex64) * theta_full
    raw_phi = np.asarray(base["ephi"], dtype=np.complex64) * phi_full
    delta_4 = np.asarray(blended_4["s_raw"], dtype=np.complex128) - np.asarray(
        nominal_4["s_raw"], dtype=np.complex128
    )
    s_raw, projection_scale = passive_full_s(
        np.asarray(base["s_raw"], dtype=np.complex128), local_s_kernel(delta_4)
    )
    s_matched, antenna_map, _series = series_network_map(s_raw, frequency * 1.0e9)
    effective = DenseExternalEEP(raw_theta, raw_phi, antenna_map)
    base_s = np.asarray(base["s_raw"], dtype=np.complex128)
    delta_s = s_raw - base_s
    metadata = {
        "frequency_ghz": frequency,
        "frequency_offset_ghz": frequency - nominal_frequency,
        "s_drift_max_abs": float(np.max(np.abs(delta_s))),
        "s_drift_relative_fro": float(np.linalg.norm(delta_s) / max(np.linalg.norm(base_s), EPS)),
        "s_projection_scale": float(projection_scale),
        "s_passivity_sigma_max": float(np.max(np.linalg.svd(s_raw, compute_uv=False))),
        "frequency_phase_sign": float(phase_sign),
        "drift_intensity": float(drift_level),
    }
    return effective, s_matched, metadata


def calibration_profile(name: str) -> dict[str, float | int]:
    if name.startswith("frequency"):
        return {"phase_rms": 8.0, "gain_rms": 0.60, "group_phase": 5.0, "soft": 2, "compression": 0.18}
    if name.startswith("patch"):
        return {"phase_rms": 6.0, "gain_rms": 0.45, "group_phase": 4.0, "soft": 1, "compression": 0.14}
    return {"phase_rms": 5.0, "gain_rms": 0.35, "group_phase": 3.0, "soft": 1, "compression": 0.10}


def calibration_state(
    name: str,
    scene_seed: int,
    element_ixiy: np.ndarray,
    high_ratio_tasks: np.ndarray,
    drift_level: float,
) -> dict[str, Any]:
    config = calibration_profile(name)
    rng = np.random.default_rng(scene_seed)
    phase = rng.normal(0.0, float(config["phase_rms"]) * drift_level, 256)
    gain = rng.normal(0.0, float(config["gain_rms"]) * drift_level, 256)
    group_bias = rng.normal(0.0, float(config["group_phase"]) * drift_level, 4)
    quadrant = (element_ixiy[:, 0] >= 8).astype(int) * 2 + (element_ixiy[:, 1] >= 8).astype(int)
    phase += group_bias[quadrant]
    factor = 10.0 ** (gain / 20.0) * np.exp(1j * np.deg2rad(phase))
    score = np.sum(np.abs(high_ratio_tasks) ** 2, axis=1)
    soft_count = int(round(float(config["soft"]) * drift_level))
    soft_ports = np.argsort(score, kind="stable")[-soft_count:]
    if soft_count:
        factor[soft_ports] *= 0.25 * np.exp(1j * rng.uniform(-np.pi, np.pi, soft_count))
    bits = 8 if drift_level <= 0.05 else 7 if drift_level <= 0.20 else 6
    return {
        "factor": factor,
        "soft_ports": soft_ports,
        "phase_rms_deg": float(np.sqrt(np.mean(phase**2))),
        "gain_rms_db": float(np.sqrt(np.mean(gain**2))),
        "group_phase_bias_rms_deg": float(np.sqrt(np.mean(group_bias**2))),
        "compression": float(config["compression"]) * drift_level,
        "temperature_offset_c": float(rng.uniform(-25.0, 25.0) * drift_level),
        "phase_bits": bits,
        "amplitude_bits": bits,
        "drift_intensity": float(drift_level),
    }


def apply_calibration(tasks: np.ndarray, mask: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    value = np.asarray(tasks, dtype=np.complex128) * np.asarray(state["factor"])[:, None]
    value[~mask] = 0.0
    amplitude = np.abs(value)
    active_values = amplitude[mask]
    if active_values.size:
        clip = float(np.quantile(active_values, 0.90))
        excess = amplitude > clip
        scale = np.ones_like(amplitude)
        scale[excess] = 1.0 - float(state["compression"]) * (
            1.0 - clip / np.maximum(amplitude[excess], EPS)
        )
        value *= scale
    phase_levels = float(2 ** int(state["phase_bits"]) - 1)
    amplitude_levels = float(2 ** int(state["amplitude_bits"]) - 1)
    phase_step = 2.0 * np.pi / phase_levels
    phase = np.round(np.angle(value) / phase_step) * phase_step
    maximum = max(float(np.max(np.abs(value))), EPS)
    magnitude = np.round(np.abs(value) / maximum * amplitude_levels) / amplitude_levels * maximum
    value = magnitude * np.exp(1j * phase)
    value[~mask] = 0.0
    return value.astype(np.complex64)


def grouped_split(scene_rows: list[dict[str, Any]], seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    split: dict[int, int] = {}
    for k_value in (2, 4, 6):
        scenes = np.asarray(
            [int(row["base_sample_index"]) for row in scene_rows if int(row["k_value"]) == k_value],
            dtype=np.int64,
        )
        rng.shuffle(scenes)
        train_stop = int(round(0.70 * scenes.size))
        val_stop = train_stop + int(round(0.15 * scenes.size))
        for value in scenes[:train_stop]:
            split[int(value)] = 0
        for value in scenes[train_stop:val_stop]:
            split[int(value)] = 1
        for value in scenes[val_stop:]:
            split[int(value)] = 2
    return split


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite operator-drift dataset: {args.out_dir}")
    calibration = json.loads(
        (args.drift_dir / "drift_calibration_summary.json").read_text(encoding="utf-8")
    )
    if not calibration.get("operator_drift_calibration_gate_pass"):
        raise RuntimeError("4x4 operator-drift calibration gate did not pass")
    args.out_dir.mkdir(parents=True)
    dataset_dirs = tuple(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASETS
    packages = [(path, load_npz(path / "dataset_arrays.npz")) for path in dataset_dirs]
    base_candidates, scene_rows = select_base_candidates(
        packages,
        supported_scene_ids(args.supported_scenes),
        int(args.scenes_per_k),
    )
    split_by_base = grouped_split(scene_rows, int(args.seed))
    base_operator = load_npz(args.operator_path)
    nominal_4 = load_npz(
        args.drift_dir / "profiles" / "nominal" / "eep" / "grounded_patch_eep_operator_16port.npz"
    )
    s_nominal, antenna_map_nominal, _series = series_network_map(
        np.asarray(base_operator["s_raw"], dtype=np.complex128), 10.0e9
    )
    nominal_effective = DenseExternalEEP(
        base_operator["etheta"], base_operator["ephi"], antenna_map_nominal
    )
    nominal_fast = FastPatternEvaluator(
        nominal_effective, base_operator["theta_deg"], base_operator["phi_deg"]
    )
    element_ixiy = np.asarray(base_operator["element_ixiy"], dtype=np.int64)

    by_scene: dict[int, list[dict[str, Any]]] = {}
    for candidate in base_candidates:
        by_scene.setdefault(int(candidate["base_sample_index"]), []).append(candidate)
    for values in by_scene.values():
        values.sort(key=lambda item: float(item["ratio"]))

    nominal_cache: dict[tuple[int, float], tuple[dict[str, float], np.ndarray]] = {}
    for base_scene, candidates in by_scene.items():
        first = candidates[0]
        data = first["data"]
        index = int(first["index"])
        k_value = int(first["k_value"])
        targets = np.asarray(data["targets_deg"][index, :k_value], dtype=np.float64)
        tasks = np.stack(
            [
                np.asarray(item["data"]["nominal_external_task_weights_real_imag"][item["index"], :, :k_value, 0], dtype=np.float32)
                + 1j
                * np.asarray(item["data"]["nominal_external_task_weights_real_imag"][item["index"], :, :k_value, 1], dtype=np.float32)
                for item in candidates
            ]
        )
        batch = nominal_fast.evaluate(tasks, targets)
        for position, item in enumerate(candidates):
            metrics = metric_at(batch, position)
            reference = {
                name: float(item["data"]["reference_metrics"][item["index"], metric_index])
                for metric_index, name in enumerate(METRIC_NAMES)
            }
            active = full_active_metrics(tasks[position], np.asarray(item["data"]["masks"][item["index"]], dtype=bool), s_nominal)
            nominal_cache[(base_scene, float(item["ratio"]))] = (
                metrics,
                physical_margins(metrics, reference, active),
            )

    arrays: dict[str, list[np.ndarray | float | int | str]] = {
        key: []
        for key in (
            "sample_index", "base_sample_index", "split_id", "k_values", "ratio", "num_active",
            "targets", "task_valid", "masks", "nominal_tasks", "actual_tasks", "reference_metrics",
            "nominal_metrics", "actual_metrics", "nominal_margins", "actual_margins", "profile",
            "target_hash", "source_index", "source_dir", "min_separation", "max_theta",
            "large_scan", "small_gap", "implementation_delta_norm", "implementation_delta_max",
            "phase_rms", "gain_rms", "dropout_count", "phase_bits", "amplitude_bits",
            "frequency_offset", "patch_offset", "permittivity_offset", "s_drift_fro", "s_drift_max",
            "s_projection_scale", "group_phase_rms", "soft_failure_count", "temperature_offset",
            "drift_intensity",
        )
    }
    manifest: list[dict[str, Any]] = []
    drift_scene_rows: list[dict[str, Any]] = []
    profile_metadata: dict[str, dict[str, float]] = {}
    candidate_index = 0
    for profile_index, profile_name in enumerate(PROFILE_NAMES):
        actual_4 = load_npz(
            args.drift_dir
            / "profiles"
            / profile_name
            / "eep"
            / "grounded_patch_eep_operator_16port.npz"
        )
        for level_index, drift_level in enumerate(DRIFT_LEVELS):
            profile_variant = f"{profile_name}_x{drift_level:.2f}"
            effective, s_matched, metadata = build_drift_operator(
                base_operator, nominal_4, actual_4, drift_level
            )
            profile_metadata[profile_variant] = metadata
            fast = FastPatternEvaluator(
                effective, base_operator["theta_deg"], base_operator["phi_deg"]
            )
            patch_offset = drift_level * {
                "patch_length_low": -0.10,
                "patch_length_high": 0.10,
            }.get(profile_name, 0.0)
            er_offset = drift_level * {"dielectric_low": -0.04, "dielectric_high": 0.04}.get(
                profile_name, 0.0
            )
            for scene_position, (base_scene, candidates) in enumerate(sorted(by_scene.items())):
                first = candidates[0]
                data = first["data"]
                first_index = int(first["index"])
                k_value = int(first["k_value"])
                targets = np.asarray(data["targets_deg"][first_index, :k_value], dtype=np.float64)
                nominal_tasks = np.stack(
                    [
                        np.asarray(item["data"]["nominal_external_task_weights_real_imag"][item["index"], :, :k_value, 0], dtype=np.float32)
                        + 1j
                        * np.asarray(item["data"]["nominal_external_task_weights_real_imag"][item["index"], :, :k_value, 1], dtype=np.float32)
                        for item in candidates
                    ]
                )
                high_ratio = nominal_tasks[-1]
                scene_seed = (
                    int(args.seed)
                    + 100003 * profile_index
                    + 1009 * level_index
                    + 97 * int(base_scene)
                )
                state = calibration_state(
                    profile_name, scene_seed, element_ixiy, high_ratio, drift_level
                )
                actual_tasks = np.stack(
                    [
                        apply_calibration(
                            tasks,
                            np.asarray(item["data"]["masks"][item["index"]], dtype=bool),
                            state,
                        )
                        for tasks, item in zip(nominal_tasks, candidates)
                    ]
                )
                actual_batch = fast.evaluate(actual_tasks, targets)
                drift_scene_id = (
                    1_400_000
                    + profile_index * 100_000
                    + level_index * 1000
                    + scene_position
                )
                drift_scene_rows.append(
                    {
                        "sample_index": drift_scene_id,
                        "base_sample_index": base_scene,
                        "operator_profile": profile_variant,
                        "operator_family": profile_name,
                        "drift_intensity": drift_level,
                        "split": ("train", "val", "test")[split_by_base[base_scene]],
                        "k_value": k_value,
                        "candidate_count": len(candidates),
                    }
                )
                for position, item in enumerate(candidates):
                    source = item["data"]
                    source_index = int(item["index"])
                    mask = np.asarray(source["masks"][source_index], dtype=bool)
                    actual_metrics = metric_at(actual_batch, position)
                    reference = {
                        name: float(source["reference_metrics"][source_index, metric_index])
                        for metric_index, name in enumerate(METRIC_NAMES)
                    }
                    actual_active = full_active_metrics(actual_tasks[position], mask, s_matched)
                    actual_margins = physical_margins(actual_metrics, reference, actual_active)
                    nominal_metrics, nominal_margins = nominal_cache[(base_scene, float(item["ratio"]))]
                    delta = actual_tasks[position] - nominal_tasks[position]
                    nominal_norm = max(float(np.linalg.norm(nominal_tasks[position])), EPS)
                    valid = np.zeros(KMAX, dtype=np.int8)
                    valid[:k_value] = 1
                    targets_full = np.full((KMAX, 2), np.nan, dtype=np.float32)
                    targets_full[:k_value] = targets.astype(np.float32)
                    nominal_full = np.zeros((256, KMAX), dtype=np.complex64)
                    actual_full = np.zeros((256, KMAX), dtype=np.complex64)
                    nominal_full[:, :k_value] = nominal_tasks[position]
                    actual_full[:, :k_value] = actual_tasks[position]
                    soft_active = int(np.sum(mask[np.asarray(state["soft_ports"], dtype=int)]))
                    arrays["sample_index"].append(drift_scene_id)
                    arrays["base_sample_index"].append(base_scene)
                    arrays["split_id"].append(split_by_base[base_scene])
                    arrays["k_values"].append(k_value)
                    arrays["ratio"].append(float(item["ratio"]))
                    arrays["num_active"].append(int(np.sum(mask)))
                    arrays["targets"].append(targets_full)
                    arrays["task_valid"].append(valid)
                    arrays["masks"].append(mask.astype(np.int8))
                    arrays["nominal_tasks"].append(nominal_full)
                    arrays["actual_tasks"].append(actual_full)
                    arrays["reference_metrics"].append(metric_vector(reference))
                    arrays["nominal_metrics"].append(metric_vector(nominal_metrics))
                    arrays["actual_metrics"].append(metric_vector(actual_metrics))
                    arrays["nominal_margins"].append(nominal_margins)
                    arrays["actual_margins"].append(actual_margins)
                    arrays["profile"].append(profile_variant)
                    arrays["target_hash"].append(str(source["target_hashes"][source_index]))
                    arrays["source_index"].append(source_index)
                    arrays["source_dir"].append(str(item["path"].resolve()))
                    arrays["min_separation"].append(float(source["min_target_separation_deg"][source_index]))
                    arrays["max_theta"].append(float(source["max_target_theta_deg"][source_index]))
                    arrays["large_scan"].append(int(source["large_scan"][source_index]))
                    arrays["small_gap"].append(int(source["small_target_gap"][source_index]))
                    arrays["implementation_delta_norm"].append(float(np.linalg.norm(delta) / nominal_norm))
                    arrays["implementation_delta_max"].append(float(np.max(np.abs(delta))))
                    arrays["phase_rms"].append(float(state["phase_rms_deg"]))
                    arrays["gain_rms"].append(float(state["gain_rms_db"]))
                    arrays["dropout_count"].append(soft_active)
                    arrays["phase_bits"].append(int(state["phase_bits"]))
                    arrays["amplitude_bits"].append(int(state["amplitude_bits"]))
                    arrays["frequency_offset"].append(metadata["frequency_offset_ghz"])
                    arrays["patch_offset"].append(patch_offset)
                    arrays["permittivity_offset"].append(er_offset)
                    arrays["s_drift_fro"].append(metadata["s_drift_relative_fro"])
                    arrays["s_drift_max"].append(metadata["s_drift_max_abs"])
                    arrays["s_projection_scale"].append(metadata["s_projection_scale"])
                    arrays["group_phase_rms"].append(float(state["group_phase_bias_rms_deg"]))
                    arrays["soft_failure_count"].append(soft_active)
                    arrays["temperature_offset"].append(float(state["temperature_offset_c"]))
                    arrays["drift_intensity"].append(float(drift_level))
                    strict = int(np.all(actual_margins >= 0.0))
                    nominal_strict = int(np.all(nominal_margins >= 0.0))
                    manifest.append(
                        {
                            "candidate_index": candidate_index,
                            "sample_index": drift_scene_id,
                            "base_sample_index": base_scene,
                            "operator_profile": profile_variant,
                            "operator_family": profile_name,
                            "drift_intensity": drift_level,
                            "k_value": k_value,
                            "ratio": float(item["ratio"]),
                            "split": ("train", "val", "test")[split_by_base[base_scene]],
                            "nominal_strict_gate20": nominal_strict,
                            "actual_strict_gate20": strict,
                            "hard_negative": int(nominal_strict == 1 and strict == 0),
                            "hard_positive": strict,
                            "near_boundary": int(float(np.min(np.abs(actual_margins))) <= 1.5),
                            "actual_psll_db": float(actual_metrics["psll_db"]),
                            "actual_nearest_iso_db": float(actual_metrics["nearest_iso_db"]),
                            "actual_local_iso_db": float(actual_metrics["local_iso_db"]),
                            "actual_active_rl_floor_db": float(actual_active["active_rl_floor_db"]),
                            "worst_actual_margin_db": float(np.min(actual_margins)),
                            "source_candidate_index": source_index,
                        }
                    )
                    candidate_index += 1
            del fast, effective

    nominal_tasks_array = np.stack(arrays["nominal_tasks"]).astype(np.complex64)
    actual_tasks_array = np.stack(arrays["actual_tasks"]).astype(np.complex64)
    nominal_margins_array = np.stack(arrays["nominal_margins"]).astype(np.float32)
    actual_margins_array = np.stack(arrays["actual_margins"]).astype(np.float32)
    strict = np.all(actual_margins_array >= 0.0, axis=1)
    gate15 = (
        (actual_margins_array[:, 0] >= 0.0)
        & (actual_margins_array[:, 1] >= 0.0)
        & (actual_margins_array[:, 2] >= -5.0)
    )
    near = np.min(np.abs(actual_margins_array), axis=1) <= 1.5
    nominal_strict = np.all(nominal_margins_array >= 0.0, axis=1)
    hard_negative = nominal_strict & ~strict
    count = candidate_index
    payload = {
        "candidate_index": np.arange(count, dtype=np.int64),
        "candidate_indices": np.arange(count, dtype=np.int64),
        "sample_index": np.asarray(arrays["sample_index"], dtype=np.int64),
        "sample_indices": np.asarray(arrays["sample_index"], dtype=np.int64),
        "base_sample_index": np.asarray(arrays["base_sample_index"], dtype=np.int64),
        "sample_ids": np.asarray([f"v14_drift_{value}" for value in arrays["sample_index"]]),
        "scene_ids": np.asarray([f"v14_drift_{value}" for value in arrays["sample_index"]]),
        "target_hashes": np.asarray(arrays["target_hash"]),
        "source_dataset": np.asarray(["v14_operator_drift_proxy"] * count),
        "source_sample_indices": np.asarray(arrays["base_sample_index"], dtype=np.int64),
        "selection_roles": np.asarray(["operator_drift"] * count),
        "variant_kind": np.asarray(arrays["profile"]),
        "operator_profile": np.asarray(arrays["profile"]),
        "split_id": np.asarray(arrays["split_id"], dtype=np.int8),
        "k_values": np.asarray(arrays["k_values"], dtype=np.int8),
        "active_ratios_requested": np.asarray(arrays["ratio"], dtype=np.float32),
        "active_ratios_actual": np.asarray(arrays["ratio"], dtype=np.float32),
        "num_active": np.asarray(arrays["num_active"], dtype=np.int16),
        "targets_deg": np.stack(arrays["targets"]).astype(np.float32),
        "task_valid": np.stack(arrays["task_valid"]).astype(np.int8),
        "mask": np.stack(arrays["masks"]).astype(np.int8),
        "masks": np.stack(arrays["masks"]).astype(np.int8),
        "w_tasks_real_imag": complex_to_ri(nominal_tasks_array),
        "task_weights_real_imag": complex_to_ri(np.conjugate(nominal_tasks_array)),
        "w_combined_real_imag": complex_to_ri(np.sum(nominal_tasks_array, axis=2)),
        "combined_weights_real_imag": complex_to_ri(np.conjugate(np.sum(nominal_tasks_array, axis=2))),
        "hfss_actual_task_weights_real_imag": complex_to_ri(np.conjugate(actual_tasks_array)),
        "hfss_actual_combined_weights_real_imag": complex_to_ri(np.conjugate(np.sum(actual_tasks_array, axis=2))),
        "hfss_weights_real_imag": complex_to_ri(np.conjugate(np.sum(actual_tasks_array, axis=2))),
        "nominal_external_task_weights_real_imag": complex_to_ri(nominal_tasks_array),
        "actual_external_task_weights_real_imag": complex_to_ri(actual_tasks_array),
        "reference_metrics": np.stack(arrays["reference_metrics"]).astype(np.float32),
        "nominal_metrics": np.stack(arrays["nominal_metrics"]).astype(np.float32),
        "actual_metrics": np.stack(arrays["actual_metrics"]).astype(np.float32),
        "metric_names": METRIC_NAMES,
        "nominal_margins": nominal_margins_array,
        "actual_margins": actual_margins_array,
        "margin_residuals": actual_margins_array - nominal_margins_array,
        "margin_names": MARGIN_NAMES,
        "gate15": gate15.astype(np.int8),
        "strict_gate20": strict.astype(np.int8),
        "mainlobe_gate": (actual_margins_array[:, 3] >= 0.0).astype(np.int8),
        "active_rl_gate": (actual_margins_array[:, 4] >= 0.0).astype(np.int8),
        "near_boundary": near.astype(np.int8),
        "hard_negative": hard_negative.astype(np.int8),
        "hard_positive": strict.astype(np.int8),
        "strict_violation": np.maximum(-actual_margins_array, 0.0).sum(axis=1).astype(np.float32),
        "min_target_separation_deg": np.asarray(arrays["min_separation"], dtype=np.float32),
        "max_target_theta_deg": np.asarray(arrays["max_theta"], dtype=np.float32),
        "large_scan": np.asarray(arrays["large_scan"], dtype=np.int8),
        "small_target_gap": np.asarray(arrays["small_gap"], dtype=np.int8),
        "implementation_delta_norm": np.asarray(arrays["implementation_delta_norm"], dtype=np.float32),
        "implementation_delta_max": np.asarray(arrays["implementation_delta_max"], dtype=np.float32),
        "phase_error_rms_deg": np.asarray(arrays["phase_rms"], dtype=np.float32),
        "gain_error_rms_db": np.asarray(arrays["gain_rms"], dtype=np.float32),
        "dropout_count": np.asarray(arrays["dropout_count"], dtype=np.int16),
        "peak_shape_margin_db": np.zeros(count, dtype=np.float32),
        "phase_bits": np.asarray(arrays["phase_bits"], dtype=np.int16),
        "amplitude_bits": np.asarray(arrays["amplitude_bits"], dtype=np.int16),
        "frequency_offset_ghz": np.asarray(arrays["frequency_offset"], dtype=np.float32),
        "patch_length_offset_mm": np.asarray(arrays["patch_offset"], dtype=np.float32),
        "relative_permittivity_offset": np.asarray(arrays["permittivity_offset"], dtype=np.float32),
        "s_drift_relative_fro": np.asarray(arrays["s_drift_fro"], dtype=np.float32),
        "s_drift_max_abs": np.asarray(arrays["s_drift_max"], dtype=np.float32),
        "s_projection_scale": np.asarray(arrays["s_projection_scale"], dtype=np.float32),
        "group_phase_bias_rms_deg": np.asarray(arrays["group_phase_rms"], dtype=np.float32),
        "soft_failure_count": np.asarray(arrays["soft_failure_count"], dtype=np.int16),
        "temperature_offset_c": np.asarray(arrays["temperature_offset"], dtype=np.float32),
        "drift_intensity": np.asarray(arrays["drift_intensity"], dtype=np.float32),
        "source_candidate_indices": np.asarray(arrays["source_index"], dtype=np.int64),
        "source_dataset_dirs": np.asarray(arrays["source_dir"]),
        "port_names": np.asarray(base_operator["port_names"]),
        "element_ixiy": element_ixiy,
        "positions_lambda": np.asarray(
            [[(ix - 7.5) * 0.5, (iy - 7.5) * 0.5, 0.0] for ix in range(16) for iy in range(16)],
            dtype=np.float64,
        ),
    }
    np.savez_compressed(args.out_dir / "dataset_arrays.npz", **payload)
    write_csv(args.out_dir / "candidate_manifest.csv", manifest)
    write_csv(args.out_dir / "base_scene_manifest.csv", scene_rows)
    write_csv(args.out_dir / "drift_scene_manifest.csv", drift_scene_rows)
    split_summary = []
    for split_id, split_name in enumerate(("train", "val", "test")):
        member = np.asarray(payload["split_id"]) == split_id
        split_summary.append(
            {
                "split": split_name,
                "base_scene_count": int(np.unique(payload["base_sample_index"][member]).size),
                "drift_scene_count": int(np.unique(payload["sample_index"][member]).size),
                "candidate_count": int(np.sum(member)),
                "strict_positive_count": int(np.sum(strict[member])),
                "strict_negative_count": int(np.sum(~strict[member])),
            }
        )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": count,
        "base_scene_count": int(np.unique(payload["base_sample_index"]).size),
        "drift_scene_count": int(np.unique(payload["sample_index"]).size),
        "k_base_scene_counts": {
            str(k): sum(int(row["k_value"]) == k for row in scene_rows) for k in (2, 4, 6)
        },
        "profile_counts": dict(Counter(str(value) for value in arrays["profile"])),
        "drift_intensity_counts": dict(
            Counter(f"{float(value):.2f}" for value in arrays["drift_intensity"])
        ),
        "strict_positive_count": int(np.sum(strict)),
        "strict_negative_count": int(np.sum(~strict)),
        "hard_negative_count": int(np.sum(hard_negative)),
        "near_boundary_count": int(np.sum(near)),
        "scene_oracle_rate": float(
            np.mean(
                [
                    np.any(strict[np.asarray(payload["sample_index"]) == scene])
                    for scene in np.unique(payload["sample_index"])
                ]
            )
        ),
        "split": split_summary,
        "base_scene_leakage_free": True,
        "profile_metadata": profile_metadata,
        "label_scope": "4x4-HFSS-calibrated 16x16 EEP/S256 proxy; not 16x16 HFSS",
        "critic_promotion_requires_16x16_hfss": True,
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
