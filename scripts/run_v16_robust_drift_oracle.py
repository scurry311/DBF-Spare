#!/usr/bin/env python3
"""Build and evaluate the preregistered v1.6 multi-corner robust oracle.

This is an EEP/S256 development experiment.  A candidate owns one mask and
one commanded task-weight matrix.  Hardware-corner calibration is applied
only while evaluating that command; corner-specific commands are never saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from generate_expanded_independent_residual_scenes import target_hash
from generate_gate15_boundary_scenes import FastPatternEvaluator, metric_at
from generate_iso_lcmv_teacher import mutate_mask
from generate_v09_eep_development_candidates import (
    MARGIN_NAMES,
    METRIC_NAMES,
    full_active_metrics,
    matched_steering_tasks,
    metric_vector,
    physical_margins,
)
from generate_v14_operator_drift_dataset import (
    PROFILE_NAMES,
    apply_calibration,
    build_drift_operator,
    calibration_state,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from refine_trusted_dense_local_eep_joint import (
    DenseConfig,
    DenseExternalEEP,
    build_constraints,
    nearest_grid_index,
    refine_one,
)
from validate_trusted_eep_hfss_residuals import series_network_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v16_robust_drift_oracle_preregistered.json"
DEFAULT_NEW_POOL = ROOT / "hfss_outputs" / "v16_robust_new_scenes_20260727_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v16_robust_drift_oracle_20260727_run01"
DEFAULT_BASE_SCENES = (
    ROOT / "hfss_outputs" / "v14_operator_drift_dataset_20260727_run03" / "base_scene_manifest.csv"
)
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_DRIFT = ROOT / "hfss_outputs" / "v14_operator_drift_4x4_smoke_20260727_run01"
DEFAULT_FROZEN_DATASETS = (
    ROOT / "hfss_outputs" / "v09_second_prospective_eep_candidates_20260726_run01",
    ROOT / "hfss_outputs" / "v13_frozen_k246_hfss_smoke_dataset_20260727_run01",
    ROOT / "hfss_outputs" / "v14_operator_drift_dataset_20260727_run03",
)
RATIOS = (0.5, 0.6, 0.7, 0.8)
LEVELS = (0.05, 0.20, 0.50, 1.00)
ENVELOPE_BY_LEVEL = {0.05: "E1", 0.20: "E2", 0.50: "E3", 1.00: "E3"}
KMAX = 6
EPS = 1.0e-12
ROBUST_MARGIN_NAMES = np.concatenate((MARGIN_NAMES, np.asarray(["hardware"])))
AUGMENT_CONFIG = DenseConfig(
    "v16_mask_augment",
    True,
    7,
    2,
    0.014,
    40.0,
    10.0,
    4.0,
    -20.0,
    1,
    48,
    2.0,
)
ROBUST_CONFIG = DenseConfig(
    "v16_common_command_corner_projection",
    True,
    4,
    2,
    0.012,
    44.0,
    12.0,
    5.0,
    -18.0,
    1,
    64,
    2.5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "screen",
            "evaluate-initial",
            "refine",
            "evaluate-final",
            "rescue",
            "evaluate-rescue",
            "all",
        ),
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--new-pool", type=Path, default=DEFAULT_NEW_POOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--existing-scenes", type=Path, default=DEFAULT_BASE_SCENES)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--selected-per-ratio", type=int, default=4)
    parser.add_argument("--augment-per-ratio", type=int, default=8)
    parser.add_argument("--refine-sweeps", type=int, default=2)
    parser.add_argument("--rescue-masks-per-ratio", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def complex_to_ri(values: np.ndarray) -> np.ndarray:
    return np.stack((values.real, values.imag), axis=-1).astype(np.float32)


def ri_to_complex(values: np.ndarray) -> np.ndarray:
    return (
        np.asarray(values[..., 0], dtype=np.float32)
        + 1j * np.asarray(values[..., 1], dtype=np.float32)
    ).astype(np.complex64)


def target_digest(targets: np.ndarray) -> str:
    return target_hash(np.asarray(targets, dtype=np.float64))


def mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()[:16]


def dataset_hashes(path: Path) -> set[str]:
    source = path / "dataset_arrays.npz" if path.is_dir() else path
    if not source.exists():
        return set()
    data = load_npz(source)
    targets_key = "targets_deg" if "targets_deg" in data else "targets" if "targets" in data else None
    if targets_key is None:
        return set()
    k_values = np.asarray(data.get("k_values", data.get("task_valid").sum(axis=1)), dtype=int)
    return {
        target_digest(np.asarray(data[targets_key][index, : int(k_value)], dtype=float))
        for index, k_value in enumerate(k_values)
    }


def load_nominal_operator(path: Path) -> tuple[dict[str, np.ndarray], DenseExternalEEP, FastPatternEvaluator, np.ndarray]:
    base = load_npz(path)
    if "s_matched" in base:
        s_matrix, antenna_map, _series = series_network_map(
            np.asarray(base["s_raw"], dtype=np.complex128),
            float(base["frequency_ghz"]) * 1.0e9,
        )
        effective = DenseExternalEEP(base["etheta"], base["ephi"], antenna_map)
        if float(np.max(np.abs(s_matrix - np.asarray(base["s_matched"])))) > 1.0e-6:
            raise RuntimeError("Reconstructed matched S256 does not match the trusted export")
    else:
        raise RuntimeError("The nominal operator does not contain matched S256")
    fast = FastPatternEvaluator(effective, base["theta_deg"], base["phi_deg"])
    return base, effective, fast, s_matrix


def source_tasks(data: dict[str, np.ndarray], indices: np.ndarray, k_value: int) -> np.ndarray:
    key = (
        "nominal_external_task_weights_real_imag"
        if "nominal_external_task_weights_real_imag" in data
        else "task_weights_real_imag"
    )
    return ri_to_complex(data[key][indices, :, :k_value])


def normalize_task_norms(tasks: np.ndarray, reference: np.ndarray) -> np.ndarray:
    output = np.asarray(tasks, dtype=np.complex128).copy()
    for task_index in range(output.shape[1]):
        desired = max(float(np.linalg.norm(reference[:, task_index])), EPS)
        output[:, task_index] *= desired / max(float(np.linalg.norm(output[:, task_index])), EPS)
    return output.astype(np.complex64)


def append_candidate(
    store: dict[str, list[Any]],
    *,
    sample_index: int,
    k_value: int,
    ratio: float,
    targets: np.ndarray,
    mask: np.ndarray,
    tasks: np.ndarray,
    scene_origin: str,
    candidate_origin: str,
    source_dataset: str,
    source_index: int,
) -> None:
    padded_targets = np.full((KMAX, 2), np.nan, dtype=np.float32)
    padded_targets[:k_value] = np.asarray(targets, dtype=np.float32)
    padded_tasks = np.zeros((256, KMAX), dtype=np.complex64)
    padded_tasks[:, :k_value] = np.asarray(tasks, dtype=np.complex64)
    store["sample_index"].append(int(sample_index))
    store["k_values"].append(int(k_value))
    store["ratio"].append(float(ratio))
    store["targets"].append(padded_targets)
    store["masks"].append(np.asarray(mask, dtype=np.int8))
    store["tasks"].append(padded_tasks)
    store["target_hash"].append(target_digest(targets))
    store["mask_hash"].append(mask_digest(mask))
    store["scene_origin"].append(scene_origin)
    store["candidate_origin"].append(candidate_origin)
    store["source_dataset"].append(source_dataset)
    store["source_index"].append(int(source_index))


def prepare_pool(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    pool_dir = args.out_dir / "pool"
    if pool_dir.exists() and any(pool_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prepared pool: {pool_dir}")
    pool_dir.mkdir(parents=True, exist_ok=True)
    base_operator, nominal_effective, _nominal_fast, nominal_s = load_nominal_operator(args.operator)
    grid_dirs = pattern_grid_dirs(base_operator["theta_deg"], base_operator["phi_deg"])
    scene_rows = read_csv(args.existing_scenes)
    if args.max_scenes > 0:
        scene_rows = scene_rows[: int(args.max_scenes)]
    package_cache: dict[str, dict[str, np.ndarray]] = {}
    store: dict[str, list[Any]] = defaultdict(list)
    scene_manifest: list[dict[str, Any]] = []
    started = time.time()

    for scene_position, row in enumerate(scene_rows):
        package_path = Path(row["source_dataset"])
        key = str(package_path.resolve())
        if key not in package_cache:
            package_cache[key] = load_npz(package_path / "dataset_arrays.npz")
        data = package_cache[key]
        sample_index = int(row["base_sample_index"])
        all_members = np.flatnonzero(np.asarray(data["sample_index"], dtype=np.int64) == sample_index)
        if all_members.size != 96:
            raise RuntimeError(f"Expected 96 source candidates for scene {sample_index}, got {all_members.size}")
        k_value = int(row["k_value"])
        first = int(all_members[0])
        targets = np.asarray(data["targets_deg"][first, :k_value], dtype=np.float64)
        scene_manifest.append(
            {
                "sample_index": sample_index,
                "k_value": k_value,
                "target_hash": target_digest(targets),
                "scene_origin": "existing45",
                "max_target_theta_deg": float(np.max(targets[:, 0])),
                "min_target_separation_deg": float(row["min_target_separation_deg"]),
                "source_dataset": key,
            }
        )
        for ratio in RATIOS:
            members = all_members[
                np.isclose(
                    np.asarray(data["active_ratios_requested"][all_members], dtype=float),
                    ratio,
                    atol=1.0e-5,
                )
            ]
            tasks_batch = source_tasks(data, members, k_value)
            for member, tasks in zip(members, tasks_batch):
                append_candidate(
                    store,
                    sample_index=sample_index,
                    k_value=k_value,
                    ratio=ratio,
                    targets=targets,
                    mask=np.asarray(data["masks"][member], dtype=bool),
                    tasks=tasks,
                    scene_origin="existing45",
                    candidate_origin=str(data.get("variant_kind", np.full(len(data["sample_index"]), "historical"))[member]),
                    source_dataset=key,
                    source_index=int(member),
                )

            floors = np.min(np.asarray(data["nominal_margins"][members], dtype=float), axis=1)
            parent_order = np.argsort(floors, kind="stable")[::-1]
            seen = {
                mask_digest(np.asarray(data["masks"][member], dtype=bool)) for member in members
            }
            rng = np.random.default_rng(args.seed + 1009 * sample_index + int(round(ratio * 100)))
            made = 0
            attempts = 0
            while made < int(args.augment_per_ratio):
                parent_local = int(parent_order[attempts % len(parent_order)])
                parent_index = int(members[parent_local])
                parent_mask = np.asarray(data["masks"][parent_index], dtype=bool)
                candidate_mask = mutate_mask(
                    parent_mask,
                    rng,
                    max_swaps=2 + attempts % 8,
                )
                digest = mask_digest(candidate_mask)
                attempts += 1
                if digest in seen:
                    continue
                seen.add(digest)
                parent_tasks = source_tasks(data, np.asarray([parent_index]), k_value)[0]
                steering = matched_steering_tasks(parent_tasks, targets, nominal_effective, grid_dirs)
                warm = 0.65 * parent_tasks + 0.35 * steering
                warm[~candidate_mask] = 0.0
                warm = normalize_task_norms(warm, parent_tasks)
                constraints, combined, _stats = build_constraints(
                    warm,
                    candidate_mask,
                    targets,
                    grid_dirs,
                    nominal_effective,
                    local_radius_deg=5.0,
                    nearest_isolation_db=25.0,
                    local_isolation_db=20.0,
                )
                refined, _diagnostics = refine_one(
                    warm,
                    warm,
                    candidate_mask,
                    constraints,
                    combined,
                    nominal_s,
                    AUGMENT_CONFIG,
                    rl_min_db=11.0,
                    task_relative_db=-20.0,
                )
                append_candidate(
                    store,
                    sample_index=sample_index,
                    k_value=k_value,
                    ratio=ratio,
                    targets=targets,
                    mask=candidate_mask,
                    tasks=refined,
                    scene_origin="existing45",
                    candidate_origin=f"local_swap_augment_{made:02d}",
                    source_dataset=key,
                    source_index=parent_index,
                )
                made += 1
        print(
            f"existing scene {scene_position + 1:03d}/{len(scene_rows):03d} "
            f"K={k_value} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    new_data = load_npz(args.new_pool / "dataset_arrays.npz")
    new_samples = sorted(set(np.asarray(new_data["sample_index"], dtype=np.int64).tolist()))
    if args.max_scenes > 0:
        new_samples = new_samples[: max(0, int(args.max_scenes) - len(scene_rows))]
    for sample_index in new_samples:
        members = np.flatnonzero(np.asarray(new_data["sample_index"], dtype=np.int64) == sample_index)
        first = int(members[0])
        k_value = int(new_data["k_values"][first])
        targets = np.asarray(new_data["targets_deg"][first, :k_value], dtype=np.float64)
        scene_manifest.append(
            {
                "sample_index": int(sample_index),
                "k_value": k_value,
                "target_hash": target_digest(targets),
                "scene_origin": "new30",
                "max_target_theta_deg": float(new_data["max_target_theta_deg"][first]),
                "min_target_separation_deg": float(new_data["min_target_separation_deg"][first]),
                "source_dataset": str(args.new_pool.resolve()),
            }
        )
        tasks_batch = source_tasks(new_data, members, k_value)
        for member, tasks in zip(members, tasks_batch):
            append_candidate(
                store,
                sample_index=int(sample_index),
                k_value=k_value,
                ratio=float(new_data["active_ratios_requested"][member]),
                targets=targets,
                mask=np.asarray(new_data["masks"][member], dtype=bool),
                tasks=tasks,
                scene_origin="new30",
                candidate_origin=str(new_data["variant_kind"][member]),
                source_dataset=str(args.new_pool.resolve()),
                source_index=int(member),
            )

    arrays = {
        "candidate_index": np.arange(len(store["sample_index"]), dtype=np.int64),
        "sample_index": np.asarray(store["sample_index"], dtype=np.int64),
        "k_values": np.asarray(store["k_values"], dtype=np.int8),
        "ratio": np.asarray(store["ratio"], dtype=np.float32),
        "targets": np.stack(store["targets"]),
        "masks": np.stack(store["masks"]),
        "tasks_real_imag": complex_to_ri(np.stack(store["tasks"])),
        "target_hash": np.asarray(store["target_hash"]),
        "mask_hash": np.asarray(store["mask_hash"]),
        "scene_origin": np.asarray(store["scene_origin"]),
        "candidate_origin": np.asarray(store["candidate_origin"]),
        "source_dataset": np.asarray(store["source_dataset"]),
        "source_index": np.asarray(store["source_index"], dtype=np.int64),
        "margin_names": MARGIN_NAMES,
        "metric_names": METRIC_NAMES,
    }
    np.savez_compressed(pool_dir / "candidate_pool.npz", **arrays)
    write_csv(pool_dir / "scene_manifest.csv", scene_manifest)
    protocol_copy = pool_dir / "preregistered_protocol.json"
    protocol_copy.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    new_hashes = {row["target_hash"] for row in scene_manifest if row["scene_origin"] == "new30"}
    frozen_overlaps: dict[str, int] = {}
    for path in DEFAULT_FROZEN_DATASETS:
        frozen_overlaps[str(path.resolve())] = len(new_hashes & dataset_hashes(path))
    counts = Counter(
        (int(k), str(origin)) for k, origin in zip(arrays["k_values"], arrays["scene_origin"])
    )
    unique_by_scene_ratio = []
    for sample in sorted(set(arrays["sample_index"].tolist())):
        for ratio in RATIOS:
            members = np.flatnonzero(
                (arrays["sample_index"] == sample) & np.isclose(arrays["ratio"], ratio, atol=1.0e-5)
            )
            unique_by_scene_ratio.append(
                len(set(np.asarray(arrays["mask_hash"])[members].tolist()))
            )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": int(len(arrays["candidate_index"])),
        "scene_count": int(len(scene_manifest)),
        "existing_scene_count": int(sum(row["scene_origin"] == "existing45" for row in scene_manifest)),
        "new_scene_count": int(sum(row["scene_origin"] == "new30" for row in scene_manifest)),
        "new_scene_counts_by_k": {
            str(k): int(sum(row["scene_origin"] == "new30" and int(row["k_value"]) == k for row in scene_manifest))
            for k in (2, 4, 6)
        },
        "unique_masks_per_scene_ratio_min": int(min(unique_by_scene_ratio)),
        "unique_masks_per_scene_ratio_max": int(max(unique_by_scene_ratio)),
        "frozen_target_hash_overlaps": frozen_overlaps,
        "target_hash_overlap_gate_pass": bool(max(frozen_overlaps.values(), default=0) == 0),
        "ratio_1_present": bool(np.any(np.isclose(arrays["ratio"], 1.0))),
        "elapsed_seconds": time.time() - started,
        "evidence_scope": protocol["evidence_scope"],
    }
    (pool_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_corners(
    args: argparse.Namespace,
    *,
    levels: tuple[float, ...] = LEVELS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    base = load_npz(args.operator)
    nominal_4 = load_npz(
        args.drift_dir / "profiles" / "nominal" / "eep" / "grounded_patch_eep_operator_16port.npz"
    )
    corners: dict[str, dict[str, Any]] = {}
    for profile in PROFILE_NAMES:
        actual_4 = load_npz(
            args.drift_dir / "profiles" / profile / "eep" / "grounded_patch_eep_operator_16port.npz"
        )
        for level in levels:
            effective, s_matrix, metadata = build_drift_operator(base, nominal_4, actual_4, level)
            name = f"{profile}_x{level:.2f}"
            corners[name] = {
                "name": name,
                "profile": profile,
                "level": level,
                "envelope": ENVELOPE_BY_LEVEL[level],
                "effective": effective,
                "fast": FastPatternEvaluator(effective, base["theta_deg"], base["phi_deg"]),
                "s": s_matrix,
                "metadata": metadata,
            }
    return base, corners


def scene_calibration_states(
    pool: dict[str, np.ndarray],
    scene_members: np.ndarray,
    corners: dict[str, dict[str, Any]],
    element_ixiy: np.ndarray,
    seed: int,
) -> dict[str, dict[str, Any]]:
    ratios = np.asarray(pool["ratio"][scene_members], dtype=float)
    high = scene_members[np.flatnonzero(np.isclose(ratios, max(RATIOS), atol=1.0e-5))[0]]
    k_value = int(pool["k_values"][high])
    high_tasks = ri_to_complex(pool["tasks_real_imag"][high, :, :k_value])
    sample = int(pool["sample_index"][high])
    states: dict[str, dict[str, Any]] = {}
    for name, corner in corners.items():
        profile_index = PROFILE_NAMES.index(str(corner["profile"]))
        level_index = LEVELS.index(float(corner["level"]))
        states[name] = calibration_state(
            str(corner["profile"]),
            int(seed + sample * 97 + profile_index * 100003 + level_index * 1009),
            element_ixiy,
            high_tasks,
            float(corner["level"]),
        )
    return states


def nominal_candidate_metrics(
    pool: dict[str, np.ndarray],
    nominal_fast: FastPatternEvaluator,
    nominal_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(pool["candidate_index"])
    metrics = np.zeros((count, len(METRIC_NAMES)), dtype=np.float32)
    margins = np.zeros((count, len(MARGIN_NAMES)), dtype=np.float32)
    samples = np.asarray(pool["sample_index"], dtype=np.int64)
    for position, sample in enumerate(sorted(set(samples.tolist()))):
        members = np.flatnonzero(samples == sample)
        k_value = int(pool["k_values"][members[0]])
        targets = np.asarray(pool["targets"][members[0], :k_value], dtype=float)
        for start in range(0, len(members), 32):
            batch_members = members[start : start + 32]
            tasks = ri_to_complex(pool["tasks_real_imag"][batch_members, :, :k_value])
            batch_metrics = nominal_fast.evaluate(tasks, targets)
            for local, candidate in enumerate(batch_members):
                item = metric_at(batch_metrics, local)
                active = full_active_metrics(
                    tasks[local], np.asarray(pool["masks"][candidate], dtype=bool), nominal_s
                )
                metrics[candidate] = metric_vector(item)
                margins[candidate] = physical_margins(item, item, active)
        print(f"nominal metrics scene {position + 1:03d}", flush=True)
    return metrics, margins


def screen_pool(args: argparse.Namespace) -> None:
    screen_dir = args.out_dir / "screen"
    if screen_dir.exists() and any(screen_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite screening: {screen_dir}")
    screen_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    strict_gates = protocol["strict_corner_gate"]
    pool = load_npz(args.out_dir / "pool" / "candidate_pool.npz")
    base, nominal_effective, nominal_fast, nominal_s = load_nominal_operator(args.operator)
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    _base, e2_corners = build_corners(args, levels=(0.20,))
    nominal_metrics, nominal_margins = nominal_candidate_metrics(pool, nominal_fast, nominal_s)
    nominal_hardware_margin = np.zeros(len(pool["candidate_index"]), dtype=np.float32)
    for candidate in range(len(pool["candidate_index"])):
        k_value = int(pool["k_values"][candidate])
        tasks = ri_to_complex(pool["tasks_real_imag"][candidate, :, :k_value])
        mask = np.asarray(pool["masks"][candidate], dtype=bool)
        targets = np.asarray(pool["targets"][candidate, :k_value], dtype=float)
        nominal_hardware_margin[candidate], _metrics = hardware_margin(
            tasks,
            tasks,
            mask,
            targets,
            nominal_effective,
            grid_dirs,
            strict_gates,
        )
    worst_e2_rl_margin = np.full(len(pool["candidate_index"]), np.inf, dtype=np.float32)
    samples = np.asarray(pool["sample_index"], dtype=np.int64)
    for scene_position, sample in enumerate(sorted(set(samples.tolist()))):
        members = np.flatnonzero(samples == sample)
        k_value = int(pool["k_values"][members[0]])
        states = scene_calibration_states(pool, members, e2_corners, base["element_ixiy"], args.seed)
        for name, corner in e2_corners.items():
            state = states[name]
            for candidate in members:
                tasks = ri_to_complex(pool["tasks_real_imag"][candidate, :, :k_value])
                mask = np.asarray(pool["masks"][candidate], dtype=bool)
                actual = apply_calibration(tasks, mask, state)
                active = full_active_metrics(actual, mask, corner["s"])
                worst_e2_rl_margin[candidate] = min(
                    worst_e2_rl_margin[candidate], float(active["active_rl_floor_db"]) - 10.0
                )
        print(f"E2 RL screen scene {scene_position + 1:03d}", flush=True)

    screen_margin = np.minimum.reduce(
        (np.min(nominal_margins[:, :4], axis=1), worst_e2_rl_margin, nominal_hardware_margin)
    )
    selected: list[int] = []
    rank_rows: list[dict[str, Any]] = []
    for sample in sorted(set(samples.tolist())):
        for ratio in RATIOS:
            members = np.flatnonzero(
                (samples == sample) & np.isclose(pool["ratio"], ratio, atol=1.0e-5)
            )
            order = members[np.argsort(screen_margin[members], kind="stable")[::-1]]
            chosen = order[: max(1, int(args.selected_per_ratio) - 1)].tolist()
            remaining = np.asarray([value for value in members if int(value) not in chosen], dtype=int)
            if len(chosen) < int(args.selected_per_ratio) and remaining.size:
                uncertainty = remaining[np.argmin(np.abs(screen_margin[remaining]))]
                chosen.append(int(uncertainty))
            selected.extend(chosen)
            for local_rank, candidate in enumerate(order):
                rank_rows.append(
                    {
                        "candidate_index": int(candidate),
                        "sample_index": int(sample),
                        "k_value": int(pool["k_values"][candidate]),
                        "ratio": float(ratio),
                        "rank": local_rank + 1,
                        "selected_for_dense_corners": int(int(candidate) in chosen),
                        "screen_worst_margin_db": float(screen_margin[candidate]),
                        "nominal_worst_pattern_margin_db": float(np.min(nominal_margins[candidate, :4])),
                        "E2_worst_active_rl_margin_db": float(worst_e2_rl_margin[candidate]),
                        "nominal_hardware_margin_db": float(nominal_hardware_margin[candidate]),
                        "candidate_origin": str(pool["candidate_origin"][candidate]),
                    }
                )
    np.savez_compressed(
        screen_dir / "screening_arrays.npz",
        selected_candidate_indices=np.asarray(sorted(set(selected)), dtype=np.int64),
        nominal_metrics=nominal_metrics,
        nominal_margins=nominal_margins,
        worst_e2_active_rl_margin=worst_e2_rl_margin,
        nominal_hardware_margin=nominal_hardware_margin,
        screen_margin=screen_margin,
    )
    write_csv(screen_dir / "candidate_screening.csv", rank_rows)
    summary = {
        "candidate_count": int(len(pool["candidate_index"])),
        "dense_selected_count": int(len(set(selected))),
        "selected_per_scene_ratio": int(args.selected_per_ratio),
        "selection_policy": "top physics margins plus one near-boundary uncertainty candidate",
        "uses_HFSS_labels": False,
    }
    (screen_dir / "screen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def apply_analog_state(tasks: np.ndarray, mask: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    value = np.asarray(tasks, dtype=np.complex128) * np.asarray(state["factor"])[:, None]
    value[~mask] = 0.0
    return value.astype(np.complex64)


def hardware_margin(
    command: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    effective: DenseExternalEEP,
    grid_dirs: np.ndarray,
    gates: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    combined = np.sum(np.asarray(actual, dtype=np.complex128), axis=1)
    active_amplitude = np.abs(combined[mask])
    maximum = max(float(np.max(active_amplitude)), EPS)
    minimum = max(float(np.min(active_amplitude)), EPS)
    rms = max(float(np.sqrt(np.mean(active_amplitude**2))), EPS)
    dynamic_range_db = float(20.0 * np.log10(maximum / minimum))
    peak_to_rms_db = float(20.0 * np.log10(maximum / rms))
    command_power = max(float(np.sum(np.abs(command) ** 2)), EPS)
    actual_power = max(float(np.sum(np.abs(actual) ** 2)), EPS)
    power_increase_db = float(10.0 * np.log10(actual_power / command_power))
    wng_values: list[float] = []
    active = np.flatnonzero(mask)
    for task_index, (theta, phi) in enumerate(targets):
        center = nearest_grid_index(grid_dirs, float(theta), float(phi))
        row_theta, row_phi = effective.point_rows(center, active)
        value = np.asarray(actual[active, task_index], dtype=np.complex128)
        field_power = abs(complex(value @ row_theta)) ** 2 + abs(complex(value @ row_phi)) ** 2
        denominator = max(
            float(np.vdot(value, value).real)
            * float(np.vdot(row_theta, row_theta).real + np.vdot(row_phi, row_phi).real),
            EPS,
        )
        wng_values.append(float(10.0 * np.log10(max(field_power / denominator, 1.0e-30))))
    normalized_wng_db = min(wng_values)
    margins = (
        float(gates["combined_channel_dynamic_range_max_db"]) - dynamic_range_db,
        float(gates["combined_peak_to_rms_max_db"]) - peak_to_rms_db,
        float(gates["corner_power_increase_max_db"]) - power_increase_db,
        normalized_wng_db - float(gates["normalized_wng_min_db"]),
    )
    return float(min(margins)), {
        "combined_dynamic_range_db": dynamic_range_db,
        "combined_peak_to_rms_db": peak_to_rms_db,
        "corner_power_increase_db": power_increase_db,
        "normalized_wng_db": normalized_wng_db,
    }


def robust_refine_command(
    command: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    grid_dirs: np.ndarray,
    sweeps: int,
) -> np.ndarray:
    current = np.asarray(command, dtype=np.complex64).copy()
    norm_reference = current.copy()
    for _sweep in range(max(1, sweeps)):
        for name, corner in corners.items():
            state = states[name]
            factor = np.asarray(state["factor"], dtype=np.complex128)
            actual = apply_analog_state(current, mask, state)
            constraints, combined, _stats = build_constraints(
                actual,
                mask,
                targets,
                grid_dirs,
                corner["effective"],
                local_radius_deg=5.0,
                nearest_isolation_db=25.0,
                local_isolation_db=20.0,
            )
            refined_actual, _diagnostics = refine_one(
                actual,
                actual,
                mask,
                constraints,
                combined,
                corner["s"],
                ROBUST_CONFIG,
                rl_min_db=11.0,
                task_relative_db=-20.0,
            )
            proposed = np.zeros_like(refined_actual)
            proposed[mask] = refined_actual[mask] / factor[mask, None]
            current = (0.72 * proposed + 0.28 * current).astype(np.complex64)
            current[~mask] = 0.0
            current = normalize_task_norms(current, norm_reference)
    return current


def refine_selected(args: argparse.Namespace) -> None:
    refine_dir = args.out_dir / "refine"
    if refine_dir.exists() and any(refine_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite refinement: {refine_dir}")
    refine_dir.mkdir(parents=True, exist_ok=True)
    pool = load_npz(args.out_dir / "pool" / "candidate_pool.npz")
    initial = load_npz(args.out_dir / "initial" / "robust_arrays.npz")
    base, e2_corners = build_corners(args, levels=(0.20,))
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    selected = np.asarray(initial["candidate_index"], dtype=np.int64)
    e2_margins = np.asarray(initial["E2_robust_margins"], dtype=float)
    score = np.min(e2_margins, axis=1)
    selected_lookup = {int(candidate): position for position, candidate in enumerate(selected)}
    samples = np.asarray(pool["sample_index"], dtype=np.int64)
    refined_tasks: list[np.ndarray] = []
    parent_indices: list[int] = []
    rows: list[dict[str, Any]] = []
    started = time.time()
    for scene_position, sample in enumerate(sorted(set(samples.tolist()))):
        scene_members = np.flatnonzero(samples == sample)
        k_value = int(pool["k_values"][scene_members[0]])
        targets = np.asarray(pool["targets"][scene_members[0], :k_value], dtype=float)
        states = scene_calibration_states(pool, scene_members, e2_corners, base["element_ixiy"], args.seed)
        for ratio in RATIOS:
            available = [
                int(candidate)
                for candidate in selected
                if int(samples[candidate]) == sample and np.isclose(float(pool["ratio"][candidate]), ratio)
            ]
            if not available:
                continue
            parent = max(available, key=lambda value: score[selected_lookup[value]])
            command = ri_to_complex(pool["tasks_real_imag"][parent, :, :k_value])
            mask = np.asarray(pool["masks"][parent], dtype=bool)
            refined = robust_refine_command(
                command,
                mask,
                targets,
                e2_corners,
                states,
                grid_dirs,
                int(args.refine_sweeps),
            )
            padded = np.zeros((256, KMAX), dtype=np.complex64)
            padded[:, :k_value] = refined
            refined_tasks.append(padded)
            parent_indices.append(parent)
            rows.append(
                {
                    "refined_index": len(refined_tasks) - 1,
                    "parent_candidate_index": parent,
                    "sample_index": sample,
                    "k_value": k_value,
                    "ratio": ratio,
                    "parent_E2_worst_margin_db": float(score[selected_lookup[parent]]),
                    "common_command_across_corners": 1,
                    "corner_specific_command_saved": 0,
                }
            )
        print(
            f"robust refine scene {scene_position + 1:03d} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    np.savez_compressed(
        refine_dir / "refined_commands.npz",
        parent_candidate_index=np.asarray(parent_indices, dtype=np.int64),
        tasks_real_imag=complex_to_ri(np.stack(refined_tasks)),
    )
    write_csv(refine_dir / "refinement_manifest.csv", rows)
    summary = {
        "refined_command_count": len(refined_tasks),
        "common_command_across_E2_corners": True,
        "profile_specific_weights_saved": False,
        "sweeps": int(args.refine_sweeps),
        "elapsed_seconds": time.time() - started,
    }
    (refine_dir / "refine_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def active_rl_guided_masks(
    command: np.ndarray,
    parent_mask: np.ndarray,
    targets: np.ndarray,
    nominal_effective: DenseExternalEEP,
    grid_dirs: np.ndarray,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    count: int,
    existing_hashes: set[str],
    seed: int,
) -> list[np.ndarray]:
    mask = np.asarray(parent_mask, dtype=bool)
    steering = matched_steering_tasks(command, targets, nominal_effective, grid_dirs)
    utility = np.sum(np.abs(steering) ** 2, axis=1)
    utility /= max(float(np.max(utility)), EPS)
    stress = np.zeros(256, dtype=np.float64)
    burden = np.zeros(256, dtype=np.float64)
    rho = 10.0 ** (-11.0 / 20.0)
    for name, corner in corners.items():
        actual = apply_analog_state(command, mask, states[name])
        for source_index, source in enumerate(
            [np.sum(actual, axis=1), *[actual[:, task] for task in range(actual.shape[1])]]
        ):
            reflected = corner["s"] @ source
            amplitude = np.abs(source)
            maximum = max(float(np.max(amplitude)), EPS)
            floor = maximum * (1.0e-6 if source_index == 0 else 0.1)
            considered = mask if source_index == 0 else mask & (amplitude >= floor)
            gamma = np.abs(reflected) / np.maximum(amplitude, floor)
            stress = np.maximum(stress, np.where(considered, gamma / rho, 0.0))
        burden += np.abs(np.diag(corner["s"]))
        burden += np.sqrt(np.sum(np.abs(corner["s"][:, mask]) ** 2, axis=1))
    burden /= max(len(corners), 1)
    burden = (burden - float(np.min(burden))) / max(float(np.ptp(burden)), EPS)
    active = np.flatnonzero(mask)
    inactive = np.flatnonzero(~mask)
    remove_order = active[
        np.argsort((stress - 0.35 * utility)[active], kind="stable")[::-1]
    ]
    add_order = inactive[
        np.argsort((0.70 * utility - 0.30 * burden)[inactive], kind="stable")[::-1]
    ]
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    attempts = 0
    swap_schedule = (1, 2, 3, 4, 6, 8, 10, 12, 5, 7, 9, 11)
    while len(output) < count and attempts < count * 20:
        swaps = min(swap_schedule[attempts % len(swap_schedule)], active.size, inactive.size)
        offset = (attempts // len(swap_schedule)) % 4
        remove = remove_order[offset : offset + swaps]
        add = add_order[offset : offset + swaps]
        if remove.size != swaps or add.size != swaps:
            proposal = mutate_mask(mask, rng, max_swaps=max(2, swaps))
        else:
            proposal = mask.copy()
            proposal[remove] = False
            proposal[add] = True
        digest = mask_digest(proposal)
        attempts += 1
        if digest in existing_hashes:
            continue
        existing_hashes.add(digest)
        output.append(proposal)
    if len(output) != count:
        raise RuntimeError(f"Could only generate {len(output)}/{count} rescue masks")
    return output


def rescue_failed_scenes(args: argparse.Namespace) -> None:
    rescue_dir = args.out_dir / "rescue"
    if rescue_dir.exists() and any(rescue_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite rescue: {rescue_dir}")
    rescue_dir.mkdir(parents=True, exist_ok=True)
    pool = load_npz(args.out_dir / "pool" / "candidate_pool.npz")
    final_arrays = load_npz(args.out_dir / "final" / "robust_arrays.npz")
    final_rows = read_csv(args.out_dir / "final" / "robust_candidate_metrics.csv")
    failed = {
        int(row["sample_index"])
        for row in read_csv(args.out_dir / "final" / "robust_scene_oracle.csv")
        if row["envelope"] == "E2" and int(row["robust_oracle_pass"]) == 0
    }
    base, nominal_effective, _fast, _s = load_nominal_operator(args.operator)
    _base, e2_corners = build_corners(args, levels=(0.20,))
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    final_commands = ri_to_complex(final_arrays["tasks_real_imag"])
    store: dict[str, list[Any]] = defaultdict(list)
    manifest: list[dict[str, Any]] = []
    started = time.time()
    for scene_position, sample in enumerate(sorted(failed)):
        pool_scene = np.flatnonzero(pool_samples == sample)
        k_value = int(pool["k_values"][pool_scene[0]])
        targets = np.asarray(pool["targets"][pool_scene[0], :k_value], dtype=float)
        states = scene_calibration_states(pool, pool_scene, e2_corners, base["element_ixiy"], args.seed)
        for ratio in RATIOS:
            matching = [
                row
                for row in final_rows
                if int(row["sample_index"]) == sample
                and np.isclose(float(row["ratio"]), ratio, atol=1.0e-5)
            ]
            parent_row = max(matching, key=lambda row: float(row["E2_worst_margin_db"]))
            parent_eval = int(parent_row["evaluation_index"])
            parent_candidate = int(parent_row["candidate_index"])
            parent_mask = np.asarray(pool["masks"][parent_candidate], dtype=bool)
            command = np.asarray(final_commands[parent_eval, :, :k_value], dtype=np.complex64)
            existing_hashes = {
                str(value)
                for value in pool["mask_hash"][
                    pool_scene[np.isclose(pool["ratio"][pool_scene], ratio, atol=1.0e-5)]
                ]
            }
            proposals = active_rl_guided_masks(
                command,
                parent_mask,
                targets,
                nominal_effective,
                grid_dirs,
                e2_corners,
                states,
                int(args.rescue_masks_per_ratio),
                existing_hashes,
                int(args.seed + sample * 1009 + round(ratio * 1000)),
            )
            steering = matched_steering_tasks(command, targets, nominal_effective, grid_dirs)
            for local_index, proposal in enumerate(proposals):
                warm = 0.70 * command + 0.30 * steering
                warm[~proposal] = 0.0
                warm = normalize_task_norms(warm, command)
                refined = robust_refine_command(
                    warm,
                    proposal,
                    targets,
                    e2_corners,
                    states,
                    grid_dirs,
                    int(args.refine_sweeps),
                )
                padded_targets = np.full((KMAX, 2), np.nan, dtype=np.float32)
                padded_targets[:k_value] = targets
                padded_tasks = np.zeros((256, KMAX), dtype=np.complex64)
                padded_tasks[:, :k_value] = refined
                store["sample_index"].append(sample)
                store["k_values"].append(k_value)
                store["ratio"].append(ratio)
                store["targets"].append(padded_targets)
                store["masks"].append(proposal.astype(np.int8))
                store["tasks"].append(padded_tasks)
                store["mask_hash"].append(mask_digest(proposal))
                store["parent_candidate"].append(parent_candidate)
                store["parent_evaluation"].append(parent_eval)
                manifest.append(
                    {
                        "rescue_index": len(store["sample_index"]) - 1,
                        "sample_index": sample,
                        "k_value": k_value,
                        "ratio": ratio,
                        "mask_hash": mask_digest(proposal),
                        "parent_candidate_index": parent_candidate,
                        "parent_evaluation_index": parent_eval,
                        "mask_family": f"E2_active_rl_guided_swap_{local_index:02d}",
                        "common_command_across_corners": 1,
                    }
                )
        print(
            f"rescue scene {scene_position + 1:03d}/{len(failed):03d} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    np.savez_compressed(
        rescue_dir / "rescue_candidates.npz",
        rescue_index=np.arange(len(store["sample_index"]), dtype=np.int64),
        sample_index=np.asarray(store["sample_index"], dtype=np.int64),
        k_values=np.asarray(store["k_values"], dtype=np.int8),
        ratio=np.asarray(store["ratio"], dtype=np.float32),
        targets=np.stack(store["targets"]),
        masks=np.stack(store["masks"]),
        tasks_real_imag=complex_to_ri(np.stack(store["tasks"])),
        mask_hash=np.asarray(store["mask_hash"]),
        parent_candidate_index=np.asarray(store["parent_candidate"], dtype=np.int64),
        parent_evaluation_index=np.asarray(store["parent_evaluation"], dtype=np.int64),
    )
    write_csv(rescue_dir / "rescue_manifest.csv", manifest)
    summary = {
        "failed_scene_count": len(failed),
        "rescue_candidate_count": len(store["sample_index"]),
        "rescue_masks_per_failed_scene_ratio": int(args.rescue_masks_per_ratio),
        "profile_specific_commands_saved": False,
        "E2_or_threshold_changed": False,
        "elapsed_seconds": time.time() - started,
    }
    (rescue_dir / "rescue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def evaluation_candidates(
    args: argparse.Namespace,
    final: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    pool = load_npz(args.out_dir / "pool" / "candidate_pool.npz")
    screen = load_npz(args.out_dir / "screen" / "screening_arrays.npz")
    candidate_indices = np.asarray(screen["selected_candidate_indices"], dtype=np.int64)
    tasks = ri_to_complex(pool["tasks_real_imag"][candidate_indices])
    parent = candidate_indices.copy()
    refined_flag = np.zeros(len(candidate_indices), dtype=np.int8)
    if final:
        refined = load_npz(args.out_dir / "refine" / "refined_commands.npz")
        refined_parent = np.asarray(refined["parent_candidate_index"], dtype=np.int64)
        candidate_indices = np.concatenate((candidate_indices, refined_parent))
        parent = np.concatenate((parent, refined_parent))
        tasks = np.concatenate((tasks, ri_to_complex(refined["tasks_real_imag"])), axis=0)
        refined_flag = np.concatenate((refined_flag, np.ones(len(refined_parent), dtype=np.int8)))
    return pool, candidate_indices, tasks, refined_flag


def evaluate(args: argparse.Namespace, *, final: bool) -> None:
    name = "final" if final else "initial"
    out_dir = args.out_dir / name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {name} evaluation: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    strict_gates = protocol["strict_corner_gate"]
    pool, parent_indices, commands_full, refined_flag = evaluation_candidates(args, final)
    base, nominal_effective, nominal_fast, nominal_s = load_nominal_operator(args.operator)
    _base, corners = build_corners(args)
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    count = len(parent_indices)
    reference_metrics = np.zeros((count, len(METRIC_NAMES)), dtype=np.float32)
    robust = {
        name: np.full((count, len(ROBUST_MARGIN_NAMES)), np.inf, dtype=np.float32)
        for name in ("E1", "E2", "E3")
    }
    worst_corner = {
        name: np.full((count, len(ROBUST_MARGIN_NAMES)), "", dtype="<U40")
        for name in ("E1", "E2", "E3")
    }
    samples = np.asarray(pool["sample_index"][parent_indices], dtype=np.int64)
    k_values = np.asarray(pool["k_values"][parent_indices], dtype=np.int8)
    ratios = np.asarray(pool["ratio"][parent_indices], dtype=np.float32)
    corner_pass = np.zeros((count, len(corners)), dtype=np.int8)
    corner_names = list(corners)
    started = time.time()

    for scene_position, sample in enumerate(sorted(set(samples.tolist()))):
        members = np.flatnonzero(samples == sample)
        pool_scene_members = np.flatnonzero(np.asarray(pool["sample_index"], dtype=np.int64) == sample)
        k_value = int(k_values[members[0]])
        targets = np.asarray(pool["targets"][parent_indices[members[0]], :k_value], dtype=float)
        commands = np.asarray(commands_full[members, :, :k_value], dtype=np.complex64)
        nominal_batch = nominal_fast.evaluate(commands, targets)
        for local, output_index in enumerate(members):
            reference_metrics[output_index] = metric_vector(metric_at(nominal_batch, local))
        states = scene_calibration_states(pool, pool_scene_members, corners, base["element_ixiy"], args.seed)
        for corner_index, (corner_name, corner) in enumerate(corners.items()):
            state = states[corner_name]
            actual = np.stack(
                [
                    apply_calibration(
                        commands[local],
                        np.asarray(pool["masks"][parent_indices[output_index]], dtype=bool),
                        state,
                    )
                    for local, output_index in enumerate(members)
                ]
            )
            batch_metrics = corner["fast"].evaluate(actual, targets)
            envelope = str(corner["envelope"])
            for local, output_index in enumerate(members):
                metrics = metric_at(batch_metrics, local)
                reference = {
                    str(metric): float(reference_metrics[output_index, metric_index])
                    for metric_index, metric in enumerate(METRIC_NAMES)
                }
                mask = np.asarray(pool["masks"][parent_indices[output_index]], dtype=bool)
                active = full_active_metrics(actual[local], mask, corner["s"])
                physical = physical_margins(metrics, reference, active)
                realizability, _hardware = hardware_margin(
                    commands[local],
                    actual[local],
                    mask,
                    targets,
                    corner["effective"],
                    grid_dirs,
                    strict_gates,
                )
                margins = np.concatenate((physical, np.asarray([realizability], dtype=np.float32)))
                corner_pass[output_index, corner_index] = int(np.all(margins >= 0.0))
                improve = margins < robust[envelope][output_index]
                robust[envelope][output_index, improve] = margins[improve]
                worst_corner[envelope][output_index, improve] = corner_name
        print(
            f"{name} dense corners scene {scene_position + 1:03d}/{len(set(samples.tolist())):03d} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    candidate_rows: list[dict[str, Any]] = []
    for output_index, parent in enumerate(parent_indices):
        row: dict[str, Any] = {
            "evaluation_index": output_index,
            "candidate_index": int(parent),
            "sample_index": int(samples[output_index]),
            "k_value": int(k_values[output_index]),
            "ratio": float(ratios[output_index]),
            "refined_common_command": int(refined_flag[output_index]),
            "mask_hash": str(pool["mask_hash"][parent]),
            "candidate_origin": (
                "adversarial_multi_corner_refinement"
                if refined_flag[output_index]
                else str(pool["candidate_origin"][parent])
            ),
        }
        for envelope in ("E1", "E2", "E3"):
            row[f"{envelope}_strict_pass"] = int(np.all(robust[envelope][output_index] >= 0.0))
            row[f"{envelope}_worst_margin_db"] = float(np.min(robust[envelope][output_index]))
            for margin_index, margin_name in enumerate(ROBUST_MARGIN_NAMES):
                row[f"{envelope}_{margin_name}_margin_db"] = float(robust[envelope][output_index, margin_index])
                row[f"{envelope}_{margin_name}_worst_corner"] = str(worst_corner[envelope][output_index, margin_index])
        candidate_rows.append(row)
    write_csv(out_dir / "robust_candidate_metrics.csv", candidate_rows)
    corner_metadata = {
        corner_name: {
            "profile": str(corner["profile"]),
            "drift_intensity": float(corner["level"]),
            "envelope": str(corner["envelope"]),
            **{key: float(value) for key, value in corner["metadata"].items()},
        }
        for corner_name, corner in corners.items()
    }
    (out_dir / "corner_metadata.json").write_text(
        json.dumps(corner_metadata, indent=2), encoding="utf-8"
    )

    scene_rows: list[dict[str, Any]] = []
    for sample in sorted(set(samples.tolist())):
        scene_members = np.flatnonzero(samples == sample)
        for envelope in ("E1", "E2", "E3"):
            passed = scene_members[np.all(robust[envelope][scene_members] >= 0.0, axis=1)]
            minimum_ratio = float(np.min(ratios[passed])) if passed.size else float("nan")
            best = int(scene_members[np.argmax(np.min(robust[envelope][scene_members], axis=1))])
            scene_rows.append(
                {
                    "sample_index": int(sample),
                    "scene_origin": str(pool["scene_origin"][parent_indices[scene_members[0]]]),
                    "k_value": int(k_values[scene_members[0]]),
                    "envelope": envelope,
                    "robust_oracle_pass": int(passed.size > 0),
                    "minimum_feasible_ratio": minimum_ratio,
                    "best_evaluation_index": best,
                    "best_worst_margin_db": float(np.min(robust[envelope][best])),
                    "verified_candidate_count": int(len(scene_members)),
                }
            )
    write_csv(out_dir / "robust_scene_oracle.csv", scene_rows)

    group_rows: list[dict[str, Any]] = []
    for envelope in ("E1", "E2", "E3"):
        envelope_rows = [row for row in scene_rows if row["envelope"] == envelope]
        for origin in ("all", "existing45", "new30"):
            for k_value in (0, 2, 4, 6):
                members = [
                    row
                    for row in envelope_rows
                    if (origin == "all" or row["scene_origin"] == origin)
                    and (k_value == 0 or int(row["k_value"]) == k_value)
                ]
                if not members:
                    continue
                group_rows.append(
                    {
                        "envelope": envelope,
                        "scene_origin": origin,
                        "k_value": "all" if k_value == 0 else k_value,
                        "scene_count": len(members),
                        "oracle_pass_count": sum(int(row["robust_oracle_pass"]) for row in members),
                        "robust_oracle_rate": float(np.mean([row["robust_oracle_pass"] for row in members])),
                    }
                )
    write_csv(out_dir / "robust_oracle_groups.csv", group_rows)

    def rate(envelope: str, origin: str = "all", k: int = 0) -> float:
        matches = [
            item
            for item in group_rows
            if item["envelope"] == envelope
            and item["scene_origin"] == origin
            and item["k_value"] == ("all" if k == 0 else k)
        ]
        return float(matches[0]["robust_oracle_rate"]) if matches else float("nan")

    e2_k6_low = any(
        int(row["k_value"]) == 6
        and row["envelope"] == "E2"
        and int(row["robust_oracle_pass"]) == 1
        and float(row["minimum_feasible_ratio"]) <= 0.70 + 1.0e-6
        for row in scene_rows
    )
    acceptance = {
        "E1_new_scene_robust_oracle": rate("E1", "new30"),
        "E2_overall_robust_oracle": rate("E2"),
        "E2_k2_robust_oracle": rate("E2", "all", 2),
        "E2_k4_robust_oracle": rate("E2", "all", 4),
        "E2_k6_robust_oracle": rate("E2", "all", 6),
        "E2_has_k6_positive_ratio_le_0_7": e2_k6_low,
    }
    gates = protocol["stage_b_acceptance"]
    acceptance["stage_b_gate_pass"] = bool(
        acceptance["E1_new_scene_robust_oracle"] >= float(gates["E1_new_scene_robust_oracle_min"])
        and acceptance["E2_overall_robust_oracle"] >= float(gates["E2_overall_robust_oracle_min"])
        and acceptance["E2_k2_robust_oracle"] >= float(gates["E2_k2_robust_oracle_min"])
        and acceptance["E2_k4_robust_oracle"] >= float(gates["E2_k4_robust_oracle_min"])
        and acceptance["E2_k6_robust_oracle"] >= float(gates["E2_k6_robust_oracle_min"])
        and e2_k6_low
    )
    root_causes = Counter()
    for row in candidate_rows:
        if int(row["E2_strict_pass"]) == 0:
            margins = [float(row[f"E2_{name}_margin_db"]) for name in ROBUST_MARGIN_NAMES]
            root_causes[str(ROBUST_MARGIN_NAMES[int(np.argmin(margins))])] += 1
    summary = {
        "evaluation": name,
        "verified_candidate_count": count,
        "scene_count": len(set(samples.tolist())),
        **acceptance,
        "E3_stress_oracle": rate("E3"),
        "E2_failed_candidate_root_causes": dict(root_causes),
        "calibration_seed_formula": (
            "protocol_seed + 97*sample_index + 100003*profile_index + 1009*level_index"
        ),
        "critic_retraining_allowed": bool(acceptance["stage_b_gate_pass"]),
        "hfss_smoke_allowed": bool(acceptance["stage_b_gate_pass"]),
        "elapsed_seconds": time.time() - started,
        "evidence_scope": protocol["evidence_scope"],
    }
    np.savez_compressed(
        out_dir / "robust_arrays.npz",
        candidate_index=parent_indices,
        refined_common_command=refined_flag,
        reference_metrics=reference_metrics,
        E1_robust_margins=robust["E1"],
        E2_robust_margins=robust["E2"],
        E3_robust_margins=robust["E3"],
        robust_margin_names=ROBUST_MARGIN_NAMES,
        corner_names=np.asarray(corner_names),
        corner_pass=corner_pass,
        tasks_real_imag=complex_to_ri(commands_full),
    )
    (out_dir / "stage_b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def evaluate_rescue(args: argparse.Namespace) -> None:
    out_dir = args.out_dir / "post_rescue"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite rescue evaluation: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    strict_gates = protocol["strict_corner_gate"]
    pool = load_npz(args.out_dir / "pool" / "candidate_pool.npz")
    rescue = load_npz(args.out_dir / "rescue" / "rescue_candidates.npz")
    base, _nominal_effective, nominal_fast, _nominal_s = load_nominal_operator(args.operator)
    _base, corners = build_corners(args)
    grid_dirs = pattern_grid_dirs(base["theta_deg"], base["phi_deg"])
    count = len(rescue["rescue_index"])
    commands_full = ri_to_complex(rescue["tasks_real_imag"])
    reference_metrics = np.zeros((count, len(METRIC_NAMES)), dtype=np.float32)
    robust = {
        name: np.full((count, len(ROBUST_MARGIN_NAMES)), np.inf, dtype=np.float32)
        for name in ("E1", "E2", "E3")
    }
    worst_corner = {
        name: np.full((count, len(ROBUST_MARGIN_NAMES)), "", dtype="<U40")
        for name in ("E1", "E2", "E3")
    }
    rescue_samples = np.asarray(rescue["sample_index"], dtype=np.int64)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    started = time.time()
    for scene_position, sample in enumerate(sorted(set(rescue_samples.tolist()))):
        members = np.flatnonzero(rescue_samples == sample)
        pool_scene = np.flatnonzero(pool_samples == sample)
        k_value = int(rescue["k_values"][members[0]])
        targets = np.asarray(rescue["targets"][members[0], :k_value], dtype=float)
        commands = np.asarray(commands_full[members, :, :k_value], dtype=np.complex64)
        nominal_batch = nominal_fast.evaluate(commands, targets)
        for local, candidate in enumerate(members):
            reference_metrics[candidate] = metric_vector(metric_at(nominal_batch, local))
        states = scene_calibration_states(pool, pool_scene, corners, base["element_ixiy"], args.seed)
        for corner_name, corner in corners.items():
            state = states[corner_name]
            actual = np.stack(
                [
                    apply_calibration(
                        commands[local],
                        np.asarray(rescue["masks"][candidate], dtype=bool),
                        state,
                    )
                    for local, candidate in enumerate(members)
                ]
            )
            metrics_batch = corner["fast"].evaluate(actual, targets)
            envelope = str(corner["envelope"])
            for local, candidate in enumerate(members):
                metrics = metric_at(metrics_batch, local)
                reference = {
                    str(metric): float(reference_metrics[candidate, metric_index])
                    for metric_index, metric in enumerate(METRIC_NAMES)
                }
                mask = np.asarray(rescue["masks"][candidate], dtype=bool)
                active = full_active_metrics(actual[local], mask, corner["s"])
                physical = physical_margins(metrics, reference, active)
                realizability, _hardware = hardware_margin(
                    commands[local],
                    actual[local],
                    mask,
                    targets,
                    corner["effective"],
                    grid_dirs,
                    strict_gates,
                )
                margins = np.concatenate((physical, np.asarray([realizability], dtype=np.float32)))
                improve = margins < robust[envelope][candidate]
                robust[envelope][candidate, improve] = margins[improve]
                worst_corner[envelope][candidate, improve] = corner_name
        print(
            f"rescue dense corners scene {scene_position + 1:03d} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    rescue_rows: list[dict[str, Any]] = []
    scene_origin = {
        int(sample): str(pool["scene_origin"][np.flatnonzero(pool_samples == sample)[0]])
        for sample in set(pool_samples.tolist())
    }
    for candidate in range(count):
        row: dict[str, Any] = {
            "evaluation_index": f"rescue_{candidate}",
            "candidate_index": -1,
            "rescue_index": candidate,
            "sample_index": int(rescue_samples[candidate]),
            "scene_origin": scene_origin[int(rescue_samples[candidate])],
            "k_value": int(rescue["k_values"][candidate]),
            "ratio": float(rescue["ratio"][candidate]),
            "refined_common_command": 1,
            "mask_hash": str(rescue["mask_hash"][candidate]),
            "candidate_origin": "E2_active_rl_guided_mask_rescue",
        }
        for envelope in ("E1", "E2", "E3"):
            row[f"{envelope}_strict_pass"] = int(np.all(robust[envelope][candidate] >= 0.0))
            row[f"{envelope}_worst_margin_db"] = float(np.min(robust[envelope][candidate]))
            for margin_index, margin_name in enumerate(ROBUST_MARGIN_NAMES):
                row[f"{envelope}_{margin_name}_margin_db"] = float(robust[envelope][candidate, margin_index])
                row[f"{envelope}_{margin_name}_worst_corner"] = str(worst_corner[envelope][candidate, margin_index])
        rescue_rows.append(row)
    write_csv(out_dir / "rescue_robust_candidate_metrics.csv", rescue_rows)

    final_rows = read_csv(args.out_dir / "final" / "robust_candidate_metrics.csv")
    for row in final_rows:
        row["scene_origin"] = scene_origin[int(row["sample_index"])]
    all_rows: list[dict[str, Any]] = [*final_rows, *rescue_rows]
    scene_rows: list[dict[str, Any]] = []
    for sample in sorted(set(pool_samples.tolist())):
        members = [row for row in all_rows if int(row["sample_index"]) == sample]
        k_value = int(members[0]["k_value"])
        for envelope in ("E1", "E2", "E3"):
            passed = [row for row in members if int(row[f"{envelope}_strict_pass"]) == 1]
            best = max(members, key=lambda row: float(row[f"{envelope}_worst_margin_db"]))
            scene_rows.append(
                {
                    "sample_index": sample,
                    "scene_origin": scene_origin[sample],
                    "k_value": k_value,
                    "envelope": envelope,
                    "robust_oracle_pass": int(bool(passed)),
                    "minimum_feasible_ratio": (
                        min(float(row["ratio"]) for row in passed) if passed else float("nan")
                    ),
                    "best_candidate_source": str(best["candidate_origin"]),
                    "best_evaluation_index": str(best["evaluation_index"]),
                    "best_worst_margin_db": float(best[f"{envelope}_worst_margin_db"]),
                    "verified_candidate_count": len(members),
                }
            )
    write_csv(out_dir / "post_rescue_scene_oracle.csv", scene_rows)

    group_rows: list[dict[str, Any]] = []
    for envelope in ("E1", "E2", "E3"):
        for origin in ("all", "existing45", "new30"):
            for k_value in (0, 2, 4, 6):
                members = [
                    row
                    for row in scene_rows
                    if row["envelope"] == envelope
                    and (origin == "all" or row["scene_origin"] == origin)
                    and (k_value == 0 or int(row["k_value"]) == k_value)
                ]
                if members:
                    group_rows.append(
                        {
                            "envelope": envelope,
                            "scene_origin": origin,
                            "k_value": "all" if k_value == 0 else k_value,
                            "scene_count": len(members),
                            "oracle_pass_count": sum(int(row["robust_oracle_pass"]) for row in members),
                            "robust_oracle_rate": float(np.mean([row["robust_oracle_pass"] for row in members])),
                        }
                    )
    write_csv(out_dir / "post_rescue_oracle_groups.csv", group_rows)

    def rate(envelope: str, origin: str = "all", k_value: int = 0) -> float:
        desired = "all" if k_value == 0 else k_value
        return float(
            next(
                row["robust_oracle_rate"]
                for row in group_rows
                if row["envelope"] == envelope
                and row["scene_origin"] == origin
                and row["k_value"] == desired
            )
        )

    e2_k6_low = any(
        row["envelope"] == "E2"
        and int(row["k_value"]) == 6
        and int(row["robust_oracle_pass"]) == 1
        and float(row["minimum_feasible_ratio"]) <= 0.70 + 1.0e-6
        for row in scene_rows
    )
    acceptance = {
        "E1_new_scene_robust_oracle": rate("E1", "new30"),
        "E2_overall_robust_oracle": rate("E2"),
        "E2_k2_robust_oracle": rate("E2", "all", 2),
        "E2_k4_robust_oracle": rate("E2", "all", 4),
        "E2_k6_robust_oracle": rate("E2", "all", 6),
        "E2_has_k6_positive_ratio_le_0_7": e2_k6_low,
    }
    gates = protocol["stage_b_acceptance"]
    acceptance["stage_b_gate_pass"] = bool(
        acceptance["E1_new_scene_robust_oracle"] >= float(gates["E1_new_scene_robust_oracle_min"])
        and acceptance["E2_overall_robust_oracle"] >= float(gates["E2_overall_robust_oracle_min"])
        and acceptance["E2_k2_robust_oracle"] >= float(gates["E2_k2_robust_oracle_min"])
        and acceptance["E2_k4_robust_oracle"] >= float(gates["E2_k4_robust_oracle_min"])
        and acceptance["E2_k6_robust_oracle"] >= float(gates["E2_k6_robust_oracle_min"])
        and e2_k6_low
    )
    failed_best_causes = Counter()
    for scene in scene_rows:
        if scene["envelope"] != "E2" or int(scene["robust_oracle_pass"]) == 1:
            continue
        best = next(
            row
            for row in all_rows
            if int(row["sample_index"]) == int(scene["sample_index"])
            and str(row["evaluation_index"]) == str(scene["best_evaluation_index"])
        )
        values = [float(best[f"E2_{name}_margin_db"]) for name in ROBUST_MARGIN_NAMES]
        failed_best_causes[str(ROBUST_MARGIN_NAMES[int(np.argmin(values))])] += 1
    summary = {
        "evaluation": "post_rescue",
        "rescue_candidate_count": count,
        "scene_count": len(set(pool_samples.tolist())),
        **acceptance,
        "E3_stress_oracle": rate("E3"),
        "failed_scene_best_candidate_root_causes": dict(failed_best_causes),
        "critic_retraining_allowed": bool(acceptance["stage_b_gate_pass"]),
        "hfss_smoke_allowed": bool(acceptance["stage_b_gate_pass"]),
        "elapsed_seconds": time.time() - started,
        "evidence_scope": protocol["evidence_scope"],
    }
    np.savez_compressed(
        out_dir / "rescue_robust_arrays.npz",
        rescue_index=np.asarray(rescue["rescue_index"], dtype=np.int64),
        reference_metrics=reference_metrics,
        E1_robust_margins=robust["E1"],
        E2_robust_margins=robust["E2"],
        E3_robust_margins=robust["E3"],
        robust_margin_names=ROBUST_MARGIN_NAMES,
    )
    (out_dir / "stage_b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if not args.protocol.exists():
        raise FileNotFoundError(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["drift_envelopes"]["E2_engineering"]["intensities"] != [0.20]:
        raise RuntimeError("The preregistered E2 envelope was changed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = (
        ("prepare", prepare_pool),
        ("screen", screen_pool),
        ("evaluate-initial", lambda value: evaluate(value, final=False)),
        ("refine", refine_selected),
        ("evaluate-final", lambda value: evaluate(value, final=True)),
        ("rescue", rescue_failed_scenes),
        ("evaluate-rescue", evaluate_rescue),
    )
    if args.mode == "all":
        for _name, function in modes:
            function(args)
    else:
        function = dict(modes)[args.mode]
        function(args)


if __name__ == "__main__":
    main()
