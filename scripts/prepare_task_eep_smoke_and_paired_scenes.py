"""Prepare grounded-patch EEP/HFSS smoke inputs and paired-ratio scenes.

No HFSS result is fabricated here.  The smoke teacher is selected only from
the joint-pass task-level run, while the supplemental dataset groups every
target scene across the same six requested ratios.  New close-target scenes
have a controlled minimum angular separation in the 5-10 degree interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from generate_iso_lcmv_teacher import lcmv_zf_weights, steering_rx, unit_vectors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_TASK_RUN = ROOT / "hfss_outputs" / "grounded_patch_task_lcmv_psll_20260717_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "grounded_patch_eep_hfss_smoke_20260717_run01"
RATIOS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
KMAX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task-run", type=Path, default=DEFAULT_TASK_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--normal-scenes-per-k", type=int, default=12)
    parser.add_argument("--close-k2-scenes", type=int, default=16)
    parser.add_argument("--close-k4-scenes", type=int, default=16)
    parser.add_argument("--close-k6-scenes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def angular_separation_deg(targets_deg: np.ndarray) -> float:
    if targets_deg.shape[0] <= 1:
        return 180.0
    directions = unit_vectors(targets_deg[:, 0], targets_deg[:, 1]).astype(np.float64)
    dots = np.clip(directions @ directions.T, -1.0, 1.0)
    upper = np.triu_indices(targets_deg.shape[0], 1)
    return float(np.min(np.rad2deg(np.arccos(dots[upper]))))


def smoke_score(row: dict[str, str]) -> tuple[float, float, float]:
    rf_margin = min(float(row["worst_active_rl_db"]), float(row["total_rl_db"])) - 10.0
    return (
        rf_margin,
        -float(row["mainlobe_loss_db"]),
        -float(row["final_psll_db"]),
    )


def select_smoke(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    requested = {
        1: (0.5, 1.0),
        2: (0.5, 0.7, 1.0),
        4: (0.5, 0.7, 1.0),
        6: (0.5, 0.6, 0.7, 0.8, 1.0),
    }
    selected: list[dict[str, str]] = []
    used: set[int] = set()
    for k_value, ratios in requested.items():
        for ratio in ratios:
            for large_scan in (0, 1):
                candidates = [
                    row
                    for row in rows
                    if int(row["joint_gate_pass"]) == 1
                    and int(row["k"]) == k_value
                    and abs(float(row["ratio_requested"]) - ratio) <= 1.0e-6
                    and int(float(row["max_target_theta_deg"]) >= 45.0) == large_scan
                    and int(row["sample_index"]) not in used
                ]
                if not candidates:
                    continue
                best = max(candidates, key=smoke_score)
                used.add(int(best["sample_index"]))
                selected.append(best)
    return selected


def make_smoke_teacher(
    base: dict[str, np.ndarray],
    task_payload: dict[str, np.ndarray],
    selected: list[dict[str, str]],
    out_dir: Path,
) -> None:
    teacher_dir = out_dir / "smoke_teacher"
    teacher_dir.mkdir()
    compatible = {key: np.asarray(value).copy() for key, value in base.items()}
    task_weights = np.asarray(task_payload["task_weights_real_imag"], dtype=np.float32)
    masks = np.asarray(task_payload["masks"], dtype=np.int8)
    combined = np.asarray(task_payload["combined_weights_real_imag"], dtype=np.float32)
    compatible["task_weights_real_imag"] = task_weights
    compatible["masks"] = masks
    compatible["hfss_weights_real_imag"] = combined
    compatible["hfss_magnitude_v"] = np.linalg.norm(combined, axis=-1).astype(np.float32)
    compatible["hfss_phase_deg"] = np.rad2deg(
        np.arctan2(combined[..., 1], combined[..., 0])
    ).astype(np.float32)
    compatible["num_active"] = np.sum(masks, axis=1).astype(np.int64)
    compatible["active_ratios_actual"] = np.mean(masks, axis=1).astype(np.float32)
    compatible["selected_indices"] = np.asarray(
        [int(row["sample_index"]) for row in selected], dtype=np.int64
    )
    np.savez_compressed(teacher_dir / "dataset_arrays.npz", **compatible)


def round_robin_aperture_order(
    positions: np.ndarray,
    target_dirs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    xy = positions[:, :2].astype(np.float64)
    radius = np.linalg.norm(xy, axis=1)
    radius /= max(float(np.max(radius)), 1.0e-9)
    steering = steering_rx(positions, target_dirs)
    coherence = np.mean(np.abs(steering - np.mean(steering, axis=0, keepdims=True)), axis=0)
    score = 0.70 * radius + 0.20 * coherence + 0.10 * rng.random(positions.shape[0])
    quadrant = (xy[:, 0] >= 0.0).astype(int) * 2 + (xy[:, 1] >= 0.0).astype(int)
    buckets = [list(np.flatnonzero(quadrant == value)[np.argsort(score[quadrant == value])[::-1]]) for value in range(4)]
    output: list[int] = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                output.append(bucket.pop(0))
    return np.asarray(output, dtype=np.int64)


def random_target(rng: np.random.Generator, large_scan: bool) -> tuple[float, float]:
    theta = float(rng.uniform(47.0, 65.0) if large_scan else rng.uniform(8.0, 43.0))
    phi = float(rng.uniform(0.0, 360.0))
    return theta, phi


def pairwise_separations(targets: list[tuple[float, float]], candidate: tuple[float, float]) -> np.ndarray:
    values = np.asarray(targets + [candidate], dtype=np.float32)
    directions = unit_vectors(values[:, 0], values[:, 1]).astype(np.float64)
    return np.rad2deg(np.arccos(np.clip(directions[:-1] @ directions[-1], -1.0, 1.0)))


def generate_close_targets(k_value: int, rng: np.random.Generator, large_scan: bool) -> np.ndarray:
    theta_1, phi_1 = random_target(rng, large_scan)
    separation = float(rng.uniform(5.25, 9.75))
    sign = 1.0 if theta_1 + separation <= 70.0 else -1.0
    targets: list[tuple[float, float]] = [(theta_1, phi_1), (theta_1 + sign * separation, phi_1)]
    attempts = 0
    while len(targets) < k_value and attempts < 10000:
        attempts += 1
        candidate = (float(rng.uniform(5.0, 70.0)), float(rng.uniform(0.0, 360.0)))
        if np.min(pairwise_separations(targets, candidate)) >= 12.0:
            targets.append(candidate)
    if len(targets) != k_value:
        raise RuntimeError(f"Could not generate K={k_value} close-target scene")
    values = np.asarray(targets, dtype=np.float32)
    actual = angular_separation_deg(values)
    if not 5.0 <= actual <= 10.0:
        raise AssertionError(f"Close-target separation outside requirement: {actual}")
    return values


def select_normal_scenes(
    rows: list[dict[str, str]],
    base: dict[str, np.ndarray],
    per_k: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for k_value in (2, 4, 6):
        candidates = [
            row
            for row in rows
            if int(row["joint_gate_pass"]) == 1
            and int(row["k"]) == k_value
            and float(row["min_target_separation_deg"]) >= 10.0
        ]
        candidates.sort(key=smoke_score, reverse=True)
        small = [row for row in candidates if float(row["max_target_theta_deg"]) < 45.0]
        large = [row for row in candidates if float(row["max_target_theta_deg"]) >= 45.0]
        chosen: list[dict[str, str]] = []
        for index in range(per_k):
            pool = large if index % 2 else small
            if not pool:
                pool = candidates
            item = pool.pop(0)
            if item not in chosen:
                chosen.append(item)
        for local_index, row in enumerate(chosen):
            sample_index = int(row["sample_index"])
            output.append(
                {
                    "scene_id": f"normal_k{k_value}_{local_index:03d}",
                    "scene_type": "paired_existing_joint_positive",
                    "k": k_value,
                    "targets_deg": np.asarray(base["targets_deg"][sample_index, :k_value], dtype=np.float32),
                    "source_sample_index": sample_index,
                }
            )
    return output


def build_paired_dataset(
    base: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    scenes = select_normal_scenes(rows, base, int(args.normal_scenes_per_k))
    close_counts = {2: int(args.close_k2_scenes), 4: int(args.close_k4_scenes), 6: int(args.close_k6_scenes)}
    for k_value, count in close_counts.items():
        for scene_index in range(count):
            scenes.append(
                {
                    "scene_id": f"close_k{k_value}_{scene_index:03d}",
                    "scene_type": "new_close_5to10deg",
                    "k": k_value,
                    "targets_deg": generate_close_targets(k_value, rng, large_scan=bool(scene_index % 2)),
                    "source_sample_index": -1,
                }
            )

    positions = np.asarray(base["positions_lambda"], dtype=np.float32)
    case_count = len(scenes) * len(RATIOS)
    sample_ids = np.empty(case_count, dtype="U80")
    k_values = np.zeros(case_count, dtype=np.int64)
    ratio_requested = np.zeros(case_count, dtype=np.float32)
    ratio_actual = np.zeros(case_count, dtype=np.float32)
    num_active = np.zeros(case_count, dtype=np.int64)
    targets_all = np.full((case_count, KMAX, 2), np.nan, dtype=np.float32)
    task_valid = np.zeros((case_count, KMAX), dtype=np.int8)
    masks = np.zeros((case_count, positions.shape[0]), dtype=np.int8)
    task_weights = np.zeros((case_count, positions.shape[0], KMAX), dtype=np.complex64)
    scene_ids = np.empty(case_count, dtype="U80")
    scene_types = np.empty(case_count, dtype="U80")
    source_indices = np.full(case_count, -1, dtype=np.int64)
    separation_values = np.zeros(case_count, dtype=np.float32)
    large_scan_values = np.zeros(case_count, dtype=np.int8)
    manifest_rows: list[dict[str, Any]] = []

    case_index = 0
    for scene_index, scene in enumerate(scenes):
        k_value = int(scene["k"])
        targets = np.asarray(scene["targets_deg"], dtype=np.float32)
        target_dirs = unit_vectors(targets[:, 0], targets[:, 1])
        order = round_robin_aperture_order(positions, target_dirs, rng)
        separation = angular_separation_deg(targets)
        max_theta = float(np.max(targets[:, 0]))
        for ratio in RATIOS:
            active_count = int(round(float(ratio) * positions.shape[0]))
            mask = np.zeros(positions.shape[0], dtype=bool)
            mask[order[:active_count]] = True
            valid_indices = np.arange(k_value, dtype=np.int64)
            weights, condition, solved, constraint_count = lcmv_zf_weights(
                mask=mask,
                positions=positions,
                target_dirs=target_dirs,
                valid_indices=valid_indices,
                diagonal_loading=1.0e-4,
            )
            if not solved or not np.all(np.isfinite(weights)):
                active = np.flatnonzero(mask)
                rows_target = steering_rx(positions[active], target_dirs)
                weights = np.zeros((positions.shape[0], KMAX), dtype=np.complex64)
                for task_index in range(k_value):
                    row = rows_target[task_index]
                    weights[active, task_index] = row.conj() / max(float(np.vdot(row, row).real), 1.0e-12)
            sample_ids[case_index] = f"{scene['scene_id']}_r{ratio:.1f}"
            scene_ids[case_index] = str(scene["scene_id"])
            scene_types[case_index] = str(scene["scene_type"])
            source_indices[case_index] = int(scene["source_sample_index"])
            k_values[case_index] = k_value
            ratio_requested[case_index] = ratio
            ratio_actual[case_index] = float(np.mean(mask))
            num_active[case_index] = active_count
            targets_all[case_index, :k_value] = targets
            task_valid[case_index, :k_value] = 1
            masks[case_index] = mask.astype(np.int8)
            task_weights[case_index] = weights
            separation_values[case_index] = separation
            large_scan_values[case_index] = int(max_theta >= 45.0)
            manifest_rows.append(
                {
                    "sample_index": case_index,
                    "sample_id": sample_ids[case_index],
                    "base_scene_id": scene_ids[case_index],
                    "scene_type": scene_types[case_index],
                    "source_sample_index": source_indices[case_index],
                    "k": k_value,
                    "ratio_requested": ratio,
                    "ratio_actual": ratio_actual[case_index],
                    "active_count": active_count,
                    "min_target_separation_deg": separation,
                    "max_target_theta_deg": max_theta,
                    "large_scan": large_scan_values[case_index],
                    "lcmv_condition": condition,
                    "constraint_count": constraint_count,
                }
            )
            case_index += 1

    combined = np.sum(task_weights, axis=2)
    paired_dir = out_dir / "paired_scene_dataset"
    paired_dir.mkdir()
    np.savez_compressed(
        paired_dir / "dataset_arrays.npz",
        sample_ids=sample_ids,
        base_scene_ids=scene_ids,
        scene_types=scene_types,
        source_sample_indices=source_indices,
        k_values=k_values,
        active_ratios_requested=ratio_requested,
        active_ratios_actual=ratio_actual,
        num_active=num_active,
        targets_deg=targets_all,
        task_valid=task_valid,
        masks=masks,
        task_weights_real_imag=np.stack((task_weights.real, task_weights.imag), axis=-1).astype(np.float32),
        hfss_weights_real_imag=np.stack((combined.real, combined.imag), axis=-1).astype(np.float32),
        hfss_magnitude_v=np.abs(combined).astype(np.float32),
        hfss_phase_deg=np.rad2deg(np.angle(combined)).astype(np.float32),
        positions_lambda=positions,
        port_names=np.asarray(base["port_names"]),
        element_ixiy=np.asarray(base["element_ixiy"]),
        min_target_separation_deg=separation_values,
        large_scan=large_scan_values,
    )
    write_csv(paired_dir / "manifest.csv", manifest_rows)

    scene_order = np.arange(len(scenes), dtype=np.int64)
    split_rng = np.random.default_rng(int(args.seed) + 1)
    split_rng.shuffle(scene_order)
    train_end = int(round(0.70 * len(scenes)))
    val_end = train_end + int(round(0.15 * len(scenes)))
    split_scene_indices = {
        "train": scene_order[:train_end],
        "val": scene_order[train_end:val_end],
        "test": scene_order[val_end:],
    }
    split_payload: dict[str, Any] = {"seed": int(args.seed) + 1, "group_key": "base_scene_id", "splits": {}}
    for split_name, selected_scene_indices in split_scene_indices.items():
        selected_ids = {str(scenes[index]["scene_id"]) for index in selected_scene_indices}
        split_payload["splits"][split_name] = [
            int(index) for index, scene_id in enumerate(scene_ids) if str(scene_id) in selected_ids
        ]
    (paired_dir / "training_split_manifest.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    return {
        "base_scene_count": len(scenes),
        "normal_base_scene_count": int(args.normal_scenes_per_k) * 3,
        "close_base_scene_count": sum(close_counts.values()),
        "variant_count": case_count,
        "ratios": list(RATIOS),
        "close_separation_min_db": float(np.min(separation_values[np.asarray(scene_types) == "new_close_5to10deg"])),
        "close_separation_max_db": float(np.max(separation_values[np.asarray(scene_types) == "new_close_5to10deg"])),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    base_npz = np.load(args.dataset, allow_pickle=False)
    base = {key: base_npz[key] for key in base_npz.files}
    task_npz = np.load(args.task_run / "task_level_weights.npz", allow_pickle=False)
    task_payload = {key: task_npz[key] for key in task_npz.files}
    rows = read_csv(args.task_run / "task_lcmv_psll_case_metrics.csv")
    smoke = select_smoke(rows)
    smoke_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(smoke):
        smoke_rows.append(
            {
                "smoke_rank": rank,
                "sample_index": int(row["sample_index"]),
                "sample_id": row["sample_id"],
                "k": int(row["k"]),
                "ratio_requested": float(row["ratio_requested"]),
                "ratio_effective": float(row["ratio_effective"]),
                "large_scan": int(float(row["max_target_theta_deg"]) >= 45.0),
                "max_target_theta_deg": float(row["max_target_theta_deg"]),
                "min_target_separation_deg": float(row["min_target_separation_deg"]),
                "af_psll_db": float(row["final_psll_db"]),
                "af_nearest_iso_db": float(row["final_nearest_iso_db"]),
                "af_local_iso_db": float(row["final_local_iso_db"]),
                "proxy_worst_active_rl_db": float(row["worst_active_rl_db"]),
                "proxy_total_rl_db": float(row["total_rl_db"]),
                "mainlobe_loss_db": float(row["mainlobe_loss_db"]),
                "status": "selected_pending_256port_eep",
            }
        )
    write_csv(args.out_dir / "joint_pass_eep_hfss_smoke_manifest.csv", smoke_rows)
    make_smoke_teacher(base, task_payload, smoke, args.out_dir)
    paired_summary = build_paired_dataset(base, rows, args, args.out_dir)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke_case_count": len(smoke_rows),
        "smoke_source": "joint-pass AF plus local-kernel S256 proxy candidates",
        "smoke_status": "pending grounded-patch 256-port EEP operator; no HFSS label generated",
        "paired_dataset": paired_summary,
        "geometry_guard": "Do not use the old matched_v2 dipole EEP/S256 with grounded-patch weights.",
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
