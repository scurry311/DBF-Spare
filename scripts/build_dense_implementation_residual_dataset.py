#!/usr/bin/env python3
"""Build a scene-grouped residual-critic dataset from paired trusted HFSS runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from validate_trusted_eep_hfss_residuals import active_return_metrics, series_network_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITIVE_DATASET = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_dataset_20260724_run01"
DEFAULT_POSITIVE_HFSS = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_smoke_20260724_run01"
DEFAULT_BOUNDARY_DATASET = ROOT / "hfss_outputs" / "trusted_dense_boundary_dataset_20260724_run01"
DEFAULT_BOUNDARY_HFSS = ROOT / "hfss_outputs" / "trusted_dense_boundary_hfss_20260724_run01"
DEFAULT_EXPANDED_DATASET = ROOT / "hfss_outputs" / "expanded_independent_scenes_20260724_run02"
DEFAULT_EXPANDED_HFSS = ROOT / "hfss_outputs" / "expanded_independent_scenes_hfss_20260724_run01"
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_dense_implementation_residual_dataset_20260724_run01"
KMAX = 6
NUM_ELEMENTS = 256
AF_METRIC_NAMES = (
    "psll_db",
    "iso_nearest_db",
    "iso_local_db",
    "peak_min_db",
    "peak_spread_db",
    "energy_proxy",
)
HFSS_METRIC_NAMES = (
    "psll_db",
    "iso_nearest_db",
    "iso_local_db",
    "peak_min_db",
    "peak_spread_db",
    "pointing_error_deg",
)
RESIDUAL_NAMES = AF_METRIC_NAMES[:5]
GATE_NAMES = ("gate15", "gate20", "mainlobe_gate", "strict_engineering_gate")
SCALAR_NAMES = (
    "k_norm",
    "active_ratio",
    "num_active_norm",
    "eep_psll_db",
    "eep_iso_nearest_db",
    "eep_iso_local_db",
    "eep_peak_min_db",
    "eep_peak_spread_db",
    "energy_proxy",
    "energy_normalized_hfss",
    "weight_l2",
    "max_channel_amplitude",
    "amplitude_dynamic_range_db",
    "condition_log10",
    "null_constraint_count_norm",
    "min_target_separation_deg_norm",
    "max_scan_theta_deg_norm",
    "mean_scan_theta_deg_norm",
    "reference_hfss_peak_min_db",
    "reference_hfss_peak_spread_db",
    "nominal_worst_active_rl_db",
    "nominal_total_rl_db",
    "phase_error_rms_deg_norm",
    "gain_error_rms_db_norm",
    "dropout_fraction",
    "phase_bits_norm",
    "amplitude_bits_norm",
    "ratio_delta",
    "phase_ramp_deg_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-dataset-dir", type=Path, default=DEFAULT_POSITIVE_DATASET)
    parser.add_argument("--positive-hfss-dir", type=Path, default=DEFAULT_POSITIVE_HFSS)
    parser.add_argument("--boundary-dataset-dir", type=Path, default=DEFAULT_BOUNDARY_DATASET)
    parser.add_argument("--boundary-hfss-dir", type=Path, default=DEFAULT_BOUNDARY_HFSS)
    parser.add_argument("--expanded-dataset-dir", type=Path, default=DEFAULT_EXPANDED_DATASET)
    parser.add_argument("--expanded-hfss-dir", type=Path, default=DEFAULT_EXPANDED_HFSS)
    parser.add_argument(
        "--exclude-expanded",
        action="store_true",
        help="Build the legacy two-source dataset without the expanded package.",
    )
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260725)
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


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float(default)


def i(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(f(row, key, default)))


def amplitude_stats(tasks: np.ndarray) -> tuple[float, float, float, float]:
    combined = np.sum(tasks, axis=1)
    amplitude = np.abs(combined)
    nonzero = amplitude[amplitude >= max(float(np.max(amplitude)), 1.0e-12) * 1.0e-4]
    dynamic = (
        float(20.0 * np.log10(max(float(np.max(nonzero)), 1.0e-12) / max(float(np.min(nonzero)), 1.0e-12)))
        if nonzero.size
        else 0.0
    )
    energy = float(np.sum(amplitude**2))
    return energy, float(np.linalg.norm(combined)), float(np.max(amplitude)), dynamic


def min_separation(targets: np.ndarray) -> float:
    if targets.shape[0] < 2:
        return 180.0
    theta = np.deg2rad(targets[:, 0])
    phi = np.deg2rad(targets[:, 1])
    dirs = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=1,
    )
    dot = np.clip(dirs @ dirs.T, -1.0, 1.0)
    distance = np.rad2deg(np.arccos(dot))
    distance[np.eye(targets.shape[0], dtype=bool)] = np.inf
    return float(np.min(distance))


def nominal_active_rl(
    tasks_internal: np.ndarray,
    mask: np.ndarray,
    s_matrix: np.ndarray,
) -> tuple[float, float]:
    values = []
    combined = np.sum(tasks_internal, axis=1)
    for weight in [combined, *(tasks_internal[:, task] for task in range(tasks_internal.shape[1]))]:
        external = np.conjugate(weight)
        external[~mask] = 0.0
        norm = max(float(np.linalg.norm(external)), 1.0e-12)
        values.append(active_return_metrics(s_matrix, external / norm, mask))
    return (
        min(float(item["worst_active_rl_db"]) for item in values),
        min(float(item["total_rl_db"]) for item in values),
    )


def scene_split(sample_index: np.ndarray, k_values: np.ndarray, seed: int) -> np.ndarray:
    scene_k: dict[int, int] = {}
    for scene, k_value in zip(sample_index.tolist(), k_values.tolist()):
        scene_k[int(scene)] = max(scene_k.get(int(scene), 0), int(k_value))
    rng = np.random.default_rng(seed)
    split_by_scene: dict[int, int] = {}
    for k_value in sorted(set(scene_k.values())):
        scenes = np.asarray(
            sorted(scene for scene, value in scene_k.items() if value == k_value),
            dtype=np.int64,
        )
        rng.shuffle(scenes)
        if scenes.size >= 5:
            # Ceil keeps the held-out scene count above the critic's minimum
            # support gate after K-stratification (notably the small K=6 stratum).
            n_val = max(1, int(math.ceil(0.15 * scenes.size)))
            n_test = max(1, int(math.ceil(0.15 * scenes.size)))
        elif scenes.size >= 2:
            n_val, n_test = 0, 1
        else:
            n_val, n_test = 0, 0
        n_train = scenes.size - n_val - n_test
        for position, scene in enumerate(scenes.tolist()):
            split_by_scene[int(scene)] = 0 if position < n_train else (1 if position < n_train + n_val else 2)
    return np.asarray([split_by_scene[int(scene)] for scene in sample_index], dtype=np.int8)


def variant_hash(mask: np.ndarray, weights: np.ndarray, strategy: str) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mask, dtype=np.int8).tobytes())
    digest.update(np.asarray(weights, dtype=np.float32).tobytes())
    digest.update(strategy.encode("utf-8"))
    return digest.hexdigest()[:32]


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    packages = [
        (
            "trusted_sparse_positive",
            load_npz(args.positive_dataset_dir / "dataset_arrays.npz"),
            read_csv(args.positive_hfss_dir / "candidate_residual_labels.csv"),
        ),
        (
            "dense_boundary",
            load_npz(args.boundary_dataset_dir / "dataset_arrays.npz"),
            read_csv(args.boundary_hfss_dir / "candidate_residual_labels.csv"),
        ),
    ]
    if not args.exclude_expanded:
        packages.append(
            (
                "expanded_independent",
                load_npz(args.expanded_dataset_dir / "dataset_arrays.npz"),
                read_csv(args.expanded_hfss_dir / "candidate_residual_labels.csv"),
            )
        )
    with np.load(args.operator, allow_pickle=False) as source:
        s_matrix, _antenna_map, _series_z = series_network_map(
            np.asarray(source["s_raw"], dtype=np.complex128), 1.0e10
        )

    positive_reference: dict[int, tuple[float, float]] = {}
    fallback_reference: dict[int, tuple[float, float]] = {}
    for _source_name, dataset, rows in packages:
        row_by_candidate = {i(row, "candidate_index"): row for row in rows}
        for candidate in range(int(dataset["candidate_indices"].size)):
            row = row_by_candidate[candidate]
            sample = int(dataset["sample_indices"][candidate])
            reference = (
                f(row, "hfss_mainlobe_gain_db"),
                f(row, "hfss_target_spread_db"),
            )
            fallback_reference.setdefault(sample, reference)
            variant_kind = (
                str(dataset["variant_kind"][candidate])
                if "variant_kind" in dataset
                else ""
            )
            if variant_kind == "nominal_control" or (
                _source_name == "trusted_sparse_positive" and i(row, "strict_engineering_gate")
            ):
                positive_reference[sample] = reference
    for sample, reference in fallback_reference.items():
        positive_reference.setdefault(sample, reference)

    records: list[dict[str, Any]] = []
    for source_name, dataset, rows in packages:
        if len(rows) != int(dataset["candidate_indices"].size):
            raise ValueError(f"Candidate/label count mismatch in {source_name}")
        row_by_candidate = {i(row, "candidate_index"): row for row in rows}
        for candidate in range(int(dataset["candidate_indices"].size)):
            row = row_by_candidate[candidate]
            k_value = int(dataset["k_values"][candidate])
            tasks_ri = np.asarray(dataset["task_weights_real_imag"][candidate], dtype=np.float32)
            tasks_internal = tasks_ri[..., 0] + 1j * tasks_ri[..., 1]
            tasks_internal = tasks_internal[:, :k_value]
            mask = np.asarray(dataset["masks"][candidate], dtype=bool)
            valid = np.asarray(dataset["task_valid"][candidate], dtype=bool)
            targets = np.asarray(dataset["targets_deg"][candidate][valid], dtype=np.float64)
            energy, weight_l2, max_amp, dynamic_db = amplitude_stats(tasks_internal)
            nominal_rl, nominal_total_rl = nominal_active_rl(tasks_internal, mask, s_matrix)
            sample = int(dataset["sample_indices"][candidate])
            reference_peak, reference_spread = positive_reference[sample]
            variant_kind = (
                str(dataset["variant_kind"][candidate])
                if "variant_kind" in dataset
                else source_name
            )
            phase_rms = (
                float(dataset["phase_error_rms_deg"][candidate])
                if "phase_error_rms_deg" in dataset
                else 0.0
            )
            gain_rms = (
                float(dataset["gain_error_rms_db"][candidate])
                if "gain_error_rms_db" in dataset
                else 0.0
            )
            dropout = (
                int(dataset["dropout_count"][candidate])
                if "dropout_count" in dataset
                else 0
            )
            phase_bits = (
                int(dataset["phase_bits"][candidate])
                if "phase_bits" in dataset and variant_kind != "paired_lower_ratio"
                else 16
            )
            amplitude_bits = (
                int(dataset["amplitude_bits"][candidate])
                if "amplitude_bits" in dataset and variant_kind != "paired_lower_ratio"
                else 16
            )
            ratio_delta = (
                float(dataset["ratio_delta"][candidate]) if "ratio_delta" in dataset else 0.0
            )
            phase_ramp = (
                float(dataset["phase_ramp_deg"][candidate])
                if "phase_ramp_deg" in dataset
                else 0.0
            )
            eep = np.asarray(
                [
                    f(row, "eep_psll_db"),
                    f(row, "eep_nearest_iso_db"),
                    f(row, "eep_local_iso_db"),
                    f(row, "eep_mainlobe_gain_db"),
                    f(row, "eep_target_spread_db"),
                    energy,
                ],
                dtype=np.float32,
            )
            hfss = np.asarray(
                [
                    f(row, "hfss_psll_db"),
                    f(row, "hfss_nearest_iso_db"),
                    f(row, "hfss_local_iso_db"),
                    f(row, "hfss_mainlobe_gain_db"),
                    f(row, "hfss_target_spread_db"),
                    f(row, "hfss_pointing_error_deg"),
                ],
                dtype=np.float32,
            )
            main_drop = reference_peak - float(hfss[3])
            rank = (
                max(float(hfss[0]), 0.0)
                + 0.5 * max(25.0 - float(hfss[1]), 0.0)
                + 0.35 * max(20.0 - float(hfss[2]), 0.0)
                + 0.5 * max(main_drop - 0.5, 0.0)
                + 0.25 * max(float(hfss[4]) - 3.0, 0.0)
                + 0.25 * max(float(hfss[5]) - 1.5, 0.0)
                + 0.5 * max(10.0 - f(row, "all_case_worst_active_rl_db"), 0.0)
            )
            max_theta = float(np.max(targets[:, 0])) if targets.size else 0.0
            mean_theta = float(np.mean(targets[:, 0])) if targets.size else 0.0
            scalar = np.asarray(
                [
                    k_value / KMAX,
                    float(np.mean(mask)),
                    float(np.sum(mask)) / NUM_ELEMENTS,
                    *eep[:5].tolist(),
                    energy,
                    energy / max(10.0 ** (float(hfss[3]) / 10.0), 1.0e-20),
                    weight_l2,
                    max_amp,
                    dynamic_db,
                    0.0,
                    0.0,
                    min_separation(targets) / 180.0,
                    max_theta / 90.0,
                    mean_theta / 90.0,
                    reference_peak,
                    reference_spread,
                    nominal_rl,
                    nominal_total_rl,
                    phase_rms / 35.0,
                    gain_rms / 1.75,
                    dropout / max(float(np.sum(mask)), 1.0),
                    phase_bits / 16.0,
                    amplitude_bits / 16.0,
                    ratio_delta,
                    phase_ramp / 5.0,
                ],
                dtype=np.float32,
            )
            records.append(
                {
                    "source": source_name,
                    "strategy": variant_kind,
                    "sample_index": sample,
                    "k": k_value,
                    "active_ratio": float(np.mean(mask)),
                    "num_active": int(np.sum(mask)),
                    "mask": mask.astype(np.int8),
                    "weights_ri": tasks_ri,
                    "targets_deg": np.asarray(dataset["targets_deg"][candidate], dtype=np.float32),
                    "task_valid": np.asarray(dataset["task_valid"][candidate], dtype=np.int8),
                    "eep": eep,
                    "hfss": hfss,
                    "residual": hfss[:5] - eep[:5],
                    "gates": np.asarray(
                        [
                            i(row, "gate15"),
                            i(row, "strict_gate20"),
                            i(row, "mainlobe_gate"),
                            i(row, "strict_engineering_gate"),
                        ],
                        dtype=np.float32,
                    ),
                    "scalar": scalar,
                    "rank": float(rank),
                    "hard_negative": i(row, "hard_negative"),
                    "phase_error_rms_deg": phase_rms,
                    "gain_error_rms_db": gain_rms,
                    "dropout_count": dropout,
                    "ratio_delta": ratio_delta,
                    "phase_ramp_deg": phase_ramp,
                }
            )

    sample_index = np.asarray([record["sample_index"] for record in records], dtype=np.int64)
    k_values = np.asarray([record["k"] for record in records], dtype=np.int8)
    split_id = scene_split(sample_index, k_values, int(args.seed))
    weights = np.stack([record["weights_ri"] for record in records]).astype(np.float32)
    masks = np.stack([record["mask"] for record in records]).astype(np.int8)
    strategy = np.asarray([record["strategy"] for record in records])
    payload = {
        "masks": masks,
        "weights_real_imag": weights,
        "targets_deg": np.stack([record["targets_deg"] for record in records]),
        "task_valid": np.stack([record["task_valid"] for record in records]),
        "sample_index": sample_index,
        "split_id": split_id,
        "source": np.asarray([record["source"] for record in records]),
        "strategy": strategy,
        "variant_hash": np.asarray(
            [variant_hash(mask, weight, name) for mask, weight, name in zip(masks, weights, strategy)]
        ),
        "k_values": k_values,
        "active_ratios": np.asarray([record["active_ratio"] for record in records], dtype=np.float32),
        "num_active": np.asarray([record["num_active"] for record in records], dtype=np.int16),
        "af_metrics": np.stack([record["eep"] for record in records]),
        "af_metric_names": np.asarray(AF_METRIC_NAMES),
        "hfss_metrics": np.stack([record["hfss"] for record in records]),
        "hfss_metric_names": np.asarray(HFSS_METRIC_NAMES),
        "residuals": np.stack([record["residual"] for record in records]),
        "residual_names": np.asarray(RESIDUAL_NAMES),
        "gates": np.stack([record["gates"] for record in records]),
        "gate_names": np.asarray(GATE_NAMES),
        "scalar_features": np.stack([record["scalar"] for record in records]),
        "scalar_names": np.asarray(SCALAR_NAMES),
        "rank_violation": np.asarray([record["rank"] for record in records], dtype=np.float32),
        "hard_negative": np.asarray([record["hard_negative"] for record in records], dtype=np.int8),
        "element_ixiy": np.asarray(packages[0][1]["element_ixiy"]),
        "positions_lambda": np.asarray(packages[0][1]["positions_lambda"]),
    }
    np.savez_compressed(args.out_dir / "fullwave_residual_dataset_v2.npz", **payload)

    csv_rows = []
    for index, record in enumerate(records):
        csv_rows.append(
            {
                "row_index": index,
                "sample_index": record["sample_index"],
                "split": ("train", "val", "test")[int(split_id[index])],
                "source": record["source"],
                "strategy": record["strategy"],
                "k": record["k"],
                "active_ratio": record["active_ratio"],
                "num_active": record["num_active"],
                "hard_negative": record["hard_negative"],
                "gate15": int(record["gates"][0]),
                "gate20": int(record["gates"][1]),
                "mainlobe_gate": int(record["gates"][2]),
                "strict_engineering_gate": int(record["gates"][3]),
                "delta_psll_db": float(record["residual"][0]),
                "delta_nearest_iso_db": float(record["residual"][1]),
                "delta_local_iso_db": float(record["residual"][2]),
                "delta_mainlobe_gain_db": float(record["residual"][3]),
                "delta_target_spread_db": float(record["residual"][4]),
                "phase_error_rms_deg": record["phase_error_rms_deg"],
                "gain_error_rms_db": record["gain_error_rms_db"],
                "dropout_count": record["dropout_count"],
                "ratio_delta": record["ratio_delta"],
                "phase_ramp_deg": record["phase_ramp_deg"],
                "rank_violation": record["rank"],
            }
        )
    write_csv(args.out_dir / "fullwave_residual_dataset_v2.csv", csv_rows)

    split_summary = {}
    for split_value, split_name in enumerate(("train", "val", "test")):
        select = split_id == split_value
        split_summary[split_name] = {
            "candidate_count": int(np.sum(select)),
            "scene_count": int(np.unique(sample_index[select]).size),
            "hard_negative_count": int(np.sum(payload["hard_negative"][select])),
            "strict_positive_count": int(np.sum(payload["gates"][select, 3])),
            "sample_indices": sorted(np.unique(sample_index[select]).astype(int).tolist()),
        }
    sets = [set(split_summary[name]["sample_indices"]) for name in ("train", "val", "test")]
    leakage_free = not any(sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3))
    residual_std = np.std(payload["residuals"], axis=0)
    hard_negative_count = int(np.sum(payload["hard_negative"]))
    active_positive_count = int(np.sum(payload["gates"][:, 3]))
    training_allowed = bool(
        leakage_free
        and hard_negative_count >= 10
        and active_positive_count >= 10
        and float(np.max(residual_std)) >= 1.0e-3
        and all(summary["candidate_count"] > 0 for summary in split_summary.values())
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(records),
        "independent_scene_count": int(np.unique(sample_index).size),
        "source_counts": {
            source_name: int(sum(record["source"] == source_name for record in records))
            for source_name in sorted({record["source"] for record in records})
        },
        "hard_negative_count": hard_negative_count,
        "strict_engineering_positive_count": active_positive_count,
        "residual_std_db": {
            name: float(value) for name, value in zip(RESIDUAL_NAMES, residual_std)
        },
        "scene_leakage_free": leakage_free,
        "split_summary": split_summary,
        "residual_critic_training_allowed": training_allowed,
    }
    (args.out_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
