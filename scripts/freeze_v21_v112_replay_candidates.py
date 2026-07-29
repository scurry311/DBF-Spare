#!/usr/bin/env python3
"""Freeze the v1.12 scene-level masks and task commands for hardware replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import metric_at
from generate_v09_eep_development_candidates import METRIC_NAMES
from run_v16_robust_drift_oracle import complex_to_ri, load_npz, ri_to_complex
from run_v19_nominal_9p96_joint_projection import identity_state
from run_v20_three_frequency_mask_weight_joint import operator_bundle, scene_states
from generate_v14_operator_drift_dataset import apply_calibration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v21_three_frequency_broadband_matching.json"
DEFAULT_V20 = ROOT / "hfss_outputs" / "v20_three_frequency_joint_search_20260729_run02"
DEFAULT_ALPHA = ROOT / "hfss_outputs" / "v201_dense_alpha_eep_20260729_run01"
DEFAULT_PROJECTED = ROOT / "hfss_outputs" / "v19_nominal_9p96_joint_projection_20260729_run01" / "projected_commands.npz"
DEFAULT_SOURCE = ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_evaluation_20260729_run01" / "dataset_arrays.npz"
DEFAULT_POOL = ROOT / "hfss_outputs" / "v16_robust_drift_oracle_20260727_run01" / "pool" / "candidate_pool.npz"
DEFAULT_NOMINAL = ROOT / "hfss_outputs" / "fixed_mesh_eep256_20260723_run05" / "grounded_patch_eep_operator_256port.npz"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v21_frozen_v112_replay_20260729_run03"
STATE_NAMES = (
    "nominal_identity",
    "frequency_low_identity",
    "frequency_low_E2_source",
    "frequency_high_identity",
    "frequency_high_E2_source",
)
KMAX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--v20-dir", type=Path, default=DEFAULT_V20)
    parser.add_argument("--alpha-dir", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--projected", type=Path, default=DEFAULT_PROJECTED)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packed_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def row_choice_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row["ratio"]), -float(row["robust_worst_margin_db"]))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen replay set: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    v20_rows = read_csv(args.v20_dir / "full_refined_candidate_metrics.csv")
    v20_arrays = load_npz(args.v20_dir / "full_refined_candidates.npz")
    alpha_rows = read_csv(args.alpha_dir / "dense_alpha_metrics.csv")
    alpha_arrays = load_npz(args.alpha_dir / "dense_alpha_commands.npz")
    generated = load_npz(args.v20_dir / "generated_mask_pool.npz")
    projected = load_npz(args.projected)
    source = load_npz(args.source)
    pool = load_npz(args.pool)
    nominal_base, nominal = operator_bundle(args.nominal_operator, 10.0)

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(v20_rows):
        evaluation = int(row["evaluation_index"])
        if evaluation != index or int(v20_arrays["evaluation_index"][evaluation]) != evaluation:
            raise RuntimeError("v20 CSV/NPZ evaluation ordering mismatch")
        candidates.append({"source": "v20", "row_index": index, **row})
    for index, row in enumerate(alpha_rows):
        candidates.append({"source": "dense_alpha", "row_index": index, **row})

    selected: list[dict[str, Any]] = []
    for sample in sorted({int(row["sample_index"]) for row in candidates}):
        members = [row for row in candidates if int(row["sample_index"]) == sample]
        strict = [row for row in members if int(row["all_corner_strict_pass"]) == 1]
        if strict:
            minimum_ratio = min(float(row["ratio"]) for row in strict)
            eligible = [row for row in strict if np.isclose(float(row["ratio"]), minimum_ratio)]
            best = max(eligible, key=lambda row: float(row["robust_worst_margin_db"]))
            policy = "minimum_strict_ratio_then_max_margin"
        else:
            best = max(members, key=lambda row: float(row["robust_worst_margin_db"]))
            policy = "maximum_worst_margin_failed_scene"
        selected.append({**best, "freeze_policy": policy})

    if len(selected) != 20:
        raise RuntimeError(f"Expected 20 frozen scenes, got {len(selected)}")

    projected_samples = np.asarray(projected["sample_index"], dtype=np.int64)
    projected_lookup = {int(sample): index for index, sample in enumerate(projected_samples)}
    source_indices = np.asarray(projected["source_candidate_index"], dtype=np.int64)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    tasks_out: list[np.ndarray] = []
    state_tasks_out: list[np.ndarray] = []
    masks_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    references_out: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    identity = identity_state()
    for freeze_index, row in enumerate(selected):
        sample = int(row["sample_index"])
        k_value = int(float(row["k"]))
        if row["source"] == "v20":
            source_index = int(row["row_index"])
            command = ri_to_complex(v20_arrays["tasks_real_imag"][source_index, :, :k_value])
            mask = np.asarray(v20_arrays["masks"][source_index], dtype=bool)
            targets = np.asarray(v20_arrays["targets"][source_index, :k_value], dtype=float)
        else:
            source_index = int(row["row_index"])
            command = ri_to_complex(alpha_arrays["commands_real_imag"][source_index, :, :k_value])
            generated_index = int(row["generated_index"])
            mask = np.asarray(generated["masks"][generated_index], dtype=bool)
            v20_member = next(
                item for item in v20_rows if int(item["sample_index"]) == sample
            )
            evaluation = int(v20_member["evaluation_index"])
            targets = np.asarray(v20_arrays["targets"][evaluation, :k_value], dtype=float)
        command = np.asarray(command, dtype=np.complex64)
        command[~mask] = 0.0
        projected_index = projected_lookup[sample]
        original = ri_to_complex(
            source["nominal_external_task_weights_real_imag"][
                int(source_indices[projected_index]), :, :k_value
            ]
        )
        reference = metric_at(nominal["fast"].evaluate(original, targets), 0)
        states, _ = scene_states(
            pool,
            np.flatnonzero(pool_samples == sample),
            np.asarray(nominal_base["element_ixiy"], dtype=np.int64),
            int(protocol["frozen_state_seed"]),
        )
        states["nominal_identity"] = identity
        states["frequency_low_identity"] = identity
        states["frequency_high_identity"] = identity
        padded_command = np.zeros((256, KMAX), dtype=np.complex64)
        padded_command[:, :k_value] = command
        padded_targets = np.full((KMAX, 2), np.nan, dtype=np.float32)
        padded_targets[:k_value] = targets
        state_tasks = np.zeros((len(STATE_NAMES), 256, KMAX), dtype=np.complex64)
        for state_index, state_name in enumerate(STATE_NAMES):
            state_tasks[state_index, :, :k_value] = apply_calibration(
                command, mask, states[state_name]
            )
        tasks_out.append(padded_command)
        state_tasks_out.append(state_tasks)
        masks_out.append(mask.astype(np.int8))
        targets_out.append(padded_targets)
        references_out.append(np.asarray([float(reference[name]) for name in METRIC_NAMES], np.float32))
        manifest_rows.append(
            {
                "freeze_index": freeze_index,
                "sample_index": sample,
                "k": k_value,
                "ratio": float(row["ratio"]),
                "source": row["source"],
                "source_row_index": source_index,
                "freeze_policy": row["freeze_policy"],
                "pre_freeze_strict_pass": int(row["all_corner_strict_pass"]),
                "pre_freeze_worst_margin_db": float(row["robust_worst_margin_db"]),
                "mask_hash": packed_hash(mask.astype(np.uint8)),
                "command_hash": packed_hash(complex_to_ri(padded_command)),
                "active_count": int(np.sum(mask)),
            }
        )

    package_path = args.out_dir / "frozen_v112_replay_candidates.npz"
    np.savez_compressed(
        package_path,
        freeze_index=np.arange(len(selected), dtype=np.int64),
        sample_index=np.asarray([int(row["sample_index"]) for row in selected], np.int64),
        k_values=np.asarray([int(float(row["k"])) for row in selected], np.int8),
        ratio=np.asarray([float(row["ratio"]) for row in selected], np.float32),
        masks=np.stack(masks_out),
        targets=np.stack(targets_out),
        tasks_real_imag=complex_to_ri(np.stack(tasks_out)),
        state_tasks_real_imag=complex_to_ri(np.stack(state_tasks_out)),
        reference_metrics=np.stack(references_out),
        reference_metric_names=np.asarray(METRIC_NAMES),
        state_names=np.asarray(STATE_NAMES),
    )
    manifest_path = args.out_dir / "frozen_candidate_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    source_paths = {
        "v20_metrics": args.v20_dir / "full_refined_candidate_metrics.csv",
        "v20_arrays": args.v20_dir / "full_refined_candidates.npz",
        "alpha_metrics": args.alpha_dir / "dense_alpha_metrics.csv",
        "alpha_arrays": args.alpha_dir / "dense_alpha_commands.npz",
        "protocol": args.protocol,
    }
    summary = {
        "protocol": protocol["protocol"],
        "parent_tag": protocol["parent_tag"],
        "scene_count": len(selected),
        "strict_count": sum(int(row["pre_freeze_strict_pass"]) for row in manifest_rows),
        "strict_counts_by_k": {
            str(k): sum(
                int(row["pre_freeze_strict_pass"])
                for row in manifest_rows
                if int(row["k"]) == k
            )
            for k in (2, 4, 6)
        },
        "package_sha256": sha256(package_path),
        "manifest_sha256": sha256(manifest_path),
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "mask_or_weight_changes_allowed": False,
        "thresholds_changed": False,
        "candidate_optimization_allowed": False,
        "hfss_allowed": False,
        "critic_training_allowed": False,
    }
    (args.out_dir / "freeze_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
