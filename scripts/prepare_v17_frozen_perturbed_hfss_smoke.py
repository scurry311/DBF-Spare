#!/usr/bin/env python3
"""Freeze 20 unseen-scene v1.7 candidates for perturbed-source 16x16 HFSS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import metric_at
from generate_v09_eep_development_candidates import (
    MARGIN_NAMES,
    METRIC_NAMES,
    full_active_metrics,
    metric_vector,
    physical_margins,
)
from run_v16_robust_drift_oracle import (
    LEVELS,
    PROFILE_NAMES,
    ROBUST_MARGIN_NAMES,
    apply_calibration,
    build_corners,
    load_nominal_operator,
    load_npz,
    ri_to_complex,
    scene_calibration_states,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "hfss_outputs" / "v16_robust_drift_oracle_20260727_run01"
DEFAULT_FIRST = ROOT / "hfss_outputs" / "v16_k6_multifrequency_rescue_20260727_run01"
DEFAULT_SECOND = ROOT / "hfss_outputs" / "v16_k6_quantized_boundary_rescue_20260727_run01"
DEFAULT_PROTOCOL = ROOT / "configs" / "v17_frozen_16x16_perturbed_hfss_smoke.json"
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_DRIFT = ROOT / "hfss_outputs" / "v14_operator_drift_4x4_smoke_20260727_run01"
DEFAULT_PREVIOUS_SMOKE = ROOT / "hfss_outputs" / "v13_frozen_k246_hfss_smoke_dataset_20260727_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v17_frozen_16x16_perturbed_hfss_smoke_dataset_20260728_run01"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--first-rescue-dir", type=Path, default=DEFAULT_FIRST)
    parser.add_argument("--second-rescue-dir", type=Path, default=DEFAULT_SECOND)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--previous-smoke-dir", type=Path, default=DEFAULT_PREVIOUS_SMOKE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


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


def digest(*values: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def full_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def margin_values(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row[f"E2_{name}_margin_db"]) for name in ROBUST_MARGIN_NAMES],
        dtype=np.float32,
    )


def target_metrics(targets: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(targets, dtype=float)
    maximum = float(np.max(valid[:, 0]))
    vectors = np.column_stack(
        (
            np.sin(np.deg2rad(valid[:, 0])) * np.cos(np.deg2rad(valid[:, 1])),
            np.sin(np.deg2rad(valid[:, 0])) * np.sin(np.deg2rad(valid[:, 1])),
            np.cos(np.deg2rad(valid[:, 0])),
        )
    )
    cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
    distance = np.rad2deg(np.arccos(cosine))
    distance[np.eye(len(valid), dtype=bool)] = np.inf
    return maximum, float(np.min(distance))


def record_score(record: dict[str, Any], role: str) -> tuple[float, ...]:
    margins = margin_values(record["row"])
    worst = float(np.min(margins))
    if role == "reserve":
        return (-worst, -float(np.sum(np.clip(margins, 0.0, 6.0))))
    if role == "large_scan":
        return (-float(record["max_target_theta_deg"]), -worst)
    if role == "small_gap":
        return (float(record["min_target_separation_deg"]), -worst)
    index = {
        "psll_boundary": 0,
        "nearest_boundary": 1,
        "local_boundary": 2,
        "mainlobe_boundary": 3,
        "active_rl_boundary": 4,
    }[role]
    other = np.delete(margins, index)
    return (float(margins[index]), -float(np.min(other)), -worst)


def assign_slots(
    records: list[dict[str, Any]], slots: list[tuple[str, float]]
) -> list[dict[str, Any]]:
    eligible: list[list[dict[str, Any]]] = []
    for role, ratio in slots:
        matches = [
            record
            for record in records
            if np.isclose(float(record["row"]["ratio"]), ratio, atol=1.0e-5)
        ]
        best_by_scene: dict[int, dict[str, Any]] = {}
        for record in matches:
            sample = int(record["row"]["sample_index"])
            current = best_by_scene.get(sample)
            if current is None or record_score(record, role) < record_score(current, role):
                best_by_scene[sample] = record
        ordered = sorted(best_by_scene.values(), key=lambda record: record_score(record, role))
        if not ordered:
            raise RuntimeError(f"No E2-strict candidate for role={role}, ratio={ratio}")
        eligible.append(ordered)

    assignment: dict[int, dict[str, Any]] = {}
    used_scenes: set[int] = set()
    order = sorted(range(len(slots)), key=lambda index: len(eligible[index]))

    def visit(position: int) -> bool:
        if position == len(order):
            return True
        slot_index = order[position]
        for record in eligible[slot_index]:
            sample = int(record["row"]["sample_index"])
            if sample in used_scenes:
                continue
            assignment[slot_index] = record
            used_scenes.add(sample)
            if visit(position + 1):
                return True
            used_scenes.remove(sample)
            del assignment[slot_index]
        return False

    if not visit(0):
        raise RuntimeError(f"Cannot assign independent scenes to slots: {slots}")
    return [assignment[index] for index in range(len(slots))]


def load_candidate_records(
    args: argparse.Namespace,
    pool: dict[str, np.ndarray],
    scene_metadata: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    final = load_npz(args.parent_dir / "final" / "robust_arrays.npz")
    final_tasks = ri_to_complex(final["tasks_real_imag"])
    for row in read_csv(args.parent_dir / "final" / "robust_candidate_metrics.csv"):
        sample = int(row["sample_index"])
        if scene_metadata[sample]["scene_origin"] != "new30" or int(row["E2_strict_pass"]) != 1:
            continue
        evaluation = int(row["evaluation_index"])
        candidate = int(row["candidate_index"])
        targets = np.asarray(pool["targets"][candidate], dtype=np.float32)
        k_value = int(row["k_value"])
        maximum, separation = target_metrics(targets[:k_value])
        records.append(
            {
                "row": row,
                "source": "v16_final",
                "source_index": evaluation,
                "command": final_tasks[evaluation],
                "mask": np.asarray(pool["masks"][candidate], dtype=bool),
                "targets": targets,
                "max_target_theta_deg": maximum,
                "min_target_separation_deg": separation,
            }
        )

    rescue = load_npz(args.parent_dir / "rescue" / "rescue_candidates.npz")
    rescue_tasks = ri_to_complex(rescue["tasks_real_imag"])
    for row in read_csv(args.parent_dir / "post_rescue" / "rescue_robust_candidate_metrics.csv"):
        sample = int(row["sample_index"])
        if scene_metadata[sample]["scene_origin"] != "new30" or int(row["E2_strict_pass"]) != 1:
            continue
        index = int(row["rescue_index"])
        targets = np.asarray(rescue["targets"][index], dtype=np.float32)
        k_value = int(row["k_value"])
        maximum, separation = target_metrics(targets[:k_value])
        records.append(
            {
                "row": row,
                "source": "v16_mask_rescue",
                "source_index": index,
                "command": rescue_tasks[index],
                "mask": np.asarray(rescue["masks"][index], dtype=bool),
                "targets": targets,
                "max_target_theta_deg": maximum,
                "min_target_separation_deg": separation,
            }
        )

    for source, directory in (
        ("v17_multifrequency", args.first_rescue_dir),
        ("v17_quantized_boundary", args.second_rescue_dir),
    ):
        arrays = load_npz(directory / "candidates" / "candidate_commands.npz")
        tasks = ri_to_complex(arrays["tasks_real_imag"])
        for row in read_csv(directory / "evaluation" / "candidate_metrics.csv"):
            sample = int(row["sample_index"])
            if scene_metadata[sample]["scene_origin"] != "new30" or int(row["E2_strict_pass"]) != 1:
                continue
            index = int(row["candidate_index"])
            targets = np.asarray(arrays["targets"][index], dtype=np.float32)
            maximum, separation = target_metrics(targets[:6])
            records.append(
                {
                    "row": row,
                    "source": source,
                    "source_index": index,
                    "command": tasks[index],
                    "mask": np.asarray(arrays["masks"][index], dtype=bool),
                    "targets": targets,
                    "max_target_theta_deg": maximum,
                    "min_target_separation_deg": separation,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen HFSS smoke: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    pool = load_npz(args.parent_dir / "pool" / "candidate_pool.npz")
    scene_rows = read_csv(args.parent_dir / "pool" / "scene_manifest.csv")
    scene_metadata = {int(row["sample_index"]): row for row in scene_rows}
    records = load_candidate_records(args, pool, scene_metadata)

    selected: list[tuple[int, str, float, dict[str, Any]]] = []
    for k_value in (2, 4, 6):
        slots = [
            (str(role), float(ratio))
            for role, ratio in protocol["selection_schedule"][f"K{k_value}"]
        ]
        members = [record for record in records if int(record["row"]["k_value"]) == k_value]
        for (role, ratio), record in zip(slots, assign_slots(members, slots)):
            selected.append((k_value, role, ratio, record))

    samples = [int(record["row"]["sample_index"]) for _k, _role, _ratio, record in selected]
    if len(selected) != int(protocol["candidate_count"]) or len(set(samples)) != len(selected):
        raise RuntimeError("Frozen selection is incomplete or leaks independent scenes")
    with np.load(args.previous_smoke_dir / "dataset_arrays.npz", allow_pickle=False) as previous:
        old_hashes = set(np.asarray(previous["target_hashes"]).tolist())
    selected_hashes = {scene_metadata[sample]["target_hash"] for sample in samples}
    overlap = old_hashes & selected_hashes
    if overlap:
        raise RuntimeError(f"Previous HFSS target-set overlap: {sorted(overlap)}")

    corner_args = SimpleNamespace(operator=args.operator, drift_dir=args.drift_dir)
    base, corners = build_corners(corner_args, levels=(0.20,))
    _base, _effective, nominal_fast, nominal_s = load_nominal_operator(args.operator)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    count = len(selected)
    k_values = np.asarray([item[0] for item in selected], dtype=np.int8)
    masks = np.stack([item[3]["mask"] for item in selected]).astype(np.int8)
    targets = np.stack([item[3]["targets"] for item in selected]).astype(np.float32)
    commands = np.stack([item[3]["command"] for item in selected]).astype(np.complex64)
    actual_commands = np.zeros_like(commands)
    reference_metrics = np.zeros((count, len(METRIC_NAMES)), dtype=np.float32)
    nominal_metrics = np.zeros_like(reference_metrics)
    actual_metrics = np.zeros_like(reference_metrics)
    nominal_margins = np.zeros((count, len(MARGIN_NAMES)), dtype=np.float32)
    robust_margins = np.zeros_like(nominal_margins)
    manifest: list[dict[str, Any]] = []
    corner_names: list[str] = []
    perturbation_seeds: list[int] = []
    phase_rms: list[float] = []
    gain_rms: list[float] = []
    dropout_counts: list[int] = []
    mask_hashes: list[str] = []
    nominal_hashes: list[str] = []
    actual_hashes: list[str] = []

    for index, (k_value, role, ratio, record) in enumerate(selected):
        sample = int(record["row"]["sample_index"])
        command = np.asarray(commands[index], dtype=np.complex64)
        command[:, k_value:] = 0.0
        commands[index] = command
        mask = np.asarray(masks[index], dtype=bool)
        target = np.asarray(targets[index, :k_value], dtype=float)
        values = margin_values(record["row"])
        cause = str(ROBUST_MARGIN_NAMES[int(np.argmin(values))])
        corner_name = str(record["row"][f"E2_{cause}_worst_corner"])
        if corner_name not in corners:
            raise KeyError(f"Unknown frozen E2 corner: {corner_name}")
        pool_scene = np.flatnonzero(pool_samples == sample)
        states = scene_calibration_states(
            pool, pool_scene, corners, base["element_ixiy"], int(protocol["frozen_E2"]["seed"])
        )
        state = states[corner_name]
        actual = apply_calibration(command, mask, state)
        actual[:, k_value:] = 0.0
        actual_commands[index] = actual

        nominal_item = metric_at(nominal_fast.evaluate(command[None, :, :k_value], target), 0)
        actual_item = metric_at(corners[corner_name]["fast"].evaluate(actual[None, :, :k_value], target), 0)
        reference_metrics[index] = metric_vector(nominal_item)
        nominal_metrics[index] = metric_vector(nominal_item)
        actual_metrics[index] = metric_vector(actual_item)
        nominal_active = full_active_metrics(command[:, :k_value], mask, nominal_s)
        nominal_margins[index] = physical_margins(nominal_item, nominal_item, nominal_active)
        robust_margins[index] = values[: len(MARGIN_NAMES)]

        profile = str(corners[corner_name]["profile"])
        profile_index = PROFILE_NAMES.index(profile)
        level_index = LEVELS.index(0.20)
        perturbation_seed = int(
            int(protocol["frozen_E2"]["seed"])
            + sample * 97
            + profile_index * 100003
            + level_index * 1009
        )
        mask_hash = digest(mask.astype(np.int8))
        nominal_hash = digest(command[:, :k_value], np.sum(command[:, :k_value], axis=1))
        actual_hash = digest(actual[:, :k_value], np.sum(actual[:, :k_value], axis=1))
        mask_hashes.append(mask_hash)
        nominal_hashes.append(nominal_hash)
        actual_hashes.append(actual_hash)
        corner_names.append(corner_name)
        perturbation_seeds.append(perturbation_seed)
        phase_rms.append(float(state["phase_rms_deg"]))
        gain_rms.append(float(state["gain_rms_db"]))
        dropout = int(np.sum(mask[np.asarray(state["soft_ports"], dtype=int)]))
        dropout_counts.append(dropout)
        manifest.append(
            {
                "candidate_index": index,
                "sample_index": sample,
                "target_hash": scene_metadata[sample]["target_hash"],
                "k_value": k_value,
                "ratio": ratio,
                "active_count": int(np.sum(mask)),
                "selection_role": role,
                "candidate_source": record["source"],
                "source_candidate_index": record["source_index"],
                "candidate_origin": record["row"].get("candidate_origin", ""),
                "frozen_E2_corner": corner_name,
                "frozen_E2_root_margin": cause,
                "E2_worst_margin_db": float(np.min(values)),
                **{
                    f"E2_{name}_margin_db": float(value)
                    for name, value in zip(ROBUST_MARGIN_NAMES, values)
                },
                "max_target_theta_deg": float(record["max_target_theta_deg"]),
                "min_target_separation_deg": float(record["min_target_separation_deg"]),
                "phase_rms_deg": float(state["phase_rms_deg"]),
                "gain_rms_db": float(state["gain_rms_db"]),
                "soft_active_port_count": dropout,
                "phase_bits": int(state["phase_bits"]),
                "amplitude_bits": int(state["amplitude_bits"]),
                "perturbation_seed": perturbation_seed,
                "mask_sha256": mask_hash,
                "nominal_weights_sha256": nominal_hash,
                "actual_weights_sha256": actual_hash,
                "physical_16x16_operator_corner_included": 0,
            }
        )

    task_valid = np.arange(6)[None, :] < k_values[:, None]
    combined = np.sum(commands, axis=2)
    actual_combined = np.sum(actual_commands, axis=2)
    internal_commands = np.conjugate(commands)
    actual_internal_commands = np.conjugate(actual_commands)
    internal_combined = np.conjugate(combined)
    actual_internal_combined = np.conjugate(actual_combined)
    if not np.allclose(np.conjugate(internal_commands), commands, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("Nominal command convention round-trip failed")
    if not np.allclose(
        np.conjugate(actual_internal_commands), actual_commands, rtol=0.0, atol=1.0e-7
    ):
        raise RuntimeError("Perturbed command convention round-trip failed")
    element_ixiy = np.asarray(base["element_ixiy"], dtype=np.int64)
    positions = np.column_stack(
        ((element_ixiy[:, 0] - 7.5) * 0.5, (element_ixiy[:, 1] - 7.5) * 0.5, np.zeros(256))
    )
    implementation_delta = actual_commands - commands
    payload = {
        "candidate_index": np.arange(count, dtype=np.int64),
        "candidate_indices": np.arange(count, dtype=np.int64),
        "sample_index": np.asarray(samples, dtype=np.int64),
        "sample_indices": np.asarray(samples, dtype=np.int64),
        "sample_ids": np.asarray([f"v17_{sample}" for sample in samples]),
        "scene_ids": np.asarray([f"v17_scene_{sample}" for sample in samples]),
        "target_hashes": np.asarray([scene_metadata[sample]["target_hash"] for sample in samples]),
        "source_dataset": np.asarray([item[3]["source"] for item in selected]),
        "source_sample_indices": np.asarray(samples, dtype=np.int64),
        "selection_roles": np.asarray([item[1] for item in selected]),
        "variant_kind": np.asarray(
            [item[3]["row"].get("candidate_origin", item[3]["source"]) for item in selected]
        ),
        "split_id": np.full(count, -1, dtype=np.int8),
        "k_values": k_values,
        "active_ratios_requested": np.asarray([item[2] for item in selected], dtype=np.float32),
        "active_ratios_actual": np.mean(masks, axis=1, dtype=np.float32),
        "num_active": np.sum(masks, axis=1).astype(np.int16),
        "targets_deg": targets,
        "task_valid": task_valid.astype(np.int8),
        "mask": masks,
        "masks": masks,
        "w_tasks_real_imag": np.stack((commands.real, commands.imag), axis=-1).astype(np.float32),
        "task_weights_real_imag": np.stack(
            (internal_commands.real, internal_commands.imag), axis=-1
        ).astype(np.float32),
        "w_combined_real_imag": np.stack((combined.real, combined.imag), axis=-1).astype(np.float32),
        "combined_weights_real_imag": np.stack(
            (internal_combined.real, internal_combined.imag), axis=-1
        ).astype(np.float32),
        "hfss_actual_task_weights_real_imag": np.stack(
            (actual_internal_commands.real, actual_internal_commands.imag), axis=-1
        ).astype(np.float32),
        "hfss_actual_combined_weights_real_imag": np.stack(
            (actual_internal_combined.real, actual_internal_combined.imag), axis=-1
        ).astype(np.float32),
        "hfss_weights_real_imag": np.stack(
            (actual_internal_combined.real, actual_internal_combined.imag), axis=-1
        ).astype(np.float32),
        "nominal_external_task_weights_real_imag": np.stack(
            (commands.real, commands.imag), axis=-1
        ).astype(np.float32),
        "actual_external_task_weights_real_imag": np.stack(
            (actual_commands.real, actual_commands.imag), axis=-1
        ).astype(np.float32),
        "weight_storage_convention": np.asarray(
            ["legacy_internal_conjugate; external commands stored explicitly"] * count
        ),
        "reference_metrics": reference_metrics,
        "nominal_metrics": nominal_metrics,
        "actual_metrics": actual_metrics,
        "metric_names": METRIC_NAMES,
        "nominal_margins": nominal_margins,
        "actual_margins": robust_margins,
        "margin_residuals": robust_margins - nominal_margins,
        "margin_names": MARGIN_NAMES,
        "gate15": np.ones(count, dtype=np.int8),
        "strict_gate20": np.ones(count, dtype=np.int8),
        "mainlobe_gate": np.ones(count, dtype=np.int8),
        "active_rl_gate": np.ones(count, dtype=np.int8),
        "near_boundary": (np.min(robust_margins, axis=1) <= 0.5).astype(np.int8),
        "hard_negative": np.zeros(count, dtype=np.int8),
        "hard_positive": np.asarray(
            [(k == 6 or theta >= 40.0 or ratio <= 0.7) for k, theta, ratio in zip(
                k_values,
                [item[3]["max_target_theta_deg"] for item in selected],
                [item[2] for item in selected],
            )],
            dtype=np.int8,
        ),
        "strict_violation": np.zeros(count, dtype=np.float32),
        "min_target_separation_deg": np.asarray(
            [item[3]["min_target_separation_deg"] for item in selected], dtype=np.float32
        ),
        "max_target_theta_deg": np.asarray(
            [item[3]["max_target_theta_deg"] for item in selected], dtype=np.float32
        ),
        "large_scan": np.asarray(
            [item[3]["max_target_theta_deg"] >= 40.0 for item in selected], dtype=np.int8
        ),
        "small_target_gap": np.asarray(
            [item[3]["min_target_separation_deg"] <= 16.0 for item in selected], dtype=np.int8
        ),
        "implementation_delta_norm": np.linalg.norm(
            implementation_delta.reshape(count, -1), axis=1
        ).astype(np.float32),
        "implementation_delta_max": np.max(
            np.abs(implementation_delta).reshape(count, -1), axis=1
        ).astype(np.float32),
        "phase_error_rms_deg": np.asarray(phase_rms, dtype=np.float32),
        "gain_error_rms_db": np.asarray(gain_rms, dtype=np.float32),
        "dropout_count": np.asarray(dropout_counts, dtype=np.int16),
        "peak_shape_margin_db": robust_margins[:, 3].astype(np.float32),
        "phase_bits": np.full(count, 7, dtype=np.int16),
        "amplitude_bits": np.full(count, 7, dtype=np.int16),
        "perturbation_seed": np.asarray(perturbation_seeds, dtype=np.int64),
        "frozen_E2_corner": np.asarray(corner_names),
        "port_names": np.asarray(base["port_names"]),
        "element_ixiy": element_ixiy,
        "positions_lambda": positions.astype(np.float64),
        "frozen_mask_hashes": np.asarray(mask_hashes),
        "frozen_nominal_weight_hashes": np.asarray(nominal_hashes),
        "frozen_actual_weight_hashes": np.asarray(actual_hashes),
        "physical_16x16_operator_corner_included": np.zeros(count, dtype=np.int8),
    }
    dataset_path = args.out_dir / "dataset_arrays.npz"
    np.savez_compressed(dataset_path, **payload)
    write_csv(args.out_dir / "frozen_selection_manifest.csv", manifest)
    (args.out_dir / "frozen_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    summary = {
        "candidate_count": count,
        "independent_scene_count": len(set(samples)),
        "k_counts": {str(k): int(np.sum(k_values == k)) for k in (2, 4, 6)},
        "ratio_counts": {
            str(ratio): int(np.sum(np.isclose(payload["active_ratios_requested"], ratio)))
            for ratio in (0.5, 0.6, 0.7, 0.8)
        },
        "expected_hfss_case_count": int(np.sum(1 + k_values)),
        "previous_hfss_target_hash_overlap": len(overlap),
        "all_selected_E2_strict": bool(np.all(np.min(robust_margins, axis=1) >= 0.0)),
        "weights_frozen": True,
        "thresholds_frozen": True,
        "dataset_sha256": full_file_hash(dataset_path),
        "physical_16x16_operator_corner_included": False,
        "hfss_scope": "trusted nominal 16x16 field basis with frozen E2 source/calibration perturbations",
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
