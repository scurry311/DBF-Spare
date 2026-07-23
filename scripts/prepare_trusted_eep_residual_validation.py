#!/usr/bin/env python3
"""Build a scene-grouped 96-candidate dataset for trusted EEP/HFSS validation."""

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


ROOT = Path(__file__).resolve().parents[1]
BASE_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
TASK_RUN = ROOT / "hfss_outputs" / "grounded_patch_task_lcmv_psll_20260717_run02"
PAIRED_DIR = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_eep_hfss_smoke_20260717_run01"
    / "paired_scene_dataset"
)
DEFAULT_OUT = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "validation_dataset"
)
RATIOS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
K_VALUES = (1, 2, 4, 6)
KMAX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, default=BASE_DATASET)
    parser.add_argument("--task-run", type=Path, default=TASK_RUN)
    parser.add_argument("--paired-dir", type=Path, default=PAIRED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260723)
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


def finite(value: str, default: float = 1000.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boundary_score(row: dict[str, str]) -> float:
    k_value = int(row["k"])
    terms = [
        abs(finite(row["final_psll_db"]) - 0.0) / 3.0,
        abs(finite(row["mainlobe_loss_db"]) - 0.5) / 0.5,
        abs(finite(row["worst_active_rl_db"]) - 10.0) / 2.0,
        abs(finite(row["total_rl_db"]) - 10.0) / 2.0,
    ]
    if k_value > 1:
        terms.extend(
            [
                abs(finite(row["final_nearest_iso_db"]) - 25.0) / 5.0,
                abs(finite(row["final_local_iso_db"]) - 20.0) / 5.0,
            ]
        )
    return float(np.mean(np.clip(terms, 0.0, 20.0)))


def positive_score(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        float(int(row["joint_gate_pass"])),
        min(finite(row["worst_active_rl_db"]), finite(row["total_rl_db"])),
        -finite(row["final_psll_db"]),
    )


def choose_cell(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    if len(rows) < 3:
        raise RuntimeError("A K/ratio cell has fewer than three source candidates")
    selected: list[tuple[str, dict[str, str]]] = []
    used: set[int] = set()

    positives = [row for row in rows if int(row["joint_gate_pass"]) == 1]
    positive = max(positives or rows, key=positive_score)
    selected.append(("proxy_joint_positive", positive))
    used.add(int(positive["sample_index"]))

    near = min(
        (row for row in rows if int(row["sample_index"]) not in used),
        key=boundary_score,
    )
    selected.append(("proxy_near_boundary", near))
    used.add(int(near["sample_index"]))

    risk_pool = [
        row
        for row in rows
        if int(row["sample_index"]) not in used
        and int(row["large_scan"]) == 1
        and (int(row["af_gate_pass"]) == 1 or int(row["joint_gate_pass"]) == 0)
    ]
    if not risk_pool:
        risk_pool = [row for row in rows if int(row["sample_index"]) not in used]
    risk = max(
        risk_pool,
        key=lambda row: (
            int(row["large_scan"]),
            -int(row["joint_gate_pass"]),
            finite(row["max_target_theta_deg"], 0.0),
            finite(row["final_psll_db"], 0.0),
        ),
    )
    selected.append(("proxy_large_scan_risk", risk))
    return selected


def scene_hash(targets: np.ndarray, k_value: int) -> str:
    payload = np.round(np.asarray(targets[:k_value], dtype=np.float64), 6).tobytes()
    return hashlib.sha256(payload).hexdigest()[:20]


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite validation dataset: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    base_npz = np.load(args.base_dataset, allow_pickle=False)
    base = {key: base_npz[key] for key in base_npz.files}
    task_npz = np.load(args.task_run / "task_level_weights.npz", allow_pickle=False)
    task = {key: task_npz[key] for key in task_npz.files}
    task_rows = read_csv(args.task_run / "task_lcmv_psll_case_metrics.csv")
    paired_npz = np.load(args.paired_dir / "dataset_arrays.npz", allow_pickle=False)
    paired = {key: paired_npz[key] for key in paired_npz.files}
    paired_rows = read_csv(args.paired_dir / "manifest.csv")

    selected: list[dict[str, Any]] = []
    for k_value in K_VALUES:
        for ratio in RATIOS:
            cell = [
                row
                for row in task_rows
                if int(row["k"]) == k_value
                and abs(float(row["ratio_requested"]) - ratio) <= 1.0e-6
            ]
            for role, row in choose_cell(cell):
                selected.append(
                    {
                        "source_dataset": "task_level_run02",
                        "source_index": int(row["sample_index"]),
                        "selection_role": role,
                        "source_row": row,
                        "scene_key": f"task_{row['sample_index']}",
                    }
                )

    close_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in paired_rows:
        if row["scene_type"] == "new_close_5to10deg" and int(row["k"]) == 6:
            close_by_scene[row["base_scene_id"]].append(row)
    complete_close = [
        rows
        for rows in close_by_scene.values()
        if {round(float(row["ratio_requested"]), 1) for row in rows} == set(RATIOS)
    ]
    complete_close.sort(key=lambda rows: rows[0]["base_scene_id"])
    low_scan = [rows for rows in complete_close if int(rows[0]["large_scan"]) == 0]
    large_scan = [rows for rows in complete_close if int(rows[0]["large_scan"]) == 1]
    chosen_close = low_scan[:2] + large_scan[:2]
    if len(chosen_close) != 4:
        raise RuntimeError("Could not select two normal and two large-scan K=6 close scenes")
    for scene_rows in chosen_close:
        for row in sorted(scene_rows, key=lambda item: float(item["ratio_requested"])):
            selected.append(
                {
                    "source_dataset": "paired_close_k6",
                    "source_index": int(row["sample_index"]),
                    "selection_role": "k6_close_ratio_pair",
                    "source_row": row,
                    "scene_key": f"close_{row['base_scene_id']}",
                }
            )

    scene_index_by_key: dict[str, int] = {}
    n = len(selected)
    candidate_indices = np.arange(n, dtype=np.int64)
    sample_indices = np.zeros(n, dtype=np.int64)
    sample_ids = np.empty(n, dtype="U96")
    scene_ids = np.empty(n, dtype="U96")
    source_dataset = np.empty(n, dtype="U32")
    source_indices = np.zeros(n, dtype=np.int64)
    selection_roles = np.empty(n, dtype="U40")
    k_values = np.zeros(n, dtype=np.int64)
    ratios_requested = np.zeros(n, dtype=np.float32)
    ratios_actual = np.zeros(n, dtype=np.float32)
    num_active = np.zeros(n, dtype=np.int64)
    targets = np.full((n, KMAX, 2), np.nan, dtype=np.float32)
    task_valid = np.zeros((n, KMAX), dtype=np.int8)
    masks = np.zeros((n, 256), dtype=np.int8)
    task_weights_ri = np.zeros((n, 256, KMAX, 2), dtype=np.float32)
    combined_weights_ri = np.zeros((n, 256, 2), dtype=np.float32)
    min_separation = np.zeros(n, dtype=np.float32)
    max_scan = np.zeros(n, dtype=np.float32)
    large_scan_values = np.zeros(n, dtype=np.int8)
    manifest: list[dict[str, Any]] = []

    for candidate_index, item in enumerate(selected):
        scene_key = str(item["scene_key"])
        if scene_key not in scene_index_by_key:
            scene_index_by_key[scene_key] = len(scene_index_by_key)
        scene_index = scene_index_by_key[scene_key]
        source_index = int(item["source_index"])
        source_name = str(item["source_dataset"])
        if source_name == "task_level_run02":
            arrays = base
            weights = task["task_weights_real_imag"][source_index]
            combined = task["combined_weights_real_imag"][source_index]
            mask = task["masks"][source_index]
            row = item["source_row"]
            ratio_actual = float(row["ratio_effective"])
            separation = float(row["min_target_separation_deg"])
            scan = float(row["max_target_theta_deg"])
        else:
            arrays = paired
            weights = arrays["task_weights_real_imag"][source_index]
            combined = arrays["hfss_weights_real_imag"][source_index]
            mask = arrays["masks"][source_index]
            row = item["source_row"]
            ratio_actual = float(row["ratio_actual"])
            separation = float(row["min_target_separation_deg"])
            scan = float(row["max_target_theta_deg"])

        k_value = int(arrays["k_values"][source_index])
        ratio = float(arrays["active_ratios_requested"][source_index])
        valid = np.asarray(arrays["task_valid"][source_index], dtype=np.int8)
        target = np.asarray(arrays["targets_deg"][source_index], dtype=np.float32)
        digest = scene_hash(target, k_value)
        sample_id = f"tv{candidate_index:03d}_s{scene_index:03d}_k{k_value}_r{ratio:.1f}"

        candidate_indices[candidate_index] = candidate_index
        sample_indices[candidate_index] = scene_index
        sample_ids[candidate_index] = sample_id
        scene_ids[candidate_index] = f"scene_{scene_index:03d}_{digest}"
        source_dataset[candidate_index] = source_name
        source_indices[candidate_index] = source_index
        selection_roles[candidate_index] = str(item["selection_role"])
        k_values[candidate_index] = k_value
        ratios_requested[candidate_index] = ratio
        ratios_actual[candidate_index] = ratio_actual
        num_active[candidate_index] = int(np.sum(mask))
        targets[candidate_index] = target
        task_valid[candidate_index] = valid
        masks[candidate_index] = mask
        task_weights_ri[candidate_index] = weights
        combined_weights_ri[candidate_index] = combined
        min_separation[candidate_index] = separation
        max_scan[candidate_index] = scan
        large_scan_values[candidate_index] = int(scan >= 45.0)
        manifest.append(
            {
                "candidate_index": candidate_index,
                "sample_index": scene_index,
                "sample_id": sample_id,
                "scene_id": scene_ids[candidate_index],
                "source_dataset": source_name,
                "source_sample_index": source_index,
                "selection_role": item["selection_role"],
                "k": k_value,
                "ratio_requested": ratio,
                "ratio_actual": ratio_actual,
                "active_count": int(np.sum(mask)),
                "min_target_separation_deg": separation,
                "max_target_theta_deg": scan,
                "large_scan": int(scan >= 45.0),
                "target_hash": digest,
                "label_status": "pending_trusted_eep_hfss",
            }
        )

    if n != 96:
        raise AssertionError(f"Expected 96 candidates, got {n}")
    if len(scene_index_by_key) != 76:
        raise AssertionError(f"Expected 76 independent scenes, got {len(scene_index_by_key)}")

    np.savez_compressed(
        args.out_dir / "dataset_arrays.npz",
        candidate_index=candidate_indices,
        candidate_indices=candidate_indices,
        sample_index=sample_indices,
        sample_indices=sample_indices,
        sample_ids=sample_ids,
        scene_ids=scene_ids,
        source_dataset=source_dataset,
        source_sample_indices=source_indices,
        selection_roles=selection_roles,
        k_values=k_values,
        active_ratios_requested=ratios_requested,
        active_ratios_actual=ratios_actual,
        num_active=num_active,
        targets_deg=targets,
        task_valid=task_valid,
        mask=masks,
        masks=masks,
        w_tasks_real_imag=task_weights_ri,
        task_weights_real_imag=task_weights_ri,
        w_combined_real_imag=combined_weights_ri,
        combined_weights_real_imag=combined_weights_ri,
        hfss_weights_real_imag=combined_weights_ri,
        min_target_separation_deg=min_separation,
        max_target_theta_deg=max_scan,
        large_scan=large_scan_values,
        port_names=np.asarray(base["port_names"]),
        element_ixiy=np.asarray(base["element_ixiy"]),
        positions_lambda=np.asarray(base["positions_lambda"]),
        selected_indices=candidate_indices,
    )
    write_csv(args.out_dir / "candidate_manifest.csv", manifest)

    by_k_ratio = Counter(
        (int(row["k"]), round(float(row["ratio_requested"]), 1)) for row in manifest
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": n,
        "independent_scene_count": len(scene_index_by_key),
        "sample_index_semantics": "independent target-direction scene; repeated across paired ratio candidates",
        "candidate_index_semantics": "unique candidate row",
        "base_candidate_count": sum(row["source_dataset"] == "task_level_run02" for row in manifest),
        "k6_close_paired_candidate_count": sum(row["source_dataset"] == "paired_close_k6" for row in manifest),
        "large_scan_candidate_count": sum(int(row["large_scan"]) for row in manifest),
        "by_k": dict(Counter(str(row["k"]) for row in manifest)),
        "by_ratio": dict(Counter(f"{float(row['ratio_requested']):.1f}" for row in manifest)),
        "by_k_ratio": {f"k{k}_r{ratio:.1f}": count for (k, ratio), count in sorted(by_k_ratio.items())},
        "old_fullwave_labels_included": False,
        "next_gate": "Run no-scale matched-operator EEP/HFSS reconstruction on the trusted field-enabled baseline.",
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
