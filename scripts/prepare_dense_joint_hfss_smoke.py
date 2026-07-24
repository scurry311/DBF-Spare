#!/usr/bin/env python3
"""Package sparse multibeam dense-EEP positives for a gated HFSS smoke."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "dataset_v2_20260724"
    / "residual_critic_dataset_v2.npz"
)
DEFAULT_DENSE = ROOT / "hfss_outputs" / "trusted_dense_local_eep_joint_20260724_run02"
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_dense_joint_hfss_dataset_20260724_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dense-dir", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-selected", type=int, default=0)
    parser.add_argument("--k6-first", action="store_true")
    parser.add_argument("--min-case-count", type=int, default=50)
    parser.add_argument("--max-case-count", type=int, default=100)
    parser.add_argument("--allow-small-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite HFSS shortlist package: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as source:
        base = {key: source[key] for key in source.files}
    with np.load(args.dense_dir / "dense_refined_task_weights.npz", allow_pickle=False) as source:
        dense = {key: source[key] for key in source.files}
    rows = read_csv(args.dense_dir / "dense_refinement_candidate_metrics.csv")
    dense_position = {
        int(candidate): position
        for position, candidate in enumerate(np.asarray(dense["candidate_index"], dtype=int))
    }
    selected = [
        row
        for row in rows
        if int(float(row["strict_engineering_gate"])) == 1
        and int(float(row["k"])) in (2, 4, 6)
        and float(row["ratio"]) < 0.999
    ]
    selected.sort(
        key=lambda row: (
            0 if args.k6_first and int(float(row["k"])) == 6 else 1,
            float(row["ratio"]),
            -int(float(row["k"])),
            int(float(row["candidate_index"])),
        )
    )
    if int(args.max_selected) > 0:
        selected = selected[: int(args.max_selected)]
    source_indices = np.asarray(
        [int(float(row["candidate_index"])) for row in selected], dtype=np.int64
    )
    positions = np.asarray([dense_position[int(index)] for index in source_indices], dtype=int)
    external_tasks = np.asarray(dense["refined_external_task_weights"][positions], dtype=np.complex64)
    external_combined = np.sum(external_tasks, axis=2)
    task_weights = np.conjugate(external_tasks)
    combined_weights = np.conjugate(external_combined)
    local_indices = np.arange(source_indices.size, dtype=np.int64)
    case_count = int(sum(1 + int(base["k_values"][index]) for index in source_indices))
    if not args.allow_small_smoke and not (
        int(args.min_case_count) <= case_count <= int(args.max_case_count)
    ):
        raise RuntimeError(
            f"HFSS shortlist has {case_count} cases, outside "
            f"[{args.min_case_count}, {args.max_case_count}]"
        )

    np.savez_compressed(
        args.out_dir / "dataset_arrays.npz",
        candidate_index=local_indices,
        candidate_indices=local_indices,
        source_dense_candidate_indices=source_indices,
        sample_index=np.asarray(base["sample_index"][source_indices]),
        sample_indices=np.asarray(base["sample_index"][source_indices]),
        sample_ids=np.asarray(
            [f"dense_hfss_c{index:03d}" for index in source_indices]
        ),
        scene_ids=np.asarray(base["scene_ids"][source_indices]),
        source_dataset=np.full(source_indices.size, "dense_local_eep_joint_run02"),
        source_sample_indices=source_indices,
        selection_roles=np.full(source_indices.size, "sparse_multibeam_joint_positive"),
        k_values=np.asarray(base["k_values"][source_indices]),
        active_ratios_requested=np.asarray(base["active_ratios_requested"][source_indices]),
        active_ratios_actual=np.asarray(base["active_ratios_actual"][source_indices]),
        num_active=np.asarray(base["num_active"][source_indices]),
        targets_deg=np.asarray(base["targets_deg"][source_indices]),
        task_valid=np.asarray(base["task_valid"][source_indices]),
        mask=np.asarray(base["mask"][source_indices]),
        masks=np.asarray(base["mask"][source_indices]),
        w_tasks_real_imag=np.stack([task_weights.real, task_weights.imag], axis=-1),
        task_weights_real_imag=np.stack([task_weights.real, task_weights.imag], axis=-1),
        w_combined_real_imag=np.stack([combined_weights.real, combined_weights.imag], axis=-1),
        combined_weights_real_imag=np.stack([combined_weights.real, combined_weights.imag], axis=-1),
        hfss_weights_real_imag=np.stack([combined_weights.real, combined_weights.imag], axis=-1),
        min_target_separation_deg=np.asarray(base["min_target_separation_deg"][source_indices]),
        max_target_theta_deg=np.asarray(base["max_target_theta_deg"][source_indices]),
        large_scan=np.asarray(base["large_scan"][source_indices]),
        port_names=np.asarray(base["port_names"]),
        element_ixiy=np.asarray(base["element_ixiy"]),
        positions_lambda=np.asarray(base["positions_lambda"]),
    )
    manifest = []
    for local_index, (source_index, metric) in enumerate(zip(source_indices, selected)):
        manifest.append(
            {
                "candidate_index": local_index,
                "source_dense_candidate_index": int(source_index),
                "sample_index": int(base["sample_index"][source_index]),
                "k": int(base["k_values"][source_index]),
                "ratio": float(base["active_ratios_requested"][source_index]),
                "large_scan": int(base["large_scan"][source_index]),
                "min_target_separation_deg": float(base["min_target_separation_deg"][source_index]),
                "eep_psll_db": float(metric["refined_psll_db"]),
                "eep_nearest_iso_db": float(metric["refined_nearest_iso_db"]),
                "eep_local_iso_db": float(metric["refined_local_iso_db"]),
                "eep_combined_active_rl_db": float(metric["refined_combined_active_rl_db"]),
                "eep_strict_engineering_gate": int(float(metric["strict_engineering_gate"])),
                "expected_hfss_case_count": 1 + int(base["k_values"][source_index]),
            }
        )
    write_csv(args.out_dir / "candidate_manifest.csv", manifest)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(selected),
        "expected_hfss_case_count": case_count,
        "source_dense_candidate_indices": source_indices.tolist(),
        "k_counts": {
            str(k): int(sum(int(float(row["k"])) == k for row in selected))
            for k in (2, 4, 6)
        },
        "ratio1_included": False,
        "all_selected_eep_strict_engineering_positive": True,
        "small_mapping_smoke": bool(args.allow_small_smoke),
        "full_50_100_case_gate_pass": bool(
            int(args.min_case_count) <= case_count <= int(args.max_case_count)
        ),
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
