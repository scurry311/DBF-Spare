"""Generate a guarded fixed-active-count teacher for the HFSS dataset.

Policy:
* optimize only selected K/active_ratio cells;
* keep the active element count unchanged;
* try greedy swaps from the original mask first;
* optionally try v2 as an enhancer;
* accept a candidate only after fine-grid validation;
* keep original labels for rejected and unselected samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from experiment_fixed_count_swap_psll import (
    analytic_task_weights,
    compute_isolation_db,
    evaluate_weights,
    make_method_args,
    parse_float_list,
    parse_int_list,
    safe_mean,
    safe_percentile,
    write_csv,
)
from generate_optimized_teacher_labels import (
    EPS,
    KMAX,
    deterministic_element_order,
    make_grid,
    optimize_sample,
    target_dirs_for_sample,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--teacher-name", default="guarded_greedy_v2_k46_ar05_09")
    parser.add_argument("--k-values", default="4,6")
    parser.add_argument("--active-ratios", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--limit-selected", type=int, default=0, help="Smoke-test cap on selected samples; 0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-v2", action="store_true")
    parser.add_argument("--write-hdf5", action="store_true", default=True)
    parser.add_argument("--no-write-hdf5", dest="write_hdf5", action="store_false")

    parser.add_argument("--theta-step", type=float, default=5.0)
    parser.add_argument("--phi-step", type=float, default=10.0)
    parser.add_argument("--eval-theta-step", type=float, default=2.0)
    parser.add_argument("--eval-phi-step", type=float, default=5.0)
    parser.add_argument("--max-swaps", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=16)
    parser.add_argument("--random-pairs", type=int, default=64)
    parser.add_argument("--min-improve-db", type=float, default=0.02)

    parser.add_argument("--v2-starts", type=int, default=8)
    parser.add_argument("--v2-greedy-swaps", type=int, default=6)
    parser.add_argument("--v2-final-candidates", type=int, default=8)
    parser.add_argument("--nsga-population", type=int, default=18)
    parser.add_argument("--nsga-generations", type=int, default=4)
    parser.add_argument("--nsga-offspring", type=int, default=18)
    parser.add_argument("--softmask-steps", type=int, default=28)
    parser.add_argument("--softmask-lr", type=float, default=0.75)

    parser.add_argument("--spread-margin-db", type=float, default=5.0)
    parser.add_argument("--spread-weight", type=float, default=0.20)
    parser.add_argument("--weak-peak-weight", type=float, default=0.03)

    parser.add_argument("--guard-min-psll-improve-db", type=float, default=0.02)
    parser.add_argument("--guard-max-isolation-drop-db", type=float, default=1.0)
    parser.add_argument("--guard-max-weak-peak-drop-db", type=float, default=0.5)
    return parser.parse_args()


def selected_indices(arrays: dict[str, np.ndarray], k_values: list[int], active_ratios: list[float]) -> np.ndarray:
    mask = np.isin(arrays["k_values"], np.asarray(k_values, dtype=np.int64))
    ratio_mask = np.zeros_like(mask, dtype=np.bool_)
    for active_ratio in active_ratios:
        ratio_mask |= np.isclose(arrays["active_ratios_requested"], float(active_ratio), atol=1.0e-6)
    return np.flatnonzero(mask & ratio_mask).astype(np.int64)


def load_original_weights(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return (
        arrays["task_weights_real_imag"][..., 0].astype(np.float32)
        + 1j * arrays["task_weights_real_imag"][..., 1].astype(np.float32)
    ).astype(np.complex64)


def candidate_passes_guard(
    *,
    original_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    active_count_delta: int,
    args: argparse.Namespace,
) -> bool:
    if active_count_delta != 0:
        return False
    delta_psll = candidate_metrics["psll_to_weakest_peak_db"] - original_metrics["psll_to_weakest_peak_db"]
    delta_iso = candidate_metrics["isolation_min_db"] - original_metrics["isolation_min_db"]
    delta_weak = candidate_metrics["weak_peak_db"] - original_metrics["weak_peak_db"]
    return (
        delta_psll <= -float(args.guard_min_psll_improve_db)
        and delta_iso >= -float(args.guard_max_isolation_drop_db)
        and delta_weak >= -float(args.guard_max_weak_peak_drop_db)
    )


def build_assignment(masks: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    assignment = np.zeros((masks.shape[0], masks.shape[1], KMAX), dtype=np.float64)
    for i, k in enumerate(k_values.astype(np.int32)):
        if k > 0:
            assignment[i, :, : int(k)] = masks[i, :, None].astype(np.float64) / float(k)
    return assignment


def write_hdf5_teacher(
    *,
    dataset_path: Path,
    teacher_name: str,
    masks: np.ndarray,
    weights: np.ndarray,
    k_values: np.ndarray,
    rows_by_index: list[dict[str, Any]],
    overwrite: bool,
) -> str:
    num_samples, num_elements = masks.shape
    weights_ri = np.zeros((num_samples, num_elements, KMAX, 2), dtype=np.float64)
    weights_ri[..., 0] = weights.real
    weights_ri[..., 1] = weights.imag
    assignment = build_assignment(masks, k_values)
    status = np.asarray([str(row["status"]) for row in rows_by_index], dtype=object)
    objective = np.asarray([float(row["chosen_psll_to_weakest_peak_db"]) for row in rows_by_index], dtype=np.float64)
    iterations = np.asarray([int(row["chosen_swaps"]) for row in rows_by_index], dtype=np.int32)
    diagnostics = np.asarray(
        [json.dumps(row, ensure_ascii=False, allow_nan=True) for row in rows_by_index],
        dtype=object,
    )
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(dataset_path, "a") as handle:
        labels = handle.require_group("labels")
        if teacher_name in labels:
            if not overwrite:
                raise RuntimeError(f"labels/{teacher_name} exists; pass --overwrite to replace it.")
            del labels[teacher_name]
        group = labels.create_group(teacher_name)
        group.create_dataset("weights_real_imag", data=weights_ri)
        group.create_dataset("activation", data=masks.astype(np.float64))
        group.create_dataset("assignment", data=assignment)
        group.create_dataset("status", data=status, dtype=string_dtype)
        group.create_dataset("solve_time_ms", data=np.zeros(num_samples, dtype=np.float64))
        group.create_dataset("e2e_time_ms", data=np.zeros(num_samples, dtype=np.float64))
        group.create_dataset("objective", data=objective)
        group.create_dataset("iterations", data=iterations)
        group.create_dataset("task_count", data=k_values.astype(np.int32))
        group.create_dataset("scenario_index", data=np.arange(num_samples, dtype=np.int32))
        group.create_dataset("diagnostics_json", data=diagnostics, dtype=string_dtype)
    return f"{dataset_path}:/labels/{teacher_name}"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: list[tuple[str, str, str]] = [("all", "all", "all")]
    for k in sorted({str(row["k"]) for row in rows}):
        keys.append(("all", k, "all"))
        for active_ratio in sorted({str(row["active_ratio"]) for row in rows if str(row["k"]) == k}):
            keys.append(("all", k, active_ratio))
    for status in sorted({str(row["status"]) for row in rows}):
        keys.append((status, "all", "all"))

    out: list[dict[str, Any]] = []
    for status_label, k_label, ratio_label in keys:
        group = rows
        if status_label != "all":
            group = [row for row in group if str(row["status"]) == status_label]
        if k_label != "all":
            group = [row for row in group if str(row["k"]) == k_label]
        if ratio_label != "all":
            group = [row for row in group if str(row["active_ratio"]) == ratio_label]
        if not group:
            continue
        out.append(
            {
                "status": status_label,
                "k": k_label,
                "active_ratio": ratio_label,
                "n": len(group),
                "accepted_rate": safe_mean([1.0 if bool(row["accepted"]) else 0.0 for row in group]),
                "greedy_accept_rate": safe_mean([1.0 if bool(row["greedy_accepted"]) else 0.0 for row in group]),
                "v2_accept_rate": safe_mean([1.0 if bool(row["v2_accepted"]) else 0.0 for row in group]),
                "chosen_greedy_rate": safe_mean([1.0 if row["chosen_method"] == "greedy_original" else 0.0 for row in group]),
                "chosen_v2_rate": safe_mean([1.0 if row["chosen_method"] == "v2" else 0.0 for row in group]),
                "original_psll_mean_db": safe_mean([float(row["original_psll_to_weakest_peak_db"]) for row in group]),
                "chosen_psll_mean_db": safe_mean([float(row["chosen_psll_to_weakest_peak_db"]) for row in group]),
                "delta_psll_mean_db": safe_mean([float(row["delta_psll_to_weakest_peak_db"]) for row in group]),
                "original_psll_p95_db": safe_percentile([float(row["original_psll_to_weakest_peak_db"]) for row in group], 95),
                "chosen_psll_p95_db": safe_percentile([float(row["chosen_psll_to_weakest_peak_db"]) for row in group], 95),
                "delta_weak_peak_mean_db": safe_mean([float(row["delta_weak_peak_db"]) for row in group]),
                "delta_isolation_mean_db": safe_mean([float(row["delta_isolation_min_db"]) for row in group]),
                "delta_energy_mean": safe_mean([float(row["delta_energy_proxy"]) for row in group]),
                "max_abs_active_count_delta": max(abs(int(row["active_count_delta"])) for row in group),
                "mean_swaps": safe_mean([float(row["chosen_swaps"]) for row in group]),
            }
        )
    return out


def base_row_for_original(
    *,
    sample_index: int,
    arrays: dict[str, np.ndarray],
    original_metrics: dict[str, float],
    status: str,
) -> dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "sample_id": str(arrays["sample_ids"][sample_index]),
        "k": int(arrays["k_values"][sample_index]),
        "active_ratio": f"{float(arrays['active_ratios_requested'][sample_index]):.1f}",
        "num_active": int(arrays["num_active"][sample_index]),
        "status": status,
        "accepted": False,
        "greedy_accepted": False,
        "v2_accepted": False,
        "chosen_method": "original",
        "chosen_swaps": 0,
        "active_count_delta": 0,
        "original_psll_to_weakest_peak_db": float(original_metrics["psll_to_weakest_peak_db"]),
        "chosen_psll_to_weakest_peak_db": float(original_metrics["psll_to_weakest_peak_db"]),
        "delta_psll_to_weakest_peak_db": 0.0,
        "original_weak_peak_db": float(original_metrics["weak_peak_db"]),
        "chosen_weak_peak_db": float(original_metrics["weak_peak_db"]),
        "delta_weak_peak_db": 0.0,
        "original_isolation_min_db": float(original_metrics["isolation_min_db"]),
        "chosen_isolation_min_db": float(original_metrics["isolation_min_db"]),
        "delta_isolation_min_db": 0.0,
        "original_energy_proxy": float(original_metrics["energy_proxy"]),
        "chosen_energy_proxy": float(original_metrics["energy_proxy"]),
        "delta_energy_proxy": 0.0,
        "greedy_delta_psll_db": float("nan"),
        "v2_delta_psll_db": float("nan"),
        "candidate_notes": "",
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    teacher_name = str(args.teacher_name)
    out_dir = dataset_dir / "optimized_teachers" / teacher_name
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays_npz = np.load(dataset_dir / "dataset_arrays.npz", allow_pickle=False)
    arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    num_samples = int(arrays["k_values"].shape[0])
    num_elements = int(arrays["masks"].shape[1])
    k_values = parse_int_list(args.k_values)
    active_ratios = parse_float_list(args.active_ratios)
    target_indices = selected_indices(arrays, k_values, active_ratios)
    if int(args.limit_selected) > 0:
        target_indices = target_indices[: int(args.limit_selected)]
    target_set = {int(index) for index in target_indices.tolist()}

    original_masks = arrays["masks"].astype(np.int8).copy()
    final_masks = original_masks.copy()
    original_weights = load_original_weights(arrays)
    final_weights = original_weights.copy()
    positions = arrays["positions_lambda"].astype(np.float32)

    _theta, _phi, opt_grid_dirs = make_grid(args.theta_step, args.phi_step)
    _eval_theta, _eval_phi, eval_grid_dirs = make_grid(args.eval_theta_step, args.eval_phi_step)
    grid_phase = 2.0 * np.pi * (eval_grid_dirs @ positions.T)
    eval_grid_steer = np.exp(-1j * grid_phase).astype(np.complex64)
    order = deterministic_element_order(arrays["element_ixiy"])
    greedy_args = make_method_args(args, "greedy_original")
    v2_args = make_method_args(args, "v2")
    rng = np.random.default_rng(int(args.seed))

    rows_selected: list[dict[str, Any]] = []
    rows_by_index: list[dict[str, Any] | None] = [None] * num_samples
    start = time.time()
    last = start

    for sample_index in range(num_samples):
        original_mask = original_masks[sample_index].astype(bool)
        num_active = int(arrays["num_active"][sample_index])
        original_eval_weights = analytic_task_weights(
            mask=original_mask,
            targets_deg=arrays["targets_deg"][sample_index],
            task_valid=arrays["task_valid"][sample_index],
            positions=positions,
            num_active=num_active,
        )
        original_metrics = evaluate_weights(
            weights=original_eval_weights,
            targets_deg=arrays["targets_deg"][sample_index],
            task_valid=arrays["task_valid"][sample_index],
            positions=positions,
            grid_dirs=eval_grid_dirs,
            grid_steer=eval_grid_steer,
        )
        if sample_index not in target_set:
            rows_by_index[sample_index] = base_row_for_original(
                sample_index=sample_index,
                arrays=arrays,
                original_metrics=original_metrics,
                status="original_unselected",
            )
            continue

        row = base_row_for_original(
            sample_index=sample_index,
            arrays=arrays,
            original_metrics=original_metrics,
            status="original_rejected",
        )
        best_mask = original_mask
        best_weights = original_weights[sample_index]
        best_metrics = original_metrics

        greedy_result = optimize_sample(
            sample_index=sample_index,
            arrays=arrays,
            initial_order=order,
            grid_dirs=opt_grid_dirs,
            rng=rng,
            args=greedy_args,
        )
        greedy_mask = greedy_result["mask"].astype(bool)
        greedy_metrics = evaluate_weights(
            weights=greedy_result["weights"],
            targets_deg=arrays["targets_deg"][sample_index],
            task_valid=arrays["task_valid"][sample_index],
            positions=positions,
            grid_dirs=eval_grid_dirs,
            grid_steer=eval_grid_steer,
        )
        greedy_active_delta = int(greedy_mask.sum()) - int(original_mask.sum())
        greedy_accepted = candidate_passes_guard(
            original_metrics=original_metrics,
            candidate_metrics=greedy_metrics,
            active_count_delta=greedy_active_delta,
            args=args,
        )
        row["greedy_delta_psll_db"] = (
            greedy_metrics["psll_to_weakest_peak_db"] - original_metrics["psll_to_weakest_peak_db"]
        )
        row["greedy_accepted"] = bool(greedy_accepted)
        row["greedy_swaps"] = int(greedy_result["swaps"])
        if greedy_accepted:
            best_mask = greedy_mask
            best_weights = greedy_result["weights"]
            best_metrics = greedy_metrics
            row["chosen_method"] = "greedy_original"
            row["chosen_swaps"] = int(greedy_result["swaps"])

        v2_accepted = False
        if not bool(args.disable_v2):
            v2_result = optimize_sample(
                sample_index=sample_index,
                arrays=arrays,
                initial_order=order,
                grid_dirs=opt_grid_dirs,
                rng=rng,
                args=v2_args,
            )
            v2_mask = v2_result["mask"].astype(bool)
            v2_metrics = evaluate_weights(
                weights=v2_result["weights"],
                targets_deg=arrays["targets_deg"][sample_index],
                task_valid=arrays["task_valid"][sample_index],
                positions=positions,
                grid_dirs=eval_grid_dirs,
                grid_steer=eval_grid_steer,
            )
            v2_active_delta = int(v2_mask.sum()) - int(original_mask.sum())
            v2_accepted = candidate_passes_guard(
                original_metrics=original_metrics,
                candidate_metrics=v2_metrics,
                active_count_delta=v2_active_delta,
                args=args,
            )
            row["v2_delta_psll_db"] = v2_metrics["psll_to_weakest_peak_db"] - original_metrics["psll_to_weakest_peak_db"]
            row["v2_accepted"] = bool(v2_accepted)
            row["v2_swaps"] = int(v2_result["swaps"])
            row["v2_seed_count"] = int(v2_result.get("v2_seed_count", 0))
            row["v2_candidate_count"] = int(v2_result.get("v2_candidate_count", 0))
            if v2_accepted and (
                v2_metrics["psll_to_weakest_peak_db"] < best_metrics["psll_to_weakest_peak_db"]
            ):
                best_mask = v2_mask
                best_weights = v2_result["weights"]
                best_metrics = v2_metrics
                row["chosen_method"] = "v2"
                row["chosen_swaps"] = int(v2_result["swaps"])

        accepted = row["chosen_method"] != "original"
        row["accepted"] = bool(accepted)
        row["status"] = f"{row['chosen_method']}_accepted" if accepted else "original_rejected"
        row["active_count_delta"] = int(best_mask.sum()) - int(original_mask.sum())
        row["chosen_psll_to_weakest_peak_db"] = float(best_metrics["psll_to_weakest_peak_db"])
        row["delta_psll_to_weakest_peak_db"] = (
            float(best_metrics["psll_to_weakest_peak_db"]) - float(original_metrics["psll_to_weakest_peak_db"])
        )
        row["chosen_weak_peak_db"] = float(best_metrics["weak_peak_db"])
        row["delta_weak_peak_db"] = float(best_metrics["weak_peak_db"]) - float(original_metrics["weak_peak_db"])
        row["chosen_isolation_min_db"] = float(best_metrics["isolation_min_db"])
        row["delta_isolation_min_db"] = float(best_metrics["isolation_min_db"]) - float(original_metrics["isolation_min_db"])
        row["chosen_energy_proxy"] = float(best_metrics["energy_proxy"])
        row["delta_energy_proxy"] = float(best_metrics["energy_proxy"]) - float(original_metrics["energy_proxy"])

        final_masks[sample_index] = best_mask.astype(np.int8)
        final_weights[sample_index] = best_weights.astype(np.complex64)
        rows_selected.append(row)
        rows_by_index[sample_index] = row

        now = time.time()
        done = len(rows_selected)
        if done == len(target_indices) or now - last >= 30.0:
            accepted_count = sum(1 for item in rows_selected if bool(item["accepted"]))
            print(
                f"processed selected {done}/{len(target_indices)}; "
                f"accepted {accepted_count}; elapsed {now - start:.1f}s"
            )
            last = now

    for sample_index, row in enumerate(rows_by_index):
        if row is None:
            raise RuntimeError(f"Missing diagnostics for sample {sample_index}")

    compatible = dict(arrays)
    compatible["masks"] = final_masks
    compatible["task_weights_real_imag"] = np.stack([final_weights.real, final_weights.imag], axis=-1)
    np.savez_compressed(out_dir / "dataset_arrays.npz", **compatible)
    np.savez_compressed(
        out_dir / "guarded_teacher_arrays.npz",
        masks=final_masks,
        task_weights_real_imag=np.stack([final_weights.real, final_weights.imag], axis=-1),
        sample_ids=arrays["sample_ids"],
        k_values=arrays["k_values"],
        active_ratios_requested=arrays["active_ratios_requested"],
        active_ratios_actual=arrays["active_ratios_actual"],
        num_active=arrays["num_active"],
        targets_deg=arrays["targets_deg"],
        task_valid=arrays["task_valid"],
        positions_lambda=arrays["positions_lambda"],
        element_ixiy=arrays["element_ixiy"],
    )
    shutil.copy2(dataset_dir / "training_split_manifest.json", out_dir / "training_split_manifest.json")
    write_csv(out_dir / "guarded_teacher_selected_metrics.csv", rows_selected)
    summary_rows = summarize(rows_selected)
    write_csv(out_dir / "guarded_teacher_summary_by_k_active.csv", summary_rows)

    hdf5_label = None
    if bool(args.write_hdf5):
        hdf5_label = write_hdf5_teacher(
            dataset_path=dataset_dir / "training_dataset.h5",
            teacher_name=teacher_name,
            masks=final_masks,
            weights=final_weights,
            k_values=arrays["k_values"],
            rows_by_index=[row for row in rows_by_index if row is not None],
            overwrite=bool(args.overwrite),
        )

    elapsed = time.time() - start
    accepted_rows = [row for row in rows_selected if bool(row["accepted"])]
    run_summary = {
        "teacher_name": teacher_name,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "selected_samples": int(len(target_indices)),
        "accepted_samples": int(len(accepted_rows)),
        "accepted_rate": float(len(accepted_rows) / max(len(target_indices), 1)),
        "greedy_chosen": int(sum(1 for row in accepted_rows if row["chosen_method"] == "greedy_original")),
        "v2_chosen": int(sum(1 for row in accepted_rows if row["chosen_method"] == "v2")),
        "elapsed_s": elapsed,
        "guard": {
            "min_psll_improve_db": float(args.guard_min_psll_improve_db),
            "max_isolation_drop_db": float(args.guard_max_isolation_drop_db),
            "max_weak_peak_drop_db": float(args.guard_max_weak_peak_drop_db),
        },
        "headline": {
            "delta_psll_mean_db": safe_mean([float(row["delta_psll_to_weakest_peak_db"]) for row in rows_selected]),
            "delta_weak_peak_mean_db": safe_mean([float(row["delta_weak_peak_db"]) for row in rows_selected]),
            "delta_isolation_mean_db": safe_mean([float(row["delta_isolation_min_db"]) for row in rows_selected]),
            "max_abs_active_count_delta": max(abs(int(row["active_count_delta"])) for row in rows_selected)
            if rows_selected
            else 0,
            "delta_energy_mean": safe_mean([float(row["delta_energy_proxy"]) for row in rows_selected]),
        },
        "outputs": {
            "dataset_arrays": str(out_dir / "dataset_arrays.npz"),
            "guarded_teacher_arrays": str(out_dir / "guarded_teacher_arrays.npz"),
            "selected_metrics": str(out_dir / "guarded_teacher_selected_metrics.csv"),
            "summary": str(out_dir / "guarded_teacher_summary_by_k_active.csv"),
            "hdf5_label": hdf5_label,
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
