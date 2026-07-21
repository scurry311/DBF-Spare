"""Fixed-active-count mask/weight refinement experiment.

This script is intentionally non-destructive: it reads dataset_arrays.npz,
optimizes selected samples with the same active element count, recomputes
analytic per-task steering weights, and writes CSV/JSON summaries.
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

import numpy as np

from generate_optimized_teacher_labels import (
    EPS,
    KMAX,
    deterministic_element_order,
    make_grid,
    objective_from_fields,
    optimize_sample,
    side_mask_for_targets,
    target_dirs_for_sample,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--run-name", default="fixed_count_swap_experiment")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--k-values", default="4,6")
    parser.add_argument("--active-ratios", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--samples-per-cell", type=int, default=5)
    parser.add_argument("--methods", default="greedy_original,v2")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--overwrite", action="store_true")

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
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_method_list(text: str) -> list[str]:
    methods = [part.strip() for part in text.split(",") if part.strip()]
    valid = {"greedy_original", "v2"}
    unknown = [method for method in methods if method not in valid]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; valid={sorted(valid)}")
    return methods


def load_splits(dataset_dir: Path) -> dict[str, np.ndarray]:
    with (dataset_dir / "training_split_manifest.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload["splits"].items()}


def select_indices(
    arrays: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    *,
    split: str,
    k_values: list[int],
    active_ratios: list[float],
    samples_per_cell: int,
) -> np.ndarray:
    base = np.asarray(splits[split], dtype=np.int64)
    selected: list[int] = []
    for k in k_values:
        for active_ratio in active_ratios:
            cell = base[
                (arrays["k_values"][base] == int(k))
                & np.isclose(arrays["active_ratios_requested"][base], float(active_ratio), atol=1.0e-6)
            ]
            cell = np.sort(cell)
            if samples_per_cell > 0:
                cell = cell[:samples_per_cell]
            selected.extend(int(index) for index in cell)
    return np.asarray(selected, dtype=np.int64)


def make_method_args(base: argparse.Namespace, method: str) -> argparse.Namespace:
    return argparse.Namespace(
        strategy="v2" if method == "v2" else "v1",
        init_mode="original" if method == "greedy_original" else "deterministic",
        max_swaps=int(base.max_swaps),
        top_candidates=int(base.top_candidates),
        random_pairs=int(base.random_pairs),
        theta_step=float(base.theta_step),
        phi_step=float(base.phi_step),
        eval_theta_step=float(base.eval_theta_step),
        eval_phi_step=float(base.eval_phi_step),
        spread_margin_db=float(base.spread_margin_db),
        spread_weight=float(base.spread_weight),
        weak_peak_weight=float(base.weak_peak_weight),
        min_improve_db=float(base.min_improve_db),
        v2_starts=int(base.v2_starts),
        v2_greedy_swaps=int(base.v2_greedy_swaps),
        v2_final_candidates=int(base.v2_final_candidates),
        nsga_population=int(base.nsga_population),
        nsga_generations=int(base.nsga_generations),
        nsga_offspring=int(base.nsga_offspring),
        softmask_steps=int(base.softmask_steps),
        softmask_lr=float(base.softmask_lr),
        softmask_min_k=4,
        softmask_temperature_db=1.5,
        seed=int(base.seed),
    )


def analytic_task_weights(
    *,
    mask: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    positions: np.ndarray,
    num_active: int,
) -> np.ndarray:
    weights = np.zeros((positions.shape[0], KMAX), dtype=np.complex64)
    valid = task_valid.astype(bool)
    target_dirs = target_dirs_for_sample(targets_deg, valid)
    target_phase = 2.0 * np.pi * (target_dirs @ positions.T)
    target_tx = np.exp(1j * target_phase).astype(np.complex64).T
    valid_indices = np.flatnonzero(valid)
    weights[:, valid_indices] = target_tx * mask[:, None].astype(np.complex64) / float(num_active)
    return weights


def evaluate_weights(
    *,
    weights: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    positions: np.ndarray,
    grid_dirs: np.ndarray,
    grid_steer: np.ndarray,
) -> dict[str, float]:
    valid = task_valid.astype(bool)
    target_dirs = target_dirs_for_sample(targets_deg, valid)
    side_mask = side_mask_for_targets(grid_dirs, target_dirs)
    combined = weights.sum(axis=1)
    af_grid = grid_steer @ combined
    target_phase = 2.0 * np.pi * (target_dirs @ positions.T)
    target_steer = np.exp(-1j * target_phase).astype(np.complex64)
    target_resp = target_steer @ combined
    _objective, metrics = objective_from_fields(
        af_grid,
        target_resp,
        side_mask,
        spread_margin_db=5.0,
        spread_weight=0.0,
        weak_peak_weight=0.0,
    )
    isolation = compute_isolation_db(weights=weights, target_steer=target_steer, valid=valid)
    metrics["isolation_min_db"] = isolation
    metrics["energy_proxy"] = float(np.sum(np.abs(weights) ** 2))
    return metrics


def compute_isolation_db(*, weights: np.ndarray, target_steer: np.ndarray, valid: np.ndarray) -> float:
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size <= 1:
        return float("nan")
    response = target_steer @ weights[:, valid_indices]
    mag = np.maximum(np.abs(response), math.sqrt(EPS))
    diag = np.diag(mag)
    isolation_values: list[float] = []
    for task_index in range(valid_indices.size):
        off = np.delete(mag[:, task_index], task_index)
        isolation_values.append(20.0 * math.log10(float(diag[task_index]) / max(float(off.max()), math.sqrt(EPS))))
    return float(min(isolation_values))


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


def finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def safe_mean(values: list[float]) -> float:
    arr = finite(values)
    return float(arr.mean()) if arr.size else float("nan")


def safe_percentile(values: list[float], q: float) -> float:
    arr = finite(values)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: list[tuple[str, str, str]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        keys.append((method, "all", "all"))
        for k in sorted({int(row["k"]) for row in rows if row["method"] == method}):
            keys.append((method, str(k), "all"))
            ratios = sorted({str(row["active_ratio"]) for row in rows if row["method"] == method and int(row["k"]) == k})
            for ratio in ratios:
                keys.append((method, str(k), ratio))

    out: list[dict[str, Any]] = []
    for method, k_label, ratio_label in keys:
        group = [row for row in rows if row["method"] == method]
        if k_label != "all":
            group = [row for row in group if int(row["k"]) == int(k_label)]
        if ratio_label != "all":
            group = [row for row in group if str(row["active_ratio"]) == ratio_label]
        if not group:
            continue
        delta_psll = [float(row["delta_psll_to_weakest_peak_db"]) for row in group]
        delta_iso = [float(row["delta_isolation_min_db"]) for row in group]
        row = {
            "method": method,
            "k": k_label,
            "active_ratio": ratio_label,
            "n": len(group),
            "original_psll_mean_db": safe_mean([float(r["original_psll_to_weakest_peak_db"]) for r in group]),
            "optimized_psll_mean_db": safe_mean([float(r["optimized_psll_to_weakest_peak_db"]) for r in group]),
            "delta_psll_mean_db": safe_mean(delta_psll),
            "original_psll_p95_db": safe_percentile([float(r["original_psll_to_weakest_peak_db"]) for r in group], 95),
            "optimized_psll_p95_db": safe_percentile([float(r["optimized_psll_to_weakest_peak_db"]) for r in group], 95),
            "improved_rate": safe_mean([1.0 if float(r["delta_psll_to_weakest_peak_db"]) < -1.0e-9 else 0.0 for r in group]),
            "original_weak_peak_mean_db": safe_mean([float(r["original_weak_peak_db"]) for r in group]),
            "optimized_weak_peak_mean_db": safe_mean([float(r["optimized_weak_peak_db"]) for r in group]),
            "delta_weak_peak_mean_db": safe_mean([float(r["delta_weak_peak_db"]) for r in group]),
            "original_isolation_p05_db": safe_percentile([float(r["original_isolation_min_db"]) for r in group], 5),
            "optimized_isolation_p05_db": safe_percentile([float(r["optimized_isolation_min_db"]) for r in group], 5),
            "delta_isolation_mean_db": safe_mean(delta_iso),
            "delta_energy_mean": safe_mean([float(r["delta_energy_proxy"]) for r in group]),
            "max_abs_active_count_delta": max(abs(int(r["active_count_delta"])) for r in group),
            "mean_swaps": safe_mean([float(r["swaps"]) for r in group]),
        }
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    out_dir = dataset_dir / "optimized_teachers" / args.run_name
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays_npz = np.load(dataset_dir / "dataset_arrays.npz", allow_pickle=False)
    arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    splits = load_splits(dataset_dir)
    k_values = parse_int_list(args.k_values)
    active_ratios = parse_float_list(args.active_ratios)
    methods = parse_method_list(args.methods)
    selected = select_indices(
        arrays,
        splits,
        split=args.split,
        k_values=k_values,
        active_ratios=active_ratios,
        samples_per_cell=int(args.samples_per_cell),
    )
    if selected.size == 0:
        raise RuntimeError("No samples selected.")

    _theta, _phi, opt_grid_dirs = make_grid(args.theta_step, args.phi_step)
    _eval_theta, _eval_phi, eval_grid_dirs = make_grid(args.eval_theta_step, args.eval_phi_step)
    positions = arrays["positions_lambda"].astype(np.float32)
    grid_phase = 2.0 * np.pi * (eval_grid_dirs @ positions.T)
    eval_grid_steer = np.exp(-1j * grid_phase).astype(np.complex64)
    order = deterministic_element_order(arrays["element_ixiy"])

    rows: list[dict[str, Any]] = []
    start = time.time()
    total_jobs = int(selected.size * len(methods))
    done = 0
    for method in methods:
        method_args = make_method_args(args, method)
        rng = np.random.default_rng(int(args.seed))
        for sample_index in selected:
            sample_index = int(sample_index)
            original_mask = arrays["masks"][sample_index].astype(bool)
            num_active = int(arrays["num_active"][sample_index])
            original_weights = analytic_task_weights(
                mask=original_mask,
                targets_deg=arrays["targets_deg"][sample_index],
                task_valid=arrays["task_valid"][sample_index],
                positions=positions,
                num_active=num_active,
            )
            original_eval = evaluate_weights(
                weights=original_weights,
                targets_deg=arrays["targets_deg"][sample_index],
                task_valid=arrays["task_valid"][sample_index],
                positions=positions,
                grid_dirs=eval_grid_dirs,
                grid_steer=eval_grid_steer,
            )

            result = optimize_sample(
                sample_index=sample_index,
                arrays=arrays,
                initial_order=order,
                grid_dirs=opt_grid_dirs,
                rng=rng,
                args=method_args,
            )
            optimized_mask = result["mask"].astype(bool)
            optimized_weights = result["weights"]
            optimized_eval = evaluate_weights(
                weights=optimized_weights,
                targets_deg=arrays["targets_deg"][sample_index],
                task_valid=arrays["task_valid"][sample_index],
                positions=positions,
                grid_dirs=eval_grid_dirs,
                grid_steer=eval_grid_steer,
            )

            active_count_delta = int(optimized_mask.sum()) - int(original_mask.sum())
            rows.append(
                {
                    "method": method,
                    "sample_index": sample_index,
                    "sample_id": str(arrays["sample_ids"][sample_index]),
                    "split": args.split,
                    "k": int(arrays["k_values"][sample_index]),
                    "active_ratio": f"{float(arrays['active_ratios_requested'][sample_index]):.1f}",
                    "num_active": num_active,
                    "optimized_num_active": int(optimized_mask.sum()),
                    "active_count_delta": active_count_delta,
                    "swaps": int(result["swaps"]),
                    "v2_seed_count": int(result.get("v2_seed_count", 0)),
                    "v2_candidate_count": int(result.get("v2_candidate_count", 0)),
                    "v2_source_rank": int(result.get("v2_source_rank", -1)),
                    "v2_softmask_used": int(result.get("v2_softmask_used", 0)),
                    "original_psll_to_weakest_peak_db": original_eval["psll_to_weakest_peak_db"],
                    "optimized_psll_to_weakest_peak_db": optimized_eval["psll_to_weakest_peak_db"],
                    "delta_psll_to_weakest_peak_db": optimized_eval["psll_to_weakest_peak_db"]
                    - original_eval["psll_to_weakest_peak_db"],
                    "original_worst_sidelobe_db": original_eval["worst_sidelobe_db"],
                    "optimized_worst_sidelobe_db": optimized_eval["worst_sidelobe_db"],
                    "delta_worst_sidelobe_db": optimized_eval["worst_sidelobe_db"]
                    - original_eval["worst_sidelobe_db"],
                    "original_weak_peak_db": original_eval["weak_peak_db"],
                    "optimized_weak_peak_db": optimized_eval["weak_peak_db"],
                    "delta_weak_peak_db": optimized_eval["weak_peak_db"] - original_eval["weak_peak_db"],
                    "original_target_spread_db": original_eval["target_spread_db"],
                    "optimized_target_spread_db": optimized_eval["target_spread_db"],
                    "original_isolation_min_db": original_eval["isolation_min_db"],
                    "optimized_isolation_min_db": optimized_eval["isolation_min_db"],
                    "delta_isolation_min_db": optimized_eval["isolation_min_db"] - original_eval["isolation_min_db"],
                    "original_energy_proxy": original_eval["energy_proxy"],
                    "optimized_energy_proxy": optimized_eval["energy_proxy"],
                    "delta_energy_proxy": optimized_eval["energy_proxy"] - original_eval["energy_proxy"],
                }
            )
            done += 1
            if done == total_jobs or done % 10 == 0:
                print(f"completed {done}/{total_jobs} jobs, elapsed {time.time() - start:.1f}s")

    summary_rows = summarize(rows)
    write_csv(out_dir / "sample_metrics.csv", rows)
    write_csv(out_dir / "summary_by_method_k_active.csv", summary_rows)
    headline = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "split": args.split,
        "selected_samples": int(selected.size),
        "methods": methods,
        "k_values": k_values,
        "active_ratios": active_ratios,
        "samples_per_cell": int(args.samples_per_cell),
        "elapsed_s": time.time() - start,
        "outputs": {
            "sample_metrics": str(out_dir / "sample_metrics.csv"),
            "summary": str(out_dir / "summary_by_method_k_active.csv"),
        },
        "overall": [row for row in summary_rows if row["k"] == "all" and row["active_ratio"] == "all"],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(headline, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
