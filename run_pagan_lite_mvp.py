"""PAGAN-lite MVP evaluation for the URA16 HFSS dataset.

This script is deliberately non-destructive. It compares:

1. original HFSS dataset masks/weights;
2. a physics-aware optimized teacher directory, e.g. greedy_psll_v2_canonical;
3. an analytic full-array baseline for the same target directions.

The optimized teacher is evaluated with the same array-factor physics layer used
by the quick-model workflow. Existing HFSS GainTotal exports are used as the
original full-wave reference; optimized masks still need follow-up HFSS solves
for final full-wave confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_TEACHER_DIR = DEFAULT_DATASET_DIR / "optimized_teachers" / "greedy_psll_v2_canonical"
KMAX = 6
NUM_ELEMENTS = 256
EPS = 1.0e-12
SPARSE_LABEL = "sparse_0.5_0.9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--theta-step", type=float, default=2.0)
    parser.add_argument("--phi-step", type=float, default=5.0)
    parser.add_argument("--sidelobe-exclusion-deg", type=float, default=8.0)
    parser.add_argument("--mainlobe-drop-limit-db", type=float, default=0.5)
    parser.add_argument("--full-psll-margin-db", type=float, default=1.0)
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument("--teacher-label", default=None)
    return parser.parse_args()


def unit_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    ).astype(np.float32)


def make_grid(theta_step: float, phi_step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_values = np.arange(0.0, 90.0 + 0.1, theta_step, dtype=np.float32)
    phi_values = np.arange(0.0, 360.0, phi_step, dtype=np.float32)
    theta, phi = np.meshgrid(theta_values, phi_values, indexing="ij")
    theta_flat = theta.reshape(-1)
    phi_flat = phi.reshape(-1)
    return theta_flat, phi_flat, unit_vectors(theta_flat, phi_flat)


def complex_from_ri(weights_ri: np.ndarray) -> np.ndarray:
    return (weights_ri[..., 0] + 1j * weights_ri[..., 1]).astype(np.complex64)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    arrays_npz = np.load(path, allow_pickle=False)
    return {key: arrays_npz[key] for key in arrays_npz.files}


def load_splits(path: Path, num_samples: int) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    splits = {
        key.replace("_id", ""): np.asarray(value, dtype=np.int64)
        for key, value in payload["splits"].items()
        if key in {"train_id", "val_id", "test_id", "train", "val", "test"}
    }
    # Prefer canonical names if both train and train_id exist.
    for key in ("train", "val", "test"):
        if key not in splits and f"{key}_id" in payload["splits"]:
            splits[key] = np.asarray(payload["splits"][f"{key}_id"], dtype=np.int64)
    splits["all"] = np.arange(num_samples, dtype=np.int64)
    return {key: value for key, value in splits.items() if key in {"train", "val", "test", "all"}}


def load_hfss_metric_map(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idx = int(row["sample_index"])
            out[idx] = {
                "hfss_target_peak_min_db": parse_float(row.get("target_peak_min_db")),
                "hfss_target_spread_db": parse_float(row.get("target_peak_spread_db")),
                "hfss_psll_to_weakest_peak_db": parse_float(row.get("psll_to_weakest_peak_db")),
                "hfss_worst_sidelobe_abs_db": parse_float(row.get("worst_sidelobe_abs_db")),
            }
    return out


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def make_full_array_weights(
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    num_samples = targets_deg.shape[0]
    weights = np.zeros((num_samples, NUM_ELEMENTS, KMAX), dtype=np.complex64)
    for i in range(num_samples):
        valid_indices = np.flatnonzero(task_valid[i].astype(bool))
        if valid_indices.size == 0:
            continue
        target_dirs = unit_vectors(targets_deg[i, valid_indices, 0], targets_deg[i, valid_indices, 1])
        phase = 2.0 * np.pi * (target_dirs @ positions.T)
        tx = np.exp(1j * phase).astype(np.complex64).T / float(NUM_ELEMENTS)
        for col, task_index in enumerate(valid_indices):
            weights[i, :, task_index] = tx[:, col]
    return weights


def compute_af_metrics(
    *,
    weights: np.ndarray,
    masks: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    positions: np.ndarray,
    grid_dirs: np.ndarray,
    sidelobe_exclusion_deg: float,
) -> dict[str, np.ndarray]:
    num_samples = weights.shape[0]
    grid_phase = 2.0 * np.pi * (grid_dirs @ positions.T)
    grid_steer = np.exp(-1j * grid_phase).astype(np.complex64)

    target_peak_min = np.full(num_samples, np.nan, dtype=np.float32)
    target_peak_mean = np.full(num_samples, np.nan, dtype=np.float32)
    target_spread = np.full(num_samples, np.nan, dtype=np.float32)
    worst_sidelobe = np.full(num_samples, np.nan, dtype=np.float32)
    psll_weak = np.full(num_samples, np.nan, dtype=np.float32)
    psll_strong = np.full(num_samples, np.nan, dtype=np.float32)
    isolation_worst = np.full(num_samples, np.nan, dtype=np.float32)
    energy_proxy = np.sum(np.abs(weights) ** 2, axis=(1, 2)).astype(np.float32)
    active_count = masks.astype(bool).sum(axis=1).astype(np.int32)

    for i in range(num_samples):
        valid_indices = np.flatnonzero(task_valid[i].astype(bool))
        if valid_indices.size == 0:
            continue
        target_dirs = unit_vectors(targets_deg[i, valid_indices, 0], targets_deg[i, valid_indices, 1])
        target_phase = 2.0 * np.pi * (target_dirs @ positions.T)
        target_steer = np.exp(-1j * target_phase).astype(np.complex64)

        combined = weights[i].sum(axis=1)
        target_resp = target_steer @ combined
        target_db = 10.0 * np.log10(np.maximum(np.abs(target_resp) ** 2, EPS))
        target_peak_min[i] = float(target_db.min())
        target_peak_mean[i] = float(target_db.mean())
        target_spread[i] = float(target_db.max() - target_db.min())

        grid_resp = grid_steer @ combined
        grid_db = 10.0 * np.log10(np.maximum(np.abs(grid_resp) ** 2, EPS))
        dots = np.clip(grid_dirs @ target_dirs.T, -1.0, 1.0)
        dists = np.rad2deg(np.arccos(dots))
        side_mask = dists.min(axis=1) > sidelobe_exclusion_deg
        if np.any(side_mask):
            side_max = float(grid_db[side_mask].max())
            worst_sidelobe[i] = side_max
            psll_weak[i] = side_max - float(target_db.min())
            psll_strong[i] = side_max - float(target_db.max())

        if valid_indices.size > 1:
            task_weights = weights[i][:, valid_indices]
            response = target_steer @ task_weights
            mag = np.maximum(np.abs(response), math.sqrt(EPS))
            iso_values: list[float] = []
            for task_idx in range(valid_indices.size):
                desired = float(mag[task_idx, task_idx])
                leakage = float(np.delete(mag[:, task_idx], task_idx).max())
                iso_values.append(20.0 * math.log10(desired / max(leakage, math.sqrt(EPS))))
            isolation_worst[i] = float(min(iso_values))

    return {
        "active_count": active_count,
        "target_peak_min_db": target_peak_min,
        "target_peak_mean_db": target_peak_mean,
        "target_spread_db": target_spread,
        "worst_sidelobe_db": worst_sidelobe,
        "psll_to_weakest_peak_db": psll_weak,
        "psll_to_strongest_peak_db": psll_strong,
        "isolation_worst_db": isolation_worst,
        "energy_proxy": energy_proxy,
    }


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def finite_rate(values: Iterable[bool]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def row_for_sample(
    idx: int,
    split: str,
    base: dict[str, np.ndarray],
    original: dict[str, np.ndarray],
    teacher: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    hfss_metrics: dict[int, dict[str, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    original_psll = float(original["psll_to_weakest_peak_db"][idx])
    teacher_psll = float(teacher["psll_to_weakest_peak_db"][idx])
    full_psll = float(full["psll_to_weakest_peak_db"][idx])
    original_peak = float(original["target_peak_min_db"][idx])
    teacher_peak = float(teacher["target_peak_min_db"][idx])
    full_peak = float(full["target_peak_min_db"][idx])
    delta_psll = teacher_psll - original_psll
    delta_peak = teacher_peak - original_peak
    active_count = int(teacher["active_count"][idx])
    channel_saving = 1.0 - active_count / float(NUM_ELEMENTS)
    mainlobe_kept = delta_peak >= -float(args.mainlobe_drop_limit_db)
    improved = delta_psll < -1.0e-9
    full_nonworse = teacher_psll <= full_psll + float(args.full_psll_margin_db)
    hamming_delta = int(np.count_nonzero(base["masks"][idx] != teacher["masks"][idx]))
    targets = [
        [float(theta), float(phi)]
        for theta, phi in base["targets_deg"][idx, base["task_valid"][idx].astype(bool)]
    ]
    hfss = hfss_metrics.get(idx, {})
    hfss_psll = hfss.get("hfss_psll_to_weakest_peak_db", float("nan"))
    return {
        "sample_index": idx,
        "sample_id": str(base["sample_ids"][idx]),
        "split": split,
        "k": int(base["k_values"][idx]),
        "active_ratio": f"{float(base['active_ratios_requested'][idx]):.1f}",
        "active_count": active_count,
        "channel_saving_vs_full": channel_saving,
        "targets_json": json.dumps(targets, ensure_ascii=False),
        "original_hfss_psll_weak_db": hfss_psll,
        "original_af_psll_weak_db": original_psll,
        "teacher_af_psll_weak_db": teacher_psll,
        "full_af_psll_weak_db": full_psll,
        "delta_teacher_vs_original_psll_db": delta_psll,
        "delta_teacher_vs_full_psll_db": teacher_psll - full_psll,
        "original_af_weak_peak_db": original_peak,
        "teacher_af_weak_peak_db": teacher_peak,
        "full_af_weak_peak_db": full_peak,
        "delta_teacher_vs_original_weak_peak_db": delta_peak,
        "delta_teacher_vs_full_weak_peak_db": teacher_peak - full_peak,
        "original_af_target_spread_db": float(original["target_spread_db"][idx]),
        "teacher_af_target_spread_db": float(teacher["target_spread_db"][idx]),
        "original_af_isolation_worst_db": float(original["isolation_worst_db"][idx]),
        "teacher_af_isolation_worst_db": float(teacher["isolation_worst_db"][idx]),
        "original_energy_proxy": float(original["energy_proxy"][idx]),
        "teacher_energy_proxy": float(teacher["energy_proxy"][idx]),
        "mask_hamming_delta": hamming_delta,
        "psll_improved": int(improved),
        "mainlobe_kept": int(mainlobe_kept),
        "full_psll_within_margin": int(full_nonworse),
        "joint_success": int(improved and mainlobe_kept and active_count < NUM_ELEMENTS),
        "needs_hfss_resolve": int(hamming_delta > 0),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: list[tuple[str, str, str]] = []
    splits = ["all", "train", "val", "test"]
    for split in splits:
        split_rows = rows if split == "all" else [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        keys.append((split, "all", "all"))
        keys.append((split, "all", SPARSE_LABEL))
        for ratio in sorted({str(row["active_ratio"]) for row in split_rows}):
            keys.append((split, "all", ratio))
        for k in sorted({int(row["k"]) for row in split_rows}):
            keys.append((split, str(k), "all"))
            keys.append((split, str(k), SPARSE_LABEL))
            for ratio in sorted({str(row["active_ratio"]) for row in split_rows if int(row["k"]) == k}):
                keys.append((split, str(k), ratio))

    out: list[dict[str, Any]] = []
    for split, k_label, ratio_label in keys:
        group = rows if split == "all" else [row for row in rows if row["split"] == split]
        if k_label != "all":
            group = [row for row in group if int(row["k"]) == int(k_label)]
        if ratio_label == SPARSE_LABEL:
            group = [row for row in group if int(row["active_count"]) < NUM_ELEMENTS]
        elif ratio_label != "all":
            group = [row for row in group if str(row["active_ratio"]) == ratio_label]
        if not group:
            continue
        out.append(
            {
                "split": split,
                "k": k_label,
                "active_ratio": ratio_label,
                "n": len(group),
                "channel_saving_mean": finite_mean(float(row["channel_saving_vs_full"]) for row in group),
                "active_count_mean": finite_mean(float(row["active_count"]) for row in group),
                "original_hfss_psll_weak_mean_db": finite_mean(
                    float(row["original_hfss_psll_weak_db"]) for row in group
                ),
                "original_af_psll_weak_mean_db": finite_mean(float(row["original_af_psll_weak_db"]) for row in group),
                "teacher_af_psll_weak_mean_db": finite_mean(float(row["teacher_af_psll_weak_db"]) for row in group),
                "full_af_psll_weak_mean_db": finite_mean(float(row["full_af_psll_weak_db"]) for row in group),
                "delta_teacher_vs_original_psll_mean_db": finite_mean(
                    float(row["delta_teacher_vs_original_psll_db"]) for row in group
                ),
                "delta_teacher_vs_original_psll_p50_db": finite_percentile(
                    (float(row["delta_teacher_vs_original_psll_db"]) for row in group), 50
                ),
                "delta_teacher_vs_original_psll_p95_db": finite_percentile(
                    (float(row["delta_teacher_vs_original_psll_db"]) for row in group), 95
                ),
                "delta_teacher_vs_full_psll_mean_db": finite_mean(
                    float(row["delta_teacher_vs_full_psll_db"]) for row in group
                ),
                "delta_teacher_vs_original_weak_peak_mean_db": finite_mean(
                    float(row["delta_teacher_vs_original_weak_peak_db"]) for row in group
                ),
                "delta_teacher_vs_full_weak_peak_mean_db": finite_mean(
                    float(row["delta_teacher_vs_full_weak_peak_db"]) for row in group
                ),
                "teacher_target_spread_mean_db": finite_mean(
                    float(row["teacher_af_target_spread_db"]) for row in group
                ),
                "teacher_isolation_worst_p05_db": finite_percentile(
                    (float(row["teacher_af_isolation_worst_db"]) for row in group), 5
                ),
                "psll_improved_rate": finite_rate(bool(row["psll_improved"]) for row in group),
                "mainlobe_kept_rate": finite_rate(bool(row["mainlobe_kept"]) for row in group),
                "full_psll_within_margin_rate": finite_rate(bool(row["full_psll_within_margin"]) for row in group),
                "joint_success_rate": finite_rate(bool(row["joint_success"]) for row in group),
                "mean_mask_hamming_delta": finite_mean(float(row["mask_hamming_delta"]) for row in group),
                "needs_hfss_resolve_rate": finite_rate(bool(row["needs_hfss_resolve"]) for row in group),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def assert_compatible(base: dict[str, np.ndarray], teacher: dict[str, np.ndarray], teacher_dir: Path) -> None:
    for key in ("sample_ids", "k_values", "targets_deg", "task_valid", "positions_lambda"):
        if key not in teacher:
            raise RuntimeError(f"{teacher_dir} missing required array {key!r}")
    if not np.array_equal(base["sample_ids"], teacher["sample_ids"]):
        raise RuntimeError("Teacher sample_ids do not match base dataset.")
    if not np.allclose(base["targets_deg"], teacher["targets_deg"], equal_nan=True):
        raise RuntimeError("Teacher target directions do not match base dataset.")
    if not np.array_equal(base["task_valid"], teacher["task_valid"]):
        raise RuntimeError("Teacher task_valid does not match base dataset.")


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    teacher_dir: Path = args.teacher_dir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (dataset_dir / "training_runs" / f"pagan_lite_mvp_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_npz(dataset_dir / "dataset_arrays.npz")
    teacher = load_npz(teacher_dir / "dataset_arrays.npz")
    assert_compatible(base, teacher, teacher_dir)

    splits = load_splits(dataset_dir / "training_split_manifest.json", int(base["sample_ids"].shape[0]))
    split_by_index = np.full(base["sample_ids"].shape[0], "unassigned", dtype=object)
    for split in ("train", "val", "test"):
        if split in splits:
            split_by_index[splits[split]] = split

    theta, phi, grid_dirs = make_grid(float(args.theta_step), float(args.phi_step))
    positions = base["positions_lambda"].astype(np.float32)
    original_weights = complex_from_ri(base["task_weights_real_imag"])
    teacher_weights = complex_from_ri(teacher["task_weights_real_imag"])
    full_weights = make_full_array_weights(base["targets_deg"], base["task_valid"], positions)
    full_masks = np.ones_like(base["masks"], dtype=np.int8)

    original_metrics = compute_af_metrics(
        weights=original_weights,
        masks=base["masks"],
        targets_deg=base["targets_deg"],
        task_valid=base["task_valid"],
        positions=positions,
        grid_dirs=grid_dirs,
        sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
    )
    teacher_metrics = compute_af_metrics(
        weights=teacher_weights,
        masks=teacher["masks"],
        targets_deg=teacher["targets_deg"],
        task_valid=teacher["task_valid"],
        positions=positions,
        grid_dirs=grid_dirs,
        sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
    )
    full_metrics = compute_af_metrics(
        weights=full_weights,
        masks=full_masks,
        targets_deg=base["targets_deg"],
        task_valid=base["task_valid"],
        positions=positions,
        grid_dirs=grid_dirs,
        sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
    )
    # Keep masks in metric dict for concise downstream comparisons.
    teacher_metrics["masks"] = teacher["masks"]

    hfss_metric_path = dataset_dir / "hfss_analysis" / "sample_hfss_metrics_2400.csv"
    hfss_metrics = load_hfss_metric_map(hfss_metric_path)

    rows = [
        row_for_sample(
            int(idx),
            str(split_by_index[idx]),
            base,
            original_metrics,
            teacher_metrics,
            full_metrics,
            hfss_metrics,
            args,
        )
        for idx in range(base["sample_ids"].shape[0])
    ]
    summary_rows = summarize(rows)

    candidate_rows = [
        row
        for row in rows
        if row["split"] == "test"
        and int(row["active_count"]) < NUM_ELEMENTS
        and int(row["mainlobe_kept"]) == 1
        and int(row["psll_improved"]) == 1
    ]
    candidate_rows.sort(key=lambda row: float(row["delta_teacher_vs_original_psll_db"]))
    candidate_rows = candidate_rows[: max(0, int(args.candidate_count))]

    write_csv(out_dir / "mvp_sample_metrics.csv", rows)
    write_csv(out_dir / "mvp_summary_by_split_k_active.csv", summary_rows)
    write_csv(out_dir / "mvp_hfss_resolve_candidates.csv", candidate_rows)

    headline_summary = {
        "dataset_dir": str(dataset_dir),
        "teacher_dir": str(teacher_dir),
        "teacher_label": args.teacher_label or teacher_dir.name,
        "out_dir": str(out_dir),
        "grid": {
            "theta_step_deg": float(args.theta_step),
            "phi_step_deg": float(args.phi_step),
            "num_points": int(grid_dirs.shape[0]),
            "theta_points": int(np.unique(theta).size),
            "phi_points": int(np.unique(phi).size),
            "sidelobe_exclusion_deg": float(args.sidelobe_exclusion_deg),
        },
        "thresholds": {
            "mainlobe_drop_limit_db": float(args.mainlobe_drop_limit_db),
            "full_psll_margin_db": float(args.full_psll_margin_db),
        },
        "num_samples": int(base["sample_ids"].shape[0]),
        "split_counts": {key: int(value.shape[0]) for key, value in splits.items() if key != "all"},
        "headline_rows": [
            row
            for row in summary_rows
            if row["split"] in {"test", "all"} and row["k"] == "all" and row["active_ratio"] == "all"
        ],
        "test_sparse_rows": [
            row
            for row in summary_rows
            if row["split"] == "test"
            and row["k"] == "all"
            and row["active_ratio"] in {SPARSE_LABEL, "0.5", "0.6", "0.7", "0.8", "0.9"}
        ],
        "outputs": {
            "sample_metrics": str(out_dir / "mvp_sample_metrics.csv"),
            "summary": str(out_dir / "mvp_summary_by_split_k_active.csv"),
            "hfss_resolve_candidates": str(out_dir / "mvp_hfss_resolve_candidates.csv"),
        },
        "caveat": (
            "Optimized/teacher masks are evaluated by the array-factor physics layer. "
            "The original HFSS GainTotal metrics are included as reference; shortlisted "
            "optimized masks should be re-solved in HFSS for final full-wave validation."
        ),
    }
    (out_dir / "mvp_headline.json").write_text(
        json.dumps(headline_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(headline_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
