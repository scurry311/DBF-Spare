"""Build a leakage-safe full-wave residual-critic dataset.

The v2 dataset keeps only complete task-level HFSS validations, removes exact
mask/weight duplicates, and assigns every variant of one scene to the same
train/validation/test split.  It also derives AF baselines, full-wave residuals,
two isolation gates, and a mainlobe-quality gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from build_stage1_metric_dataset import (
    combined_hfss_metrics,
    complex_from_ri,
    compute_af_metrics,
    load_npz,
    make_grid,
    read_hfss_pattern,
    unit_vectors,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_OUT_DIR = DEFAULT_DATASET_DIR / "stage1_fullwave_residual_dataset_v2_20260714"
KMAX = 6
NUM_ELEMENTS = 256
EPS = 1.0e-12
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
    "pointing_error_max_deg",
)
RESIDUAL_NAMES = (
    "psll_db",
    "iso_nearest_db",
    "iso_local_db",
    "peak_min_db",
    "peak_spread_db",
)
GATE_NAMES = ("gate15", "gate20", "mainlobe_gate", "strict_engineering_gate")
SCALAR_NAMES = (
    "k_norm",
    "active_ratio",
    "num_active_norm",
    "af_psll_db",
    "af_iso_nearest_db",
    "af_iso_local_db",
    "af_peak_min_db",
    "af_peak_spread_db",
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--theta-step", type=float, default=2.0)
    parser.add_argument("--phi-step", type=float, default=5.0)
    parser.add_argument("--sidelobe-exclusion-deg", type=float, default=8.0)
    parser.add_argument("--local-radius-deg", type=float, default=5.0)
    parser.add_argument("--gate-psll-max-db", type=float, default=0.0)
    parser.add_argument("--gate-nearest-iso-min-db", type=float, default=25.0)
    parser.add_argument("--gate-local15-min-db", type=float, default=15.0)
    parser.add_argument("--gate-local20-min-db", type=float, default=20.0)
    parser.add_argument("--mainlobe-drop-max-db", type=float, default=0.5)
    parser.add_argument("--mainlobe-spread-max-db", type=float, default=3.0)
    parser.add_argument("--pointing-error-max-deg", type=float, default=3.0)
    parser.add_argument("--condition-max", type=float, default=1.0e8)
    parser.add_argument(
        "--max-channel-amplitude",
        type=float,
        default=0.0,
        help="0 disables an absolute channel-amplitude gate; the feature is still recorded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


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


def resolve_teacher_dir(run_dir: Path) -> Path | None:
    for name in ("analysis_summary.json", "prepare_summary.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        raw = payload.get("teacher_dir")
        if raw:
            teacher_dir = Path(raw)
            if not teacher_dir.is_absolute():
                teacher_dir = (ROOT / teacher_dir).resolve()
            if (teacher_dir / "dataset_arrays.npz").exists():
                return teacher_dir
    return None


def strategy_name(source: str) -> str:
    for pattern in (
        r"_(a[1-4]_.+)_chunk\d+$",
        r"_(s[1-4]_.+)_chunk\d+$",
    ):
        match = re.search(pattern, source)
        if match:
            return match.group(1)
    return source


def load_teacher_diagnostics(teacher_dir: Path) -> dict[int, dict[str, str]]:
    path = teacher_dir / "iso_lcmv_metrics.csv"
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {int(row["sample_index"]): row for row in csv.DictReader(handle)}


def variant_hash(mask: np.ndarray, weights_ri: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mask, dtype=np.int8).tobytes())
    rounded = np.round(np.asarray(weights_ri, dtype=np.float32), decimals=7)
    digest.update(rounded.tobytes())
    return digest.hexdigest()


def pointing_error_max_deg(
    pattern_path: Path,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    local_radius_deg: float,
) -> float:
    if not pattern_path.exists():
        return float("nan")
    pattern = read_hfss_pattern(pattern_path)
    valid_targets = np.asarray(targets_deg[task_valid.astype(bool)], dtype=np.float32)
    if not valid_targets.size:
        return float("nan")
    target_dirs = unit_vectors(valid_targets[:, 0], valid_targets[:, 1])
    distances = np.rad2deg(np.arccos(np.clip(pattern["dirs"] @ target_dirs.T, -1.0, 1.0)))
    errors: list[float] = []
    for col in range(valid_targets.shape[0]):
        local = np.flatnonzero(distances[:, col] <= float(local_radius_deg))
        if not local.size:
            return float("nan")
        peak_index = int(local[int(np.argmax(pattern["gain_db"][local]))])
        errors.append(float(distances[peak_index, col]))
    return float(max(errors))


def min_target_separation_deg(targets_deg: np.ndarray, task_valid: np.ndarray) -> float:
    valid = np.asarray(targets_deg[task_valid.astype(bool)], dtype=np.float32)
    if valid.shape[0] < 2:
        return 180.0
    dirs = unit_vectors(valid[:, 0], valid[:, 1])
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    distances = np.rad2deg(np.arccos(dots))
    distances += np.eye(valid.shape[0], dtype=np.float32) * 360.0
    return float(np.min(distances))


def amplitude_stats(weights: np.ndarray) -> tuple[float, float, float, float]:
    amplitude = np.abs(weights)
    nonzero = amplitude[amplitude > 1.0e-9]
    energy = float(np.sum(amplitude**2))
    l2 = math.sqrt(max(energy, 0.0))
    max_amp = float(nonzero.max()) if nonzero.size else 0.0
    if nonzero.size:
        dynamic_db = 20.0 * math.log10(max_amp / max(float(nonzero.min()), 1.0e-12))
    else:
        dynamic_db = 0.0
    return energy, l2, max_amp, dynamic_db


def bool_gate(*conditions: bool) -> float:
    return float(all(conditions))


def assign_scene_splits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[int, int]:
    by_scene: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scene[int(record["sample_index"])].append(record)
    strata: dict[tuple[int, float, int, int], list[int]] = defaultdict(list)
    for sample_index, group in by_scene.items():
        first = group[0]
        key = (
            int(first["k"]),
            round(float(first["active_ratio"]), 3),
            int(any(float(item["gate15"]) >= 0.5 for item in group)),
            int(any(float(item["gate20"]) >= 0.5 for item in group)),
        )
        strata[key].append(sample_index)

    rng = np.random.default_rng(int(seed))
    split_by_scene: dict[int, int] = {}
    for key in sorted(strata):
        indices = np.asarray(sorted(strata[key]), dtype=np.int64)
        rng.shuffle(indices)
        n = int(indices.size)
        if n < 7:
            n_train, n_val = n, 0
        else:
            n_train = int(round(n * float(train_fraction)))
            n_val = int(round(n * float(val_fraction)))
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        for pos, sample_index in enumerate(indices.tolist()):
            split_by_scene[int(sample_index)] = 0 if pos < n_train else (1 if pos < n_train + n_val else 2)
    return split_by_scene


def collect_complete_variants(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation_root = dataset_dir / "hfss_fullwave_validations"
    candidates: list[dict[str, Any]] = []
    scanned_runs = 0
    accepted_runs = 0
    incomplete_rows = 0
    missing_teacher_rows = 0
    run_dirs = sorted(
        (path for path in validation_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        metrics_path = run_dir / "hfss_task_fullwave_metrics.csv"
        if not metrics_path.exists():
            continue
        scanned_runs += 1
        teacher_dir = resolve_teacher_dir(run_dir)
        if teacher_dir is None:
            with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
                missing_teacher_rows += sum(1 for _ in csv.DictReader(handle))
            continue
        teacher = load_npz(teacher_dir / "dataset_arrays.npz")
        diagnostics = load_teacher_diagnostics(teacher_dir)
        accepted_runs += 1
        with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if int(row.get("combined_complete", "0")) != 1 or int(row.get("isolation_complete", "0")) != 1:
                    incomplete_rows += 1
                    continue
                sample_index = int(row["sample_index"])
                diag = diagnostics.get(sample_index, {})
                mask = teacher["masks"][sample_index].astype(np.int8)
                weights_ri = teacher["task_weights_real_imag"][sample_index].astype(np.float32)
                output_dir = Path(row.get("combined_output_dir", ""))
                pattern_path = output_dir / "hfss_gain_total_theta_phi.csv"
                candidates.append(
                    {
                        "source": run_dir.name,
                        "strategy": strategy_name(run_dir.name),
                        "source_mtime": metrics_path.stat().st_mtime,
                        "teacher_dir": str(teacher_dir),
                        "sample_index": sample_index,
                        "mask": mask,
                        "weights_ri": weights_ri,
                        "variant_hash": variant_hash(mask, weights_ri),
                        "active_ratio": finite_float(row.get("active_ratio")),
                        "num_active": int(round(finite_float(row.get("active_count")))),
                        "hfss_psll_db": finite_float(row.get("combined_psll_to_weakest_peak_db")),
                        "hfss_iso_nearest_db": finite_float(row.get("isolation_worst_nearest_db")),
                        "hfss_iso_local_db": finite_float(row.get("isolation_worst_local_db")),
                        "hfss_peak_min_db": finite_float(row.get("combined_target_peak_min_db")),
                        "hfss_peak_spread_db": finite_float(row.get("combined_target_spread_db")),
                        "combined_pattern_path": str(pattern_path),
                        "condition": finite_float(diag.get("condition")),
                        "null_constraint_count": finite_float(diag.get("null_constraint_count")),
                    }
                )

    unique: dict[tuple[int, str], dict[str, Any]] = {}
    duplicate_sources: dict[tuple[int, str], list[str]] = defaultdict(list)
    for candidate in candidates:
        key = (int(candidate["sample_index"]), str(candidate["variant_hash"]))
        duplicate_sources[key].append(str(candidate["source"]))
        if key not in unique:
            unique[key] = candidate
    records = list(unique.values())
    for record in records:
        key = (int(record["sample_index"]), str(record["variant_hash"]))
        record["duplicate_count"] = len(duplicate_sources[key])
        record["duplicate_sources"] = "|".join(sorted(set(duplicate_sources[key])))
    diagnostics = {
        "scanned_metric_runs": scanned_runs,
        "accepted_metric_runs": accepted_runs,
        "raw_complete_variant_rows": len(candidates),
        "deduplicated_variant_rows": len(records),
        "duplicate_rows_removed": len(candidates) - len(records),
        "incomplete_rows_rejected": incomplete_rows,
        "missing_teacher_rows_rejected": missing_teacher_rows,
    }
    return records, diagnostics


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base = load_npz(args.dataset_dir / "dataset_arrays.npz")
    positions = base["positions_lambda"].astype(np.float32)
    grid_dirs = make_grid(float(args.theta_step), float(args.phi_step))
    records, scan_summary = collect_complete_variants(args.dataset_dir)
    records.sort(key=lambda row: (int(row["sample_index"]), str(row["source"]), str(row["variant_hash"])))

    reference_cache: dict[int, dict[str, float]] = {}
    processed: list[dict[str, Any]] = []
    for pos, record in enumerate(records, start=1):
        sample_index = int(record["sample_index"])
        task_valid = base["task_valid"][sample_index].astype(bool)
        targets_deg = base["targets_deg"][sample_index].astype(np.float32)
        weights = complex_from_ri(record["weights_ri"])
        af = compute_af_metrics(
            weights=weights,
            targets_deg=targets_deg,
            task_valid=task_valid,
            positions=positions,
            grid_dirs=grid_dirs,
            sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
            local_radius_deg=float(args.local_radius_deg),
        )
        if sample_index not in reference_cache:
            sample_id = str(base["sample_ids"][sample_index])
            reference_path = args.dataset_dir / "samples" / sample_id / "hfss_gain_total_theta_phi.csv"
            if reference_path.exists():
                reference_cache[sample_index] = combined_hfss_metrics(
                    read_hfss_pattern(reference_path),
                    targets_deg,
                    task_valid,
                    target_radius_deg=float(args.local_radius_deg),
                    sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
                )
            else:
                reference_cache[sample_index] = {
                    "hfss_peak_min_db": float("nan"),
                    "hfss_peak_spread_db": float("nan"),
                }
        reference = reference_cache[sample_index]
        pointing = pointing_error_max_deg(
            Path(str(record["combined_pattern_path"])),
            targets_deg,
            task_valid,
            float(args.local_radius_deg),
        )
        energy, weight_l2, max_amp, dynamic_db = amplitude_stats(weights)
        hfss_peak_min = float(record["hfss_peak_min_db"])
        energy_normalized = energy / max(10.0 ** (hfss_peak_min / 10.0), EPS)
        reference_peak_min = finite_float(reference.get("hfss_peak_min_db"))
        reference_spread = finite_float(reference.get("hfss_peak_spread_db"))
        mainlobe_drop = reference_peak_min - hfss_peak_min
        k_value = int(base["k_values"][sample_index])
        valid_targets = targets_deg[task_valid]
        max_theta = float(valid_targets[:, 0].max()) if valid_targets.size else 0.0
        mean_theta = float(valid_targets[:, 0].mean()) if valid_targets.size else 0.0
        min_sep = min_target_separation_deg(targets_deg, task_valid)
        condition = finite_float(record.get("condition"))
        null_count = finite_float(record.get("null_constraint_count"))
        condition_ok = (not np.isfinite(condition)) or condition <= float(args.condition_max)
        amplitude_ok = float(args.max_channel_amplitude) <= 0.0 or max_amp <= float(args.max_channel_amplitude)
        gate15 = bool_gate(
            np.isfinite(record["hfss_psll_db"]) and record["hfss_psll_db"] <= float(args.gate_psll_max_db),
            np.isfinite(record["hfss_iso_nearest_db"])
            and record["hfss_iso_nearest_db"] >= float(args.gate_nearest_iso_min_db),
            np.isfinite(record["hfss_iso_local_db"])
            and record["hfss_iso_local_db"] >= float(args.gate_local15_min_db),
        )
        gate20 = bool_gate(
            np.isfinite(record["hfss_psll_db"]) and record["hfss_psll_db"] <= float(args.gate_psll_max_db),
            np.isfinite(record["hfss_iso_nearest_db"])
            and record["hfss_iso_nearest_db"] >= float(args.gate_nearest_iso_min_db),
            np.isfinite(record["hfss_iso_local_db"])
            and record["hfss_iso_local_db"] >= float(args.gate_local20_min_db),
        )
        mainlobe_gate = bool_gate(
            np.isfinite(mainlobe_drop) and mainlobe_drop <= float(args.mainlobe_drop_max_db),
            np.isfinite(record["hfss_peak_spread_db"])
            and record["hfss_peak_spread_db"] <= float(args.mainlobe_spread_max_db),
            np.isfinite(pointing) and pointing <= float(args.pointing_error_max_deg),
        )
        strict_engineering = bool_gate(gate20 >= 0.5, mainlobe_gate >= 0.5, condition_ok, amplitude_ok)
        af_gate15 = bool_gate(
            np.isfinite(af["af_psll_db"]) and af["af_psll_db"] <= float(args.gate_psll_max_db),
            np.isfinite(af["af_iso_nearest_db"])
            and af["af_iso_nearest_db"] >= float(args.gate_nearest_iso_min_db),
            np.isfinite(af["af_iso_local_db"])
            and af["af_iso_local_db"] >= float(args.gate_local15_min_db),
        )
        rank_violation = (
            max(float(record["hfss_psll_db"]) - float(args.gate_psll_max_db), 0.0)
            + 0.5 * max(float(args.gate_nearest_iso_min_db) - float(record["hfss_iso_nearest_db"]), 0.0)
            + 0.35 * max(float(args.gate_local20_min_db) - float(record["hfss_iso_local_db"]), 0.0)
            + 0.5 * max(mainlobe_drop - float(args.mainlobe_drop_max_db), 0.0)
            + 0.25 * max(float(record["hfss_peak_spread_db"]) - float(args.mainlobe_spread_max_db), 0.0)
            + 0.25 * max(pointing - float(args.pointing_error_max_deg), 0.0)
        )
        processed.append(
            {
                **record,
                "k": k_value,
                "targets_deg": targets_deg,
                "task_valid": task_valid.astype(np.int8),
                "af_psll_db": float(af["af_psll_db"]),
                "af_iso_nearest_db": float(af["af_iso_nearest_db"]),
                "af_iso_local_db": float(af["af_iso_local_db"]),
                "af_peak_min_db": float(af["af_peak_min_db"]),
                "af_peak_spread_db": float(af["af_peak_spread_db"]),
                "energy_proxy": energy,
                "energy_normalized_hfss": energy_normalized,
                "weight_l2": weight_l2,
                "max_channel_amplitude": max_amp,
                "amplitude_dynamic_range_db": dynamic_db,
                "pointing_error_max_deg": pointing,
                "reference_hfss_peak_min_db": reference_peak_min,
                "reference_hfss_peak_spread_db": reference_spread,
                "mainlobe_drop_db": mainlobe_drop,
                "min_target_separation_deg": min_sep,
                "max_scan_theta_deg": max_theta,
                "mean_scan_theta_deg": mean_theta,
                "gate15": gate15,
                "gate20": gate20,
                "mainlobe_gate": mainlobe_gate,
                "strict_engineering_gate": strict_engineering,
                "af_gate15": af_gate15,
                "hard_negative": float(af_gate15 >= 0.5 and gate15 < 0.5),
                "rank_violation": float(rank_violation),
            }
        )
        if pos % 250 == 0 or pos == len(records):
            print(f"processed {pos}/{len(records)} variants, elapsed={time.time() - started:.1f}s")

    split_by_scene = assign_scene_splits(
        processed,
        seed=int(args.seed),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
    )
    split_names = np.asarray(["train", "val", "test"])
    n = len(processed)
    masks = np.zeros((n, NUM_ELEMENTS), dtype=np.int8)
    weights_ri = np.zeros((n, NUM_ELEMENTS, KMAX, 2), dtype=np.float32)
    targets_deg = np.zeros((n, KMAX, 2), dtype=np.float32)
    task_valid = np.zeros((n, KMAX), dtype=np.int8)
    sample_index = np.zeros(n, dtype=np.int64)
    split_id = np.zeros(n, dtype=np.int8)
    k_values = np.zeros(n, dtype=np.int8)
    active_ratios = np.zeros(n, dtype=np.float32)
    num_active = np.zeros(n, dtype=np.int16)
    source = np.empty(n, dtype="U128")
    strategy = np.empty(n, dtype="U128")
    variant_hashes = np.empty(n, dtype="U64")
    af_metrics = np.zeros((n, len(AF_METRIC_NAMES)), dtype=np.float32)
    hfss_metrics = np.zeros((n, len(HFSS_METRIC_NAMES)), dtype=np.float32)
    residuals = np.zeros((n, len(RESIDUAL_NAMES)), dtype=np.float32)
    gates = np.zeros((n, len(GATE_NAMES)), dtype=np.float32)
    scalars = np.zeros((n, len(SCALAR_NAMES)), dtype=np.float32)
    rank_violation = np.zeros(n, dtype=np.float32)
    hard_negative = np.zeros(n, dtype=np.int8)
    csv_rows: list[dict[str, Any]] = []
    for row_index, item in enumerate(processed):
        idx = int(item["sample_index"])
        split = int(split_by_scene[idx])
        mask = np.asarray(item["mask"], dtype=np.int8)
        weight_ri = np.asarray(item["weights_ri"], dtype=np.float32)
        af_values = np.asarray(
            [
                item["af_psll_db"],
                item["af_iso_nearest_db"],
                item["af_iso_local_db"],
                item["af_peak_min_db"],
                item["af_peak_spread_db"],
                item["energy_proxy"],
            ],
            dtype=np.float32,
        )
        hfss_values = np.asarray(
            [
                item["hfss_psll_db"],
                item["hfss_iso_nearest_db"],
                item["hfss_iso_local_db"],
                item["hfss_peak_min_db"],
                item["hfss_peak_spread_db"],
                item["pointing_error_max_deg"],
            ],
            dtype=np.float32,
        )
        residual_values = hfss_values[:5] - af_values[:5]
        condition = finite_float(item.get("condition"))
        null_count = finite_float(item.get("null_constraint_count"))
        scalar_values = np.asarray(
            [
                float(item["k"]) / float(KMAX),
                float(item["active_ratio"]),
                float(item["num_active"]) / float(NUM_ELEMENTS),
                *af_values[:5].tolist(),
                float(item["energy_proxy"]),
                float(item["energy_normalized_hfss"]),
                float(item["weight_l2"]),
                float(item["max_channel_amplitude"]),
                float(item["amplitude_dynamic_range_db"]),
                math.log10(max(condition, 1.0)) if np.isfinite(condition) else 0.0,
                (null_count / 256.0) if np.isfinite(null_count) else 0.0,
                float(item["min_target_separation_deg"]) / 180.0,
                float(item["max_scan_theta_deg"]) / 90.0,
                float(item["mean_scan_theta_deg"]) / 90.0,
                float(item["reference_hfss_peak_min_db"]),
                float(item["reference_hfss_peak_spread_db"]),
            ],
            dtype=np.float32,
        )
        masks[row_index] = mask
        weights_ri[row_index] = weight_ri
        targets_deg[row_index] = np.asarray(item["targets_deg"], dtype=np.float32)
        task_valid[row_index] = np.asarray(item["task_valid"], dtype=np.int8)
        sample_index[row_index] = idx
        split_id[row_index] = split
        k_values[row_index] = int(item["k"])
        active_ratios[row_index] = float(item["active_ratio"])
        num_active[row_index] = int(item["num_active"])
        source[row_index] = str(item["source"])
        strategy[row_index] = str(item["strategy"])
        variant_hashes[row_index] = str(item["variant_hash"])
        af_metrics[row_index] = af_values
        hfss_metrics[row_index] = hfss_values
        residuals[row_index] = residual_values
        gates[row_index] = np.asarray([item[name] for name in GATE_NAMES], dtype=np.float32)
        scalars[row_index] = np.nan_to_num(scalar_values, nan=0.0, posinf=0.0, neginf=0.0)
        rank_violation[row_index] = float(item["rank_violation"])
        hard_negative[row_index] = int(item["hard_negative"])
        csv_rows.append(
            {
                "example_id": row_index,
                "sample_index": idx,
                "sample_id": str(base["sample_ids"][idx]),
                "split": str(split_names[split]),
                "source": str(item["source"]),
                "strategy": str(item["strategy"]),
                "variant_hash": str(item["variant_hash"]),
                "duplicate_count": int(item["duplicate_count"]),
                "k": int(item["k"]),
                "active_ratio": float(item["active_ratio"]),
                "num_active": int(item["num_active"]),
                **{f"af_{name}": float(af_values[col]) for col, name in enumerate(AF_METRIC_NAMES)},
                **{f"hfss_{name}": float(hfss_values[col]) for col, name in enumerate(HFSS_METRIC_NAMES)},
                **{f"residual_{name}": float(residual_values[col]) for col, name in enumerate(RESIDUAL_NAMES)},
                **{name: int(item[name]) for name in GATE_NAMES},
                "mainlobe_drop_db": float(item["mainlobe_drop_db"]),
                "hard_negative": int(item["hard_negative"]),
                "rank_violation": float(item["rank_violation"]),
                "min_target_separation_deg": float(item["min_target_separation_deg"]),
                "max_scan_theta_deg": float(item["max_scan_theta_deg"]),
                "mean_scan_theta_deg": float(item["mean_scan_theta_deg"]),
                "condition": finite_float(item.get("condition")),
                "null_constraint_count": finite_float(item.get("null_constraint_count")),
                "energy_normalized_hfss": float(item["energy_normalized_hfss"]),
                "weight_l2": float(item["weight_l2"]),
                "max_channel_amplitude": float(item["max_channel_amplitude"]),
                "amplitude_dynamic_range_db": float(item["amplitude_dynamic_range_db"]),
            }
        )

    np.savez_compressed(
        args.out_dir / "fullwave_residual_dataset_v2.npz",
        masks=masks,
        weights_real_imag=weights_ri,
        targets_deg=targets_deg,
        task_valid=task_valid,
        sample_index=sample_index,
        split_id=split_id,
        source=source,
        strategy=strategy,
        variant_hash=variant_hashes,
        k_values=k_values,
        active_ratios=active_ratios,
        num_active=num_active,
        af_metrics=af_metrics,
        af_metric_names=np.asarray(AF_METRIC_NAMES),
        hfss_metrics=hfss_metrics,
        hfss_metric_names=np.asarray(HFSS_METRIC_NAMES),
        residuals=residuals,
        residual_names=np.asarray(RESIDUAL_NAMES),
        gates=gates,
        gate_names=np.asarray(GATE_NAMES),
        scalar_features=scalars,
        scalar_names=np.asarray(SCALAR_NAMES),
        rank_violation=rank_violation,
        hard_negative=hard_negative,
        element_ixiy=base["element_ixiy"],
        positions_lambda=base["positions_lambda"],
    )
    write_csv(args.out_dir / "fullwave_residual_dataset_v2.csv", csv_rows)

    distribution_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, float, str], list[int]] = defaultdict(list)
    for row_index in range(n):
        key = (
            str(split_names[int(split_id[row_index])]),
            int(k_values[row_index]),
            round(float(active_ratios[row_index]), 3),
            str(strategy[row_index]),
        )
        groups[key].append(row_index)
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        distribution_rows.append(
            {
                "split": key[0],
                "k": key[1],
                "active_ratio": key[2],
                "strategy": key[3],
                "variant_count": int(indices.size),
                "unique_scene_count": int(np.unique(sample_index[indices]).size),
                "gate15_rate": float(gates[indices, 0].mean()),
                "gate20_rate": float(gates[indices, 1].mean()),
                "mainlobe_gate_rate": float(gates[indices, 2].mean()),
                "strict_engineering_gate_rate": float(gates[indices, 3].mean()),
                "hard_negative_rate": float(hard_negative[indices].mean()),
                "max_scan_theta_mean_deg": finite_mean(csv_rows[i]["max_scan_theta_deg"] for i in indices),
                "min_target_separation_mean_deg": finite_mean(
                    csv_rows[i]["min_target_separation_deg"] for i in indices
                ),
            }
        )
    write_csv(args.out_dir / "dataset_distribution.csv", distribution_rows)
    split_manifest = {
        "seed": int(args.seed),
        "split_names": ["train", "val", "test"],
        "scene_indices": {
            name: sorted(int(idx) for idx, split in split_by_scene.items() if split == split_index)
            for split_index, name in enumerate(("train", "val", "test"))
        },
    }
    (args.out_dir / "scene_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        **scan_summary,
        "variant_count": n,
        "unique_scene_count": int(np.unique(sample_index).size),
        "split_variant_counts": {
            name: int(np.sum(split_id == split_index)) for split_index, name in enumerate(split_names.tolist())
        },
        "split_scene_counts": {
            name: int(len(split_manifest["scene_indices"][name])) for name in split_names.tolist()
        },
        "gate_rates": {name: float(gates[:, col].mean()) for col, name in enumerate(GATE_NAMES)},
        "hard_negative_count": int(hard_negative.sum()),
        "strategy_counts": dict(Counter(str(value) for value in strategy)),
        "thresholds": {
            "psll_max_db": float(args.gate_psll_max_db),
            "nearest_iso_min_db": float(args.gate_nearest_iso_min_db),
            "local15_min_db": float(args.gate_local15_min_db),
            "local20_min_db": float(args.gate_local20_min_db),
            "mainlobe_drop_max_db": float(args.mainlobe_drop_max_db),
            "mainlobe_spread_max_db": float(args.mainlobe_spread_max_db),
            "pointing_error_max_deg": float(args.pointing_error_max_deg),
            "condition_max": float(args.condition_max),
            "max_channel_amplitude": float(args.max_channel_amplitude),
        },
        "elapsed_s": time.time() - started,
        "outputs": {
            "npz": str(args.out_dir / "fullwave_residual_dataset_v2.npz"),
            "csv": str(args.out_dir / "fullwave_residual_dataset_v2.csv"),
            "distribution": str(args.out_dir / "dataset_distribution.csv"),
            "scene_split": str(args.out_dir / "scene_split_manifest.json"),
        },
    }
    (args.out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
