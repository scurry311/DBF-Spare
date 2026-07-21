"""Build the stage-1 metric dataset for HFSS-aware sparse-mask training.

The output cache combines:

* every original 2400-sample mask/weight pair with array-factor labels;
* original combined HFSS GainTotal labels when exported pattern CSV files exist;
* task-level HFSS validation rows for optimized teacher variants.

This dataset is meant for a fast metric critic, not for replacing the main
mask generator. Missing full-wave task-isolation labels are stored as NaN.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
K_VALUES = (1, 2, 4, 6)
KMAX = 6
NUM_ELEMENTS = 256
EPS = 1.0e-12
LABEL_NAMES = (
    "af_psll_db",
    "af_iso_nearest_db",
    "af_iso_local_db",
    "af_peak_min_db",
    "af_peak_spread_db",
    "energy_proxy",
    "hfss_psll_db",
    "hfss_peak_min_db",
    "hfss_peak_spread_db",
    "hfss_iso_nearest_db",
    "hfss_iso_local_db",
    "af_gate_pass",
    "hfss_gate_pass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--theta-step", type=float, default=2.0)
    parser.add_argument("--phi-step", type=float, default=5.0)
    parser.add_argument("--sidelobe-exclusion-deg", type=float, default=8.0)
    parser.add_argument("--local-radius-deg", type=float, default=5.0)
    parser.add_argument("--gate-psll-max-db", type=float, default=0.0)
    parser.add_argument("--gate-nearest-iso-min-db", type=float, default=25.0)
    parser.add_argument("--gate-local-iso-min-db", type=float, default=15.0)
    parser.add_argument("--skip-original-hfss", action="store_true")
    parser.add_argument("--max-original-hfss", type=int, default=0, help="0 means all original samples.")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    return {key: arrays[key] for key in arrays.files}


def load_splits(path: Path, n: int) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, np.ndarray] = {"all": np.arange(n, dtype=np.int64)}
    for key in ("train", "val", "test"):
        raw = payload["splits"].get(key) or payload["splits"].get(f"{key}_id")
        if raw is not None:
            splits[key] = np.asarray(raw, dtype=np.int64)
    return splits


def split_ids_from_manifest(splits: dict[str, np.ndarray], n: int) -> np.ndarray:
    out = np.full(n, 3, dtype=np.int64)
    for split_id, key in enumerate(("train", "val", "test")):
        if key in splits:
            out[splits[key]] = split_id
    return out


def complex_from_ri(weights_ri: np.ndarray) -> np.ndarray:
    return (weights_ri[..., 0] + 1j * weights_ri[..., 1]).astype(np.complex64)


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


def make_grid(theta_step: float, phi_step: float) -> np.ndarray:
    theta_values = np.arange(0.0, 90.0 + 0.1, theta_step, dtype=np.float32)
    phi_values = np.arange(0.0, 360.0, phi_step, dtype=np.float32)
    theta, phi = np.meshgrid(theta_values, phi_values, indexing="ij")
    return unit_vectors(theta.reshape(-1), phi.reshape(-1))


def steering_rx(positions: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    phase = 2.0 * np.pi * (dirs @ positions.T)
    return np.exp(-1j * phase).astype(np.complex64)


def local_probe_dirs(theta_deg: float, phi_deg: float, radius_deg: float) -> np.ndarray:
    offsets = sorted({2.0, float(radius_deg)})
    points: list[tuple[float, float]] = [(float(theta_deg), float(phi_deg) % 360.0)]
    seen = {(round(float(theta_deg), 6), round(float(phi_deg) % 360.0, 6))}
    for offset in offsets:
        if offset <= 0.0:
            continue
        for dtheta in (-offset, 0.0, offset):
            for dphi in (-offset, 0.0, offset):
                if dtheta == 0.0 and dphi == 0.0:
                    continue
                theta_i = min(90.0, max(0.0, float(theta_deg) + dtheta))
                phi_i = (float(phi_deg) + dphi) % 360.0
                key = (round(theta_i, 6), round(phi_i, 6))
                if key not in seen:
                    seen.add(key)
                    points.append((theta_i, phi_i))
    arr = np.asarray(points, dtype=np.float32)
    return unit_vectors(arr[:, 0], arr[:, 1])


def compute_af_metrics(
    *,
    weights: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    positions: np.ndarray,
    grid_dirs: np.ndarray,
    sidelobe_exclusion_deg: float,
    local_radius_deg: float,
) -> dict[str, float]:
    valid_indices = np.flatnonzero(task_valid.astype(bool))
    if valid_indices.size == 0:
        return {
            "af_psll_db": float("nan"),
            "af_iso_nearest_db": float("nan"),
            "af_iso_local_db": float("nan"),
            "af_peak_min_db": float("nan"),
            "af_peak_spread_db": float("nan"),
            "energy_proxy": float(np.sum(np.abs(weights) ** 2)),
        }
    valid_targets = np.nan_to_num(targets_deg[valid_indices], nan=0.0).astype(np.float32)
    target_dirs = unit_vectors(valid_targets[:, 0], valid_targets[:, 1])
    target_steer = steering_rx(positions, target_dirs)
    combined = weights.sum(axis=1)
    target_resp = target_steer @ combined
    target_db = 10.0 * np.log10(np.maximum(np.abs(target_resp) ** 2, EPS))

    grid_steer = steering_rx(positions, grid_dirs)
    grid_resp = grid_steer @ combined
    grid_db = 10.0 * np.log10(np.maximum(np.abs(grid_resp) ** 2, EPS))
    dots = np.clip(grid_dirs @ target_dirs.T, -1.0, 1.0)
    dists = np.rad2deg(np.arccos(dots))
    side_mask = dists.min(axis=1) > float(sidelobe_exclusion_deg)
    side_max = float(grid_db[side_mask].max()) if np.any(side_mask) else float("nan")

    nearest_iso = float("nan")
    local_iso = float("nan")
    if valid_indices.size > 1:
        task_weights = weights[:, valid_indices]
        response = target_steer @ task_weights
        mag = np.maximum(np.abs(response), math.sqrt(EPS))
        nearest_values: list[float] = []
        local_values: list[float] = []
        probe_cache = [
            local_probe_dirs(float(theta), float(phi), float(local_radius_deg))
            for theta, phi in valid_targets
        ]
        for task_pos in range(valid_indices.size):
            desired = float(mag[task_pos, task_pos])
            nearest_leak = float(np.delete(mag[:, task_pos], task_pos).max())
            nearest_values.append(20.0 * math.log10(desired / max(nearest_leak, math.sqrt(EPS))))
            local_leak = math.sqrt(EPS)
            w_task = task_weights[:, task_pos]
            for other_pos, probe_dirs in enumerate(probe_cache):
                if other_pos == task_pos:
                    continue
                probe_steer = steering_rx(positions, probe_dirs)
                local_leak = max(local_leak, float(np.abs(probe_steer @ w_task).max()))
            local_values.append(20.0 * math.log10(max(desired, math.sqrt(EPS)) / max(local_leak, math.sqrt(EPS))))
        nearest_iso = float(min(nearest_values))
        local_iso = float(min(local_values))

    return {
        "af_psll_db": side_max - float(target_db.min()) if np.isfinite(side_max) else float("nan"),
        "af_iso_nearest_db": nearest_iso,
        "af_iso_local_db": local_iso,
        "af_peak_min_db": float(target_db.min()),
        "af_peak_spread_db": float(target_db.max() - target_db.min()),
        "energy_proxy": float(np.sum(np.abs(weights) ** 2)),
    }


def read_hfss_pattern(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    theta = data[:, 0]
    phi = data[:, 1]
    gain = data[:, 2]
    dirs = unit_vectors(theta.astype(np.float32), phi.astype(np.float32))
    return {"theta": theta, "phi": phi, "gain_db": gain, "dirs": dirs}


def combined_hfss_metrics(
    pattern: dict[str, np.ndarray],
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    *,
    target_radius_deg: float,
    sidelobe_exclusion_deg: float,
) -> dict[str, float]:
    valid_indices = np.flatnonzero(task_valid.astype(bool))
    if valid_indices.size == 0:
        return {
            "hfss_psll_db": float("nan"),
            "hfss_peak_min_db": float("nan"),
            "hfss_peak_spread_db": float("nan"),
        }
    targets = np.nan_to_num(targets_deg[valid_indices], nan=0.0)
    target_dirs = unit_vectors(targets[:, 0], targets[:, 1])
    dists = np.rad2deg(np.arccos(np.clip(pattern["dirs"] @ target_dirs.T, -1.0, 1.0)))
    peaks: list[float] = []
    for col in range(targets.shape[0]):
        local = dists[:, col] <= float(target_radius_deg)
        if np.any(local):
            peaks.append(float(np.max(pattern["gain_db"][local])))
        else:
            peaks.append(float(pattern["gain_db"][int(np.argmin(dists[:, col]))]))
    side_mask = dists.min(axis=1) > float(sidelobe_exclusion_deg)
    side_max = float(np.max(pattern["gain_db"][side_mask])) if np.any(side_mask) else float("nan")
    peak_arr = np.asarray(peaks, dtype=np.float64)
    return {
        "hfss_psll_db": side_max - float(peak_arr.min()) if np.isfinite(side_max) else float("nan"),
        "hfss_peak_min_db": float(peak_arr.min()),
        "hfss_peak_spread_db": float(peak_arr.max() - peak_arr.min()),
    }


def gate_pass(psll: float, nearest_iso: float, local_iso: float, args: argparse.Namespace) -> float:
    if not (np.isfinite(psll) and np.isfinite(nearest_iso) and np.isfinite(local_iso)):
        return float("nan")
    return float(
        psll <= float(args.gate_psll_max_db)
        and nearest_iso >= float(args.gate_nearest_iso_min_db)
        and local_iso >= float(args.gate_local_iso_min_db)
    )


def row_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def collect_validation_examples(dataset_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    root = dataset_dir / "hfss_fullwave_validations"
    if not root.exists():
        return out
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = run_dir / "hfss_task_fullwave_metrics.csv"
        summary_path = run_dir / "analysis_summary.json"
        prepare_path = run_dir / "prepare_summary.json"
        if not metrics_path.exists():
            continue
        teacher_dir: Path | None = None
        for candidate in (summary_path, prepare_path):
            if candidate.exists():
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    raw = payload.get("teacher_dir")
                    if raw:
                        teacher_dir = Path(raw)
                        if not teacher_dir.is_absolute():
                            teacher_dir = (Path.cwd() / teacher_dir).resolve()
                        break
                except json.JSONDecodeError:
                    pass
        if teacher_dir is None or not (teacher_dir / "dataset_arrays.npz").exists():
            continue
        teacher = load_npz(teacher_dir / "dataset_arrays.npz")
        with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                idx = int(row["sample_index"])
                out.append(
                    {
                        "source": run_dir.name,
                        "sample_index": idx,
                        "masks": teacher["masks"][idx].astype(np.int8),
                        "weights": complex_from_ri(teacher["task_weights_real_imag"][idx]),
                        "active_ratio": row_float(row, "active_ratio"),
                        "num_active": int(round(row_float(row, "active_count"))),
                        "hfss_psll_db": row_float(row, "combined_psll_to_weakest_peak_db"),
                        "hfss_peak_min_db": row_float(row, "combined_target_peak_min_db"),
                        "hfss_peak_spread_db": row_float(row, "combined_target_spread_db"),
                        "hfss_iso_nearest_db": row_float(row, "isolation_worst_nearest_db"),
                        "hfss_iso_local_db": row_float(row, "isolation_worst_local_db"),
                    }
                )
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


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    out_dir = args.out_dir or dataset_dir / "stage1_metric_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_npz(dataset_dir / "dataset_arrays.npz")
    splits = load_splits(dataset_dir / "training_split_manifest.json", arrays["k_values"].shape[0])
    base_split_ids = split_ids_from_manifest(splits, arrays["k_values"].shape[0])
    positions = arrays["positions_lambda"].astype(np.float32)
    grid_dirs = make_grid(float(args.theta_step), float(args.phi_step))

    examples: list[dict[str, Any]] = []
    start = time.time()
    n_base = arrays["k_values"].shape[0]
    original_hfss_limit = n_base if int(args.max_original_hfss) <= 0 else min(n_base, int(args.max_original_hfss))
    for idx in range(n_base):
        weights = complex_from_ri(arrays["task_weights_real_imag"][idx])
        af = compute_af_metrics(
            weights=weights,
            targets_deg=arrays["targets_deg"][idx],
            task_valid=arrays["task_valid"][idx],
            positions=positions,
            grid_dirs=grid_dirs,
            sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
            local_radius_deg=float(args.local_radius_deg),
        )
        hfss = {"hfss_psll_db": float("nan"), "hfss_peak_min_db": float("nan"), "hfss_peak_spread_db": float("nan")}
        if not args.skip_original_hfss and idx < original_hfss_limit:
            pattern_path = dataset_dir / "samples" / str(arrays["sample_ids"][idx]) / "hfss_gain_total_theta_phi.csv"
            if pattern_path.exists():
                pattern = read_hfss_pattern(pattern_path)
                hfss = combined_hfss_metrics(
                    pattern,
                    arrays["targets_deg"][idx],
                    arrays["task_valid"][idx],
                    target_radius_deg=float(args.local_radius_deg),
                    sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
                )
        examples.append(
            {
                "source": "original",
                "sample_index": idx,
                "masks": arrays["masks"][idx].astype(np.int8),
                "weights": weights,
                "active_ratio": float(arrays["active_ratios_requested"][idx]),
                "num_active": int(arrays["num_active"][idx]),
                **af,
                **hfss,
                "hfss_iso_nearest_db": float("nan"),
                "hfss_iso_local_db": float("nan"),
            }
        )
        if (idx + 1) % 300 == 0:
            print(f"processed original {idx + 1}/{n_base}, elapsed {time.time() - start:.1f}s")

    validation_examples = collect_validation_examples(dataset_dir)
    print(f"found {len(validation_examples)} task-fullwave validation examples")
    for item in validation_examples:
        idx = int(item["sample_index"])
        af = compute_af_metrics(
            weights=item["weights"],
            targets_deg=arrays["targets_deg"][idx],
            task_valid=arrays["task_valid"][idx],
            positions=positions,
            grid_dirs=grid_dirs,
            sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
            local_radius_deg=float(args.local_radius_deg),
        )
        examples.append({**item, **af})

    n = len(examples)
    masks = np.zeros((n, NUM_ELEMENTS), dtype=np.int8)
    weights_ri = np.zeros((n, NUM_ELEMENTS, KMAX, 2), dtype=np.float32)
    sample_index = np.zeros(n, dtype=np.int64)
    split_id = np.zeros(n, dtype=np.int64)
    active_ratio_values = np.zeros(n, dtype=np.float32)
    num_active_values = np.zeros(n, dtype=np.int64)
    source = np.empty(n, dtype="U96")
    labels = np.full((n, len(LABEL_NAMES)), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(examples):
        idx = int(item["sample_index"])
        sample_index[i] = idx
        split_id[i] = base_split_ids[idx]
        active_ratio_values[i] = float(item.get("active_ratio", float(arrays["active_ratios_requested"][idx])))
        num_active_values[i] = int(item.get("num_active", int(arrays["num_active"][idx])))
        source[i] = str(item["source"])
        masks[i] = item["masks"]
        weights = item["weights"].astype(np.complex64)
        weights_ri[i, ..., 0] = weights.real
        weights_ri[i, ..., 1] = weights.imag
        item["af_gate_pass"] = gate_pass(
            float(item.get("af_psll_db", float("nan"))),
            float(item.get("af_iso_nearest_db", float("nan"))),
            float(item.get("af_iso_local_db", float("nan"))),
            args,
        )
        item["hfss_gate_pass"] = gate_pass(
            float(item.get("hfss_psll_db", float("nan"))),
            float(item.get("hfss_iso_nearest_db", float("nan"))),
            float(item.get("hfss_iso_local_db", float("nan"))),
            args,
        )
        for col, name in enumerate(LABEL_NAMES):
            labels[i, col] = float(item.get(name, float("nan")))
        rows.append(
            {
                "example_id": i,
                "source": str(item["source"]),
                "sample_index": idx,
                "sample_id": str(arrays["sample_ids"][idx]),
                "split_id": int(split_id[i]),
                "k": int(arrays["k_values"][idx]),
                "active_ratio": f"{float(active_ratio_values[i]):.1f}",
                "num_active": int(num_active_values[i]),
                **{name: float(labels[i, col]) for col, name in enumerate(LABEL_NAMES)},
            }
        )

    np.savez_compressed(
        out_dir / "stage1_metric_dataset.npz",
        masks=masks,
        weights_real_imag=weights_ri,
        sample_index=sample_index,
        split_id=split_id,
        source=source,
        labels=labels,
        label_names=np.asarray(LABEL_NAMES),
        k_values=arrays["k_values"][sample_index],
        active_ratios_requested=active_ratio_values,
        num_active=num_active_values,
        targets_deg=arrays["targets_deg"][sample_index],
        task_valid=arrays["task_valid"][sample_index],
        element_ixiy=arrays["element_ixiy"],
        positions_lambda=arrays["positions_lambda"],
    )
    write_csv(out_dir / "stage1_metric_dataset.csv", rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "example_count": int(n),
        "original_count": int(n_base),
        "validation_variant_count": int(len(validation_examples)),
        "hfss_psll_label_count": int(np.isfinite(labels[:, LABEL_NAMES.index("hfss_psll_db")]).sum()),
        "hfss_iso_label_count": int(np.isfinite(labels[:, LABEL_NAMES.index("hfss_iso_nearest_db")]).sum()),
        "af_gate_pass_rate": finite_mean(labels[:, LABEL_NAMES.index("af_gate_pass")]),
        "hfss_gate_pass_rate_labeled": finite_mean(labels[:, LABEL_NAMES.index("hfss_gate_pass")]),
        "label_names": list(LABEL_NAMES),
        "elapsed_s": time.time() - start,
        "outputs": {
            "npz": str(out_dir / "stage1_metric_dataset.npz"),
            "csv": str(out_dir / "stage1_metric_dataset.csv"),
        },
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
