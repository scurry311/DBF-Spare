"""Create a mixed teacher from guarded and canonical HFSS labels.

The mixed label is canonical by default. For selected low-active K=4/6 groups,
guarded labels are used only when both group-level and per-sample gates pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_COMPARE_CSV = (
    DEFAULT_DATASET_DIR
    / "training_runs"
    / "hardcase_teacher_compare"
    / "hardcase_teacher_compare_k46_active_test.csv"
)
KMAX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--compare-csv", type=Path, default=DEFAULT_COMPARE_CSV)
    parser.add_argument("--teacher-name", default="mixed_guarded_canonical_k46_lowar")
    parser.add_argument("--canonical-name", default="greedy_psll_v2_canonical")
    parser.add_argument("--guarded-name", default="guarded_greedy_v2_k46_ar05_09")
    parser.add_argument("--k-values", default="4,6")
    parser.add_argument("--low-active-ratios", default="0.5,0.6,0.7")
    parser.add_argument("--group-p95-margin-db", type=float, default=0.20)
    parser.add_argument("--group-mean-margin-db", type=float, default=0.15)
    parser.add_argument("--group-isolation-floor-db", type=float, default=28.0)
    parser.add_argument("--group-isolation-margin-db", type=float, default=1.0)
    parser.add_argument("--sample-min-psll-improve-db", type=float, default=0.02)
    parser.add_argument("--sample-max-isolation-drop-db", type=float, default=1.0)
    parser.add_argument("--sample-max-weak-peak-drop-db", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-write-hdf5", dest="write_hdf5", action="store_false")
    parser.set_defaults(write_hdf5=True)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def ratio_key(value: float, allowed: list[float] | None = None) -> str:
    if allowed:
        nearest = min(allowed, key=lambda item: abs(float(value) - float(item)))
        if abs(float(value) - nearest) <= 0.035:
            return f"{nearest:.1f}"
    return f"{float(value):.1f}"


def load_compare_rows(path: Path, active_ratios: list[float]) -> dict[tuple[str, int, str], dict[str, float]]:
    rows: dict[tuple[str, int, str], dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                str(row["teacher"]),
                int(row["k"]),
                ratio_key(float(row["active_ratio"]), active_ratios),
            )
            rows[key] = {
                "mean_worst_psll_db": float(row["mean_worst_psll_db"]),
                "p95_worst_psll_db": float(row["p95_worst_psll_db"]),
                "p05_min_isolation_db": float(row["p05_min_isolation_db"]),
                "mean_task_psll_db": float(row.get("mean_task_psll_db", "nan")),
            }
    return rows


def allowed_group_decisions(
    *,
    compare_rows: dict[tuple[str, int, str], dict[str, float]],
    canonical_name: str,
    guarded_name: str,
    k_values: list[int],
    active_ratios: list[float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for k in k_values:
        for active_ratio in active_ratios:
            ar = f"{float(active_ratio):.1f}"
            guarded = compare_rows.get((guarded_name, int(k), ar))
            canonical = compare_rows.get((canonical_name, int(k), ar))
            if guarded is None or canonical is None:
                decisions.append({"k": k, "active_ratio": ar, "allow_guarded": False, "reason": "missing_compare_row"})
                continue
            p95_ok = guarded["p95_worst_psll_db"] <= canonical["p95_worst_psll_db"] + float(args.group_p95_margin_db)
            mean_ok = guarded["mean_worst_psll_db"] <= canonical["mean_worst_psll_db"] + float(args.group_mean_margin_db)
            iso_floor_ok = guarded["p05_min_isolation_db"] >= float(args.group_isolation_floor_db)
            iso_margin_ok = guarded["p05_min_isolation_db"] >= (
                canonical["p05_min_isolation_db"] - float(args.group_isolation_margin_db)
            )
            allow = bool(p95_ok and mean_ok and iso_floor_ok and iso_margin_ok)
            decisions.append(
                {
                    "k": int(k),
                    "active_ratio": ar,
                    "allow_guarded": allow,
                    "reason": "pass" if allow else "gate_failed",
                    "guarded_mean_worst_psll_db": guarded["mean_worst_psll_db"],
                    "canonical_mean_worst_psll_db": canonical["mean_worst_psll_db"],
                    "guarded_p95_worst_psll_db": guarded["p95_worst_psll_db"],
                    "canonical_p95_worst_psll_db": canonical["p95_worst_psll_db"],
                    "guarded_p05_min_isolation_db": guarded["p05_min_isolation_db"],
                    "canonical_p05_min_isolation_db": canonical["p05_min_isolation_db"],
                    "p95_ok": p95_ok,
                    "mean_ok": mean_ok,
                    "iso_floor_ok": iso_floor_ok,
                    "iso_margin_ok": iso_margin_ok,
                }
            )
    return decisions


def decode_status(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_diagnostics(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def sample_guard_ok(diagnostics: dict[str, Any], status: str, args: argparse.Namespace) -> bool:
    if not status.endswith("_accepted"):
        return False
    if "accepted" in diagnostics and not bool(diagnostics["accepted"]):
        return False
    delta_psll = float(diagnostics.get("delta_psll_to_weakest_peak_db", 0.0))
    delta_iso = float(diagnostics.get("delta_isolation_min_db", 0.0))
    delta_weak = float(diagnostics.get("delta_weak_peak_db", 0.0))
    active_delta = int(diagnostics.get("active_count_delta", 0))
    return (
        active_delta == 0
        and delta_psll <= -float(args.sample_min_psll_improve_db)
        and delta_iso >= -float(args.sample_max_isolation_drop_db)
        and delta_weak >= -float(args.sample_max_weak_peak_drop_db)
    )


def copy_label_payload(group: h5py.Group) -> dict[str, np.ndarray]:
    return {
        "weights_real_imag": group["weights_real_imag"][...],
        "activation": group["activation"][...],
        "assignment": group["assignment"][...],
        "status": group["status"][...],
        "solve_time_ms": group["solve_time_ms"][...],
        "e2e_time_ms": group["e2e_time_ms"][...],
        "objective": group["objective"][...],
        "iterations": group["iterations"][...],
        "task_count": group["task_count"][...],
        "scenario_index": group["scenario_index"][...],
        "diagnostics_json": group["diagnostics_json"][...],
    }


def write_hdf5_label(
    *,
    dataset_path: Path,
    teacher_name: str,
    payload: dict[str, np.ndarray],
    diagnostics: list[dict[str, Any]],
    overwrite: bool,
) -> str:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    diagnostics_json = np.asarray(
        [json.dumps(item, ensure_ascii=False, allow_nan=True) for item in diagnostics],
        dtype=object,
    )
    status = np.asarray([str(item["status"]) for item in diagnostics], dtype=object)
    with h5py.File(dataset_path, "a") as handle:
        labels = handle.require_group("labels")
        if teacher_name in labels:
            if not overwrite:
                raise RuntimeError(f"labels/{teacher_name} exists; pass --overwrite to replace it.")
            del labels[teacher_name]
        group = labels.create_group(teacher_name)
        for key in (
            "weights_real_imag",
            "activation",
            "assignment",
            "solve_time_ms",
            "e2e_time_ms",
            "objective",
            "iterations",
            "task_count",
            "scenario_index",
        ):
            group.create_dataset(key, data=payload[key])
        group.create_dataset("status", data=status, dtype=string_dtype)
        group.create_dataset("diagnostics_json", data=diagnostics_json, dtype=string_dtype)
    return f"{dataset_path}:/labels/{teacher_name}"


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


def summarize_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = [("all", "all")]
    for k in sorted({int(row["k"]) for row in rows}):
        keys.append((str(k), "all"))
        for ar in sorted({row["active_ratio"] for row in rows if int(row["k"]) == k}):
            keys.append((str(k), ar))
    for k_label, ar_label in keys:
        group = rows
        if k_label != "all":
            group = [row for row in group if int(row["k"]) == int(k_label)]
        if ar_label != "all":
            group = [row for row in group if row["active_ratio"] == ar_label]
        if not group:
            continue
        out.append(
            {
                "k": k_label,
                "active_ratio": ar_label,
                "n": len(group),
                "guarded_count": sum(1 for row in group if row["source_teacher"] == "guarded"),
                "canonical_count": sum(1 for row in group if row["source_teacher"] == "canonical"),
                "guarded_rate": sum(1 for row in group if row["source_teacher"] == "guarded") / len(group),
                "group_allowed_rate": sum(1 for row in group if bool(row["group_allowed"])) / len(group),
                "sample_guard_pass_rate": sum(1 for row in group if bool(row["sample_guard_ok"])) / len(group),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    dataset_path = dataset_dir / "training_dataset.h5"
    out_dir = dataset_dir / "optimized_teachers" / str(args.teacher_name)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_int_list(args.k_values)
    low_active_ratios = parse_float_list(args.low_active_ratios)
    arrays = np.load(dataset_dir / "dataset_arrays.npz", allow_pickle=False)
    compare_rows = load_compare_rows(Path(args.compare_csv), low_active_ratios + [0.8, 0.9, 1.0])
    group_decisions = allowed_group_decisions(
        compare_rows=compare_rows,
        canonical_name=str(args.canonical_name),
        guarded_name=str(args.guarded_name),
        k_values=k_values,
        active_ratios=low_active_ratios,
        args=args,
    )
    allowed_groups = {
        (int(row["k"]), str(row["active_ratio"]))
        for row in group_decisions
        if bool(row["allow_guarded"])
    }

    with h5py.File(dataset_path, "r") as handle:
        canonical = copy_label_payload(handle["labels"][str(args.canonical_name)])
        guarded = copy_label_payload(handle["labels"][str(args.guarded_name)])

    mixed = {key: np.array(value, copy=True) for key, value in canonical.items()}
    diagnostics: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    changed = 0
    for index in range(int(arrays["k_values"].shape[0])):
        k = int(arrays["k_values"][index])
        ar = ratio_key(float(arrays["active_ratios_requested"][index]), low_active_ratios + [0.8, 0.9, 1.0])
        candidate_group = k in set(k_values) and ar in {f"{x:.1f}" for x in low_active_ratios}
        group_allowed = (k, ar) in allowed_groups
        status = decode_status(guarded["status"][index])
        guarded_diag = parse_diagnostics(guarded["diagnostics_json"][index])
        per_sample_ok = sample_guard_ok(guarded_diag, status, args)
        use_guarded = bool(candidate_group and group_allowed and per_sample_ok)
        source = "guarded" if use_guarded else "canonical"
        if use_guarded:
            changed += 1
            for key in (
                "weights_real_imag",
                "activation",
                "assignment",
                "solve_time_ms",
                "e2e_time_ms",
                "objective",
                "iterations",
                "task_count",
                "scenario_index",
            ):
                mixed[key][index] = guarded[key][index]
        diag = {
            "teacher": str(args.teacher_name),
            "source_teacher": source,
            "canonical_name": str(args.canonical_name),
            "guarded_name": str(args.guarded_name),
            "sample_index": int(index),
            "k": k,
            "active_ratio": ar,
            "candidate_group": bool(candidate_group),
            "group_allowed": bool(group_allowed),
            "sample_guard_ok": bool(per_sample_ok),
            "guarded_status": status,
            "guarded_delta_psll_to_weakest_peak_db": guarded_diag.get("delta_psll_to_weakest_peak_db"),
            "guarded_delta_isolation_min_db": guarded_diag.get("delta_isolation_min_db"),
            "guarded_delta_weak_peak_db": guarded_diag.get("delta_weak_peak_db"),
            "status": f"mixed_{source}",
        }
        diagnostics.append(diag)
        if candidate_group:
            selection_rows.append(diag)

    hdf5_label = None
    if bool(args.write_hdf5):
        hdf5_label = write_hdf5_label(
            dataset_path=dataset_path,
            teacher_name=str(args.teacher_name),
            payload=mixed,
            diagnostics=diagnostics,
            overwrite=bool(args.overwrite),
        )

    compatible = {key: arrays[key] for key in arrays.files}
    compatible["masks"] = mixed["activation"].astype(np.int8)
    compatible["task_weights_real_imag"] = mixed["weights_real_imag"]
    np.savez_compressed(out_dir / "dataset_arrays.npz", **compatible)
    shutil.copy2(dataset_dir / "training_split_manifest.json", out_dir / "training_split_manifest.json")
    write_csv(out_dir / "group_gate_decisions.csv", group_decisions)
    write_csv(out_dir / "selection_rows.csv", selection_rows)
    selection_summary = summarize_selection(selection_rows)
    write_csv(out_dir / "selection_summary_by_k_active.csv", selection_summary)

    active_delta = mixed["activation"].sum(axis=1) - canonical["activation"].sum(axis=1)
    run_summary = {
        "teacher_name": str(args.teacher_name),
        "dataset_dir": str(dataset_dir),
        "hdf5_label": hdf5_label,
        "changed_samples": int(changed),
        "candidate_samples": int(len(selection_rows)),
        "max_abs_active_count_delta_vs_canonical": float(np.max(np.abs(active_delta))),
        "allowed_groups": sorted([{"k": k, "active_ratio": ar} for k, ar in allowed_groups], key=lambda row: (row["k"], row["active_ratio"])),
        "group_gate_decisions": group_decisions,
        "outputs": {
            "dataset_arrays": str(out_dir / "dataset_arrays.npz"),
            "group_gate_decisions": str(out_dir / "group_gate_decisions.csv"),
            "selection_rows": str(out_dir / "selection_rows.csv"),
            "selection_summary": str(out_dir / "selection_summary_by_k_active.csv"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
