#!/usr/bin/env python3
"""Audit active-return definitions on the trusted 96-candidate EEP dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
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
DEFAULT_EXCITATIONS = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "eep_hfss_validation"
    / "case_excitations.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "trusted_active_rl_audit_20260724_run02"
RELATIVE_THRESHOLDS_DB: tuple[float | None, ...] = (None, -40.0, -30.0, -20.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--excitations", type=Path, default=DEFAULT_EXCITATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--return-loss-min-db", type=float, default=10.0)
    return parser.parse_args()


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


def threshold_name(relative_db: float | None) -> str:
    return "strict_mask" if relative_db is None else f"rel{abs(int(relative_db))}db"


def port_class(ix: int, iy: int) -> str:
    x_edge = ix in (0, 15)
    y_edge = iy in (0, 15)
    if x_edge and y_edge:
        return "corner"
    if x_edge or y_edge:
        return "edge"
    return "interior"


def normalized_external(weight: np.ndarray, mask: np.ndarray) -> np.ndarray:
    excitation = np.conjugate(np.asarray(weight, dtype=np.complex128))
    excitation = np.where(mask, excitation, 0.0)
    norm = float(np.linalg.norm(excitation))
    if norm <= 1.0e-14:
        raise ValueError("Zero external excitation")
    return excitation / norm


def active_metrics(
    s_matrix: np.ndarray,
    excitation: np.ndarray,
    command_mask: np.ndarray,
    relative_db: float | None,
    element_ixiy: np.ndarray,
    rl_min_db: float,
) -> dict[str, Any]:
    reflected = s_matrix @ excitation
    amplitude = np.abs(excitation)
    maximum = max(float(np.max(amplitude)), 1.0e-30)
    if relative_db is None:
        driven = command_mask & (amplitude > 1.0e-10)
    else:
        driven = command_mask & (amplitude >= maximum * 10.0 ** (relative_db / 20.0))
    if not np.any(driven):
        raise ValueError("No driven ports under active-return definition")
    indices = np.flatnonzero(driven)
    gamma = np.abs(reflected[indices] / excitation[indices])
    worst_pos = int(np.argmax(gamma))
    worst_index = int(indices[worst_pos])
    worst_rl = float(-20.0 * np.log10(max(float(gamma[worst_pos]), 1.0e-30)))
    incident_power = max(float(np.vdot(excitation, excitation).real), 1.0e-30)
    reflected_power = float(np.vdot(reflected, reflected).real)
    total_rl = float(-10.0 * np.log10(max(reflected_power / incident_power, 1.0e-30)))
    ix, iy = (int(value) for value in element_ixiy[worst_index])
    return {
        "definition": threshold_name(relative_db),
        "relative_threshold_db": "strict" if relative_db is None else relative_db,
        "evaluated_port_count": int(indices.size),
        "worst_active_rl_db": worst_rl,
        "total_rl_db": total_rl,
        "active_gate_pass": int(worst_rl >= rl_min_db and total_rl >= rl_min_db),
        "worst_port_index": worst_index,
        "worst_port_ix": ix,
        "worst_port_iy": iy,
        "worst_port_class": port_class(ix, iy),
        "worst_port_amplitude_relative_db": float(
            20.0 * np.log10(max(amplitude[worst_index] / maximum, 1.0e-30))
        ),
        "worst_port_gamma_magnitude": float(gamma[worst_pos]),
    }


def grouped_summary(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        keys = (
            ("all", "all", "all"),
            (str(row["k"]), "all", "all"),
            (str(row["k"]), f"{float(row['ratio']):.1f}", str(row["large_scan"])),
        )
        for key in keys:
            groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (k_value, ratio, large_scan), rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "k": k_value,
            "ratio": ratio,
            "large_scan": large_scan,
            "candidate_count": len(rows),
        }
        for definition in ("strict_mask", "rel40db", "rel30db", "rel20db"):
            prefix = f"combined_{definition}"
            values = np.asarray([float(row[f"{prefix}_worst_active_rl_db"]) for row in rows])
            record[f"{prefix}_worst_min_db"] = float(np.min(values))
            record[f"{prefix}_worst_mean_db"] = float(np.mean(values))
            record[f"{prefix}_worst_max_db"] = float(np.max(values))
            record[f"{prefix}_gate_rate"] = float(
                np.mean([int(row[f"{prefix}_gate_pass"]) for row in rows])
            )
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite active-RL audit: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    with np.load(args.excitations, allow_pickle=False) as source:
        s_matrix = np.asarray(source["matched_s"], dtype=np.complex128)

    masks = np.asarray(data["mask"], dtype=bool)
    task_weights = data["w_tasks_real_imag"][..., 0] + 1j * data["w_tasks_real_imag"][..., 1]
    combined_weights = data["w_combined_real_imag"][..., 0] + 1j * data["w_combined_real_imag"][..., 1]
    element_ixiy = np.asarray(data["element_ixiy"], dtype=int)
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for candidate in range(masks.shape[0]):
        mask = masks[candidate]
        k_value = int(data["k_values"][candidate])
        base = {
            "candidate_index": int(data["candidate_index"][candidate]),
            "sample_index": int(data["sample_index"][candidate]),
            "sample_id": str(data["sample_ids"][candidate]),
            "k": k_value,
            "ratio": float(data["active_ratios_requested"][candidate]),
            "large_scan": int(data["large_scan"][candidate]),
            "min_target_separation_deg": float(data["min_target_separation_deg"][candidate]),
        }
        combined = normalized_external(combined_weights[candidate], mask)
        combined_by_definition: dict[str, dict[str, Any]] = {}
        task_by_definition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relative_db in RELATIVE_THRESHOLDS_DB:
            metrics = active_metrics(
                s_matrix, combined, mask, relative_db, element_ixiy, float(args.return_loss_min_db)
            )
            combined_by_definition[metrics["definition"]] = metrics
            case_rows.append({**base, "case_kind": "combined", "task_index": -1, **metrics})
        for task_index in range(k_value):
            task = normalized_external(task_weights[candidate, :, task_index], mask)
            for relative_db in RELATIVE_THRESHOLDS_DB:
                metrics = active_metrics(
                    s_matrix, task, mask, relative_db, element_ixiy, float(args.return_loss_min_db)
                )
                task_by_definition[metrics["definition"]].append(metrics)
                case_rows.append(
                    {**base, "case_kind": "task", "task_index": task_index, **metrics}
                )
        row = dict(base)
        for definition, metrics in combined_by_definition.items():
            prefix = f"combined_{definition}"
            for key in (
                "evaluated_port_count",
                "worst_active_rl_db",
                "total_rl_db",
                "active_gate_pass",
                "worst_port_index",
                "worst_port_ix",
                "worst_port_iy",
                "worst_port_class",
                "worst_port_amplitude_relative_db",
            ):
                output_key = "gate_pass" if key == "active_gate_pass" else key
                row[f"{prefix}_{output_key}"] = metrics[key]
        for definition, members in task_by_definition.items():
            prefix = f"all_tasks_{definition}"
            row[f"{prefix}_worst_active_rl_db"] = min(
                float(item["worst_active_rl_db"]) for item in members
            )
            row[f"{prefix}_worst_total_rl_db"] = min(float(item["total_rl_db"]) for item in members)
            row[f"{prefix}_gate_pass"] = int(all(int(item["active_gate_pass"]) for item in members))
        candidate_rows.append(row)

    write_csv(args.out_dir / "active_rl_case_audit.csv", case_rows)
    write_csv(args.out_dir / "active_rl_candidate_audit.csv", candidate_rows)
    write_csv(args.out_dir / "active_rl_group_summary.csv", grouped_summary(candidate_rows))

    summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(args.dataset.resolve()),
        "matched_s_source": str(args.excitations.resolve()),
        "candidate_count": len(candidate_rows),
        "case_count": len(case_rows),
        "return_loss_min_db": float(args.return_loss_min_db),
        "definitions": {},
        "worst_port_class_counts_combined_strict": dict(
            Counter(str(row["combined_strict_mask_worst_port_class"]) for row in candidate_rows)
        ),
    }
    for definition in ("strict_mask", "rel40db", "rel30db", "rel20db"):
        prefix = f"combined_{definition}"
        combined_values = np.asarray(
            [float(row[f"{prefix}_worst_active_rl_db"]) for row in candidate_rows]
        )
        summary["definitions"][definition] = {
            "combined_gate_count": int(sum(int(row[f"{prefix}_gate_pass"]) for row in candidate_rows)),
            "combined_gate_rate": float(np.mean([int(row[f"{prefix}_gate_pass"]) for row in candidate_rows])),
            "combined_worst_active_rl_min_db": float(np.min(combined_values)),
            "combined_worst_active_rl_mean_db": float(np.mean(combined_values)),
            "combined_worst_active_rl_max_db": float(np.max(combined_values)),
            "all_tasks_gate_count": int(
                sum(int(row[f"all_tasks_{definition}_gate_pass"]) for row in candidate_rows)
            ),
            "all_tasks_gate_rate": float(
                np.mean([int(row[f"all_tasks_{definition}_gate_pass"]) for row in candidate_rows])
            ),
        }
    (args.out_dir / "active_rl_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
