#!/usr/bin/env python3
"""Evaluate all 84 v0.9 held-out candidates after full-wave validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from train_v09_physical_margin_critic import MARGIN_SCALE, binary_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT = ROOT / "hfss_outputs" / "v09_margin_development_dataset_20260726_run01"
DEFAULT_CRITIC = ROOT / "hfss_outputs" / "v09_physical_margin_critic_20260726_run02"
DEFAULT_ADAPTIVE = ROOT / "hfss_outputs" / "v09_adaptive_ratio_eep_loop_20260726_run01"
DEFAULT_SMOKE_DATASET = ROOT / "hfss_outputs" / "v09_hfss_smoke_dataset_20260726_run02"
DEFAULT_SMOKE_HFSS = ROOT / "hfss_outputs" / "v09_hfss_smoke_20260726_run02"
DEFAULT_FULL_DATASET = ROOT / "hfss_outputs" / "v09_fullwave_validation_dataset_20260726_run02"
DEFAULT_FULL_HFSS = ROOT / "hfss_outputs" / "v09_fullwave_validation_20260726_run02"
DEFAULT_ADAPTIVE_DATASET = ROOT / "hfss_outputs" / "v09_adaptive_hfss_confirmation_dataset_20260726_run01"
DEFAULT_ADAPTIVE_HFSS = ROOT / "hfss_outputs" / "v09_adaptive_hfss_confirmation_20260726_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_fullwave_evaluation_20260726_run02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--adaptive-dir", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--smoke-dataset-dir", type=Path, default=DEFAULT_SMOKE_DATASET)
    parser.add_argument("--smoke-hfss-dir", type=Path, default=DEFAULT_SMOKE_HFSS)
    parser.add_argument("--full-dataset-dir", type=Path, default=DEFAULT_FULL_DATASET)
    parser.add_argument("--full-hfss-dir", type=Path, default=DEFAULT_FULL_HFSS)
    parser.add_argument("--adaptive-dataset-dir", type=Path, default=DEFAULT_ADAPTIVE_DATASET)
    parser.add_argument("--adaptive-hfss-dir", type=Path, default=DEFAULT_ADAPTIVE_HFSS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def mapped_labels(dataset_dir: Path, hfss_dir: Path) -> list[dict[str, Any]]:
    with np.load(dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        development_indices = np.asarray(source["development_candidate_indices"], dtype=int)
    labels = read_csv(hfss_dir / "candidate_residual_labels.csv")
    rows: list[dict[str, Any]] = []
    for label in labels:
        local = int(label["candidate_index"])
        rows.append({**label, "development_candidate_index": int(development_indices[local])})
    return rows


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v0.9 full-wave evaluation: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.development_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    labels = mapped_labels(args.smoke_dataset_dir, args.smoke_hfss_dir)
    labels += mapped_labels(args.full_dataset_dir, args.full_hfss_dir)
    by_development = {int(row["development_candidate_index"]): row for row in labels}
    if len(labels) != 84 or len(by_development) != 84:
        raise RuntimeError(f"Expected 84 unique held-out candidates, got {len(labels)}/{len(by_development)}")

    predictions = read_csv(args.critic_dir / "test_predictions.csv")
    prediction_groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        prediction_groups[int(row["candidate_index"])].append(row)
    combined_rows: list[dict[str, Any]] = []
    for candidate in sorted(by_development):
        prediction = prediction_groups[candidate]
        if not prediction:
            raise RuntimeError(f"Missing critic prediction for candidate {candidate}")
        label = by_development[candidate]
        combined_rows.append(
            {
                "development_candidate_index": candidate,
                "sample_index": int(data["sample_index"][candidate]),
                "k_value": int(data["k_values"][candidate]),
                "ratio": float(data["active_ratios_requested"][candidate]),
                "mask_family": str(data["variant_kind"][candidate]),
                "mean_strict_probability": float(
                    np.mean([float(row["prob_strict_gate20"]) for row in prediction])
                ),
                "mean_gate15_probability": float(
                    np.mean([float(row["prob_gate15"]) for row in prediction])
                ),
                "mean_ranking_score": float(
                    np.mean([float(row["ranking_score"]) for row in prediction])
                ),
                "eep_strict_gate": int(data["strict_gate20"][candidate]),
                "hfss_gate15": int(label["gate15"]),
                "hfss_strict_gate20": int(label["strict_gate20"]),
                "hfss_mainlobe_gate": int(label["mainlobe_gate"]),
                "hfss_active_rl_gate": int(label["active_RL_gate"]),
                "hfss_strict_engineering_gate": int(label["strict_engineering_gate"]),
                "hfss_psll_db": float(label["hfss_psll_db"]),
                "hfss_nearest_iso_db": float(label["hfss_nearest_iso_db"]),
                "hfss_local_iso_db": float(label["hfss_local_iso_db"]),
                "hfss_worst_active_rl_db": float(label["all_case_worst_active_rl_db"]),
            }
        )
    write_csv(args.out_dir / "heldout_candidate_evaluation.csv", combined_rows)
    candidate_by_index = {
        int(row["development_candidate_index"]): row for row in combined_rows
    }
    indices = np.asarray(sorted(candidate_by_index), dtype=int)
    probability = np.asarray(
        [candidate_by_index[index]["mean_strict_probability"] for index in indices], dtype=float
    )
    hfss_strict = np.asarray(
        [candidate_by_index[index]["hfss_strict_engineering_gate"] for index in indices], dtype=int
    )
    strict_metrics = binary_metrics(probability, hfss_strict)

    scene_rows: list[dict[str, Any]] = []
    nominal_score = np.min(np.asarray(data["nominal_margins"]) / MARGIN_SCALE[None, :], axis=1)
    for scene in np.unique(data["sample_index"][indices]):
        group = indices[data["sample_index"][indices] == scene]
        critic_choice = max(group.tolist(), key=lambda index: candidate_by_index[index]["mean_ranking_score"])
        fixed = group[np.isclose(data["active_ratios_requested"][group], 0.6)]
        fixed_choice = int(fixed[np.argmax(nominal_score[fixed])])
        oracle = max(
            group.tolist(),
            key=lambda index: candidate_by_index[index]["hfss_strict_engineering_gate"],
        )
        scene_rows.append(
            {
                "sample_index": int(scene),
                "k_value": int(data["k_values"][critic_choice]),
                "critic_candidate_index": critic_choice,
                "critic_ratio": float(data["active_ratios_requested"][critic_choice]),
                "critic_hfss_strict_gate": int(candidate_by_index[critic_choice]["hfss_strict_engineering_gate"]),
                "fixed_ratio06_candidate_index": fixed_choice,
                "fixed_ratio06_hfss_strict_gate": int(candidate_by_index[fixed_choice]["hfss_strict_engineering_gate"]),
                "oracle_hfss_strict_gate": int(candidate_by_index[oracle]["hfss_strict_engineering_gate"]),
            }
        )
    adaptive = [
        row for row in read_csv(args.adaptive_dir / "adaptive_scene_selections.csv")
        if int(row["split_id"]) == 2
    ]
    adaptive_admitted = [row for row in adaptive if int(row["admitted"]) == 1]
    pool_to_development = {
        int(source): local
        for local, source in enumerate(
            np.asarray(data["source_candidate_indices"], dtype=int)
        )
    }
    with np.load(
        args.adaptive_dataset_dir / "dataset_arrays.npz", allow_pickle=False
    ) as source:
        confirmation_pool_indices = np.asarray(
            source["subset_source_candidate_indices"], dtype=int
        )
    confirmation_labels = read_csv(
        args.adaptive_hfss_dir / "candidate_residual_labels.csv"
    )
    confirmation_by_pool = {
        int(pool_index): label
        for pool_index, label in zip(confirmation_pool_indices, confirmation_labels)
    }

    def adaptive_hfss_gate(row: dict[str, str]) -> int:
        pool_index = int(row["selected_candidate_index"])
        if pool_index in pool_to_development:
            development_index = pool_to_development[pool_index]
            return int(
                candidate_by_index[development_index]["hfss_strict_engineering_gate"]
            )
        return int(confirmation_by_pool[pool_index]["strict_engineering_gate"])

    adaptive_hfss = [
        adaptive_hfss_gate(row)
        for row in adaptive_admitted
    ]
    write_csv(args.out_dir / "scene_selection_evaluation.csv", scene_rows)

    group_rows: list[dict[str, Any]] = []
    for k_value in (2, 4, 6):
        for ratio in (0.5, 0.6, 0.7, 0.8):
            group = [
                row for row in combined_rows
                if int(row["k_value"]) == k_value and np.isclose(float(row["ratio"]), ratio)
            ]
            if not group:
                continue
            group_rows.append(
                {
                    "k_value": k_value,
                    "ratio": ratio,
                    "count": len(group),
                    "gate15_rate": float(np.mean([row["hfss_gate15"] for row in group])),
                    "strict_engineering_rate": float(
                        np.mean([row["hfss_strict_engineering_gate"] for row in group])
                    ),
                    "mean_psll_db": float(np.mean([row["hfss_psll_db"] for row in group])),
                    "mean_nearest_iso_db": float(np.mean([row["hfss_nearest_iso_db"] for row in group])),
                    "mean_local_iso_db": float(np.mean([row["hfss_local_iso_db"] for row in group])),
                    "mean_worst_active_rl_db": float(
                        np.mean([row["hfss_worst_active_rl_db"] for row in group])
                    ),
                }
            )
    write_csv(args.out_dir / "fullwave_by_k_ratio.csv", group_rows)
    smoke_summary = json.loads((args.smoke_hfss_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    full_summary = json.loads((args.full_hfss_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    eep_strict = np.asarray([row["eep_strict_gate"] for row in combined_rows], dtype=int)
    full_hfss_strict = np.asarray(
        [row["hfss_strict_engineering_gate"] for row in combined_rows], dtype=int
    )
    acceptance = {
        "all_420_cases_complete": int(smoke_summary["complete_case_count"])
        + int(full_summary["complete_case_count"]) == 420,
        "all_no_scale_reconstruction_pass": bool(smoke_summary["all_no_scale_reconstruction_pass"])
        and bool(full_summary["all_no_scale_reconstruction_pass"]),
        "strict_auroc_ge_0_88": strict_metrics["auroc"] >= 0.88,
        "strict_ece_le_0_08": strict_metrics["ece"] <= 0.08,
        "critic_top1_beats_fixed_ratio": float(
            np.mean([row["critic_hfss_strict_gate"] for row in scene_rows])
        ) > float(np.mean([row["fixed_ratio06_hfss_strict_gate"] for row in scene_rows])),
        "adaptive_admitted_precision_ge_0_80": float(np.mean(adaptive_hfss)) >= 0.80,
        "k6_fullwave_positive_exists": any(
            int(row["k_value"]) == 6 and int(row["hfss_strict_engineering_gate"]) == 1
            for row in combined_rows
        ),
    }
    acceptance["v09_stage_complete"] = bool(all(acceptance.values()))
    summary = {
        "candidate_count": len(combined_rows),
        "independent_scene_count": len(scene_rows),
        "case_count": int(smoke_summary["complete_case_count"]) + int(full_summary["complete_case_count"]),
        "strict_engineering_positive_count": int(np.sum(full_hfss_strict)),
        "eep_hfss_strict_agreement": float(np.mean(eep_strict == full_hfss_strict)),
        "critic_strict_metrics": strict_metrics,
        "critic_top1_hfss_rate": float(np.mean([row["critic_hfss_strict_gate"] for row in scene_rows])),
        "fixed_ratio06_hfss_rate": float(
            np.mean([row["fixed_ratio06_hfss_strict_gate"] for row in scene_rows])
        ),
        "oracle_hfss_rate": float(np.mean([row["oracle_hfss_strict_gate"] for row in scene_rows])),
        "adaptive_admission_rate": len(adaptive_admitted) / max(len(adaptive), 1),
        "adaptive_admitted_hfss_precision": float(np.mean(adaptive_hfss)),
        "complex_nmse_max": max(
            float(smoke_summary["complex_nmse_max"]), float(full_summary["complex_nmse_max"])
        ),
        "magnitude_rmse_db_max": max(
            float(smoke_summary["magnitude_rmse_db_max"]),
            float(full_summary["magnitude_rmse_db_max"]),
        ),
        "acceptance": acceptance,
    }
    (args.out_dir / "v09_fullwave_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
