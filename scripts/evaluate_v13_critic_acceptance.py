#!/usr/bin/env python3
"""Compare the v1.3 nominal critic with the actual-weight EEP physics gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--critic-dataset-dir", type=Path, required=True)
    parser.add_argument("--critic-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite critic acceptance: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate = load(args.candidate_dir / "dataset_arrays.npz")
    fullwave = load(args.critic_dataset_dir / "dataset_arrays.npz")
    training = json.loads((args.critic_dir / "training_summary.json").read_text(encoding="utf-8"))
    eep_margin = np.asarray(candidate["actual_margins"], dtype=float)
    hfss_margin = np.asarray(fullwave["actual_margins"], dtype=float)
    strict = np.asarray(fullwave["strict_gate20"], dtype=int)
    split_id = np.asarray(fullwave["split_id"], dtype=int)
    test = split_id == 2
    eep_score = np.min(eep_margin, axis=1)
    eep_gate = np.all(eep_margin >= 0.0, axis=1).astype(int)
    delta = hfss_margin - eep_margin
    aggregate = training["five_seed_aggregate"]
    critic_metrics = {
        "strict_auroc_mean": float(aggregate["test_strict_gate20_auroc"]["mean"]),
        "strict_auroc_ci95_half_width": float(
            aggregate["test_strict_gate20_auroc"]["ci95_half_width"]
        ),
        "strict_ece_mean": float(aggregate["test_strict_gate20_ece"]["mean"]),
        "strict_precision_mean": float(
            aggregate["test_strict_gate20_precision"]["mean"]
        ),
        "strict_recall_mean": float(aggregate["test_strict_gate20_recall"]["mean"]),
        "top1_strict_rate_mean": float(aggregate["top1_strict_rate"]["mean"]),
        "test_oracle_strict_rate": float(aggregate["oracle_strict_rate"]["mean"]),
    }
    acceptance = {
        "strict_auroc_ge_0_88": critic_metrics["strict_auroc_mean"] >= 0.88,
        "strict_ece_le_0_08": critic_metrics["strict_ece_mean"] <= 0.08,
        "strict_precision_ge_0_90": critic_metrics["strict_precision_mean"] >= 0.90,
        "top1_ge_0_80": critic_metrics["top1_strict_rate_mean"] >= 0.80,
    }
    summary = {
        "candidate_count": int(strict.size),
        "independent_scene_count": int(np.unique(fullwave["sample_index"]).size),
        "test_candidate_count": int(np.sum(test)),
        "test_scene_count": int(np.unique(fullwave["sample_index"][test]).size),
        "critic_metrics": critic_metrics,
        "critic_acceptance": acceptance,
        "critic_promoted": bool(all(acceptance.values())),
        "actual_weight_eep_physics_gate": {
            "all_candidate_agreement": float(np.mean(eep_gate == strict)),
            "all_candidate_precision": float(
                np.sum((eep_gate == 1) & (strict == 1)) / max(np.sum(eep_gate == 1), 1)
            ),
            "all_candidate_recall": float(
                np.sum((eep_gate == 1) & (strict == 1)) / max(np.sum(strict == 1), 1)
            ),
            "test_auroc": float(roc_auc_score(strict[test], eep_score[test])),
            "test_auprc": float(average_precision_score(strict[test], eep_score[test])),
            "test_agreement": float(np.mean(eep_gate[test] == strict[test])),
        },
        "hfss_minus_actual_weight_eep_margin": {
            "mean_db": np.mean(delta, axis=0).tolist(),
            "std_db": np.std(delta, axis=0).tolist(),
            "max_abs_db": np.max(np.abs(delta), axis=0).tolist(),
            "margin_names": fullwave["margin_names"].astype(str).tolist(),
        },
        "decision": (
            "Do not promote the nominal-command residual critic and do not start adaptive-ratio "
            "prospective HFSS. Use actual implementation weights in EEP/S256 for deterministic "
            "physics gating. Collect geometry, frequency, calibration, S-parameter, and hardware "
            "drift labels before attempting a non-trivial residual critic."
        ),
        "adaptive_ratio_allowed": False,
        "final_prospective_hfss_allowed": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
