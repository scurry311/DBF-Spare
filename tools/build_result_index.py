#!/usr/bin/env python3
"""Create a compact, checksummed baseline from selected local result files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Artifact:
    name: str
    source: str
    level: str
    status: str
    note: str


LEGACY_ARTIFACTS = (
    Artifact("requirements", "docs/EXPERIMENT_REQUIREMENTS_AND_REBUILD_DECISION_20260717.md", "A", "policy", "Frozen research gates and rebuild decision."),
    Artifact("patch_1x1", "hfss_outputs/grounded_patch_direct_1x1_20260717_run01/stage_summary.json", "A", "passed", "Converged matched 1x1 HFSS gate."),
    Artifact("patch_4x4", "hfss_outputs/grounded_patch_direct_4x4_20260717_run01/stage_summary.json", "A", "passed", "Converged matched 4x4 HFSS gate."),
    Artifact("patch_8x8", "hfss_outputs/grounded_patch_direct_8x8_eep_gate_20260717_run02/stage_summary.json", "A", "passed", "Converged matched 8x8 HFSS gate."),
    Artifact("eep_operator", "hfss_outputs/grounded_patch_eep256_resource_smoke_20260717_run01/operator_analysis_summary.json", "C", "pipeline_only", "256-port EEP structural audit."),
    Artifact("eep_superposition", "hfss_outputs/grounded_patch_eep256_resource_smoke_20260717_run01/superposition_analysis_summary.json", "C", "pipeline_only", "Three-case direct superposition validation."),
    Artifact("joint_smoke", "hfss_outputs/grounded_patch_eep_hfss_joint_smoke_stage_20260717_run01/stage_decision.json", "B", "blocked", "EEP/HFSS smoke stage decision."),
    Artifact("label_decision", "hfss_outputs/grounded_patch_eep_hfss_joint_smoke_stage_20260717_run01/hfss_training_label_decision.json", "B", "locked", "Authoritative full-wave label decision."),
    Artifact("task_lcmv", "hfss_outputs/grounded_patch_task_lcmv_psll_20260717_run01/summary.json", "D", "proxy_only", "AF plus synthesized S256 task optimization."),
    Artifact("s256_proxy", "hfss_outputs/grounded_patch_s256_proxy_20260717_run02/summary.json", "D", "proxy_only", "Local-kernel S256 proxy audit."),
    Artifact("active_projection", "hfss_outputs/grounded_patch_active_rl_projection_20260717_run04_prune12_strong/active_return_projection_summary.json", "D", "proxy_only", "Active-return projection results."),
    Artifact("dataset", "hfss_outputs/multitask_dataset/stage1_fullwave_residual_dataset_v2_20260714/build_summary.json", "B", "legacy_baseline", "Scene-grouped residual dataset summary."),
    Artifact("critic", "hfss_outputs/multitask_dataset/training_runs/fullwave_residual_critic_v2_20260714/five_seed_summary.json", "B", "pretraining_only", "Five-seed residual critic metrics."),
    Artifact("stage1_acceptance", "hfss_outputs/multitask_dataset/stage1_acceptance_v2_20260714/stage1_acceptance_summary.json", "B", "failed", "Stage-1 and Stage-2 acceptance decision."),
    Artifact("ddm_pass1_metrics", "hfss_outputs/grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02/stages/pass01/stage_metrics.json", "B", "blocked", "Numerically valid 16x16 DDM pass1."),
    Artifact("ddm_pass1_profile", "hfss_outputs/grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02/stages/pass01/DV1332_S2193_V0.profile", "B", "blocked", "HFSS pass1 profile."),
    Artifact("ddm_pass1_s256", "hfss_outputs/grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02/stages/pass01/grounded_patch_16x16_ddm_pass01.s256p", "B", "blocked", "Physical pass1 S256."),
    Artifact("ddm_pass2_metrics", "hfss_outputs/grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02/stages/pass02/stage_metrics.json", "B", "blocked", "Pass2 Delta S and physical gates."),
    Artifact("ddm_pass2_profile", "hfss_outputs/grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02/stages/pass02/DV1332_S2193_V2409.profile", "B", "blocked", "HFSS pass2 profile."),
    Artifact("ddm_pass2_s256", "hfss_outputs/grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02/stages/pass02/grounded_patch_16x16_ddm_pass02.s256p", "B", "blocked", "Physical pass2 S256."),
)


TRUSTED_EEP_ARTIFACTS = (
    Artifact("fieldsolve_validation", "hfss_outputs/fixed_mesh_eep_fieldsolve_20260723_run05_ddm80/solve/fieldsolve_validation.json", "A", "passed", "Field-enabled fixed-mesh DDM solve cross-checked against the trusted direct S256."),
    Artifact("eep_operator", "hfss_outputs/fixed_mesh_eep256_20260723_run05/operator_analysis_summary.json", "A", "passed", "Complete 256-port complex Etheta/Ephi EEP operator audit."),
    Artifact("eep_superposition", "hfss_outputs/fixed_mesh_eep256_20260723_run05/superposition_analysis_summary.json", "A", "passed", "Initial three-case no-scale complex-field superposition smoke."),
    Artifact("candidate_prepare", "hfss_outputs/trusted_eep_residual_20260723_run02/validation_dataset/prepare_summary.json", "A", "passed", "New-label candidate distribution and scene semantics."),
    Artifact("reconstruction_validation", "hfss_outputs/trusted_eep_residual_20260723_run02/eep_hfss_validation/analysis_summary.json", "A", "pattern_labels_only", "Ninety-six candidates and 474 no-scale EEP/HFSS task cases."),
    Artifact("grouped_split", "hfss_outputs/trusted_eep_residual_20260723_run02/eep_hfss_validation/grouped_split_manifest.json", "A", "passed", "Leakage-free sample_index grouped train/validation/test split."),
    Artifact("residual_labels", "hfss_outputs/trusted_eep_residual_20260723_run02/eep_hfss_validation/candidate_residual_labels.csv", "A", "pattern_labels_only", "Per-candidate EEP/HFSS residuals, gates, and S256 matching metrics."),
    Artifact("dataset_package", "hfss_outputs/trusted_eep_residual_20260723_run02/dataset_v2_20260724/package_summary.json", "A", "passed", "Canonical non-destructive dataset package audit."),
    Artifact("dataset_schema", "hfss_outputs/trusted_eep_residual_20260723_run02/dataset_v2_20260724/dataset_schema.json", "A", "passed", "Names, shapes, and dtypes for every packaged array."),
    Artifact("residual_dataset", "hfss_outputs/trusted_eep_residual_20260723_run02/dataset_v2_20260724/residual_critic_dataset_v2.npz", "A", "pattern_labels_only", "Compact arrays containing canonical scene IDs, masks, task weights, combined weights, metrics, gates, and splits."),
    Artifact("group_summary", "hfss_outputs/trusted_eep_residual_20260723_run02/eep_hfss_validation/fullwave_summary_by_k_ratio_scan.csv", "A", "diagnostic", "K, ratio, and scan-angle grouped full-wave statistics."),
    Artifact("critic_decision", "hfss_outputs/trusted_eep_residual_20260723_run02/eep_hfss_validation/critic_training_decision.json", "A", "held", "Residual-critic readiness decision based on residual scale and label support."),
    Artifact("critic_priority", "hfss_outputs/trusted_eep_residual_20260723_run02/eep_hfss_validation/critic_priority_candidates.csv", "A", "diagnostic", "Near-boundary and hard-positive candidate priority list."),
    Artifact("null_metrics", "hfss_outputs/trusted_eep_residual_20260723_run02/eep_hfss_validation/null_residual_baseline_metrics.csv", "A", "baseline_only", "Zero/train-mean residual baseline metrics."),
    Artifact("null_checkpoint", "hfss_outputs/trusted_eep_residual_20260723_run02/eep_hfss_validation/null_residual_critic_checkpoint.npz", "A", "baseline_only", "Reproducible null checkpoint; not an engineering feasibility critic."),
)


ACTIVE_RL_JOINT_ARTIFACTS = (
    Artifact("active_rl_audit", "hfss_outputs/trusted_active_rl_audit_20260724_run02/active_rl_audit_summary.json", "A", "passed", "Combined/task and amplitude-threshold active-RL semantics audit on 96 trusted candidates."),
    Artifact("active_rl_groups", "hfss_outputs/trusted_active_rl_audit_20260724_run02/active_rl_group_summary.csv", "A", "diagnostic", "Active-RL audit grouped by K, ratio, and scan class."),
    Artifact("joint_optimization_summary", "hfss_outputs/trusted_eep_s256_joint_optimization_20260724_run03/optimization_summary.json", "A", "passed", "Full 96-candidate trusted EEP/S256 joint projection summary."),
    Artifact("joint_candidate_metrics", "hfss_outputs/trusted_eep_s256_joint_optimization_20260724_run03/optimization_candidate_metrics.csv", "A", "diagnostic", "Paired baseline/optimized pattern, matching, gain, and gate metrics."),
    Artifact("joint_trials", "hfss_outputs/trusted_eep_s256_joint_optimization_20260724_run03/optimization_trials.csv", "A", "diagnostic", "Three projection configurations per candidate."),
    Artifact("joint_weights", "hfss_outputs/trusted_eep_s256_joint_optimization_20260724_run03/optimized_task_weights.npz", "A", "eep_only", "Task-level and combined external weights with w=sum(w_k)."),
    Artifact("smoke_decision", "hfss_outputs/trusted_eep_s256_joint_smoke_decision_20260724_run01/hfss_label_generation_decision.json", "A", "hfss_locked", "Authoritative decision after the 96-candidate optimization smoke."),
    Artifact("smoke_groups", "hfss_outputs/trusted_eep_s256_joint_smoke_decision_20260724_run01/smoke_summary_by_k_ratio_scan.csv", "A", "diagnostic", "Joint-gate statistics by K, ratio, and scan class."),
    Artifact("scene_oracle", "hfss_outputs/trusted_eep_s256_joint_smoke_decision_20260724_run01/scene_minimum_ratio_oracle.csv", "A", "diagnostic", "Scene-level minimum sparse-ratio oracle."),
    Artifact("next_priority", "hfss_outputs/trusted_eep_s256_joint_smoke_decision_20260724_run01/next_optimization_priority_candidates.csv", "A", "diagnostic", "Prioritized sparse positives and multibeam direction failures for dense regional projection."),
)


DENSE_LOCAL_HFSS_ARTIFACTS = (
    Artifact("dense_summary", "hfss_outputs/trusted_dense_local_eep_joint_20260724_run02/dense_refinement_summary.json", "A", "passed", "Dense local-5-degree EEP projection summary for all 96 candidates."),
    Artifact("dense_candidate_metrics", "hfss_outputs/trusted_dense_local_eep_joint_20260724_run02/dense_refinement_candidate_metrics.csv", "A", "diagnostic", "Per-candidate dense EEP, active-RL, mainlobe, and strict-gate metrics."),
    Artifact("dense_trials", "hfss_outputs/trusted_dense_local_eep_joint_20260724_run02/dense_refinement_trials.csv", "A", "diagnostic", "Paired preserve-target and equalized-target dense projection trials."),
    Artifact("dense_weights", "hfss_outputs/trusted_dense_local_eep_joint_20260724_run02/dense_refined_task_weights.npz", "A", "eep_hfss_validated", "Task-level dense-local weights with exact combined-weight reconstruction."),
    Artifact("shortlist_prepare", "hfss_outputs/trusted_dense_joint_hfss_dataset_20260724_run01/prepare_summary.json", "A", "passed", "Gated 15-candidate and 65-case sparse multibeam HFSS shortlist."),
    Artifact("shortlist_manifest", "hfss_outputs/trusted_dense_joint_hfss_dataset_20260724_run01/candidate_manifest.csv", "A", "passed", "Traceable mapping from local HFSS candidates to the 96-candidate dense smoke."),
    Artifact("hfss_analysis", "hfss_outputs/trusted_dense_joint_hfss_smoke_20260724_run01/analysis_summary.json", "A", "passed", "Complete no-scale EEP/HFSS analysis for all 65 combined and task cases."),
    Artifact("hfss_case_metrics", "hfss_outputs/trusted_dense_joint_hfss_smoke_20260724_run01/case_reconstruction_metrics.csv", "A", "diagnostic", "Per-case complex-field and magnitude reconstruction errors."),
    Artifact("hfss_candidate_labels", "hfss_outputs/trusted_dense_joint_hfss_smoke_20260724_run01/candidate_residual_labels.csv", "A", "positive_labels", "Fifteen trusted sparse multibeam positive labels; no ratio-1 control is included."),
    Artifact("hfss_decision", "hfss_outputs/trusted_dense_joint_hfss_decision_20260724_run01/dense_joint_hfss_smoke_decision.json", "A", "labels_allowed_critic_held", "Experiment-specific label and residual-critic decision without the legacy 96-row assumption."),
    Artifact("hfss_groups", "hfss_outputs/trusted_dense_joint_hfss_decision_20260724_run01/dense_joint_hfss_group_summary.csv", "A", "diagnostic", "HFSS strict-gate and residual statistics grouped by K, ratio, and scan class."),
    Artifact("mapping_smoke", "hfss_outputs/trusted_dense_joint_hfss_mapping_smoke_20260724_run01/analysis_summary.json", "A", "passed", "Seven-case K=6 mapping smoke run before the 65-case batch."),
)


BASELINE_CONFIGS = {
    "2026-07-21": {
        "version": "v0.1.0-physics-gated",
        "artifacts": LEGACY_ARTIFACTS,
        "training_labels_locked": True,
        "pattern_labels_allowed": False,
        "strict_benchmark_gate_pass": False,
    },
    "2026-07-24": {
        "version": "v0.2.0-trusted-eep",
        "artifacts": TRUSTED_EEP_ARTIFACTS,
        "training_labels_locked": True,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": False,
    },
    "2026-07-24-active-rl-joint": {
        "version": "v0.3.0-active-rl-joint",
        "artifacts": ACTIVE_RL_JOINT_ARTIFACTS,
        "training_labels_locked": True,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": False,
    },
    "2026-07-24-dense-local-hfss": {
        "version": "v0.4.0-dense-local-hfss",
        "artifacts": DENSE_LOCAL_HFSS_ARTIFACTS,
        "training_labels_locked": False,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": True,
        "hfss_physical_labels_allowed": True,
        "residual_critic_training_locked": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_name(artifact: Artifact, source: Path) -> str:
    return f"{artifact.name}{source.suffix.lower()}"


def main() -> None:
    args = parse_args()
    if args.tag not in BASELINE_CONFIGS:
        known = ", ".join(sorted(BASELINE_CONFIGS))
        raise ValueError(f"Unknown baseline tag {args.tag!r}; choose one of: {known}")
    config = BASELINE_CONFIGS[args.tag]
    baseline = ROOT / "baselines" / args.tag
    snapshots = baseline / "snapshots"
    manifest_path = baseline / "artifact_manifest.csv"
    if args.verify_only:
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
        failed = []
        for row in rows:
            path = ROOT / row["snapshot_path"]
            if not path.exists() or sha256(path) != row["sha256"]:
                failed.append(row["name"])
        print(json.dumps({"verified": not failed, "failed": failed}, indent=2))
        raise SystemExit(1 if failed else 0)

    snapshots.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for artifact in config["artifacts"]:
        source = ROOT / artifact.source
        if not source.exists():
            missing.append(artifact.source)
            continue
        target = snapshots / snapshot_name(artifact, source)
        shutil.copy2(source, target)
        digest = sha256(target)
        if digest != sha256(source):
            raise RuntimeError(f"Snapshot hash mismatch: {artifact.name}")
        rows.append(
            {
                "name": artifact.name,
                "level": artifact.level,
                "status": artifact.status,
                "source_path": artifact.source,
                "snapshot_path": target.relative_to(ROOT).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": digest,
                "note": artifact.note,
            }
        )
    if missing:
        raise FileNotFoundError("Missing baseline artifacts:\n" + "\n".join(missing))
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "tag": args.tag,
        "version": config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(rows),
        "snapshot_bytes": sum(int(row["size_bytes"]) for row in rows),
        "training_labels_locked": config["training_labels_locked"],
        "pattern_labels_allowed": config["pattern_labels_allowed"],
        "strict_benchmark_gate_pass": config["strict_benchmark_gate_pass"],
    }
    for optional_key in (
        "hfss_physical_labels_allowed",
        "residual_critic_training_locked",
    ):
        if optional_key in config:
            metadata[optional_key] = config[optional_key]
    (baseline / "baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
