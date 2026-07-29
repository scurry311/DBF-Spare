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


IMPLEMENTATION_RESIDUAL_ARTIFACTS = (
    Artifact("boundary_prepare", "hfss_outputs/trusted_dense_boundary_dataset_20260724_run01/prepare_summary.json", "A", "passed", "Twenty-one implementation-boundary and lower-ratio candidates forming a 95-case HFSS batch."),
    Artifact("boundary_manifest", "hfss_outputs/trusted_dense_boundary_dataset_20260724_run01/candidate_manifest.csv", "A", "diagnostic", "Per-candidate perturbation, ratio, nominal EEP, predicted actual-basis, and active-RL metadata."),
    Artifact("mapping_smoke", "hfss_outputs/trusted_dense_boundary_hfss_mapping_smoke_20260724_run01/analysis_summary.json", "A", "passed", "Seven-case K=6 dual-excitation mapping smoke before the full batch."),
    Artifact("boundary_hfss_analysis", "hfss_outputs/trusted_dense_boundary_hfss_20260724_run01/analysis_summary.json", "A", "passed", "Complete 95-case actual-EEP/direct-HFSS mapping and nominal-command residual analysis."),
    Artifact("boundary_hfss_labels", "hfss_outputs/trusted_dense_boundary_hfss_20260724_run01/candidate_residual_labels.csv", "A", "hard_negative_labels", "Fifteen EEP-pass/HFSS-fail hard negatives and six lower-ratio paired labels."),
    Artifact("critic_dataset_summary", "hfss_outputs/trusted_dense_implementation_residual_dataset_20260724_run01/build_summary.json", "A", "training_open", "Scene-grouped 36-candidate dataset readiness and residual-scale audit."),
    Artifact("critic_dataset", "hfss_outputs/trusted_dense_implementation_residual_dataset_20260724_run01/fullwave_residual_dataset_v2.npz", "A", "training_dataset", "Compact mask, task-weight, target, hardware-condition, residual, gate, and split arrays."),
    Artifact("five_seed_summary", "hfss_outputs/trusted_dense_implementation_residual_critic_20260724_run01/five_seed_summary.json", "A", "trained_experimental", "Five-seed aggregate critic metrics."),
    Artifact("five_seed_metrics", "hfss_outputs/trusted_dense_implementation_residual_critic_20260724_run01/five_seed_summary.csv", "A", "diagnostic", "Five-seed AUROC, AUPRC, ECE, ranking, and confidence intervals."),
    Artifact("best_checkpoint", "hfss_outputs/trusted_dense_implementation_residual_critic_20260724_run01/seed_20260725/residual_critic_v2.pt", "A", "experimental_checkpoint", "Best validation-seed 188343-parameter residual critic checkpoint."),
    Artifact("best_run_summary", "hfss_outputs/trusted_dense_implementation_residual_critic_20260724_run01/seed_20260725/run_summary.json", "A", "diagnostic", "Best-seed train, validation, test, calibration, and selection metrics."),
    Artifact("critic_acceptance", "hfss_outputs/trusted_dense_implementation_critic_decision_20260724_run01/critic_acceptance.json", "A", "not_promoted", "Authoritative engineering-promotion decision and failure reasons."),
    Artifact("critic_seed_test", "hfss_outputs/trusted_dense_implementation_critic_decision_20260724_run01/five_seed_test_metrics.csv", "A", "diagnostic", "Per-seed scene-test residual and gate metrics."),
)


EXPANDED_RESIDUAL_ARTIFACTS = (
    Artifact("expanded_prepare", "hfss_outputs/expanded_independent_scenes_20260724_run02/prepare_summary.json", "A", "passed", "Forty-five independent target scenes, 105 paired candidates, and the pre-HFSS physics gate."),
    Artifact("expanded_manifest", "hfss_outputs/expanded_independent_scenes_20260724_run02/candidate_manifest.csv", "A", "diagnostic", "Per-candidate control, intermediate-error, targeted-mainlobe, and lower-ratio metadata."),
    Artifact("mapping_smoke", "hfss_outputs/expanded_independent_scenes_hfss_smoke_20260724_run01/analysis_summary.json", "A", "passed", "Twenty-one-case K=6 control, mainlobe-failure, and lower-ratio mapping smoke."),
    Artifact("expanded_hfss_analysis", "hfss_outputs/expanded_independent_scenes_hfss_20260724_run01/analysis_summary.json", "A", "passed", "Complete 455-case no-scale EEP/direct-HFSS analysis for 105 candidates."),
    Artifact("expanded_hfss_labels", "hfss_outputs/expanded_independent_scenes_hfss_20260724_run01/candidate_residual_labels.csv", "A", "fullwave_labels", "Full-wave residual and engineering-gate labels for all expanded candidates."),
    Artifact("critic_dataset_summary", "hfss_outputs/trusted_dense_implementation_residual_dataset_20260724_run02/build_summary.json", "A", "training_open", "Scene-grouped 141-candidate and 60-scene three-source dataset audit."),
    Artifact("critic_dataset", "hfss_outputs/trusted_dense_implementation_residual_dataset_20260724_run02/fullwave_residual_dataset_v2.npz", "A", "training_dataset", "Compact mask, task-weight, target, hardware-condition, residual, gate, and split arrays."),
    Artifact("five_seed_summary", "hfss_outputs/trusted_dense_implementation_residual_critic_20260724_run02/five_seed_summary.json", "A", "trained_experimental", "Five-seed aggregate critic metrics on eleven held-out scenes."),
    Artifact("five_seed_metrics", "hfss_outputs/trusted_dense_implementation_residual_critic_20260724_run02/five_seed_summary.csv", "A", "diagnostic", "Five-seed AUROC, AUPRC, ECE, ranking, and confidence intervals."),
    Artifact("best_checkpoint", "hfss_outputs/trusted_dense_implementation_residual_critic_20260724_run02/seed_20260726/residual_critic_v2.pt", "A", "experimental_checkpoint", "Best validation-seed 188407-parameter residual critic checkpoint."),
    Artifact("best_run_summary", "hfss_outputs/trusted_dense_implementation_residual_critic_20260724_run02/seed_20260726/run_summary.json", "A", "diagnostic", "Best-seed train, validation, test, calibration, and selection metrics."),
    Artifact("critic_acceptance", "hfss_outputs/trusted_dense_implementation_critic_decision_20260724_run04/critic_acceptance.json", "A", "not_promoted", "Authoritative promotion decision including gate15 AUROC and calibration failures."),
    Artifact("critic_seed_test", "hfss_outputs/trusted_dense_implementation_critic_decision_20260724_run04/five_seed_test_metrics.csv", "A", "diagnostic", "Per-seed held-out scene residual and gate metrics."),
)


GATE15_BOUNDARY_ARTIFACTS = (
    Artifact("boundary_prepare", "hfss_outputs/gate15_boundary_scenes_20260725_run01/prepare_summary.json", "A", "passed", "Thirty independent PSLL, nearest-isolation, and local-isolation boundary scenes with ninety paired candidates."),
    Artifact("boundary_manifest", "hfss_outputs/gate15_boundary_scenes_20260725_run01/candidate_manifest.csv", "A", "diagnostic", "Per-candidate boundary type, side, physical perturbation, EEP metrics, and active-return metadata."),
    Artifact("mapping_smoke", "hfss_outputs/gate15_boundary_scenes_hfss_smoke_20260725_run01/analysis_summary.json", "A", "passed", "Thirty-four-case no-scale EEP/direct-HFSS mapping smoke across all three boundary mechanisms."),
    Artifact("boundary_hfss_analysis", "hfss_outputs/gate15_boundary_scenes_hfss_20260725_run01/analysis_summary.json", "A", "passed", "Complete 444-case full-wave analysis for ninety candidates and thirty independent scenes."),
    Artifact("boundary_hfss_summary", "hfss_outputs/gate15_boundary_scenes_hfss_20260725_run01/gate15_boundary_hfss_summary.json", "A", "passed", "Strict audit of inside/outside threshold crossings with zero new mainlobe failures."),
    Artifact("boundary_hfss_groups", "hfss_outputs/gate15_boundary_scenes_hfss_20260725_run01/gate15_boundary_hfss_summary_by_type_side.csv", "A", "diagnostic", "Full-wave margins by PSLL/nearest/local boundary type and control/inside/outside side."),
    Artifact("boundary_hfss_labels", "hfss_outputs/gate15_boundary_scenes_hfss_20260725_run01/candidate_residual_labels.csv", "A", "pattern_boundary_labels", "Thirty isolated gate15 hard negatives, thirty just-inside positives, and thirty controls."),
    Artifact("critic_dataset_summary", "hfss_outputs/trusted_dense_implementation_residual_dataset_20260725_run02/build_summary.json", "A", "training_open", "Scene-grouped 231-candidate and 90-scene four-source dataset audit."),
    Artifact("critic_dataset", "hfss_outputs/trusted_dense_implementation_residual_dataset_20260725_run02/fullwave_residual_dataset_v2.npz", "A", "training_dataset", "Mask, task-weight, target, implementation-condition, residual, gate, and leakage-free split arrays."),
    Artifact("five_seed_summary", "hfss_outputs/trusted_dense_implementation_residual_critic_20260725_run01/five_seed_summary.json", "A", "trained", "Five-seed discrimination, residual-regression, and ranking metrics before pooled calibration."),
    Artifact("five_seed_metrics", "hfss_outputs/trusted_dense_implementation_residual_critic_20260725_run01/five_seed_summary.csv", "A", "diagnostic", "Per-seed gate AUROC, AUPRC, temperature ECE, and ranking confidence intervals."),
    Artifact("best_checkpoint", "hfss_outputs/trusted_dense_implementation_residual_critic_20260725_run01/seed_20260727/residual_critic_v2.pt", "A", "calibrator_required", "Best validation-seed 188855-parameter checkpoint; never deploy without the pooled calibrator."),
    Artifact("best_run_summary", "hfss_outputs/trusted_dense_implementation_residual_critic_20260725_run01/seed_20260727/run_summary.json", "A", "diagnostic", "Best-seed train, validation, test, residual, gate, and ranking metrics."),
    Artifact("pooled_calibration", "hfss_outputs/trusted_dense_implementation_residual_calibration_20260725_run01/pooled_calibration_summary.json", "A", "passed", "Scene-grouped regularized-isotonic calibration selected without test-label access."),
    Artifact("calibration_cv", "hfss_outputs/trusted_dense_implementation_residual_calibration_20260725_run01/calibration_cross_validation.csv", "A", "diagnostic", "Grouped validation OOF alpha selection for gate15 and gate20 calibrators."),
    Artifact("calibrated_seed_metrics", "hfss_outputs/trusted_dense_implementation_residual_calibration_20260725_run01/pooled_calibrated_five_seed_test_metrics.csv", "A", "diagnostic", "Five-seed test AUROC, AUPRC, Brier, ECE, precision, and recall after pooled calibration."),
    Artifact("critic_acceptance", "hfss_outputs/trusted_dense_implementation_critic_decision_20260725_run02/critic_acceptance.json", "A", "stage1_promoted", "Promotion decision requiring the checkpoint and pooled calibrator as one inference package."),
    Artifact("critic_seed_test", "hfss_outputs/trusted_dense_implementation_critic_decision_20260725_run02/five_seed_test_metrics.csv", "A", "diagnostic", "Per-seed held-out scene metrics and mainlobe-negative support audit."),
)


PROSPECTIVE_HFSS_ARTIFACTS = (
    Artifact("prospective_prepare", "hfss_outputs/prospective_gate15_scenes_20260725_run01/prepare_summary.json", "A", "passed", "Twenty-four unseen target-direction scenes with balanced PSLL, nearest-isolation, and local-isolation boundary triplets."),
    Artifact("freeze_manifest", "hfss_outputs/prospective_frozen_critic_20260725_run01/prospective_freeze_manifest.json", "A", "pre_registered", "Pre-HFSS checkpoint, calibrator, dataset, thresholds, substitutions, and SHA-256 freeze record."),
    Artifact("frozen_predictions", "hfss_outputs/prospective_frozen_critic_20260725_run01/frozen_predictions_before_hfss.csv", "A", "pre_registered", "Seventy-two candidate probabilities, residual predictions, confidence bounds, and admission decisions saved before HFSS."),
    Artifact("frozen_selections", "hfss_outputs/prospective_frozen_critic_20260725_run01/frozen_scene_selections_before_hfss.csv", "A", "pre_registered", "Four frozen top-one selection rules for every unseen scene."),
    Artifact("mapping_smoke", "hfss_outputs/prospective_gate15_hfss_smoke_20260725_run01/analysis_summary.json", "A", "passed", "Thirty-case unseen-scene no-scale mapping smoke before the full prospective batch."),
    Artifact("prospective_hfss_analysis", "hfss_outputs/prospective_gate15_hfss_20260725_run01/analysis_summary.json", "A", "passed", "Complete 354-case prospective no-scale EEP/direct-HFSS analysis."),
    Artifact("prospective_boundary_audit", "hfss_outputs/prospective_gate15_hfss_20260725_run01/gate15_boundary_hfss_summary.json", "A", "passed", "Dynamic audit of all twenty-four inside/outside boundary crossings with zero mainlobe failures."),
    Artifact("prospective_boundary_groups", "hfss_outputs/prospective_gate15_hfss_20260725_run01/gate15_boundary_hfss_summary_by_type_side.csv", "A", "diagnostic", "Prospective HFSS margins grouped by boundary type and side."),
    Artifact("prospective_hfss_labels", "hfss_outputs/prospective_gate15_hfss_20260725_run01/candidate_residual_labels.csv", "A", "prospective_locked", "Prospective labels retained for evaluation only and excluded from post-HFSS retraining."),
    Artifact("prospective_evaluation", "hfss_outputs/prospective_frozen_critic_evaluation_20260725_run03/prospective_validation_summary.json", "A", "failed_acceptance", "Frozen critic prospective decision: gate15 AUROC and gate20 ECE failed the inherited v0.7 protocol."),
    Artifact("prospective_candidate_metrics", "hfss_outputs/prospective_frozen_critic_evaluation_20260725_run03/prospective_candidate_evaluation.csv", "A", "diagnostic", "Frozen prediction, HFSS truth, residual error, admission, and threshold-consistency fields for every candidate."),
    Artifact("prospective_group_metrics", "hfss_outputs/prospective_frozen_critic_evaluation_20260725_run03/prospective_group_metrics.csv", "A", "diagnostic", "Gate discrimination and calibration grouped by PSLL, nearest, local, and boundary side."),
    Artifact("prospective_pair_metrics", "hfss_outputs/prospective_frozen_critic_evaluation_20260725_run03/prospective_boundary_pair_metrics.csv", "A", "diagnostic", "Inside/outside pair ordering and control-free discrimination by boundary mechanism."),
    Artifact("prospective_scene_metrics", "hfss_outputs/prospective_frozen_critic_evaluation_20260725_run03/prospective_scene_selection_metrics.csv", "A", "diagnostic", "Frozen scene-level top-one and oracle pass rates."),
    Artifact("prospective_bootstrap", "hfss_outputs/prospective_frozen_critic_evaluation_20260725_run03/prospective_scene_bootstrap_95ci.csv", "A", "diagnostic", "Two-thousand-repeat sample_index-grouped prospective uncertainty intervals."),
    Artifact("prospective_failures", "hfss_outputs/prospective_frozen_critic_evaluation_20260725_run03/prospective_failures.csv", "A", "diagnostic", "Candidate-level pattern or strict admission errors retained without post-HFSS tuning."),
    Artifact("prospective_calibration", "hfss_outputs/prospective_frozen_critic_evaluation_20260725_run03/prospective_calibration_bins.csv", "A", "diagnostic", "Prospective reliability bins for all frozen critic gates."),
)


V09_ADAPTIVE_HFSS_ARTIFACTS = (
    Artifact("eep_pool_summary", "hfss_outputs/v09_eep_development_candidates_20260726_run01/prepare_summary.json", "A", "passed", "Sixty independent scenes and 1,920 no-control sparse EEP/S256 candidates."),
    Artifact("development_summary", "hfss_outputs/v09_margin_development_dataset_20260726_run01/prepare_summary.json", "A", "passed", "Scene-grouped 420-candidate physical-margin development set."),
    Artifact("development_groups", "hfss_outputs/v09_margin_development_dataset_20260726_run01/distribution_by_split_k_ratio.csv", "A", "diagnostic", "K, ratio, split, and boundary-label support."),
    Artifact("critic_summary", "hfss_outputs/v09_physical_margin_critic_20260726_run02/training_summary.json", "A", "passed", "Five-seed scene-conditioned physical-margin residual critic."),
    Artifact("critic_metrics", "hfss_outputs/v09_physical_margin_critic_20260726_run02/five_seed_metrics.csv", "A", "diagnostic", "Five-seed discrimination, calibration, and no-control ranking metrics."),
    Artifact("checkpoint_20260726", "hfss_outputs/v09_physical_margin_critic_20260726_run02/seed_20260726/best_checkpoint.pt", "A", "ensemble_checkpoint", "Frozen physical-margin ensemble member."),
    Artifact("checkpoint_20260727", "hfss_outputs/v09_physical_margin_critic_20260726_run02/seed_20260727/best_checkpoint.pt", "A", "ensemble_checkpoint", "Frozen physical-margin ensemble member."),
    Artifact("checkpoint_20260728", "hfss_outputs/v09_physical_margin_critic_20260726_run02/seed_20260728/best_checkpoint.pt", "A", "ensemble_checkpoint", "Frozen physical-margin ensemble member."),
    Artifact("checkpoint_20260729", "hfss_outputs/v09_physical_margin_critic_20260726_run02/seed_20260729/best_checkpoint.pt", "A", "ensemble_checkpoint", "Frozen physical-margin ensemble member."),
    Artifact("checkpoint_20260730", "hfss_outputs/v09_physical_margin_critic_20260726_run02/seed_20260730/best_checkpoint.pt", "A", "ensemble_checkpoint", "Frozen physical-margin ensemble member."),
    Artifact("adaptive_eep_summary", "hfss_outputs/v09_adaptive_ratio_eep_loop_20260726_run01/adaptive_summary.json", "A", "passed_conservative", "Minimum-ratio EEP/S256 search with conservative five-margin admission."),
    Artifact("smoke_analysis", "hfss_outputs/v09_hfss_smoke_20260726_run02/analysis_summary.json", "A", "passed", "Eighteen-candidate and 82-case HFSS smoke."),
    Artifact("smoke_acceptance", "hfss_outputs/v09_hfss_smoke_20260726_run02/v09_smoke_acceptance.json", "A", "passed", "Hard gate authorizing the 50-100 candidate stage."),
    Artifact("fullwave_analysis", "hfss_outputs/v09_fullwave_validation_20260726_run02/analysis_summary.json", "A", "passed", "Sixty-six new candidates and 338 direct-HFSS cases."),
    Artifact("fullwave_evaluation", "hfss_outputs/v09_fullwave_evaluation_20260726_run02/v09_fullwave_summary.json", "A", "calibration_required", "Combined 84-candidate held-out HFSS critic and ranking audit."),
    Artifact("fullwave_groups", "hfss_outputs/v09_fullwave_evaluation_20260726_run02/fullwave_by_k_ratio.csv", "A", "diagnostic", "Held-out full-wave metrics grouped by K and ratio."),
    Artifact("calibrator", "hfss_outputs/v09_fullwave_calibrator_20260726_run01/calibrator.json", "A", "frozen", "Regularized Platt calibrator fit on twelve independent HFSS validation scenes."),
    Artifact("calibration_summary", "hfss_outputs/v09_fullwave_calibrator_20260726_run01/calibration_summary.json", "A", "ece_pass_brier_tradeoff", "Held-out ECE improvement with the recorded small Brier tradeoff."),
    Artifact("prospective_pool", "hfss_outputs/v09_second_prospective_eep_candidates_20260726_run01/prepare_summary.json", "A", "passed", "Twelve second-cycle unseen scenes and 384 sparse candidates."),
    Artifact("prospective_adaptive", "hfss_outputs/v09_second_prospective_adaptive_20260726_run01/adaptive_summary.json", "A", "pre_registered", "Frozen calibrated adaptive-ratio selections before HFSS."),
    Artifact("prospective_freeze", "hfss_outputs/v09_second_prospective_hfss_dataset_20260726_run01/prospective_freeze_manifest.json", "A", "pre_registered", "Checkpoint hashes, threshold, target hashes, and no-tuning policy."),
    Artifact("prospective_analysis", "hfss_outputs/v09_second_prospective_hfss_20260726_run01/analysis_summary.json", "A", "passed", "Complete sixty-case no-scale prospective HFSS analysis."),
    Artifact("prospective_summary", "hfss_outputs/v09_second_prospective_hfss_20260726_run01/second_prospective_summary.json", "A", "passed_conservative", "Five of five admitted candidates passed strict HFSS gates."),
    Artifact("prospective_selections", "hfss_outputs/v09_second_prospective_hfss_20260726_run01/prospective_selection_evaluation.csv", "A", "diagnostic", "Per-scene ratio, PSLL, isolation, active-RL, admission, and HFSS truth."),
)


V112_PARETO_RESCUE_ARTIFACTS = (
    Artifact("protocol", "configs/v201_pareto_joint_feasibility_rescue.json", "A", "pre_registered", "Frozen v20.1 alpha, Pareto, rescue, stage-gate, and downstream-lock protocol."),
    Artifact("alpha_summary", "hfss_outputs/v201_dense_alpha_eep_20260729_run01/summary.json", "A", "passed_no_reserve", "Dense five-state EEP alpha validation for the two preregistered candidate neighborhoods."),
    Artifact("alpha_metrics", "hfss_outputs/v201_dense_alpha_eep_20260729_run01/dense_alpha_summary.csv", "A", "diagnostic", "Best exact alpha and physical margins for both requested candidates."),
    Artifact("warm_summary", "hfss_outputs/v201_warm_mask_pareto_screen_20260729_run01/summary.json", "A", "completed", "Five-state warm EEP screening of all 2,304 existing masks."),
    Artifact("pareto_selection", "hfss_outputs/v201_warm_mask_pareto_screen_20260729_run01/pareto_selection.csv", "A", "diagnostic", "Four-role pattern, active-RL, max-min, and Pareto-knee selections for 72 scene-ratio groups."),
    Artifact("rescue_summary", "hfss_outputs/v201_k24_progressive_rescue_20260729_run01/summary.json", "A", "failed_gate", "K=2/K=4 progressive three-frequency joint-feasibility rescue summary."),
    Artifact("scene_oracle", "hfss_outputs/v201_k24_progressive_rescue_20260729_run01/k24_scene_oracle.csv", "A", "diagnostic", "Combined v20, dense-alpha, and Pareto-rescue scene oracle."),
    Artifact("selected_paths", "hfss_outputs/v201_k24_progressive_rescue_20260729_run01/selected_path_metrics.csv", "A", "diagnostic", "Best command and physical margins on each of the 128 rescued Pareto paths."),
    Artifact("stop_audit", "hfss_outputs/v201_k24_progressive_rescue_20260729_run01/k24_stop_decision_audit.csv", "A", "diagnostic", "Per-scene v20 comparison and limiting state/constraint audit."),
    Artifact("stop_decision", "hfss_outputs/v201_k24_progressive_rescue_20260729_run01/stop_decision.json", "A", "algorithm_expansion_stopped", "Machine-readable preregistered stop decision."),
    Artifact("stage_decision", "hfss_outputs/v201_k24_progressive_rescue_20260729_run01/STAGE_DECISION.md", "A", "authoritative", "Human-readable v20.1 result and downstream decision."),
)


V113_BROADBAND_MATCH_REPLAY_ARTIFACTS = (
    Artifact("protocol", "configs/v21_three_frequency_broadband_matching.json", "A", "pre_registered", "Frozen-command three-frequency matching, replay gate, and downstream lock protocol."),
    Artifact("freeze_summary", "hfss_outputs/v21_frozen_v112_replay_20260729_run03/freeze_summary.json", "A", "frozen", "Corrected v1.12 replay package provenance and SHA-256 chain."),
    Artifact("freeze_manifest", "hfss_outputs/v21_frozen_v112_replay_20260729_run03/frozen_candidate_manifest.csv", "A", "frozen", "Twenty scene-level masks, commands, ratios, source rows, and hashes."),
    Artifact("uniform_design", "hfss_outputs/v21_three_frequency_broadband_match_20260729_run02_q50_uniform01/design_summary.json", "A", "failed_gate", "Uniform finite-Q S-L-S network screening result."),
    Artifact("geometry_design", "hfss_outputs/v21_three_frequency_broadband_match_20260729_run02_q50_geometry01/design_summary.json", "A", "selected_for_replay", "Corner/edge/interior finite-Q S-L-S design summary."),
    Artifact("geometry_parameters", "hfss_outputs/v21_three_frequency_broadband_match_20260729_run02_q50_geometry01/network_variant_summary.csv", "A", "diagnostic", "Three-class component parameters and active-match design metrics."),
    Artifact("geometry_active_replay", "hfss_outputs/v21_three_frequency_broadband_match_20260729_run02_q50_geometry01/candidate_active_rl_replay.csv", "A", "diagnostic", "Active-RL-only precheck on the frozen candidates."),
    Artifact("operator_validation", "hfss_outputs/v21_broadband_s256_eep_replay_20260729_run01/operator_structural_validation.csv", "A", "passed", "Three-frequency external S256 reciprocity, passivity, passive RL, and EEP map validation."),
    Artifact("frozen_replay_groups", "hfss_outputs/v21_broadband_s256_eep_replay_20260729_run01/frozen_replay_by_variant_k.csv", "A", "failed_gate", "Strict, pattern, active-RL, reserve, and efficiency counts by K."),
    Artifact("frozen_replay_metrics", "hfss_outputs/v21_broadband_s256_eep_replay_20260729_run01/frozen_replay_candidate_metrics.csv", "A", "diagnostic", "Paired old/new full five-state EEP/S256 replay metrics for all scenes."),
    Artifact("stage_decision", "hfss_outputs/v21_broadband_s256_eep_replay_20260729_run01/stage_decision.json", "A", "downstream_locked", "Machine-readable replay gate and downstream decision."),
    Artifact("stage_report", "hfss_outputs/v21_broadband_s256_eep_replay_20260729_run01/STAGE_REPORT.md", "A", "authoritative", "Human-readable matching, operator, replay, and stop decision."),
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
    "2026-07-24-implementation-residual": {
        "version": "v0.5.0-implementation-residual",
        "artifacts": IMPLEMENTATION_RESIDUAL_ARTIFACTS,
        "training_labels_locked": False,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": False,
        "hfss_physical_labels_allowed": True,
        "residual_critic_training_locked": False,
        "residual_critic_engineering_promoted": False,
    },
    "2026-07-24-expanded-residual": {
        "version": "v0.6.0-expanded-residual",
        "artifacts": EXPANDED_RESIDUAL_ARTIFACTS,
        "training_labels_locked": False,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": False,
        "hfss_physical_labels_allowed": True,
        "residual_critic_training_locked": False,
        "residual_critic_engineering_promoted": False,
    },
    "2026-07-25-gate15-boundary": {
        "version": "v0.7.0-gate15-boundary",
        "artifacts": GATE15_BOUNDARY_ARTIFACTS,
        "training_labels_locked": False,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": False,
        "hfss_physical_labels_allowed": True,
        "residual_critic_training_locked": False,
        "residual_critic_engineering_promoted": True,
        "prospective_validation_required": True,
    },
    "2026-07-25-prospective-hfss": {
        "version": "v0.8.0-prospective-hfss",
        "artifacts": PROSPECTIVE_HFSS_ARTIFACTS,
        "training_labels_locked": True,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": False,
        "hfss_physical_labels_allowed": True,
        "residual_critic_training_locked": True,
        "residual_critic_engineering_promoted": False,
        "prospective_validation_required": False,
        "prospective_validation_completed": True,
        "prospective_validation_pass": False,
        "automatic_hfss_admission_allowed": False,
    },
    "2026-07-26-v09-adaptive-hfss": {
        "version": "v0.9.0-physical-margin-adaptive",
        "artifacts": V09_ADAPTIVE_HFSS_ARTIFACTS,
        "training_labels_locked": True,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": True,
        "hfss_physical_labels_allowed": True,
        "residual_critic_training_locked": True,
        "residual_critic_engineering_promoted": True,
        "prospective_validation_required": False,
        "prospective_validation_completed": True,
        "prospective_validation_pass": True,
        "automatic_hfss_admission_allowed": True,
        "final_adaptive_coverage_target_met": False,
    },
    "2026-07-29-v112-pareto-joint-feasibility": {
        "version": "v1.12.0-pareto-joint-feasibility",
        "artifacts": V112_PARETO_RESCUE_ARTIFACTS,
        "training_labels_locked": True,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": False,
        "hfss_physical_labels_allowed": False,
        "residual_critic_training_locked": True,
        "residual_critic_engineering_promoted": False,
        "automatic_hfss_admission_allowed": False,
        "final_adaptive_coverage_target_met": False,
    },
    "2026-07-29-v113-broadband-match-replay": {
        "version": "v1.13.0-broadband-match-replay",
        "artifacts": V113_BROADBAND_MATCH_REPLAY_ARTIFACTS,
        "training_labels_locked": True,
        "pattern_labels_allowed": True,
        "strict_benchmark_gate_pass": False,
        "hfss_physical_labels_allowed": False,
        "residual_critic_training_locked": True,
        "residual_critic_engineering_promoted": False,
        "automatic_hfss_admission_allowed": False,
        "final_adaptive_coverage_target_met": False,
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
        "residual_critic_engineering_promoted",
        "prospective_validation_required",
        "prospective_validation_completed",
        "prospective_validation_pass",
        "automatic_hfss_admission_allowed",
        "final_adaptive_coverage_target_met",
    ):
        if optional_key in config:
            metadata[optional_key] = config[optional_key]
    (baseline / "baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
