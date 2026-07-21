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


ARTIFACTS = (
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
    for artifact in ARTIFACTS:
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
        "version": "v0.1.0-physics-gated",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(rows),
        "snapshot_bytes": sum(int(row["size_bytes"]) for row in rows),
        "training_labels_locked": True,
        "strict_benchmark_gate_pass": False,
    }
    (baseline / "baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
