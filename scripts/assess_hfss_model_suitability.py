"""Create a machine-readable suitability decision for the corrected HFSS/EEP model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "hfss_outputs" / "multitask_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-eep-dir",
        type=Path,
        default=DATASET_ROOT / "eep_smoke_16port_modalpower_20260714",
    )
    parser.add_argument(
        "--corrected-eep-dir",
        type=Path,
        default=DATASET_ROOT / "eep_smoke_16port_matched_v2_20260714",
    )
    parser.add_argument(
        "--matched-model-manifest",
        type=Path,
        default=ROOT / "hfss_outputs" / "matched_model_v2_20260714" / "matched_model_manifest.json",
    )
    parser.add_argument(
        "--active-return-summary",
        type=Path,
        default=(
            DATASET_ROOT
            / "full_s256p_matched_v2_20260714"
            / "active_return_analysis_20260714"
            / "active_return_summary.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATASET_ROOT / "model_suitability_assessment_20260714",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_source_conventions(root: Path) -> dict[str, int]:
    counts = {
        "legacy_magnitude_v_csv": 0,
        "incident_power_w_csv": 0,
        "unclassified_source_csv": 0,
        "vbs_use_incident_voltage": 0,
    }
    candidates: set[Path] = set(root.rglob("hfss_sources.csv"))
    candidates.update(root.rglob("sources_*.csv"))
    for path in candidates:
        try:
            header = path.open("r", encoding="utf-8-sig", errors="replace").readline()
        except OSError:
            continue
        if "MagnitudeV" in header:
            counts["legacy_magnitude_v_csv"] += 1
        elif "IncidentPowerW" in header or "incident_power_w" in header:
            counts["incident_power_w_csv"] += 1
        else:
            counts["unclassified_source_csv"] += 1
    for path in root.rglob("*.vbs"):
        try:
            if "UseIncidentVoltage" in path.read_text(encoding="utf-8-sig", errors="replace"):
                counts["vbs_use_incident_voltage"] += 1
        except OSError:
            continue
    counts["source_csv_total"] = sum(
        counts[key]
        for key in ("legacy_magnitude_v_csv", "incident_power_w_csv", "unclassified_source_csv")
    )
    return counts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    legacy_operator = load_json(args.legacy_eep_dir / "operator_analysis_summary.json")
    legacy_superposition = load_json(args.legacy_eep_dir / "superposition_analysis_summary.json")
    corrected_operator = load_json(args.corrected_eep_dir / "operator_analysis_summary.json")
    corrected_superposition = load_json(args.corrected_eep_dir / "superposition_analysis_summary.json")
    matching = load_json(args.corrected_eep_dir / "matching_50ohm_lsection" / "matching_network_summary.json")
    model_manifest = load_json(args.matched_model_manifest)
    active_return = load_json(args.active_return_summary) if args.active_return_summary.exists() else None
    source_audit = scan_source_conventions(ROOT / "hfss_outputs")

    normalization_passed = bool(
        corrected_superposition.get("all_cases_passed")
        and corrected_superposition.get("complex_nmse_max", 1.0) <= 1.0e-6
        and max(abs(value - 1.0) for value in corrected_superposition.get("best_fit_scale_magnitude_range", [0.0]))
        <= 1.0e-4
    )
    geometry_non_touching = bool(model_manifest["interelement_tip_gap_mm"] > 0.0)
    raw_match_passed = bool(corrected_operator.get("engineering_port_matching_passed"))
    ideal_match_passed = bool(matching.get("engineering_matching_smoke_passed"))
    eep_structural_passed = bool(corrected_operator.get("smoke_passed"))
    full256_validated = bool(active_return and active_return.get("matrix_validation_passed"))
    active_return_passed = bool(active_return and active_return.get("hard_gate_passed"))

    decisions = {
        "eep_linear_superposition_smoke": "suitable" if normalization_passed and eep_structural_passed else "not_suitable",
        "single_frequency_af_to_eep_algorithm_development": (
            "conditionally_suitable" if normalization_passed and geometry_non_touching and ideal_match_passed else "not_suitable"
        ),
        "new_fullwave_critic_training": (
            "allowed_after_new_labels" if active_return_passed else "blocked_due_full256_active_return_failure"
        ),
        "absolute_power_or_energy_claims": "not_suitable_with_legacy_labels",
        "bandwidth_or_hardware_engineering_claims": "not_suitable_with_current_pec_single_frequency_proxy",
    }
    overall = "algorithmic_proxy_blocked_by_full256_active_match"
    hard_blockers = [
        "Raw corrected-array 50-ohm return loss is below the 10 dB requirement.",
        "Full 256-port active-return validation failed the 10 dB all-active-channel requirement for every evaluated model.",
        "The L-section is ideal and single-frequency and is not embedded in the feed geometry.",
        "The model omits substrate, balun/feed network, conductor/dielectric loss, phase-shifter loss, and bandwidth.",
        "Legacy full-wave source files use the pre-correction magnitude convention and require quarantine/audit.",
    ]
    next_acceptance = [
        "Cluster edge, corner, and interior ports by passive and K=1 active impedance using the full 256-port matrix.",
        "Design port-class matching or a multiport decoupling/matching network rather than one identical L-section.",
        "Repeat K=1/2/4/6 active-return validation and require every source-active port >= 10 dB.",
        "Validate 50-100 corrected-project direct HFSS cases against the matched EEP operator without fitted scale.",
        "Regenerate a small corrected full-wave training tranche before retraining the residual critic.",
    ]
    assessment = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_assessment": overall,
        "source_normalization_passed": normalization_passed,
        "power_normalization_policy": "sum(abs(complex_field_coefficient)**2)=1W_per_case",
        "corrected_geometry_non_touching": geometry_non_touching,
        "corrected_tip_gap_mm": model_manifest["interelement_tip_gap_mm"],
        "eep_structural_passed": eep_structural_passed,
        "unscaled_superposition_nmse_max": corrected_superposition.get("complex_nmse_max"),
        "unscaled_superposition_scale_range": corrected_superposition.get("best_fit_scale_magnitude_range"),
        "legacy_raw_return_loss_min_db": legacy_operator.get("return_loss_min_db"),
        "corrected_raw_return_loss_min_db": corrected_operator.get("return_loss_min_db"),
        "corrected_raw_port_match_passed": raw_match_passed,
        "ideal_external_lmatch_return_loss_min_db": matching.get("matched_return_loss_min_db"),
        "ideal_external_lmatch_passed": ideal_match_passed,
        "full256_s_matrix_validated": full256_validated,
        "full256_active_return_hard_gate_passed": active_return_passed,
        "full256_active_return_models": active_return.get("models") if active_return else None,
        "corrected_selected_port_coupling_worst_db": corrected_operator.get("mutual_coupling_worst_db"),
        "matched_selected_port_coupling_worst_db": matching.get("matched_mutual_coupling_worst_db"),
        "source_convention_audit": source_audit,
        "decisions": decisions,
        "hard_blockers": hard_blockers,
        "next_acceptance": next_acceptance,
        "legacy_data_policy": {
            "status": "quarantine_from_absolute_power_and_residual_labels",
            "allowed_use": "shape-only pretraining after per-case source-convention audit",
            "forbidden_use": "accepted-power, efficiency, calibrated EEP residual, or engineering gate labels",
        },
    }
    json_path = args.out_dir / "model_suitability_assessment.json"
    json_path.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    lines = [
        "# HFSS/EEP model suitability assessment",
        "",
        f"Overall: **{overall}**",
        "",
        "## Verified",
        f"- Driven Modal incident-power normalization: {normalization_passed}",
        "- System power normalization: sum(abs(a_i)^2) = 1 W per case",
        f"- Non-touching corrected geometry: {geometry_non_touching} (tip gap {model_manifest['interelement_tip_gap_mm']:.3f} mm)",
        f"- Unscaled EEP/direct NMSE max: {corrected_superposition.get('complex_nmse_max'):.3e}",
        f"- Best-fit scale range: {corrected_superposition.get('best_fit_scale_magnitude_range')}",
        f"- Ideal 50-ohm L-match smoke: {ideal_match_passed} (minimum return loss {matching.get('matched_return_loss_min_db'):.2f} dB)",
        f"- Full 256-port S-matrix validation: {full256_validated}",
        f"- Full 256-port active-return 10 dB hard gate: {active_return_passed}",
        "",
        "## Blocking Results",
    ]
    lines.extend(f"- {item}" for item in hard_blockers)
    lines.extend(["", "## Decision", ""])
    lines.extend(f"- {name}: {value}" for name, value in decisions.items())
    lines.extend(["", "## Next Acceptance", ""])
    lines.extend(f"{index}. {item}" for index, item in enumerate(next_acceptance, 1))
    lines.extend(
        [
            "",
            "## Legacy Source Audit",
            "",
            f"- Legacy MagnitudeV source CSV: {source_audit['legacy_magnitude_v_csv']}",
            f"- Corrected incident-power source CSV: {source_audit['incident_power_w_csv']}",
            f"- Generated VBS containing UseIncidentVoltage: {source_audit['vbs_use_incident_voltage']}",
        ]
    )
    (args.out_dir / "model_suitability_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
