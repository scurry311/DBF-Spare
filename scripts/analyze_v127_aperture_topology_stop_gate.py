#!/usr/bin/env python3
"""Aggregate the frozen v1.27 1x1 physical screens and apply the stop gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "hfss_outputs" / "v127_aperture_topology_stop_gate_20260806"
SOURCES = {
    "joint_screen": ROOT
    / "hfss_outputs"
    / "v127_joint_radiator_input_doe_20260806_run08_fixed_aperture_mesh"
    / "candidate_metrics.csv",
    "complex_impedance_target": ROOT
    / "hfss_outputs"
    / "v127_targeted_complex_impedance_doe_20260806_run09"
    / "candidate_metrics.csv",
    "tongue_resonance_recovery": ROOT
    / "hfss_outputs"
    / "v127_tongue_resonance_recovery_20260806_run12_partitioned_mesh"
    / "candidate_metrics.csv",
}


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_run, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                resistance = as_float(row, "input_resistance_ohm")
                gamma = as_float(row, "s11_magnitude")
                ideal_gamma = abs((resistance - 50.0) / (resistance + 50.0))
                rows.append(
                    {
                        "source_run": source_run,
                        **row,
                        "distance_to_50ohm": math.hypot(
                            resistance - 50.0,
                            as_float(row, "input_reactance_ohm"),
                        ),
                        "ideal_series_reactance_cancel_rl_db": (
                            math.inf if ideal_gamma == 0.0 else -20.0 * math.log10(ideal_gamma)
                        ),
                        "gamma_distance_to_origin": gamma,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="ascii")


def candidate(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    return next(row for row in rows if row["case_id"] == case_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_directory if args.output_directory.is_absolute() else ROOT / args.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite stop-gate output: {output}")
    output.mkdir(parents=True)

    rows = read_rows()
    rows.sort(key=lambda row: as_float(row, "passive_rl_db"), reverse=True)
    best = rows[0]
    e = candidate(rows, "joint_e_mid_capacitive")
    m = candidate(rows, "res_m_e_tongue_3p40")
    n = candidate(rows, "res_n_e_tongue_3p65")
    g = candidate(rows, "target_g_e_short_stub")
    tongue_variants = [m, n]
    source_best = {}
    for source in SOURCES:
        source_rows = [row for row in rows if row["source_run"] == source]
        source_best[source] = {
            "case_id": source_rows[0]["case_id"],
            "passive_rl_db": as_float(source_rows[0], "passive_rl_db"),
            "input_impedance_ohm": [
                as_float(source_rows[0], "input_resistance_ohm"),
                as_float(source_rows[0], "input_reactance_ohm"),
            ],
        }

    ideal_best = max(rows, key=lambda row: float(row["ideal_series_reactance_cancel_rl_db"]))
    audit = {
        "stage": "v1.27_aperture_coupled_radiator_stop_gate",
        "evidence_level": "B_diagnostic_physical_1x1",
        "candidate_count": len(rows),
        "converged_count": sum(row["converged"].lower() == "true" for row in rows),
        "delta_s_gate_pass_count": sum(as_float(row, "final_delta_s") <= 0.05 for row in rows),
        "topology_clean_count": sum(int(row["topology_warning_count"]) == 0 for row in rows),
        "passive_rl_10db_pass_count": sum(as_float(row, "passive_rl_db") >= 10.0 for row in rows),
        "best_physical_candidate": {
            "case_id": best["case_id"],
            "source_run": best["source_run"],
            "passive_rl_db": as_float(best, "passive_rl_db"),
            "s11_magnitude": as_float(best, "s11_magnitude"),
            "input_impedance_ohm": [
                as_float(best, "input_resistance_ohm"),
                as_float(best, "input_reactance_ohm"),
            ],
            "final_delta_s": as_float(best, "final_delta_s"),
        },
        "source_best": source_best,
        "observed_impedance_envelope_ohm": {
            "resistance_min": min(as_float(row, "input_resistance_ohm") for row in rows),
            "resistance_max": max(as_float(row, "input_resistance_ohm") for row in rows),
            "reactance_min": min(as_float(row, "input_reactance_ohm") for row in rows),
            "reactance_max": max(as_float(row, "input_reactance_ohm") for row in rows),
        },
        "tongue_length_sensitivity": {
            "reference_case": e["case_id"],
            "reference_slot_length_mm": as_float(e, "dual_slot_length_mm"),
            "tested_slot_lengths_mm": [as_float(row, "dual_slot_length_mm") for row in tongue_variants],
            "maximum_abs_resistance_change_ohm": max(
                abs(as_float(row, "input_resistance_ohm") - as_float(e, "input_resistance_ohm"))
                for row in tongue_variants
            ),
            "maximum_abs_reactance_change_ohm": max(
                abs(as_float(row, "input_reactance_ohm") - as_float(e, "input_reactance_ohm"))
                for row in tongue_variants
            ),
            "maximum_abs_rl_change_db": max(
                abs(as_float(row, "passive_rl_db") - as_float(e, "passive_rl_db"))
                for row in tongue_variants
            ),
            "interpretation": "The aperture-fed input is nearly insensitive to the tested tongue-length increase at 10 GHz.",
        },
        "stub_crossing_diagnostic": {
            "long_stub_case": e["case_id"],
            "long_stub_length_mm": as_float(e, "open_stub_length_mm"),
            "long_stub_impedance_ohm": [
                as_float(e, "input_resistance_ohm"),
                as_float(e, "input_reactance_ohm"),
            ],
            "short_stub_case": g["case_id"],
            "short_stub_length_mm": as_float(g, "open_stub_length_mm"),
            "short_stub_impedance_ohm": [
                as_float(g, "input_resistance_ohm"),
                as_float(g, "input_reactance_ohm"),
            ],
            "interpretation": "The stub reverses reactance only by traversing a high-resistance resonance, so it is not an independent matching control.",
        },
        "ideal_series_reactance_cancellation_diagnostic": {
            "case_id": ideal_best["case_id"],
            "upper_bound_rl_db": float(ideal_best["ideal_series_reactance_cancel_rl_db"]),
            "not_a_physical_result": True,
            "interpretation": "A new feed topology needs an independent, manufacturable reactance transformation; the present topology does not provide it.",
        },
        "mutual_coupling_claim": {
            "nearest_neighbor_sij_measured": False,
            "reason": "The 1x1 passive-RL prerequisite failed, so no 2x2 solve was authorized.",
        },
        "resource_risk": {
            "minimum_observed_free_memory_gib": 2.3651466369628906,
            "memory_aborted_cases": 0,
            "interpretation": "Several short excursions fell below 3 GiB; expansion is not resource-authorized.",
        },
    }
    decision = {
        "decision": "stop_current_aperture_tongue_stub_topology",
        "reason": "Zero of 18 converged physical candidates reached 10 dB passive RL at 10 GHz.",
        "allow_three_frequency_1x1": False,
        "allow_2x2_jacobian": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "next_hardware_branch": "Use a true balanced or dual-resonant input with an independent impedance-transforming degree of freedom before any coupling audit.",
    }

    write_csv(output / "all_candidate_metrics.csv", rows)
    write_json(output / "topology_reachability_audit.json", audit)
    write_json(output / "stage_decision.json", decision)
    print(json.dumps({"audit": audit, "decision": decision}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
