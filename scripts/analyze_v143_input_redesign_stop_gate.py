#!/usr/bin/env python3
"""Compare v1.39/v1.42/v1.43 and issue the input-redesign stop decision."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V139 = ROOT / "hfss_outputs/v139_physical_2x2_differential_array_20260808_run01/initial_10ghz/direct01_repair03"
V142 = ROOT / "hfss_outputs/v142_corrected_s4_mapped_eep_joint_20260809_run01"
V143_1X1 = ROOT / "hfss_outputs/v143_y_balanced_element_1x1_20260809_run01"
V143_2X2 = ROOT / "hfss_outputs/v143_y_balanced_element_input_20260809_run02/initial_10ghz/direct01_repair02"
OUT = ROOT / "hfss_outputs/v143_y_balanced_element_input_20260809_run02/comparison"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def coupling_summary(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(float(row["coupling_db"]))
    return {name: sum(values) / len(values) for name, values in groups.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old_summary = read_json(V139 / "case_summary.json")
    new_summary = read_json(V143_2X2 / "case_summary.json")
    one_summary = read_json(V143_1X1 / "stage_summary.json")
    old_modal = {row["mode"]: row for row in read_csv(V139 / "modal_impedance.csv")}
    new_modal = {row["mode"]: row for row in read_csv(V143_2X2 / "modal_impedance.csv")}
    modal_rows = []
    for name in ("even", "x_odd", "y_odd", "checker"):
        old, new = old_modal[name], new_modal[name]
        modal_rows.append(
            {
                "mode": name,
                "old_rl_db": float(old["rl_db"]),
                "new_rl_db": float(new["rl_db"]),
                "rl_change_db": float(new["rl_db"]) - float(old["rl_db"]),
                "old_resistance_ohm": float(old["resistance_ohm"]),
                "new_resistance_ohm": float(new["resistance_ohm"]),
                "old_reactance_ohm": float(old["reactance_ohm"]),
                "new_reactance_ohm": float(new["reactance_ohm"]),
            }
        )
    write_csv(OUT / "modal_impedance_comparison.csv", modal_rows)

    old_coupling = coupling_summary(V139 / "coupling_metrics.csv")
    new_coupling = coupling_summary(V143_2X2 / "coupling_metrics.csv")
    coupling_rows = [
        {
            "group": name,
            "old_mean_coupling_db": old_coupling[name],
            "new_mean_coupling_db": new_coupling[name],
            "isolation_improvement_db": old_coupling[name] - new_coupling[name],
        }
        for name in ("x", "y", "diagonal")
    ]
    write_csv(OUT / "coupling_comparison.csv", coupling_rows)

    oracle = read_csv(V142 / "scene_oracle.csv")
    candidates = {
        (int(row["sample_index"]), int(row["candidate_index_in_scene"])): row
        for row in read_csv(V142 / "candidate_metrics.csv")
    }
    failure_rows = []
    counters: dict[int, Counter[str]] = {2: Counter(), 4: Counter(), 6: Counter()}
    for scene in oracle:
        if int(scene["reserve11_strict_pass"]):
            continue
        k_value = int(scene["k_value"])
        row = candidates[(int(scene["sample_index"]), int(scene["best_candidate_index_in_scene"]))]
        failures = []
        tests = {
            "pattern": float(row["pattern_margin_db"]) < 0.0,
            "active11": float(row["active_rl_floor_db"]) < 11.0,
            "hardware": float(row["hardware_margin_db"]) < 0.0,
            "dense": int(row["dense_constraint_pass"]) == 0,
        }
        for name, failed in tests.items():
            if failed:
                failures.append(name)
                counters[k_value][name] += 1
        failure_rows.append(
            {
                "sample_index": int(scene["sample_index"]),
                "k_value": k_value,
                "ratio": float(scene["ratio"]),
                "failed_gates": "|".join(failures),
                "pattern_margin_db": float(row["pattern_margin_db"]),
                "active_rl_floor_db": float(row["active_rl_floor_db"]),
                "hardware_margin_db": float(row["hardware_margin_db"]),
                "dense_constraint_pass": int(row["dense_constraint_pass"]),
            }
        )
    write_csv(OUT / "v142_failed_scene_gate_audit.csv", failure_rows)
    write_csv(
        OUT / "v142_failure_counts_by_k.csv",
        [
            {"k_value": k_value, **{name: counters[k_value][name] for name in ("pattern", "active11", "hardware", "dense")}}
            for k_value in (2, 4, 6)
        ],
    )

    metrics = [
        ("final_delta_s", "final_delta_s", "lower"),
        ("minimum_passive_rl_db", "minimum_passive_rl_db", "higher"),
        ("minimum_active_rl_db", "minimum_active_rl_db", "higher"),
        ("minimum_total_rl_db", "minimum_total_rl_db", "higher"),
        ("minimum_system_efficiency", "minimum_system_efficiency", "higher"),
    ]
    physical_rows = []
    for label, key, preferred in metrics:
        old_value, new_value = float(old_summary[key]), float(new_summary[key])
        physical_rows.append(
            {
                "metric": label,
                "old_v139": old_value,
                "new_v143": new_value,
                "new_minus_old": new_value - old_value,
                "preferred_direction": preferred,
            }
        )
    write_csv(OUT / "physical_gate_comparison.csv", physical_rows)

    v142_summary = read_json(V142 / "stage_summary.json")
    summary = {
        "v142": {
            "scene_count": v142_summary["scene_count"],
            "engineering_strict_oracle_count": v142_summary["engineering_strict_oracle_count"],
            "reserve11_counts_by_k": v142_summary["reserve11_counts_by_k"],
            "stop_v141_network_topology": True,
        },
        "v143_1x1": {
            "input_prescreen_gate_pass": one_summary["input_prescreen_gate_pass"],
            "input_impedance_ohm": [one_summary["input_resistance_ohm"], one_summary["input_reactance_ohm"]],
            "target_impedance_error_ohm": one_summary["target_impedance_error_ohm"],
            "passive_rl_db": one_summary["passive_rl_db"],
        },
        "v143_2x2": {
            "numerical_physical_gate_pass": new_summary["numerical_physical_gate_pass"],
            "strict_initial_gate_pass": new_summary["strict_initial_gate_pass"],
            "minimum_active_rl_db": new_summary["minimum_active_rl_db"],
            "minimum_total_rl_db": new_summary["minimum_total_rl_db"],
            "minimum_system_efficiency": new_summary["minimum_system_efficiency"],
            "y_coupling_isolation_improvement_db": old_coupling["y"] - new_coupling["y"],
            "stop_y_balanced_dual_branch_topology": True,
        },
        "decision": {
            "allow_crosscheck_or_three_frequency": False,
            "allow_4x4_or_16x16": False,
            "allow_eep_labels_or_critic": False,
            "next_authorized_branch": "Replace the free-standing differential radiator with a ground-backed/cavity-backed element or an element-level neutralization topology that can reduce x/y coupling by several dB while independently tuning self impedance. Do not continue tuning the mirrored secondary branch or add another external correction stage.",
        },
        "evidence_scope": "v1.42 is corrected-S4/mapped-EEP circuit evidence; v1.43 1x1 and 2x2 values are physical 10 GHz HFSS evidence. No 16x16 or EEP claim is made.",
    }
    write_json(OUT / "stage_summary.json", summary)
    write_json(OUT / "stage_decision.json", summary["decision"])
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
