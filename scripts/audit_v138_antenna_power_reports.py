#!/usr/bin/env python3
"""Audit v1.38 HFSS radiation efficiency against exported antenna powers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from run_v121_parametric_feed_post import aedt_processes
from run_v125_feedpoint_input_impedance import write_csv, write_json
from run_v128_true_balanced_dual_resonant import vp
from run_v130_fixed_reference_cps_transformer import read_json, resolve


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v138_vertical_differential_independent_validation_preregistered.json"
QUANTITIES = (
    "RadiationEfficiency",
    "SystemEfficiency",
    "RadiatedPower",
    "AcceptedPower",
    "IncidentPower",
    "ReflectedPower",
)


def export_script(project: Path, frequency_ghz: float, output: Path) -> str:
    lines = []
    for quantity in QUANTITIES:
        report = f"V138_PowerAudit_{quantity}"
        target = output / f"{quantity}.csv"
        lines.append(
            f'''Err.Clear
oReport.DeleteReports Array("{report}")
Err.Clear
oReport.CreateReport "{report}", "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("{frequency_ghz:g}GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("{quantity}"))
If Err.Number = 0 Then oReport.ExportToFile "{report}", "{vp(target)}"
statusFile.WriteLine "{quantity}," & CStr(Err.Number)
Err.Clear'''
        )
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oReport, fso, quantityFile, statusFile, quantities, item
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("V132_VerticalDifferential")
Set oReport = oDesign.GetModule("ReportSetup")
Set fso = CreateObject("Scripting.FileSystemObject")
Set quantityFile = fso.CreateTextFile("{vp(output / 'available_quantities.txt')}", True)
On Error Resume Next
quantities = oReport.GetAllQuantities("Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"))
For Each item In quantities
    quantityFile.WriteLine CStr(item)
Next
quantityFile.Close
Set statusFile = fso.CreateTextFile("{vp(output / 'export_status.csv')}", True)
statusFile.WriteLine "quantity,error_number"
{chr(10).join(lines)}
statusFile.Close
On Error GoTo 0
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def numeric_value(path: Path) -> float | None:
    if not path.exists() or path.stat().st_size < 10:
        return None
    values: list[float] = []
    with path.open(encoding="utf-8-sig", errors="ignore") as handle:
        for row in csv.reader(handle):
            for token in row[1:]:
                try:
                    value = float(token)
                except ValueError:
                    continue
                if math.isfinite(value):
                    values.append(value)
    return values[0] if values else None


def run(config: dict[str, Any]) -> dict[str, Any]:
    if aedt_processes():
        raise RuntimeError("Refusing the power-report audit while AEDT/HFSS is active")
    root = resolve(config["output_directory"])
    decision = read_json(root / "stage_decision.json")
    if not decision.get("three_frequency_efficiency_gate_pass"):
        raise RuntimeError("The v1.38 three-frequency solve is incomplete")
    source_root = root / "three_frequency_efficiency"
    cases = read_json(source_root / "case_manifest.json")["cases"]
    audit_root = root / "power_report_audit"
    if audit_root.exists():
        raise FileExistsError(f"Refusing to overwrite power-report audit: {audit_root}")
    audit_root.mkdir(parents=True)
    executable = str(resolve(config["ansys_executable"]))
    manifests = []
    for case in cases:
        case_root = audit_root / case["case_id"]
        case_root.mkdir()
        script = case_root / "export_power_reports.vbs"
        script.write_text(
            export_script(Path(case["project_path"]), float(case["frequency_ghz"]), case_root),
            encoding="ascii",
        )
        log = case_root / "export_power_reports.log"
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                [executable, "-ng", "-RunScriptAndExit", str(script.resolve())],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        manifests.append(
            {
                "case_id": case["case_id"],
                "frequency_ghz": float(case["frequency_ghz"]),
                "project_path": case["project_path"],
                "return_code": process.returncode,
                "output_directory": str(case_root.resolve()),
            }
        )
        if process.returncode != 0:
            raise RuntimeError(f"Power-report export failed for {case['case_id']}")
    write_json(audit_root / "manifest.json", {"cases": manifests})
    return {"case_count": len(manifests), "root": str(audit_root)}


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    root = resolve(config["output_directory"]) / "power_report_audit"
    manifests = read_json(root / "manifest.json")["cases"]
    rows: list[dict[str, Any]] = []
    for case in manifests:
        folder = Path(case["output_directory"])
        available = (
            [line.strip() for line in (folder / "available_quantities.txt").read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
            if (folder / "available_quantities.txt").exists()
            else []
        )
        values = {quantity: numeric_value(folder / f"{quantity}.csv") for quantity in QUANTITIES}
        power_ratio = None
        if values["RadiatedPower"] is not None and values["AcceptedPower"] not in (None, 0.0):
            power_ratio = values["RadiatedPower"] / values["AcceptedPower"]
        reported = values["RadiationEfficiency"]
        rows.append(
            {
                "case_id": case["case_id"],
                "frequency_ghz": case["frequency_ghz"],
                "available_quantity_count": len(available),
                "radiation_efficiency": reported,
                "system_efficiency": values["SystemEfficiency"],
                "radiated_power_w": values["RadiatedPower"],
                "accepted_power_w": values["AcceptedPower"],
                "incident_power_w": values["IncidentPower"],
                "reflected_power_w": values["ReflectedPower"],
                "radiated_over_accepted": power_ratio,
                "efficiency_vs_power_ratio_abs_error": (
                    abs(reported - power_ratio) if reported is not None and power_ratio is not None else None
                ),
                "efficiency_excess_above_unity": max((reported or 0.0) - 1.0, 0.0),
                "available_quantities_json": json.dumps(available, separators=(",", ":")),
            }
        )
    write_csv(root / "power_balance_metrics.csv", rows)
    ratio_available = all(row["radiated_over_accepted"] is not None for row in rows)
    maximum_excess = max(row["efficiency_excess_above_unity"] for row in rows)
    maximum_ratio_error = (
        max(row["efficiency_vs_power_ratio_abs_error"] for row in rows)
        if ratio_available
        else None
    )
    summary = {
        "frequency_count": len(rows),
        "power_ratio_available_all_frequencies": ratio_available,
        "minimum_reported_radiation_efficiency": min(row["radiation_efficiency"] for row in rows if row["radiation_efficiency"] is not None),
        "maximum_reported_radiation_efficiency": max(row["radiation_efficiency"] for row in rows if row["radiation_efficiency"] is not None),
        "maximum_efficiency_excess_above_unity": maximum_excess,
        "maximum_efficiency_vs_power_ratio_abs_error": maximum_ratio_error,
        "interpretation": (
            "HFSS radiation efficiency is internally consistent with RadiatedPower/AcceptedPower; values slightly above unity are retained as numerical power-balance error, not physical gain."
            if ratio_available and maximum_ratio_error is not None and maximum_ratio_error <= 1.0e-6
            else "The antenna-power reports do not independently close the efficiency balance; do not promote the efficiency gate."
        ),
        "efficiency_lower_bound_95_percent_supported": bool(
            ratio_available
            and maximum_ratio_error is not None
            and maximum_ratio_error <= 1.0e-6
            and all(row["radiation_efficiency"] is not None and row["radiation_efficiency"] >= 0.95 for row in rows)
            and maximum_excess <= 0.005
        ),
    }
    write_json(root / "power_balance_summary.json", summary)
    return {"summary": summary, "rows": rows}


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["output_directory"])
    summary = read_json(output / "power_report_audit" / "power_balance_summary.json")
    three_frequency = read_json(output / "three_frequency_efficiency" / "stage_summary.json")
    crosscheck = read_json(output / "independent_10ghz" / "crosscheck_summary.json")
    destination = output / "final_stage_decision.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite final v1.38 decision: {destination}")
    gate = bool(
        crosscheck["residual_warning_benign_gate_pass"]
        and three_frequency["three_frequency_efficiency_gate_pass"]
        and summary["efficiency_lower_bound_95_percent_supported"]
    )
    decision = {
        "stage": "E_power_balance_audit_complete",
        "independent_direct_ddm_gate_pass": crosscheck["residual_warning_benign_gate_pass"],
        "three_frequency_passive_rl_gate_pass": three_frequency["three_frequency_efficiency_gate_pass"],
        "power_balance_audit_pass": summary["efficiency_lower_bound_95_percent_supported"],
        "maximum_raw_radiation_efficiency_excess_above_unity": summary["maximum_efficiency_excess_above_unity"],
        "raw_efficiency_interpretation": "numerical power-balance error; do not report values above unity as physical efficiency",
        "v138_gate_pass": gate,
        "allow_physical_2x2": gate,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(destination, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("run", "analyze", "finalize"), required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    actions = {"run": run, "analyze": analyze, "finalize": finalize}
    result = actions[args.mode](config)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
