#!/usr/bin/env python3
"""Build and gate the v1.49 SIW cavity-backed stacked-patch element.

The workflow is evidence-gated. Periodic, finite-array, label-generation, and
critic stages cannot be opened by a surrogate or an incomplete HFSS export.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "v149_siw_cavity_stacked_patch_preregistered.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="ascii")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def scan_states(config: dict[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "frequency_ghz": float(frequency),
            "theta_deg": float(theta),
            "phi_deg": float(phi),
        }
        for frequency in config["frequencies_ghz"]
        for theta in config["scan"]["theta_deg"]
        for phi in config["scan"]["phi_deg"]
    ]


def validate_config(config: dict[str, Any]) -> None:
    geometry = config["nominal_geometry"]
    if float(geometry["period_x_mm"]) != 15.0:
        raise ValueError("v1.49 requires period_x_mm = 15.0")
    if float(geometry["period_y_mm"]) != 15.0:
        raise ValueError("v1.49 requires period_y_mm = 15.0")
    total_height = (
        float(geometry["main_substrate_thickness_mm"])
        + float(geometry["stack_spacer_thickness_mm"])
        + float(geometry["copper_thickness_mm"])
    )
    if total_height > 3.5:
        raise ValueError("Stacked element exceeds the 3.5 mm height limit")
    sample_count = int(config["doe"]["sample_count"])
    if not 12 <= sample_count <= 20:
        raise ValueError("Periodic DOE sample_count must be in [12, 20]")
    pareto_count = int(config["doe"]["pareto_candidate_count"])
    if not 3 <= pareto_count <= 5:
        raise ValueError("Pareto candidate count must be in [3, 5]")
    if len(config["manufacturing_ranges"]) not in range(10, 13):
        raise ValueError("Exactly 10-12 physical variables may be optimized")
    expected_states = int(
        config["gates"]["required_periodic_scan_state_count"]
    )
    if len(scan_states(config)) != expected_states:
        raise ValueError("Scan-state count does not match the frozen gate")
    for forbidden in (
        "allow_2x2",
        "allow_4x4",
        "allow_16x16",
        "allow_training_labels",
        "allow_critic_training",
    ):
        if config["scope"].get(forbidden):
            raise ValueError(f"Preregistered scope must lock {forbidden}")


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def memory_available_gib() -> float:
    if hasattr(ctypes, "windll"):
        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available_physical / (1024.0**3)
    return math.nan


def aedt_processes() -> list[str]:
    result = subprocess.run(
        ["tasklist", "/fo", "csv", "/nh"],
        capture_output=True,
        text=True,
        check=False,
    )
    names = ("ansysedt.exe", "hfss.exe", "ansysedtsv.exe")
    return [
        line
        for line in result.stdout.splitlines()
        if any(name in line.lower() for name in names)
    ]


def memory_allows_stage(
    config: dict[str, Any],
    free_memory_gib: float,
    aedt_instance_count: int,
) -> bool:
    minimum = float(
        config["resources"]["minimum_free_memory_before_solve_gib"]
    )
    maximum_instances = int(
        config["resources"]["maximum_concurrent_aedt_instances"]
    )
    return (
        math.isfinite(free_memory_gib)
        and free_memory_gib >= minimum
        and aedt_instance_count < maximum_instances
    )


def evaluate_periodic_gate(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "real_hfss_evidence": (
            metrics.get("evidence_source") == "HFSS_periodic_fullwave"
        ),
        "evidence_complete": metrics.get("evidence_complete") is True,
        "scan_state_count": int(metrics.get("scan_state_count", 0))
        >= int(gates["required_periodic_scan_state_count"]),
        "active_rl": float(metrics.get("minimum_active_rl_db", -math.inf))
        >= float(gates["minimum_periodic_active_rl_db"]),
        "broadside_passive_rl": float(
            metrics.get("broadside_passive_rl_db", -math.inf)
        )
        >= float(gates["minimum_broadside_passive_rl_db"]),
        "efficiency": float(metrics.get("minimum_efficiency", -math.inf))
        >= float(gates["minimum_periodic_efficiency"]),
        "scan_gain_drop": float(
            metrics.get("maximum_scan_gain_drop_db", math.inf)
        )
        <= float(gates["maximum_scan_gain_drop_db"]),
        "delta_s": float(metrics.get("maximum_final_delta_s", math.inf))
        <= float(gates["maximum_final_delta_s"]),
        "no_scan_blindness": metrics.get("scan_blindness_detected") is False,
        "critical_warnings": int(
            metrics.get("critical_warning_count", 10**9)
        )
        <= int(gates["maximum_critical_warning_count"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def stage_decision_after_periodic(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    gate = evaluate_periodic_gate(metrics, gates)
    return {
        "stage": "periodic_gate_evaluated",
        "periodic_gate_passed": gate["passed"],
        "periodic_gate_checks": gate["checks"],
        "failed_checks": gate["failed_checks"],
        "allow_1x1": gate["passed"],
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "Periodic full-wave gate passed; only the finite 1x1 stage is open."
            if gate["passed"]
            else "Periodic full-wave gate failed or lacks complete evidence; all finite-array and learning stages remain locked."
        ),
    }


def allocate_output_root(config: dict[str, Any]) -> Path:
    prefix = resolve(config["output_prefix"])
    for index in range(1, 100):
        candidate = Path(f"{prefix}{index:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("No free v1.49 run index remains")


def status(config: dict[str, Any]) -> dict[str, Any]:
    processes = aedt_processes()
    free = memory_available_gib()
    return {
        "protocol": config["protocol"],
        "free_memory_gib": free,
        "aedt_process_count": len(processes),
        "aedt_processes": processes,
        "solve_preflight_pass": memory_allows_stage(
            config, free, len(processes)
        ),
        "locked_stages": [
            name
            for name, allowed in config["scope"].items()
            if name.startswith("allow_") and not allowed
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("validate-config", "status")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    result = {"valid": True} if args.command == "validate-config" else status(config)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
