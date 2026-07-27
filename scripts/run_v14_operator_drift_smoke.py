#!/usr/bin/env python3
"""Run the pre-registered 4x4 HFSS operator-drift calibration smoke."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v14_operator_drift_preregistered.json"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v14_operator_drift_4x4_smoke_20260727_run01"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
ARRAY_SCRIPT = ROOT / "scripts" / "run_grounded_patch_array_rebuild.py"
EEP_SCRIPT = ROOT / "scripts" / "export_grounded_patch_eep_operator.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "all", "status"), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def profile_paths(out_dir: Path, name: str) -> dict[str, Path]:
    root = out_dir / "profiles" / name
    model = root / "model"
    array = model / "grounded_patch_4x4"
    return {
        "root": root,
        "model": model,
        "array": array,
        "project": array / "grounded_patch_4x4.aedt",
        "touchstone": array / "grounded_patch_4x4.s16p",
        "eep": root / "eep",
        "operator": root / "eep" / "grounded_patch_eep_operator_16port.npz",
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite operator-drift smoke: {args.out_dir}")
    protocol = load_json(args.protocol)
    args.out_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for profile in protocol["profiles"]:
        paths = profile_paths(args.out_dir, str(profile["name"]))
        rows.append(
            {
                **profile,
                "model_dir": str(paths["model"].resolve()),
                "project_path": str(paths["project"].resolve()),
                "touchstone_path": str(paths["touchstone"].resolve()),
                "eep_dir": str(paths["eep"].resolve()),
                "operator_path": str(paths["operator"].resolve()),
            }
        )
    write_csv(args.out_dir / "profile_manifest.csv", rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": str(args.protocol.resolve()),
        "profile_count": len(rows),
        "profiles": [row["name"] for row in rows],
        "evidence_scope": "4x4 HFSS S16 and complex EEP sensitivity calibration; not 16x16 HFSS",
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def command_for_profile(
    args: argparse.Namespace, profile: dict[str, Any], mode: str
) -> list[str]:
    command = [
        sys.executable,
        str(ARRAY_SCRIPT),
        "--mode",
        mode,
        "--side",
        "4",
        "--out-dir",
        str(profile_paths(args.out_dir, str(profile["name"]))["model"]),
        "--ansys-exe",
        str(args.ansys_exe),
        "--feed-model",
        "coax",
        "--frequency-ghz",
        str(profile.get("frequency_ghz", 10.0)),
    ]
    options = {
        "patch_length_y_mm": "--patch-length-y-mm",
        "patch_width_x_mm": "--patch-width-x-mm",
        "relative_permittivity": "--relative-permittivity",
        "substrate_thickness_mm": "--substrate-thickness-mm",
        "loss_tangent": "--loss-tangent",
    }
    for key, option in options.items():
        if key in profile:
            command.extend((option, str(profile[key])))
    return command


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.out_dir / "profile_manifest.csv").exists():
        raise RuntimeError("Prepare the operator-drift smoke before running")
    protocol = load_json(args.protocol)
    progress: list[dict[str, Any]] = []
    for profile in protocol["profiles"]:
        name = str(profile["name"])
        paths = profile_paths(args.out_dir, name)
        if not paths["operator"].exists():
            if not paths["project"].exists():
                run_checked(command_for_profile(args, profile, "run"), paths["root"] / "array_run.log")
            run_checked(command_for_profile(args, profile, "analyze"), paths["root"] / "array_analyze.log")
            if not (paths["eep"] / "eep_jobs.csv").exists():
                run_checked(
                    [
                        sys.executable,
                        str(EEP_SCRIPT),
                        "--mode",
                        "prepare",
                        "--project-path",
                        str(paths["project"]),
                        "--touchstone-path",
                        str(paths["touchstone"]),
                        "--out-dir",
                        str(paths["eep"]),
                        "--side",
                        "4",
                        "--ports-per-job",
                        "16",
                        "--ansys-exe",
                        str(args.ansys_exe),
                        "--frequency-ghz",
                        str(profile.get("frequency_ghz", 10.0)),
                    ],
                    paths["root"] / "eep_prepare.log",
                )
            for eep_mode in ("run", "analyze"):
                run_checked(
                    [
                        sys.executable,
                        str(EEP_SCRIPT),
                        "--mode",
                        eep_mode,
                        "--project-path",
                        str(paths["project"]),
                        "--touchstone-path",
                        str(paths["touchstone"]),
                        "--out-dir",
                        str(paths["eep"]),
                        "--side",
                        "4",
                        "--ports-per-job",
                        "16",
                        "--ansys-exe",
                        str(args.ansys_exe),
                        "--frequency-ghz",
                        str(profile.get("frequency_ghz", 10.0)),
                    ],
                    paths["root"] / f"eep_{eep_mode}.log",
                )
        progress.append(
            {
                "profile": name,
                "project_complete": paths["project"].exists(),
                "touchstone_complete": paths["touchstone"].exists(),
                "operator_complete": paths["operator"].exists(),
            }
        )
        write_csv(args.out_dir / "run_progress.csv", progress)
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_count": len(progress),
        "complete_operator_count": sum(bool(row["operator_complete"]) for row in progress),
        "run_complete": all(bool(row["operator_complete"]) for row in progress),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def field_metrics(nominal: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> dict[str, float]:
    ref = np.concatenate((nominal["etheta"].ravel(), nominal["ephi"].ravel()))
    value = np.concatenate((actual["etheta"].ravel(), actual["ephi"].ravel()))
    delta = value - ref
    ref_power = max(float(np.vdot(ref, ref).real), 1.0e-30)
    magnitude_ref = np.sqrt(np.abs(nominal["etheta"]) ** 2 + np.abs(nominal["ephi"]) ** 2)
    magnitude = np.sqrt(np.abs(actual["etheta"]) ** 2 + np.abs(actual["ephi"]) ** 2)
    threshold = float(np.max(magnitude_ref)) * 1.0e-6
    keep = magnitude_ref >= threshold
    db_ref = 20.0 * np.log10(np.maximum(magnitude_ref[keep], 1.0e-30))
    db_value = 20.0 * np.log10(np.maximum(magnitude[keep], 1.0e-30))
    return {
        "complex_field_nmse": float(np.vdot(delta, delta).real / ref_power),
        "magnitude_rmse_db": float(np.sqrt(np.mean((db_value - db_ref) ** 2))),
        "magnitude_max_abs_db": float(np.max(np.abs(db_value - db_ref))),
    }


def load_operator(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.protocol)
    operators: dict[str, dict[str, np.ndarray]] = {}
    for profile in protocol["profiles"]:
        name = str(profile["name"])
        path = profile_paths(args.out_dir, name)["operator"]
        if not path.exists():
            raise FileNotFoundError(f"Missing operator for {name}: {path}")
        operators[name] = load_operator(path)
    nominal = operators["nominal"]
    rows: list[dict[str, Any]] = []
    for profile in protocol["profiles"]:
        name = str(profile["name"])
        paths = profile_paths(args.out_dir, name)
        operator = operators[name]
        if not (
            np.array_equal(operator["theta_deg"], nominal["theta_deg"])
            and np.array_equal(operator["phi_deg"], nominal["phi_deg"])
        ):
            raise RuntimeError(f"EEP grid mismatch for {name}")
        s_delta = np.asarray(operator["s_raw"], dtype=np.complex128) - np.asarray(
            nominal["s_raw"], dtype=np.complex128
        )
        s_reference = max(float(np.linalg.norm(nominal["s_raw"])), 1.0e-30)
        stage = load_json(paths["model"] / "stage_summary.json")
        eep = load_json(paths["eep"] / "operator_analysis_summary.json")
        rows.append(
            {
                "profile": name,
                "frequency_ghz": float(profile.get("frequency_ghz", 10.0)),
                "patch_length_y_mm": float(profile.get("patch_length_y_mm", 9.35)),
                "relative_permittivity": float(profile.get("relative_permittivity", 2.2)),
                **field_metrics(nominal, operator),
                "s_raw_delta_max_abs": float(np.max(np.abs(s_delta))),
                "s_raw_delta_relative_fro": float(np.linalg.norm(s_delta) / s_reference),
                "solve_converged": bool(stage["metric"].get("converged")),
                "final_delta_s": stage["metric"].get("final_delta_s"),
                "port_topology_warning_count": int(stage["metric"].get("port_topology_warning_count", 0)),
                "s_reciprocity_max_abs": float(eep["s_reciprocity_max_abs"]),
                "s_passivity_sigma_max": float(eep["s_passivity_sigma_max"]),
                "matched_passive_rl_min_db": float(eep["matched_passive_rl_min_db"]),
                "structural_gate_pass": bool(eep["structural_gate_pass"]),
            }
        )
    write_csv(args.out_dir / "operator_drift_metrics.csv", rows)
    non_nominal = [row for row in rows if row["profile"] != "nominal"]
    gate = bool(
        all(row["solve_converged"] for row in rows)
        and all(float(row["final_delta_s"]) <= 0.05 for row in rows)
        and all(row["port_topology_warning_count"] == 0 for row in rows)
        and all(row["structural_gate_pass"] for row in rows)
        and all(row["complex_field_nmse"] > 1.0e-8 for row in non_nominal)
        and all(row["s_raw_delta_max_abs"] > 1.0e-5 for row in non_nominal)
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_count": len(rows),
        "all_hfss_operators_valid": all(row["structural_gate_pass"] for row in rows),
        "all_non_nominal_residuals_nonzero": all(
            row["complex_field_nmse"] > 1.0e-8 for row in non_nominal
        ),
        "operator_drift_calibration_gate_pass": gate,
        "field_nmse_range_non_nominal": [
            min(row["complex_field_nmse"] for row in non_nominal),
            max(row["complex_field_nmse"] for row in non_nominal),
        ],
        "s_delta_max_range_non_nominal": [
            min(row["s_raw_delta_max_abs"] for row in non_nominal),
            max(row["s_raw_delta_max_abs"] for row in non_nominal),
        ],
        "label_scope": "4x4 HFSS-calibrated operator sensitivity; not 16x16 HFSS",
        "allow_16x16_proxy_dataset": gate,
    }
    (args.out_dir / "drift_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def status(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.protocol)
    rows = []
    for profile in protocol["profiles"]:
        name = str(profile["name"])
        paths = profile_paths(args.out_dir, name)
        rows.append(
            {
                "profile": name,
                "project": paths["project"].exists(),
                "touchstone": paths["touchstone"].exists(),
                "operator": paths["operator"].exists(),
            }
        )
    return {
        "prepared": (args.out_dir / "profile_manifest.csv").exists(),
        "profiles": rows,
        "complete_operator_count": sum(row["operator"] for row in rows),
        "analyzed": (args.out_dir / "drift_calibration_summary.json").exists(),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        result = {"prepare": prepare(args), "run": run(args), "analyze": analyze(args)}
    else:
        result = {
            "prepare": prepare,
            "run": run,
            "analyze": analyze,
            "status": status,
        }[args.mode](args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
