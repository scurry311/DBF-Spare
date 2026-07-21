#!/usr/bin/env python3
"""Continue the grounded-patch 16x16 HFSS solution in auditable stages."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DEFAULT_USER_CONFIG = Path.home() / "Documents" / "Ansoft" / "ElectronicsDesktop2023.1" / "config" / "scurry.cfg"
DESIGN_NAME = "URA_GroundedPatch_10GHz"
SETUP_NAME = "Setup_10GHz"
NPORTS = 256
DELTA_LIMIT = 0.05
RL_LIMIT_DB = 10.0
RECIPROCITY_LIMIT = 1.0e-4
PASSIVITY_SIGMA_LIMIT = 1.001
SERIES_L_NH = 0.533
SERIES_Q = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("init", "run-stage", "analyze", "status"), required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--maximum-passes", type=int)
    parser.add_argument("--percent-refinement", type=float, default=15.0)
    parser.add_argument("--memory-stop-gb", type=float, default=18.0)
    parser.add_argument("--disk-stop-gb", type=float, default=10.0)
    parser.add_argument("--temp-disk-stop-gb", type=float, default=5.0)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--aedt-temp-dir", type=Path)
    parser.add_argument(
        "--solver-type",
        choices=("direct", "iterative", "ddm", "auto"),
        default="direct",
        help="HFSS driven solver used for this stage.",
    )
    parser.add_argument(
        "--iterative-residual",
        type=float,
        default=1.0e-4,
        help="Residual tolerance when --solver-type iterative is selected.",
    )
    parser.add_argument(
        "--save-fields",
        action="store_true",
        help="Retain adaptive fields. Disabled by default for S-matrix convergence.",
    )
    parser.add_argument(
        "--retry-failed-pass",
        action="store_true",
        help="Retry the same pass after execution failure or invalid S-matrix physics.",
    )
    return parser.parse_args()


def project_folder(out_dir: Path) -> Path:
    return out_dir / "grounded_patch_16x16"


def project_path(out_dir: Path) -> Path:
    return project_folder(out_dir) / "grounded_patch_16x16.aedt"


def touchstone_path(out_dir: Path) -> Path:
    return project_folder(out_dir) / "grounded_patch_16x16.s256p"


def vbs_path(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def continuation_vbs(
    project: Path,
    touchstone: Path,
    maximum_passes: int,
    percent_refinement: float,
    save_fields: bool,
    solver_type: str,
    iterative_residual: float,
) -> str:
    save_fields_vbs = "True" if save_fields else "False"
    driven_solver_type = {
        "direct": "Direct Solver",
        "iterative": "Iterative Solver",
        "ddm": "Domain Decomposition",
        "auto": "Auto Select Direct/Iterative",
    }[solver_type]
    iterative_options = ""
    if solver_type in ("iterative", "ddm"):
        iterative_options = (
            f'    "DrivenSolverType:=", "{driven_solver_type}", _\n'
            f'    "IterativeResidual:=", {iterative_residual:.8g}, _\n'
            f'    "DDMSolverResidual:=", {iterative_residual:.8g}, _\n'
        )
    else:
        iterative_options = f'    "DrivenSolverType:=", "{driven_solver_type}", _\n'
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oAnalysis, oSolutions, vars, variation
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vbs_path(project)}"
Set oProject = oDesktop.SetActiveProject("grounded_patch_16x16")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
oAnalysis.EditSetup "{SETUP_NAME}", Array( _
    "NAME:{SETUP_NAME}", _
    "SolveType:=", "Single", _
    "Frequency:=", "10GHz", _
    "MaxDeltaS:=", {DELTA_LIMIT}, _
    "MaximumPasses:=", {maximum_passes}, _
    "MinimumPasses:=", 2, _
    "MinimumConvergedPasses:=", 2, _
    "PercentRefinement:=", {percent_refinement:.6g}, _
    "BasisOrder:=", 1, _
    "DoLambdaRefine:=", True, _
    "DoMaterialLambda:=", True, _
    "SetLambdaTarget:=", False, _
    "UseMaxTetIncrease:=", False, _
    "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, _
    "SetPortMinMaxTri:=", False, _
{iterative_options}    "SaveAnyFields:=", {save_fields_vbs}, _
    "SaveRadFieldsOnly:=", False)
oProject.Save
oDesign.Analyze "{SETUP_NAME}"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("{SETUP_NAME}:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("{SETUP_NAME}:LastAdaptive"), 3, "{vbs_path(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


@contextmanager
def temporary_aedt_directory(temp_dir: Path | None):
    if temp_dir is None:
        yield
        return
    temp_dir = temp_dir.resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    config_path = DEFAULT_USER_CONFIG
    original = config_path.read_text(encoding="utf-8")
    target = str(temp_dir).replace(chr(92), "/")
    current_match = re.search(r"tempdirectory='([^']*)'", original)
    if current_match and current_match.group(1).rstrip("/").lower() == target.rstrip("/").lower():
        yield
        return
    replacement = re.sub(
        r"tempdirectory='[^']*'",
        f"tempdirectory='{target}'",
        original,
    )
    if replacement == original and "tempdirectory=''" not in original:
        raise ValueError(f"Cannot safely patch AEDT tempdirectory in {config_path}")
    config_path.write_text(replacement, encoding="utf-8")
    try:
        yield
    finally:
        config_path.write_text(original, encoding="utf-8")


def init_workspace(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_dir is None:
        raise ValueError("--source-dir is required for init")
    source_folder = args.source_dir / "grounded_patch_16x16"
    if not (source_folder / "grounded_patch_16x16.aedt").exists():
        raise FileNotFoundError(source_folder)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite staged workspace: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    shutil.copytree(source_folder, project_folder(args.out_dir))
    (args.out_dir / "stages").mkdir()
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(args.source_dir.resolve()),
        "working_project": str(project_path(args.out_dir).resolve()),
        "working_touchstone": str(touchstone_path(args.out_dir).resolve()),
        "policy": {
            "delta_s_max": DELTA_LIMIT,
            "consecutive_converged_passes": 2,
            "matched_passive_rl_min_db": RL_LIMIT_DB,
            "memory_stop_gb": float(args.memory_stop_gb),
            "disk_stop_gb": float(args.disk_stop_gb),
            "percent_refinement": float(args.percent_refinement),
        },
        "status": "initialized_from_source_project",
    }
    (args.out_dir / "staged_convergence_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_profile(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    current_pass: int | None = None
    deltas: dict[int, float] = {}
    matrices: dict[int, int] = {}
    memories: dict[int, int] = {}
    pass_pattern = re.compile(r"Name='Adaptive Pass\s+(\d+)'")
    delta_pattern = re.compile(r"Max Mag\. Delta S.*?,\s*([0-9.+\-Ee]+)\s*,")
    matrix_pattern = re.compile(
        r"ProfileItem\('Matrix Solve',\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*(\d+).*?Matrix size.*?,\s*(\d+)",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        pass_match = pass_pattern.search(line)
        if pass_match:
            current_pass = int(pass_match.group(1))
        delta_match = delta_pattern.search(line)
        if current_pass is not None and delta_match:
            deltas[current_pass] = float(delta_match.group(1))
        matrix_match = matrix_pattern.search(line)
        if current_pass is not None and matrix_match:
            memories[current_pass] = int(matrix_match.group(1))
            matrices[current_pass] = int(matrix_match.group(2))

    pass_numbers = sorted(set(deltas) | set(matrices))
    final_pass = max(pass_numbers, default=0)
    consecutive = 0
    for pass_number in sorted(deltas, reverse=True):
        if pass_number != final_pass - consecutive:
            break
        if deltas[pass_number] <= DELTA_LIMIT:
            consecutive += 1
        else:
            break
    return {
        "profile_path": str(path),
        "final_pass": final_pass,
        "delta_s_by_pass": {str(key): value for key, value in sorted(deltas.items())},
        "final_delta_s": deltas.get(final_pass),
        "consecutive_delta_pass_count": consecutive,
        "matrix_size_by_pass": {str(key): value for key, value in sorted(matrices.items())},
        "final_matrix_size": matrices.get(final_pass),
        "peak_matrix_size": max(matrices.values(), default=0),
        "solver_memory_gb_by_pass": {
            str(key): value / 1.0e6 for key, value in sorted(memories.items())
        },
        "peak_solver_memory_gb": max(memories.values(), default=0) / 1.0e6,
        "hfss_converged_message": bool(
            "Adaptive Passes converged" in text and "did not converge" not in text
        ),
    }


def select_profile(folder: Path) -> Path:
    candidates = list(folder.rglob("*.profile"))
    if not candidates:
        raise FileNotFoundError(f"No HFSS profile under {folder}")
    scored = []
    for path in candidates:
        parsed = parse_profile(path)
        scored.append((int(parsed["final_pass"]), path.stat().st_mtime_ns, path))
    return max(scored)[2]


def parse_touchstone(path: Path, nports: int) -> np.ndarray:
    tokens: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith(("!", "#", "[")):
            tokens.extend(float(value) for value in line.split())
    values = np.asarray(tokens[1:], dtype=float).reshape(nports * nports, 2)
    return (values[:, 0] * np.exp(1j * np.deg2rad(values[:, 1]))).reshape(nports, nports)


def s_metrics(path: Path) -> dict[str, Any]:
    s_raw = parse_touchstone(path, NPORTS).astype(np.complex128)
    identity = np.eye(NPORTS, dtype=np.complex128)
    z0 = 50.0
    z_ant = z0 * (identity + s_raw) @ np.linalg.inv(identity - s_raw)
    omega_l = 2.0 * np.pi * 10.0e9 * SERIES_L_NH * 1.0e-9
    series_impedance = omega_l / SERIES_Q + 1j * omega_l
    z_matched = z_ant + series_impedance * identity
    s_matched = (z_matched - z0 * identity) @ np.linalg.inv(z_matched + z0 * identity)
    raw_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s_raw)), 1.0e-15))
    matched_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s_matched)), 1.0e-15))
    return {
        "touchstone_complete": bool(s_raw.shape == (NPORTS, NPORTS) and np.all(np.isfinite(s_raw))),
        "s_reciprocity_max_abs": float(np.max(np.abs(s_raw - s_raw.T))),
        "s_passivity_sigma_max": float(np.max(np.linalg.svd(s_raw, compute_uv=False))),
        "raw_passive_rl_min_db": float(np.min(raw_rl)),
        "raw_passive_rl_median_db": float(np.median(raw_rl)),
        "matched_passive_rl_min_db": float(np.min(matched_rl)),
        "matched_passive_rl_median_db": float(np.median(matched_rl)),
        "matched_passive_rl_10db_port_pass_rate": float(np.mean(matched_rl >= RL_LIMIT_DB)),
    }


def analyze_stage(args: argparse.Namespace, maximum_passes: int, return_code: int | None) -> dict[str, Any]:
    folder = project_folder(args.out_dir)
    profile = select_profile(folder)
    metrics = parse_profile(profile)
    touchstone = touchstone_path(args.out_dir)
    current_pass_complete = bool(
        int(metrics.get("final_pass", 0)) == int(maximum_passes)
        and metrics.get("final_delta_s") is not None
        and return_code in (None, 0)
    )
    if touchstone.exists() and touchstone.stat().st_size > 1000 and current_pass_complete:
        metrics.update(s_metrics(touchstone))
        metrics["touchstone_stale"] = False
    else:
        metrics.update(
            {
                "touchstone_complete": False,
                "touchstone_stale": bool(touchstone.exists()),
                "matched_passive_rl_min_db": float("nan"),
                "matched_passive_rl_median_db": float("nan"),
                "matched_passive_rl_10db_port_pass_rate": float("nan"),
            }
        )
    metrics["requested_maximum_passes"] = int(maximum_passes)
    metrics["percent_refinement"] = float(args.percent_refinement)
    metrics["solver_type"] = args.solver_type
    metrics["iterative_residual"] = float(args.iterative_residual)
    metrics["solve_return_code"] = return_code
    reciprocity = float(metrics.get("s_reciprocity_max_abs", np.inf))
    passivity_sigma = float(metrics.get("s_passivity_sigma_max", np.inf))
    metrics["numerical_smatrix_valid"] = bool(
        metrics.get("touchstone_complete") is True
        and reciprocity <= RECIPROCITY_LIMIT
        and passivity_sigma <= PASSIVITY_SIGMA_LIMIT
    )
    metrics["reciprocity_limit"] = RECIPROCITY_LIMIT
    metrics["passivity_sigma_limit"] = PASSIVITY_SIGMA_LIMIT
    metrics["small_mesh_segment_count"] = sum(
        item.read_text(encoding="utf-8", errors="ignore").lower().count("small mesh segment")
        for item in folder.rglob("*.g3derr")
    )
    disk_free_gb = shutil.disk_usage(args.out_dir).free / 1.0e9
    metrics["disk_free_gb"] = disk_free_gb
    temp_disk_free_gb = None
    if args.aedt_temp_dir is not None:
        probe = args.aedt_temp_dir.resolve()
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        temp_disk_free_gb = shutil.disk_usage(probe).free / 1.0e9
    metrics["temp_disk_free_gb"] = temp_disk_free_gb

    memory_by_pass = [
        float(value) for _, value in sorted(
            ((int(key), value) for key, value in metrics.get("solver_memory_gb_by_pass", {}).items())
        )
    ]
    latest_memory_increment = (
        max(0.0, memory_by_pass[-1] - memory_by_pass[-2])
        if len(memory_by_pass) >= 2
        else 1.0
    )
    previous_percent_refinement = float(args.percent_refinement)
    latest_path = args.out_dir / "latest_stage_decision.json"
    if latest_path.exists():
        previous_metrics = json.loads(latest_path.read_text(encoding="utf-8"))
        previous_percent_refinement = float(
            previous_metrics.get("percent_refinement", args.percent_refinement)
        )
    refinement_scale = float(args.percent_refinement) / max(previous_percent_refinement, 1.0e-9)
    projected_increment = max(0.5, latest_memory_increment * refinement_scale * 1.2)
    current_memory = max(memory_by_pass[-1:], default=float("inf"))
    metrics["projected_next_memory_increment_gb"] = projected_increment
    metrics["memory_projection_refinement_scale"] = refinement_scale
    metrics["predicted_next_solver_memory_gb"] = current_memory + projected_increment
    metrics["strict_gate_pass"] = bool(
        return_code in (None, 0)
        and metrics.get("touchstone_complete") is True
        and metrics.get("numerical_smatrix_valid") is True
        and metrics.get("final_delta_s") is not None
        and float(metrics["final_delta_s"]) <= DELTA_LIMIT
        and int(metrics["consecutive_delta_pass_count"]) >= 2
        and float(metrics.get("matched_passive_rl_min_db", -np.inf)) >= RL_LIMIT_DB
    )
    metrics["resource_continue_allowed"] = bool(
        float(metrics["predicted_next_solver_memory_gb"]) < float(args.memory_stop_gb)
        and disk_free_gb >= float(args.disk_stop_gb)
        and (
            temp_disk_free_gb is None
            or temp_disk_free_gb >= float(args.temp_disk_stop_gb)
        )
    )
    delta_values = [
        float(value) for _, value in sorted(
            ((int(key), value) for key, value in metrics.get("delta_s_by_pass", {}).items())
        )
    ]
    recent_deltas = delta_values[-6:]
    recent_steps = [
        current - previous for previous, current in zip(recent_deltas, recent_deltas[1:])
    ]
    alternating_oscillation = bool(
        len(recent_steps) == 5
        and all(step != 0.0 for step in recent_steps)
        and all(
            np.sign(previous) != np.sign(current)
            for previous, current in zip(recent_steps, recent_steps[1:])
        )
        and max(recent_deltas) - min(recent_deltas) > DELTA_LIMIT
        and min(recent_deltas) > DELTA_LIMIT
    )
    metrics["recent_delta_s"] = recent_deltas
    metrics["alternating_delta_s_oscillation"] = alternating_oscillation
    metrics["convergence_strategy_continue_allowed"] = not alternating_oscillation
    metrics["solver_execution_error"] = bool(return_code not in (None, 0))
    if metrics["strict_gate_pass"]:
        metrics["decision"] = "strict_gate_pass_allow_fresh_eep_hfss_export"
    elif metrics["solver_execution_error"]:
        metrics["decision"] = "stop_solver_execution_error"
    elif not metrics["numerical_smatrix_valid"]:
        metrics["decision"] = "stop_invalid_smatrix_reciprocity_or_passivity"
    elif not metrics["resource_continue_allowed"]:
        metrics["decision"] = "stop_resource_ceiling"
    elif not metrics["convergence_strategy_continue_allowed"]:
        metrics["decision"] = "stop_convergence_oscillation_requires_port_mesh_stabilization"
    else:
        metrics["decision"] = "continue_next_staged_checkpoint"
    return metrics


def save_checkpoint(
    args: argparse.Namespace,
    maximum_passes: int,
    metrics: dict[str, Any],
    log_path: Path | None,
    stage_dir: Path | None = None,
) -> Path:
    stage_dir = stage_dir or args.out_dir / "stages" / f"pass{maximum_passes:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    profile = Path(metrics["profile_path"])
    shutil.copy2(profile, stage_dir / profile.name)
    touchstone = touchstone_path(args.out_dir)
    if touchstone.exists():
        shutil.copy2(touchstone, stage_dir / touchstone.name)
    if log_path is not None and log_path.exists() and log_path.parent != stage_dir:
        shutil.copy2(log_path, stage_dir / log_path.name)
    history_path = args.out_dir / "stage_history.csv"
    history = read_csv(history_path)
    history = [row for row in history if int(row["requested_maximum_passes"]) != maximum_passes]
    history.append(
        {
            "requested_maximum_passes": maximum_passes,
            "final_pass": metrics.get("final_pass"),
            "final_delta_s": metrics.get("final_delta_s"),
            "consecutive_delta_pass_count": metrics.get("consecutive_delta_pass_count"),
            "peak_matrix_size": metrics.get("peak_matrix_size"),
            "peak_solver_memory_gb": metrics.get("peak_solver_memory_gb"),
            "solver_type": metrics.get("solver_type"),
            "iterative_residual": metrics.get("iterative_residual"),
            "matched_passive_rl_min_db": metrics.get("matched_passive_rl_min_db"),
            "matched_passive_rl_10db_port_pass_rate": metrics.get(
                "matched_passive_rl_10db_port_pass_rate"
            ),
            "strict_gate_pass": int(bool(metrics.get("strict_gate_pass"))),
            "numerical_smatrix_valid": int(bool(metrics.get("numerical_smatrix_valid"))),
            "resource_continue_allowed": int(bool(metrics.get("resource_continue_allowed"))),
            "decision": metrics.get("decision"),
        }
    )
    history.sort(key=lambda row: int(row["requested_maximum_passes"]))
    write_csv(history_path, history)
    (args.out_dir / "latest_stage_decision.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return stage_dir


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.maximum_passes is None or args.maximum_passes < 3:
        raise ValueError("--maximum-passes >= 3 is required")
    if not project_path(args.out_dir).exists():
        raise FileNotFoundError("Run --mode init first")
    prior_history = read_csv(args.out_dir / "stage_history.csv")
    latest_decision = args.out_dir / "latest_stage_decision.json"
    latest: dict[str, Any] = {}
    if latest_decision.exists():
        latest = json.loads(latest_decision.read_text(encoding="utf-8"))
        if args.retry_failed_pass:
            reciprocity = float(latest.get("s_reciprocity_max_abs", float("inf")))
            passivity_sigma = float(latest.get("s_passivity_sigma_max", float("inf")))
            numerical_invalid = bool(
                latest.get("touchstone_complete") is not True
                or reciprocity > RECIPROCITY_LIMIT
                or passivity_sigma > PASSIVITY_SIGMA_LIMIT
            )
            if latest.get("solver_execution_error") is not True and not numerical_invalid:
                raise RuntimeError(
                    "Retry is allowed only after execution failure or invalid S-matrix physics"
                )
            if int(latest.get("requested_maximum_passes", -1)) != args.maximum_passes:
                raise RuntimeError("Retry must use the same failed maximum pass")
            if float(latest.get("peak_solver_memory_gb", float("inf"))) >= args.memory_stop_gb:
                raise RuntimeError("Failed pass peak memory exceeds the retry memory ceiling")
            if args.aedt_temp_dir is None:
                raise RuntimeError("Retry requires an explicit AEDT temporary directory")
            probe = args.aedt_temp_dir.resolve()
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            temp_free_gb = shutil.disk_usage(probe).free / 1.0e9
            if temp_free_gb < args.temp_disk_stop_gb:
                raise RuntimeError(
                    f"Retry temp disk has {temp_free_gb:.2f} GB free; "
                    f"requires {args.temp_disk_stop_gb:.2f} GB"
                )
        elif latest.get("resource_continue_allowed") is False:
            raise RuntimeError(
                "Latest checkpoint blocks another pass on predicted memory or disk resources"
            )
        if latest.get("convergence_strategy_continue_allowed") is False:
            raise RuntimeError(
                "Latest checkpoint blocks another pass because Delta S is oscillating; "
                "apply deterministic port-region mesh controls first"
            )
    if (
        prior_history
        and not args.retry_failed_pass
        and args.maximum_passes
        <= max(int(row["requested_maximum_passes"]) for row in prior_history)
    ):
        raise ValueError("maximum passes must increase at every staged continuation")
    stage_name = f"pass{args.maximum_passes:02d}"
    if args.retry_failed_pass:
        retry_number = 1
        while (args.out_dir / "stages" / f"{stage_name}_retry{retry_number:02d}").exists():
            retry_number += 1
        stage_name = f"{stage_name}_retry{retry_number:02d}"
    stage_dir = args.out_dir / "stages" / stage_name
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(stage_dir)
    stage_dir.mkdir(parents=True)
    vbs = stage_dir / f"continue_to_pass{args.maximum_passes:02d}.vbs"
    log = stage_dir / f"continue_to_pass{args.maximum_passes:02d}.log"
    existing_touchstone = touchstone_path(args.out_dir)
    if args.retry_failed_pass and existing_touchstone.exists():
        shutil.copy2(existing_touchstone, stage_dir / "touchstone_before_retry.s256p")
    vbs.write_text(
        continuation_vbs(
            project_path(args.out_dir),
            touchstone_path(args.out_dir),
            args.maximum_passes,
            args.percent_refinement,
            args.save_fields,
            args.solver_type,
            args.iterative_residual,
        ),
        encoding="ascii",
    )
    with temporary_aedt_directory(args.aedt_temp_dir):
        with log.open("w", encoding="utf-8", errors="replace") as handle:
            process = subprocess.run(
                [str(args.ansys_exe), "-ng", "-RunScriptAndExit", str(vbs.resolve())],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    metrics = analyze_stage(args, args.maximum_passes, int(process.returncode))
    save_checkpoint(args, args.maximum_passes, metrics, log, stage_dir=stage_dir)
    return metrics


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    maximum_passes = args.maximum_passes or max(
        (int(row["requested_maximum_passes"]) for row in read_csv(args.out_dir / "stage_history.csv")),
        default=2,
    )
    metrics = analyze_stage(args, maximum_passes, None)
    save_checkpoint(args, maximum_passes, metrics, None)
    return metrics


def status(args: argparse.Namespace) -> dict[str, Any]:
    latest = args.out_dir / "latest_stage_decision.json"
    return {
        "initialized": project_path(args.out_dir).exists(),
        "history": read_csv(args.out_dir / "stage_history.csv"),
        "latest": json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else None,
    }


def main() -> None:
    args = parse_args()
    result = {
        "init": init_workspace,
        "run-stage": run_stage,
        "analyze": analyze,
        "status": status,
    }[args.mode](args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
