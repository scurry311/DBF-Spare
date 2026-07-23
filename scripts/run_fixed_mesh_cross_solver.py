#!/usr/bin/env python3
"""Prepare and run an auditable fixed-mesh direct/DDM cross-solver check."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ddm_recovery_stage import profile_metrics  # noqa: E402
from run_staged_16x16_convergence import (  # noqa: E402
    NPORTS,
    PASSIVITY_SIGMA_LIMIT,
    RECIPROCITY_LIMIT,
    parse_touchstone,
    s_metrics,
)


ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
SOURCE_RUN = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02"
)
SOURCE_PROJECT = SOURCE_RUN / "grounded_patch_16x16" / "grounded_patch_16x16.aedt"
SOURCE_TOUCHSTONE = (
    SOURCE_RUN / "stages" / "pass05" / "grounded_patch_16x16_ddm_pass05.s256p"
)
DEFAULT_RUN = ROOT / "hfss_outputs" / "fixed_mesh_cross_solver_20260723_run01"
TARGET_NAME = "frozen_mesh_cross_solver"
DESIGN = "URA_GroundedPatch_10GHz"
SOURCE_SETUP = "Setup_DDM_Recovery : LastAdaptive"
DIRECT_SETUP = "Setup_Frozen_Direct"
DDM_SETUP = "Setup_Frozen_DDM"

MAX_DELTA_S = 0.05
PREFERRED_DELTA_S = 0.02
MIN_MATCHED_RL_DB = 10.0


class MemoryStatus(ctypes.Structure):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "controller", "analyze", "status"), required=True
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def memory_metrics() -> dict[str, float]:
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    scale = 1024.0**3
    return {
        "physical_total_gb": float(status.total_physical / scale),
        "physical_available_gb": float(status.available_physical / scale),
        "commit_limit_gb": float(status.total_page_file / scale),
        "commit_headroom_gb": float(status.available_page_file / scale),
    }


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_digest(text: str, pattern: str) -> tuple[int, str]:
    blocks = [match.group(0) for match in re.finditer(pattern, text, re.DOTALL)]
    digest = hashlib.sha256("\n".join(blocks).encode("utf-8")).hexdigest()
    return len(blocks), digest


def status_path(run_dir: Path) -> Path:
    return run_dir / "fixed_mesh_cross_solver_status.json"


def read_status(run_dir: Path) -> dict[str, Any]:
    path = status_path(run_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_status(run_dir: Path, **values: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    current = read_status(run_dir)
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path(run_dir).write_text(json.dumps(current, indent=2), encoding="utf-8")


def mesh_link(source_project: Path) -> str:
    return f'''Array( _
        "NAME:MeshLink", _
        "ImportMesh:=", True, _
        "Project:=", "{vp(source_project)}", _
        "Product:=", "HFSS", _
        "Design:=", "{DESIGN}", _
        "Soln:=", "{SOURCE_SETUP}", _
        Array("NAME:Params"), _
        "ForceSourceToSolve:=", False, _
        "PreservePartnerSoln:=", True, _
        "PathRelativeTo:=", "TargetProject")'''


def setup_array(name: str, solver: str, source_project: Path) -> str:
    solver_options = f'    "DrivenSolverType:=", "{solver}", _\n'
    if solver == "Domain Decomposition":
        solver_options += (
            '    "IterativeResidual:=", 0.000001, _\n'
            '    "DDMSolverResidual:=", 0.000001, _\n'
        )
    return f'''Array( _
    "NAME:{name}", _
    "SolveType:=", "Single", _
    "Frequency:=", "10GHz", _
    "MaxDeltaS:=", 0.05, _
    "UseMatrixConv:=", False, _
    "MaximumPasses:=", 1, _
    "MinimumPasses:=", 1, _
    "MinimumConvergedPasses:=", 1, _
    "PercentRefinement:=", 1, _
    "IsEnabled:=", True, _
    {mesh_link(source_project)}, _
    "BasisOrder:=", 1, _
    "DoLambdaRefine:=", False, _
    "DoMaterialLambda:=", False, _
    "SetLambdaTarget:=", False, _
    "UseMaxTetIncrease:=", False, _
    "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, _
    "SetPortMinMaxTri:=", False, _
{solver_options}    "SaveRadFieldsOnly:=", False, _
    "SaveAnyFields:=", False)'''


def prepare_vbs(target: Path, source: Path, marker: Path) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oAnalysis, fso, outFile
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{vp(marker)}", True)
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(target)}"
Set oProject = oDesktop.SetActiveProject("{TARGET_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN}")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
On Error Resume Next
oDesign.DeleteFullVariation "All", False
oAnalysis.DeleteSetups Array("Setup_10GHz")
oAnalysis.DeleteSetups Array("Setup_DDM_Recovery")
oAnalysis.DeleteSetups Array("Setup_VolFeed_DDM")
oAnalysis.DeleteSetups Array("{DIRECT_SETUP}")
oAnalysis.DeleteSetups Array("{DDM_SETUP}")
On Error GoTo 0
oAnalysis.InsertSetup "HfssDriven", {setup_array(DIRECT_SETUP, "Direct Solver", source)}
oAnalysis.InsertSetup "HfssDriven", {setup_array(DDM_SETUP, "Domain Decomposition", source)}
oProject.Save
outFile.WriteLine "PREPARED=1"
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def solve_vbs(target: Path, setup: str, touchstone: Path, messages: Path) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oSolutions, fso, outFile
Dim values, variation, msgs, i, analyzeErr, analyzeDescription
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{vp(messages)}", True)
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(target)}"
Set oProject = oDesktop.SetActiveProject("{TARGET_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN}")
On Error Resume Next
Err.Clear
oDesign.Analyze "{setup}"
analyzeErr = Err.Number
analyzeDescription = Err.Description
On Error GoTo 0
outFile.WriteLine "ANALYZE_ERR_NUMBER=" & CStr(analyzeErr)
outFile.WriteLine "ANALYZE_ERR_DESCRIPTION=" & analyzeDescription
msgs = oDesktop.GetMessages(oProject.GetName(), oDesign.GetName(), 0)
If IsArray(msgs) Then
    For i = LBound(msgs) To UBound(msgs)
        outFile.WriteLine "AEDT_MESSAGE=" & CStr(msgs(i))
    Next
End If
If analyzeErr = 0 Then
    Set oSolutions = oDesign.GetModule("Solutions")
    values = oSolutions.ListVariations("{setup}:LastAdaptive")
    variation = CStr(values(LBound(values)))
    oSolutions.ExportNetworkData variation, Array("{setup}:LastAdaptive"), 3, _
        "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
End If
oProject.Save
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def prepare(run_dir: Path) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    if not SOURCE_PROJECT.exists() or not SOURCE_TOUCHSTONE.exists():
        raise FileNotFoundError("The old pass5 source project or S256 is missing")
    project_dir = run_dir / "project"
    project_dir.mkdir(parents=True)
    target = project_dir / f"{TARGET_NAME}.aedt"
    shutil.copy2(SOURCE_PROJECT, target)
    marker = run_dir / "prepare_marker.txt"
    vbs = run_dir / "prepare_fixed_mesh_cross_solver.vbs"
    vbs.write_text(prepare_vbs(target, SOURCE_PROJECT, marker), encoding="ascii")
    log = run_dir / "prepare_fixed_mesh_cross_solver.log"
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(
            [str(ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    source_text = SOURCE_PROJECT.read_text(encoding="utf-8", errors="ignore")
    target_text = target.read_text(encoding="utf-8", errors="ignore")
    port_pattern = r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'"
    mesh_pattern = (
        r"\$begin '(?:PortFeedUniform_0p180mm|FeedSheetUniform_0p180mm)'"
        r".*?\$end '(?:PortFeedUniform_0p180mm|FeedSheetUniform_0p180mm)'"
    )
    source_ports = block_digest(source_text, port_pattern)
    target_ports = block_digest(target_text, port_pattern)
    source_mesh = block_digest(source_text, mesh_pattern)
    target_mesh = block_digest(target_text, mesh_pattern)
    source_path_token = str(SOURCE_PROJECT.resolve()).replace("\\", "\\\\")
    setup_bindings_present = bool(
        DIRECT_SETUP in target_text
        and DDM_SETUP in target_text
        and SOURCE_SETUP in target_text
        and (str(SOURCE_PROJECT.resolve()) in target_text or source_path_token in target_text)
    )
    passed = bool(
        result.returncode == 0
        and marker.exists()
        and source_ports == target_ports
        and source_ports[0] == NPORTS
        and source_mesh == target_mesh
        and setup_bindings_present
    )
    summary = {
        "source_project": str(SOURCE_PROJECT),
        "source_touchstone": str(SOURCE_TOUCHSTONE),
        "target_project": str(target),
        "source_project_sha256": sha256(SOURCE_PROJECT),
        "source_touchstone_sha256": sha256(SOURCE_TOUCHSTONE),
        "prepare_return_code": int(result.returncode),
        "port_count": source_ports[0],
        "port_definition_hash_unchanged": source_ports == target_ports,
        "surface_mesh_operation_hash_unchanged": source_mesh == target_mesh,
        "fixed_setup_bindings_present": setup_bindings_present,
        "source_mesh_solution": SOURCE_SETUP,
        "direct_setup": DIRECT_SETUP,
        "ddm_setup": DDM_SETUP,
        "maximum_passes": 1,
        "adaptive_refinement_enabled": False,
        "prepare_gate_pass": passed,
        "training_labels_locked": True,
    }
    (run_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_status(
        run_dir,
        state="prepared" if passed else "prepare_failed",
        prepare=summary,
        training_labels_locked=True,
    )
    if not passed:
        raise RuntimeError("Fixed-mesh project preparation gate failed")
    return summary


def wait_for_resources(run_dir: Path, solver: str) -> None:
    requirements = {
        "ddm": {"physical_available_gb": 5.0, "commit_headroom_gb": 20.0},
        "direct": {"physical_available_gb": 6.0, "commit_headroom_gb": 24.0},
    }[solver]
    deadline = time.time() + 8.0 * 3600.0
    while True:
        memory = memory_metrics()
        passed = all(memory[key] >= value for key, value in requirements.items())
        write_status(
            run_dir,
            state=(f"resource_gate_passed_{solver}" if passed else f"waiting_memory_{solver}"),
            current_solver=solver,
            memory=memory,
            memory_requirements=requirements,
        )
        if passed:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Memory gate for {solver} did not pass within eight hours")
        time.sleep(60.0)


def run_solver(run_dir: Path, solver: str) -> dict[str, Any]:
    setup = DDM_SETUP if solver == "ddm" else DIRECT_SETUP
    target = run_dir / "project" / f"{TARGET_NAME}.aedt"
    result_dir = run_dir / solver
    result_dir.mkdir(parents=True, exist_ok=True)
    touchstone = result_dir / f"frozen_{solver}.s256p"
    messages = result_dir / f"frozen_{solver}_messages.txt"
    vbs = result_dir / f"run_frozen_{solver}.vbs"
    vbs.write_text(solve_vbs(target, setup, touchstone, messages), encoding="ascii")
    stdout = result_dir / f"frozen_{solver}.stdout.log"
    stderr = result_dir / f"frozen_{solver}.stderr.log"
    results_dir = target.with_suffix(".aedtresults")
    before = {path.resolve() for path in results_dir.rglob("*.profile")} if results_dir.exists() else set()
    write_status(run_dir, state=f"running_{solver}", current_solver=solver)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(
            [str(ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=out,
            stderr=err,
        )
        low_resource_count = 0
        while process.poll() is None:
            memory = memory_metrics()
            low = bool(
                memory["physical_available_gb"] < 1.25
                or memory["commit_headroom_gb"] < 2.5
            )
            low_resource_count = low_resource_count + 1 if low else 0
            write_status(
                run_dir,
                state=f"running_{solver}",
                current_solver=solver,
                solver_pid=process.pid,
                memory=memory,
                consecutive_low_resource_polls=low_resource_count,
            )
            if low_resource_count >= 3:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                process.wait(timeout=120)
                write_status(run_dir, state=f"stopped_resource_guard_{solver}")
                raise MemoryError(f"Stopped {solver} after three low-resource polls")
            time.sleep(60.0)
        return_code = int(process.returncode)

    profiles = list(results_dir.rglob("*.profile")) if results_dir.exists() else []
    new_profiles = [path for path in profiles if path.resolve() not in before]
    candidates = new_profiles or profiles
    profile = max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None
    message_text = messages.read_text(encoding="utf-8", errors="ignore") if messages.exists() else ""
    solve_ok = bool(
        return_code == 0
        and touchstone.exists()
        and touchstone.stat().st_size > 1_000_000
        and "ANALYZE_ERR_NUMBER=0" in message_text
        and profile is not None
    )
    metrics: dict[str, Any] = {
        "solver": solver,
        "setup": setup,
        "return_code": return_code,
        "touchstone": str(touchstone),
        "profile": str(profile) if profile else None,
        "solve_gate_pass": solve_ok,
    }
    if solve_ok and profile is not None:
        copied_profile = result_dir / profile.name
        if profile.resolve() != copied_profile.resolve():
            shutil.copy2(profile, copied_profile)
        metrics.update(profile_metrics(copied_profile))
        metrics.update(s_metrics(touchstone))
        metrics["numerical_smatrix_valid"] = bool(
            metrics["touchstone_complete"]
            and float(metrics["s_reciprocity_max_abs"]) <= RECIPROCITY_LIMIT
            and float(metrics["s_passivity_sigma_max"]) <= PASSIVITY_SIGMA_LIMIT
        )
    (result_dir / "solver_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if not solve_ok:
        raise RuntimeError(f"Frozen-mesh {solver} solve or S256 export failed")
    return metrics


def compare(first: Path, second: Path) -> dict[str, Any]:
    first_s = parse_touchstone(first, NPORTS)
    second_s = parse_touchstone(second, NPORTS)
    delta = np.abs(second_s - first_s)
    diagonal = np.diag(delta)
    off_diagonal = delta[~np.eye(NPORTS, dtype=bool)]
    return {
        "first": str(first),
        "second": str(second),
        "max_abs_delta_s": float(np.max(delta)),
        "rms_abs_delta_s": float(np.sqrt(np.mean(delta**2))),
        "max_diagonal_delta_s": float(np.max(diagonal)),
        "median_diagonal_delta_s": float(np.median(diagonal)),
        "fraction_diagonal_gt_0p05": float(np.mean(diagonal > MAX_DELTA_S)),
        "max_offdiagonal_delta_s": float(np.max(off_diagonal)),
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    direct_path = run_dir / "direct" / "frozen_direct.s256p"
    ddm_path = run_dir / "ddm" / "frozen_ddm.s256p"
    if not direct_path.exists() or not ddm_path.exists():
        raise FileNotFoundError("Both frozen direct and DDM S256 files are required")
    direct_metrics = s_metrics(direct_path)
    ddm_metrics = s_metrics(ddm_path)
    comparisons = {
        "direct_vs_ddm": compare(direct_path, ddm_path),
        "source_pass05_vs_direct": compare(SOURCE_TOUCHSTONE, direct_path),
        "source_pass05_vs_ddm": compare(SOURCE_TOUCHSTONE, ddm_path),
    }
    numerical_valid = bool(
        direct_metrics["touchstone_complete"]
        and ddm_metrics["touchstone_complete"]
        and float(direct_metrics["s_reciprocity_max_abs"]) <= RECIPROCITY_LIMIT
        and float(ddm_metrics["s_reciprocity_max_abs"]) <= RECIPROCITY_LIMIT
        and float(direct_metrics["s_passivity_sigma_max"]) <= PASSIVITY_SIGMA_LIMIT
        and float(ddm_metrics["s_passivity_sigma_max"]) <= PASSIVITY_SIGMA_LIMIT
    )
    cross_gate = comparisons["direct_vs_ddm"]["max_abs_delta_s"] <= MAX_DELTA_S
    source_gate = bool(
        comparisons["source_pass05_vs_direct"]["max_abs_delta_s"] <= MAX_DELTA_S
        and comparisons["source_pass05_vs_ddm"]["max_abs_delta_s"] <= MAX_DELTA_S
    )
    rl_gate = bool(
        float(direct_metrics["matched_passive_rl_min_db"]) >= MIN_MATCHED_RL_DB
        and float(ddm_metrics["matched_passive_rl_min_db"]) >= MIN_MATCHED_RL_DB
    )
    accepted = bool(numerical_valid and cross_gate and source_gate and rl_gate)
    result = {
        "scope": "old pass5 frozen tetrahedral mesh direct/DDM cross validation",
        "direct_metrics": direct_metrics,
        "ddm_metrics": ddm_metrics,
        "comparisons": comparisons,
        "gates": {
            "numerical_smatrix_valid": numerical_valid,
            "direct_ddm_max_delta_s_le_0p05": cross_gate,
            "direct_ddm_preferred_delta_s_le_0p02": bool(
                comparisons["direct_vs_ddm"]["max_abs_delta_s"] <= PREFERRED_DELTA_S
            ),
            "both_reproduce_source_pass05_le_0p05": source_gate,
            "both_matched_min_rl_ge_10db": rl_gate,
        },
        "engineering_baseline_accepted": accepted,
        "training_labels_locked": True,
        "decision": (
            "accept_old_pass05_as_fixed_engineering_baseline"
            if accepted
            else "discard_old_pass05_and_build_deterministic_uniform_mesh_baseline"
        ),
    }
    (run_dir / "cross_solver_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# Fixed-mesh cross-solver validation",
        "",
        f"- Direct vs DDM max |Delta S|: {comparisons['direct_vs_ddm']['max_abs_delta_s']:.6f}",
        f"- Source vs direct max |Delta S|: {comparisons['source_pass05_vs_direct']['max_abs_delta_s']:.6f}",
        f"- Source vs DDM max |Delta S|: {comparisons['source_pass05_vs_ddm']['max_abs_delta_s']:.6f}",
        f"- Direct matched minimum RL: {direct_metrics['matched_passive_rl_min_db']:.3f} dB",
        f"- DDM matched minimum RL: {ddm_metrics['matched_passive_rl_min_db']:.3f} dB",
        f"- Decision: {result['decision']}",
        "",
        "Training labels remain locked until the accepted baseline is used for a separately gated EEP/HFSS rebuild.",
    ]
    (run_dir / "cross_solver_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_status(
        run_dir,
        state="completed_pass" if accepted else "completed_fail_discard_old_pass05",
        validation=result,
        training_labels_locked=True,
    )
    return result


def controller(run_dir: Path) -> None:
    summary = run_dir / "prepare_summary.json"
    if not summary.exists() or not json.loads(summary.read_text(encoding="utf-8")).get(
        "prepare_gate_pass"
    ):
        raise RuntimeError("Run prepare mode and pass its gate before controller mode")
    write_status(run_dir, state="controller_started", training_labels_locked=True)
    try:
        wait_for_resources(run_dir, "ddm")
        ddm_metrics = run_solver(run_dir, "ddm")
        write_status(run_dir, state="ddm_complete", ddm_metrics=ddm_metrics)
        time.sleep(30.0)
        wait_for_resources(run_dir, "direct")
        direct_metrics = run_solver(run_dir, "direct")
        write_status(run_dir, state="direct_complete", direct_metrics=direct_metrics)
        analyze(run_dir)
    except Exception as exc:
        write_status(
            run_dir,
            state="controller_failed",
            failure_type=type(exc).__name__,
            failure=str(exc),
            training_labels_locked=True,
        )
        raise


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.mode == "prepare":
        print(json.dumps(prepare(run_dir), indent=2))
    elif args.mode == "controller":
        controller(run_dir)
    elif args.mode == "analyze":
        print(json.dumps(analyze(run_dir), indent=2))
    else:
        print(json.dumps(read_status(run_dir), indent=2))


if __name__ == "__main__":
    main()
