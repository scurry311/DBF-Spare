#!/usr/bin/env python3
"""Run an independent fixed-mesh direct solve and cross-check run02 DDM."""

from __future__ import annotations

import argparse
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

import run_fixed_mesh_ddm_run02 as fixed  # noqa: E402
from run_staged_16x16_convergence import (  # noqa: E402
    NPORTS,
    PASSIVITY_SIGMA_LIMIT,
    RECIPROCITY_LIMIT,
    parse_touchstone,
    s_metrics,
    temporary_aedt_directory,
)


RUN02_ROOT = fixed.DEFAULT_RUN
DIRECT_ROOT = RUN02_ROOT / "direct_independent"
TARGET_NAME = "frozen_mesh_direct_run02"
DIRECT_SETUP = "Setup_Frozen_Direct_Run02"
TARGET_PROJECT = DIRECT_ROOT / "project" / f"{TARGET_NAME}.aedt"
DIRECT_TOUCHSTONE = DIRECT_ROOT / "solve" / "frozen_mesh_run02_direct.s256p"
DDM_TOUCHSTONE = RUN02_ROOT / "ddm" / "frozen_mesh_run02_ddm.s256p"
DDM_VALIDATION = RUN02_ROOT / "ddm" / "ddm_validation.json"
BASELINE_DIR = RUN02_ROOT / "engineering_baseline"

MIN_START_FREE_RAM_GB = 15.0
MIN_START_COMMIT_HEADROOM_GB = 28.0
MIN_START_D_FREE_GB = 70.0
CRITICAL_FREE_RAM_GB = 0.5
CRITICAL_COMMIT_HEADROOM_GB = 2.5
CRITICAL_D_FREE_GB = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("prepare", "smoke", "direct", "cross", "all", "status"),
        required=True,
    )
    return parser.parse_args()


def status_path() -> Path:
    return DIRECT_ROOT / "direct_crosscheck_status.json"


def read_status() -> dict[str, Any]:
    path = status_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_status(**values: Any) -> None:
    DIRECT_ROOT.mkdir(parents=True, exist_ok=True)
    current = read_status()
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path().write_text(json.dumps(current, indent=2), encoding="utf-8")


def d_free_gb() -> float:
    return float(shutil.disk_usage(ROOT).free / 1024.0**3)


def resource_metrics() -> dict[str, float]:
    return {**fixed.memory_metrics(), "d_free_gb": d_free_gb()}


def mesh_link() -> str:
    return f'''Array( _
        "NAME:MeshLink", _
        "ImportMesh:=", True, _
        "Project:=", "{fixed.vp(fixed.SOURCE_PROJECT)}", _
        "Product:=", "HFSS", _
        "Design:=", "{fixed.DESIGN}", _
        "Soln:=", "{fixed.SOURCE_SETUP}", _
        Array("NAME:Params"), _
        "ForceSourceToSolve:=", False, _
        "PreservePartnerSoln:=", True, _
        "PathRelativeTo:=", "TargetProject")'''


def direct_setup_array() -> str:
    return f'''Array( _
    "NAME:{DIRECT_SETUP}", _
    "SolveType:=", "Single", _
    "Frequency:=", "10GHz", _
    "MaxDeltaS:=", 0.05, _
    "UseMatrixConv:=", False, _
    "MaximumPasses:=", 1, _
    "MinimumPasses:=", 1, _
    "MinimumConvergedPasses:=", 1, _
    "PercentRefinement:=", 1, _
    "IsEnabled:=", True, _
    {mesh_link()}, _
    "BasisOrder:=", 1, _
    "DoLambdaRefine:=", False, _
    "DoMaterialLambda:=", False, _
    "SetLambdaTarget:=", False, _
    "UseMaxTetIncrease:=", False, _
    "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, _
    "SetPortMinMaxTri:=", False, _
    "DrivenSolverType:=", "Direct Solver", _
    "SaveRadFieldsOnly:=", False, _
    "SaveAnyFields:=", False)'''


def prepare_vbs(marker: Path) -> str:
    delete_ops = "\n".join(
        f'oMesh.DeleteOp Array("{name}")' for name in fixed.LOCAL_MESH_OPS
    )
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oAnalysis, oMesh, fso, outFile
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{fixed.vp(marker)}", True)
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{fixed.vp(TARGET_PROJECT)}"
Set oProject = oDesktop.SetActiveProject("{TARGET_NAME}")
Set oDesign = oProject.SetActiveDesign("{fixed.DESIGN}")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
Set oMesh = oDesign.GetModule("MeshSetup")
On Error Resume Next
oDesign.DeleteFullVariation "All", False
oAnalysis.DeleteSetups Array("Setup_10GHz")
oAnalysis.DeleteSetups Array("Setup_DDM_Recovery")
oAnalysis.DeleteSetups Array("Setup_VolFeed_DDM")
oAnalysis.DeleteSetups Array("Setup_Frozen_Direct")
oAnalysis.DeleteSetups Array("Setup_Frozen_DDM")
oAnalysis.DeleteSetups Array("Setup_Frozen_DDM_Run02")
oAnalysis.DeleteSetups Array("{DIRECT_SETUP}")
{delete_ops}
On Error GoTo 0
oAnalysis.InsertSetup "HfssDriven", {direct_setup_array()}
oProject.Save
outFile.WriteLine "PREPARED=1"
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def smoke_vbs(messages: Path) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, fso, outFile, msgs, i, meshErr, meshDescription
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{fixed.vp(messages)}", True)
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{fixed.vp(TARGET_PROJECT)}"
Set oProject = oDesktop.SetActiveProject("{TARGET_NAME}")
Set oDesign = oProject.SetActiveDesign("{fixed.DESIGN}")
On Error Resume Next
Err.Clear
oDesign.GenerateMesh Array("{DIRECT_SETUP}")
meshErr = Err.Number
meshDescription = Err.Description
On Error GoTo 0
outFile.WriteLine "GENERATE_MESH_ERR_NUMBER=" & CStr(meshErr)
outFile.WriteLine "GENERATE_MESH_ERR_DESCRIPTION=" & meshDescription
msgs = oDesktop.GetMessages(oProject.GetName(), oDesign.GetName(), 0)
If IsArray(msgs) Then
    For i = LBound(msgs) To UBound(msgs)
        outFile.WriteLine "AEDT_MESSAGE=" & CStr(msgs(i))
    Next
End If
oProject.Save
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def solve_vbs(messages: Path) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oSolutions, fso, outFile
Dim values, variation, msgs, i, analyzeErr, analyzeDescription
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{fixed.vp(messages)}", True)
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{fixed.vp(TARGET_PROJECT)}"
Set oProject = oDesktop.SetActiveProject("{TARGET_NAME}")
Set oDesign = oProject.SetActiveDesign("{fixed.DESIGN}")
On Error Resume Next
Err.Clear
oDesign.Analyze "{DIRECT_SETUP}"
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
    values = oSolutions.ListVariations("{DIRECT_SETUP}:LastAdaptive")
    variation = CStr(values(LBound(values)))
    oSolutions.ExportNetworkData variation, Array("{DIRECT_SETUP}:LastAdaptive"), 3, _
        "{fixed.vp(DIRECT_TOUCHSTONE)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
End If
oProject.Save
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def run_aedt(vbs: Path, stdout: Path, stderr: Path) -> int:
    with stdout.open("w", encoding="utf-8") as out, stderr.open(
        "w", encoding="utf-8"
    ) as err:
        result = subprocess.run(
            [str(fixed.ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=out,
            stderr=err,
            check=False,
        )
    return int(result.returncode)


def prepare() -> dict[str, Any]:
    if DIRECT_ROOT.exists() and any(DIRECT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite independent direct run: {DIRECT_ROOT}")
    required = (
        fixed.ANSYS,
        fixed.SOURCE_PROJECT,
        fixed.SOURCE_TOUCHSTONE,
        fixed.SOURCE_STATS,
        DDM_TOUCHSTONE,
        DDM_VALIDATION,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing direct cross-check prerequisites: {missing}")
    TARGET_PROJECT.parent.mkdir(parents=True)
    shutil.copy2(fixed.SOURCE_PROJECT, TARGET_PROJECT)
    source_lock = fixed.clear_stale_source_lock(DIRECT_ROOT)
    marker = DIRECT_ROOT / "prepare_marker.txt"
    vbs = DIRECT_ROOT / "prepare_direct.vbs"
    vbs.write_text(prepare_vbs(marker), encoding="ascii")
    return_code = run_aedt(
        vbs, DIRECT_ROOT / "prepare.stdout.log", DIRECT_ROOT / "prepare.stderr.log"
    )

    source_text = fixed.SOURCE_PROJECT.read_text(encoding="utf-8", errors="ignore")
    target_text = TARGET_PROJECT.read_text(encoding="utf-8", errors="ignore")
    port_pattern = r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'"
    mesh_pattern = (
        r"\$begin '(?:PortFeedUniform_0p180mm|FeedSheetUniform_0p180mm)'"
        r".*?\$end '(?:PortFeedUniform_0p180mm|FeedSheetUniform_0p180mm)'"
    )
    source_ports = fixed.block_digest(source_text, port_pattern)
    target_ports = fixed.block_digest(target_text, port_pattern)
    target_mesh = fixed.block_digest(target_text, mesh_pattern)
    source_path_token = str(fixed.SOURCE_PROJECT.resolve()).replace("\\", "\\\\")
    setup_bound = bool(
        DIRECT_SETUP in target_text
        and fixed.SOURCE_SETUP in target_text
        and (
            str(fixed.SOURCE_PROJECT.resolve()) in target_text
            or source_path_token in target_text
        )
    )
    passed = bool(
        return_code == 0
        and marker.exists()
        and source_ports == target_ports
        and source_ports[0] == NPORTS
        and target_mesh[0] == 0
        and setup_bound
    )
    result = {
        "scope": "independent direct fixed-mesh cross-check",
        "source_lock_preflight": source_lock,
        "target_project": str(TARGET_PROJECT),
        "source_project_sha256": fixed.sha256(fixed.SOURCE_PROJECT),
        "target_project_sha256": fixed.sha256(TARGET_PROJECT),
        "port_count": source_ports[0],
        "port_definition_hash_unchanged": source_ports == target_ports,
        "target_local_mesh_operation_count": target_mesh[0],
        "mesh_link_binding_present": setup_bound,
        "direct_setup": DIRECT_SETUP,
        "maximum_passes": 1,
        "prepare_return_code": return_code,
        "prepare_gate_pass": passed,
    }
    (DIRECT_ROOT / "prepare_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_status(state="prepared" if passed else "prepare_failed", prepare=result)
    if not passed:
        raise RuntimeError("Independent direct prepare gate failed")
    return result


def latest_file(root: Path, pattern: str) -> Path | None:
    candidates = list(root.rglob(pattern)) if root.exists() else []
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def smoke() -> dict[str, Any]:
    prep = DIRECT_ROOT / "prepare_summary.json"
    if not prep.exists() or not json.loads(prep.read_text(encoding="utf-8")).get(
        "prepare_gate_pass"
    ):
        raise RuntimeError("Prepare gate must pass before direct import smoke")
    source_lock = fixed.clear_stale_source_lock(DIRECT_ROOT)
    smoke_dir = DIRECT_ROOT / "import_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    messages = smoke_dir / "import_smoke_messages.txt"
    vbs = smoke_dir / "run_import_smoke.vbs"
    vbs.write_text(smoke_vbs(messages), encoding="ascii")
    write_status(state="running_import_smoke", memory=resource_metrics())
    return_code = run_aedt(
        vbs, smoke_dir / "import_smoke.stdout.log", smoke_dir / "import_smoke.stderr.log"
    )
    results_dir = TARGET_PROJECT.with_suffix(".aedtresults")
    profile = latest_file(results_dir, "*.profile")
    stats = latest_file(results_dir, "current.stats")
    message_text = messages.read_text(encoding="utf-8", errors="ignore") if messages.exists() else ""
    counts = fixed.mesh_profile_counts(profile) if profile else {}
    source_rows, source_sum = fixed.stats_tet_sum(fixed.SOURCE_STATS)
    target_rows, target_sum = fixed.stats_tet_sum(stats) if stats else (0, 0)
    relative_delta = abs(target_sum - source_sum) / source_sum if source_sum else None
    passed = bool(
        return_code == 0
        and "GENERATE_MESH_ERR_NUMBER=0" in message_text
        and profile is not None
        and stats is not None
        and target_rows == NPORTS + 4
        and relative_delta is not None
        and relative_delta <= fixed.IMPORT_TET_REL_TOL
        and counts.get("linked_mesh_profile_present") is True
        and counts.get("manual_refine_present") is False
    )
    copied_profile = None
    if profile:
        copied_profile = smoke_dir / profile.name
        shutil.copy2(profile, copied_profile)
    if stats:
        shutil.copy2(stats, smoke_dir / "current.stats")
    result = {
        "return_code": return_code,
        "source_lock_preflight": source_lock,
        "messages": str(messages),
        "profile": str(copied_profile or profile) if profile else None,
        **counts,
        "source_stats_rows": source_rows,
        "source_stats_tetrahedra_sum": source_sum,
        "target_stats_rows": target_rows,
        "target_stats_tetrahedra_sum": target_sum,
        "stats_relative_delta": relative_delta,
        "import_smoke_gate_pass": passed,
    }
    (smoke_dir / "import_smoke_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_status(
        state="import_smoke_passed" if passed else "import_smoke_failed",
        import_smoke=result,
    )
    if not passed:
        raise RuntimeError("Independent direct import smoke gate failed")
    return result


def wait_for_resources() -> dict[str, float]:
    requirements = {
        "physical_available_gb": MIN_START_FREE_RAM_GB,
        "commit_headroom_gb": MIN_START_COMMIT_HEADROOM_GB,
        "d_free_gb": MIN_START_D_FREE_GB,
    }
    deadline = time.time() + 8.0 * 3600.0
    while True:
        current = resource_metrics()
        passed = all(current[key] >= value for key, value in requirements.items())
        write_status(
            state="direct_resource_gate_passed" if passed else "waiting_direct_resources",
            resource_metrics=current,
            start_requirements=requirements,
        )
        if passed:
            return current
        if time.time() >= deadline:
            raise TimeoutError("Direct resource gate did not pass within eight hours")
        time.sleep(60.0)


def profile_snapshot(results_dir: Path) -> dict[Path, int]:
    if not results_dir.exists():
        return {}
    return {path.resolve(): path.stat().st_mtime_ns for path in results_dir.rglob("*.profile")}


def changed_profile(results_dir: Path, before: dict[Path, int]) -> Path | None:
    candidates = []
    for path in results_dir.rglob("*.profile") if results_dir.exists() else []:
        resolved = path.resolve()
        if resolved not in before or path.stat().st_mtime_ns > before[resolved]:
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else latest_file(results_dir, "*.profile")


def direct_profile_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matrix_lines = re.findall(
        r"ProfileItem\('Matrix Solve'.*?Type\\', \\'([^']+).*?"
        r"Matrix size\\',\s*(\d+).*?Disk\\', \\'([^']+)",
        text,
    )
    memory_kb = [
        int(value)
        for value in re.findall(
            r"ProfileItem\('Matrix Solve',\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*(\d+)",
            text,
        )
    ]
    return {
        "normal_completion": "Normal Completion" in text,
        "manual_refine_present": "ProfileItem('Manual Refine'" in text,
        "matrix_solve_count": len(matrix_lines),
        "matrix_solver_type": matrix_lines[-1][0].rstrip("\\") if matrix_lines else None,
        "matrix_size": int(matrix_lines[-1][1]) if matrix_lines else None,
        "matrix_disk": matrix_lines[-1][2].rstrip("\\") if matrix_lines else None,
        "matrix_memory_gb": float(memory_kb[-1] / 1024.0**2) if memory_kb else None,
    }


def compare(first: Path, second: Path) -> dict[str, Any]:
    first_s = parse_touchstone(first, NPORTS)
    second_s = parse_touchstone(second, NPORTS)
    delta = np.abs(second_s - first_s)
    diagonal = np.diag(delta)
    off_diagonal = delta[~np.eye(NPORTS, dtype=bool)]
    max_index = np.unravel_index(np.argmax(delta), delta.shape)
    return {
        "first": str(first),
        "second": str(second),
        "max_abs_delta_s": float(np.max(delta)),
        "max_abs_delta_s_ports_1based": [int(value) + 1 for value in max_index],
        "rms_abs_delta_s": float(np.sqrt(np.mean(delta**2))),
        "max_diagonal_delta_s": float(np.max(diagonal)),
        "max_offdiagonal_delta_s": float(np.max(off_diagonal)),
    }


def run_direct() -> dict[str, Any]:
    smoke_summary = DIRECT_ROOT / "import_smoke" / "import_smoke_summary.json"
    if not smoke_summary.exists() or not json.loads(
        smoke_summary.read_text(encoding="utf-8")
    ).get("import_smoke_gate_pass"):
        raise RuntimeError("Direct import smoke gate must pass before solve")
    source_lock = fixed.clear_stale_source_lock(DIRECT_ROOT)
    start_resources = wait_for_resources()
    solve_dir = DIRECT_ROOT / "solve"
    solve_dir.mkdir(parents=True, exist_ok=True)
    scratch = DIRECT_ROOT / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    messages = solve_dir / "direct_messages.txt"
    vbs = solve_dir / "run_direct.vbs"
    vbs.write_text(solve_vbs(messages), encoding="ascii")
    stdout = solve_dir / "direct.stdout.log"
    stderr = solve_dir / "direct.stderr.log"
    results_dir = TARGET_PROJECT.with_suffix(".aedtresults")
    before = profile_snapshot(results_dir)
    write_status(
        state="running_direct",
        direct_start_resources=start_resources,
        scratch_directory=str(scratch),
    )
    with temporary_aedt_directory(scratch), stdout.open(
        "w", encoding="utf-8"
    ) as out, stderr.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(
            [str(fixed.ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=out,
            stderr=err,
        )
        critical_polls = 0
        while process.poll() is None:
            current = resource_metrics()
            critical = bool(
                current["physical_available_gb"] < CRITICAL_FREE_RAM_GB
                or current["commit_headroom_gb"] < CRITICAL_COMMIT_HEADROOM_GB
                or current["d_free_gb"] < CRITICAL_D_FREE_GB
            )
            critical_polls = critical_polls + 1 if critical else 0
            write_status(
                state="running_direct",
                direct_pid=process.pid,
                resource_metrics=current,
                consecutive_critical_resource_polls=critical_polls,
            )
            if critical_polls >= 3:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                process.wait(timeout=120)
                raise MemoryError("Direct solve stopped after three critical resource polls")
            time.sleep(30.0)
        return_code = int(process.returncode)

    profile = changed_profile(results_dir, before)
    message_text = messages.read_text(encoding="utf-8", errors="ignore") if messages.exists() else ""
    solve_ok = bool(
        return_code == 0
        and "ANALYZE_ERR_NUMBER=0" in message_text
        and DIRECT_TOUCHSTONE.exists()
        and DIRECT_TOUCHSTONE.stat().st_size > 1_000_000
        and profile is not None
    )
    copied_profile = None
    if profile:
        copied_profile = solve_dir / profile.name
        if profile.resolve() != copied_profile.resolve():
            shutil.copy2(profile, copied_profile)
    result: dict[str, Any] = {
        "return_code": return_code,
        "source_lock_preflight": source_lock,
        "touchstone": str(DIRECT_TOUCHSTONE),
        "profile": str(copied_profile or profile) if profile else None,
        "solve_and_export_gate_pass": solve_ok,
        "completion_resources": resource_metrics(),
    }
    if solve_ok and profile is not None:
        result.update(direct_profile_metrics(copied_profile or profile))
        result.update(s_metrics(DIRECT_TOUCHSTONE))
        result["numerical_smatrix_valid"] = bool(
            result["touchstone_complete"]
            and float(result["s_reciprocity_max_abs"]) <= RECIPROCITY_LIMIT
            and float(result["s_passivity_sigma_max"]) <= PASSIVITY_SIGMA_LIMIT
        )
    (solve_dir / "direct_solver_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_status(
        state="direct_complete" if solve_ok else "direct_failed",
        direct=result,
    )
    if not solve_ok:
        raise RuntimeError("Independent direct solve or S256 export failed")
    return result


def cross_validate() -> dict[str, Any]:
    direct_metrics_path = DIRECT_ROOT / "solve" / "direct_solver_metrics.json"
    required = (direct_metrics_path, DIRECT_TOUCHSTONE, DDM_TOUCHSTONE, DDM_VALIDATION)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cross-validation inputs: {missing}")
    direct_metrics = json.loads(direct_metrics_path.read_text(encoding="utf-8"))
    for key in ("matrix_solver_type", "matrix_disk"):
        if isinstance(direct_metrics.get(key), str):
            direct_metrics[key] = direct_metrics[key].rstrip("\\")
    direct_metrics_path.write_text(
        json.dumps(direct_metrics, indent=2), encoding="utf-8"
    )
    ddm_metrics = json.loads(DDM_VALIDATION.read_text(encoding="utf-8"))
    direct_s = s_metrics(DIRECT_TOUCHSTONE)
    ddm_s = s_metrics(DDM_TOUCHSTONE)
    comparisons = {
        "direct_vs_ddm": compare(DIRECT_TOUCHSTONE, DDM_TOUCHSTONE),
        "source_pass05_vs_direct": compare(fixed.SOURCE_TOUCHSTONE, DIRECT_TOUCHSTONE),
        "source_pass05_vs_ddm": compare(fixed.SOURCE_TOUCHSTONE, DDM_TOUCHSTONE),
    }
    numerical_valid = bool(
        direct_metrics.get("numerical_smatrix_valid")
        and ddm_metrics.get("numerical_smatrix_valid")
    )
    cross_gate = comparisons["direct_vs_ddm"]["max_abs_delta_s"] <= fixed.MAX_DELTA_S
    preferred_gate = (
        comparisons["direct_vs_ddm"]["max_abs_delta_s"] <= fixed.PREFERRED_DELTA_S
    )
    source_gate = bool(
        comparisons["source_pass05_vs_direct"]["max_abs_delta_s"] <= fixed.MAX_DELTA_S
        and comparisons["source_pass05_vs_ddm"]["max_abs_delta_s"] <= fixed.MAX_DELTA_S
    )
    rl_gate = bool(
        float(direct_s["matched_passive_rl_min_db"]) >= fixed.MIN_MATCHED_RL_DB
        and float(ddm_s["matched_passive_rl_min_db"]) >= fixed.MIN_MATCHED_RL_DB
    )
    mesh_gate = bool(
        json.loads(
            (DIRECT_ROOT / "import_smoke" / "import_smoke_summary.json").read_text(
                encoding="utf-8"
            )
        ).get("import_smoke_gate_pass")
        and json.loads(
            (RUN02_ROOT / "import_smoke" / "import_smoke_summary.json").read_text(
                encoding="utf-8"
            )
        ).get("import_smoke_gate_pass")
    )
    accepted = bool(numerical_valid and cross_gate and source_gate and rl_gate and mesh_gate)
    result = {
        "baseline_version": "fixed_mesh_old_pass05_run02_20260723_v1",
        "accepted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "old pass5 fixed-mesh independent direct/DDM engineering baseline",
        "comparisons": comparisons,
        "direct_metrics": direct_metrics,
        "ddm_metrics": ddm_metrics,
        "gates": {
            "both_import_smokes_pass": mesh_gate,
            "both_numerical_smatrices_valid": numerical_valid,
            "direct_vs_ddm_max_delta_s_le_0p05": cross_gate,
            "direct_vs_ddm_preferred_delta_s_le_0p02": preferred_gate,
            "both_reproduce_source_pass05_le_0p05": source_gate,
            "both_matched_min_rl_ge_10db": rl_gate,
        },
        "engineering_baseline_accepted": accepted,
        "decision": (
            "accept_old_pass05_fixed_mesh_engineering_baseline"
            if accepted
            else "reject_old_pass05_fixed_mesh_engineering_baseline"
        ),
        "new_baseline_label_generation_allowed": accepted,
        "legacy_labels_unchanged": True,
    }
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    direct_copy = BASELINE_DIR / "fixed_mesh_direct.s256p"
    ddm_copy = BASELINE_DIR / "fixed_mesh_ddm.s256p"
    direct_profile = Path(str(direct_metrics["profile"]))
    ddm_profile = Path(str(ddm_metrics["profile"]))
    direct_profile_copy = BASELINE_DIR / "fixed_mesh_direct.profile"
    ddm_profile_copy = BASELINE_DIR / "fixed_mesh_ddm.profile"
    shutil.copy2(DIRECT_TOUCHSTONE, direct_copy)
    shutil.copy2(DDM_TOUCHSTONE, ddm_copy)
    shutil.copy2(direct_profile, direct_profile_copy)
    shutil.copy2(ddm_profile, ddm_profile_copy)
    result["baseline_artifacts"] = {
        "direct_s256": str(direct_copy),
        "direct_s256_sha256": fixed.sha256(direct_copy),
        "ddm_s256": str(ddm_copy),
        "ddm_s256_sha256": fixed.sha256(ddm_copy),
        "direct_profile": str(direct_profile_copy),
        "direct_profile_sha256": fixed.sha256(direct_profile_copy),
        "ddm_profile": str(ddm_profile_copy),
        "ddm_profile_sha256": fixed.sha256(ddm_profile_copy),
        "source_project": str(fixed.SOURCE_PROJECT),
        "source_project_sha256": fixed.sha256(fixed.SOURCE_PROJECT),
        "direct_project": str(TARGET_PROJECT),
        "direct_project_sha256": fixed.sha256(TARGET_PROJECT),
    }
    manifest = BASELINE_DIR / "engineering_baseline_manifest.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cross = comparisons["direct_vs_ddm"]
    lines = [
        "# Fixed-mesh engineering baseline",
        "",
        f"- Decision: {result['decision']}",
        f"- Direct vs DDM max |Delta S|: {cross['max_abs_delta_s']:.6f}",
        f"- Direct vs DDM RMS |Delta S|: {cross['rms_abs_delta_s']:.6f}",
        f"- Direct matched minimum RL: {direct_s['matched_passive_rl_min_db']:.3f} dB",
        f"- DDM matched minimum RL: {ddm_s['matched_passive_rl_min_db']:.3f} dB",
        f"- New baseline label generation allowed: {accepted}",
        "- Legacy labels remain unchanged.",
    ]
    (BASELINE_DIR / "ENGINEERING_BASELINE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_status(
        state="baseline_accepted" if accepted else "baseline_rejected",
        cross_validation=result,
    )
    fixed.write_status(
        RUN02_ROOT,
        state=("engineering_baseline_accepted" if accepted else "engineering_baseline_rejected"),
        engineering_baseline=result,
        training_labels_locked=not accepted,
        new_baseline_label_generation_allowed=accepted,
        legacy_labels_unchanged=True,
    )
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.mode == "smoke":
        print(json.dumps(smoke(), indent=2))
    elif args.mode == "direct":
        print(json.dumps(run_direct(), indent=2))
    elif args.mode == "cross":
        print(json.dumps(cross_validate(), indent=2))
    elif args.mode == "all":
        prepare()
        smoke()
        run_direct()
        print(json.dumps(cross_validate(), indent=2))
    else:
        print(json.dumps(read_status(), indent=2))


if __name__ == "__main__":
    main()
