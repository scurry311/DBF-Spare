#!/usr/bin/env python3
"""Build and solve a field-enabled fixed-mesh project for 256-port EEP export."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_fixed_mesh_ddm_run02 as fixed  # noqa: E402
import run_fixed_mesh_direct_crosscheck as direct  # noqa: E402
from run_staged_16x16_convergence import (  # noqa: E402
    NPORTS,
    PASSIVITY_SIGMA_LIMIT,
    RECIPROCITY_LIMIT,
    s_metrics,
    temporary_aedt_directory,
)


OUT_DIR = ROOT / "hfss_outputs" / "fixed_mesh_eep_fieldsolve_20260723_run01"
TARGET_NAME = "fixed_mesh_eep_fieldsolve_run01"
TARGET_PROJECT = OUT_DIR / "project" / f"{TARGET_NAME}.aedt"
SETUP_NAME = "Setup_Frozen_Direct_EEP_Run01"
TOUCHSTONE = OUT_DIR / "solve" / "fixed_mesh_eep_fieldsolve.s256p"
REFERENCE_TOUCHSTONE = (
    fixed.DEFAULT_RUN / "engineering_baseline" / "fixed_mesh_direct.s256p"
)
MIN_D_FREE_GB = 70.0
MIN_COMMIT_HEADROOM_GB = 26.5
MIN_PHYSICAL_AVAILABLE_GB = 15.0
AEDT_DISTRIBUTED_MACHINE_LIST: str | None = None
CRITICAL_PHYSICAL_GB = 0.5
CRITICAL_COMMIT_GB = 2.5
CRITICAL_D_FREE_GB = 10.0
MAX_CRITICAL_POLLS = 3
RESOURCE_POLL_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "smoke", "solve", "all", "status"), required=True
    )
    return parser.parse_args()


def status_path() -> Path:
    return OUT_DIR / "fieldsolve_status.json"


def read_status() -> dict[str, Any]:
    path = status_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_status(**values: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current = read_status()
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path().write_text(json.dumps(current, indent=2), encoding="utf-8")


def resources() -> dict[str, float]:
    return {
        **fixed.memory_metrics(),
        "d_free_gb": float(shutil.disk_usage(ROOT).free / 1024.0**3),
    }


def append_resource_sample(stage: str, values: dict[str, float], solver_pid: int | None = None) -> None:
    path = OUT_DIR / "resource_history.csv"
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "solver_pid": "" if solver_pid is None else solver_pid,
        **values,
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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


def setup_array() -> str:
    return f'''Array( _
    "NAME:{SETUP_NAME}", _
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
    "SaveAnyFields:=", True)'''


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
oAnalysis.DeleteSetups Array("Setup_Frozen_Direct_Run02")
oAnalysis.DeleteSetups Array("{SETUP_NAME}")
{delete_ops}
On Error GoTo 0
oAnalysis.InsertSetup "HfssDriven", {setup_array()}
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
oDesign.GenerateMesh Array("{SETUP_NAME}")
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
oDesign.Analyze "{SETUP_NAME}"
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
    values = oSolutions.ListVariations("{SETUP_NAME}:LastAdaptive")
    variation = CStr(values(LBound(values)))
    oSolutions.ExportNetworkData variation, Array("{SETUP_NAME}:LastAdaptive"), 3, _
        "{fixed.vp(TOUCHSTONE)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
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
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite fieldsolve output: {OUT_DIR}")
    required = (fixed.SOURCE_PROJECT, fixed.SOURCE_STATS, REFERENCE_TOUCHSTONE)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fieldsolve prerequisites: {missing}")
    TARGET_PROJECT.parent.mkdir(parents=True)
    shutil.copy2(fixed.SOURCE_PROJECT, TARGET_PROJECT)
    source_lock = fixed.clear_stale_source_lock(OUT_DIR)
    marker = OUT_DIR / "prepare_marker.txt"
    vbs = OUT_DIR / "prepare_fieldsolve.vbs"
    vbs.write_text(prepare_vbs(marker), encoding="ascii")
    return_code = run_aedt(
        vbs, OUT_DIR / "prepare.stdout.log", OUT_DIR / "prepare.stderr.log"
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
    fields_enabled = bool(
        SETUP_NAME in target_text and re.search(r"SaveAnyFields\s*=\s*true", target_text)
    )
    setup_bound = bool(SETUP_NAME in target_text and fixed.SOURCE_SETUP in target_text)
    passed = bool(
        return_code == 0
        and marker.exists()
        and source_ports == target_ports
        and source_ports[0] == NPORTS
        and target_mesh[0] == 0
        and fields_enabled
        and setup_bound
    )
    result = {
        "source_lock_preflight": source_lock,
        "target_project": str(TARGET_PROJECT),
        "port_count": source_ports[0],
        "port_definition_hash_unchanged": source_ports == target_ports,
        "target_local_mesh_operation_count": target_mesh[0],
        "mesh_link_binding_present": setup_bound,
        "save_any_fields_enabled": fields_enabled,
        "prepare_return_code": return_code,
        "prepare_gate_pass": passed,
    }
    (OUT_DIR / "prepare_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_status(state="prepared" if passed else "prepare_failed", prepare=result)
    if not passed:
        raise RuntimeError("Fieldsolve prepare gate failed")
    return result


def smoke() -> dict[str, Any]:
    prep = OUT_DIR / "prepare_summary.json"
    if not prep.exists() or not json.loads(prep.read_text(encoding="utf-8")).get(
        "prepare_gate_pass"
    ):
        raise RuntimeError("Prepare gate must pass before fieldsolve smoke")
    fixed.clear_stale_source_lock(OUT_DIR)
    smoke_dir = OUT_DIR / "import_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    messages = smoke_dir / "import_smoke_messages.txt"
    vbs = smoke_dir / "run_import_smoke.vbs"
    vbs.write_text(smoke_vbs(messages), encoding="ascii")
    write_status(state="running_import_smoke", resources=resources())
    return_code = run_aedt(
        vbs, smoke_dir / "import_smoke.stdout.log", smoke_dir / "import_smoke.stderr.log"
    )
    results_dir = TARGET_PROJECT.with_suffix(".aedtresults")
    profile = direct.latest_file(results_dir, "*.profile")
    stats = direct.latest_file(results_dir, "current.stats")
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
    if profile:
        shutil.copy2(profile, smoke_dir / profile.name)
    if stats:
        shutil.copy2(stats, smoke_dir / "current.stats")
    result = {
        "return_code": return_code,
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
        raise RuntimeError("Fieldsolve import smoke gate failed")
    return result


def wait_for_resources() -> dict[str, float]:
    requirements = {
        "physical_available_gb": MIN_PHYSICAL_AVAILABLE_GB,
        "commit_headroom_gb": MIN_COMMIT_HEADROOM_GB,
        "d_free_gb": MIN_D_FREE_GB,
    }
    deadline = time.time() + 8.0 * 3600.0
    while True:
        current = resources()
        append_resource_sample("waiting_resources", current)
        passed = all(current[key] >= value for key, value in requirements.items())
        write_status(
            state="resource_gate_passed" if passed else "waiting_resources",
            resources=current,
            start_requirements=requirements,
        )
        if passed:
            return current
        if time.time() >= deadline:
            raise TimeoutError("Fieldsolve resource gate did not pass within eight hours")
        time.sleep(60.0)


def solve() -> dict[str, Any]:
    smoke_summary = OUT_DIR / "import_smoke" / "import_smoke_summary.json"
    if not smoke_summary.exists() or not json.loads(
        smoke_summary.read_text(encoding="utf-8")
    ).get("import_smoke_gate_pass"):
        raise RuntimeError("Fieldsolve smoke gate must pass before solve")
    fixed.clear_stale_source_lock(OUT_DIR)
    start_resources = wait_for_resources()
    append_resource_sample("fieldsolve_start", start_resources)
    solve_dir = OUT_DIR / "solve"
    solve_dir.mkdir(parents=True, exist_ok=True)
    scratch = OUT_DIR / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    messages = solve_dir / "fieldsolve_messages.txt"
    vbs = solve_dir / "run_fieldsolve.vbs"
    vbs.write_text(solve_vbs(messages), encoding="ascii")
    stdout = solve_dir / "fieldsolve.stdout.log"
    stderr = solve_dir / "fieldsolve.stderr.log"
    results_dir = TARGET_PROJECT.with_suffix(".aedtresults")
    before = direct.profile_snapshot(results_dir)
    write_status(
        state="running_fieldsolve",
        start_resources=start_resources,
        scratch_directory=str(scratch),
    )
    with temporary_aedt_directory(scratch), stdout.open(
        "w", encoding="utf-8"
    ) as out, stderr.open("w", encoding="utf-8") as err:
        command = [str(fixed.ANSYS), "-ng"]
        if AEDT_DISTRIBUTED_MACHINE_LIST:
            command.extend(
                ["-Distributed", "-MachineList", AEDT_DISTRIBUTED_MACHINE_LIST]
            )
        command.extend(["-RunScriptAndExit", str(vbs)])
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=out,
            stderr=err,
        )
        critical_polls = 0
        while process.poll() is None:
            current = resources()
            append_resource_sample("running_fieldsolve", current, process.pid)
            critical = bool(
                current["physical_available_gb"] < CRITICAL_PHYSICAL_GB
                or current["commit_headroom_gb"] < CRITICAL_COMMIT_GB
                or current["d_free_gb"] < CRITICAL_D_FREE_GB
            )
            critical_polls = critical_polls + 1 if critical else 0
            write_status(
                state="running_fieldsolve",
                solver_pid=process.pid,
                resources=current,
                consecutive_critical_resource_polls=critical_polls,
            )
            if critical_polls >= MAX_CRITICAL_POLLS:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                process.wait(timeout=120)
                raise MemoryError(
                    "Fieldsolve stopped after "
                    f"{MAX_CRITICAL_POLLS} consecutive critical resource polls"
                )
            time.sleep(RESOURCE_POLL_SECONDS)
        return_code = int(process.returncode)
    profile = direct.changed_profile(results_dir, before)
    message_text = messages.read_text(encoding="utf-8", errors="ignore") if messages.exists() else ""
    solve_ok = bool(
        return_code == 0
        and "ANALYZE_ERR_NUMBER=0" in message_text
        and TOUCHSTONE.exists()
        and TOUCHSTONE.stat().st_size > 1_000_000
        and profile is not None
    )
    copied_profile = None
    if profile:
        copied_profile = solve_dir / profile.name
        if profile.resolve() != copied_profile.resolve():
            shutil.copy2(profile, copied_profile)
    result: dict[str, Any] = {
        "return_code": return_code,
        "touchstone": str(TOUCHSTONE),
        "profile": str(copied_profile or profile) if profile else None,
        "solve_and_export_gate_pass": solve_ok,
        "completion_resources": resources(),
    }
    append_resource_sample("fieldsolve_complete", result["completion_resources"])
    if solve_ok and profile is not None:
        result.update(direct.direct_profile_metrics(copied_profile or profile))
        result.update(s_metrics(TOUCHSTONE))
        comparison = direct.compare(REFERENCE_TOUCHSTONE, TOUCHSTONE)
        numerical_valid = bool(
            result["touchstone_complete"]
            and float(result["s_reciprocity_max_abs"]) <= RECIPROCITY_LIMIT
            and float(result["s_passivity_sigma_max"]) <= PASSIVITY_SIGMA_LIMIT
        )
        result.update(
            {
                "reference_direct_comparison": comparison,
                "numerical_smatrix_valid": numerical_valid,
                "delta_s_gate_pass": comparison["max_abs_delta_s"] <= fixed.MAX_DELTA_S,
                "matched_rl_gate_pass": float(result["matched_passive_rl_min_db"])
                >= fixed.MIN_MATCHED_RL_DB,
            }
        )
        result["fieldsolve_gate_pass"] = bool(
            numerical_valid and result["delta_s_gate_pass"] and result["matched_rl_gate_pass"]
        )
    else:
        result["fieldsolve_gate_pass"] = False
    (solve_dir / "fieldsolve_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_status(
        state="fieldsolve_complete_pass" if result["fieldsolve_gate_pass"] else "fieldsolve_complete_fail",
        solve=result,
    )
    if not solve_ok:
        raise RuntimeError("Fieldsolve or S256 export failed")
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.mode == "smoke":
        print(json.dumps(smoke(), indent=2))
    elif args.mode == "solve":
        print(json.dumps(solve(), indent=2))
    elif args.mode == "all":
        prepare()
        smoke()
        print(json.dumps(solve(), indent=2))
    else:
        print(json.dumps(read_status(), indent=2))


if __name__ == "__main__":
    main()
