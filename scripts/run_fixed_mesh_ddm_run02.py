#!/usr/bin/env python3
"""Run an import-only mesh smoke and one fixed-mesh DDM validation."""

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
SOURCE_LOCK = SOURCE_PROJECT.with_suffix(SOURCE_PROJECT.suffix + ".lock")
SOURCE_TOUCHSTONE = (
    SOURCE_RUN / "stages" / "pass05" / "grounded_patch_16x16_ddm_pass05.s256p"
)
SOURCE_STATS = (
    SOURCE_RUN
    / "grounded_patch_16x16"
    / "grounded_patch_16x16.aedtresults"
    / "URA_GroundedPatch_10GHz.results"
    / "DV1332_S2193_MI0_V2409.sd"
    / "current.stats"
)
DEFAULT_RUN = ROOT / "hfss_outputs" / "fixed_mesh_cross_solver_20260723_run02"
TARGET_NAME = "frozen_mesh_ddm_run02"
DESIGN = "URA_GroundedPatch_10GHz"
SOURCE_SETUP = "Setup_DDM_Recovery : LastAdaptive"
DDM_SETUP = "Setup_Frozen_DDM_Run02"
LOCAL_MESH_OPS = ("FeedSheetUniform_0p180mm", "PortFeedUniform_0p180mm")

SOURCE_DDM_TETS = 936_505
IMPORT_TET_REL_TOL = 0.01
FINAL_STATS_REL_REPORT_TOL = 0.02
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
        "--mode", choices=("prepare", "smoke", "ddm", "all", "status"), required=True
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


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


def process_exists(pid: int) -> bool:
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information, False, pid
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def clear_stale_source_lock(run_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_lock": str(SOURCE_LOCK),
        "lock_found": SOURCE_LOCK.exists(),
        "lock_removed": False,
    }
    if not SOURCE_LOCK.exists():
        return result
    lock_text = SOURCE_LOCK.read_text(encoding="utf-8", errors="ignore")
    pid_match = re.search(r"DesktopProcessID=(\d+)", lock_text)
    owner_pid = int(pid_match.group(1)) if pid_match else None
    owner_alive = bool(owner_pid is not None and process_exists(owner_pid))
    result.update(
        {
            "lock_sha256": sha256(SOURCE_LOCK),
            "owner_pid": owner_pid,
            "owner_process_alive": owner_alive,
        }
    )
    if owner_alive:
        raise RuntimeError(f"Source MeshLink project is actively locked by PID {owner_pid}")
    archive = run_dir / "stale_source_lock_archive" / SOURCE_LOCK.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_LOCK, archive)
    SOURCE_LOCK.unlink()
    result.update({"lock_removed": True, "lock_archive": str(archive)})
    return result


def status_path(run_dir: Path) -> Path:
    return run_dir / "run02_status.json"


def read_status(run_dir: Path) -> dict[str, Any]:
    path = status_path(run_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_status(run_dir: Path, **values: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    current = read_status(run_dir)
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path(run_dir).write_text(json.dumps(current, indent=2), encoding="utf-8")


def target_project(run_dir: Path) -> Path:
    return run_dir / "project" / f"{TARGET_NAME}.aedt"


def mesh_link() -> str:
    return f'''Array( _
        "NAME:MeshLink", _
        "ImportMesh:=", True, _
        "Project:=", "{vp(SOURCE_PROJECT)}", _
        "Product:=", "HFSS", _
        "Design:=", "{DESIGN}", _
        "Soln:=", "{SOURCE_SETUP}", _
        Array("NAME:Params"), _
        "ForceSourceToSolve:=", False, _
        "PreservePartnerSoln:=", True, _
        "PathRelativeTo:=", "TargetProject")'''


def setup_array() -> str:
    return f'''Array( _
    "NAME:{DDM_SETUP}", _
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
    "DrivenSolverType:=", "Domain Decomposition", _
    "IterativeResidual:=", 0.000001, _
    "DDMSolverResidual:=", 0.000001, _
    "SaveRadFieldsOnly:=", False, _
    "SaveAnyFields:=", False)'''


def prepare_vbs(target: Path, marker: Path) -> str:
    delete_ops = "\n".join(f'oMesh.DeleteOp Array("{name}")' for name in LOCAL_MESH_OPS)
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oAnalysis, oMesh, fso, outFile
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{vp(marker)}", True)
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(target)}"
Set oProject = oDesktop.SetActiveProject("{TARGET_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN}")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
Set oMesh = oDesign.GetModule("MeshSetup")
On Error Resume Next
oDesign.DeleteFullVariation "All", False
oAnalysis.DeleteSetups Array("Setup_10GHz")
oAnalysis.DeleteSetups Array("Setup_DDM_Recovery")
oAnalysis.DeleteSetups Array("Setup_VolFeed_DDM")
oAnalysis.DeleteSetups Array("Setup_Frozen_Direct")
oAnalysis.DeleteSetups Array("Setup_Frozen_DDM")
oAnalysis.DeleteSetups Array("{DDM_SETUP}")
{delete_ops}
On Error GoTo 0
oAnalysis.InsertSetup "HfssDriven", {setup_array()}
oProject.Save
outFile.WriteLine "PREPARED=1"
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def smoke_vbs(target: Path, messages: Path) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, fso, outFile, msgs, i, meshErr, meshDescription
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{vp(messages)}", True)
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(target)}"
Set oProject = oDesktop.SetActiveProject("{TARGET_NAME}")
Set oDesign = oProject.SetActiveDesign("{DESIGN}")
On Error Resume Next
Err.Clear
oDesign.GenerateMesh Array("{DDM_SETUP}")
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


def solve_vbs(target: Path, touchstone: Path, messages: Path) -> str:
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
oDesign.Analyze "{DDM_SETUP}"
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
    values = oSolutions.ListVariations("{DDM_SETUP}:LastAdaptive")
    variation = CStr(values(LBound(values)))
    oSolutions.ExportNetworkData variation, Array("{DDM_SETUP}:LastAdaptive"), 3, _
        "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
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
            [str(ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=out,
            stderr=err,
            check=False,
        )
    return int(result.returncode)


def prepare(run_dir: Path) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    required = (ANSYS, SOURCE_PROJECT, SOURCE_TOUCHSTONE, SOURCE_STATS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run02 prerequisites: {missing}")

    project_dir = run_dir / "project"
    project_dir.mkdir(parents=True)
    target = target_project(run_dir)
    shutil.copy2(SOURCE_PROJECT, target)
    marker = run_dir / "prepare_marker.txt"
    vbs = run_dir / "prepare_run02.vbs"
    vbs.write_text(prepare_vbs(target, marker), encoding="ascii")
    return_code = run_aedt(
        vbs, run_dir / "prepare.stdout.log", run_dir / "prepare.stderr.log"
    )

    source_text = SOURCE_PROJECT.read_text(encoding="utf-8", errors="ignore")
    target_text = target.read_text(encoding="utf-8", errors="ignore")
    port_pattern = r"\$begin 'P\d{3}'.*?\$end 'P\d{3}'"
    local_mesh_pattern = (
        r"\$begin '(?:PortFeedUniform_0p180mm|FeedSheetUniform_0p180mm)'"
        r".*?\$end '(?:PortFeedUniform_0p180mm|FeedSheetUniform_0p180mm)'"
    )
    source_ports = block_digest(source_text, port_pattern)
    target_ports = block_digest(target_text, port_pattern)
    source_local_mesh = block_digest(source_text, local_mesh_pattern)
    target_local_mesh = block_digest(target_text, local_mesh_pattern)
    source_path_token = str(SOURCE_PROJECT.resolve()).replace("\\", "\\\\")
    setup_bound = bool(
        DDM_SETUP in target_text
        and SOURCE_SETUP in target_text
        and (
            str(SOURCE_PROJECT.resolve()) in target_text
            or source_path_token in target_text
        )
    )
    passed = bool(
        return_code == 0
        and marker.exists()
        and source_ports == target_ports
        and source_ports[0] == NPORTS
        and source_local_mesh[0] == 2
        and target_local_mesh[0] == 0
        and setup_bound
    )
    result = {
        "scope": "independent run02 old-pass5 fixed-mesh DDM validation",
        "source_project": str(SOURCE_PROJECT),
        "source_touchstone": str(SOURCE_TOUCHSTONE),
        "source_current_stats": str(SOURCE_STATS),
        "target_project": str(target),
        "source_project_sha256": sha256(SOURCE_PROJECT),
        "source_touchstone_sha256": sha256(SOURCE_TOUCHSTONE),
        "source_current_stats_sha256": sha256(SOURCE_STATS),
        "prepare_return_code": return_code,
        "port_count": source_ports[0],
        "port_definition_hash_unchanged": source_ports == target_ports,
        "source_local_mesh_operation_count": source_local_mesh[0],
        "target_local_mesh_operation_count": target_local_mesh[0],
        "target_local_mesh_operations_removed": target_local_mesh[0] == 0,
        "mesh_link_binding_present": setup_bound,
        "source_mesh_solution": SOURCE_SETUP,
        "ddm_setup": DDM_SETUP,
        "maximum_passes": 1,
        "adaptive_volume_refinement_enabled": False,
        "prepare_gate_pass": passed,
        "training_labels_locked": True,
    }
    (run_dir / "prepare_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_status(
        run_dir,
        state="prepared" if passed else "prepare_failed",
        prepare=result,
        training_labels_locked=True,
    )
    if not passed:
        raise RuntimeError("run02 preparation gate failed")
    return result


def latest_file(root: Path, pattern: str) -> Path | None:
    candidates = list(root.rglob(pattern)) if root.exists() else []
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def stats_tet_sum(path: Path) -> tuple[int, int]:
    rows = 0
    total = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"\s*\d+\s*\|\s*(\d+)\s*\|", line)
        if match:
            rows += 1
            total += int(match.group(1))
    return rows, total


def mesh_profile_counts(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    result: dict[str, Any] = {
        "manual_refine_present": "ProfileItem('Manual Refine'" in text,
        "linked_mesh_profile_present": bool(
            "\\'Type\\', \\'Link\\'" in text
            and "Setup_DDM_Recovery : LastAdaptive" in text
        ),
        "port_adapt_tetrahedra": [],
        "port_refine_tetrahedra": [],
        "domain_partition_tetrahedra": [],
    }
    patterns = {
        "port_adapt_tetrahedra": r"ProfileItem\('Port Adapt'.*?Tetrahedra.*?,\s*(\d+)",
        "port_refine_tetrahedra": r"ProfileItem\('Port Refine'.*?Tetrahedra.*?,\s*(\d+)",
        "domain_partition_tetrahedra": r"ProfileItem\('Domain Partitioning'.*?Tetrahedra.*?,\s*(\d+)",
    }
    for key, pattern in patterns.items():
        result[key] = [int(value) for value in re.findall(pattern, text)]
    return result


def smoke(run_dir: Path) -> dict[str, Any]:
    prepare_summary = run_dir / "prepare_summary.json"
    if not prepare_summary.exists() or not json.loads(
        prepare_summary.read_text(encoding="utf-8")
    ).get("prepare_gate_pass"):
        raise RuntimeError("run02 prepare gate must pass before smoke")

    source_lock = clear_stale_source_lock(run_dir)
    target = target_project(run_dir)
    smoke_dir = run_dir / "import_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    messages = smoke_dir / "import_smoke_messages.txt"
    vbs = smoke_dir / "run_import_smoke.vbs"
    vbs.write_text(smoke_vbs(target, messages), encoding="ascii")
    write_status(run_dir, state="running_import_smoke", memory=memory_metrics())
    return_code = run_aedt(
        vbs, smoke_dir / "import_smoke.stdout.log", smoke_dir / "import_smoke.stderr.log"
    )

    results_dir = target.with_suffix(".aedtresults")
    profile = latest_file(results_dir, "*.profile")
    stats = latest_file(results_dir, "current.stats")
    message_text = messages.read_text(encoding="utf-8", errors="ignore") if messages.exists() else ""
    counts = mesh_profile_counts(profile) if profile else {}
    source_rows, source_sum = stats_tet_sum(SOURCE_STATS)
    target_rows, target_sum = stats_tet_sum(stats) if stats else (0, 0)
    port_adapt_values = counts.get("port_adapt_tetrahedra", [])
    import_tets = port_adapt_values[-1] if port_adapt_values else None
    import_relative_delta = (
        abs(import_tets - SOURCE_DDM_TETS) / SOURCE_DDM_TETS
        if import_tets is not None
        else None
    )
    final_stats_relative_delta = (
        abs(target_sum - source_sum) / source_sum if target_sum and source_sum else None
    )
    stats_count_gate = bool(
        final_stats_relative_delta is not None
        and final_stats_relative_delta <= IMPORT_TET_REL_TOL
    )
    port_adapt_count_gate = bool(
        import_relative_delta is not None and import_relative_delta <= IMPORT_TET_REL_TOL
    )
    no_manual_refine_gate = bool(
        profile is not None and not counts.get("manual_refine_present", True)
    )
    passed = bool(
        return_code == 0
        and "GENERATE_MESH_ERR_NUMBER=0" in message_text
        and profile is not None
        and stats is not None
        and target_rows == NPORTS + 4
        and counts.get("linked_mesh_profile_present") is True
        and stats_count_gate
        and no_manual_refine_gate
    )
    copied_profile = None
    copied_stats = None
    if profile:
        copied_profile = smoke_dir / profile.name
        shutil.copy2(profile, copied_profile)
    if stats:
        copied_stats = smoke_dir / "current.stats"
        shutil.copy2(stats, copied_stats)
    result = {
        "return_code": return_code,
        "source_lock_preflight": source_lock,
        "messages": str(messages),
        "profile": str(copied_profile or profile) if profile else None,
        "current_stats": str(copied_stats or stats) if stats else None,
        **counts,
        "source_domain_partition_tetrahedra": SOURCE_DDM_TETS,
        "import_port_adapt_tetrahedra": import_tets,
        "import_tetrahedra_relative_delta": import_relative_delta,
        "port_adapt_count_within_1pct_if_reported": (
            port_adapt_count_gate if import_tets is not None else None
        ),
        "source_stats_rows": source_rows,
        "source_stats_tetrahedra_sum": source_sum,
        "target_stats_rows": target_rows,
        "target_stats_tetrahedra_sum": target_sum,
        "final_stats_relative_delta": final_stats_relative_delta,
        "linked_stats_tetrahedra_within_1pct": stats_count_gate,
        "final_stats_within_2pct_diagnostic": bool(
            final_stats_relative_delta is not None
            and final_stats_relative_delta <= FINAL_STATS_REL_REPORT_TOL
        ),
        "no_target_manual_refine": no_manual_refine_gate,
        "import_smoke_gate_pass": passed,
        "training_labels_locked": True,
    }
    (smoke_dir / "import_smoke_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_status(
        run_dir,
        state="import_smoke_passed" if passed else "import_smoke_failed",
        import_smoke=result,
        training_labels_locked=True,
    )
    if not passed:
        raise RuntimeError("run02 import-only smoke gate failed")
    return result


def wait_for_ddm_resources(run_dir: Path) -> dict[str, float]:
    requirements = {"physical_available_gb": 14.0, "commit_headroom_gb": 28.0}
    deadline = time.time() + 8.0 * 3600.0
    while True:
        memory = memory_metrics()
        passed = all(memory[key] >= value for key, value in requirements.items())
        write_status(
            run_dir,
            state="ddm_resource_gate_passed" if passed else "waiting_ddm_resources",
            memory=memory,
            ddm_start_requirements=requirements,
        )
        if passed:
            return memory
        if time.time() >= deadline:
            raise TimeoutError("run02 DDM resource gate did not pass within eight hours")
        time.sleep(60.0)


def compare_touchstones(first: Path, second: Path) -> dict[str, Any]:
    first_s = parse_touchstone(first, NPORTS)
    second_s = parse_touchstone(second, NPORTS)
    delta = np.abs(second_s - first_s)
    diagonal = np.diag(delta)
    off_diagonal = delta[~np.eye(NPORTS, dtype=bool)]
    max_index = np.unravel_index(np.argmax(delta), delta.shape)
    return {
        "reference": str(first),
        "candidate": str(second),
        "max_abs_delta_s": float(np.max(delta)),
        "max_abs_delta_s_ports_1based": [int(value) + 1 for value in max_index],
        "rms_abs_delta_s": float(np.sqrt(np.mean(delta**2))),
        "max_diagonal_delta_s": float(np.max(diagonal)),
        "max_offdiagonal_delta_s": float(np.max(off_diagonal)),
    }


def ddm(run_dir: Path) -> dict[str, Any]:
    smoke_summary = run_dir / "import_smoke" / "import_smoke_summary.json"
    if not smoke_summary.exists() or not json.loads(
        smoke_summary.read_text(encoding="utf-8")
    ).get("import_smoke_gate_pass"):
        raise RuntimeError("run02 import smoke gate must pass before DDM")

    source_lock = clear_stale_source_lock(run_dir)
    starting_memory = wait_for_ddm_resources(run_dir)
    target = target_project(run_dir)
    ddm_dir = run_dir / "ddm"
    ddm_dir.mkdir(parents=True, exist_ok=True)
    touchstone = ddm_dir / "frozen_mesh_run02_ddm.s256p"
    messages = ddm_dir / "ddm_messages.txt"
    vbs = ddm_dir / "run_ddm.vbs"
    vbs.write_text(solve_vbs(target, touchstone, messages), encoding="ascii")
    stdout = ddm_dir / "ddm.stdout.log"
    stderr = ddm_dir / "ddm.stderr.log"
    results_dir = target.with_suffix(".aedtresults")
    before_profiles = (
        {path.resolve() for path in results_dir.rglob("*.profile")}
        if results_dir.exists()
        else set()
    )
    write_status(
        run_dir,
        state="running_ddm",
        ddm_start_memory=starting_memory,
        training_labels_locked=True,
    )
    with stdout.open("w", encoding="utf-8") as out, stderr.open(
        "w", encoding="utf-8"
    ) as err:
        process = subprocess.Popen(
            [str(ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=out,
            stderr=err,
        )
        low_resource_polls = 0
        while process.poll() is None:
            memory = memory_metrics()
            low = bool(
                memory["physical_available_gb"] < 0.5
                or memory["commit_headroom_gb"] < 2.5
            )
            low_resource_polls = low_resource_polls + 1 if low else 0
            write_status(
                run_dir,
                state="running_ddm",
                ddm_pid=process.pid,
                memory=memory,
                consecutive_low_resource_polls=low_resource_polls,
                training_labels_locked=True,
            )
            if low_resource_polls >= 3:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                process.wait(timeout=120)
                raise MemoryError("run02 DDM stopped after three critical-resource polls")
            time.sleep(30.0)
        return_code = int(process.returncode)

    profiles = list(results_dir.rglob("*.profile")) if results_dir.exists() else []
    new_profiles = [path for path in profiles if path.resolve() not in before_profiles]
    profile = max(new_profiles or profiles, key=lambda path: path.stat().st_mtime_ns) if profiles else None
    message_text = messages.read_text(encoding="utf-8", errors="ignore") if messages.exists() else ""
    solve_ok = bool(
        return_code == 0
        and "ANALYZE_ERR_NUMBER=0" in message_text
        and touchstone.exists()
        and touchstone.stat().st_size > 1_000_000
        and profile is not None
    )
    copied_profile = None
    if profile:
        copied_profile = ddm_dir / profile.name
        if profile.resolve() != copied_profile.resolve():
            shutil.copy2(profile, copied_profile)

    result: dict[str, Any] = {
        "return_code": return_code,
        "source_lock_preflight": source_lock,
        "touchstone": str(touchstone),
        "profile": str(copied_profile or profile) if profile else None,
        "solve_and_export_gate_pass": solve_ok,
        "training_labels_locked": True,
    }
    if solve_ok and profile is not None:
        result.update(profile_metrics(copied_profile or profile))
        result.update(s_metrics(touchstone))
        comparison = compare_touchstones(SOURCE_TOUCHSTONE, touchstone)
        numerical_valid = bool(
            result["touchstone_complete"]
            and float(result["s_reciprocity_max_abs"]) <= RECIPROCITY_LIMIT
            and float(result["s_passivity_sigma_max"]) <= PASSIVITY_SIGMA_LIMIT
        )
        delta_gate = comparison["max_abs_delta_s"] <= MAX_DELTA_S
        rl_gate = float(result["matched_passive_rl_min_db"]) >= MIN_MATCHED_RL_DB
        result.update(
            {
                "source_pass05_comparison": comparison,
                "numerical_smatrix_valid": numerical_valid,
                "source_delta_s_gate_pass": delta_gate,
                "source_delta_s_preferred_pass": comparison["max_abs_delta_s"]
                <= PREFERRED_DELTA_S,
                "matched_min_rl_gate_pass": rl_gate,
                "ddm_validation_gate_pass": bool(numerical_valid and delta_gate and rl_gate),
                "decision": (
                    "ddm_pass_proceed_to_independent_direct_cross_check"
                    if numerical_valid and delta_gate and rl_gate
                    else "ddm_fail_discard_old_pass5_fixed_mesh_candidate"
                ),
            }
        )
    else:
        result.update(
            {
                "ddm_validation_gate_pass": False,
                "decision": "ddm_solve_or_export_failed",
            }
        )
    (ddm_dir / "ddm_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# run02 fixed-mesh DDM validation",
        "",
        f"- Solve/export passed: {solve_ok}",
        f"- Decision: {result['decision']}",
        f"- Training labels locked: {result['training_labels_locked']}",
    ]
    if solve_ok:
        comparison = result["source_pass05_comparison"]
        lines[2:2] = [
            f"- Source vs run02 DDM max |Delta S|: {comparison['max_abs_delta_s']:.6f}",
            f"- Preferred <= 0.02: {result['source_delta_s_preferred_pass']}",
            f"- Required <= 0.05: {result['source_delta_s_gate_pass']}",
            f"- Matched minimum RL: {result['matched_passive_rl_min_db']:.3f} dB",
            f"- Matched RL >= 10 dB: {result['matched_min_rl_gate_pass']}",
        ]
    (ddm_dir / "ddm_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_status(
        run_dir,
        state="ddm_complete_pass" if result.get("ddm_validation_gate_pass") else "ddm_complete_fail",
        ddm=result,
        training_labels_locked=True,
    )
    if not solve_ok:
        raise RuntimeError("run02 DDM solve or S256 export failed")
    return result


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.mode == "prepare":
        print(json.dumps(prepare(run_dir), indent=2))
    elif args.mode == "smoke":
        print(json.dumps(smoke(run_dir), indent=2))
    elif args.mode == "ddm":
        print(json.dumps(ddm(run_dir), indent=2))
    elif args.mode == "all":
        prepare(run_dir)
        smoke(run_dir)
        print(json.dumps(ddm(run_dir), indent=2))
    else:
        print(json.dumps(read_status(run_dir), indent=2))


if __name__ == "__main__":
    main()
