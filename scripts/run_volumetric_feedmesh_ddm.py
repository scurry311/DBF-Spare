#!/usr/bin/env python3
"""Run guarded pass1-pass3 DDM validation for the volumetric feed-mesh branch."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "hfss_outputs" / "grounded_patch_direct_16x16_volumetric_feedmesh_20260722_run01"
PROJECT_DIR = RUN / "grounded_patch_16x16"
PROJECT = PROJECT_DIR / "grounded_patch_16x16.aedt"
ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN = "URA_GroundedPatch_10GHz"
SETUP = "Setup_VolFeed_DDM"
STATUS = RUN / "volumetric_ddm_status.json"
ANALYZER = ROOT / "scripts" / "analyze_ddm_recovery_stage.py"
OLD_SMALL_SEGMENT_COUNT = 990
DDM_TASKS = 4


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


def available_memory_gb() -> float:
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return float(status.available_physical) / (1024.0**3)


def resource_gate(metrics: dict[str, object], prepare: dict[str, object]) -> bool:
    matrix_sizes = [int(value) for value in metrics.get("matrix_size_by_pass", {}).values()]
    total_memory = metrics.get("adaptive_total_domain_memory_gb")
    return bool(
        matrix_sizes
        and max(matrix_sizes) <= int(prepare.get("max_matrix_size", 5_800_000))
        and total_memory is not None
        and float(total_memory) <= float(prepare.get("max_total_domain_memory_gb", 18.5))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--ddm-tasks", type=int, choices=(4, 6), default=4)
    return parser.parse_args()


def write_status(**values: object) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(current, indent=2), encoding="utf-8")


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def vbs_text(pass_number: int, touchstone: Path, messages: Path) -> str:
    setup_action = "" if pass_number > 1 else f'''On Error Resume Next
oAnalysis.DeleteSetups Array("{SETUP}")
On Error GoTo 0
oAnalysis.InsertSetup "HfssDriven", Array( _
    "NAME:{SETUP}", "SolveType:=", "Single", "Frequency:=", "10GHz", _
    "MaxDeltaS:=", 0.05, "UseMatrixConv:=", False, _
    "MaximumPasses:=", 1, "MinimumPasses:=", 1, _
    "MinimumConvergedPasses:=", 1, "PercentRefinement:=", 5, _
    "IsEnabled:=", True, "BasisOrder:=", 1, "DoLambdaRefine:=", True, _
    "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, _
    "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False, _
    "DrivenSolverType:=", "Domain Decomposition", _
    "IterativeResidual:=", 0.000001, "DDMSolverResidual:=", 0.000001, _
    "SaveRadFieldsOnly:=", False, "SaveAnyFields:=", False)
'''
    if pass_number > 1:
        setup_action = f'''oAnalysis.EditSetup "{SETUP}", Array( _
    "NAME:{SETUP}", "SolveType:=", "Single", "Frequency:=", "10GHz", _
    "MaxDeltaS:=", 0.05, "UseMatrixConv:=", False, _
    "MaximumPasses:=", {pass_number}, "MinimumPasses:=", 2, _
    "MinimumConvergedPasses:=", 2, "PercentRefinement:=", 5, _
    "IsEnabled:=", True, "BasisOrder:=", 1, "DoLambdaRefine:=", True, _
    "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, _
    "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False, _
    "DrivenSolverType:=", "Domain Decomposition", _
    "IterativeResidual:=", 0.000001, "DDMSolverResidual:=", 0.000001, _
    "SaveRadFieldsOnly:=", False, "SaveAnyFields:=", False)
'''
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oAnalysis, oSolutions
Dim fso, outFile, values, variation, msgs, i, analyzeErr, analyzeDescription
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{vp(messages)}", True)
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.OpenProject "{vp(PROJECT)}"
Set oProject = oDesktop.SetActiveProject("grounded_patch_16x16")
Set oDesign = oProject.SetActiveDesign("{DESIGN}")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
{setup_action}
oProject.Save
On Error Resume Next
Err.Clear
oDesign.Analyze "{SETUP}"
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
    values = oSolutions.ListVariations("{SETUP}:LastAdaptive")
    variation = CStr(values(LBound(values)))
    oSolutions.ExportNetworkData variation, Array("{SETUP}:LastAdaptive"), 3, _
        "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
End If
oProject.Save
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def g3derr_summary() -> dict[str, object]:
    files = list((PROJECT_DIR / "grounded_patch_16x16.aedtresults").rglob("*.g3derr"))
    if not files:
        return {
            "file_count": 0,
            "error_block_count": 0,
            "feed_or_port_error_count": 0,
            "min_length_mm": None,
            "max_length_mm": None,
            "top_bodies": [],
        }
    path = max(files, key=lambda item: item.stat().st_mtime_ns)
    text = path.read_text(encoding="utf-8", errors="ignore")
    counts: Counter[str] = Counter()
    lengths: list[float] = []
    for block in re.findall(r"BEGIN_ERR(.*?)END_ERR", text, re.DOTALL):
        body = re.search(r"Small mesh segment detected on body\s*:\s*([^\r\n]+)", block, re.I)
        length = re.search(r"Segment length\(s\)\s*:\s*([0-9.eE+\-]+)mm", block, re.I)
        if body:
            counts[body.group(1).strip()] += 1
        if length:
            lengths.append(float(length.group(1)))
    feed_or_port = sum(
        count
        for name, count in counts.items()
        if name.startswith(("FeedNbr_", "PortSheet_"))
    )
    feed_core = sum(count for name, count in counts.items() if name.startswith("FeedCore_"))
    return {
        "path": str(path),
        "file_count": len(files),
        "error_block_count": sum(counts.values()),
        "feed_or_port_error_count": feed_or_port,
        "feed_core_error_count": feed_core,
        "all_feed_region_error_count": feed_or_port + feed_core,
        "min_length_mm": min(lengths) if lengths else None,
        "max_length_mm": max(lengths) if lengths else None,
        "top_bodies": counts.most_common(30),
    }


def topology_warning_count(messages: Path) -> int:
    text = messages.read_text(encoding="utf-8", errors="ignore") if messages.exists() else ""
    patterns = (
        "Too many conductors touch lumped port",
        "'0' conductors touch lumped port",
        "'1' conductors touch lumped port",
        "typically a lumped port contains 2 conductors",
        "Both endpoints of port lines must lie on port",
    )
    return sum(text.count(pattern) for pattern in patterns)


def run_pass(pass_number: int, reference: Path) -> dict[str, object]:
    minimum_free_gb = 3.5
    free_gb = available_memory_gb()
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"Refusing pass{pass_number:02d}: only {free_gb:.2f} GB physical memory free; "
            f"require {minimum_free_gb:.2f} GB"
        )
    stage = RUN / "stages" / f"pass{pass_number:02d}"
    stage.mkdir(parents=True, exist_ok=True)
    touchstone = PROJECT_DIR / f"grounded_patch_16x16_volfeed_pass{pass_number:02d}.s256p"
    messages = RUN / f"volfeed_pass{pass_number:02d}_messages.txt"
    vbs = RUN / f"run_volfeed_pass{pass_number:02d}.vbs"
    vbs.write_text(vbs_text(pass_number, touchstone, messages), encoding="ascii")
    stdout = RUN / f"volfeed_pass{pass_number:02d}.stdout.log"
    stderr = RUN / f"volfeed_pass{pass_number:02d}.stderr.log"
    write_status(state=f"running_pass{pass_number:02d}", current_pass=pass_number)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        command = [str(ANSYS), "-ng"]
        if DDM_TASKS > 4:
            command.extend(
                [
                    "-Distributed",
                    "-MachineList",
                    f"list=scurry:{DDM_TASKS}:12:90%:0",
                ]
            )
        command.extend(["-RunScriptAndExit", str(vbs)])
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=out,
            stderr=err,
            check=False,
        )
    if result.returncode != 0 or not touchstone.exists():
        raise RuntimeError(
            f"pass{pass_number:02d} failed: return={result.returncode}, s={touchstone.exists()}"
        )
    profiles = list((PROJECT_DIR / "grounded_patch_16x16.aedtresults").rglob("*.profile"))
    profile = max(profiles, key=lambda item: item.stat().st_mtime_ns)
    stage_profile = stage / profile.name
    stage_s = stage / touchstone.name
    shutil.copy2(profile, stage_profile)
    shutil.copy2(touchstone, stage_s)
    shutil.copy2(messages, stage / messages.name)
    metrics_path = stage / "stage_metrics.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "scripts")
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--stage",
            f"volfeed_pass{pass_number:02d}",
            "--profile",
            str(stage_profile),
            "--touchstone",
            str(stage_s),
            "--reference",
            str(reference),
            "--output",
            str(metrics_path),
            "--port-csv",
            str(stage / f"port_stability_vs_reference_pass{pass_number:02d}.csv"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["topology_warning_count"] = topology_warning_count(messages)
    metrics["g3derr"] = g3derr_summary()
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    global RUN, PROJECT_DIR, PROJECT, STATUS, DDM_TASKS
    args = parse_args()
    DDM_TASKS = args.ddm_tasks
    RUN = args.run_dir.resolve()
    PROJECT_DIR = RUN / "grounded_patch_16x16"
    PROJECT = PROJECT_DIR / "grounded_patch_16x16.aedt"
    STATUS = RUN / "volumetric_ddm_status.json"
    prepare_paths = (
        RUN / "clean_ddm_restart_prepare_summary.json",
        RUN / "volumetric_feedmesh_prepare_summary.json",
        RUN / "pecsheet_prepare_summary.json",
        RUN / "pecsheet_perfecte_prepare_summary.json",
        RUN / "ddm_domain_prepare_summary.json",
        RUN / "layered_feedmesh_prepare_summary.json",
    )
    prepare_path = next((path for path in prepare_paths if path.exists()), None)
    if prepare_path is None:
        raise FileNotFoundError("No feed-mesh configuration summary found")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    if not prepare.get("configuration_smoke_pass"):
        raise RuntimeError("Configuration smoke did not pass")
    reference = RUN / "reference_pass05" / "grounded_patch_16x16_ddm_pass05.s256p"
    layered = "core_mesh_mm" in prepare
    write_status(
        state="starting_initial_mesh_smoke",
        policy={
            "configuration_mode": prepare.get("configuration_mode", "volumetric_feedmesh"),
            "mesh_mm": prepare.get("port_surface_mesh_mm", prepare.get("mesh_mm", 0.18)),
            "region_mm": prepare.get("region_xy_mm"),
            "region_depth_mm": prepare.get("region_z_mm"),
            "refine_inside": prepare.get("refine_inside", False),
            "percent_refinement": 5.0,
            "ddm_residual": 1.0e-6,
            "old_small_segment_count": OLD_SMALL_SEGMENT_COUNT,
            "ddm_tasks": DDM_TASKS,
            "requested_cores": 12 if DDM_TASKS > 4 else 4,
            "layered_feedmesh": layered,
            "core_mesh_mm": prepare.get("core_mesh_mm"),
            "halo_mesh_mm": prepare.get("halo_mesh_mm"),
            "first_pass_tetrahedra_target": [1_000_000, 1_400_000] if layered else None,
            "reference_max_abs_delta_s_limit": 0.02 if layered else None,
            "reference_max_rl_delta_db_limit": 0.2 if layered else None,
            "minimum_free_memory_before_pass_gb": 3.5,
            "max_total_domain_memory_gb": prepare.get(
                "max_total_domain_memory_gb", 18.5
            ),
            "max_matrix_size": prepare.get("max_matrix_size", 5_800_000),
        },
        training_labels_locked=True,
    )
    try:
        pass1 = run_pass(1, reference)
        pass1["resource_gate_pass"] = resource_gate(pass1, prepare)
        smoke_pass = bool(
            pass1["numerical_smatrix_valid"]
            and pass1["topology_warning_count"] == 0
            and pass1["g3derr"]["feed_or_port_error_count"] == 0
            and pass1["g3derr"]["error_block_count"] <= OLD_SMALL_SEGMENT_COUNT
            and pass1["resource_gate_pass"]
        )
        pass1_tetrahedra = int(pass1.get("tetrahedra_by_pass", {}).get("1", 0))
        pass1["tetrahedra_target_min"] = 1_000_000 if layered else None
        pass1["tetrahedra_target_max"] = 1_400_000 if layered else None
        pass1["tetrahedra_target_pass"] = bool(
            not layered or 1_000_000 <= pass1_tetrahedra <= 1_400_000
        )
        if layered:
            comparison = pass1.get("comparison", {})
            pass1["reference_delta_s_gate_pass"] = bool(
                float(comparison.get("max_abs_delta_s", float("inf"))) <= 0.02
            )
            pass1["reference_rl_delta_gate_pass"] = bool(
                float(comparison.get("max_abs_passive_rl_delta_db", float("inf"))) <= 0.2
            )
            smoke_pass = bool(
                smoke_pass
                and pass1["reference_delta_s_gate_pass"]
                and pass1["reference_rl_delta_gate_pass"]
            )
        (RUN / "stages" / "pass01" / "stage_metrics.json").write_text(
            json.dumps(pass1, indent=2), encoding="utf-8"
        )
        write_status(initial_mesh_smoke=pass1, initial_mesh_smoke_pass=smoke_pass)
        if not smoke_pass:
            write_status(state="stopped_initial_mesh_smoke_failed")
            return
        pass2 = run_pass(2, RUN / "stages" / "pass01" / "grounded_patch_16x16_volfeed_pass01.s256p")
        pass2["resource_gate_pass"] = resource_gate(pass2, prepare)
        if (
            not pass2["numerical_smatrix_valid"]
            or pass2["topology_warning_count"]
            or not pass2["resource_gate_pass"]
        ):
            write_status(state="stopped_pass02_invalid", pass02=pass2)
            return
        write_status(pass02=pass2)
        pass3 = run_pass(3, RUN / "stages" / "pass02" / "grounded_patch_16x16_volfeed_pass02.s256p")
        pass3["resource_gate_pass"] = resource_gate(pass3, prepare)
        delta2 = float(pass2["final_delta_s"])
        delta3 = float(pass3["final_delta_s"])
        monotonic = bool(
            pass3["numerical_smatrix_valid"]
            and pass3["resource_gate_pass"]
            and delta3 < delta2
        )
        consecutive_two_round_gate = bool(
            pass2["numerical_smatrix_valid"]
            and pass3["numerical_smatrix_valid"]
            and pass2["resource_gate_pass"]
            and pass3["resource_gate_pass"]
            and pass2["topology_warning_count"] == 0
            and pass3["topology_warning_count"] == 0
            and pass2["g3derr"]["feed_or_port_error_count"] == 0
            and pass3["g3derr"]["feed_or_port_error_count"] == 0
            and delta2 <= 0.05
            and delta3 <= 0.05
        )
        write_status(
            pass03=pass3,
            delta_s_monotonic_decrease=monotonic,
            consecutive_two_round_delta_s_gate_pass=consecutive_two_round_gate,
            state=(
                "completed_two_consecutive_delta_s_passes"
                if consecutive_two_round_gate
                else "stopped_two_consecutive_delta_s_gate_failed"
            ),
        )
    except Exception as exc:
        write_status(state="stopped_execution_error", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
