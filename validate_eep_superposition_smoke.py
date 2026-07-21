"""Validate 16-port EEP superposition against direct HFSS combinations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_eep_16port_operator import read_complex_field


ROOT = Path(__file__).resolve().parent
DEFAULT_EEP_DIR = ROOT / "hfss_outputs" / "multitask_dataset" / "eep_smoke_16port_20260714"
DEFAULT_PROJECT = ROOT / "ura16_quick_10ghz_fullarray_run.aedt"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze"), default="prepare")
    parser.add_argument("--eep-dir", type=Path, default=DEFAULT_EEP_DIR)
    parser.add_argument("--operator-path", type=Path)
    parser.add_argument("--project-path", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--design-name", default="URA16_Quick_10GHz")
    parser.add_argument("--solution-name", default="Setup_10GHz : LastAdaptive")
    parser.add_argument("--sphere-name", default="InfiniteSphere_Theta0_90_Phi0_360")
    parser.add_argument("--frequency-ghz", type=float, default=10.0)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--nmse-max", type=float, default=1.0e-6)
    parser.add_argument("--magnitude-rmse-max-db", type=float, default=0.02)
    return parser.parse_args()


def operator_path(args: argparse.Namespace) -> Path:
    if args.operator_path is not None:
        return args.operator_path
    legacy = args.eep_dir / "eep_operator_16port.npz"
    if legacy.exists():
        return legacy
    candidates = sorted(args.eep_dir.glob("grounded_patch_eep_operator_*port.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one EEP operator in {args.eep_dir}, found {len(candidates)}; pass --operator-path"
        )
    return candidates[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_cases(ports: list[str], count: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(20260714)
    case_ids: list[str] = []
    weights: list[np.ndarray] = []
    for index in range(count):
        if index == 0:
            vector = np.zeros(len(ports), dtype=np.complex64)
            vector[0] = 1.0 + 0.0j
        elif index == 1:
            vector = np.ones(len(ports), dtype=np.complex64) / math.sqrt(len(ports))
        else:
            amplitude = rng.uniform(0.5, 1.0, size=len(ports))
            phase = rng.uniform(-math.pi, math.pi, size=len(ports))
            vector = amplitude * np.exp(1j * phase)
            vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
        case_ids.append(f"combo_{index:02d}")
        weights.append(vector.astype(np.complex64))
    return case_ids, np.stack(weights)


def write_vbs(
    path: Path,
    project_path: Path,
    eep_dir: Path,
    case_ids: list[str],
    source_files: list[Path],
    *,
    design_name: str,
    solution_name: str,
    sphere_name: str,
    frequency_ghz: float,
) -> None:
    cases_vbs = ", ".join(f'"{item}"' for item in case_ids)
    files_vbs = ", ".join(f'"{item.resolve()}"' for item in source_files)
    text = f'''Option Explicit
Dim projectPath, projectName, designName, solutionName, sphereName, outDir
Dim caseIds, sourceFiles, i, reportName, outputPath
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol, oReport, fso, statusFile

projectPath = "{project_path.resolve()}"
projectName = "{project_path.stem}"
designName = "{design_name}"
solutionName = "{solution_name}"
sphereName = "{sphere_name}"
outDir = "{eep_dir.resolve()}"
caseIds = Array({cases_vbs})
sourceFiles = Array({files_vbs})

Set fso = CreateObject("Scripting.FileSystemObject")
Set statusFile = fso.CreateTextFile(outDir & "\superposition_export_status.csv", True)
statusFile.WriteLine "case_id,status,output_path"
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject projectPath
Set oProject = oDesktop.SetActiveProject(projectName)
Set oDesign = oProject.SetActiveDesign(designName)
Set oSol = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")

For i = LBound(caseIds) To UBound(caseIds)
    ApplySourcesFromCsv oSol, sourceFiles(i)
    reportName = "EEPVAL_" & CStr(caseIds(i))
    outputPath = outDir & "\direct_" & CStr(caseIds(i)) & "_complex.csv"
    DeleteIfExists oReport, reportName
    On Error Resume Next
    Err.Clear
    oReport.CreateReport reportName, "Far Fields", "Rectangular Contour Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array("{frequency_ghz:.9g}GHz"), "Theta:=", Array("All"), "Phi:=", Array("All")), _
        Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", _
            Array("re(rETheta)", "im(rETheta)", "re(rEPhi)", "im(rEPhi)")), _
        Array()
    If Err.Number = 0 Then oReport.ExportToFile reportName, outputPath
    On Error GoTo 0
    If OutputFileOk(outputPath, 1000) Then
        statusFile.WriteLine CStr(caseIds(i)) & ",complete," & outputPath
    Else
        statusFile.WriteLine CStr(caseIds(i)) & ",failed," & outputPath
    End If
    DeleteIfExists oReport, reportName
Next

statusFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ApplySourcesFromCsv(solModule, csvPath)
    Dim magMap, phaseMap, srcFile, header, line, parts
    Dim sources, editArgs(), i, sourceName, baseName, magnitude, phase
    Set magMap = CreateObject("Scripting.Dictionary")
    Set phaseMap = CreateObject("Scripting.Dictionary")
    Set srcFile = fso.OpenTextFile(csvPath, 1)
    If Not srcFile.AtEndOfStream Then header = srcFile.ReadLine
    Do Until srcFile.AtEndOfStream
        line = Trim(srcFile.ReadLine)
        If Len(line) > 0 Then
            parts = Split(line, ",")
            magMap(LCase(parts(0))) = parts(1)
            phaseMap(LCase(parts(0))) = parts(2)
        End If
    Loop
    srcFile.Close
    sources = solModule.GetAllSources()
    ReDim editArgs(UBound(sources) + 1)
    editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)
    For i = LBound(sources) To UBound(sources)
        sourceName = CStr(sources(i))
        baseName = LCase(Split(sourceName, ":")(0))
        magnitude = "0W"
        phase = "0deg"
        If magMap.Exists(baseName) Then
            magnitude = CStr(magMap(baseName)) & "W"
            phase = CStr(phaseMap(baseName)) & "deg"
        End If
        editArgs(i + 1) = Array("Name:=", sourceName, "Magnitude:=", magnitude, "Phase:=", phase)
    Next
    solModule.EditSources editArgs
End Sub

Sub DeleteIfExists(reportModule, reportName)
    On Error Resume Next
    reportModule.DeleteReports Array(reportName)
    On Error GoTo 0
End Sub

Function OutputFileOk(filePath, minBytes)
    OutputFileOk = False
    If fso.FileExists(filePath) Then
        If fso.GetFile(filePath).Size >= minBytes Then OutputFileOk = True
    End If
End Function
'''
    path.write_text(text, encoding="ascii")


def write_runner(path: Path, ansys: Path, vbs: Path, log_path: Path) -> None:
    path.write_text(
        f'''$ErrorActionPreference = "Stop"
& "{ansys.resolve()}" -ng -RunScriptAndExit "{vbs.resolve()}" *>&1 | Tee-Object -FilePath "{log_path.resolve()}"
if ($LASTEXITCODE -ne 0) {{ throw "AEDT superposition smoke failed: $LASTEXITCODE" }}
''',
        encoding="ascii",
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    selected_operator = operator_path(args)
    operator = np.load(selected_operator, allow_pickle=False)
    ports = [str(port) for port in operator["port_names"]]
    case_ids, weights = generate_cases(ports, int(args.case_count))
    source_files: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    for case_pos, case_id in enumerate(case_ids):
        source_path = args.eep_dir / f"sources_{case_id}.csv"
        source_rows = []
        for port_pos, port in enumerate(ports):
            value = weights[case_pos, port_pos]
            source_rows.append(
                {
                    "port_name": port,
                    "incident_power_w": float(abs(value) ** 2),
                    "phase_deg": float(np.rad2deg(np.angle(value))),
                }
            )
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "port_name": port,
                    "weight_real": float(value.real),
                    "weight_imag": float(value.imag),
                }
            )
        write_csv(source_path, source_rows)
        source_files.append(source_path)
    write_csv(args.eep_dir / "superposition_cases.csv", manifest_rows)
    np.savez_compressed(args.eep_dir / "superposition_cases.npz", case_id=np.asarray(case_ids), weights=weights)
    vbs = args.eep_dir / "export_eep_superposition_cases.vbs"
    runner = args.eep_dir / "run_eep_superposition_cases.ps1"
    write_vbs(
        vbs,
        args.project_path,
        args.eep_dir,
        case_ids,
        source_files,
        design_name=str(args.design_name),
        solution_name=str(args.solution_name),
        sphere_name=str(args.sphere_name),
        frequency_ghz=float(args.frequency_ghz),
    )
    write_runner(runner, args.ansys_exe, vbs, args.eep_dir / "export_eep_superposition_cases.log")
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(case_ids),
        "port_count": len(ports),
        "operator_path": str(selected_operator),
        "project_path": str(args.project_path),
        "design_name": str(args.design_name),
        "solution_name": str(args.solution_name),
        "sphere_name": str(args.sphere_name),
        "hfss_solve_required": False,
        "runner": str(runner),
    }
    (args.eep_dir / "superposition_prepare_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    operator = np.load(operator_path(args), allow_pickle=False)
    cases = np.load(args.eep_dir / "superposition_cases.npz", allow_pickle=False)
    etheta_operator = operator["etheta"]
    ephi_operator = operator["ephi"]
    rows: list[dict[str, Any]] = []
    for case_pos, case_id_raw in enumerate(cases["case_id"]):
        case_id = str(case_id_raw)
        direct_path = args.eep_dir / f"direct_{case_id}_complex.csv"
        if not direct_path.exists():
            rows.append({"case_id": case_id, "complete": 0})
            continue
        angles, direct_theta, direct_phi = read_complex_field(direct_path)
        weights = cases["weights"][case_pos]
        reconstructed_theta = weights @ etheta_operator
        reconstructed_phi = weights @ ephi_operator
        direct = np.concatenate([direct_theta, direct_phi])
        reconstructed = np.concatenate([reconstructed_theta, reconstructed_phi])
        error = reconstructed - direct
        nmse = float(np.sum(np.abs(error) ** 2) / max(float(np.sum(np.abs(direct) ** 2)), 1.0e-20))
        correlation = float(abs(np.vdot(direct, reconstructed)) / max(float(np.linalg.norm(direct) * np.linalg.norm(reconstructed)), 1.0e-20))
        scale = np.vdot(reconstructed, direct) / max(
            float(np.vdot(reconstructed, reconstructed).real), 1.0e-20
        )
        reconstructed_scaled = scale * reconstructed
        scaled_error = reconstructed_scaled - direct
        scaled_nmse = float(
            np.sum(np.abs(scaled_error) ** 2) / max(float(np.sum(np.abs(direct) ** 2)), 1.0e-20)
        )
        direct_db = 20.0 * np.log10(np.maximum(np.abs(direct), 1.0e-12))
        reconstructed_db = 20.0 * np.log10(np.maximum(np.abs(reconstructed), 1.0e-12))
        reconstructed_scaled_db = 20.0 * np.log10(np.maximum(np.abs(reconstructed_scaled), 1.0e-12))
        visible = direct_db >= float(direct_db.max() - 40.0)
        magnitude_rmse_db = float(np.sqrt(np.mean((reconstructed_db[visible] - direct_db[visible]) ** 2)))
        scaled_magnitude_rmse_db = float(
            np.sqrt(np.mean((reconstructed_scaled_db[visible] - direct_db[visible]) ** 2))
        )
        phase_error = np.angle(reconstructed[visible] * np.conj(direct[visible]))
        phase_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(phase_error**2))))
        rows.append(
            {
                "case_id": case_id,
                "complete": 1,
                "grid_point_count": int(angles.shape[0]),
                "complex_nmse": nmse,
                "complex_correlation": correlation,
                "best_fit_scale_magnitude": float(abs(scale)),
                "best_fit_scale_phase_deg": float(np.rad2deg(np.angle(scale))),
                "scale_corrected_complex_nmse": scaled_nmse,
                "magnitude_rmse_db_visible40": magnitude_rmse_db,
                "scale_corrected_magnitude_rmse_db_visible40": scaled_magnitude_rmse_db,
                "phase_rmse_deg_visible40": phase_rmse_deg,
                "passed": int(nmse <= float(args.nmse_max) and magnitude_rmse_db <= float(args.magnitude_rmse_max_db)),
                "shape_passed": int(scaled_nmse <= 0.02 and correlation >= 0.99),
            }
        )
    write_csv(args.eep_dir / "superposition_validation_metrics.csv", rows)
    complete = [row for row in rows if row.get("complete") == 1]
    passed = bool(len(complete) == int(args.case_count) and all(row.get("passed") == 1 for row in complete))
    shape_passed = bool(
        len(complete) == int(args.case_count) and all(row.get("shape_passed") == 1 for row in complete)
    )
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_case_count": int(args.case_count),
        "complete_case_count": len(complete),
        "all_cases_passed": passed,
        "all_shapes_passed_after_complex_scale": shape_passed,
        "complex_nmse_max": max((row["complex_nmse"] for row in complete), default=float("nan")),
        "magnitude_rmse_db_max": max((row["magnitude_rmse_db_visible40"] for row in complete), default=float("nan")),
        "complex_correlation_min": min((row["complex_correlation"] for row in complete), default=float("nan")),
        "scale_corrected_complex_nmse_max": max(
            (row["scale_corrected_complex_nmse"] for row in complete), default=float("nan")
        ),
        "best_fit_scale_magnitude_range": [
            min((row["best_fit_scale_magnitude"] for row in complete), default=float("nan")),
            max((row["best_fit_scale_magnitude"] for row in complete), default=float("nan")),
        ],
        "allow_256_port_export": False,
        "next_gate": "Keep the 256-port export blocked until 50-100 full-array HFSS cases confirm reconstruction error.",
        "metrics_csv": str(args.eep_dir / "superposition_validation_metrics.csv"),
    }
    (args.eep_dir / "superposition_analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "run":
        if not (args.eep_dir / "run_eep_superposition_cases.ps1").exists():
            prepare(args)
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(args.eep_dir / "run_eep_superposition_cases.ps1")],
            cwd=str(ROOT),
            check=False,
        )
        result = analyze(args)
        result["runner_exit_code"] = int(completed.returncode)
        (args.eep_dir / "superposition_analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = analyze(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
