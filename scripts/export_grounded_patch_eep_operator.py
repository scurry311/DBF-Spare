"""Export and analyze grounded-patch EEP operators in resumable port chunks."""

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
from run_grounded_patch_array_rebuild import parse_touchstone


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "URA_GroundedPatch_10GHz"
SOLUTION_NAME = "Setup_10GHz : LastAdaptive"
SPHERE_NAME = "InfiniteSphere_Theta0_90_Phi0_360"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="prepare")
    parser.add_argument("--project-path", type=Path, required=True)
    parser.add_argument("--touchstone-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--side", type=int, choices=(8, 16), required=True)
    parser.add_argument("--ports-per-job", type=int, default=16)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--design-name", default=DESIGN_NAME)
    parser.add_argument("--solution-name", default=SOLUTION_NAME)
    parser.add_argument("--sphere-name", default=SPHERE_NAME)
    parser.add_argument("--frequency-ghz", type=float, default=10.0)
    parser.add_argument("--series-match-inductance-nh", type=float, default=0.533)
    parser.add_argument("--series-match-q", type=float, default=50.0)
    parser.add_argument("--reciprocity-max", type=float, default=1.0e-4)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def vbs_path(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def chunk_vbs(
    project_path: Path,
    out_dir: Path,
    ports: list[str],
    chunk_id: str,
    design_name: str,
    solution_name: str,
    sphere_name: str,
    frequency_ghz: float,
) -> str:
    port_values = ", ".join(f'"{port}"' for port in ports)
    return f'''Option Explicit
Dim projectPath, projectName, designName, solutionName, sphereName, outDir, frequencyValue
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol, oReport, fso
Dim ports, portIndex, portName, reportName, outputPath, logFile, representation
projectPath = "{vbs_path(project_path)}"
projectName = "{project_path.stem}"
designName = "{design_name}"
solutionName = "{solution_name}"
sphereName = "{sphere_name}"
frequencyValue = "{frequency_ghz:.9g}GHz"
outDir = "{vbs_path(out_dir)}"
ports = Array({port_values})
Set fso = CreateObject("Scripting.FileSystemObject")
Set logFile = fso.CreateTextFile(outDir & "\eep_export_status_{chunk_id}.csv", True)
logFile.WriteLine "port_name,status,representation,output_path"
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject projectPath
Set oProject = oDesktop.SetActiveProject(projectName)
Set oDesign = oProject.SetActiveDesign(designName)
Set oSol = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")
For portIndex = LBound(ports) To UBound(ports)
    portName = CStr(ports(portIndex))
    reportName = "EEP_{chunk_id}_" & portName
    outputPath = outDir & "\eep_" & LCase(portName) & "_complex.csv"
    ApplySinglePort oSol, portName
    DeleteIfExists oReport, reportName
    representation = ExportComplexField(oReport, reportName, outputPath)
    If OutputFileOk(outputPath, 1000) Then
        logFile.WriteLine portName & ",complete," & representation & "," & outputPath
    Else
        logFile.WriteLine portName & ",failed," & representation & "," & outputPath
    End If
    DeleteIfExists oReport, reportName
Next
logFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ApplySinglePort(solModule, selectedPort)
    Dim sources, editArgs(), i, sourceName, magnitude
    sources = solModule.GetAllSources()
    ReDim editArgs(UBound(sources) + 1)
    editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)
    For i = LBound(sources) To UBound(sources)
        sourceName = CStr(sources(i))
        magnitude = "0W"
        If LCase(Split(sourceName, ":")(0)) = LCase(selectedPort) Then magnitude = "1W"
        editArgs(i + 1) = Array("Name:=", sourceName, "Magnitude:=", magnitude, "Phase:=", "0deg")
    Next
    solModule.EditSources editArgs
End Sub

Function ExportComplexField(reportModule, reportName, outputPath)
    On Error Resume Next
    Err.Clear
    reportModule.CreateReport reportName, "Far Fields", "Rectangular Contour Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array(frequencyValue), "Theta:=", Array("All"), "Phi:=", Array("All")), _
        Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", _
            Array("re(rETheta)", "im(rETheta)", "re(rEPhi)", "im(rEPhi)")), Array()
    If Err.Number = 0 Then reportModule.ExportToFile reportName, outputPath
    If Err.Number = 0 And OutputFileOk(outputPath, 1000) Then
        ExportComplexField = "real_imag"
        On Error GoTo 0
        Exit Function
    End If
    Err.Clear
    DeleteIfExists reportModule, reportName
    reportModule.CreateReport reportName, "Far Fields", "Rectangular Contour Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array(frequencyValue), "Theta:=", Array("All"), "Phi:=", Array("All")), _
        Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", _
            Array("mag(rETheta)", "ang_deg(rETheta)", "mag(rEPhi)", "ang_deg(rEPhi)")), Array()
    If Err.Number = 0 Then reportModule.ExportToFile reportName, outputPath
    If Err.Number = 0 And OutputFileOk(outputPath, 1000) Then
        ExportComplexField = "magnitude_phase_deg"
    Else
        ExportComplexField = "failed"
    End If
    On Error GoTo 0
End Function

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


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing EEP output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    port_count = int(args.side) ** 2
    ports = [f"P{index:03d}" for index in range(port_count)]
    manifest_rows = []
    job_rows = []
    for index, port in enumerate(ports):
        ix, iy = divmod(index, int(args.side))
        manifest_rows.append({"element_index": index, "port_name": port, "ix": ix, "iy": iy})
    for start in range(0, port_count, int(args.ports_per_job)):
        stop = min(port_count, start + int(args.ports_per_job))
        chunk_id = f"p{start:03d}_{stop - 1:03d}"
        script = args.out_dir / f"export_eep_{chunk_id}.vbs"
        script.write_text(
            chunk_vbs(
                args.project_path,
                args.out_dir,
                ports[start:stop],
                chunk_id,
                str(args.design_name),
                str(args.solution_name),
                str(args.sphere_name),
                float(args.frequency_ghz),
            ),
            encoding="ascii",
        )
        job_rows.append(
            {
                "chunk_id": chunk_id,
                "start_port": start,
                "stop_port_exclusive": stop,
                "expected_count": stop - start,
                "vbs_path": str(script.resolve()),
                "status": "pending",
            }
        )
    write_csv(args.out_dir / "port_manifest.csv", manifest_rows)
    write_csv(args.out_dir / "eep_jobs.csv", job_rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_scope": "grounded rectangular patch full-array EEP",
        "side": int(args.side),
        "port_count": port_count,
        "ports_per_job": int(args.ports_per_job),
        "job_count": len(job_rows),
        "project_path": str(args.project_path.resolve()),
        "touchstone_path": str(args.touchstone_path.resolve()),
        "status": "prepared_not_run",
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def job_complete(out_dir: Path, row: dict[str, str]) -> bool:
    for index in range(int(row["start_port"]), int(row["stop_port_exclusive"])):
        path = out_dir / f"eep_p{index:03d}_complex.csv"
        if not path.exists() or path.stat().st_size < 1000:
            return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.out_dir / "eep_jobs.csv").exists():
        prepare(args)
    jobs = read_csv(args.out_dir / "eep_jobs.csv")
    run_rows: list[dict[str, Any]] = []
    for row in jobs:
        if job_complete(args.out_dir, row):
            run_rows.append({**row, "status": "already_complete", "return_code": 0})
            continue
        log_path = args.out_dir / f"export_{row['chunk_id']}.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            process = subprocess.run(
                [str(args.ansys_exe), "-ng", "-RunScriptAndExit", row["vbs_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        complete = job_complete(args.out_dir, row)
        run_rows.append(
            {
                **row,
                "status": "complete" if complete else "failed_or_incomplete",
                "return_code": int(process.returncode),
            }
        )
        write_csv(args.out_dir / "eep_run_progress.csv", run_rows)
        if process.returncode != 0 or not complete:
            break
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_count": len(jobs),
        "complete_job_count": sum(row["status"] in ("complete", "already_complete") for row in run_rows),
        "run_complete": len(run_rows) == len(jobs) and all(
            row["status"] in ("complete", "already_complete") for row in run_rows
        ),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def matched_s_matrix(s_raw: np.ndarray, frequency_hz: float, inductance_nh: float, q_value: float) -> np.ndarray:
    identity = np.eye(s_raw.shape[0], dtype=np.complex128)
    z0 = 50.0
    z_ant = z0 * (identity + s_raw) @ np.linalg.inv(identity - s_raw)
    omega_l = 2.0 * np.pi * frequency_hz * float(inductance_nh) * 1.0e-9
    series_impedance = omega_l / float(q_value) + 1j * omega_l
    z_matched = z_ant + series_impedance * identity
    return (z_matched - z0 * identity) @ np.linalg.inv(z_matched + z0 * identity)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_csv(args.out_dir / "port_manifest.csv")
    ports = [row["port_name"] for row in manifest]
    etheta_rows: list[np.ndarray] = []
    ephi_rows: list[np.ndarray] = []
    theta_ref: np.ndarray | None = None
    phi_ref: np.ndarray | None = None
    failures: list[str] = []
    for port in ports:
        path = args.out_dir / f"eep_{port.lower()}_complex.csv"
        if not path.exists():
            failures.append(f"missing:{path.name}")
            continue
        try:
            angles, etheta, ephi = read_complex_field(path)
        except (OSError, ValueError) as error:
            failures.append(f"invalid:{path.name}:{error}")
            continue
        if theta_ref is None:
            theta_ref = angles[:, 0]
            phi_ref = angles[:, 1]
        elif not (np.array_equal(theta_ref, angles[:, 0]) and np.array_equal(phi_ref, angles[:, 1])):
            failures.append(f"grid_mismatch:{path.name}")
            continue
        etheta_rows.append(etheta)
        ephi_rows.append(ephi)
    field_complete = len(etheta_rows) == len(ports) and not failures
    s_complete = args.touchstone_path.exists() and args.touchstone_path.stat().st_size > 1000
    s_raw = np.empty((0, 0), dtype=np.complex128)
    if s_complete:
        try:
            s_raw = parse_touchstone(args.touchstone_path, len(ports)).astype(np.complex128)
            s_complete = bool(s_raw.shape == (len(ports), len(ports)) and np.all(np.isfinite(s_raw)))
        except (OSError, ValueError) as error:
            failures.append(f"touchstone:{error}")
            s_complete = False
    if field_complete:
        etheta = np.stack(etheta_rows).astype(np.complex64)
        ephi = np.stack(ephi_rows).astype(np.complex64)
        field_finite = bool(np.all(np.isfinite(etheta)) and np.all(np.isfinite(ephi)))
        field_nonzero = bool(max(float(np.max(np.abs(etheta))), float(np.max(np.abs(ephi)))) > 0.0)
    else:
        etheta = np.empty((0, 0), dtype=np.complex64)
        ephi = np.empty((0, 0), dtype=np.complex64)
        field_finite = False
        field_nonzero = False
    if s_complete:
        s_matched = matched_s_matrix(
            s_raw,
            float(args.frequency_ghz) * 1.0e9,
            float(args.series_match_inductance_nh),
            float(args.series_match_q),
        )
        reciprocity = float(np.max(np.abs(s_raw - s_raw.T)))
        passivity_sigma_max = float(np.max(np.linalg.svd(s_raw, compute_uv=False)))
        raw_rl_min = float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(s_raw)), 1.0e-15))))
        matched_rl_min = float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(s_matched)), 1.0e-15))))
    else:
        s_matched = np.empty((0, 0), dtype=np.complex128)
        reciprocity = passivity_sigma_max = raw_rl_min = matched_rl_min = float("nan")
    structural_gate = bool(
        field_complete
        and field_finite
        and field_nonzero
        and s_complete
        and reciprocity <= float(args.reciprocity_max)
        and passivity_sigma_max <= 1.0001
    )
    operator_path = args.out_dir / f"grounded_patch_eep_operator_{len(ports)}port.npz"
    if structural_gate:
        np.savez_compressed(
            operator_path,
            port_names=np.asarray(ports),
            element_indices=np.arange(len(ports), dtype=np.int16),
            element_ixiy=np.asarray([[int(row["ix"]), int(row["iy"])] for row in manifest], dtype=np.int8),
            theta_deg=theta_ref,
            phi_deg=phi_ref,
            etheta=etheta,
            ephi=ephi,
            s_raw=s_raw.astype(np.complex64),
            s_matched=s_matched.astype(np.complex64),
            frequency_ghz=np.asarray(float(args.frequency_ghz), dtype=np.float32),
            model_scope=np.asarray("grounded-patch HFSS EEP; one-port 1W incident-power basis"),
        )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port_count": len(ports),
        "field_complete": field_complete,
        "grid_point_count": int(theta_ref.size) if theta_ref is not None else 0,
        "s_parameter_complete": s_complete,
        "s_reciprocity_max_abs": reciprocity,
        "s_passivity_sigma_max": passivity_sigma_max,
        "raw_passive_rl_min_db": raw_rl_min,
        "matched_passive_rl_min_db": matched_rl_min,
        "structural_gate_pass": structural_gate,
        "operator_path": str(operator_path) if structural_gate else "",
        "failures": failures,
        "engineering_claim_allowed": False,
        "next_gate": "Direct HFSS superposition and selected joint-pass case validation are required.",
    }
    (args.out_dir / "operator_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def status(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.out_dir / "port_manifest.csv"
    expected = int(args.side) ** 2
    complete = sum(
        (args.out_dir / f"eep_p{index:03d}_complex.csv").exists()
        and (args.out_dir / f"eep_p{index:03d}_complex.csv").stat().st_size >= 1000
        for index in range(expected)
    )
    return {
        "prepared": manifest_path.exists(),
        "expected_port_count": expected,
        "complete_eep_count": complete,
        "touchstone_exists": args.touchstone_path.exists(),
        "analyzed": (args.out_dir / "operator_analysis_summary.json").exists(),
    }


def main() -> None:
    args = parse_args()
    result = {
        "prepare": prepare,
        "run": run,
        "analyze": analyze,
        "status": status,
    }[args.mode](args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
