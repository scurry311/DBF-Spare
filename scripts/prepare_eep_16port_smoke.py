"""Prepare or run a 16-port embedded-element-pattern smoke export.

The script never changes the solved project on disk.  It writes a standalone
AEDT/VBScript post-processing job that excites one representative port at a
time, exports complex Etheta/Ephi fields, and exports a complex S-parameter
submatrix for the same ports.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = ROOT / "models" / "hfss" / "ura16_quick_10ghz_fullarray_run.aedt"
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_OUT_DIR = DEFAULT_DATASET / "eep_smoke_16port_20260714"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "URA16_Quick_10GHz"
SOLUTION_NAME = "Setup_10GHz : LastAdaptive"
SPHERE_NAME = "InfiniteSphere_Theta0_90_Phi0_360"
REPRESENTATIVE_INDICES = (0, 7, 8, 15, 112, 119, 120, 127, 128, 135, 136, 143, 240, 247, 248, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze"), default="prepare")
    parser.add_argument("--project-path", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument(
        "--port-indices",
        default=",".join(str(index) for index in REPRESENTATIVE_INDICES),
        help="Comma-separated element indices; use a single index for an interface smoke test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def project_name(path: Path) -> str:
    return path.stem


def write_vbs(path: Path, project_path: Path, out_dir: Path, ports: list[str]) -> None:
    port_values = ", ".join(f'"{port}"' for port in ports)
    text = f'''Option Explicit

Dim projectPath, projectName, designName, solutionName, sphereName, outDir
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol, oReport, fso
Dim ports, portIndex, portName, reportName, outputPath
Dim logFile, representation

projectPath = "{project_path}"
projectName = "{project_name(project_path)}"
designName = "{DESIGN_NAME}"
solutionName = "{SOLUTION_NAME}"
sphereName = "{SPHERE_NAME}"
outDir = "{out_dir}"
ports = Array({port_values})

Set fso = CreateObject("Scripting.FileSystemObject")
EnsureFolder outDir
Set logFile = fso.CreateTextFile(outDir & "\eep_export_status.csv", True)
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
    reportName = "EEP16_" & portName
    outputPath = outDir & "\eep_" & LCase(portName) & "_complex.csv"
    ApplySinglePort oSol, portName
    DeleteIfExists oReport, reportName
    representation = ExportComplexField(oReport, reportName, outputPath)
    If OutputFileOk(outputPath, 1000) Then
        logFile.WriteLine CsvCell(portName) & ",complete," & CsvCell(representation) & "," & CsvCell(outputPath)
    Else
        logFile.WriteLine CsvCell(portName) & ",failed_missing_output," & CsvCell(representation) & "," & CsvCell(outputPath)
    End If
    DeleteIfExists oReport, reportName
Next

ExportSSubmatrix oReport, ports, outDir & "\s_parameter_submatrix_complex.csv"
logFile.Close
' Do not save: source edits and temporary reports are post-processing state only.
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
        If LCase(BasePortName(sourceName)) = LCase(selectedPort) Then magnitude = "1W"
        editArgs(i + 1) = Array("Name:=", sourceName, "Magnitude:=", magnitude, "Phase:=", "0deg")
    Next
    solModule.EditSources editArgs
End Sub

Function BasePortName(sourceName)
    Dim parts
    parts = Split(CStr(sourceName), ":")
    BasePortName = parts(0)
End Function

Function ExportComplexField(reportModule, reportName, outputPath)
    On Error Resume Next
    Err.Clear
    reportModule.CreateReport reportName, "Far Fields", "Rectangular Contour Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("All")), _
        Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", _
            Array("re(rETheta)", "im(rETheta)", "re(rEPhi)", "im(rEPhi)")), _
        Array()
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
        Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("All")), _
        Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", _
            Array("mag(rETheta)", "ang_deg(rETheta)", "mag(rEPhi)", "ang_deg(rEPhi)")), _
        Array()
    If Err.Number = 0 Then reportModule.ExportToFile reportName, outputPath
    If Err.Number = 0 And OutputFileOk(outputPath, 1000) Then
        ExportComplexField = "magnitude_phase_deg"
    Else
        ExportComplexField = "failed"
    End If
    On Error GoTo 0
End Function

Sub ExportSSubmatrix(reportModule, selectedPorts, outputPath)
    Dim reportName, traces(), i, j, cursor, expression
    reportName = "EEP16_S_Submatrix"
    ReDim traces(2 * (UBound(selectedPorts) + 1) * (UBound(selectedPorts) + 1) - 1)
    cursor = 0
    For i = LBound(selectedPorts) To UBound(selectedPorts)
        For j = LBound(selectedPorts) To UBound(selectedPorts)
            expression = "S(" & CStr(selectedPorts(i)) & "," & CStr(selectedPorts(j)) & ")"
            traces(cursor) = "re(" & expression & ")"
            traces(cursor + 1) = "im(" & expression & ")"
            cursor = cursor + 2
        Next
    Next
    DeleteIfExists reportModule, reportName
    On Error Resume Next
    reportModule.CreateReport reportName, "Modal Solution Data", "Rectangular Plot", solutionName, _
        Array(), _
        Array("Freq:=", Array("All")), _
        Array("X Component:=", "Freq", "Y Component:=", traces), _
        Array()
    If Err.Number = 0 Then reportModule.ExportToFile reportName, outputPath
    On Error GoTo 0
    DeleteIfExists reportModule, reportName
End Sub

Sub EnsureFolder(folderPath)
    If Not fso.FolderExists(folderPath) Then fso.CreateFolder(folderPath)
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

Function CsvCell(value)
    CsvCell = Replace(CStr(value), ",", ";")
End Function
'''
    path.write_text(text, encoding="ascii")


def write_runner(path: Path, ansys_exe: Path, script_path: Path, log_path: Path) -> None:
    text = f'''$ErrorActionPreference = "Stop"
$ansys = "{ansys_exe}"
$script = "{script_path}"
$log = "{log_path}"
if (-not (Test-Path -LiteralPath $ansys)) {{ throw "ansysedt.exe not found: $ansys" }}
& $ansys -ng -RunScriptAndExit $script *>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) {{ throw "AEDT EEP smoke failed with exit code $LASTEXITCODE" }}
'''
    path.write_text(text, encoding="ascii")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace generated control files: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_npz(args.dataset_dir / "dataset_arrays.npz")
    indices = [int(part.strip()) for part in str(args.port_indices).split(",") if part.strip()]
    if not indices or any(index < 0 or index >= int(arrays["port_names"].shape[0]) for index in indices):
        raise ValueError(f"Invalid --port-indices: {args.port_indices}")
    ports = [str(arrays["port_names"][index]) for index in indices]
    manifest_rows = [
        {
            "representative_order": order,
            "element_index": index,
            "port_name": ports[order],
            "ix": int(arrays["element_ixiy"][index, 0]),
            "iy": int(arrays["element_ixiy"][index, 1]),
        }
        for order, index in enumerate(indices)
    ]
    write_csv(args.out_dir / "representative_ports.csv", manifest_rows)
    vbs_path = args.out_dir / "export_eep_16port.vbs"
    ps1_path = args.out_dir / "run_eep_16port.ps1"
    write_vbs(vbs_path, args.project_path.resolve(), args.out_dir.resolve(), ports)
    write_runner(ps1_path, args.ansys_exe.resolve(), vbs_path.resolve(), (args.out_dir / "export_eep_16port.log").resolve())
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "prepare",
        "project_path": str(args.project_path.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "port_count": len(ports),
        "ports": ports,
        "frequency_ghz": 10.0,
        "source_convention": "driven_modal_incident_power_watt",
        "eep_basis": "one_port_1W_incident_power",
        "field_components": ["Etheta_real", "Etheta_imag", "Ephi_real", "Ephi_imag"],
        "fallback_components": ["Etheta_mag", "Etheta_phase_deg", "Ephi_mag", "Ephi_phase_deg"],
        "hfss_solve_required": False,
        "large_scale_allowed": False,
        "outputs": {
            "manifest": str(args.out_dir / "representative_ports.csv"),
            "vbs": str(vbs_path),
            "runner": str(ps1_path),
        },
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    status_path = args.out_dir / "eep_export_status.csv"
    rows: list[dict[str, str]] = []
    if status_path.exists():
        with status_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    complete = [row for row in rows if row.get("status") == "complete"]
    manifest_path = args.out_dir / "representative_ports.csv"
    expected = 0
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            expected = sum(1 for _row in csv.DictReader(handle))
    s_path = args.out_dir / "s_parameter_submatrix_complex.csv"
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_port_count": expected,
        "status_row_count": len(rows),
        "complete_eep_count": len(complete),
        "s_parameter_complete": bool(s_path.exists() and s_path.stat().st_size >= 1000),
        "representations": sorted({row.get("representation", "") for row in complete}),
        "smoke_passed": bool(expected > 0 and len(complete) == expected and s_path.exists() and s_path.stat().st_size >= 1000),
        "allow_256_port_expansion": False,
        "next_gate": "Validate complex-field reconstruction on 50-100 existing HFSS cases before 256-port export.",
    }
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "run":
        if not (args.out_dir / "run_eep_16port.ps1").exists():
            prepare(args)
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(args.out_dir / "run_eep_16port.ps1")],
            cwd=str(ROOT),
            check=False,
        )
        result = analyze(args)
        result["runner_exit_code"] = int(completed.returncode)
        (args.out_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = analyze(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
