"""HFSS full-wave validation for sparse LCMV/ZF mask and weights.

This script prepares a small, stratified set of optimized teacher samples for
HFSS re-export and then analyzes the exported full-wave patterns.

Two case types are generated for every selected sample:

* combined: sum of all per-task weights, matching the original dataset's
  multi-beam HFSS export convention. This is used for PSLL validation.
* task_j: one task weight column at a time. These per-task full-wave patterns
  are used to build a target-gain matrix and compute beam isolation after HFSS
  embedded-pattern/mutual-coupling effects.

The script reuses the solved HFSS setup through Solutions.EditSources; it does
not rebuild geometry or remesh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_TEACHER_DIR = DEFAULT_DATASET_DIR / "optimized_teachers" / "iso_lcmv_zf_psll_test_v1"
DEFAULT_OUT_ROOT = DEFAULT_DATASET_DIR / "hfss_fullwave_validations"
DEFAULT_PROJECT_PATH = ROOT / "models" / "hfss" / "ura16_quick_10ghz_fullarray_run.aedt"
DEFAULT_ANSYS_EXE = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "URA16_Quick_10GHz"
SPHERE_NAME = "InfiniteSphere_Theta0_90_Phi0_360"
SOLUTION_NAME = "Setup_10GHz : LastAdaptive"
KMAX = 6
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="iso_lcmv_zf_fullwave_task_mvp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k-values", default="2,4,6")
    parser.add_argument("--active-ratios", default="0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--samples-per-cell", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means no additional cap.")
    parser.add_argument("--mode", choices=("prepare", "analyze", "both"), default="both")
    parser.add_argument("--project-path", type=Path, default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS_EXE)
    parser.add_argument("--design-name", default=DESIGN_NAME)
    parser.add_argument("--solution-name", default=SOLUTION_NAME)
    parser.add_argument("--sphere-name", default=SPHERE_NAME)
    parser.add_argument("--export-batch-size", type=int, default=0, help="0 means export all queued cases.")
    parser.add_argument("--target-local-radius-deg", type=float, default=5.0)
    parser.add_argument("--sidelobe-exclusion-deg", type=float, default=8.0)
    parser.add_argument("--isolation-thresholds-db", default="20,25,30")
    parser.add_argument("--gate-psll-max-db", type=float, default=0.0)
    parser.add_argument("--gate-nearest-iso-min-db", type=float, default=25.0)
    parser.add_argument("--gate-local-iso-min-db", type=float, default=15.0)
    parser.add_argument("--gate-local-iso-strict-db", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def load_split_indices(dataset_dir: Path, split_name: str, num_samples: int) -> np.ndarray:
    if split_name == "all":
        return np.arange(num_samples, dtype=np.int64)
    manifest_path = dataset_dir / "training_split_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    splits = payload["splits"]
    key = split_name if split_name in splits else f"{split_name}_id"
    if key not in splits:
        raise KeyError(f"Split {split_name!r} not found in {manifest_path}")
    return np.asarray(splits[key], dtype=np.int64)


def select_samples(
    *,
    arrays: dict[str, np.ndarray],
    dataset_dir: Path,
    split_name: str,
    k_values: list[int],
    active_ratios: list[float],
    samples_per_cell: int,
    max_samples: int,
) -> list[int]:
    split_indices = load_split_indices(dataset_dir, split_name, arrays["sample_ids"].shape[0])
    split_set = set(int(i) for i in split_indices)
    selected_teacher = set(int(i) for i in arrays.get("selected_indices", np.asarray([], dtype=np.int64)))
    out: list[int] = []
    for k in k_values:
        for ratio in active_ratios:
            candidates = [
                int(i)
                for i in split_indices
                if int(i) in split_set
                and int(arrays["k_values"][i]) == int(k)
                and abs(float(arrays["active_ratios_requested"][i]) - float(ratio)) < 1.0e-6
            ]
            if selected_teacher:
                teacher_candidates = [idx for idx in candidates if idx in selected_teacher]
                if not teacher_candidates:
                    continue
                candidates = teacher_candidates
            candidates.sort()
            take = candidates if samples_per_cell <= 0 else candidates[:samples_per_cell]
            out.extend(take)
    seen: set[int] = set()
    unique = []
    for idx in out:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    if max_samples > 0:
        unique = unique[:max_samples]
    return unique


def phase_deg(value: complex) -> float:
    if abs(value) <= 0.0:
        return 0.0
    return math.degrees(math.atan2(float(value.imag), float(value.real)))


def normalize_hfss_excitation(weights: np.ndarray) -> np.ndarray:
    excitation = np.conjugate(weights.astype(np.complex128))
    l2_norm = float(np.linalg.norm(excitation))
    if l2_norm <= 0.0:
        return excitation
    # In Driven Modal, sum(abs(a)**2) is total incident power in watts.
    return excitation / l2_norm


def case_id_for(sample_id: str, case_kind: str, task_index: int | None) -> str:
    if case_kind == "combined":
        return f"{sample_id}_combined"
    if task_index is None:
        raise ValueError("task_index required for task case")
    return f"{sample_id}_task{task_index}"


def write_sources_csv(
    *,
    path: Path,
    port_names: np.ndarray,
    mask: np.ndarray,
    excitation: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PortName", "IncidentPowerW", "PhaseDeg", "Active", "ElementIndex"])
        for idx, port_name in enumerate(port_names):
            value = complex(excitation[idx])
            coefficient_magnitude = abs(value)
            incident_power_w = coefficient_magnitude**2
            active = int(bool(mask[idx]) and coefficient_magnitude > 1.0e-10)
            writer.writerow([str(port_name), f"{incident_power_w:.12e}", f"{phase_deg(value):.6f}", active, idx])


def write_case_weights_csv(
    *,
    path: Path,
    port_names: np.ndarray,
    element_ixiy: np.ndarray,
    mask: np.ndarray,
    task_weights: np.ndarray,
    excitation: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "element_index",
        "port_name",
        "ix",
        "iy",
        "active",
        "source_convention",
        "hfss_field_coefficient_magnitude",
        "hfss_incident_power_w",
        "hfss_magnitude_v",
        "hfss_phase_deg",
        "hfss_real",
        "hfss_imag",
        "weight_real",
        "weight_imag",
        "weight_magnitude",
        "weight_phase_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, port_name in enumerate(port_names):
            value = complex(excitation[idx])
            weight = complex(task_weights[idx])
            writer.writerow(
                {
                    "element_index": idx,
                    "port_name": str(port_name),
                    "ix": int(element_ixiy[idx, 0]),
                    "iy": int(element_ixiy[idx, 1]),
                    "active": int(bool(mask[idx])),
                    "source_convention": "driven_modal_incident_power_watt",
                    "hfss_field_coefficient_magnitude": f"{abs(value):.9f}",
                    "hfss_incident_power_w": f"{abs(value) ** 2:.12e}",
                    # Retained for readers of legacy files; this is a field coefficient, not volts.
                    "hfss_magnitude_v": f"{abs(value):.9f}",
                    "hfss_phase_deg": f"{phase_deg(value):.6f}",
                    "hfss_real": f"{value.real:.12e}",
                    "hfss_imag": f"{value.imag:.12e}",
                    "weight_real": f"{weight.real:.12e}",
                    "weight_imag": f"{weight.imag:.12e}",
                    "weight_magnitude": f"{abs(weight):.12e}",
                    "weight_phase_deg": f"{phase_deg(weight):.6f}",
                }
            )


def prepare_cases(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    arrays = load_npz(args.teacher_dir / "dataset_arrays.npz")
    if "selected_indices" not in arrays and (args.teacher_dir / "iso_lcmv_teacher_arrays.npz").exists():
        aux = load_npz(args.teacher_dir / "iso_lcmv_teacher_arrays.npz")
        arrays["selected_indices"] = aux.get("selected_indices", np.asarray([], dtype=np.int64))
    k_values = parse_int_list(args.k_values)
    active_ratios = parse_float_list(args.active_ratios)
    selected = select_samples(
        arrays=arrays,
        dataset_dir=args.dataset_dir,
        split_name=args.split,
        k_values=k_values,
        active_ratios=active_ratios,
        samples_per_cell=int(args.samples_per_cell),
        max_samples=int(args.max_samples),
    )
    if not selected:
        raise RuntimeError("No samples selected for HFSS full-wave validation.")

    sample_root = out_dir / "samples"
    queue_path = out_dir / "hfss_task_export_queue.csv"
    manifest_path = out_dir / "hfss_task_case_manifest.csv"
    summary_path = out_dir / "prepare_summary.json"
    sample_root.mkdir(parents=True, exist_ok=True)

    port_names = arrays.get("port_names", np.asarray([f"P{i:03d}" for i in range(256)]))
    element_ixiy = arrays["element_ixiy"]
    masks = arrays["masks"].astype(bool)
    weights = arrays["task_weights_real_imag"][..., 0] + 1j * arrays["task_weights_real_imag"][..., 1]

    manifest_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, str]] = []

    for sample_index in selected:
        sample_id = str(arrays["sample_ids"][sample_index])
        k = int(arrays["k_values"][sample_index])
        valid_indices = np.flatnonzero(arrays["task_valid"][sample_index].astype(bool))
        active_ratio = float(arrays["active_ratios_requested"][sample_index])
        active_count = int(arrays["num_active"][sample_index])
        mask = masks[sample_index]
        sample_dir = sample_root / sample_id
        targets = arrays["targets_deg"][sample_index, valid_indices]
        sample_dir.mkdir(parents=True, exist_ok=True)

        with (sample_dir / "sample.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sample_index": int(sample_index),
                    "sample_id": sample_id,
                    "k": k,
                    "active_ratio": active_ratio,
                    "active_count": active_count,
                    "targets": [
                        {"task_index": int(task), "theta_deg": float(theta), "phi_deg": float(phi)}
                        for task, (theta, phi) in zip(valid_indices, targets)
                    ],
                },
                handle,
                indent=2,
            )

        sample_weights = weights[sample_index]
        combined_weights = sample_weights[:, valid_indices].sum(axis=1)
        cases: list[tuple[str, int | None, np.ndarray]] = [("combined", None, combined_weights)]
        for task_index in valid_indices:
            cases.append(("task", int(task_index), sample_weights[:, int(task_index)]))

        for case_kind, task_index, case_weights in cases:
            case_id = case_id_for(sample_id, case_kind, task_index)
            case_dir = sample_dir / ("combined" if case_kind == "combined" else f"task_{task_index}")
            sources_csv = case_dir / "hfss_sources.csv"
            weights_csv = case_dir / "case_weights.csv"
            excitation = normalize_hfss_excitation(case_weights)
            write_sources_csv(path=sources_csv, port_names=port_names, mask=mask, excitation=excitation)
            write_case_weights_csv(
                path=weights_csv,
                port_names=port_names,
                element_ixiy=element_ixiy,
                mask=mask,
                task_weights=case_weights,
                excitation=excitation,
            )
            row = {
                "case_id": case_id,
                "sample_index": int(sample_index),
                "sample_id": sample_id,
                "case_kind": case_kind,
                "task_index": "" if task_index is None else int(task_index),
                "k": k,
                "active_ratio": f"{active_ratio:.6f}",
                "active_count": active_count,
                "targets_json": json.dumps([[float(t), float(p)] for t, p in targets], separators=(",", ":")),
                "sources_csv": str(sources_csv),
                "weights_csv": str(weights_csv),
                "output_dir": str(case_dir),
            }
            manifest_rows.append(row)
            queue_rows.append({"CaseID": case_id, "SourcesCSV": str(sources_csv), "OutputDir": str(case_dir)})

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case_id",
            "sample_index",
            "sample_id",
            "case_kind",
            "task_index",
            "k",
            "active_ratio",
            "active_count",
            "targets_json",
            "sources_csv",
            "weights_csv",
            "output_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["CaseID", "SourcesCSV", "OutputDir"])
        writer.writeheader()
        writer.writerows(queue_rows)

    write_export_vbs(
        out_dir / "export_hfss_task_patterns.vbs",
        project_path=args.project_path,
        queue_path=queue_path,
        batch_size=int(args.export_batch_size),
        design_name=str(args.design_name),
        solution_name=str(args.solution_name),
        sphere_name=str(args.sphere_name),
    )
    write_export_ps1(
        out_dir / "run_hfss_task_export.ps1",
        ansys_exe=args.ansys_exe,
        script_path=out_dir / "export_hfss_task_patterns.vbs",
        queue_path=queue_path,
        log_path=out_dir / "export_hfss_task_patterns.log",
    )

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teacher_dir": str(args.teacher_dir),
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(out_dir),
        "split": args.split,
        "k_values": k_values,
        "active_ratios": active_ratios,
        "samples_per_cell": int(args.samples_per_cell),
        "selected_sample_count": len(selected),
        "queued_case_count": len(queue_rows),
        "combined_case_count": len(selected),
        "task_case_count": len(queue_rows) - len(selected),
        "source_convention": "driven_modal_incident_power_watt",
        "source_mapping": "incident_power_w=abs(complex_field_coefficient)**2",
        "power_normalization": "sum(incident_power_w)=1W_per_case",
        "project_path": str(args.project_path),
        "design_name": str(args.design_name),
        "solution_name": str(args.solution_name),
        "sphere_name": str(args.sphere_name),
        "manifest": str(manifest_path),
        "queue": str(queue_path),
        "run_ps1": str(out_dir / "run_hfss_task_export.ps1"),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def write_export_vbs(
    path: Path,
    *,
    project_path: Path,
    queue_path: Path,
    batch_size: int,
    design_name: str,
    solution_name: str,
    sphere_name: str,
) -> None:
    progress_path = path.parent / "hfss_task_export_progress.csv"
    summary_path = path.parent / "hfss_task_export_batch_summary.csv"
    batch = int(batch_size)
    path.write_text(
        f'''Option Explicit

Dim projectPath, designName, queuePath, sphereName, solutionName
Dim progressPath, summaryPath
Dim BATCH_SIZE, KEEP_REPORTS, RESUME_ENABLED, FORCE_EXPORT
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol, oReport, fso
Dim queueFile, header, line, parts, caseId, sourcesCsv, outputDir
Dim exportedCount, skippedCount, scannedCount

projectPath = "{str(project_path)}"
designName = "{design_name}"
queuePath = "{str(queue_path)}"
sphereName = "{sphere_name}"
solutionName = "{solution_name}"
BATCH_SIZE = {batch}
RESUME_ENABLED = True
FORCE_EXPORT = False
KEEP_REPORTS = False

Set fso = CreateObject("Scripting.FileSystemObject")
progressPath = "{str(progress_path)}"
summaryPath = "{str(summary_path)}"
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject projectPath
Set oProject = oDesktop.SetActiveProject("{project_path.stem}")
Set oDesign = oProject.SetActiveDesign(designName)
Set oSol = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")

Set queueFile = fso.OpenTextFile(queuePath, 1)
If Not queueFile.AtEndOfStream Then header = queueFile.ReadLine
exportedCount = 0
skippedCount = 0
scannedCount = 0

Do Until queueFile.AtEndOfStream
    line = Trim(queueFile.ReadLine)
    If Len(line) > 0 Then
        scannedCount = scannedCount + 1
        parts = Split(line, ",")
        caseId = parts(0)
        sourcesCsv = parts(1)
        outputDir = parts(2)
        EnsureFolder outputDir
        If RESUME_ENABLED And Not FORCE_EXPORT And SampleComplete(outputDir) Then
            skippedCount = skippedCount + 1
        Else
            ApplySourcesFromCsv oSol, sourcesCsv
            ExportCaseReports oReport, caseId, outputDir
            If Not SampleComplete(outputDir) Then
                AppendProgress caseId, "failed_missing_outputs", outputDir
                Err.Raise vbObjectError + 1001, "ExportCaseReports", "Missing HFSS export CSV files for " & caseId
            End If
            AppendProgress caseId, "exported", outputDir
            exportedCount = exportedCount + 1
            If BATCH_SIZE > 0 And exportedCount >= BATCH_SIZE Then Exit Do
        End If
    End If
Loop

queueFile.Close
AppendSummary scannedCount, skippedCount, exportedCount, BATCH_SIZE
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ApplySourcesFromCsv(solModule, csvPath)
    Dim srcFile, srcHeader, srcLine, srcParts
    Dim magMap, phaseMap, sources, editArgs()
    Dim i, name, mag, phase

    Set magMap = CreateObject("Scripting.Dictionary")
    Set phaseMap = CreateObject("Scripting.Dictionary")

    Set srcFile = fso.OpenTextFile(csvPath, 1)
    If Not srcFile.AtEndOfStream Then srcHeader = srcFile.ReadLine
    Do Until srcFile.AtEndOfStream
        srcLine = Trim(srcFile.ReadLine)
        If Len(srcLine) > 0 Then
            srcParts = Split(srcLine, ",")
            name = srcParts(0)
            magMap(name) = srcParts(1)
            phaseMap(name) = srcParts(2)
        End If
    Loop
    srcFile.Close

    sources = solModule.GetAllSources()
    ReDim editArgs(UBound(sources) + 1)
    editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)

    For i = LBound(sources) To UBound(sources)
        name = CStr(sources(i))
        If magMap.Exists(name) Then
            mag = CStr(magMap(name)) & "W"
            phase = CStr(phaseMap(name)) & "deg"
        Else
            mag = "0W"
            phase = "0deg"
        End If
        editArgs(i + 1) = Array("Name:=", name, "Magnitude:=", mag, "Phase:=", phase)
    Next

    solModule.EditSources editArgs
End Sub

Sub ExportCaseReports(reportModule, caseId, outputDir)
    Dim rptThetaPhi, rptPhi0, rptPhi90

    rptThetaPhi = "FW_" & caseId & "_GainTotal_ThetaPhi"
    rptPhi0 = "FW_" & caseId & "_GainTotal_Phi0"
    rptPhi90 = "FW_" & caseId & "_GainTotal_Phi90"

    DeleteIfExists reportModule, rptThetaPhi
    DeleteIfExists reportModule, rptPhi0
    DeleteIfExists reportModule, rptPhi90

    reportModule.CreateReport rptThetaPhi, "Far Fields", "Rectangular Contour Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("All")), _
        Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", Array("dB(GainTotal)")), _
        Array()
    reportModule.ExportToFile rptThetaPhi, outputDir & "\\hfss_gain_total_theta_phi.csv"

    reportModule.CreateReport rptPhi0, "Far Fields", "Rectangular Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("0deg")), _
        Array("X Component:=", "Theta", "Y Component:=", Array("dB(GainTotal)")), _
        Array()
    reportModule.ExportToFile rptPhi0, outputDir & "\\hfss_gain_total_phi0.csv"

    reportModule.CreateReport rptPhi90, "Far Fields", "Rectangular Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("90deg")), _
        Array("X Component:=", "Theta", "Y Component:=", Array("dB(GainTotal)")), _
        Array()
    reportModule.ExportToFile rptPhi90, outputDir & "\\hfss_gain_total_phi90.csv"

    If Not KEEP_REPORTS Then
        DeleteIfExists reportModule, rptThetaPhi
        DeleteIfExists reportModule, rptPhi0
        DeleteIfExists reportModule, rptPhi90
    End If
End Sub

Sub EnsureFolder(folderPath)
    If Not fso.FolderExists(folderPath) Then
        fso.CreateFolder(folderPath)
    End If
End Sub

Function SampleComplete(outputDir)
    SampleComplete = OutputFileOk(outputDir & "\\hfss_gain_total_theta_phi.csv", 1000) _
        And OutputFileOk(outputDir & "\\hfss_gain_total_phi0.csv", 100) _
        And OutputFileOk(outputDir & "\\hfss_gain_total_phi90.csv", 100)
End Function

Function OutputFileOk(filePath, minBytes)
    OutputFileOk = False
    If fso.FileExists(filePath) Then
        If fso.GetFile(filePath).Size >= minBytes Then
            OutputFileOk = True
        End If
    End If
End Function

Sub AppendProgress(caseId, status, outputDir)
    Dim progressFile
    Set progressFile = OpenCsvAppend(progressPath, "timestamp,case_id,status,output_dir")
    progressFile.WriteLine CsvCell(TimestampNow()) & "," & CsvCell(caseId) & "," & CsvCell(status) & "," & CsvCell(outputDir)
    progressFile.Close
End Sub

Sub AppendSummary(scannedCount, skippedCount, exportedCount, batchSize)
    Dim summaryFile
    Set summaryFile = OpenCsvAppend(summaryPath, "timestamp,scanned_rows,skipped_existing,exported_new,batch_size,resume_enabled,force_export")
    summaryFile.WriteLine CsvCell(TimestampNow()) & "," & scannedCount & "," & skippedCount & "," & exportedCount & "," & batchSize & "," & RESUME_ENABLED & "," & FORCE_EXPORT
    summaryFile.Close
End Sub

Function OpenCsvAppend(filePath, headerLine)
    Dim shouldWriteHeader, csvFile
    shouldWriteHeader = True
    If fso.FileExists(filePath) Then
        If fso.GetFile(filePath).Size > 0 Then shouldWriteHeader = False
    End If
    Set csvFile = fso.OpenTextFile(filePath, 8, True)
    If shouldWriteHeader Then csvFile.WriteLine headerLine
    Set OpenCsvAppend = csvFile
End Function

Function CsvCell(value)
    CsvCell = """" & Replace(CStr(value), """", """""") & """"
End Function

Function TimestampNow()
    Dim d
    d = Now
    TimestampNow = Year(d) & "-" & Pad2(Month(d)) & "-" & Pad2(Day(d)) & " " & Pad2(Hour(d)) & ":" & Pad2(Minute(d)) & ":" & Pad2(Second(d))
End Function

Function Pad2(value)
    Pad2 = Right("0" & CStr(value), 2)
End Function

Sub DeleteIfExists(reportModule, name)
    On Error Resume Next
    reportModule.DeleteReports Array(name)
    On Error GoTo 0
End Sub
''',
        encoding="utf-8",
    )


def write_export_ps1(path: Path, *, ansys_exe: Path, script_path: Path, queue_path: Path, log_path: Path) -> None:
    path.write_text(
        f'''$ErrorActionPreference = "Stop"

$ansys = "{str(ansys_exe)}"
$script = "{str(script_path)}"
$log = "{str(log_path)}"
$queue = "{str(queue_path)}"

function Get-HfssTaskExportStatus {{
    param([string]$QueuePath)

    $rows = Import-Csv -LiteralPath $QueuePath
    $done = 0
    foreach ($row in $rows) {{
        $thetaPhi = Join-Path $row.OutputDir "hfss_gain_total_theta_phi.csv"
        $phi0 = Join-Path $row.OutputDir "hfss_gain_total_phi0.csv"
        $phi90 = Join-Path $row.OutputDir "hfss_gain_total_phi90.csv"
        if ((Test-Path -LiteralPath $thetaPhi) -and
            (Test-Path -LiteralPath $phi0) -and
            (Test-Path -LiteralPath $phi90) -and
            ((Get-Item -LiteralPath $thetaPhi).Length -ge 1000) -and
            ((Get-Item -LiteralPath $phi0).Length -ge 100) -and
            ((Get-Item -LiteralPath $phi90).Length -ge 100)) {{
            $done += 1
        }}
    }}

    [pscustomobject]@{{
        Done = $done
        Total = $rows.Count
        Remaining = $rows.Count - $done
    }}
}}

if (-not (Test-Path -LiteralPath $ansys)) {{
    throw "ansysedt.exe not found: $ansys"
}}
if (-not (Test-Path -LiteralPath $script)) {{
    throw "HFSS export script not found: $script"
}}
if (-not (Test-Path -LiteralPath $queue)) {{
    throw "HFSS export queue not found: $queue"
}}

Remove-Item -LiteralPath $log -Force -ErrorAction SilentlyContinue
$before = Get-HfssTaskExportStatus -QueuePath $queue
Write-Host "HFSS task export status before run: $($before.Done)/$($before.Total) completed, $($before.Remaining) remaining."

$proc = Start-Process -FilePath $ansys -ArgumentList @("-RunScriptAndExit", $script) -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddHours(12)

while (-not $proc.HasExited) {{
    if ((Get-Date) -gt $deadline) {{
        throw "Timed out waiting for HFSS task export."
    }}
    Start-Sleep -Seconds 10
    $proc.Refresh()
}}

if ($proc.ExitCode -ne 0) {{
    throw "HFSS task export exited with code $($proc.ExitCode)."
}}

$after = Get-HfssTaskExportStatus -QueuePath $queue
Write-Host "HFSS task pattern export completed."
Write-Host "HFSS task export status after run: $($after.Done)/$($after.Total) completed, $($after.Remaining) remaining."
''',
        encoding="utf-8",
    )


def unit_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return np.asarray(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        dtype=np.float64,
    )


def pattern_grid_dirs(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    return np.stack(
        [
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad),
        ],
        axis=-1,
    )


def read_hfss_pattern(path: Path) -> dict[str, np.ndarray]:
    theta: list[float] = []
    phi: list[float] = []
    gain: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            theta.append(float(row["Theta[deg]"]))
            phi.append(float(row["Phi[deg]"]))
            gain.append(float(row["dB(GainTotal)"]))
    theta_arr = np.asarray(theta, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    gain_arr = np.asarray(gain, dtype=np.float64)
    return {"theta": theta_arr, "phi": phi_arr, "gain_db": gain_arr, "dirs": pattern_grid_dirs(theta_arr, phi_arr)}


def angular_dist_deg(dirs: np.ndarray, target_dir: np.ndarray) -> np.ndarray:
    dots = np.clip(dirs @ target_dir, -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def nearest_gain_db(pattern: dict[str, np.ndarray], theta_deg: float, phi_deg: float) -> tuple[float, float, float, float]:
    target = unit_vector(theta_deg, phi_deg)
    dist = angular_dist_deg(pattern["dirs"], target)
    idx = int(np.argmin(dist))
    return (
        float(pattern["gain_db"][idx]),
        float(pattern["theta"][idx]),
        float(pattern["phi"][idx]),
        float(dist[idx]),
    )


def local_peak_db(pattern: dict[str, np.ndarray], theta_deg: float, phi_deg: float, radius_deg: float) -> tuple[float, float]:
    target = unit_vector(theta_deg, phi_deg)
    dist = angular_dist_deg(pattern["dirs"], target)
    mask = dist <= float(radius_deg)
    if not np.any(mask):
        idx = int(np.argmin(dist))
        return float(pattern["gain_db"][idx]), float(dist[idx])
    local_indices = np.flatnonzero(mask)
    best_local = int(local_indices[np.argmax(pattern["gain_db"][mask])])
    return float(pattern["gain_db"][best_local]), float(dist[best_local])


def combined_metrics(
    pattern: dict[str, np.ndarray],
    targets: list[list[float]],
    *,
    target_radius_deg: float,
    sidelobe_exclusion_deg: float,
) -> dict[str, Any]:
    target_peaks = [
        local_peak_db(pattern, float(theta), float(phi), target_radius_deg)[0]
        for theta, phi in targets
    ]
    target_dirs = np.stack([unit_vector(float(theta), float(phi)) for theta, phi in targets], axis=0)
    dists = np.rad2deg(np.arccos(np.clip(pattern["dirs"] @ target_dirs.T, -1.0, 1.0)))
    side_mask = dists.min(axis=1) > float(sidelobe_exclusion_deg)
    worst_side = float(np.max(pattern["gain_db"][side_mask])) if np.any(side_mask) else float("nan")
    peaks = np.asarray(target_peaks, dtype=np.float64)
    return {
        "combined_target_peak_min_db": float(np.min(peaks)),
        "combined_target_peak_mean_db": float(np.mean(peaks)),
        "combined_target_peak_max_db": float(np.max(peaks)),
        "combined_target_spread_db": float(np.max(peaks) - np.min(peaks)),
        "combined_worst_sidelobe_abs_db": worst_side,
        "combined_psll_to_weakest_peak_db": worst_side - float(np.min(peaks)),
        "combined_target_peaks_json": json.dumps([float(v) for v in peaks], separators=(",", ":")),
    }


def isolation_metrics(
    task_patterns: dict[int, dict[str, np.ndarray]],
    targets: list[list[float]],
    *,
    target_radius_deg: float,
) -> dict[str, Any]:
    k = len(targets)
    nearest = np.full((k, k), np.nan, dtype=np.float64)
    local = np.full((k, k), np.nan, dtype=np.float64)
    nearest_distance = np.full((k, k), np.nan, dtype=np.float64)
    task_indices = sorted(task_patterns)
    if len(task_indices) != k:
        return {
            "isolation_complete": 0,
            "isolation_worst_nearest_db": float("nan"),
            "isolation_mean_nearest_db": float("nan"),
            "isolation_worst_local_db": float("nan"),
            "isolation_mean_local_db": float("nan"),
            "isolation_matrix_nearest_json": "[]",
            "isolation_matrix_local_json": "[]",
            "isolation_per_task_nearest_json": "[]",
            "isolation_per_task_local_json": "[]",
            "nearest_distance_matrix_json": "[]",
        }

    for col, task_index in enumerate(task_indices):
        pattern = task_patterns[task_index]
        for row, (theta, phi) in enumerate(targets):
            gain, _, _, dist = nearest_gain_db(pattern, float(theta), float(phi))
            nearest[row, col] = gain
            nearest_distance[row, col] = dist
            local[row, col] = local_peak_db(pattern, float(theta), float(phi), target_radius_deg)[0]

    if k == 1:
        no_intertask_leakage_db = 300.0
        return {
            "isolation_complete": 1,
            "isolation_worst_nearest_db": no_intertask_leakage_db,
            "isolation_mean_nearest_db": no_intertask_leakage_db,
            "isolation_worst_local_db": no_intertask_leakage_db,
            "isolation_mean_local_db": no_intertask_leakage_db,
            "isolation_matrix_nearest_json": json.dumps(nearest.tolist(), separators=(",", ":")),
            "isolation_matrix_local_json": json.dumps(local.tolist(), separators=(",", ":")),
            "isolation_per_task_nearest_json": json.dumps([no_intertask_leakage_db]),
            "isolation_per_task_local_json": json.dumps([no_intertask_leakage_db]),
            "nearest_distance_matrix_json": json.dumps(
                nearest_distance.tolist(), separators=(",", ":")
            ),
        }

    nearest_iso = []
    local_iso = []
    for col in range(k):
        nearest_leakage = float(np.max(np.delete(nearest[:, col], col)))
        local_leakage = float(np.max(np.delete(local[:, col], col)))
        nearest_iso.append(float(nearest[col, col] - nearest_leakage))
        local_iso.append(float(local[col, col] - local_leakage))

    return {
        "isolation_complete": 1,
        "isolation_worst_nearest_db": float(np.min(nearest_iso)),
        "isolation_mean_nearest_db": float(np.mean(nearest_iso)),
        "isolation_worst_local_db": float(np.min(local_iso)),
        "isolation_mean_local_db": float(np.mean(local_iso)),
        "isolation_matrix_nearest_json": json.dumps(nearest.tolist(), separators=(",", ":")),
        "isolation_matrix_local_json": json.dumps(local.tolist(), separators=(",", ":")),
        "isolation_per_task_nearest_json": json.dumps(nearest_iso, separators=(",", ":")),
        "isolation_per_task_local_json": json.dumps(local_iso, separators=(",", ":")),
        "nearest_distance_matrix_json": json.dumps(nearest_distance.tolist(), separators=(",", ":")),
    }


def parse_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def finite_mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return float(np.mean(values)) if values.size else float("nan")


def finite_percentile(rows: Iterable[dict[str, Any]], key: str, q: float) -> float:
    values = finite_values(rows, key)
    return float(np.percentile(values, q)) if values.size else float("nan")


def finite_min(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return float(np.min(values)) if values.size else float("nan")


def pass_rate(rows: Iterable[dict[str, Any]], key: str, threshold: float) -> float:
    values = finite_values(rows, key)
    return float(np.mean(values >= float(threshold))) if values.size else float("nan")


def row_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def write_fullwave_gate_outputs(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    psll_max_db: float,
    nearest_iso_min_db: float,
    local_iso_min_db: float,
    local_iso_strict_db: float,
) -> dict[str, Any]:
    gate_rows: list[dict[str, Any]] = []
    for row in rows:
        psll = row_float(row, "combined_psll_to_weakest_peak_db")
        nearest_iso = row_float(row, "isolation_worst_nearest_db")
        local_iso = row_float(row, "isolation_worst_local_db")
        combined_complete = int(row.get("combined_complete", 0)) == 1
        isolation_complete = int(row.get("isolation_complete", 0)) == 1

        fail_incomplete = not (combined_complete and isolation_complete)
        fail_psll = (not np.isfinite(psll)) or psll > float(psll_max_db)
        fail_nearest = (not np.isfinite(nearest_iso)) or nearest_iso < float(nearest_iso_min_db)
        fail_local = (not np.isfinite(local_iso)) or local_iso < float(local_iso_min_db)
        fail_local_strict = (not np.isfinite(local_iso)) or local_iso < float(local_iso_strict_db)

        reasons: list[str] = []
        if fail_incomplete:
            reasons.append("missing_fullwave_exports")
        if fail_psll:
            reasons.append(f"psll>{psll_max_db:g}dB")
        if fail_nearest:
            reasons.append(f"nearest_iso<{nearest_iso_min_db:g}dB")
        if fail_local:
            reasons.append(f"local_iso<{local_iso_min_db:g}dB")
        strict_reasons = list(reasons)
        if fail_local_strict and not fail_local:
            strict_reasons.append(f"local_iso<{local_iso_strict_db:g}dB")

        primary_pass = not (fail_incomplete or fail_psll or fail_nearest or fail_local)
        strict_pass = primary_pass and not fail_local_strict
        gate_rows.append(
            {
                "sample_index": int(row["sample_index"]),
                "sample_id": str(row["sample_id"]),
                "k": int(row["k"]),
                "active_ratio": f"{float(row['active_ratio']):.1f}",
                "active_count": int(row["active_count"]),
                "fullwave_gate_pass": int(primary_pass),
                "fullwave_gate_pass_strict_local": int(strict_pass),
                "fail_reasons": ";".join(reasons),
                "strict_fail_reasons": ";".join(strict_reasons),
                "combined_psll_to_weakest_peak_db": psll,
                "isolation_worst_nearest_db": nearest_iso,
                "isolation_worst_local_db": local_iso,
                "combined_target_spread_db": row_float(row, "combined_target_spread_db"),
                "combined_complete": int(row.get("combined_complete", 0)),
                "isolation_complete": int(row.get("isolation_complete", 0)),
            }
        )

    fields = [
        "sample_index",
        "sample_id",
        "k",
        "active_ratio",
        "active_count",
        "fullwave_gate_pass",
        "fullwave_gate_pass_strict_local",
        "fail_reasons",
        "strict_fail_reasons",
        "combined_psll_to_weakest_peak_db",
        "isolation_worst_nearest_db",
        "isolation_worst_local_db",
        "combined_target_spread_db",
        "combined_complete",
        "isolation_complete",
    ]
    gate_path = out_dir / "hfss_fullwave_gate_results.csv"
    failure_path = out_dir / "hfss_fullwave_gate_failures.csv"
    strict_failure_path = out_dir / "hfss_fullwave_gate_strict_failures.csv"
    reopt_path = out_dir / "hfss_fullwave_reoptimize_indices.txt"
    strict_reopt_path = out_dir / "hfss_fullwave_reoptimize_indices_strict.txt"
    pass_path = out_dir / "hfss_fullwave_gate_pass_indices.txt"

    with gate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(gate_rows)

    primary_failures = [row for row in gate_rows if int(row["fullwave_gate_pass"]) == 0]
    strict_failures = [row for row in gate_rows if int(row["fullwave_gate_pass_strict_local"]) == 0]
    primary_passes = [row for row in gate_rows if int(row["fullwave_gate_pass"]) == 1]

    with failure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(primary_failures)
    with strict_failure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(strict_failures)
    reopt_path.write_text("\n".join(str(row["sample_index"]) for row in primary_failures) + ("\n" if primary_failures else ""), encoding="utf-8")
    strict_reopt_path.write_text("\n".join(str(row["sample_index"]) for row in strict_failures) + ("\n" if strict_failures else ""), encoding="utf-8")
    pass_path.write_text("\n".join(str(row["sample_index"]) for row in primary_passes) + ("\n" if primary_passes else ""), encoding="utf-8")

    return {
        "gate_results": str(gate_path),
        "gate_failures": str(failure_path),
        "gate_strict_failures": str(strict_failure_path),
        "reoptimize_indices": str(reopt_path),
        "reoptimize_indices_strict": str(strict_reopt_path),
        "gate_pass_indices": str(pass_path),
        "gate_pass_count": len(primary_passes),
        "gate_fail_count": len(primary_failures),
        "gate_strict_fail_count": len(strict_failures),
        "gate_thresholds": {
            "psll_max_db": float(psll_max_db),
            "nearest_iso_min_db": float(nearest_iso_min_db),
            "local_iso_min_db": float(local_iso_min_db),
            "local_iso_strict_db": float(local_iso_strict_db),
        },
    }


def analyze_outputs(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    manifest = parse_manifest(out_dir / "hfss_task_case_manifest.csv")
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for sample_id, case_rows in sorted(by_sample.items()):
        combined_row = next((row for row in case_rows if row["case_kind"] == "combined"), None)
        if combined_row is None:
            continue
        targets = json.loads(combined_row["targets_json"])
        combined_path = Path(combined_row["output_dir"]) / "hfss_gain_total_theta_phi.csv"
        base = {
            "sample_index": int(combined_row["sample_index"]),
            "sample_id": sample_id,
            "k": int(combined_row["k"]),
            "active_ratio": float(combined_row["active_ratio"]),
            "active_count": int(combined_row["active_count"]),
            "targets_json": combined_row["targets_json"],
            "combined_output_dir": combined_row["output_dir"],
        }
        if combined_path.exists() and combined_path.stat().st_size >= 1000:
            pattern = read_hfss_pattern(combined_path)
            base.update(
                combined_metrics(
                    pattern,
                    targets,
                    target_radius_deg=float(args.target_local_radius_deg),
                    sidelobe_exclusion_deg=float(args.sidelobe_exclusion_deg),
                )
            )
            base["combined_complete"] = 1
        else:
            base.update(
                {
                    "combined_target_peak_min_db": float("nan"),
                    "combined_target_peak_mean_db": float("nan"),
                    "combined_target_peak_max_db": float("nan"),
                    "combined_target_spread_db": float("nan"),
                    "combined_worst_sidelobe_abs_db": float("nan"),
                    "combined_psll_to_weakest_peak_db": float("nan"),
                    "combined_target_peaks_json": "[]",
                    "combined_complete": 0,
                }
            )

        task_patterns: dict[int, dict[str, np.ndarray]] = {}
        for row in case_rows:
            if row["case_kind"] != "task":
                continue
            task_index = int(row["task_index"])
            path = Path(row["output_dir"]) / "hfss_gain_total_theta_phi.csv"
            if path.exists() and path.stat().st_size >= 1000:
                task_patterns[task_index] = read_hfss_pattern(path)
        base.update(isolation_metrics(task_patterns, targets, target_radius_deg=float(args.target_local_radius_deg)))
        rows.append(base)

    metrics_path = out_dir / "hfss_task_fullwave_metrics.csv"
    fieldnames = [
        "sample_index",
        "sample_id",
        "k",
        "active_ratio",
        "active_count",
        "combined_complete",
        "isolation_complete",
        "combined_target_peak_min_db",
        "combined_target_peak_mean_db",
        "combined_target_peak_max_db",
        "combined_target_spread_db",
        "combined_worst_sidelobe_abs_db",
        "combined_psll_to_weakest_peak_db",
        "isolation_worst_nearest_db",
        "isolation_mean_nearest_db",
        "isolation_worst_local_db",
        "isolation_mean_local_db",
        "combined_target_peaks_json",
        "isolation_per_task_nearest_json",
        "isolation_per_task_local_json",
        "isolation_matrix_nearest_json",
        "isolation_matrix_local_json",
        "nearest_distance_matrix_json",
        "targets_json",
        "combined_output_dir",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary_rows = build_summary(rows, parse_float_list(args.isolation_thresholds_db))
    summary_path = out_dir / "hfss_task_fullwave_summary_by_ratio.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    gate_summary = write_fullwave_gate_outputs(
        out_dir=out_dir,
        rows=rows,
        psll_max_db=float(args.gate_psll_max_db),
        nearest_iso_min_db=float(args.gate_nearest_iso_min_db),
        local_iso_min_db=float(args.gate_local_iso_min_db),
        local_iso_strict_db=float(args.gate_local_iso_strict_db),
    )

    summary = {
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "metrics": str(metrics_path),
        "summary_by_ratio": str(summary_path),
        "fullwave_gate": gate_summary,
        "sample_count": len(rows),
        "combined_complete_count": int(sum(int(row.get("combined_complete", 0)) for row in rows)),
        "isolation_complete_count": int(sum(int(row.get("isolation_complete", 0)) for row in rows)),
    }
    with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_summary(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    ratios = sorted({float(row["active_ratio"]) for row in rows})
    groups.extend((f"{ratio:.1f}", [row for row in rows if abs(float(row["active_ratio"]) - ratio) < 1.0e-6]) for ratio in ratios)
    out: list[dict[str, Any]] = []
    for ratio_label, group in groups:
        if not group:
            continue
        record: dict[str, Any] = {
            "active_ratio": ratio_label,
            "n": len(group),
            "combined_complete_n": int(sum(int(row.get("combined_complete", 0)) for row in group)),
            "isolation_complete_n": int(sum(int(row.get("isolation_complete", 0)) for row in group)),
            "active_count_mean": finite_mean(group, "active_count"),
            "combined_psll_mean_db": finite_mean(group, "combined_psll_to_weakest_peak_db"),
            "combined_psll_p50_db": finite_percentile(group, "combined_psll_to_weakest_peak_db", 50),
            "combined_psll_p95_db": finite_percentile(group, "combined_psll_to_weakest_peak_db", 95),
            "combined_target_spread_mean_db": finite_mean(group, "combined_target_spread_db"),
            "isolation_worst_nearest_mean_db": finite_mean(group, "isolation_worst_nearest_db"),
            "isolation_worst_nearest_p05_db": finite_percentile(group, "isolation_worst_nearest_db", 5),
            "isolation_worst_nearest_p50_db": finite_percentile(group, "isolation_worst_nearest_db", 50),
            "isolation_worst_nearest_min_db": finite_min(group, "isolation_worst_nearest_db"),
            "isolation_worst_local_mean_db": finite_mean(group, "isolation_worst_local_db"),
            "isolation_worst_local_p05_db": finite_percentile(group, "isolation_worst_local_db", 5),
            "isolation_worst_local_min_db": finite_min(group, "isolation_worst_local_db"),
        }
        for threshold in thresholds:
            key = f"isolation_ge_{threshold:g}db_rate"
            record[key] = pass_rate(group, "isolation_worst_nearest_db", threshold)
        out.append(record)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.out_root / args.run_name)
    if args.mode in {"prepare", "both"}:
        if out_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output directory exists, pass --overwrite to replace: {out_dir}")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prepare_summary = prepare_cases(args, out_dir)
        print(json.dumps(prepare_summary, indent=2))
    if args.mode in {"analyze", "both"}:
        if not (out_dir / "hfss_task_case_manifest.csv").exists():
            print(f"Analysis skipped: manifest not found in {out_dir}")
            return
        analysis_summary = analyze_outputs(args, out_dir)
        print(json.dumps(analysis_summary, indent=2))


if __name__ == "__main__":
    main()
