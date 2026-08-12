#!/usr/bin/env python3
"""Audit and gate the Marlin U-slot and MyriadRF Vivaldi reference branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from run_v114_small_cell_broadband_feed import memory_available_gb
from run_v121_parametric_feed_post import aedt_processes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v148_reference_element_import_preregistered.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else None


def parse_component(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="latin-1", errors="replace")
    variables = {
        name: value
        for name, value in re.findall(r"VariableProp\('([^']+)',\s*'[^']*',\s*'[^']*',\s*'([^']*)'\)", text)
    }
    parts = []
    for match in re.finditer(r"\$begin 'GeometryPart'(.*?)\$end 'GeometryPart'", text, re.DOTALL):
        body = match.group(1)
        name = first_match(r"\bName='([^']+)'", body)
        material = first_match(r"\bMaterialValue='([^']*)'", body)
        part_id = first_match(r"\bID=(\d+)", body)
        if name:
            parts.append({"name": name, "material": material, "first_operation_id": part_id})
    materials = re.findall(r"\$begin '([^']+)'\s+CoordinateSystemType=", text)
    included = first_match(r"IncludedParts\[\d+:\s*([^\]]*)\]", text)
    assigned = first_match(r"AssignedObject\[\d+:\s*([^\]]*)\]", text)
    version = re.search(r"Version\((\d+),\s*(\d+)\)", text)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "component_name": first_match(r"ComponentName='([^']+)'", text),
        "owner": first_match(r"Owner='([^']*)'", text),
        "product": first_match(r"ProductName='([^']+)'", text),
        "solution_type": first_match(r"SolutionType='([^']+)'", text),
        "source_aedt_version": [int(version.group(1)), int(version.group(2))] if version else None,
        "format_version": first_match(r"FormatVersion=(\d+)", text),
        "component_meshing": first_match(r"\bType='([^']+)'", text),
        "variable_count": len(variables),
        "variables": variables,
        "part_count": len(parts),
        "parts": parts,
        "material_definitions": materials,
        "included_part_ids": included,
        "assigned_entity_ids": assigned,
        "contains_ro4350_text": "RO4350" in text.upper(),
        "contains_port_token": bool(re.search(r"LumpedPort|WavePort|Excitation", text, re.IGNORECASE)),
        "contains_boundary_token": bool(re.search(r"PerfectE|FiniteCond|Radiation", text, re.IGNORECASE)),
        "static_risks": [
            "Source component was exported by AEDT 2025.2 while the installed solver is AEDT 2023.1.",
            "The component header exposes copper/vacuum definitions only; dielectric assignment must be verified after import.",
            "Static text does not prove that a 50-ohm excitation and radiation boundary survive component insertion.",
            "The upstream Marlin repository does not publish measured S11 or an explicit license file.",
        ],
    }


def parse_vivaldi_board(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    edge_points = []
    for x1, y1, x2, y2 in re.findall(
        r"\(gr_line \(start ([\d.\-]+) ([\d.\-]+)\) \(end ([\d.\-]+) ([\d.\-]+)\).*?\(layer Edge\.Cuts\)",
        text,
    ):
        edge_points.extend([(float(x1), float(y1)), (float(x2), float(y2))])
    width = max(x for x, _ in edge_points) - min(x for x, _ in edge_points)
    height = max(y for _, y in edge_points) - min(y for _, y in edge_points)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "edge_width_mm": width,
        "edge_height_mm": height,
        "sma_module_count": len(re.findall(r"\(module SMA_H", text)),
        "via_count": len(re.findall(r"\(via ", text)),
        "front_copper_zone_count": len(re.findall(r"\(zone .*?\(layer F\.Cu\)", text)),
        "back_copper_zone_count": len(re.findall(r"\(zone .*?\(layer B\.Cu\)", text)),
    }


def vbs_path(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def import_vbs(component: Path, project: Path, audit: Path) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oEditor, oBoundary, fso, outFile
Dim solids, sheets, lines, unclassified, boundaries, excitations, instances
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{vbs_path(audit)}", True)
On Error Resume Next
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.InsertDesign "HFSS", "MarlinUslotImport", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("MarlinUslotImport")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
Err.Clear
oEditor.Insert3DComponent Array("NAME:InsertComponentData", "Parameters:=", "", "TargetCS:=", "Global", "ComponentFile:=", "{vbs_path(component)}")
outFile.WriteLine "insert_error_number=" & CStr(Err.Number)
outFile.WriteLine "insert_error_description=" & Replace(CStr(Err.Description), vbCrLf, " ")
Err.Clear
solids = oEditor.GetObjectsInGroup("Solids")
outFile.WriteLine "solids=" & JoinSafe(solids)
sheets = oEditor.GetObjectsInGroup("Sheets")
outFile.WriteLine "sheets=" & JoinSafe(sheets)
lines = oEditor.GetObjectsInGroup("Lines")
outFile.WriteLine "lines=" & JoinSafe(lines)
unclassified = oEditor.GetObjectsInGroup("Unclassified")
outFile.WriteLine "unclassified=" & JoinSafe(unclassified)
instances = oEditor.Get3DComponentInstanceNames()
outFile.WriteLine "component_instances=" & JoinSafe(instances)
Set oBoundary = oDesign.GetModule("BoundarySetup")
boundaries = oBoundary.GetBoundaries()
outFile.WriteLine "boundaries=" & JoinSafe(boundaries)
excitations = oBoundary.GetExcitations()
outFile.WriteLine "excitations=" & JoinSafe(excitations)
outFile.WriteLine "audit_error_number=" & CStr(Err.Number)
outFile.WriteLine "audit_error_description=" & Replace(CStr(Err.Description), vbCrLf, " ")
On Error GoTo 0
oProject.SaveAs "{vbs_path(project)}", True
outFile.WriteLine "project_saved=" & CStr(fso.FileExists("{vbs_path(project)}"))
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Function JoinSafe(value)
    On Error Resume Next
    JoinSafe = Join(value, "|")
    If Err.Number <> 0 Then
        Err.Clear
        JoinSafe = CStr(value)
    End If
    On Error GoTo 0
End Function
'''


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing v1.48 output: {out}")
    out.mkdir(parents=True)
    sources = config["sources"]
    marlin_repo = resolve(sources["marlin_repository"])
    vivaldi_repo = resolve(sources["vivaldi_repository"])
    files = {
        key: resolve(value)
        for key, value in sources.items()
        if key not in {"marlin_repository", "marlin_commit", "vivaldi_repository", "vivaldi_commit"}
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")
    external = {
        "marlin_expected_commit": sources["marlin_commit"],
        "marlin_actual_commit": git(marlin_repo, "rev-parse", "HEAD"),
        "vivaldi_expected_commit": sources["vivaldi_commit"],
        "vivaldi_actual_commit": git(vivaldi_repo, "rev-parse", "HEAD"),
        "files": {key: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for key, path in files.items()},
    }
    external["commit_gate_pass"] = bool(
        external["marlin_expected_commit"] == external["marlin_actual_commit"]
        and external["vivaldi_expected_commit"] == external["vivaldi_actual_commit"]
    )
    marlin = parse_component(files["marlin_component"])
    vivaldi = {
        **parse_vivaldi_board(files["vivaldi_board"]),
        **config["vivaldi_calibration"],
        "report_path": str(files["vivaldi_report"]),
        "report_sha256": sha256(files["vivaldi_report"]),
        "evidence_boundary": "Measured plot and tabulated boresight gain exist, but raw VNA Touchstone data do not.",
    }
    write_json(out / "external_source_manifest.json", external)
    write_json(out / "marlin_static_component_audit.json", marlin)
    write_json(out / "myriadrf_vivaldi_calibration_preregistration.json", vivaldi)
    import_dir = out / "marlin_import_smoke"
    import_dir.mkdir()
    project = import_dir / "v148_marlin_import_smoke.aedt"
    audit = import_dir / "aedt_import_audit.txt"
    script = import_dir / "import_audit.vbs"
    script.write_text(import_vbs(files["marlin_component"], project, audit), encoding="ascii")
    preregistration = {
        **config,
        "prepared_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "external_source_manifest": str(out / "external_source_manifest.json"),
        "marlin_static_audit": str(out / "marlin_static_component_audit.json"),
        "vivaldi_calibration": str(out / "myriadrf_vivaldi_calibration_preregistration.json"),
        "import_script": str(script),
        "import_project": str(project),
        "import_audit": str(audit),
    }
    write_json(out / "preregistration.json", preregistration)
    write_json(out / "stage_decision.json", {
        "stage": "prepared",
        "allow_marlin_import_smoke": external["commit_gate_pass"],
        "allow_marlin_1x1_solve": False,
        "allow_marlin_independent_validation": False,
        "allow_marlin_2x2": False,
        **config["locks"],
    })
    return {"output_directory": str(out), "external_commit_gate_pass": external["commit_gate_pass"], "marlin_static_risk_count": len(marlin["static_risks"])}


def parse_key_value(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key.strip()] = value.strip()
    return rows


def import_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    prereg = read_json(out / "preregistration.json")
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_marlin_import_smoke"):
        raise RuntimeError("Marlin import smoke is locked by the source-integrity gate")
    if aedt_processes():
        raise RuntimeError(f"Concurrent AEDT processes are not allowed: {aedt_processes()}")
    folder = Path(prereg["import_script"]).parent
    log = folder / "import_audit.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", prereg["import_script"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    audit_path = Path(prereg["import_audit"])
    audit = parse_key_value(audit_path) if audit_path.exists() else {}
    insert_error = int(audit.get("insert_error_number", "-1") or -1)
    excitation_text = audit.get("excitations", "")
    boundary_text = audit.get("boundaries", "")
    object_text = "|".join(audit.get(key, "") for key in ("solids", "sheets", "lines"))
    compatibility_pass = result.returncode == 0 and insert_error == 0 and Path(prereg["import_project"]).exists()
    port_gate = bool(excitation_text and excitation_text.lower() not in {"empty", "none"})
    boundary_gate = bool(boundary_text and boundary_text.lower() not in {"empty", "none"})
    object_gate = bool(object_text.strip("|"))
    passed = bool(compatibility_pass and port_gate and boundary_gate and object_gate)
    summary = {
        "return_code": result.returncode,
        "free_memory_gib_after": memory_available_gb(),
        "aedt_processes_after": aedt_processes(),
        "raw_audit": audit,
        "compatibility_gate_pass": compatibility_pass,
        "object_gate_pass": object_gate,
        "excitation_gate_pass": port_gate,
        "boundary_gate_pass": boundary_gate,
        "import_smoke_gate_pass": passed,
        "next_action": "Build a frozen 1x1 sweep wrapper around the imported component." if passed else "Stop before solving; repair AEDT-version compatibility or reconstruct missing material/port/boundary data from the source project.",
    }
    write_json(folder / "import_smoke_summary.json", summary)
    write_json(out / "stage_decision.json", {
        "stage": "marlin_import_smoke_complete",
        "allow_marlin_import_smoke": True,
        "allow_marlin_1x1_solve": passed,
        "allow_marlin_independent_validation": False,
        "allow_marlin_2x2": False,
        **config["locks"],
        "next_action": summary["next_action"],
    })
    return summary


def compatibility_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    prereg = read_json(out / "preregistration.json")
    original = resolve(config["sources"]["marlin_component"])
    settings = config["compatibility_rescue"]
    folder = out / "marlin_import_compatibility_smoke"
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Refusing to overwrite compatibility smoke: {folder}")
    folder.mkdir(parents=True)
    adapted = folder / "u_slot_patch_antenna_ro4350b_4L_sma_aedt231_diagnostic.a3dcomp"
    payload = original.read_bytes()
    source = str(settings["source_version_text"]).encode("ascii")
    target = str(settings["diagnostic_version_text"]).encode("ascii")
    if len(source) != len(target):
        raise ValueError("Compatibility header replacement must preserve byte length")
    if payload.count(source) != 1:
        raise ValueError("Expected exactly one component version marker")
    adapted.write_bytes(payload.replace(source, target, 1))
    project = folder / "v148_marlin_compatibility_import_smoke.aedt"
    audit_path = folder / "aedt_import_audit.txt"
    script = folder / "import_audit.vbs"
    script.write_text(import_vbs(adapted, project, audit_path), encoding="ascii")
    manifest = {
        "source_component": str(original),
        "source_sha256": sha256(original),
        "adapted_component": str(adapted),
        "adapted_sha256": sha256(adapted),
        "byte_count_unchanged": original.stat().st_size == adapted.stat().st_size,
        "header_change_only": True,
        "scope": settings["scope"],
        "engineering_evidence_allowed": settings["engineering_evidence_allowed"],
    }
    write_json(folder / "compatibility_artifact_manifest.json", manifest)
    if aedt_processes():
        raise RuntimeError(f"Concurrent AEDT processes are not allowed: {aedt_processes()}")
    with (folder / "import_audit.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", str(script)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    audit = parse_key_value(audit_path) if audit_path.exists() else {}
    insert_error = int(audit.get("insert_error_number", "-1") or -1)
    compatibility_pass = bool(result.returncode == 0 and insert_error == 0 and project.exists())
    object_text = "|".join(audit.get(key, "") for key in ("solids", "sheets", "lines"))
    object_gate = bool(object_text.strip("|"))
    excitation_gate = bool(audit.get("excitations", ""))
    boundary_gate = bool(audit.get("boundaries", ""))
    import_pass = bool(compatibility_pass and object_gate)
    engineering_ready = bool(import_pass and excitation_gate and boundary_gate and manifest["engineering_evidence_allowed"])
    summary = {
        "return_code": result.returncode,
        "raw_audit": audit,
        "compatibility_gate_pass": compatibility_pass,
        "object_gate_pass": object_gate,
        "excitation_present": excitation_gate,
        "boundary_present": boundary_gate,
        "diagnostic_import_gate_pass": import_pass,
        "engineering_model_ready": engineering_ready,
        "free_memory_gib_after": memory_available_gb(),
        "aedt_processes_after": aedt_processes(),
        "next_action": "Reconstruct the missing substrate, port, reference plane, radiation region, and solution setup around the imported exact copper geometry." if import_pass else "Reject AEDT 2023.1 component import and rebuild from Gerber/PCB artwork or use AEDT 2025.2.",
    }
    write_json(folder / "compatibility_smoke_summary.json", summary)
    write_json(out / "stage_decision.json", {
        "stage": "marlin_compatibility_smoke_complete",
        "allow_marlin_1x1_model_reconstruction": import_pass,
        "allow_marlin_1x1_solve": False,
        "allow_marlin_independent_validation": False,
        "allow_marlin_2x2": False,
        **config["locks"],
        "next_action": summary["next_action"],
    })
    return summary


def project_audit_vbs(project: Path, audit: Path, design_name: str) -> str:
    return f'''Option Explicit
Dim oApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, fso, outFile
Dim designs, solids, sheets, boundaries, excitations, setups, sweeps
Set fso = CreateObject("Scripting.FileSystemObject")
Set outFile = fso.CreateTextFile("{vbs_path(audit)}", True)
On Error Resume Next
Set oApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oApp.GetAppDesktop()
Err.Clear
oDesktop.OpenProject "{vbs_path(project)}"
outFile.WriteLine "open_error_number=" & CStr(Err.Number)
outFile.WriteLine "open_error_description=" & Replace(CStr(Err.Description), vbCrLf, " ")
Set oProject = oDesktop.GetActiveProject()
designs = oProject.GetTopDesignList()
outFile.WriteLine "designs=" & JoinSafe(designs)
Err.Clear
Set oDesign = oProject.SetActiveDesign("{design_name}")
outFile.WriteLine "set_design_error_number=" & CStr(Err.Number)
outFile.WriteLine "set_design_error_description=" & Replace(CStr(Err.Description), vbCrLf, " ")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
solids = oEditor.GetObjectsInGroup("Solids")
outFile.WriteLine "solids=" & JoinSafe(solids)
sheets = oEditor.GetObjectsInGroup("Sheets")
outFile.WriteLine "sheets=" & JoinSafe(sheets)
Set oBoundary = oDesign.GetModule("BoundarySetup")
boundaries = oBoundary.GetBoundaries()
outFile.WriteLine "boundaries=" & JoinSafe(boundaries)
excitations = oBoundary.GetExcitations()
outFile.WriteLine "excitations=" & JoinSafe(excitations)
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
setups = oAnalysis.GetSetups()
outFile.WriteLine "setups=" & JoinSafe(setups)
If IsArray(setups) Then
    If UBound(setups) >= LBound(setups) Then
        sweeps = oAnalysis.GetSweeps(CStr(setups(LBound(setups))))
        outFile.WriteLine "first_setup_sweeps=" & JoinSafe(sweeps)
    End If
End If
outFile.WriteLine "audit_error_number=" & CStr(Err.Number)
outFile.WriteLine "audit_error_description=" & Replace(CStr(Err.Description), vbCrLf, " ")
outFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Function JoinSafe(value)
    On Error Resume Next
    JoinSafe = Join(value, "|")
    If Err.Number <> 0 Then
        Err.Clear
        JoinSafe = CStr(value)
    End If
    On Error GoTo 0
End Function
'''


def project_compatibility_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    original = resolve(config["sources"]["marlin_project"])
    settings = config["compatibility_rescue"]
    folder = out / "marlin_project_compatibility_smoke"
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Refusing to overwrite project compatibility smoke: {folder}")
    folder.mkdir(parents=True)
    adapted = folder / "MARLIN_aedt231_diagnostic.aedt"
    payload = original.read_bytes()
    source = str(settings["source_version_text"]).encode("ascii")
    target = str(settings["diagnostic_version_text"]).encode("ascii")
    count = payload.count(source)
    if count < 1:
        raise ValueError("No project version markers found")
    adapted.write_bytes(payload.replace(source, target))
    audit_path = folder / "aedt_project_audit.txt"
    script = folder / "project_audit.vbs"
    design_name = "u_slot_patch_antenna_ro4350b_4L_sma"
    script.write_text(project_audit_vbs(adapted, audit_path, design_name), encoding="ascii")
    manifest = {
        "source_project": str(original),
        "source_sha256": sha256(original),
        "adapted_project": str(adapted),
        "adapted_sha256": sha256(adapted),
        "version_marker_replacement_count": count,
        "byte_count_unchanged": original.stat().st_size == adapted.stat().st_size,
        "scope": "read-only project audit",
        "engineering_evidence_allowed": False,
    }
    write_json(folder / "compatibility_artifact_manifest.json", manifest)
    if aedt_processes():
        raise RuntimeError(f"Concurrent AEDT processes are not allowed: {aedt_processes()}")
    with (folder / "project_audit.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", str(script)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    audit = parse_key_value(audit_path) if audit_path.exists() else {}
    open_pass = bool(result.returncode == 0 and audit.get("open_error_number") == "0")
    design_pass = bool(audit.get("set_design_error_number") == "0")
    object_pass = bool((audit.get("solids", "") + audit.get("sheets", "")).strip())
    port_pass = bool(audit.get("excitations", "").strip())
    boundary_pass = bool(audit.get("boundaries", "").strip())
    setup_pass = bool(audit.get("setups", "").strip())
    passed = bool(open_pass and design_pass and object_pass and port_pass and boundary_pass and setup_pass)
    summary = {
        "return_code": result.returncode,
        "raw_audit": audit,
        "project_open_gate_pass": open_pass,
        "target_design_gate_pass": design_pass,
        "object_gate_pass": object_pass,
        "excitation_gate_pass": port_pass,
        "boundary_gate_pass": boundary_pass,
        "setup_gate_pass": setup_pass,
        "project_audit_gate_pass": passed,
        "engineering_model_ready": False,
        "free_memory_gib_after": memory_available_gb(),
        "aedt_processes_after": aedt_processes(),
        "next_action": "Export the audited source design into a new AEDT 2023.1 project, freeze settings, and run the nominal 1x1 sweep." if passed else "Stop the Marlin solve branch on AEDT 2023.1; rebuild from manufacturing artwork or install AEDT 2025.2.",
    }
    write_json(folder / "project_compatibility_smoke_summary.json", summary)
    write_json(out / "stage_decision.json", {
        "stage": "marlin_project_compatibility_smoke_complete",
        "allow_marlin_1x1_model_export": passed,
        "allow_marlin_1x1_solve": False,
        "allow_marlin_independent_validation": False,
        "allow_marlin_2x2": False,
        **config["locks"],
        "next_action": summary["next_action"],
    })
    return summary


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    return {
        "output_directory": str(out),
        "prepared": (out / "preregistration.json").exists(),
        "decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
        "import_smoke": read_json(out / "marlin_import_smoke/import_smoke_summary.json") if (out / "marlin_import_smoke/import_smoke_summary.json").exists() else None,
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
    }


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    import_summary = read_json(out / "marlin_import_smoke/import_smoke_summary.json")
    compatibility_summary = read_json(
        out / "marlin_import_compatibility_smoke/compatibility_smoke_summary.json"
    )
    project_summary = read_json(
        out / "marlin_project_compatibility_smoke/project_compatibility_smoke_summary.json"
    )
    project_log = (
        out / "marlin_project_compatibility_smoke/project_audit.log"
    ).read_text(encoding="utf-8", errors="ignore")
    calibration_config = read_json(resolve(config["myriadrf_calibration_config"]))
    calibration_out = resolve(calibration_config["output_directory"])
    calibration_summary = read_json(calibration_out / "nominal_direct/stage_summary.json")
    marlin_ready = bool(
        import_summary["import_smoke_gate_pass"]
        and project_summary["project_audit_gate_pass"]
    )
    result = {
        "version": config["version"],
        "parent_commit": config["parent_commit"],
        "parent_tag": config["parent_tag"],
        "marlin": {
            "source_component_aedt_version": [2025, 2],
            "installed_aedt_version": [2023, 1],
            "original_component_import_gate_pass": import_summary["import_smoke_gate_pass"],
            "header_only_diagnostic_import_compatible": compatibility_summary["compatibility_gate_pass"],
            "header_only_diagnostic_engineering_ready": compatibility_summary["engineering_model_ready"],
            "source_project_audit_gate_pass": project_summary["project_audit_gate_pass"],
            "source_project_encrypted_and_not_decryptable": "Unable to decrypt encrypted project file" in project_log,
            "trusted_1x1_available": marlin_ready,
            "nominal_sweep_completed": False,
            "independent_direct_completed": False,
            "independent_ddm_completed": False,
            "direct_ddm_gate_pass": False,
            "two_by_two_active_rl_completed": False,
            "stop_reason": "AEDT 2023.1 cannot import the original 2025.2 component, the header-only diagnostic has no auditable top-level excitation/boundary, and the complete MARLIN project is encrypted and cannot be opened.",
        },
        "myriadrf": {
            "source_commit": calibration_config["source"]["commit"],
            "pcb_derived_hfss_solve_completed": True,
            "numerical_gate_pass": calibration_summary["numerical_gate_pass"],
            "final_delta_s": calibration_summary["final_delta_s"],
            "gate3_worst_passive_rl_db": calibration_summary["gate_worst_passive_rl_db"],
            "center_passive_rl_db": calibration_summary["center_passive_rl_db"],
            "contiguous_10db_bandwidth_ghz": calibration_summary["contiguous_10db_bandwidth_ghz"],
            "contiguous_10db_band_low_ghz": calibration_summary["contiguous_10db_band_low_ghz"],
            "contiguous_10db_band_high_ghz": calibration_summary["contiguous_10db_band_high_ghz"],
            "mesh_warning_audit": calibration_summary["mesh_warning_audit"],
            "radiation_efficiency_observed": calibration_summary["radiation_efficiency"] is not None,
            "endfire_gain_observed": calibration_summary["endfire_gain_10ghz_dbi"] is not None,
            "raw_measured_touchstone_available": calibration_config["evidence_policy"]["raw_measured_touchstone_available"],
            "simulated_reference_consistency_gate_pass": calibration_summary["simulated_reference_consistency_gate_pass"],
            "measured_validation_gate_pass": calibration_summary["measured_validation_gate_pass"],
            "allowed_use": "qualitative S11 port/material/reference-plane calibration evidence only",
        },
        "physical_gate_pass": False,
        "allow_marlin_1x1_solve": False,
        "allow_independent_direct_ddm": False,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_s256_export": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "next_action": "Obtain AEDT 2025.2 or an unencrypted Marlin source project with dielectric, port, and boundary definitions; obtain raw MyriadRF VNA data if exact port/reference-plane regression is required.",
    }
    write_json(out / "stage_summary.json", result)
    write_json(out / "stage_decision.json", {"stage": "v148_final_stop_gate", **result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("prepare", "import-smoke", "compatibility-smoke", "project-compatibility-smoke", "finalize", "status"), required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    result = {
        "prepare": prepare,
        "import-smoke": import_smoke,
        "compatibility-smoke": compatibility_smoke,
        "project-compatibility-smoke": project_compatibility_smoke,
        "finalize": finalize,
        "status": status,
    }[args.mode](config)
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
