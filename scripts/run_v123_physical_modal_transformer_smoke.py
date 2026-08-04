#!/usr/bin/env python3
"""Build, solve, and gate the single authorized v1.23 10 GHz physical S8 smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from design_v115_grounded_modal_network import terminate_network
from design_v119_multiport_post_decoupler import active_metrics, reordered_network
from run_v114_small_cell_broadband_feed import load_stimuli, memory_available_gb, parse_touchstone, profile_metrics
from run_v115_physical_modal_feed_fixture import touchstone_port_names
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v122_balanced_modal_branch import common_s_metrics, loaded_efficiencies, small_segment_diagnostics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v123_physical_modal_transformer_smoke_preregistered.json"
DESIGN_NAME = "V123_Physical_Modal_Transformer_S8"
EPS = 1.0e-15


def resolve(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def cascade_s(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cascade equal-width 2N-port networks with [left, right] partitions."""
    n = a.shape[0] // 2
    a11, a12 = a[:n, :n], a[:n, n:]
    a21, a22 = a[n:, :n], a[n:, n:]
    b11, b12 = b[:n, :n], b[:n, n:]
    b21, b22 = b[n:, :n], b[n:, n:]
    d = np.linalg.inv(np.eye(n, dtype=complex) - a22 @ b11)
    result = np.empty_like(a)
    result[:n, :n] = a11 + a12 @ b11 @ d @ a21
    result[:n, n:] = a12 @ (b11 @ d @ a22 @ b12 + b12)
    result[n:, :n] = b21 @ d @ a21
    result[n:, n:] = b22 + b21 @ d @ a22 @ b12
    return result


def helpers_vbs() -> str:
    return r'''Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(190 120 50)", "Transparency:=", 0.1, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheetX(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "X"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(210 150 55)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub AssignDifferentialPort(boundary, portName, sheetName, x, yNegative, yPositive, z)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x), Mm(yNegative), Mm(z)), "End:=", Array(Mm(x), Mm(yPositive), Mm(z))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Sub AssignRLC(boundary, boundaryName, sheetName, useR, resistanceOhm, useL, inductanceNh, useC, capacitancePf, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedRLC Array("NAME:" & boundaryName, "Objects:=", Array(sheetName), Array("NAME:CurrentLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "RLC Type:=", "Serial", "UseResist:=", useR, "Resistance:=", CStr(resistanceOhm) & "ohm", "UseInduct:=", useL, "Inductance:=", CStr(inductanceNh) & "nH", "UseCap:=", useC, "Capacitance:=", CStr(capacitancePf) & "pF")
End Sub
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function'''


def component_values(config: dict[str, Any]) -> dict[str, float]:
    block = config["single_local_block"]
    frequency_hz = float(config["frequency_ghz"]) * 1.0e9
    omega = 2.0 * math.pi * frequency_hz
    cap_pf = float(block["per_conductor_ground_capacitance_pf"])
    cap_q = float(block["ground_capacitor_q"])
    inductance_nh = float(block["per_polarity_bridge_inductance_nh"])
    inductance_q = float(block["bridge_inductor_q"])
    return {
        "ground_capacitance_pf": cap_pf,
        "ground_capacitor_esr_ohm": 1.0 / (omega * cap_pf * 1.0e-12 * cap_q),
        "bridge_inductance_nh": inductance_nh,
        "bridge_inductor_esr_ohm": omega * inductance_nh * 1.0e-9 / inductance_q,
    }


def builder_text(project: Path, touchstone: Path, config: dict[str, Any]) -> str:
    launch = config["frozen_launch"]
    block = config["single_local_block"]
    mesh = config["mesh"]
    values = component_values(config)
    h = float(launch["substrate_thickness_mm"])
    copper = float(launch["copper_thickness_mm"])
    board_l = float(launch["board_length_mm"])
    board_w = float(launch["board_width_mm"])
    pre_x = float(launch["pre_reference_x_mm"])
    post_x = float(launch["post_reference_x_mm"])
    pitch = float(launch["pair_center_pitch_mm"])
    trace_w = float(launch["trace_width_mm"])
    y_by_port = [float(value) for value in launch["channel_y_mm_by_antenna_port"]]
    physical_order = [int(value) for value in launch["physical_channel_order_negative_to_positive_y"]]
    load_x = float(block["ground_load_x_mm"])
    load_w = float(block["ground_sheet_width_mm"])
    bridge_x = [float(value) for value in block["bridge_x_mm_by_polarity"]]
    bridge_z = float(block["bridge_height_mm"])
    bridge_w = float(block["bridge_sheet_width_mm"])

    geometry: list[str] = []
    assignments: list[str] = []
    trace_names: list[str] = []
    component_names: list[str] = []
    conductor_names = ["ReferenceGround"]
    trace_y: dict[tuple[int, str], float] = {}
    for port in physical_order:
        center = y_by_port[port]
        for polarity, y_value in (("N", center - pitch / 2.0), ("P", center + pitch / 2.0)):
            trace_y[(port, polarity)] = y_value
            name = f"Trace{polarity}_{port}"
            geometry.append(
                f'CreateSheetZ oEditor, "{name}", {pre_x:.7f}, {y_value-trace_w/2:.7f}, 0, {post_x-pre_x:.7f}, {trace_w:.7f}'
            )
            trace_names.append(name)
            conductor_names.append(name)
            cap_name = f"GroundCapSheet_{polarity}_{port}"
            geometry.append(
                f'CreateSheetX oEditor, "{cap_name}", {load_x:.7f}, {y_value-load_w/2:.7f}, {-h:.7f}, {load_w:.7f}, {h:.7f}'
            )
            assignments.append(
                f'AssignRLC oBoundary, "GroundCap_{polarity}_{port}", "{cap_name}", True, {values["ground_capacitor_esr_ohm"]:.12f}, False, 1, True, {values["ground_capacitance_pf"]:.12f}, {load_x:.7f}, {y_value:.7f}, 0, {load_x:.7f}, {y_value:.7f}, {-h:.7f}'
            )
            component_names.append(cap_name)
        negative_y = center - pitch / 2.0
        positive_y = center + pitch / 2.0
        geometry.extend(
            [
                f'CreateSheetX oEditor, "PrePortSheet_{port}", {pre_x:.7f}, {negative_y:.7f}, 0, {pitch:.7f}, 0.1',
                f'CreateSheetX oEditor, "PostPortSheet_{port}", {post_x:.7f}, {negative_y:.7f}, 0, {pitch:.7f}, 0.1',
            ]
        )
        assignments.append(
            f'AssignDifferentialPort oBoundary, "PRE_{port}", "PrePortSheet_{port}", {pre_x:.7f}, {negative_y:.7f}, {positive_y:.7f}, 0'
        )
    for port in range(4):
        center = y_by_port[port]
        assignments.append(
            f'AssignDifferentialPort oBoundary, "POST_{port}", "PostPortSheet_{port}", {post_x:.7f}, {center-pitch/2:.7f}, {center+pitch/2:.7f}, 0'
        )

    for pair_index, pair in enumerate(launch["local_x_neighbor_pairs"]):
        first, second = int(pair[0]), int(pair[1])
        for polarity_index, polarity in enumerate(("N", "P")):
            x_value = bridge_x[polarity_index]
            ys = sorted((trace_y[(first, polarity)], trace_y[(second, polarity)]))
            for endpoint, y_value in enumerate(ys):
                via = f"BridgeVia_{pair_index}_{polarity}_{endpoint}"
                geometry.append(
                    f'CreateBox oEditor, "{via}", {x_value-bridge_w/2:.7f}, {y_value-bridge_w/2:.7f}, 0, {bridge_w:.7f}, {bridge_w:.7f}, {bridge_z:.7f}, "copper", False'
                )
                component_names.append(via)
            bridge = f"BridgeSheet_{pair_index}_{polarity}"
            geometry.append(
                f'CreateSheetZ oEditor, "{bridge}", {x_value-bridge_w/2:.7f}, {ys[0]:.7f}, {bridge_z:.7f}, {bridge_w:.7f}, {ys[1]-ys[0]:.7f}'
            )
            assignments.append(
                f'AssignRLC oBoundary, "BridgeL_{pair_index}_{polarity}", "{bridge}", True, {values["bridge_inductor_esr_ohm"]:.12f}, True, {values["bridge_inductance_nh"]:.12f}, False, 1, {x_value:.7f}, {ys[0]:.7f}, {bridge_z:.7f}, {x_value:.7f}, {ys[1]:.7f}, {bridge_z:.7f}'
            )
            component_names.append(bridge)

    finite_objects = ", ".join(f'"{name}"' for name in conductor_names)
    trace_mesh = ", ".join(f'"{name}"' for name in trace_names)
    component_mesh = ", ".join(f'"{name}"' for name in component_names)
    expected_ports = [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)]
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:{launch['substrate_material']}", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{launch['relative_permittivity']}", "dielectric_loss_tangent:=", "{launch['loss_tangent']}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h:.7f}, {board_l:.7f}, {board_w:.7f}, {h:.7f}, "{launch['substrate_material']}", True
CreateSheetZ oEditor, "ReferenceGround", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h:.7f}, {board_l:.7f}, {board_w:.7f}
{chr(10).join(geometry)}
oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", "Objects:=", Array({finite_objects}), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)
{chr(10).join(assignments)}
CreateBox oEditor, "AirRegion", {-board_l/2-4:.7f}, {-board_w/2-4:.7f}, -3, {board_l+8:.7f}, {board_w+8:.7f}, 6.5, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:FrozenTraceMesh_0p120mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({trace_mesh}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(mesh['trace_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
oMesh.AssignLengthOp Array("NAME:LocalBlockMesh_0p060mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({component_mesh}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(mesh['component_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{float(config['frequency_ghz']):g}GHz", "MaxDeltaS:=", {float(config['gates']['maximum_final_delta_s']):.7f}, "MaximumPasses:=", {int(mesh['maximum_passes'])}, "MinimumPasses:=", {int(mesh['minimum_passes'])}, "MinimumConvergedPasses:=", {int(mesh['minimum_converged_passes'])}, "PercentRefinement:=", {float(mesh['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "PortAccuracy:=", 2, "DrivenSolverType:=", "Iterative Solver", "IterativeResidual:=", {float(mesh['iterative_residual']):.10g})
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{helpers_vbs()}
'''


def solver_text(project: Path, touchstone: Path) -> str:
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, vars, variation
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def verify_upstream(config: dict[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    upstream = config["upstream"]
    for key in ("stage_decision", "finite_q_upper_bound", "selected_circuit_operator", "frozen_run06_s8_10ghz"):
        path = resolve(upstream[key])
        actual = sha256_file(path)
        expected = str(upstream[f"{key}_sha256"]).lower()
        if actual != expected:
            raise RuntimeError(f"Upstream hash mismatch for {key}: {actual} != {expected}")
        verified[key] = actual
    decision = read_json(resolve(upstream["stage_decision"]))
    if not decision.get("allow_initial_10ghz_network_only_s8"):
        raise RuntimeError("The v1.23 circuit decision does not authorize the 10 GHz physical S8")
    if decision.get("allow_three_frequency_hfss") or decision.get("allow_integrated_2x2"):
        raise RuntimeError("Upstream decision unexpectedly authorizes a later HFSS stage")
    return verified


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.23 physical output: {out}")
    out.mkdir(parents=True)
    verified = verify_upstream(config)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    finite_q = read_json(resolve(config["upstream"]["finite_q_upper_bound"]))
    values = component_values(config)
    expected_cap = 2.0 * float(finite_q["components"]["symmetric_ground_branch"]["value_pf"])
    expected_inductor = 0.5 * float(finite_q["components"]["pair_bridge_branch"]["value_nh"])
    if abs(values["ground_capacitance_pf"] - expected_cap) > 1.0e-10:
        raise RuntimeError("Balanced ground-capacitor mapping is inconsistent with the circuit upper bound")
    if abs(values["bridge_inductance_nh"] - expected_inductor) > 1.0e-10:
        raise RuntimeError("Balanced bridge-inductor mapping is inconsistent with the circuit upper bound")
    case = out / "network_only_s8_10ghz"
    case.mkdir()
    project = case / "v123_physical_modal_transformer_10ghz.aedt"
    touchstone = case / "v123_physical_modal_transformer_10ghz.s8p"
    builder = case / "build.vbs"
    solver = case / "solve_export.vbs"
    builder.write_text(builder_text(project, touchstone, config), encoding="ascii")
    solver.write_text(solver_text(project, touchstone), encoding="ascii")
    manifest = {
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "expected_port_order": [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)],
        "solution": "Setup_10GHz:LastAdaptive",
        "evidence_scope": "network-only balanced differential HFSS terminated post hoc by the trusted antenna S4; not integrated HFSS",
    }
    write_json(out / "case_manifest.json", manifest)
    write_csv(out / "component_mapping.csv", [{**values, **config["single_local_block"]}])
    preregistration = {
        **config,
        "runtime_audit": {
            "head_commit": head,
            "parent_tag_commit": tag,
            "verified_upstream_hashes": verified,
            "free_memory_gib": memory_available_gb(),
            "aedt_processes": aedt_processes(),
        },
        "evidence_rules": {
            "only_one_10ghz_network_only_s8_authorized": True,
            "no_three_frequency_hfss_before_this_gate": True,
            "no_integrated_model_or_training_labels": True,
            "lumped_component_smoke_is_not_a_packaged_component_manufacturing_validation": True,
        },
    }
    write_json(out / "preregistration.json", preregistration)
    decision = {
        "stage": "A_physical_smoke_preregistered",
        "allow_build_smoke": True,
        "allow_10ghz_solve": False,
        "allow_three_frequency_hfss": False,
        "allow_integrated_2x2": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "manifest": manifest, "decision": decision}


def run_build_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_build_smoke"):
        raise RuntimeError("Build smoke is not authorized")
    require_no_aedt()
    free = memory_available_gb()
    if free < float(config["resources"]["minimum_free_memory_before_build_gib"]):
        raise MemoryError(f"Only {free:.2f} GiB free before build smoke")
    manifest = read_json(out / "case_manifest.json")
    project = Path(manifest["project_path"])
    if project.exists() and project.stat().st_size > 100:
        return {"status": "already_complete", "project_path": str(project)}
    log = project.parent / "build.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(resolve(config["ansys_executable"])), "-RunScriptAndExit", manifest["builder_path"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    payload = {
        "return_code": int(result.returncode),
        "project_exists": project.exists() and project.stat().st_size > 100,
        "free_memory_gib_before": free,
    }
    write_json(project.parent / "build_summary.json", payload)
    return payload


def audit_build_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest = read_json(out / "case_manifest.json")
    builder = Path(manifest["builder_path"]).read_text(encoding="ascii")
    folder = Path(manifest["project_path"]).parent
    log = (folder / "build.log").read_text(encoding="utf-8", errors="ignore")
    summary = read_json(folder / "build_summary.json")
    warnings = {
        pattern: log.lower().count(pattern)
        for pattern in ("script error", "invalid geometry", "small segment", "too many conductors touch lumped port")
    }
    audit = {
        **summary,
        "expected_port_count": 8,
        "differential_port_definition_count": builder.count("AssignDifferentialPort oBoundary"),
        "ground_capacitor_count": len(set(re.findall(r"GroundCap_[NP]_\d+", builder))),
        "bridge_inductor_count": len(set(re.findall(r"BridgeL_\d+_[NP]", builder))),
        "bridge_via_count": len(set(re.findall(r"BridgeVia_\d+_[NP]_\d+", builder))),
        "floating_branch_count": builder.count("ModalFloatingBranch"),
        "warning_counts": warnings,
    }
    audit["gate_pass"] = bool(
        audit["return_code"] == 0
        and audit["project_exists"]
        and audit["differential_port_definition_count"] == 8
        and audit["ground_capacitor_count"] == 8
        and audit["bridge_inductor_count"] == 4
        and audit["bridge_via_count"] == 8
        and audit["floating_branch_count"] == 0
        and sum(warnings.values()) == 0
    )
    write_json(folder / "build_audit.json", audit)
    decision = {
        "stage": "B_build_smoke_complete",
        "build_smoke_pass": audit["gate_pass"],
        "allow_10ghz_solve": audit["gate_pass"],
        "allow_three_frequency_hfss": False,
        "allow_integrated_2x2": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
    }
    write_json(out / "stage_decision.json", decision)
    return {"audit": audit, "decision": decision}


def run_solve(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_10ghz_solve"):
        raise RuntimeError("The 10 GHz physical S8 solve is not authorized")
    require_no_aedt()
    free = memory_available_gb()
    minimum = float(config["resources"]["minimum_free_memory_before_solve_gib"])
    if free < minimum:
        raise MemoryError(f"Only {free:.2f} GiB free; {minimum:.2f} GiB required")
    manifest = read_json(out / "case_manifest.json")
    touchstone = Path(manifest["touchstone_path"])
    if touchstone.exists() and touchstone.stat().st_size > 100:
        return {"status": "already_complete", "touchstone_path": str(touchstone)}
    folder = touchstone.parent
    code, aborted, minimum_free = run_process_with_memory_guard(
        [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", manifest["solver_path"]],
        folder / "solve_export.log",
        float(config["resources"]["abort_free_memory_during_solve_gib"]),
        float(config["resources"]["poll_interval_seconds"]),
    )
    result = {
        "return_code": code,
        "memory_aborted": aborted,
        "minimum_free_memory_gib": minimum_free,
        "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100,
    }
    write_json(folder / "run_summary.json", result)
    return result


def representative_event_rows(
    external: np.ndarray,
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    considered: np.ndarray,
    selected: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    selected_indices = np.flatnonzero(selected)
    for index in selected_indices:
        source = vectors[index, :4]
        reflected = external @ source
        active = considered[index, :4]
        port_rl = -20.0 * np.log10(np.maximum(np.abs(source), EPS) / np.maximum(np.abs(source), EPS))
        port_rl[active] = -20.0 * np.log10(
            np.maximum(np.abs(reflected[active]), EPS) / np.maximum(np.abs(source[active]), EPS)
        )
        total_rl = -10.0 * math.log10(
            max(float(np.sum(np.abs(reflected) ** 2)), EPS) / max(float(np.sum(np.abs(source) ** 2)), EPS)
        )
        considered_rl = port_rl[active]
        output.append(
            {
                "sample_index": rows[index].get("sample_index"),
                "K": rows[index].get("k_value"),
                "ratio": rows[index].get("ratio"),
                "source": rows[index].get("source_type"),
                "mode": rows[index].get("state_name"),
                "worst_active_port": int(np.flatnonzero(active)[int(np.argmin(considered_rl))]) if np.any(active) else -1,
                "active_rl_min_db": float(np.min(considered_rl)) if considered_rl.size else math.inf,
                "total_rl_db": total_rl,
            }
        )
    return sorted(output, key=lambda row: float(row["active_rl_min_db"]))


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest = read_json(out / "case_manifest.json")
    touchstone = Path(manifest["touchstone_path"])
    if not touchstone.exists() or touchstone.stat().st_size <= 100:
        raise RuntimeError("The physical S8 Touchstone is missing")
    exported_names = touchstone_port_names(touchstone)
    if set(exported_names) != set(manifest["expected_port_order"]):
        raise RuntimeError(f"Unexpected physical S8 port names: {exported_names}")
    frequencies, matrices = reordered_network(touchstone, manifest["expected_port_order"], 8)
    frequency = float(config["frequency_ghz"])
    index = int(np.argmin(np.abs(frequencies - frequency)))
    if abs(float(frequencies[index]) - frequency) > 1.0e-6:
        raise RuntimeError(f"Physical S8 frequency mismatch: {frequencies[index]}")
    network = matrices[index]
    antenna_f, antenna_matrices = parse_touchstone(resolve(config["upstream"]["trusted_antenna_s4"]), 4)
    antenna = antenna_matrices[int(np.argmin(np.abs(antenna_f - frequency)))]
    stimulus_rows, vectors, considered = load_stimuli(resolve(config["upstream"]["trusted_stimulus_root"]))
    selected = np.asarray(
        [int(row["side"]) == 2 and abs(float(row["frequency_ghz"]) - frequency) <= 1.0e-9 for row in stimulus_rows]
    )
    sources = vectors[selected, :4].T
    active = considered[selected, :4].T
    external, _, _ = terminate_network(network, antenna)
    active_rl, total_rl = active_metrics(external, sources, active)
    matched_external, load_incident, load_reflected = terminate_network(network, np.zeros((4, 4), dtype=complex))
    accepted = 1.0 - np.sum(np.abs(matched_external) ** 2, axis=0)
    delivered = np.sum(np.abs(load_incident) ** 2, axis=0) - np.sum(np.abs(load_reflected) ** 2, axis=0)
    insertion, transducer = loaded_efficiencies(network, antenna, sources)
    operator = np.load(resolve(config["upstream"]["selected_circuit_operator"]), allow_pickle=False)
    operator_f = operator["frequencies_ghz"]
    operator_index = int(np.argmin(np.abs(operator_f - frequency)))
    target = cascade_s(operator["launch_s8"][operator_index], operator["correction_s8"][operator_index])
    profile = profile_metrics(touchstone.parent)
    segments = small_segment_diagnostics(touchstone.parent)
    common = common_s_metrics(np.asarray([network]))
    summary = {
        **profile,
        **segments,
        **common,
        "frequency_ghz": frequency,
        "exported_port_order": exported_names,
        "representative_source_count": int(np.sum(selected)),
        "matched_load_passive_rl_min_db": float(
            np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matched_external)), EPS)))
        ),
        "representative_active_rl_min_db": active_rl,
        "representative_total_rl_min_db": total_rl,
        "matched_load_network_efficiency_min": float(np.min(delivered / np.maximum(accepted, EPS))),
        "actual_load_insertion_efficiency_min": insertion,
        "actual_load_transducer_efficiency_min": transducer,
        "physical_vs_target_abs_delta_s": float(np.max(np.abs(network - target))),
        "evidence_scope": manifest["evidence_scope"],
    }
    gates = config["gates"]
    checks = {
        "converged": summary.get("converged") is True,
        "final_delta_s": float(summary.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"]),
        "mesh_segment_count": int(summary.get("small_mesh_segment_count") or 0) <= int(gates["maximum_small_mesh_segment_count"]),
        "mesh_segment_length": summary.get("small_mesh_segment_min_length_mm") is None
        or float(summary["small_mesh_segment_min_length_mm"]) >= float(gates["minimum_small_mesh_segment_length_mm"]),
        "reciprocity": summary["reciprocity_error"] <= float(gates["maximum_reciprocity_error"]),
        "passivity": summary["passivity_sigma"] <= float(gates["maximum_passivity_sigma"]),
        "passive_rl": summary["matched_load_passive_rl_min_db"] >= float(gates["minimum_matched_load_passive_rl_db"]),
        "active_rl": summary["representative_active_rl_min_db"] >= float(gates["minimum_representative_active_rl_db"]),
        "total_rl": summary["representative_total_rl_min_db"] >= float(gates["minimum_representative_total_rl_db"]),
        "network_efficiency": summary["matched_load_network_efficiency_min"] >= float(gates["minimum_matched_load_network_efficiency"]),
        "actual_load_insertion_efficiency": summary["actual_load_insertion_efficiency_min"] >= float(gates["minimum_actual_load_insertion_efficiency"]),
        "actual_load_transducer_efficiency": summary["actual_load_transducer_efficiency_min"] >= float(gates["minimum_actual_load_transducer_efficiency"]),
        "physical_vs_target": summary["physical_vs_target_abs_delta_s"] <= float(gates["maximum_physical_vs_target_abs_delta_s"]),
    }
    summary["gate_checks"] = checks
    summary["failed_gates"] = [name for name, passed in checks.items() if not passed]
    summary["gate_pass"] = all(checks.values())
    write_json(touchstone.parent / "analysis.json", summary)
    write_csv(
        touchstone.parent / "representative_active_events.csv",
        representative_event_rows(external, stimulus_rows, vectors, considered, selected),
    )
    decision = {
        "stage": "C_10ghz_physical_s8_gate_complete",
        "physical_10ghz_gate_pass": summary["gate_pass"],
        "allow_three_frequency_hfss": summary["gate_pass"],
        "allow_independent_repeat": False,
        "allow_integrated_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "stop_current_topology": not summary["gate_pass"],
        "reason": (
            "The 10 GHz network-only S8 passed; exact single-frequency 9.96/10.04 GHz solves are now authorized."
            if summary["gate_pass"]
            else "The 10 GHz physical S8 failed: " + ", ".join(summary["failed_gates"])
        ),
    }
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "decision": decision}


def network_metrics(
    network: np.ndarray,
    antenna: np.ndarray,
    sources: np.ndarray,
    considered: np.ndarray,
) -> dict[str, float]:
    external, _, _ = terminate_network(network, antenna)
    active_rl, total_rl = active_metrics(external, sources, considered)
    matched_external, load_incident, load_reflected = terminate_network(network, np.zeros((4, 4), dtype=complex))
    accepted = 1.0 - np.sum(np.abs(matched_external) ** 2, axis=0)
    delivered = np.sum(np.abs(load_incident) ** 2, axis=0) - np.sum(np.abs(load_reflected) ** 2, axis=0)
    insertion, transducer = loaded_efficiencies(network, antenna, sources)
    return {
        "passive_rl_min_db": float(np.min(-20.0 * np.log10(np.maximum(np.abs(np.diag(matched_external)), EPS)))),
        "active_rl_min_db": active_rl,
        "total_rl_min_db": total_rl,
        "matched_efficiency_min": float(np.min(delivered / np.maximum(accepted, EPS))),
        "actual_load_insertion_min": insertion,
        "actual_load_transducer_min": transducer,
    }


def per_event_minimums(
    external: np.ndarray,
    vectors: np.ndarray,
    considered: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active_values = []
    total_values = []
    for index in selected_indices:
        source = vectors[index, :4]
        reflected = external @ source
        active = considered[index, :4]
        rl = -20.0 * np.log10(
            np.maximum(np.abs(reflected[active]), EPS) / np.maximum(np.abs(source[active]), EPS)
        )
        active_values.append(float(np.min(rl)))
        total_values.append(
            -10.0
            * math.log10(
                max(float(np.sum(np.abs(reflected) ** 2)), EPS)
                / max(float(np.sum(np.abs(source) ** 2)), EPS)
            )
        )
    return np.asarray(active_values), np.asarray(total_values)


def compare_old(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest = read_json(out / "case_manifest.json")
    touchstone = Path(manifest["touchstone_path"])
    _, current_matrices = reordered_network(touchstone, manifest["expected_port_order"], 8)
    current = current_matrices[0]
    old_path = resolve(config["upstream"]["frozen_run06_s8_10ghz"])
    _, old_matrices = reordered_network(old_path, manifest["expected_port_order"], 8)
    old = old_matrices[0]
    frequency = float(config["frequency_ghz"])
    antenna_f, antenna_matrices = parse_touchstone(resolve(config["upstream"]["trusted_antenna_s4"]), 4)
    antenna = antenna_matrices[int(np.argmin(np.abs(antenna_f - frequency)))]
    stimulus_rows, vectors, considered = load_stimuli(resolve(config["upstream"]["trusted_stimulus_root"]))
    selected = np.asarray(
        [int(row["side"]) == 2 and abs(float(row["frequency_ghz"]) - frequency) <= 1.0e-9 for row in stimulus_rows]
    )
    selected_indices = np.flatnonzero(selected)
    sources = vectors[selected, :4].T
    active = considered[selected, :4].T
    operator = np.load(resolve(config["upstream"]["selected_circuit_operator"]), allow_pickle=False)
    operator_index = int(np.argmin(np.abs(operator["frequencies_ghz"] - frequency)))
    target = cascade_s(operator["launch_s8"][operator_index], operator["correction_s8"][operator_index])
    transform = operator["modal_transform"].astype(complex)
    transform8 = np.zeros((8, 8), dtype=complex)
    transform8[:4, :4] = transform
    transform8[4:, 4:] = transform
    modal_current = transform8 @ current @ transform8.T
    modal_old = transform8 @ old @ transform8.T
    modal_target = transform8 @ target @ transform8.T
    rows = []
    for name, network in (("frozen_run06", old), ("physical_v123", current), ("circuit_target", target)):
        rows.append({"network": name, **network_metrics(network, antenna, sources, active)})
    write_csv(touchstone.parent / "paired_network_metrics.csv", rows)
    old_external, _, _ = terminate_network(old, antenna)
    current_external, _, _ = terminate_network(current, antenna)
    old_active, old_total = per_event_minimums(old_external, vectors, considered, selected_indices)
    new_active, new_total = per_event_minimums(current_external, vectors, considered, selected_indices)
    event_rows = []
    for position, index in enumerate(selected_indices):
        source_row = stimulus_rows[index]
        event_rows.append(
            {
                "stimulus_index": source_row.get("stimulus_index"),
                "sample_index": source_row.get("sample_index"),
                "K": source_row.get("k_value"),
                "ratio": source_row.get("ratio"),
                "state_name": source_row.get("state_name"),
                "source_type": source_row.get("source_type"),
                "old_active_rl_db": float(old_active[position]),
                "new_active_rl_db": float(new_active[position]),
                "new_minus_old_active_rl_db": float(new_active[position] - old_active[position]),
                "old_total_rl_db": float(old_total[position]),
                "new_total_rl_db": float(new_total[position]),
                "new_minus_old_total_rl_db": float(new_total[position] - old_total[position]),
            }
        )
    write_csv(touchstone.parent / "paired_stimulus_comparison.csv", event_rows)
    partition_names = ("S11", "S12", "S21", "S22")
    partition_rows = []
    for name, matrix in (("physical_minus_run06", current - old), ("physical_minus_target", current - target)):
        for partition, block in zip(partition_names, (matrix[:4, :4], matrix[:4, 4:], matrix[4:, :4], matrix[4:, 4:])):
            partition_rows.append(
                {
                    "comparison": name,
                    "partition": partition,
                    "max_abs_delta_s": float(np.max(np.abs(block))),
                    "frobenius_energy": float(np.sum(np.abs(block) ** 2)),
                }
            )
    write_csv(touchstone.parent / "s8_partition_residuals.csv", partition_rows)
    summary = {
        "physical_minus_run06_max_abs_delta_s": float(np.max(np.abs(current - old))),
        "physical_minus_target_max_abs_delta_s": float(np.max(np.abs(current - target))),
        "physical_minus_run06_modal_max_abs_delta_s": float(np.max(np.abs(modal_current - modal_old))),
        "physical_minus_target_modal_max_abs_delta_s": float(np.max(np.abs(modal_current - modal_target))),
        "physical_vs_run06_active_rl_change_db": rows[1]["active_rl_min_db"] - rows[0]["active_rl_min_db"],
        "physical_vs_run06_total_rl_change_db": rows[1]["total_rl_min_db"] - rows[0]["total_rl_min_db"],
        "physical_vs_run06_efficiency_change": rows[1]["matched_efficiency_min"] - rows[0]["matched_efficiency_min"],
        "physical_vs_run06_insertion_change": rows[1]["actual_load_insertion_min"] - rows[0]["actual_load_insertion_min"],
        "stimulus_count": len(event_rows),
        "events_improved_active_rl_count": int(np.sum(new_active > old_active)),
        "events_degraded_active_rl_count": int(np.sum(new_active < old_active)),
        "median_active_rl_change_db": float(np.median(new_active - old_active)),
        "conclusion": "The fixed finite-Q component mapping is diagnostic only; no optimization or later HFSS stage is authorized.",
    }
    write_json(touchstone.parent / "paired_comparison_summary.json", summary)
    return {"summary": summary, "network_rows": rows}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    manifest_path = out / "case_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    touchstone = Path(manifest["touchstone_path"]) if manifest else None
    return {
        "output_directory": str(out),
        "preregistered": (out / "preregistration.json").exists(),
        "project_exists": Path(manifest["project_path"]).exists() if manifest else False,
        "touchstone_exists": bool(touchstone and touchstone.exists() and touchstone.stat().st_size > 100),
        "analysis_exists": bool(touchstone and (touchstone.parent / "analysis.json").exists()),
        "free_memory_gib": memory_available_gb(),
        "aedt_processes": aedt_processes(),
        "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("preregister", "run-build-smoke", "audit-build-smoke", "run-solve", "analyze", "compare-old", "status"),
        default="status",
    )
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    actions = {
        "preregister": preregister,
        "run-build-smoke": run_build_smoke,
        "audit-build-smoke": audit_build_smoke,
        "run-solve": run_solve,
        "analyze": analyze,
        "compare-old": compare_old,
        "status": status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
