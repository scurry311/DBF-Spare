#!/usr/bin/env python3
"""Build and gate a physical 2x2 array of the frozen v1.38 differential element."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_v137_vertical_mesh_audit import segment_audit
from run_v114_small_cell_broadband_feed import (
    efficiency_from_csv,
    memory_available_gb,
    parse_touchstone,
    profile_metrics,
)
from run_v121_parametric_feed_post import aedt_processes, run_process_with_memory_guard
from run_v125_feedpoint_input_impedance import topology_warning_count, write_csv, write_json
from run_v128_true_balanced_dual_resonant import vp
from run_v130_fixed_reference_cps_transformer import read_json, resolve
from run_v132_vertical_differential_launch import helpers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v139_physical_2x2_differential_array_preregistered.json"
DESIGN_NAME = "V139_Physical2x2Differential"
EPS = 1.0e-15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ri_to_complex(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    return value[..., 0] + 1j * value[..., 1]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > EPS else vector


def unique_sorted(values: list[float]) -> list[float]:
    output: list[float] = []
    for value in sorted(values):
        if not output or abs(value - output[-1]) > 1.0e-9:
            output.append(value)
    return output


def substrate_partition_text(g: dict[str, Any], centers: list[tuple[str, float, float]]) -> str:
    h = float(g["substrate_thickness_mm"])
    half = float(g["via_radius_mm"])
    x_bounds = [-15.0, 15.0]
    y_bounds = [-15.0, 15.0]
    holes: list[tuple[float, float, float, float]] = []
    pitch = float(g["via_pair_pitch_mm"])
    for _, cx, cy in centers:
        for x in (cx - pitch / 2.0, cx + pitch / 2.0):
            holes.append((x - half, x + half, cy - half, cy + half))
            x_bounds.extend((x - half, x + half))
            y_bounds.extend((cy - half, cy + half))
    xs = unique_sorted(x_bounds)
    ys = unique_sorted(y_bounds)
    pieces: list[str] = []
    names: list[str] = []
    for ix, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        for iy, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if any(hx0 < mx < hx1 and hy0 < my < hy1 for hx0, hx1, hy0, hy1 in holes):
                continue
            name = f"SubstratePart_{ix:02d}_{iy:02d}"
            names.append(name)
            pieces.append(
                f'CreateBox oEditor, "{name}", {x0:.7f}, {y0:.7f}, {-h:.7f}, '
                f'{x1-x0:.7f}, {y1-y0:.7f}, {h:.7f}, "RO5880_V139", True'
            )
    # Keep the deterministic same-material partitions as separate touching bodies.
    # A single many-body Parasolid union proved non-regenerable in the first v1.39
    # attempt; the partitions fill one continuous board while avoiding that history.
    return "\n".join(pieces)


def element_text(g: dict[str, Any], suffix: str, cx: float, cy: float) -> tuple[str, list[str]]:
    h = float(g["substrate_thickness_mm"])
    gap = float(g["primary_inner_gap_mm"])
    primary_l = float(g["primary_arm_length_mm"])
    primary_w = float(g["primary_arm_width_mm"])
    secondary_l = float(g["secondary_arm_length_mm"])
    secondary_w = float(g["secondary_arm_width_mm"])
    secondary_y = float(g["secondary_arm_offset_y_mm"])
    neck_l = float(g["secondary_neck_length_x_mm"])
    overlap = float(g["secondary_neck_overlap_mm"])
    pitch = float(g["via_pair_pitch_mm"])
    half = float(g["via_radius_mm"])
    pad_w = float(g["bottom_pad_width_mm"])
    pad_y = float(g["bottom_pad_length_y_mm"])
    margin = float(g["port_sheet_margin_mm"])
    bottom = -h
    primary_y = cy - primary_w / 2.0
    secondary_bottom = cy + secondary_y - secondary_w / 2.0
    neck_height = secondary_bottom - primary_y + overlap
    xn, xp = cx - pitch / 2.0, cx + pitch / 2.0
    pad_gap = pitch - pad_w
    names = [
        f"PrimaryN_{suffix}", f"PrimaryP_{suffix}", f"ViaN_{suffix}", f"ViaP_{suffix}",
        f"PadN_{suffix}", f"PadP_{suffix}",
    ]
    text = f'''
' Frozen v1.38 radiator {suffix} at ({cx:.3f},{cy:.3f}) mm.
CreateMetalSheetZ oEditor, "PrimaryN_{suffix}", {cx-gap/2-primary_l:.7f}, {primary_y:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}
CreateMetalSheetZ oEditor, "SecondaryN_{suffix}", {cx-gap/2-secondary_l:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}
CreateMetalSheetZ oEditor, "NeckN_{suffix}", {cx-gap/2-neck_l:.7f}, {primary_y:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}
UniteSelection oEditor, "PrimaryN_{suffix},SecondaryN_{suffix},NeckN_{suffix}"
CreateMetalSheetZ oEditor, "PrimaryP_{suffix}", {cx+gap/2:.7f}, {primary_y:.7f}, 0, {primary_l:.7f}, {primary_w:.7f}
CreateMetalSheetZ oEditor, "SecondaryP_{suffix}", {cx+gap/2:.7f}, {secondary_bottom:.7f}, 0, {secondary_l:.7f}, {secondary_w:.7f}
CreateMetalSheetZ oEditor, "NeckP_{suffix}", {cx+gap/2:.7f}, {primary_y:.7f}, 0, {neck_l:.7f}, {neck_height:.7f}
UniteSelection oEditor, "PrimaryP_{suffix},SecondaryP_{suffix},NeckP_{suffix}"
CreateBox oEditor, "ViaN_{suffix}", {xn-half:.7f}, {cy-half:.7f}, {bottom:.7f}, {2*half:.7f}, {2*half:.7f}, {h:.7f}, "copper", False
CreateBox oEditor, "ViaP_{suffix}", {xp-half:.7f}, {cy-half:.7f}, {bottom:.7f}, {2*half:.7f}, {2*half:.7f}, {h:.7f}, "copper", False
CreateMetalSheetZ oEditor, "PadN_{suffix}", {xn-pad_w/2:.7f}, {cy-pad_y/2:.7f}, {bottom:.7f}, {pad_w:.7f}, {pad_y:.7f}
CreateMetalSheetZ oEditor, "PadP_{suffix}", {xp-pad_w/2:.7f}, {cy-pad_y/2:.7f}, {bottom:.7f}, {pad_w:.7f}, {pad_y:.7f}
CreateSheetZ oEditor, "PortSheet_{suffix}", {cx-pad_gap/2-margin:.7f}, {cy-pad_y/2:.7f}, {bottom:.7f}, {pad_gap+2*margin:.7f}, {pad_y:.7f}
AssignDifferentialPortZ oBoundary, "{suffix}", "PortSheet_{suffix}", {cx-pad_gap/2:.7f}, {cx+pad_gap/2:.7f}, {cy:.7f}, {bottom:.7f}
'''
    return text, names


def solver_suffix(solver_type: str) -> str:
    options = {
        "direct": ', "DrivenSolverType:=", "Direct Solver"',
        "ddm": ', "DrivenSolverType:=", "Domain Decomposition", "IterativeResidual:=", 0.000001, "DDMSolverResidual:=", 0.000001',
    }
    if solver_type not in options:
        raise ValueError(f"Unsupported solver type: {solver_type}")
    return options[solver_type]


def builder_text(project: Path, g: dict[str, Any], frequency_ghz: float, solver_type: str) -> str:
    centers = [("P00", -7.5, -7.5), ("P10", 7.5, -7.5), ("P01", -7.5, 7.5), ("P11", 7.5, 7.5)]
    substrate = substrate_partition_text(g, centers)
    elements: list[str] = []
    conductors: list[str] = []
    for suffix, x, y in centers:
        source, names = element_text(g, suffix, x, y)
        elements.append(source)
        conductors.extend(names)
    finite_sheets = [name for name in conductors if name.startswith("Primary") or name.startswith("Pad")]
    # Match the trusted v1.38 mesh scope exactly: the 0.10 mm operation belongs
    # on the primary radiator sheets, not on POST/pad volumes.
    mesh_names = [name for name in conductors if name.startswith("Primary")]
    mesh_objects = ", ".join(f'"{name}"' for name in mesh_names)
    sheet_objects = ", ".join(f'"{name}"' for name in finite_sheets)
    h = float(g["substrate_thickness_mm"])
    copper = float(g["copper_thickness_mm"])
    mesh = float(g["local_mesh_max_length_mm"])
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V139", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(g['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(g['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
{substrate}
{''.join(elements)}
oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", "Objects:=", Array({sheet_objects}), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)
CreateBox oEditor, "AirRegion", -27, -27, {-h-12:.7f}, 54, 54, {h+24.035:.7f}, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:UnifiedFeedRadiatorMesh_0p100mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({mesh_objects}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency_ghz:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {int(g['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {float(g['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False{solver_suffix(solver_type)})
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V139", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{helpers()}
'''


def calibration_states() -> list[tuple[str, np.ndarray]]:
    states: list[tuple[str, np.ndarray]] = []
    for i in range(4):
        vector = np.zeros(4, dtype=complex)
        vector[i] = 1.0
        states.append((f"basis_{i}", vector))
    for i in range(4):
        for j in range(i + 1, 4):
            vector = np.zeros(4, dtype=complex)
            vector[i] = vector[j] = 1.0
            states.append((f"pair_re_{i}_{j}", vector))
            vector = np.zeros(4, dtype=complex)
            vector[i], vector[j] = 1.0, 1j
            states.append((f"pair_im_{i}_{j}", vector))
    return states


def solver_text(project: Path, touchstone: Path, folder: Path, frequency_ghz: float) -> str:
    blocks: list[str] = []
    for name, vector in calibration_states():
        magnitudes = [f'"{abs(value) ** 2:.12g}W"' for value in vector]
        phases = [f'"{math.degrees(np.angle(value)):.12g}deg"' for value in vector]
        blocks.append(
            f'ApplyCalibration oSolutions, ports, Array({",".join(magnitudes)}), Array({",".join(phases)})\n'
            f'ExportEfficiency oReport, "{name}", "{vp(folder / ("efficiency_" + name + ".csv"))}", "{frequency_ghz:g}GHz"'
        )
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, ports
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
ports = Array("P00", "P10", "P01", "P11")
{chr(10).join(blocks)}
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ApplyCalibration(solModule, portArray, magnitudeArray, phaseArray)
    Dim sources, editArgs(), i, j, sourceName, portName, magnitude, phase
    sources = solModule.GetAllSources()
    ReDim editArgs(UBound(sources) + 1)
    editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)
    For i = LBound(sources) To UBound(sources)
        sourceName = CStr(sources(i))
        magnitude = "0W"
        phase = "0deg"
        For j = LBound(portArray) To UBound(portArray)
            portName = CStr(portArray(j))
            If LCase(Split(sourceName, ":")(0)) = LCase(portName) Then
                magnitude = CStr(magnitudeArray(j))
                phase = CStr(phaseArray(j))
            End If
        Next
        editArgs(i + 1) = Array("Name:=", sourceName, "Magnitude:=", magnitude, "Phase:=", phase)
    Next
    solModule.EditSources editArgs
End Sub
Sub ExportEfficiency(reportModule, stateName, outputPath, frequencyValue)
    Dim reportName
    reportName = "V139_Eff_" & stateName
    On Error Resume Next
    reportModule.DeleteReports Array(reportName)
    Err.Clear
    reportModule.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V139"), Array("Freq:=", Array(frequencyValue)), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
    If Err.Number = 0 Then reportModule.ExportToFile reportName, outputPath
    On Error GoTo 0
End Sub
'''


def prepare_case(root: Path, case_id: str, config: dict[str, Any], frequency: float, solver_type: str) -> dict[str, Any]:
    folder = root / case_id
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite v1.39 case: {folder}")
    folder.mkdir(parents=True)
    project = folder / f"v139_{case_id}.aedt"
    touchstone = folder / f"v139_{case_id}.s4p"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    source = builder_text(project, config["frozen_geometry"], frequency, solver_type)
    builder.write_text(source, encoding="ascii")
    solver.write_text(solver_text(project, touchstone, folder, frequency), encoding="ascii")
    case = {
        "case_id": case_id,
        "frequency_ghz": frequency,
        "solver_type": solver_type,
        "geometry": dict(config["frozen_geometry"]),
        "numerical_amendment": config.get("active_numerical_amendment"),
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
        "builder_sha256": sha256(builder),
        "solver_sha256": sha256(solver),
        "cad_audit": {
            "differential_port_definition_count": source.count("AssignDifferentialPortZ oBoundary"),
            "port_order": config["port_order"],
            "reference_ground_count": source.count('"ReferenceGround"') + source.count('"Ground"'),
            "finite_conductivity_sheet": "AssignFiniteCond" in source,
            "partitioned_continuous_substrate": source.count("SubstratePart_") >= 13,
            "solver_option_count": source.count("DrivenSolverType:="),
            "trusted_local_mesh_object_count": source.split("UnifiedFeedRadiatorMesh_0p100mm", 1)[1].split("RestrictElem:=", 1)[0].count("Primary"),
        },
    }
    write_json(folder / "case_manifest.json", case)
    return case


def choose_window(mask: np.ndarray, combined: np.ndarray, role: str) -> tuple[int, int, float]:
    options: list[tuple[int, int]] = []
    for x0 in range(15):
        for y0 in range(15):
            on_x = x0 in (0, 14)
            on_y = y0 in (0, 14)
            if role == "corner" and on_x and on_y:
                options.append((x0, y0))
            elif role == "edge" and on_x != on_y:
                options.append((x0, y0))
            elif role == "interior" and not on_x and not on_y:
                options.append((x0, y0))
    best = max(
        options,
        key=lambda xy: float(sum(abs(combined[(xy[0] + ix) * 16 + xy[1] + iy]) ** 2 for ix in range(2) for iy in range(2))),
    )
    score = float(sum(abs(combined[(best[0] + ix) * 16 + best[1] + iy]) ** 2 for ix in range(2) for iy in range(2)))
    return best[0], best[1], score


def prepare_stimuli(config: dict[str, Any], out: Path) -> dict[str, Any]:
    old_csv = resolve(config["inputs"]["old_stimuli_csv"])
    old_npz = resolve(config["inputs"]["old_stimuli_npz"])
    for key, path in (("old_stimuli_csv", old_csv), ("old_stimuli_npz", old_npz)):
        actual = sha256(path)
        if actual != config["locked_input_sha256"][key]:
            raise RuntimeError(f"Frozen input hash mismatch for {key}: {actual}")
    old_rows = list(csv.DictReader(old_csv.open(encoding="utf-8-sig")))
    with np.load(old_npz, allow_pickle=False) as data:
        old_vectors = ri_to_complex(data["vectors_real_imag"])
        old_considered = np.asarray(data["considered"], dtype=bool)
    permutation = np.asarray(config["old_local_to_physical_permutation"], dtype=int)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    considered: list[np.ndarray] = []
    for old_index, row in enumerate(old_rows):
        if int(row["side"]) != 2:
            continue
        vector = normalize_vector(old_vectors[old_index, :4][permutation])
        active = old_considered[old_index, :4][permutation]
        rows.append({**row, "stimulus_family": "old_285_replay", "source_row": old_index, "physical_port_order": "P00|P10|P01|P11"})
        vectors.append(vector)
        considered.append(active)

    supported = [row for row in csv.DictReader(resolve(config["inputs"]["supported_scene_list"]).open(encoding="utf-8-sig")) if int(row["k_value"]) == 6]
    manifest_rows = list(csv.DictReader(resolve(config["inputs"]["v11_candidate_manifest"]).open(encoding="utf-8-sig")))
    manifest_by_candidate = {int(row["candidate_index"]): row for row in manifest_rows}
    with np.load(resolve(config["inputs"]["v11_dataset_arrays"]), allow_pickle=False) as data:
        candidate_ids = np.asarray(data["candidate_index"], dtype=int)
        row_by_candidate = {int(value): index for index, value in enumerate(candidate_ids)}
        masks = np.asarray(data["masks"], dtype=bool)
        combined_all = ri_to_complex(data["hfss_actual_combined_weights_real_imag"])
        tasks_all = ri_to_complex(data["hfss_actual_task_weights_real_imag"])
    floor = float(config["stimuli"]["active_amplitude_floor_fraction"])
    k6_count = 0
    for scene in supported:
        candidate = int(scene["best_candidate_index"])
        index = row_by_candidate[candidate]
        manifest = manifest_by_candidate[candidate]
        if int(manifest["sample_index"]) != int(scene["sample_index"]) or int(manifest["k_value"]) != 6:
            raise RuntimeError(f"K6 candidate audit failed for {candidate}")
        mask = masks[index]
        combined = combined_all[index] * mask
        tasks = tasks_all[index] * mask[:, None]
        for role in config["stimuli"]["k6_window_roles"]:
            x0, y0, score = choose_window(mask, combined, role)
            old_indices = np.asarray([(x0 + ix) * 16 + y0 + iy for ix in range(2) for iy in range(2)], dtype=int)
            physical_indices = old_indices[permutation]
            source_vectors = [("combined", -1, combined[physical_indices])]
            source_vectors.extend(("task", task, tasks[physical_indices, task]) for task in range(6))
            for source_type, task_index, local in source_vectors:
                vector = normalize_vector(local)
                if np.linalg.norm(vector) <= EPS:
                    continue
                active = np.abs(vector) >= floor * float(np.max(np.abs(vector)))
                for frequency in config["frequencies_ghz"]:
                    rows.append({
                        "stimulus_index": len(rows), "side": 2, "window_role": role,
                        "window_x0": x0, "window_y0": y0, "window_score": score,
                        "sample_index": int(scene["sample_index"]), "scene_class": "supported_k6",
                        "k_value": 6, "ratio": float(scene["best_candidate_ratio"]),
                        "frequency_ghz": float(frequency), "state_name": "frozen_v11_actual_weight",
                        "source_type": source_type, "task_index": task_index,
                        "local_active_count": int(np.sum(active)), "stimulus_family": "k6_supported_scene",
                        "source_row": candidate, "physical_port_order": "P00|P10|P01|P11",
                    })
                    vectors.append(vector)
                    considered.append(active)
                    k6_count += 1

    modes = {
        "mode_even": np.array([1, 1, 1, 1], complex),
        "mode_x_odd": np.array([1, -1, 1, -1], complex),
        "mode_y_odd": np.array([1, 1, -1, -1], complex),
        "mode_checker": np.array([1, -1, -1, 1], complex),
    }
    c = 299792458.0
    positions = np.asarray([[-7.5, -7.5], [7.5, -7.5], [-7.5, 7.5], [7.5, 7.5]]) * 1.0e-3
    for frequency in config["frequencies_ghz"]:
        entries = list(modes.items())
        for phi in config["stimuli"]["large_scan_phi_deg"]:
            theta = math.radians(float(config["stimuli"]["large_scan_theta_deg"]))
            phir = math.radians(float(phi))
            k = 2.0 * math.pi * float(frequency) * 1.0e9 / c
            phase = -k * math.sin(theta) * (positions[:, 0] * math.cos(phir) + positions[:, 1] * math.sin(phir))
            entries.append((f"scan_t48_p{int(phi)}", np.exp(1j * phase)))
        for name, raw in entries:
            vector = normalize_vector(raw)
            rows.append({
                "stimulus_index": len(rows), "side": 2, "window_role": "full_2x2", "window_x0": -1,
                "window_y0": -1, "window_score": 1.0, "sample_index": -1, "scene_class": "canonical",
                "k_value": 0, "ratio": 1.0, "frequency_ghz": float(frequency), "state_name": name,
                "source_type": "canonical", "task_index": -1, "local_active_count": 4,
                "stimulus_family": "canonical_modal_scan", "source_row": -1,
                "physical_port_order": "P00|P10|P01|P11",
            })
            vectors.append(vector)
            considered.append(np.ones(4, dtype=bool))

    stimulus_dir = out / "stimuli"
    stimulus_dir.mkdir(parents=True)
    for index, row in enumerate(rows):
        row["stimulus_index"] = index
    write_csv(stimulus_dir / "stimuli_manifest.csv", rows)
    array = np.asarray(vectors, dtype=np.complex64)
    np.savez_compressed(
        stimulus_dir / "stimuli_vectors.npz",
        vectors_real_imag=np.stack((array.real, array.imag), axis=-1),
        considered=np.asarray(considered, dtype=np.int8),
    )
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['stimulus_family']}|K{row['k_value']}|f{float(row['frequency_ghz']):.2f}"
        counts[key] = counts.get(key, 0) + 1
    audit = {
        "total_count": len(rows), "old_replay_count": sum(row["stimulus_family"] == "old_285_replay" for row in rows),
        "k6_supported_scene_count": len(supported), "k6_stimulus_count": k6_count,
        "source_hashes": {key: sha256(resolve(path)) for key, path in config["inputs"].items()},
        "port_order": config["port_order"], "old_to_physical_permutation": permutation.tolist(), "counts": counts,
    }
    write_json(stimulus_dir / "stimuli_audit.json", audit)
    return audit


def frozen_parent(config: dict[str, Any]) -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "rev-list", "-n", "1", config["parent_tag"]], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if head != config["parent_commit"] or tag != config["parent_commit"]:
        raise RuntimeError(f"Frozen parent mismatch: HEAD={head}, tag={tag}")
    prior = read_json(config["inputs"]["v138_stage_decision"])
    if not prior.get("three_frequency_efficiency_gate_pass") or not prior.get("allow_2x2"):
        raise RuntimeError("v1.38 does not authorize physical 2x2")
    return {"head_commit": head, "tag_commit": tag}


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.39 output: {out}")
    out.mkdir(parents=True)
    parent = frozen_parent(config)
    stimuli = prepare_stimuli(config, out)
    root = out / "initial_10ghz"
    case = prepare_case(root, "direct01", config, float(config["frequency_ghz"]), "direct")
    write_json(root / "case_manifest.json", {"cases": [case]})
    write_json(out / "preregistration.json", {
        **config, "runtime_audit": {**parent, "free_memory_gib": memory_available_gb(), "aedt_processes": aedt_processes(), "runner_sha256": sha256(Path(__file__))},
        "stimuli_audit": stimuli,
        "evidence_rules": {"single_initial_direct_case": True, "new_continuous_2x2_substrate": True, "four_true_differential_ports": True, "old_stimuli_port_order_reaudited": True, "k6_uses_existing_v11_weights": True, "larger_array_eep_labels_critic_locked": True},
    })
    decision = {"stage": "A_preregistered", "allow_build_smoke": True, "allow_initial_solve": False, "allow_crosscheck": False, "allow_three_frequency": False, "allow_4x4": False, "allow_16x16": False, "allow_eep_export": False, "allow_training_labels": False, "allow_critic_training": False}
    write_json(out / "stage_decision.json", decision)
    return {"output_directory": str(out), "case": case, "stimuli": stimuli, "decision": decision}


def require_no_aedt() -> None:
    processes = aedt_processes()
    if processes:
        raise RuntimeError(f"Refusing concurrent AEDT/HFSS execution: {processes}")


def build_case(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    require_no_aedt()
    folder = Path(case["project_path"]).parent
    log = folder / "build.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run([str(resolve(config["ansys_executable"])), "-RunScriptAndExit", case["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    cad = case["cad_audit"]
    passed = bool(result.returncode == 0 and Path(case["project_path"]).exists() and cad["differential_port_definition_count"] == 4 and cad["reference_ground_count"] == 0 and cad["finite_conductivity_sheet"] and cad["partitioned_continuous_substrate"] and cad["solver_option_count"] == 1 and topology_warning_count(folder) == 0)
    audit = {"case_id": case["case_id"], "return_code": result.returncode, "project_exists": Path(case["project_path"]).exists(), "topology_warning_count": topology_warning_count(folder), "cad_audit": cad, "build_gate_pass": passed}
    write_json(folder / "build_audit.json", audit)
    if not passed:
        raise RuntimeError(f"Build smoke failed: {audit}")
    return audit


def build_smoke(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    decision = read_json(out / "stage_decision.json")
    if not decision.get("allow_build_smoke"):
        raise RuntimeError("Build smoke is not authorized")
    case = read_json(out / "initial_10ghz" / "case_manifest.json")["cases"][0]
    audit = build_case(config, case)
    decision.update({"stage": "B_build_smoke_complete", "allow_build_smoke": False, "allow_initial_solve": True})
    write_json(out / "stage_decision.json", decision)
    return {"audit": audit, "decision": decision}


def wait_for_memory(config: dict[str, Any]) -> float:
    required = float(config["resources"]["minimum_free_memory_before_2x2_gib"])
    deadline = time.time() + float(config["resources"]["memory_recovery_wait_seconds"])
    while True:
        require_no_aedt()
        free = memory_available_gb()
        if free >= required:
            return free
        if time.time() >= deadline:
            raise MemoryError(f"Memory did not recover to {required:.2f} GiB; current {free:.2f} GiB")
        time.sleep(float(config["resources"]["poll_interval_seconds"]))


def solve_case(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    project = Path(case["project_path"])
    if not project.exists():
        build_case(config, case)
    free = wait_for_memory(config)
    folder = project.parent
    code, aborted, minimum = run_process_with_memory_guard(
        [str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", case["solver_path"]],
        folder / "solve_export.log", float(config["resources"]["abort_free_memory_during_solve_gib"]), float(config["resources"]["poll_interval_seconds"]),
    )
    touchstone = Path(case["touchstone_path"])
    row = {"case_id": case["case_id"], "solve_return_code": code, "memory_aborted": aborted, "free_memory_gib_before": free, "minimum_free_memory_gib": minimum, "touchstone_exists": touchstone.exists() and touchstone.stat().st_size > 100}
    write_csv(folder / "run_progress.csv", [row])
    if code != 0 or aborted or not row["touchstone_exists"]:
        raise RuntimeError(f"Solve failed: {row}")
    return row


def run_initial(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_initial_solve"):
        raise RuntimeError("Initial solve is not authorized")
    case = read_json(out / "initial_10ghz" / "case_manifest.json")["cases"][-1]
    return solve_case(config, case)


def prepare_initial_repair(config: dict[str, Any]) -> dict[str, Any]:
    """Preserve a failed initial case and prepare a new non-overwriting CAD repair."""
    out = resolve(config["output_directory"])
    root = out / "initial_10ghz"
    manifest_path = root / "case_manifest.json"
    manifest = read_json(manifest_path)
    previous = manifest["cases"][-1]
    progress = Path(previous["project_path"]).parent / "run_progress.csv"
    if not progress.exists() or "False" not in progress.read_text(encoding="utf-8-sig"):
        raise RuntimeError("No preserved failed initial solve authorizes a repair case")
    case_id = f"direct01_repair{len(manifest['cases']):02d}"
    case = prepare_case(root, case_id, config, float(config["frequency_ghz"]), "direct")
    manifest["cases"].append(case)
    write_json(manifest_path, manifest)
    audit = build_case(config, case)
    decision = read_json(out / "stage_decision.json")
    decision.update({"stage": "B2_initial_cad_repair_built", "allow_initial_solve": True, "failed_case_preserved": previous["case_id"], "active_case_id": case_id})
    write_json(out / "stage_decision.json", decision)
    return {"case": case, "build_audit": audit, "decision": decision}


def load_stimuli(out: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((out / "stimuli" / "stimuli_manifest.csv").open(encoding="utf-8-sig")))
    with np.load(out / "stimuli" / "stimuli_vectors.npz", allow_pickle=False) as data:
        vectors = ri_to_complex(data["vectors_real_imag"])
        considered = np.asarray(data["considered"], dtype=bool)
    if len(rows) != vectors.shape[0] or vectors.shape != considered.shape:
        raise RuntimeError("Stimulus manifest/vector alignment failure")
    return rows, vectors, considered


def radiation_matrix(folder: Path, s: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values: dict[str, float] = {}
    audit: list[dict[str, Any]] = []
    for name, vector in calibration_states():
        efficiency = efficiency_from_csv(folder / f"efficiency_{name}.csv")
        if efficiency is None:
            raise RuntimeError(f"Missing HFSS efficiency calibration: {name}")
        incident = float(np.vdot(vector, vector).real)
        reflected = float(np.vdot(s @ vector, s @ vector).real)
        accepted = incident - reflected
        radiated = efficiency * accepted
        values[name] = radiated
        audit.append({"state": name, "incident_power_w": incident, "reflected_power_w": reflected, "accepted_power_w": accepted, "hfss_radiation_efficiency": efficiency, "radiated_power_w": radiated, "system_efficiency": radiated / incident})
    matrix = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        matrix[i, i] = values[f"basis_{i}"]
    for i in range(4):
        for j in range(i + 1, 4):
            real = (values[f"pair_re_{i}_{j}"] - matrix[i, i].real - matrix[j, j].real) / 2.0
            imag = -(values[f"pair_im_{i}_{j}"] - matrix[i, i].real - matrix[j, j].real) / 2.0
            matrix[i, j] = real + 1j * imag
            matrix[j, i] = np.conj(matrix[i, j])
    return matrix, audit


def touchstone_port_names(path: Path) -> list[str]:
    names: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        lower = line.lower()
        if "port[" not in lower or "=" not in line:
            continue
        try:
            left, right = line.split("=", 1)
            index = int(left.lower().split("port[", 1)[1].split("]", 1)[0])
            names.append((index, right.strip().split(":", 1)[0].strip()))
        except (ValueError, IndexError):
            continue
    return [name for _, name in sorted(names)]


def active_rows(out: Path, s: np.ndarray, radiation: np.ndarray, frequency: float) -> list[dict[str, Any]]:
    metadata, vectors, considered = load_stimuli(out)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(metadata):
        if abs(float(row["frequency_ghz"]) - frequency) > 1.0e-6:
            continue
        source = vectors[index]
        active = considered[index]
        reflected_vector = s @ source
        gamma = np.abs(reflected_vector[active]) / np.maximum(np.abs(source[active]), EPS)
        incident = float(np.vdot(source, source).real)
        reflected = float(np.vdot(reflected_vector, reflected_vector).real)
        accepted = incident - reflected
        radiated = float(np.vdot(source, radiation @ source).real)
        rows.append({
            **row, "active_rl_db": float(-20.0 * np.log10(max(float(np.max(gamma)), EPS))),
            "total_rl_db": float(-10.0 * np.log10(max(reflected / incident, EPS))),
            "incident_power_w": incident, "reflected_power_w": reflected, "accepted_power_w": accepted,
            "radiated_power_w": radiated, "radiation_efficiency": radiated / max(accepted, EPS),
            "system_efficiency": radiated / max(incident, EPS),
        })
    return rows


def grouped_active_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["stimulus_family"]), int(row["k_value"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (family, k), values in sorted(groups.items()):
        active = np.asarray([float(row["active_rl_db"]) for row in values])
        total = np.asarray([float(row["total_rl_db"]) for row in values])
        efficiency = np.asarray([float(row["system_efficiency"]) for row in values])
        output.append({"stimulus_family": family, "k_value": k, "count": len(values), "minimum_active_rl_db": float(active.min()), "q05_active_rl_db": float(np.quantile(active, 0.05)), "median_active_rl_db": float(np.median(active)), "minimum_total_rl_db": float(total.min()), "minimum_system_efficiency": float(efficiency.min()), "active_rl_ge_10_count": int(np.sum(active >= 10.0)), "active_rl_ge_11_count": int(np.sum(active >= 11.0))})
    return output


def analyze_case(config: dict[str, Any], case: dict[str, Any], write_outputs: bool = True) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    folder = Path(case["project_path"]).parent
    frequencies, matrices = parse_touchstone(Path(case["touchstone_path"]), 4)
    index = int(np.argmin(np.abs(frequencies - float(case["frequency_ghz"]))))
    s = matrices[index]
    expected_order = list(config["port_order"])
    exported_order = touchstone_port_names(Path(case["touchstone_path"]))
    port_order_pass = not exported_order or exported_order == expected_order
    if exported_order and sorted(exported_order) == sorted(expected_order) and exported_order != expected_order:
        permutation = [exported_order.index(name) for name in expected_order]
        s = s[np.ix_(permutation, permutation)]
        port_order_pass = True
    elif exported_order and not port_order_pass:
        raise RuntimeError(f"Unexpected Touchstone port names: {exported_order}")
    radiation, calibration = radiation_matrix(folder, s)
    active = active_rows(out, s, radiation, float(case["frequency_ghz"]))
    grouped = grouped_active_summary(active)
    profile = profile_metrics(folder)
    bodies, lengths = segment_audit(folder)
    conductor_names = {f"{base}_{port}" for port in expected_order for base in ("PrimaryN", "PrimaryP", "PadN", "PadP", "ViaN", "ViaP")}
    conductor_count = sum(value for name, value in bodies.items() if name in conductor_names)
    allowed = set(config["gates"]["allowed_residual_small_segment_bodies"])
    dielectric_partition_names = {name for name in bodies if name.startswith("SubstratePart_")}
    unexpected = sorted(set(bodies) - conductor_names - allowed - dielectric_partition_names)
    passive_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(s)), EPS))
    reciprocity = float(np.max(np.abs(s - s.T)))
    passivity = float(np.linalg.svd(s, compute_uv=False)[0])
    coupling_rows = []
    pair_groups = {"x": [(0, 1), (2, 3)], "y": [(0, 2), (1, 3)], "diagonal": [(0, 3), (1, 2)]}
    for group, pairs in pair_groups.items():
        for i, j in pairs:
            coupling_rows.append({"group": group, "port_i": expected_order[i], "port_j": expected_order[j], "sij_real": float(s[i, j].real), "sij_imag": float(s[i, j].imag), "coupling_db": float(20.0 * np.log10(max(abs(s[i, j]), EPS)))})
    modal_rows = []
    modes = {"even": [1, 1, 1, 1], "x_odd": [1, -1, 1, -1], "y_odd": [1, 1, -1, -1], "checker": [1, -1, -1, 1]}
    for name, raw in modes.items():
        vector = normalize_vector(np.asarray(raw, complex))
        gamma = complex(np.vdot(vector, s @ vector))
        impedance = 50.0 * (1.0 + gamma) / (1.0 - gamma)
        modal_rows.append({"mode": name, "gamma_real": gamma.real, "gamma_imag": gamma.imag, "rl_db": float(-20.0 * np.log10(max(abs(gamma), EPS))), "resistance_ohm": impedance.real, "reactance_ohm": impedance.imag})
    gates = config["gates"]
    min_active = min(float(row["active_rl_db"]) for row in active)
    min_total = min(float(row["total_rl_db"]) for row in active)
    min_system_eff = min(float(row["system_efficiency"]) for row in active)
    numerical_gate = bool(profile.get("converged") is True and float(profile.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"]) and reciprocity <= float(gates["maximum_reciprocity_error"]) and passivity <= float(gates["maximum_passivity_sigma"]) and float(passive_rl.min()) >= float(gates["minimum_passive_rl_db"]) and conductor_count <= int(gates["maximum_conductor_small_segment_count"]) and not unexpected and topology_warning_count(folder) <= int(gates["maximum_port_topology_warning_count"]) and port_order_pass)
    stop_line = min_active >= float(gates["active_rl_stop_line_db"])
    strict = bool(numerical_gate and stop_line and min_active >= float(gates["minimum_active_rl_design_db"]) and min_total >= float(gates["minimum_total_rl_db"]) and min_system_eff >= float(gates["minimum_system_efficiency"]))
    summary = {"case_id": case["case_id"], "frequency_ghz": float(frequencies[index]), **profile, "exported_port_order": exported_order, "effective_port_order": expected_order, "port_order_gate_pass": port_order_pass, "maximum_reciprocity_error": reciprocity, "maximum_passivity_sigma": passivity, "minimum_passive_rl_db": float(passive_rl.min()), "minimum_active_rl_db": min_active, "minimum_total_rl_db": min_total, "minimum_system_efficiency": min_system_eff, "conductor_small_segment_message_count": int(conductor_count), "residual_messages_confined_to_allowed_bodies": not unexpected, "unexpected_small_segment_bodies": unexpected, "minimum_segment_length_mm": min(lengths) if lengths else None, "topology_warning_count": topology_warning_count(folder), "numerical_physical_gate_pass": numerical_gate, "active_rl_stop_line_pass": stop_line, "strict_initial_gate_pass": strict, "stimulus_count": len(active)}
    if write_outputs:
        write_json(folder / "case_summary.json", summary)
        write_csv(folder / "active_metrics.csv", active)
        write_csv(folder / "active_group_summary.csv", grouped)
        write_csv(folder / "coupling_metrics.csv", coupling_rows)
        write_csv(folder / "modal_impedance.csv", modal_rows)
        write_csv(folder / "radiation_calibration.csv", calibration)
        np.savez_compressed(folder / "physical_operators.npz", s_real_imag=np.stack((s.real, s.imag), axis=-1), radiation_real_imag=np.stack((radiation.real, radiation.imag), axis=-1), port_order=np.asarray(expected_order))
    return {"summary": summary, "s": s, "active": active, "grouped": grouped}


def analyze_initial(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    case = read_json(out / "initial_10ghz" / "case_manifest.json")["cases"][-1]
    result = analyze_case(config, case)
    passed = result["summary"]["strict_initial_gate_pass"]
    stop = result["summary"]["active_rl_stop_line_pass"]
    decision = {"stage": "C_initial_10ghz_complete", "initial_strict_gate_pass": passed, "active_rl_stop_line_pass": stop, "allow_crosscheck": passed, "allow_three_frequency": False, "allow_4x4": False, "allow_16x16": False, "allow_eep_export": False, "allow_training_labels": False, "allow_critic_training": False, "reason": "Initial physical 2x2 passed all preregistered 10 GHz gates." if passed else ("Worst active RL is below 10 dB; stop solver expansion and diagnose coupling/modal sensitivity." if not stop else "A preregistered numerical, matching, total-RL, or efficiency gate failed; crosscheck remains locked.")}
    write_json(out / "stage_decision.json", decision)
    return {"summary": result["summary"], "grouped": result["grouped"], "decision": decision}


def prepare_crosscheck(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_crosscheck"):
        raise RuntimeError("Independent crosscheck is not authorized")
    root = out / "independent_crosscheck_10ghz"
    cases = [prepare_case(root, "direct02", config, 10.0, "direct"), prepare_case(root, "ddm01", config, 10.0, "ddm")]
    write_json(root / "case_manifest.json", {"cases": cases})
    decision = read_json(out / "stage_decision.json")
    decision.update({"stage": "D_crosscheck_prepared", "allow_crosscheck_run": True})
    write_json(out / "stage_decision.json", decision)
    return {"cases": cases, "decision": decision}


def run_crosscheck(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_crosscheck_run"):
        raise RuntimeError("Crosscheck run is not authorized")
    cases = read_json(out / "independent_crosscheck_10ghz" / "case_manifest.json")["cases"]
    return {case["case_id"]: solve_case(config, case) for case in cases}


def analyze_crosscheck(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    initial = read_json(out / "initial_10ghz" / "case_manifest.json")["cases"][0]
    repeats = read_json(out / "independent_crosscheck_10ghz" / "case_manifest.json")["cases"]
    results = [analyze_case(config, case) for case in [initial, *repeats]]
    comparisons = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            comparisons.append({"left": [initial, *repeats][i]["case_id"], "right": [initial, *repeats][j]["case_id"], "max_abs_delta_s": float(np.max(np.abs(results[i]["s"] - results[j]["s"])))})
    write_csv(out / "independent_crosscheck_10ghz" / "pairwise_s_comparison.csv", comparisons)
    maximum = max(row["max_abs_delta_s"] for row in comparisons)
    gate = bool(all(result["summary"]["strict_initial_gate_pass"] for result in results) and maximum <= float(config["gates"]["maximum_direct_ddm_abs_delta_s"]))
    summary = {"case_count": len(results), "maximum_pairwise_abs_delta_s": maximum, "preferred_delta_s_pass": maximum <= float(config["gates"]["preferred_direct_ddm_abs_delta_s"]), "crosscheck_gate_pass": gate}
    write_json(out / "independent_crosscheck_10ghz" / "crosscheck_summary.json", summary)
    decision = {"stage": "E_crosscheck_complete", "initial_strict_gate_pass": True, "crosscheck_gate_pass": gate, "allow_three_frequency": gate, "allow_4x4": False, "allow_16x16": False, "allow_eep_export": False, "allow_training_labels": False, "allow_critic_training": False}
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "comparisons": comparisons, "decision": decision}


def prepare_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_three_frequency"):
        raise RuntimeError("Three-frequency stage is not authorized")
    root = out / "three_frequency_independent"
    cases = [prepare_case(root, f"direct_f{float(f):.2f}".replace(".", "p"), config, float(f), "direct") for f in config["frequencies_ghz"]]
    write_json(root / "case_manifest.json", {"cases": cases})
    decision = read_json(out / "stage_decision.json")
    decision.update({"stage": "F_three_frequency_prepared", "allow_three_frequency_run": True})
    write_json(out / "stage_decision.json", decision)
    return {"cases": cases, "decision": decision}


def run_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    if not read_json(out / "stage_decision.json").get("allow_three_frequency_run"):
        raise RuntimeError("Three-frequency run is not authorized")
    cases = read_json(out / "three_frequency_independent" / "case_manifest.json")["cases"]
    return {case["case_id"]: solve_case(config, case) for case in cases}


def analyze_three_frequency(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    cases = read_json(out / "three_frequency_independent" / "case_manifest.json")["cases"]
    results = [analyze_case(config, case) for case in cases]
    summaries = [result["summary"] for result in results]
    write_csv(out / "three_frequency_independent" / "three_frequency_summary.csv", summaries)
    gate = all(row["strict_initial_gate_pass"] for row in summaries)
    summary = {"frequency_count": len(summaries), "minimum_passive_rl_db": min(row["minimum_passive_rl_db"] for row in summaries), "minimum_active_rl_db": min(row["minimum_active_rl_db"] for row in summaries), "minimum_total_rl_db": min(row["minimum_total_rl_db"] for row in summaries), "minimum_system_efficiency": min(row["minimum_system_efficiency"] for row in summaries), "three_frequency_gate_pass": gate}
    write_json(out / "three_frequency_independent" / "stage_summary.json", summary)
    decision = {"stage": "G_three_frequency_complete", "physical_2x2_gate_pass": gate, "allow_4x4": False, "allow_16x16": False, "allow_eep_export": False, "allow_training_labels": False, "allow_critic_training": False, "reason": "The physical 2x2 passed the independent three-frequency gate; a separate authorization is still required before any larger array." if gate else "The physical 2x2 failed at least one frozen three-frequency gate; larger arrays and learning remain locked."}
    write_json(out / "stage_decision.json", decision)
    return {"summary": summary, "rows": summaries, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(config["output_directory"])
    return {"output_directory": str(out), "free_memory_gib": memory_available_gb(), "aedt_processes": aedt_processes(), "stage_decision": read_json(out / "stage_decision.json") if (out / "stage_decision.json").exists() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--mode", choices=("preregister", "build-smoke", "prepare-initial-repair", "run-initial", "analyze-initial", "prepare-crosscheck", "run-crosscheck", "analyze-crosscheck", "prepare-three-frequency", "run-three-frequency", "analyze-three-frequency", "status"), default="status")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.amendment:
        amendment = read_json(args.amendment)
        if sha256(resolve(amendment["base_config"])) != amendment["base_config_sha256"]:
            raise RuntimeError("Numerical amendment base-config hash mismatch")
        allowed = {"local_mesh_max_length_mm", "adaptive_refinement_percent", "maximum_passes"}
        unknown = set(amendment["geometry_overrides"]) - allowed
        if unknown:
            raise RuntimeError(f"Numerical amendment attempts non-authorized changes: {sorted(unknown)}")
        config["frozen_geometry"] = {**config["frozen_geometry"], **amendment["geometry_overrides"]}
        config["active_numerical_amendment"] = {**amendment, "amendment_sha256": sha256(args.amendment.resolve())}
    actions = {"preregister": preregister, "build-smoke": build_smoke, "prepare-initial-repair": prepare_initial_repair, "run-initial": run_initial, "analyze-initial": analyze_initial, "prepare-crosscheck": prepare_crosscheck, "run-crosscheck": run_crosscheck, "analyze-crosscheck": analyze_crosscheck, "prepare-three-frequency": prepare_three_frequency, "run-three-frequency": run_three_frequency, "analyze-three-frequency": analyze_three_frequency, "status": status}
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
