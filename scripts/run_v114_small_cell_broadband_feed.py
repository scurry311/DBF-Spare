#!/usr/bin/env python3
"""Run the staged v1.14 physical small-cell broadband-feed feasibility gate.

The script never creates a 16x16 model.  It screens one physical dual-slot
probe-feed topology at 1x1, promotes one passing geometry to 2x2 and 4x4, and
replays frozen v1.13 task excitations on the resulting multiport S matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v114_small_cell_broadband_feed_preregistered.json"
DEFAULT_FROZEN = (
    ROOT
    / "hfss_outputs"
    / "v21_frozen_v112_replay_20260729_run03"
    / "frozen_v112_replay_candidates.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "v114_small_cell_broadband_feed_20260730_run04"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "SmallCell_DualSlot_10GHz"
FREQUENCY_STATES = {
    9.96: ("frequency_low_identity", "frequency_low_E2_source"),
    10.0: ("nominal_identity",),
    10.04: ("frequency_high_identity", "frequency_high_E2_source"),
}
EPS = 1.0e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("prepare", "prepare-side", "run", "analyze", "status"),
        default="status",
    )
    parser.add_argument("--side", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--candidate-id")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_event(out_dir: Path, event: dict[str, Any]) -> None:
    payload = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with (out_dir / "stage_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def memory_available_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available / 1024**3)
    except Exception:
        try:
            import ctypes

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

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.available_physical / 1024**3)
        except Exception:
            pass
        return float("nan")


def load_frozen(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def ri_to_complex(values: np.ndarray) -> np.ndarray:
    return np.asarray(values[..., 0], dtype=float) + 1j * np.asarray(
        values[..., 1], dtype=float
    )


def window_role(x0: int, y0: int, side: int) -> str:
    x_edge = x0 in (0, 16 - side)
    y_edge = y0 in (0, 16 - side)
    if x_edge and y_edge:
        return "corner"
    if x_edge or y_edge:
        return "edge"
    return "interior"


def window_ports(x0: int, y0: int, side: int) -> np.ndarray:
    return np.asarray(
        [(x0 + ix) * 16 + (y0 + iy) for ix in range(side) for iy in range(side)],
        dtype=int,
    )


def select_windows(
    side: int,
    tasks: np.ndarray,
    masks: np.ndarray,
    selected_rows: np.ndarray,
) -> dict[str, tuple[int, int, np.ndarray, float]]:
    energy = np.sum(np.abs(tasks[selected_rows]) ** 2, axis=(0, 1, 3))
    active_rate = np.mean(masks[selected_rows], axis=0)
    contribution = energy * (0.25 + 0.75 * active_rate)
    best: dict[str, tuple[int, int, np.ndarray, float]] = {}
    for x0 in range(17 - side):
        for y0 in range(17 - side):
            role = window_role(x0, y0, side)
            ports = window_ports(x0, y0, side)
            score = float(np.sum(contribution[ports]))
            if role not in best or score > best[role][3]:
                best[role] = (x0, y0, ports, score)
    return best


def prepare_stimuli(protocol: dict[str, Any], frozen_path: Path, out_dir: Path) -> None:
    frozen = load_frozen(frozen_path)
    samples = np.asarray(frozen["sample_index"], dtype=int)
    requested = (
        protocol["representative_scenes"]["near_boundary"]
        + protocol["representative_scenes"]["stress"]
    )
    selected_rows = np.asarray(
        [int(np.where(samples == sample)[0][0]) for sample in requested], dtype=int
    )
    tasks = ri_to_complex(frozen["state_tasks_real_imag"])
    masks = np.asarray(frozen["masks"], dtype=bool)
    state_names = [str(item) for item in frozen["state_names"]]
    k_values = np.asarray(frozen["k_values"], dtype=int)
    ratios = np.asarray(frozen["ratio"], dtype=float)
    stress = set(protocol["representative_scenes"]["stress"])

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    considered: list[np.ndarray] = []
    for side in (2, 4):
        windows = select_windows(side, tasks, masks, selected_rows)
        for role, (x0, y0, ports, score) in windows.items():
            for row_index in selected_rows:
                sample = int(samples[row_index])
                k_value = int(k_values[row_index])
                for frequency, names in FREQUENCY_STATES.items():
                    for state_name in names:
                        state_index = state_names.index(state_name)
                        local_tasks = np.asarray(
                            tasks[row_index, state_index, ports, :k_value],
                            dtype=np.complex128,
                        )
                        local_mask = masks[row_index, ports]
                        source_items = [("combined", -1, np.sum(local_tasks, axis=1))]
                        source_items.extend(
                            ("task", task_index, local_tasks[:, task_index])
                            for task_index in range(k_value)
                        )
                        for source_type, task_index, vector in source_items:
                            vector = np.where(local_mask, vector, 0.0)
                            amplitude = np.abs(vector)
                            floor = max(float(np.max(amplitude)) * 0.1, 1.0e-10)
                            active = local_mask & (amplitude >= floor)
                            if not np.any(active):
                                continue
                            norm = float(np.linalg.norm(vector))
                            if norm <= 1.0e-12:
                                continue
                            vectors.append(vector / norm)
                            considered.append(active)
                            rows.append(
                                {
                                    "stimulus_index": len(rows),
                                    "side": side,
                                    "window_role": role,
                                    "window_x0": x0,
                                    "window_y0": y0,
                                    "window_score": score,
                                    "sample_index": sample,
                                    "scene_class": "stress" if sample in stress else "near_boundary",
                                    "k_value": k_value,
                                    "ratio": float(ratios[row_index]),
                                    "frequency_ghz": frequency,
                                    "state_name": state_name,
                                    "source_type": source_type,
                                    "task_index": task_index,
                                    "local_active_count": int(np.sum(active)),
                                }
                            )
    stimulus_dir = out_dir / "stimuli"
    stimulus_dir.mkdir(parents=True, exist_ok=True)
    write_csv(stimulus_dir / "representative_stimuli.csv", rows)
    max_ports = 16
    vector_array = np.zeros((len(vectors), max_ports), dtype=np.complex64)
    considered_array = np.zeros((len(vectors), max_ports), dtype=np.int8)
    for index, (vector, active) in enumerate(zip(vectors, considered)):
        vector_array[index, : len(vector)] = vector
        considered_array[index, : len(active)] = active
    np.savez_compressed(
        stimulus_dir / "representative_stimuli.npz",
        vectors_real_imag=np.stack((vector_array.real, vector_array.imag), axis=-1),
        considered=considered_array,
    )
    write_json(
        stimulus_dir / "stimulus_summary.json",
        {
            "frozen_source": str(frozen_path.resolve()),
            "sample_indices": requested,
            "stimulus_count": len(rows),
            "counts_by_side": {
                str(side): sum(int(row["side"]) == side for row in rows) for side in (2, 4)
            },
            "normalization": "unit local incident-wave norm; active reflection is scale invariant",
        },
    )


def create_box_sub() -> str:
    return r'''Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(170 130 70)", "Transparency:=", 0.2, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateCylinderZ(editor, objName, x, y, z, radius, height, material, solveInside)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", "24"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(210 150 50)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.1, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub SubtractObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", False)
End Sub
Sub SubtractKeepObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", True)
End Sub
Sub UniteObjects(editor, firstName, secondName)
    editor.Unite Array("NAME:Selections", "Selections:=", firstName & "," & secondName), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub AssignPort(boundary, portName, sheetName, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 6)) & "mm"
End Function
Function Pad3(value)
    Pad3 = Right("000" & CStr(value), 3)
End Function'''


def builder_text(project: Path, side: int, candidate: dict[str, Any], protocol: dict[str, Any]) -> str:
    physical = protocol["physical_topology"]
    spacing = float(physical["spacing_mm"])
    h = float(physical["substrate_thickness_mm"])
    copper = float(physical["copper_thickness_mm"])
    patch_w = float(physical["patch_width_mm"])
    patch_l = float(physical["patch_length_mm"])
    slot_l = float(candidate["slot_length_mm"])
    slot_w = float(candidate["slot_width_mm"])
    tongue = float(candidate["tongue_width_mm"])
    feed_inset = float(candidate["feed_inset_from_edge_mm"])
    board = max(15.0, (side - 1) * spacing + 15.0)
    coax_inner = 0.60
    coax_outer = 1.10
    probe = 0.25
    coax_drop = 0.80
    coax_bottom = -h - copper - coax_drop
    mesh = float(physical["mesh_max_length_mm"])
    mesh_refine_inside = "True" if bool(physical.get("mesh_refine_inside", True)) else "False"
    refine = float(physical["adaptive_refinement_percent"])
    decoupler_enabled = bool(candidate.get("decoupler_x_strip_enabled", False)) and side > 1
    mesh_name_items = [f'"FeedMesh_{index:03d}"' for index in range(side * side)]
    if decoupler_enabled:
        mesh_name_items.extend(
            f'"DecX_{index:03d}"' for index in range((side - 1) * side)
        )
    local_mesh_names = ", ".join(mesh_name_items)
    mesh_width = tongue + 2.0 * slot_w + 1.2
    mesh_length = slot_l + 0.5
    decoupler_geometry = ""
    if decoupler_enabled:
        decoupler_length = float(candidate["decoupler_length_mm"])
        decoupler_width = float(candidate["decoupler_width_mm"])
        decoupler_geometry = f'''idx = 0
For ix = 0 To {side - 2}
    For iy = 0 To {side - 1}
        xc = (ix - 0.5 * ({side - 1}) + 0.5) * {spacing:.6f}
        yc = (iy - 0.5 * ({side - 1})) * {spacing:.6f}
        nameBase = Pad3(idx)
        CreateBox oEditor, "DecX_" & nameBase, xc - {decoupler_width/2:.6f}, yc - {decoupler_length/2:.6f}, 0, {decoupler_width:.6f}, {decoupler_length:.6f}, {copper:.6f}, "copper", False
        idx = idx + 1
    Next
Next'''
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Dim ix, iy, idx, xc, yc, patchBottom, feedY, slotOffset, nameBase
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V114", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{physical['relative_permittivity']}", "dielectric_loss_tangent:=", "{physical['loss_tangent']}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board/2:.6f}, {-board/2:.6f}, {-h:.6f}, {board:.6f}, {board:.6f}, {h:.6f}, "RO5880_V114", True
CreateBox oEditor, "Ground", {-board/2:.6f}, {-board/2:.6f}, {-h-copper:.6f}, {board:.6f}, {board:.6f}, {copper:.6f}, "copper", False
idx = 0
For ix = 0 To {side - 1}
    For iy = 0 To {side - 1}
        xc = (ix - 0.5 * ({side - 1})) * {spacing:.6f}
        yc = (iy - 0.5 * ({side - 1})) * {spacing:.6f}
        patchBottom = yc - {patch_l/2:.6f}
        feedY = patchBottom + {feed_inset:.6f}
        slotOffset = 0.5 * ({tongue:.6f} + {slot_w:.6f})
        nameBase = Pad3(idx)
        CreateBox oEditor, "FeedMesh_" & nameBase, xc - {mesh_width/2:.6f}, patchBottom - 0.10, {-h:.6f}, {mesh_width:.6f}, {mesh_length:.6f}, {h:.6f}, "RO5880_V114", True
        SubtractKeepObject oEditor, "Substrate", "FeedMesh_" & nameBase
        CreateCylinderZ oEditor, "GroundHole_" & nameBase, xc, feedY, {-h-copper-0.05:.6f}, {coax_inner:.6f}, {copper+0.10:.6f}, "vacuum", True
        SubtractObject oEditor, "Ground", "GroundHole_" & nameBase
        CreateCylinderZ oEditor, "SubstrateHole_" & nameBase, xc, feedY, {-h-0.01:.6f}, {probe+0.01:.6f}, {h+0.02:.6f}, "vacuum", True
        SubtractKeepObject oEditor, "Substrate", "SubstrateHole_" & nameBase
        SubtractObject oEditor, "FeedMesh_" & nameBase, "SubstrateHole_" & nameBase
        CreateBox oEditor, "Patch_" & nameBase, xc - {patch_w/2:.6f}, patchBottom, 0, {patch_w:.6f}, {patch_l:.6f}, {copper:.6f}, "copper", False
        CreateBox oEditor, "SlotL_" & nameBase, xc - slotOffset - {slot_w/2:.6f}, patchBottom - 0.01, -0.01, {slot_w:.6f}, {slot_l+0.02:.6f}, {copper+0.02:.6f}, "vacuum", True
        SubtractObject oEditor, "Patch_" & nameBase, "SlotL_" & nameBase
        CreateBox oEditor, "SlotR_" & nameBase, xc + slotOffset - {slot_w/2:.6f}, patchBottom - 0.01, -0.01, {slot_w:.6f}, {slot_l+0.02:.6f}, {copper+0.02:.6f}, "vacuum", True
        SubtractObject oEditor, "Patch_" & nameBase, "SlotR_" & nameBase
        CreateCylinderZ oEditor, "Probe_" & nameBase, xc, feedY, {coax_bottom:.6f}, {probe:.6f}, {h+2*copper+coax_drop:.6f}, "copper", False
        UniteObjects oEditor, "Patch_" & nameBase, "Probe_" & nameBase
        CreateCylinderZ oEditor, "Outer_" & nameBase, xc, feedY, {coax_bottom:.6f}, {coax_outer:.6f}, {coax_drop+copper/2:.6f}, "copper", False
        CreateCylinderZ oEditor, "OuterCut_" & nameBase, xc, feedY, {coax_bottom-0.02:.6f}, {coax_inner:.6f}, {coax_drop+copper+0.04:.6f}, "vacuum", True
        SubtractObject oEditor, "Outer_" & nameBase, "OuterCut_" & nameBase
        UniteObjects oEditor, "Ground", "Outer_" & nameBase
        CreateSheetZ oEditor, "PortSheet_" & nameBase, xc + {probe-0.02:.6f}, feedY - 0.15, {coax_bottom:.6f}, {coax_inner-probe+0.04:.6f}, 0.3
        AssignPort oBoundary, "P" & nameBase, "PortSheet_" & nameBase, xc + {probe+0.01:.6f}, feedY, {coax_bottom:.6f}, xc + {coax_inner-0.01:.6f}, feedY, {coax_bottom:.6f}
        idx = idx + 1
    Next
Next
{decoupler_geometry}
CreateBox oEditor, "AirRegion", {-board/2-15:.6f}, {-board/2-15:.6f}, -15, {board+30:.6f}, {board+30:.6f}, 30, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:FeedSlotUniform_0p180mm", "RefineInside:=", {mesh_refine_inside}, "Enabled:=", True, "Objects:=", Array({local_mesh_names}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{mesh:.6f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", 24, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {refine:.6f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False)
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "9.96GHz", "RangeEnd:=", "10.04GHz", "RangeCount:=", 3, "Type:=", "Discrete", "SaveFields:=", True, "SaveRadFields:=", True, "InterpTolerance:=", 0.5, "InterpMaxSolns:=", 250, "InterpMinSolns:=", 0, "InterpMinSubranges:=", 1, "InterpUseS:=", True, "InterpUsePortImped:=", True, "InterpUsePropConst:=", True, "UseDerivativeConvergence:=", False, "InterpDerivTolerance:=", 0.2, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V114", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

{create_box_sub()}
'''


def solve_text(project: Path, touchstone: Path, efficiency_csv: Path) -> str:
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, reportName
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:Sweep_Gate3")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Gate3"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
Set oReport = oDesign.GetModule("ReportSetup")
reportName = "V114_RadiationEfficiency"
On Error Resume Next
oReport.DeleteReports Array(reportName)
Err.Clear
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : Sweep_Gate3", Array("Context:=", "InfiniteSphere_V114"), Array("Freq:=", Array("9.96GHz", "10GHz", "10.04GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "{vp(efficiency_csv)}"
On Error GoTo 0
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def case_dir(out_dir: Path, side: int, candidate_id: str, replicate: int) -> Path:
    return out_dir / f"{side}x{side}" / f"{candidate_id}_direct{replicate:02d}"


def prepare_case(
    out_dir: Path,
    side: int,
    candidate: dict[str, Any],
    replicate: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    folder = case_dir(out_dir, side, str(candidate["candidate_id"]), replicate)
    if folder.exists():
        return load_json(folder / "case_manifest.json")
    folder.mkdir(parents=True)
    name = f"v114_{side}x{side}_{candidate['candidate_id']}_direct{replicate:02d}"
    nports = side * side
    project = folder / f"{name}.aedt"
    touchstone = folder / f"{name}.s{nports}p"
    efficiency = folder / "radiation_efficiency.csv"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    builder.write_text(builder_text(project, side, candidate, protocol), encoding="ascii")
    solver.write_text(solve_text(project, touchstone, efficiency), encoding="ascii")
    manifest = {
        "name": name,
        "side": side,
        "port_count": nports,
        "candidate": candidate,
        "replicate": replicate,
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "efficiency_csv_path": str(efficiency.resolve()),
        "builder_path": str(builder.resolve()),
        "solver_path": str(solver.resolve()),
    }
    write_json(folder / "case_manifest.json", manifest)
    return manifest


def initial_prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing v1.14 run: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    protocol = load_json(args.protocol)
    shutil.copy2(args.protocol, args.out_dir / "protocol_snapshot.json")
    prepare_stimuli(protocol, args.frozen, args.out_dir)
    cases = [
        prepare_case(args.out_dir, 1, candidate, 1, protocol)
        for candidate in protocol["one_by_one_candidates"]
    ]
    append_event(
        args.out_dir,
        {
            "event": "prepared",
            "scope": "1x1 candidates and frozen small-array stimuli",
            "case_count": len(cases),
            "downstream_locks": protocol["scope"],
        },
    )
    return {"out_dir": str(args.out_dir), "prepared_1x1_cases": len(cases)}


def prepare_side(args: argparse.Namespace) -> dict[str, Any]:
    if args.side == 1:
        raise ValueError("Use --mode prepare for the 1x1 stage")
    protocol = load_json(args.out_dir / "protocol_snapshot.json")
    candidates = protocol["one_by_one_candidates"]
    if args.candidate_id:
        candidates = [candidate_by_id(protocol, args.candidate_id)]
    cases = [prepare_case(args.out_dir, args.side, candidate, 1, protocol) for candidate in candidates]
    append_event(
        args.out_dir,
        {
            "event": "explicit_small_array_cases_prepared",
            "side": args.side,
            "candidate_ids": [item["candidate"]["candidate_id"] for item in cases],
            "allow_16x16": False,
        },
    )
    return {"side": args.side, "prepared_cases": len(cases)}


def selected_candidate(out_dir: Path, protocol: dict[str, Any]) -> dict[str, Any] | None:
    selection = out_dir / "selected_geometry.json"
    if selection.exists():
        selected = load_json(selection)["candidate"]
        return selected
    return None


def candidate_by_id(protocol: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in protocol["one_by_one_candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def discover_cases(out_dir: Path, side: int, candidate_id: str | None = None) -> list[Path]:
    root = out_dir / f"{side}x{side}"
    if not root.exists():
        return []
    cases = sorted(path.parent for path in root.glob("*/case_manifest.json"))
    if candidate_id:
        cases = [path for path in cases if load_json(path / "case_manifest.json")["candidate"]["candidate_id"] == candidate_id]
    return cases


def run_one_case(folder: Path, ansys_exe: Path) -> dict[str, Any]:
    manifest = load_json(folder / "case_manifest.json")
    touchstone = Path(manifest["touchstone_path"])
    if touchstone.exists() and touchstone.stat().st_size > 100:
        return {"name": manifest["name"], "status": "already_complete"}
    free_gb = memory_available_gb()
    minimum = {1: 2.0, 2: 4.0, 4: 8.0}[int(manifest["side"])]
    if math.isfinite(free_gb) and free_gb < minimum:
        raise MemoryError(f"Only {free_gb:.2f} GB free; {minimum:.1f} GB required")
    with (folder / "build.log").open("w", encoding="utf-8", errors="replace") as handle:
        build = subprocess.run(
            [str(ansys_exe), "-RunScriptAndExit", manifest["builder_path"]],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    solve_code: int | None = None
    if build.returncode == 0:
        with (folder / "solve_export.log").open("w", encoding="utf-8", errors="replace") as handle:
            solve = subprocess.run(
                [str(ansys_exe), "-ng", "-RunScriptAndExit", manifest["solver_path"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        solve_code = int(solve.returncode)
    summary = {
        "name": manifest["name"],
        "build_return_code": int(build.returncode),
        "solve_return_code": solve_code,
        "free_memory_gb_before": free_gb,
    }
    write_json(folder / "run_summary.json", summary)
    return summary


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.out_dir / "protocol_snapshot.json")
    selected = selected_candidate(args.out_dir, protocol)
    candidate_id = args.candidate_id or (selected["candidate_id"] if selected else None)
    cases = discover_cases(args.out_dir, args.side, candidate_id)
    if not cases:
        raise RuntimeError(f"No prepared {args.side}x{args.side} cases")
    results = []
    for folder in cases:
        print(f"Running {folder.name}", flush=True)
        results.append(run_one_case(folder, args.ansys_exe))
    append_event(
        args.out_dir,
        {"event": "run_completed", "side": args.side, "results": results},
    )
    return {"side": args.side, "results": results}


def profile_metrics(folder: Path) -> dict[str, Any]:
    selected: tuple[str, list[float]] | None = None
    peak_solver_memory_kb = 0
    maximum_solve_tetrahedra = 0
    for profile in folder.rglob("*.profile"):
        text = profile.read_text(encoding="utf-8", errors="ignore")
        values: list[float] = []
        for line in text.splitlines():
            if "ProfileItem('Matrix Solve'" in line:
                match = re.search(r",\s*(\d+),\s*'I\(", line)
                if match:
                    peak_solver_memory_kb = max(peak_solver_memory_kb, int(match.group(1)))
            if "Tetrahedra" in line and "ProfileItem" in line:
                match = re.search(r"Tetrahedra\\',\s*(\d+)", line)
                if match:
                    maximum_solve_tetrahedra = max(
                        maximum_solve_tetrahedra, int(match.group(1))
                    )
            if "Max Mag. Delta S" not in line:
                continue
            tail = line.split("Delta S", 1)[1]
            for token in tail.split(","):
                try:
                    values.append(float(token.replace("\\", "").replace("'", "").strip()))
                    break
                except ValueError:
                    continue
        if selected is None or len(values) > len(selected[1]):
            selected = (text, values)
    text, values = selected if selected else ("", [])
    errors = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in folder.rglob("*.g3derr")
    )
    return {
        "pass_count": len(values),
        "final_delta_s": values[-1] if values else None,
        "converged": "Adaptive Passes converged" in text and "did not converge" not in text,
        "small_mesh_segment_count": errors.lower().count("small mesh segment"),
        "peak_solver_memory_gb": peak_solver_memory_kb / 1024**2,
        "maximum_tetrahedra": maximum_solve_tetrahedra,
    }


def parse_touchstone(path: Path, nports: int) -> tuple[np.ndarray, np.ndarray]:
    unit = "ghz"
    data_format = "ma"
    tokens: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line[1:].lower().split()
            unit = parts[0]
            data_format = parts[2]
            continue
        if line.startswith("["):
            continue
        tokens.extend(float(value) for value in line.split())
    record = 1 + 2 * nports * nports
    if len(tokens) % record:
        raise ValueError(f"Invalid Touchstone token count in {path}")
    raw = np.asarray(tokens, dtype=float).reshape(-1, record)
    scale = {"hz": 1.0e-9, "khz": 1.0e-6, "mhz": 1.0e-3, "ghz": 1.0}[unit]
    frequencies = raw[:, 0] * scale
    pairs = raw[:, 1:].reshape(-1, nports, nports, 2)
    if data_format == "ma":
        values = pairs[..., 0] * np.exp(1j * np.deg2rad(pairs[..., 1]))
    elif data_format == "ri":
        values = pairs[..., 0] + 1j * pairs[..., 1]
    elif data_format == "db":
        values = 10.0 ** (pairs[..., 0] / 20.0) * np.exp(1j * np.deg2rad(pairs[..., 1]))
    else:
        raise ValueError(f"Unsupported Touchstone format: {data_format}")
    return frequencies, np.transpose(values, (0, 2, 1))


def efficiency_from_csv(path: Path) -> float | None:
    if not path.exists() or path.stat().st_size < 10:
        return None
    values: list[float] = []
    for row in csv.reader(path.open(encoding="utf-8-sig", errors="ignore")):
        for token in row[1:]:
            try:
                value = float(token)
            except ValueError:
                continue
            if 0.0 <= value <= 1.2:
                values.append(value)
    return min(values) if values else None


def load_stimuli(out_dir: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = list(
        csv.DictReader(
            (out_dir / "stimuli" / "representative_stimuli.csv").open(encoding="utf-8-sig")
        )
    )
    with np.load(out_dir / "stimuli" / "representative_stimuli.npz", allow_pickle=False) as data:
        vectors = ri_to_complex(data["vectors_real_imag"])
        considered = np.asarray(data["considered"], dtype=bool)
    return rows, vectors, considered


def active_metrics(
    out_dir: Path,
    side: int,
    frequencies: np.ndarray,
    matrices: np.ndarray,
) -> tuple[list[dict[str, Any]], float, float]:
    rows, vectors, considered = load_stimuli(out_dir)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if int(row["side"]) != side:
            continue
        frequency = float(row["frequency_ghz"])
        f_index = int(np.argmin(np.abs(frequencies - frequency)))
        source = vectors[index, : side * side]
        active = considered[index, : side * side]
        reflected = matrices[f_index] @ source
        gamma = np.abs(reflected[active]) / np.maximum(np.abs(source[active]), EPS)
        active_rl = float(-20.0 * np.log10(max(float(np.max(gamma)), EPS)))
        total_rl = float(
            -10.0
            * np.log10(
                max(
                    float(np.sum(np.abs(reflected) ** 2) / np.sum(np.abs(source) ** 2)),
                    EPS,
                )
            )
        )
        output.append(
            {
                **row,
                "selected_frequency_ghz": float(frequencies[f_index]),
                "active_rl_db": active_rl,
                "total_rl_db": total_rl,
            }
        )
    return (
        output,
        min(float(row["active_rl_db"]) for row in output),
        min(float(row["total_rl_db"]) for row in output),
    )


def port_class(side: int, port: int) -> str:
    ix, iy = divmod(port, side)
    x_edge = ix in (0, side - 1)
    y_edge = iy in (0, side - 1)
    if x_edge and y_edge:
        return "corner"
    if x_edge or y_edge:
        return "edge"
    return "interior"


def analyze_case(folder: Path, out_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = load_json(folder / "case_manifest.json")
    side = int(manifest["side"])
    nports = int(manifest["port_count"])
    touchstone = Path(manifest["touchstone_path"])
    run_summary = load_json(folder / "run_summary.json") if (folder / "run_summary.json").exists() else {}
    result: dict[str, Any] = {
        "name": manifest["name"],
        "side": side,
        "candidate_id": manifest["candidate"]["candidate_id"],
        "replicate": int(manifest["replicate"]),
        **run_summary,
        **profile_metrics(folder),
        "touchstone_exists": touchstone.exists(),
    }
    if not touchstone.exists() or touchstone.stat().st_size < 100:
        result["base_gate_pass"] = False
        write_json(folder / "analysis.json", result)
        return result
    frequencies, matrices = parse_touchstone(touchstone, nports)
    target_frequencies = np.asarray(protocol["frequencies_ghz"], dtype=float)
    selected_indices = [int(np.argmin(np.abs(frequencies - value))) for value in target_frequencies]
    selected = matrices[selected_indices]
    selected_frequencies = frequencies[selected_indices]
    reciprocity = float(np.max(np.abs(selected - np.transpose(selected, (0, 2, 1)))))
    passivity = float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in selected))
    port_rows: list[dict[str, Any]] = []
    for local_frequency, matrix in zip(selected_frequencies, selected):
        rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(matrix)), EPS))
        for port, value in enumerate(rl):
            port_rows.append(
                {
                    "frequency_ghz": float(local_frequency),
                    "port_index": port,
                    "port_class": port_class(side, port),
                    "passive_rl_db": float(value),
                }
            )
    write_csv(folder / "port_frequency_metrics.csv", port_rows)
    passive_min = min(float(row["passive_rl_db"]) for row in port_rows)
    efficiency = efficiency_from_csv(Path(manifest["efficiency_csv_path"]))
    result.update(
        {
            "frequency_count": len(selected_frequencies),
            "frequency_max_error_ghz": float(np.max(np.abs(selected_frequencies - target_frequencies))),
            "passive_rl_min_db": passive_min,
            "reciprocity_error_max": reciprocity,
            "passivity_sigma_max": passivity,
            "minimum_radiation_efficiency": efficiency,
            "radiation_efficiency_available": efficiency is not None,
        }
    )
    cross_mesh = protocol.get("cross_mesh_reference", {})
    reference_path = cross_mesh.get("touchstone")
    if reference_path and side == int(cross_mesh.get("side", 1)):
        reference = Path(reference_path)
        if not reference.is_absolute():
            reference = ROOT / reference
        reference_frequencies, reference_matrices = parse_touchstone(reference, nports)
        reference_indices = [
            int(np.argmin(np.abs(reference_frequencies - value)))
            for value in target_frequencies
        ]
        reference_selected = reference_matrices[reference_indices]
        result["cross_mesh_reference"] = str(reference.resolve())
        result["cross_mesh_max_abs_delta_s"] = float(
            np.max(np.abs(selected - reference_selected))
        )
        result["cross_mesh_gate_pass"] = bool(
            result["cross_mesh_max_abs_delta_s"]
            <= float(cross_mesh.get("maximum_abs_delta_s", 0.05))
        )
    gates = protocol["gates"][
        "one_by_one" if side == 1 else "two_by_two_and_four_by_four"
    ]
    if side > 1:
        replay_rows, active_min, total_min = active_metrics(
            out_dir, side, selected_frequencies, selected
        )
        write_csv(folder / "representative_active_rl_replay.csv", replay_rows)
        result["representative_active_rl_min_db"] = active_min
        result["representative_total_rl_min_db"] = total_min
    topology_warnings = 0
    for log in (folder / "build.log", folder / "solve_export.log"):
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="ignore")
            for pattern in (
                "Too many conductors touch lumped port",
                "'0' conductors touch lumped port",
                "'1' conductors touch lumped port",
                "Both endpoints of port lines must lie on port",
            ):
                topology_warnings += text.count(pattern)
    result["port_topology_warning_count"] = topology_warnings
    common = (
        result.get("build_return_code") == 0
        and result.get("solve_return_code") == 0
        and result.get("converged") is True
        and float(result.get("final_delta_s") or float("inf"))
        <= float(gates["maximum_final_delta_s"])
        and topology_warnings == 0
        and result["frequency_max_error_ghz"] <= 1.0e-6
        and passivity <= 1.0001
        and efficiency is not None
        and efficiency >= float(gates["minimum_network_efficiency"])
        and result.get("cross_mesh_gate_pass", True)
    )
    if side == 1:
        result["base_gate_pass"] = bool(
            common
            and passive_min >= float(gates["minimum_passive_rl_db_all_frequencies"])
        )
    else:
        result["base_gate_pass"] = bool(
            common
            and passive_min
            >= float(gates["minimum_passive_rl_db_all_ports_all_frequencies"])
            and result["representative_active_rl_min_db"]
            >= float(gates["minimum_active_rl_db_all_representative_sources"])
            and result["representative_total_rl_min_db"]
            >= float(gates["minimum_total_rl_db_all_representative_sources"])
        )
    write_json(folder / "analysis.json", result)
    return result


def compare_repeats(folders: list[Path]) -> dict[str, float] | None:
    complete = [folder for folder in folders if (folder / "analysis.json").exists()]
    if len(complete) < 2:
        return None
    payloads = [load_json(folder / "case_manifest.json") for folder in complete[:2]]
    nports = int(payloads[0]["port_count"])
    f0, s0 = parse_touchstone(Path(payloads[0]["touchstone_path"]), nports)
    f1, s1 = parse_touchstone(Path(payloads[1]["touchstone_path"]), nports)
    if len(f0) != len(f1) or np.max(np.abs(f0 - f1)) > 1.0e-9:
        raise ValueError("Repeat frequency grids differ")
    delta = np.abs(s0 - s1)
    return {
        "max_abs_delta_s": float(np.max(delta)),
        "rms_abs_delta_s": float(np.sqrt(np.mean(delta**2))),
    }


def prepare_repeat_or_next(
    out_dir: Path,
    side: int,
    candidate: dict[str, Any],
    protocol: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    folders = discover_cases(out_dir, side, str(candidate["candidate_id"]))
    comparison = compare_repeats(folders)
    passing = [row for row in results if row.get("base_gate_pass")]
    if not passing:
        return {"decision": f"stop_{side}x{side}_base_gate_failed"}
    if len(folders) < 2:
        prepare_case(out_dir, side, candidate, 2, protocol)
        return {"decision": f"run_{side}x{side}_independent_repeat"}
    repeat_limit = float(
        protocol["gates"][
            "one_by_one" if side == 1 else "two_by_two_and_four_by_four"
        ]["maximum_solver_repeat_delta_s"]
    )
    if comparison is None or comparison["max_abs_delta_s"] > repeat_limit:
        return {
            "decision": f"stop_{side}x{side}_repeat_consistency_failed",
            "repeat_comparison": comparison,
        }
    if not all(row.get("base_gate_pass") for row in results):
        return {
            "decision": f"stop_{side}x{side}_repeat_gate_not_reproduced",
            "repeat_comparison": comparison,
        }
    if side < 4:
        next_side = 2 if side == 1 else 4
        prepare_case(out_dir, next_side, candidate, 1, protocol)
        return {
            "decision": f"allow_{next_side}x{next_side}_base_solve",
            "repeat_comparison": comparison,
        }
    return {
        "decision": "small_cell_hardware_gate_pass",
        "repeat_comparison": comparison,
        "allow_16x16": False,
        "critic_training_allowed": False,
    }


def analyze_stage(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.out_dir / "protocol_snapshot.json")
    selected = selected_candidate(args.out_dir, protocol)
    candidate_id = args.candidate_id or (selected["candidate_id"] if selected else None)
    cases = discover_cases(args.out_dir, args.side, candidate_id)
    if not cases:
        raise RuntimeError(f"No prepared {args.side}x{args.side} cases")
    results = [analyze_case(folder, args.out_dir, protocol) for folder in cases]
    write_csv(args.out_dir / f"{args.side}x{args.side}_case_metrics.csv", results)

    if args.side == 1 and selected is None:
        first_runs = [row for row in results if int(row["replicate"]) == 1]
        passing = [row for row in first_runs if row.get("base_gate_pass")]
        if not passing:
            decision = {
                "decision": "stop_topology_no_1x1_candidate_meets_preregistered_gate",
                "passing_candidate_count": 0,
            }
        else:
            winner = max(
                passing,
                key=lambda row: (
                    float(row["passive_rl_min_db"]),
                    float(row["minimum_radiation_efficiency"]),
                ),
            )
            candidate = candidate_by_id(protocol, str(winner["candidate_id"]))
            write_json(
                args.out_dir / "selected_geometry.json",
                {
                    "selection_rule": "maximum all-corner passive-RL margin, then radiation efficiency",
                    "candidate": candidate,
                    "first_run_metrics": winner,
                },
            )
            prepare_case(args.out_dir, 1, candidate, 2, protocol)
            decision = {
                "decision": "run_selected_1x1_independent_repeat",
                "candidate_id": candidate["candidate_id"],
            }
    else:
        if selected is None and args.candidate_id:
            selected = candidate_by_id(protocol, args.candidate_id)
        if selected is None:
            raise RuntimeError("A selected 1x1 geometry or --candidate-id is required")
        relevant = [
            row for row in results if row["candidate_id"] == selected["candidate_id"]
        ]
        decision = prepare_repeat_or_next(
            args.out_dir, args.side, selected, protocol, relevant
        )
    revision = len(list(args.out_dir.glob("stage_decision_rev*.json"))) + 1
    decision_path = args.out_dir / f"stage_decision_rev{revision:02d}.json"
    write_json(
        decision_path,
        {
            "side": args.side,
            "decision": decision,
            "locks": protocol["scope"],
            "thresholds_changed": False,
            "mask_or_weight_changed": False,
        },
    )
    append_event(
        args.out_dir,
        {"event": "analysis_completed", "side": args.side, "decision": decision},
    )
    return {"metrics": results, "stage_decision": decision, "decision_path": str(decision_path)}


def status(args: argparse.Namespace) -> dict[str, Any]:
    if not args.out_dir.exists():
        return {"prepared": False, "out_dir": str(args.out_dir)}
    cases: dict[str, Any] = {}
    for side in (1, 2, 4):
        entries = []
        for folder in discover_cases(args.out_dir, side):
            manifest = load_json(folder / "case_manifest.json")
            entries.append(
                {
                    "name": manifest["name"],
                    "run_complete": (folder / "run_summary.json").exists(),
                    "touchstone_exists": Path(manifest["touchstone_path"]).exists(),
                    "analyzed": (folder / "analysis.json").exists(),
                }
            )
        cases[f"{side}x{side}"] = entries
    decisions = sorted(args.out_dir.glob("stage_decision_rev*.json"))
    return {
        "prepared": (args.out_dir / "protocol_snapshot.json").exists(),
        "selected_geometry": (
            load_json(args.out_dir / "selected_geometry.json")
            if (args.out_dir / "selected_geometry.json").exists()
            else None
        ),
        "cases": cases,
        "latest_decision": load_json(decisions[-1]) if decisions else None,
        "free_memory_gb": memory_available_gb(),
    }


def main() -> None:
    args = parse_args()
    result = {
        "prepare": initial_prepare,
        "prepare-side": prepare_side,
        "run": run_stage,
        "analyze": analyze_stage,
        "status": status,
    }[args.mode](args)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
