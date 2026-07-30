#!/usr/bin/env python3
"""Build and gate one single-stage sparse-neighbor POST/feed physical S8."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from design_v115_grounded_modal_network import terminate_network
from design_v119_multiport_post_decoupler import active_metrics, deembed_load, reordered_network
from design_v120_joint_feed_fanout_sparse_graph import sparse_pi_s8, unpack
from run_v114_small_cell_broadband_feed import load_stimuli, memory_available_gb, parse_touchstone, profile_metrics
from run_v115_physical_modal_feed_fixture import helpers_vbs, touchstone_port_names, vp
from run_v1191_multiconductor_post_block import phase_align, port_positions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v120_sparse_graph_physical_front_gate.json"
DESIGN_NAME = "V120_SparseGraph_PhysicalFront_S8"
EPS = 1.0e-15


def resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rlc_assignment(value: float, kind: str, ql: float, qc: float) -> dict[str, Any]:
    omega = 2.0 * math.pi * 10.0e9
    if kind == "series":
        if value >= 0.0:
            return {"use_l": True, "use_c": False, "l_nh": value / omega * 1.0e9, "c_pf": 1.0, "r_ohm": max(value / ql, 1.0e-6)}
        return {"use_l": False, "use_c": True, "l_nh": 1.0, "c_pf": -1.0 / (omega * value) * 1.0e12, "r_ohm": max(abs(value) / qc, 1.0e-6)}
    if value >= 0.0:
        return {"use_l": False, "use_c": True, "l_nh": 1.0, "c_pf": value / omega * 1.0e12, "r_ohm": qc / max(value, EPS)}
    return {"use_l": True, "use_c": False, "l_nh": -1.0 / (omega * value) * 1.0e9, "c_pf": 1.0, "r_ohm": ql / max(abs(value), EPS)}


def vb_bool(value: bool) -> str:
    return "True" if value else "False"


def builder_text(project: Path, config: dict[str, Any]) -> str:
    block = config["physical_block"]
    synthesis_config = read_json(resolve(config["synthesis_config"]))
    synthesis = read_json(resolve(config["synthesis_summary"]))
    graph = [tuple(int(item) for item in pair) for pair in synthesis_config["manufacturable_graph_pairs"]]
    _, series_ground, _, input_ground, input_pair, output_ground, output_pair = unpack(np.asarray(synthesis["optimized_parameters"], dtype=float))
    q = synthesis_config["network"]
    h = float(block["substrate_thickness_mm"])
    copper = float(block["copper_thickness_mm"])
    board_l = float(block["board_length_mm"])
    board_w = float(block["board_width_mm"])
    trace_l = float(block["trace_length_mm"])
    widths = [float(value) for value in block["trace_width_mm_by_port"]]
    positions = port_positions(block)
    pre_x = -trace_l / 2.0
    post_x = trace_l / 2.0
    series_x = float(block["series_plane_x_mm"])
    sheet_w = float(block["component_sheet_width_mm"])
    geometry: list[str] = []
    assignments: list[str] = []
    pre_assignments: list[str] = []
    post_assignments: list[str] = []
    mesh_names: list[str] = []
    for port in range(4):
        width = widths[port]
        y = positions[port]
        left = f"TraceLeft_{port}"
        right = f"TraceRight_{port}"
        series_sheet = f"SeriesSheet_{port}"
        pre_sheet = f"PrePortSheet_{port}"
        post_sheet = f"PostPortSheet_{port}"
        geometry.extend([
            f'CreateBox oEditor, "{left}", {pre_x:.7f}, {y-width/2:.7f}, 0, {series_x-sheet_w/2-pre_x:.7f}, {width:.7f}, {copper:.7f}, "copper", False',
            f'CreateBox oEditor, "{right}", {series_x+sheet_w/2:.7f}, {y-width/2:.7f}, 0, {post_x-series_x-sheet_w/2:.7f}, {width:.7f}, {copper:.7f}, "copper", False',
            f'CreateSheet oEditor, "{series_sheet}", "Z", {series_x-sheet_w/2:.7f}, {y-width/2:.7f}, 0, {sheet_w:.7f}, {width:.7f}',
            f'CreateSheet oEditor, "{pre_sheet}", "X", {pre_x:.7f}, {y-width/2:.7f}, {-h:.7f}, {width:.7f}, {h:.7f}',
            f'CreateSheet oEditor, "{post_sheet}", "X", {post_x:.7f}, {y-width/2:.7f}, {-h:.7f}, {width:.7f}, {h:.7f}',
        ])
        item = rlc_assignment(float(series_ground[port]), "series", float(q["series_inductor_q"]), float(q["series_capacitor_q"]))
        assignments.append(
            f'AssignRLC oBoundary, "Series_{port}", "{series_sheet}", "Serial", True, {item["r_ohm"]:.10f}, {vb_bool(item["use_l"])}, {item["l_nh"]:.10f}, {vb_bool(item["use_c"])}, {item["c_pf"]:.10f}, {series_x-sheet_w/2:.7f}, {y:.7f}, 0, {series_x+sheet_w/2:.7f}, {y:.7f}, 0'
        )
        pre_assignments.append(f'AssignPort oBoundary, "PRE_{port}", "{pre_sheet}", {pre_x:.7f}, {y:.7f}, 0, {pre_x:.7f}, {y:.7f}, {-h:.7f}')
        post_assignments.append(f'AssignPort oBoundary, "POST_{port}", "{post_sheet}", {post_x:.7f}, {y:.7f}, 0, {post_x:.7f}, {y:.7f}, {-h:.7f}')
        mesh_names.extend((left, right, series_sheet))

    for plane, values in (("Input", input_ground), ("Output", output_ground)):
        x = float(block[f"{plane.lower()}_shunt_x_mm"])
        for port, value in enumerate(values):
            if abs(float(value)) <= EPS:
                continue
            y = positions[port]
            width = widths[port]
            name = f"{plane}GroundSheet_{port}"
            item = rlc_assignment(float(value), "shunt", float(q["shunt_inductor_q"]), float(q["shunt_capacitor_q"]))
            geometry.append(f'CreateSheet oEditor, "{name}", "X", {x:.7f}, {y-width/4:.7f}, {-h:.7f}, {width/2:.7f}, {h:.7f}')
            assignments.append(
                f'AssignRLC oBoundary, "{plane}Ground_{port}", "{name}", "Parallel", True, {item["r_ohm"]:.10f}, {vb_bool(item["use_l"])}, {item["l_nh"]:.10f}, {vb_bool(item["use_c"])}, {item["c_pf"]:.10f}, {x:.7f}, {y:.7f}, 0, {x:.7f}, {y:.7f}, {-h:.7f}'
            )
            mesh_names.append(name)

    assignments.extend(pre_assignments)
    assignments.extend(post_assignments)

    for plane, values in (("Input", input_pair), ("Output", output_pair)):
        x = float(block[f"{plane.lower()}_graph_x_mm"])
        for edge, (value, pair) in enumerate(zip(values, graph)):
            if abs(float(value)) <= EPS:
                continue
            first, second = pair
            low, high = sorted((positions[first], positions[second]))
            name = f"{plane}GraphSheet_{edge}"
            item = rlc_assignment(float(value), "shunt", float(q["shunt_inductor_q"]), float(q["shunt_capacitor_q"]))
            geometry.append(f'CreateSheet oEditor, "{name}", "Z", {x-sheet_w/2:.7f}, {low+widths[first]/2:.7f}, 0, {sheet_w:.7f}, {high-low-(widths[first]+widths[second])/2:.7f}')
            assignments.append(
                f'AssignRLC oBoundary, "{plane}Graph_{edge}", "{name}", "Parallel", True, {item["r_ohm"]:.10f}, {vb_bool(item["use_l"])}, {item["l_nh"]:.10f}, {vb_bool(item["use_c"])}, {item["c_pf"]:.10f}, {x:.7f}, {low+widths[first]/2:.7f}, 0, {x:.7f}, {high-widths[second]/2:.7f}, 0'
            )
            mesh_names.append(name)

    mesh_objects = ", ".join(f'"{name}"' for name in mesh_names)
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V120", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{block['substrate_relative_permittivity']}", "dielectric_loss_tangent:=", "{block['substrate_loss_tangent']}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h:.7f}, {board_l:.7f}, {board_w:.7f}, {h:.7f}, "RO5880_V120", True
CreateBox oEditor, "Ground", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h-copper:.7f}, {board_l:.7f}, {board_w:.7f}, {copper:.7f}, "copper", False
{chr(10).join(geometry)}
{chr(10).join(assignments)}
CreateBox oEditor, "AirRegion", {-board_l/2-4:.7f}, {-board_w/2-4:.7f}, -4, {board_l+8:.7f}, {board_w+8:.7f}, 8, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:V120_SparseGraphMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({mesh_objects}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(block['mesh_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {int(block['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", {int(block['minimum_converged_passes'])}, "PercentRefinement:=", {float(block['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "PortAccuracy:=", 2)
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "9.96GHz", "RangeEnd:=", "10.04GHz", "RangeCount:=", 3, "Type:=", "Discrete", "SaveFields:=", False, "SaveRadFields:=", False, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
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
vars = oSolutions.ListVariations("Setup_10GHz:Sweep_Gate3")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_Gate3"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    out = resolve(config["output_directory"])
    case = out / "physical_s8_direct01"
    return out, case, case / "v120_sparse_graph_physical_front_direct01.s8p"


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    out, case, touchstone = paths(config)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.20 physical output: {out}")
    case.mkdir(parents=True)
    project = case / "v120_sparse_graph_physical_front_direct01.aedt"
    build = case / "build.vbs"
    solve = case / "solve_export.vbs"
    build.write_text(builder_text(project, config), encoding="ascii")
    solve.write_text(solver_text(project, touchstone), encoding="ascii")
    manifest = {
        "project_path": str(project.resolve()),
        "touchstone_path": str(touchstone.resolve()),
        "builder_path": str(build.resolve()),
        "solver_path": str(solve.resolve()),
        "pre_reference_ports": [f"PRE_{index}" for index in range(4)],
        "post_reference_ports": [f"POST_{index}" for index in range(4)],
        "physical_graph": read_json(resolve(config["synthesis_config"]))["manufacturable_graph_pairs"],
        "evidence_scope": "single-stage distributed coupled-line plus local sparse RLC loading; not integrated antenna HFSS",
    }
    write_json(case / "case_manifest.json", manifest)
    write_json(out / "config_snapshot.json", config)
    return manifest


def run(config: dict[str, Any]) -> dict[str, Any]:
    _, case, touchstone = paths(config)
    manifest = read_json(case / "case_manifest.json")
    free = memory_available_gb()
    minimum = float(config["physical_block"]["minimum_free_memory_gb"])
    if math.isfinite(free) and free < minimum:
        raise MemoryError(f"Only {free:.2f} GiB free; {minimum:.2f} GiB required")
    if touchstone.exists() and touchstone.stat().st_size > 100:
        return {"status": "already_complete", "free_memory_gb": free}
    with (case / "build.log").open("w", encoding="utf-8") as handle:
        build = subprocess.run([str(resolve(config["ansys_executable"])), "-RunScriptAndExit", manifest["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    solve_code = None
    if build.returncode == 0:
        with (case / "solve_export.log").open("w", encoding="utf-8") as handle:
            solve = subprocess.run([str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", manifest["solver_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        solve_code = int(solve.returncode)
    result = {"build_return_code": int(build.returncode), "solve_return_code": solve_code, "free_memory_gb_before": free}
    write_json(case / "run_summary.json", result)
    return result


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out, case, touchstone = paths(config)
    if not touchstone.exists() or touchstone.stat().st_size < 100:
        raise FileNotFoundError("Physical S8 Touchstone is incomplete")
    manifest = read_json(case / "case_manifest.json")
    names = touchstone_port_names(touchstone)
    desired_names = manifest["pre_reference_ports"] + manifest["post_reference_ports"]
    if set(names) != set(desired_names):
        raise RuntimeError(f"Unexpected port names: {names}")
    frequencies, physical = reordered_network(touchstone, desired_names, 8)
    synthesis_config = read_json(resolve(config["synthesis_config"]))
    synthesis = read_json(resolve(config["synthesis_summary"]))
    graph = [tuple(int(value) for value in pair) for pair in synthesis_config["manufacturable_graph_pairs"]]
    _, series_ground, series_pair, input_ground, input_pair, output_ground, output_pair = unpack(np.asarray(synthesis["optimized_parameters"], dtype=float))
    target = np.stack([
        sparse_pi_s8(float(frequency), series_ground, series_pair, input_ground, input_pair, output_ground, output_pair, graph, synthesis_config)
        for frequency in frequencies
    ])
    integrated_f, integrated = reordered_network(resolve(config["integrated_v118_s4"]), [f"PRE_{index}" for index in range(4)], 4)
    antenna_f, antenna = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    feed_f, feed = reordered_network(resolve(config["validated_feed_s8"]), desired_names, 8)
    if not all(np.allclose(frequencies, item) for item in (integrated_f, antenna_f, feed_f)):
        raise RuntimeError("Frequency grids differ")
    effective_load = np.stack([deembed_load(integrated[index], feed[index]) for index in range(3)])
    desired = np.stack([terminate_network(feed[index], antenna[index])[0] for index in range(3)])
    rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    frequency_rows: list[dict[str, Any]] = []
    for index, frequency in enumerate(frequencies):
        aligned, phases, target_delta = phase_align(physical[index], target[index])
        post = terminate_network(aligned, effective_load[index])[0]
        corrected = terminate_network(feed[index], post)[0]
        selected = np.asarray([abs(float(row["frequency_ghz"]) - float(frequency)) <= 1.0e-9 for row in rows])
        active_rl, total_rl = active_metrics(corrected, vectors[selected].T, considered[selected].T)
        matched_s, load_incident, load_reflected = terminate_network(aligned, np.zeros((4, 4), dtype=complex))
        accepted = 1.0 - np.sum(np.abs(matched_s) ** 2, axis=0)
        delivered = np.sum(np.abs(load_incident) ** 2, axis=0) - np.sum(np.abs(load_reflected) ** 2, axis=0)
        frequency_rows.append({
            "frequency_ghz": float(frequency),
            "physical_vs_target_s8_max_abs_delta": target_delta,
            "active_rl_min_db": active_rl,
            "total_rl_min_db": total_rl,
            "corrected_vs_target_max_abs_delta_s": float(np.max(np.abs(corrected - desired[index]))),
            "network_efficiency_min": float(np.min(delivered / np.maximum(accepted, EPS))),
            "reference_phase_deg": json.dumps(phases),
        })
    profile = profile_metrics(case)
    reciprocity = float(np.max(np.abs(physical - np.transpose(physical, (0, 2, 1)))))
    passivity = float(max(np.max(np.linalg.svd(matrix, compute_uv=False)) for matrix in physical))
    summary = {
        **read_json(case / "run_summary.json"),
        **profile,
        "reciprocity_error_max": reciprocity,
        "passivity_sigma_max": passivity,
        "physical_vs_target_s8_max_abs_delta": max(row["physical_vs_target_s8_max_abs_delta"] for row in frequency_rows),
        "active_rl_min_db": min(row["active_rl_min_db"] for row in frequency_rows),
        "total_rl_min_db": min(row["total_rl_min_db"] for row in frequency_rows),
        "corrected_vs_target_max_abs_delta_s": max(row["corrected_vs_target_max_abs_delta_s"] for row in frequency_rows),
        "network_efficiency_min": min(row["network_efficiency_min"] for row in frequency_rows),
        "evidence_level": manifest["evidence_scope"],
        "exported_port_order": names,
        "analysis_port_order": desired_names,
    }
    gates = config["gates"]
    gate = bool(
        summary.get("converged") is True
        and float(summary.get("final_delta_s") or math.inf) <= float(gates["maximum_final_delta_s"])
        and reciprocity <= float(gates["maximum_reciprocity_error"])
        and passivity <= float(gates["maximum_passivity_sigma"])
        and summary["network_efficiency_min"] >= float(gates["minimum_network_efficiency"])
        and summary["active_rl_min_db"] >= float(gates["minimum_active_rl_db"])
        and summary["total_rl_min_db"] >= float(gates["minimum_total_rl_db"])
        and summary["corrected_vs_target_max_abs_delta_s"] <= float(gates["maximum_corrected_vs_target_abs_delta_s"])
        and summary["physical_vs_target_s8_max_abs_delta"] <= float(gates["maximum_physical_vs_target_s8_abs_delta"])
    )
    summary["physical_front_gate_pass"] = gate
    decision = {
        "allow_integrated_2x2": gate,
        "allow_independent_repeat": gate,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "next_action": "integrate the frozen sparse graph into one 2x2 HFSS smoke" if gate else "stop HFSS expansion and remap the coupled-line widths/gaps within the same sparse graph",
    }
    write_csv(case / "frequency_gate_metrics.csv", frequency_rows)
    write_json(case / "analysis.json", summary)
    write_json(out / "stage_decision.json", decision)
    return {"analysis": summary, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out, case, touchstone = paths(config)
    return {"prepared": (case / "case_manifest.json").exists(), "touchstone_complete": touchstone.exists() and touchstone.stat().st_size > 100, "analyzed": (case / "analysis.json").exists(), "free_memory_gb": memory_available_gb(), "output_directory": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), default="status")
    args = parser.parse_args()
    config = read_json(args.config)
    result = {"prepare": prepare, "run": run, "analyze": analyze, "status": status}[args.mode](config)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
