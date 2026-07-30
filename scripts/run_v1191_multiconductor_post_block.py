#!/usr/bin/env python3
"""Build and gate one physical nonuniform four-conductor POST S8 block."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from design_v115_grounded_modal_network import terminate_network
from design_v119_multiport_post_decoupler import active_metrics, deembed_load, reordered_network
from run_v114_small_cell_broadband_feed import load_stimuli, memory_available_gb, parse_touchstone, profile_metrics
from run_v115_physical_modal_feed_fixture import helpers_vbs, touchstone_port_names, vp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v1191_multiconductor_post_block.json"
DESIGN_NAME = "V1191_Multiconductor_POST_S8"
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


def port_positions(block: dict[str, Any]) -> dict[int, float]:
    order = [int(value) for value in block["port_order_from_negative_y"]]
    widths = [float(value) for value in block["trace_width_mm_by_port"]]
    gaps = [float(value) for value in block["adjacent_gap_mm"]]
    centers = [0.0]
    for index in range(3):
        centers.append(
            centers[-1]
            + widths[order[index]] / 2.0
            + gaps[index]
            + widths[order[index + 1]] / 2.0
        )
    offset = 0.5 * (centers[0] + centers[-1])
    return {port: center - offset for port, center in zip(order, centers)}


def positions_for_gaps(block: dict[str, Any], gaps: list[float]) -> dict[int, float]:
    local = {**block, "adjacent_gap_mm": gaps}
    return port_positions(local)


def builder_text(project: Path, config: dict[str, Any]) -> str:
    block = config["physical_block"]
    h = float(block["substrate_thickness_mm"])
    copper = float(block["copper_thickness_mm"])
    board_l = float(block["board_length_mm"])
    board_w = float(block["board_width_mm"])
    sections = block.get("sections")
    if sections:
        first_length = float(sections[0]["length_mm"])
        second_length = float(sections[1]["length_mm"])
        transition_length = float(block["transition_length_mm"])
        trace_l = first_length + transition_length + second_length
        first_positions = positions_for_gaps(
            block, [float(value) for value in sections[0]["adjacent_gap_mm"]]
        )
        second_positions = positions_for_gaps(
            block, [float(value) for value in sections[1]["adjacent_gap_mm"]]
        )
    else:
        first_length = trace_l = float(block["trace_length_mm"])
        second_length = transition_length = 0.0
        first_positions = second_positions = port_positions(block)
    widths = [float(value) for value in block["trace_width_mm_by_port"]]
    positions = first_positions
    pre_x = -trace_l / 2.0
    post_x = trace_l / 2.0
    geometry = []
    assignments = []
    trace_names = []
    for port in range(4):
        width = widths[port]
        y_value = first_positions[port]
        trace_name = f"CoupledTrace_{port}"
        pre_sheet = f"PrePortSheet_{port}"
        post_sheet = f"PostPortSheet_{port}"
        trace_names.append(trace_name)
        trace_geometry = []
        if sections:
            transition_start = pre_x + first_length
            transition_steps = int(block.get("transition_steps", 8))
            trace_geometry.append(
                f'CreateBox oEditor, "{trace_name}", {pre_x:.7f}, {y_value-width/2:.7f}, 0, {first_length+0.02:.7f}, {width:.7f}, {copper:.7f}, "copper", False'
            )
            previous_name = trace_name
            for step in range(transition_steps):
                fraction = (step + 0.5) / transition_steps
                center_y = (1.0 - fraction) * first_positions[port] + fraction * second_positions[port]
                segment_name = f"Transition_{port}_{step}"
                x_start = transition_start + step * transition_length / transition_steps - 0.02
                segment_length = transition_length / transition_steps + 0.04
                trace_geometry.extend(
                    [
                        f'CreateBox oEditor, "{segment_name}", {x_start:.7f}, {center_y-width/2:.7f}, 0, {segment_length:.7f}, {width:.7f}, {copper:.7f}, "copper", False',
                        f'UniteObjects oEditor, "{previous_name}", "{segment_name}"',
                    ]
                )
            second_name = f"CoupledTraceSecond_{port}"
            trace_geometry.extend(
                [
                    f'CreateBox oEditor, "{second_name}", {transition_start+transition_length-0.02:.7f}, {second_positions[port]-width/2:.7f}, 0, {second_length+0.02:.7f}, {width:.7f}, {copper:.7f}, "copper", False',
                    f'UniteObjects oEditor, "{previous_name}", "{second_name}"',
                ]
            )
        else:
            trace_geometry.append(
                f'CreateBox oEditor, "{trace_name}", {pre_x:.7f}, {y_value-width/2:.7f}, 0, {trace_l:.7f}, {width:.7f}, {copper:.7f}, "copper", False'
            )
        geometry.extend(
            trace_geometry
            + [
                f'CreateSheet oEditor, "{pre_sheet}", "X", {pre_x:.7f}, {y_value-width/2:.7f}, {-h:.7f}, {width:.7f}, {h:.7f}',
                f'CreateSheet oEditor, "{post_sheet}", "X", {post_x:.7f}, {second_positions[port]-width/2:.7f}, {-h:.7f}, {width:.7f}, {h:.7f}',
            ]
        )
    for port in range(4):
        y_value = first_positions[port]
        assignments.append(
            f'AssignPort oBoundary, "PRE_{port}", "PrePortSheet_{port}", {pre_x:.7f}, {y_value:.7f}, 0, {pre_x:.7f}, {y_value:.7f}, {-h:.7f}'
        )
    for port in range(4):
        y_value = second_positions[port]
        assignments.append(
            f'AssignPort oBoundary, "POST_{port}", "PostPortSheet_{port}", {post_x:.7f}, {y_value:.7f}, 0, {post_x:.7f}, {y_value:.7f}, {-h:.7f}'
        )
    mesh_objects = ", ".join(f'"{name}"' for name in trace_names)
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V1191", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{block['substrate_relative_permittivity']}", "dielectric_loss_tangent:=", "{block['substrate_loss_tangent']}")
oProject.InsertDesign "HFSS", "{DESIGN_NAME}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h:.7f}, {board_l:.7f}, {board_w:.7f}, {h:.7f}, "RO5880_V1191", True
CreateBox oEditor, "Ground", {-board_l/2:.7f}, {-board_w/2:.7f}, {-h-copper:.7f}, {board_l:.7f}, {board_w:.7f}, {copper:.7f}, "copper", False
{chr(10).join(geometry)}
{chr(10).join(assignments)}
CreateBox oEditor, "AirRegion", {-board_l/2-4:.7f}, {-board_w/2-4:.7f}, -4, {board_l+8:.7f}, {board_w+8:.7f}, 8, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:V1191_CoupledTraceMesh", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({mesh_objects}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(block['mesh_max_length_mm']):.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {int(block['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", {int(block['minimum_converged_passes'])}, "PercentRefinement:=", {float(block['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "PortAccuracy:=", 2)
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_Gate3", "IsEnabled:=", True, "RangeType:=", "LinearCount", "RangeStart:=", "9.96GHz", "RangeEnd:=", "10.04GHz", "RangeCount:=", 3, "Type:=", "Discrete", "SaveFields:=", False, "SaveRadFields:=", False, "UseFullBasis:=", True, "EnforcePassivity:=", False, "EnforceCausality:=", False, "SMatrixOnlySolveMode:=", "Auto")
oProject.SaveAs "{vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub UniteObjects(editor, firstName, secondName)
    editor.Unite Array("NAME:Selections", "Selections:=", firstName & "," & secondName), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub

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
    out_dir = resolve(config["output_directory"])
    case = out_dir / "physical_s8_direct01"
    return out_dir, case, case / "v1191_multiconductor_post_direct01.s8p"


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    out_dir, case, touchstone = paths(config)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.19.1 output: {out_dir}")
    case.mkdir(parents=True)
    project = case / "v1191_multiconductor_post_direct01.aedt"
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
        "pre_port_y_mm": (
            positions_for_gaps(
                config["physical_block"],
                [float(value) for value in config["physical_block"]["sections"][0]["adjacent_gap_mm"]],
            )
            if config["physical_block"].get("sections")
            else port_positions(config["physical_block"])
        ),
        "post_port_y_mm": (
            positions_for_gaps(
                config["physical_block"],
                [float(value) for value in config["physical_block"]["sections"][1]["adjacent_gap_mm"]],
            )
            if config["physical_block"].get("sections")
            else port_positions(config["physical_block"])
        ),
    }
    write_json(case / "case_manifest.json", manifest)
    write_json(out_dir / "config_snapshot.json", config)
    return manifest


def run(config: dict[str, Any]) -> dict[str, Any]:
    _, case, touchstone = paths(config)
    manifest = read_json(case / "case_manifest.json")
    free_memory = memory_available_gb()
    minimum = float(config["physical_block"]["minimum_free_memory_gb"])
    if math.isfinite(free_memory) and free_memory < minimum:
        raise MemoryError(f"Only {free_memory:.2f} GiB free; {minimum:.2f} GiB required")
    if touchstone.exists() and touchstone.stat().st_size > 100:
        return {"status": "already_complete", "free_memory_gb": free_memory}
    with (case / "build.log").open("w", encoding="utf-8") as handle:
        build = subprocess.run([str(resolve(config["ansys_executable"])), "-RunScriptAndExit", manifest["builder_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    solve_code = None
    if build.returncode == 0:
        with (case / "solve_export.log").open("w", encoding="utf-8") as handle:
            solve = subprocess.run([str(resolve(config["ansys_executable"])), "-ng", "-RunScriptAndExit", manifest["solver_path"]], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        solve_code = int(solve.returncode)
    summary = {"build_return_code": int(build.returncode), "solve_return_code": solve_code, "free_memory_gb_before": free_memory}
    write_json(case / "run_summary.json", summary)
    return summary


def phase_align(physical: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, list[float], float]:
    def residual(free: np.ndarray) -> np.ndarray:
        phases = np.r_[0.0, free]
        delay = np.diag(np.exp(1j * phases))
        error = delay @ physical @ delay - target
        return np.r_[error.real.ravel(), error.imag.ravel()]
    fit = least_squares(residual, np.zeros(7), max_nfev=20000)
    phases = np.r_[0.0, fit.x]
    delay = np.diag(np.exp(1j * phases))
    aligned = delay @ physical @ delay
    return aligned, np.degrees(phases).tolist(), float(np.max(np.abs(aligned - target)))


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    out_dir, case, touchstone = paths(config)
    if not touchstone.exists() or touchstone.stat().st_size < 100:
        raise FileNotFoundError("Physical S8 Touchstone is incomplete")
    manifest = read_json(case / "case_manifest.json")
    frequencies, physical = reordered_network(touchstone, manifest["pre_reference_ports"] + manifest["post_reference_ports"], 8)
    target_f, target = parse_touchstone(resolve(config["target_s8"]), 8)
    integrated_f, integrated = reordered_network(resolve(config["integrated_v118_s4"]), [f"PRE_{index}" for index in range(4)], 4)
    antenna_f, antenna = parse_touchstone(resolve(config["trusted_antenna_s4"]), 4)
    feed_f, feed = reordered_network(resolve(config["validated_feed_s8"]), [f"PRE_{index}" for index in range(4)] + [f"POST_{index}" for index in range(4)], 8)
    if not all(np.allclose(frequencies, item) for item in (target_f, integrated_f, antenna_f, feed_f)):
        raise RuntimeError("Frequency grids differ")
    effective_load = np.stack([deembed_load(integrated[index], feed[index]) for index in range(3)])
    desired = np.stack([terminate_network(feed[index], antenna[index])[0] for index in range(3)])
    rows, vectors, considered = load_stimuli(resolve(config["trusted_stimulus_root"]))
    side = np.asarray([int(row["side"]) == 2 for row in rows])
    rows = [row for row, keep in zip(rows, side) if keep]
    vectors = vectors[side, :4]
    considered = considered[side, :4]
    frequency_rows = []
    aligned_all = []
    for index, frequency in enumerate(frequencies):
        aligned, phases, target_delta = phase_align(physical[index], target[index])
        aligned_all.append(aligned)
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
            "offdiagonal_transmission_max": float(np.max(np.abs(aligned[4:, :4] - np.diag(np.diag(aligned[4:, :4]))))),
        })
    aligned_array = np.stack(aligned_all)
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
        "offdiagonal_transmission_max": max(row["offdiagonal_transmission_max"] for row in frequency_rows),
        "evidence_level": "standalone physical HFSS S8 with fitted external reference phases; not integrated antenna-network HFSS",
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
    summary["physical_s8_gate_pass"] = gate
    topology_name = (
        "two-section noncommuting multiconductor"
        if config["physical_block"].get("sections")
        else "single-section multiconductor"
    )
    decision = {
        "allow_integrated_2x2": gate,
        "allow_independent_repeat": gate,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "evaluated_topology": topology_name,
        "next_action": (
            "integrate the frozen block into one 2x2 HFSS smoke"
            if gate
            else "stop this physical topology; reoptimize the antenna/feed transition under a realizable sparse coupling graph"
        ),
    }
    write_csv(case / "frequency_gate_metrics.csv", frequency_rows)
    write_json(case / "analysis.json", summary)
    write_json(out_dir / "stage_decision.json", decision)
    return {"analysis": summary, "decision": decision}


def status(config: dict[str, Any]) -> dict[str, Any]:
    out_dir, case, touchstone = paths(config)
    return {"prepared": (case / "case_manifest.json").exists(), "touchstone_complete": touchstone.exists() and touchstone.stat().st_size > 100, "analyzed": (case / "analysis.json").exists(), "free_memory_gb": memory_available_gb(), "output_directory": str(out_dir)}


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
