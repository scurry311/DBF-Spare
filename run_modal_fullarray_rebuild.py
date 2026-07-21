"""Gate, rebuild, solve, and analyze the 16x16 modal-network antenna candidate."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_full_s256p_active_return import parse_touchstone
from design_modal_subarray_network import (
    aggregate_case_metrics,
    case_metrics,
    passive_metrics,
    write_csv,
    write_touchstone,
)
from design_port_class_matching import compose_nonuniform_network


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "hfss_outputs" / "modal_decoupling_20260714_run01"
DEFAULT_DESIGN_DIR = RUN_ROOT / "s16_design"
DEFAULT_OUT = RUN_ROOT / "fullarray"
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
TRUSTED_S16_BUILDER = (
    ROOT
    / "hfss_outputs"
    / "geometry_feed_smoke_20260714_run03"
    / "retuned_tload_3p0_l10p4"
    / "build_retuned_tload_3p0_l10p4.vbs"
)
DESIGN_NAME = "URA16_Quick_10GHz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "build", "solve", "run", "status", "analyze"), default="status"
    )
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    return parser.parse_args()


def vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


def write_full_builder(path: Path, project_path: Path) -> None:
    text = TRUSTED_S16_BUILDER.read_text(encoding="utf-8-sig")
    old_project_line = next(line for line in text.splitlines() if line.startswith("projectPath = "))
    text = replace_once(text, old_project_line, f'projectPath = "{vp(project_path)}"')
    text = replace_once(text, "nx = 4", "nx = 16")
    text = replace_once(text, "ny = 4", "ny = 16")
    path.write_text(text, encoding="ascii")


def write_solve_export(path: Path, project_path: Path, touchstone: Path, port_order: Path) -> None:
    text = f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, fso, sources, vars, variation, i, pf
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vp(project_path)}"
Set oProject = oDesktop.SetActiveProject("{project_path.stem}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
sources = oSolutions.GetAllSources()
Set fso = CreateObject("Scripting.FileSystemObject")
Set pf = fso.CreateTextFile("{vp(port_order)}", True)
pf.WriteLine "touchstone_index,source_name,port_name"
For i = LBound(sources) To UBound(sources)
    pf.WriteLine CStr(i + 1) & "," & CStr(sources(i)) & "," & Split(CStr(sources(i)), ":")(0)
Next
pf.Close
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "{vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''
    path.write_text(text, encoding="ascii")


def read_gate(design_dir: Path) -> dict[str, Any]:
    summary_path = design_dir / "design_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("allow_16x16_rebuild")):
        raise RuntimeError("S16 smoke gate blocks the 16x16 rebuild")
    return summary


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    gate = read_gate(args.design_dir)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    project = args.out_dir / "ura16_tload3_l10p4_modal_candidate.aedt"
    raw_touchstone = args.out_dir / "ura16_tload3_l10p4_raw_hfss.s256p"
    builder = args.out_dir / "build_fullarray.vbs"
    solver = args.out_dir / "solve_export_raw_s256.vbs"
    port_order = args.out_dir / "aedt_port_order.csv"
    write_full_builder(builder, project)
    write_solve_export(solver, project, raw_touchstone, port_order)
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "s16_gate": gate,
        "project": str(project),
        "builder": str(builder),
        "solver": str(solver),
        "raw_hfss_s256": str(raw_touchstone),
        "network_cascaded_s256": str(
            args.out_dir / "ura16_tload3_l10p4_modal_network_circuit_cascade.s256p"
        ),
        "note": "The raw S256 is HFSS. The cascaded S256 is a circuit calculation and is labeled separately.",
    }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_step(args: argparse.Namespace, step: str) -> dict[str, Any]:
    if not (args.out_dir / "prepare_summary.json").exists():
        prepare(args)
    script = args.out_dir / ("build_fullarray.vbs" if step == "build" else "solve_export_raw_s256.vbs")
    log = args.out_dir / ("build_fullarray.log" if step == "build" else "solve_export_raw_s256.log")
    command = [str(args.ansys_exe)]
    if step == "solve":
        command.append("-ng")
    command.extend(("-RunScriptAndExit", str(script)))
    started = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    result = {
        "step": step,
        "return_code": int(completed.returncode),
        "elapsed_seconds": time.time() - started,
        "log": str(log),
    }
    (args.out_dir / f"{step}_status.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"AEDT {step} failed with code {completed.returncode}; see {log}")
    return result


def fullarray_transform(local_transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.zeros((256, 256), dtype=np.float64)
    mode_slot = np.empty(256, dtype=int)
    for x0 in range(0, 16, 4):
        for y0 in range(0, 16, 4):
            indices = [(x0 + dx) * 16 + y0 + dy for dx in range(4) for dy in range(4)]
            transform[np.ix_(indices, indices)] = local_transform
            mode_slot[indices] = np.arange(16)
    return transform, mode_slot


def load_full_scenarios(dataset_path: Path) -> dict[str, np.ndarray]:
    dataset = np.load(dataset_path, allow_pickle=False)
    weight_ri = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float64)
    weights = weight_ri[:, :, 0] + 1j * weight_ri[:, :, 1]
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1.0e-15)
    k_values = np.asarray(dataset["k_values"], dtype=int)
    targets = np.asarray(dataset["targets_deg"], dtype=np.float64)
    max_theta = np.asarray(
        [np.max(targets[index, : k_values[index], 0]) for index in range(len(k_values))]
    )
    return {
        "weights": weights,
        "masks": np.asarray(dataset["masks"], dtype=bool),
        "k": k_values,
        "ratio": np.asarray(dataset["active_ratios_actual"], dtype=float),
        "max_theta": max_theta,
        "large_scan": max_theta >= 45.0,
    }


def read_convergence(out_dir: Path) -> dict[str, Any]:
    result_root = out_dir / "ura16_tload3_l10p4_modal_candidate.aedtresults"
    profiles = list(result_root.rglob("*.profile"))
    if not profiles:
        return {
            "profile": "",
            "adaptive_pass_count": 0,
            "max_mag_delta_s_by_pass": [],
            "final_max_mag_delta_s": float("nan"),
            "converged": False,
        }
    profile = max(profiles, key=lambda path: path.stat().st_mtime)
    text = profile.read_text(encoding="utf-8", errors="replace")
    pass_count = len(re.findall(r"Name='Adaptive Pass \d+'", text))
    delta_values = [
        float(value)
        for value in re.findall(r"Max Mag\. Delta S\\',\s*([0-9.eE+-]+)", text)
    ]
    explicitly_not_converged = "Adaptive Passes did not converge" in text
    final_delta = delta_values[-1] if delta_values else float("nan")
    return {
        "profile": str(profile),
        "adaptive_pass_count": pass_count,
        "max_mag_delta_s_by_pass": delta_values,
        "final_max_mag_delta_s": final_delta,
        "converged": bool(
            delta_values and final_delta <= 0.05 and not explicitly_not_converged
        ),
        "explicit_not_converged_marker": explicitly_not_converged,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    read_gate(args.design_dir)
    raw_path = args.out_dir / "ura16_tload3_l10p4_raw_hfss.s256p"
    parsed = parse_touchstone(raw_path)
    if parsed["s_parameters"].shape != (1, 256, 256):
        raise RuntimeError(f"Expected one S256 matrix, got {parsed['s_parameters'].shape}")
    expected_ports = [f"P{index:03d}" for index in range(256)]
    if list(parsed["port_names"]) != expected_ports:
        raise RuntimeError("Raw S256 port order is not P000-P255")
    raw_s = np.asarray(parsed["s_parameters"][0], dtype=np.complex128)
    convergence = read_convergence(args.out_dir)
    network = np.load(args.design_dir / "selected_modal_network_s16.npz", allow_pickle=False)
    local_transform = np.asarray(network["mode_transform"], dtype=np.float64)
    transform, mode_slot = fullarray_transform(local_transform)
    series_x = np.asarray(network["series_reactance_ohm"], dtype=float)[mode_slot]
    shunt_b = np.asarray(network["shunt_susceptance_siemens"], dtype=float)[mode_slot]
    modal_s = transform.T @ raw_s @ transform
    composite_modal, antenna_wave_map_modal, network_parameters = compose_nonuniform_network(
        modal_s,
        series_x,
        shunt_b,
        float(parsed["reference_impedance_ohm"]),
    )
    composite_s = transform @ composite_modal @ transform.T
    antenna_wave_map = transform @ antenna_wave_map_modal @ transform.T
    scenarios = load_full_scenarios(args.dataset)
    raw_cases = case_metrics(raw_s, scenarios["weights"], scenarios["masks"], 10.0, -30.0)
    composite_cases = case_metrics(
        composite_s, scenarios["weights"], scenarios["masks"], 10.0, -30.0
    )
    case_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for model, evaluated in (("raw_hfss_s256", raw_cases), ("modal_circuit_cascade_s256", composite_cases)):
        cases, groups = aggregate_case_metrics(model, evaluated, scenarios)
        case_rows.extend(cases)
        group_rows.extend(groups)
    write_csv(args.out_dir / "fullarray_active_return_case_metrics.csv", case_rows)
    write_csv(args.out_dir / "fullarray_active_return_group_summary.csv", group_rows)
    cascade_path = args.out_dir / "ura16_tload3_l10p4_modal_network_circuit_cascade.s256p"
    write_touchstone(
        cascade_path,
        composite_s,
        float(parsed["frequency_hz"][0]),
        float(parsed["reference_impedance_ohm"]),
    )
    np.savez_compressed(
        args.out_dir / "fullarray_raw_and_modal_network_s256.npz",
        raw_hfss_s=raw_s.astype(np.complex64),
        modal_circuit_cascade_s=composite_s.astype(np.complex64),
        fullarray_mode_transform=transform.astype(np.float32),
        antenna_incident_wave_map=antenna_wave_map.astype(np.complex64),
        matching_network_parameters=network_parameters.astype(np.complex64),
        series_reactance_ohm=series_x,
        shunt_susceptance_siemens=shunt_b,
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_s256_is_hfss_fullwave": True,
        "cascaded_s256_is_hfss_fullwave": False,
        "raw_s256": str(raw_path),
        "circuit_cascaded_s256": str(cascade_path),
        "hfss_convergence": convergence,
        "eligible_as_fullwave_training_label": bool(convergence["converged"]),
        "raw_metrics": passive_metrics(raw_s),
        "circuit_cascade_metrics": passive_metrics(composite_s),
        "raw_all_active_10db_pass_rate": float(np.mean(raw_cases["all_active_pass"])),
        "cascade_all_active_10db_pass_rate": float(
            np.mean(composite_cases["all_active_pass"])
        ),
        "raw_total_10db_pass_rate": float(np.mean(raw_cases["total_pass"])),
        "cascade_total_10db_pass_rate": float(np.mean(composite_cases["total_pass"])),
        "next_gate": (
            "First obtain a converged raw antenna S256. Then build a physical feed substrate and "
            "validate one 4x4 network in HFSS/circuit co-simulation before generating new labels."
        ),
    }
    (args.out_dir / "fullarray_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def status(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "prepared": args.out_dir / "prepare_summary.json",
        "project": args.out_dir / "ura16_tload3_l10p4_modal_candidate.aedt",
        "raw_s256": args.out_dir / "ura16_tload3_l10p4_raw_hfss.s256p",
        "analysis": args.out_dir / "fullarray_analysis_summary.json",
    }
    result: dict[str, Any] = {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    result.update(
        {
            key: {
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for key, path in paths.items()
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode in ("build", "solve"):
        result = run_step(args, args.mode)
    elif args.mode == "run":
        result = {"build": run_step(args, "build"), "solve": run_step(args, "solve")}
    elif args.mode == "analyze":
        result = analyze(args)
    else:
        result = status(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
