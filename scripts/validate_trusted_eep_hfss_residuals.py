#!/usr/bin/env python3
"""Validate trusted 256-port EEP reconstruction and build new residual labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_eep_16port_operator import read_complex_field
from hfss_task_fullwave_validate import (
    combined_metrics,
    isolation_metrics,
    local_peak_db,
    pattern_grid_dirs,
)
from validate_eep_superposition_smoke import write_vbs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "validation_dataset"
)
DEFAULT_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_PROJECT = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep_fieldsolve_20260723_run05_ddm80"
    / "project"
    / "fixed_mesh_eep_ddm_fieldsolve_run05.aedt"
)
DEFAULT_OUT = (
    ROOT
    / "hfss_outputs"
    / "trusted_eep_residual_20260723_run02"
    / "eep_hfss_validation"
)
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN = "URA_GroundedPatch_10GHz"
SOLUTION = "Setup_Frozen_DDM_EEP_Run05 : LastAdaptive"
SPHERE = "InfiniteSphere_Theta0_90_Phi0_360"
ETA0 = 376.730313668
SERIES_L_NH = 0.533
SERIES_Q = 50.0
Z0 = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "analyze", "status"), required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--operator-path", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--project-path", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--design-name", default=DESIGN)
    parser.add_argument("--solution-name", default=SOLUTION)
    parser.add_argument("--sphere-name", default=SPHERE)
    parser.add_argument("--frequency-ghz", type=float, default=10.0)
    parser.add_argument("--cases-per-chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--nmse-max", type=float, default=1.0e-6)
    parser.add_argument("--magnitude-rmse-max-db", type=float, default=0.02)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def series_network_map(
    s_antenna: np.ndarray, frequency_hz: float
) -> tuple[np.ndarray, np.ndarray, complex]:
    series_z = 2.0 * np.pi * frequency_hz * SERIES_L_NH * 1.0e-9
    series_z = series_z / SERIES_Q + 1j * series_z
    normalized = series_z / Z0
    denominator = 2.0 + normalized
    s11 = normalized / denominator
    s22 = s11
    s21 = 2.0 / denominator
    s12 = s21
    identity = np.eye(s_antenna.shape[0], dtype=np.complex128)
    antenna_wave_map = np.linalg.inv(identity - s22 * s_antenna) @ (s21 * identity)
    composite_s = s11 * identity + s12 * s_antenna @ antenna_wave_map
    return composite_s, antenna_wave_map, series_z


def normalized_external_excitation(weight: np.ndarray, mask: np.ndarray) -> np.ndarray:
    value = np.conjugate(np.asarray(weight, dtype=np.complex128))
    value = np.where(np.asarray(mask, dtype=bool), value, 0.0)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-14:
        raise ValueError("Zero candidate excitation")
    return value / norm


def active_return_metrics(s_matrix: np.ndarray, excitation: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    reflected = s_matrix @ excitation
    driven = np.asarray(mask, dtype=bool) & (np.abs(excitation) >= 1.0e-8)
    gamma = np.abs(reflected[driven] / excitation[driven])
    worst_active_rl = float(np.min(-20.0 * np.log10(np.maximum(gamma, 1.0e-15)))) if gamma.size else float("nan")
    reflected_ratio = float(np.vdot(reflected, reflected).real / max(float(np.vdot(excitation, excitation).real), 1.0e-20))
    total_rl = float(-10.0 * np.log10(max(reflected_ratio, 1.0e-15)))
    return {
        "worst_active_rl_db": worst_active_rl,
        "total_rl_db": total_rl,
        "reflected_power_fraction": reflected_ratio,
    }


def source_csv(path: Path, ports: np.ndarray, excitation: np.ndarray) -> None:
    rows = []
    for port, value in zip(ports, excitation):
        rows.append(
            {
                "port_name": str(port),
                "incident_power_w": f"{abs(value) ** 2:.12e}",
                "phase_deg": f"{np.rad2deg(np.angle(value)):.9f}",
            }
        )
    write_csv(path, rows)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite validation output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_npz(args.dataset_dir / "dataset_arrays.npz")
    operator = np.load(args.operator_path, allow_pickle=False)
    s_raw = np.asarray(operator["s_raw"], dtype=np.complex128)
    s_expected = np.asarray(operator["s_matched"], dtype=np.complex128)
    s_matched, antenna_map, series_z = series_network_map(
        s_raw, float(args.frequency_ghz) * 1.0e9
    )
    matching_delta = float(np.max(np.abs(s_matched - s_expected)))
    if matching_delta > 1.0e-7:
        raise RuntimeError(f"Matching-network cascade mismatch: {matching_delta}")

    ports = np.asarray(operator["port_names"])
    masks = np.asarray(dataset["masks"], dtype=bool)
    task_weights = (
        dataset["task_weights_real_imag"][..., 0]
        + 1j * dataset["task_weights_real_imag"][..., 1]
    )
    combined_weights = (
        dataset["combined_weights_real_imag"][..., 0]
        + 1j * dataset["combined_weights_real_imag"][..., 1]
    )
    case_rows: list[dict[str, Any]] = []
    external: list[np.ndarray] = []
    antenna: list[np.ndarray] = []
    source_root = args.out_dir / "sources"
    source_root.mkdir()

    for candidate_index in range(dataset["candidate_indices"].size):
        valid_tasks = np.flatnonzero(dataset["task_valid"][candidate_index].astype(bool))
        case_specs: list[tuple[str, int, np.ndarray]] = [
            ("combined", -1, combined_weights[candidate_index])
        ]
        case_specs.extend(
            ("task", int(task_index), task_weights[candidate_index, :, int(task_index)])
            for task_index in valid_tasks
        )
        for case_kind, task_index, weight in case_specs:
            case_index = len(case_rows)
            suffix = "combined" if task_index < 0 else f"t{task_index}"
            case_id = f"c{candidate_index:03d}_{suffix}"
            a_external = normalized_external_excitation(weight, masks[candidate_index])
            a_antenna = antenna_map @ a_external
            source_path = source_root / f"sources_{case_id}.csv"
            source_csv(source_path, ports, a_antenna)
            metrics = active_return_metrics(s_matched, a_external, masks[candidate_index])
            case_rows.append(
                {
                    "case_index": case_index,
                    "case_id": case_id,
                    "candidate_index": candidate_index,
                    "sample_index": int(dataset["sample_indices"][candidate_index]),
                    "case_kind": case_kind,
                    "task_index": task_index,
                    "k": int(dataset["k_values"][candidate_index]),
                    "ratio": float(dataset["active_ratios_requested"][candidate_index]),
                    "source_csv": str(source_path.resolve()),
                    **metrics,
                }
            )
            external.append(a_external.astype(np.complex64))
            antenna.append(a_antenna.astype(np.complex64))

    np.savez_compressed(
        args.out_dir / "case_excitations.npz",
        case_ids=np.asarray([row["case_id"] for row in case_rows]),
        candidate_indices=np.asarray([row["candidate_index"] for row in case_rows], dtype=np.int64),
        task_indices=np.asarray([row["task_index"] for row in case_rows], dtype=np.int64),
        external_excitation=np.stack(external),
        antenna_excitation=np.stack(antenna),
        antenna_wave_map=antenna_map.astype(np.complex64),
        matched_s=s_matched.astype(np.complex64),
    )
    write_csv(args.out_dir / "case_manifest.csv", case_rows)

    chunks: list[dict[str, Any]] = []
    for start in range(0, len(case_rows), int(args.cases_per_chunk)):
        stop = min(len(case_rows), start + int(args.cases_per_chunk))
        chunk_id = f"cases_{start:03d}_{stop - 1:03d}"
        chunk_dir = args.out_dir / "chunks" / chunk_id
        chunk_dir.mkdir(parents=True)
        rows = case_rows[start:stop]
        vbs = chunk_dir / f"export_{chunk_id}.vbs"
        write_vbs(
            vbs,
            args.project_path,
            chunk_dir,
            [str(row["case_id"]) for row in rows],
            [Path(str(row["source_csv"])) for row in rows],
            design_name=str(args.design_name),
            solution_name=str(args.solution_name),
            sphere_name=str(args.sphere_name),
            frequency_ghz=float(args.frequency_ghz),
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_case": start,
                "stop_case_exclusive": stop,
                "case_count": stop - start,
                "chunk_dir": str(chunk_dir.resolve()),
                "vbs": str(vbs.resolve()),
            }
        )
    write_csv(args.out_dir / "chunk_manifest.csv", chunks)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": int(dataset["candidate_indices"].size),
        "independent_scene_count": int(np.unique(dataset["sample_indices"]).size),
        "case_count": len(case_rows),
        "chunk_count": len(chunks),
        "cases_per_chunk": int(args.cases_per_chunk),
        "matching_network": {
            "series_inductance_nh": SERIES_L_NH,
            "q": SERIES_Q,
            "series_impedance_ohm": [float(series_z.real), float(series_z.imag)],
            "cascade_vs_expected_max_abs": matching_delta,
        },
        "normalization": "one watt at matching-network external reference for each combined/task case",
        "hfss_antenna_excitation": "a_ant = antenna_wave_map @ a_external; no renormalization",
        "old_labels_included": False,
    }
    (args.out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def chunk_complete(row: dict[str, str]) -> bool:
    chunk_dir = Path(row["chunk_dir"])
    start = int(row["start_case"])
    stop = int(row["stop_case_exclusive"])
    cases = read_csv(chunk_dir.parents[1] / "case_manifest.csv")[start:stop]
    return all(
        (chunk_dir / f"direct_{case['case_id']}_complex.csv").exists()
        and (chunk_dir / f"direct_{case['case_id']}_complex.csv").stat().st_size >= 1000
        for case in cases
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    chunks = read_csv(args.out_dir / "chunk_manifest.csv")
    progress: list[dict[str, Any]] = []
    for row in chunks:
        if chunk_complete(row):
            progress.append({**row, "status": "already_complete", "return_code": 0})
            continue
        log = Path(row["chunk_dir"]) / "export.log"
        with log.open("w", encoding="utf-8", errors="replace") as handle:
            result = subprocess.run(
                [str(args.ansys_exe), "-ng", "-RunScriptAndExit", row["vbs"]],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        complete = chunk_complete(row)
        progress.append(
            {
                **row,
                "status": "complete" if complete else "failed_or_incomplete",
                "return_code": int(result.returncode),
            }
        )
        write_csv(args.out_dir / "run_progress.csv", progress)
        if result.returncode != 0 or not complete:
            break
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_count": len(chunks),
        "complete_chunk_count": sum(row["status"] in ("complete", "already_complete") for row in progress),
        "run_complete": len(progress) == len(chunks) and all(
            row["status"] in ("complete", "already_complete") for row in progress
        ),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def field_pattern(theta: np.ndarray, phi: np.ndarray, etheta: np.ndarray, ephi: np.ndarray) -> dict[str, np.ndarray]:
    power = np.abs(etheta) ** 2 + np.abs(ephi) ** 2
    gain_db = 10.0 * np.log10(np.maximum((2.0 * np.pi / ETA0) * power, 1.0e-30))
    return {
        "theta": theta,
        "phi": phi,
        "gain_db": gain_db.astype(np.float64),
        "dirs": pattern_grid_dirs(theta, phi),
    }


def pointing_error(pattern: dict[str, np.ndarray], targets: list[list[float]]) -> float:
    return float(
        max(local_peak_db(pattern, float(theta), float(phi), 5.0)[1] for theta, phi in targets)
    )


def margins(row: dict[str, Any]) -> float:
    values = [
        abs(float(row["hfss_psll_db"]) - 0.0) / 3.0,
        abs(float(row["hfss_nearest_iso_db"]) - 25.0) / 5.0,
        abs(float(row["hfss_local_iso_db"]) - 20.0) / 5.0,
        abs(float(row["all_case_worst_active_rl_db"]) - 10.0) / 2.0,
        abs(float(row["all_case_worst_total_rl_db"]) - 10.0) / 2.0,
    ]
    return float(min(values))


def build_group_split(rows: list[dict[str, Any]], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    by_scene: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_scene[int(row["sample_index"])].append(index)
    strata: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for scene, indices in by_scene.items():
        group = [rows[index] for index in indices]
        key = (
            max(int(row["k"]) for row in group),
            int(any(int(row["strict_gate20"]) == 1 for row in group)),
            int(any(int(row["hard_negative"]) == 1 for row in group)),
        )
        strata[key].append(scene)
    rng = np.random.default_rng(seed)
    split_by_scene: dict[int, int] = {}
    for scenes in strata.values():
        values = np.asarray(scenes, dtype=np.int64)
        rng.shuffle(values)
        n = values.size
        n_train = max(1, int(round(0.70 * n))) if n >= 3 else max(1, n - 1)
        n_val = max(1, int(round(0.15 * n))) if n >= 6 else (1 if n >= 3 else 0)
        if n_train + n_val >= n and n >= 2:
            n_train = max(1, n - n_val - 1)
        for pos, scene in enumerate(values):
            split_by_scene[int(scene)] = 0 if pos < n_train else (1 if pos < n_train + n_val else 2)
    split = np.asarray([split_by_scene[int(row["sample_index"])] for row in rows], dtype=np.int8)
    payload: dict[str, Any] = {
        "seed": seed,
        "group_key": "sample_index",
        "old_labels_included": False,
        "splits": {},
    }
    for split_value, name in enumerate(("train", "val", "test")):
        indices = np.flatnonzero(split == split_value)
        payload["splits"][name] = {
            "candidate_indices": indices.tolist(),
            "sample_indices": sorted({int(rows[index]["sample_index"]) for index in indices}),
            "candidate_count": int(indices.size),
            "scene_count": len({int(rows[index]["sample_index"]) for index in indices}),
        }
    sets = [set(payload["splits"][name]["sample_indices"]) for name in ("train", "val", "test")]
    payload["scene_leakage_free"] = not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    return split, payload


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_npz(args.dataset_dir / "dataset_arrays.npz")
    operator = np.load(args.operator_path, allow_pickle=False)
    cases = load_npz(args.out_dir / "case_excitations.npz")
    case_rows = read_csv(args.out_dir / "case_manifest.csv")
    chunks = read_csv(args.out_dir / "chunk_manifest.csv")
    chunk_by_case: dict[int, Path] = {}
    for row in chunks:
        for case_index in range(int(row["start_case"]), int(row["stop_case_exclusive"])):
            chunk_by_case[case_index] = Path(row["chunk_dir"])

    theta = np.asarray(operator["theta_deg"], dtype=np.float64)
    phi = np.asarray(operator["phi_deg"], dtype=np.float64)
    antenna_excitation = np.asarray(cases["antenna_excitation"], dtype=np.complex64)
    reconstructed_theta = antenna_excitation @ np.asarray(operator["etheta"], dtype=np.complex64)
    reconstructed_phi = antenna_excitation @ np.asarray(operator["ephi"], dtype=np.complex64)

    case_metrics: list[dict[str, Any]] = []
    candidate_case_indices: dict[int, list[int]] = defaultdict(list)
    direct_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for case_index, row in enumerate(case_rows):
        candidate_case_indices[int(row["candidate_index"])].append(case_index)
        path = chunk_by_case[case_index] / f"direct_{row['case_id']}_complex.csv"
        if not path.exists() or path.stat().st_size < 1000:
            case_metrics.append({**row, "complete": 0})
            continue
        angles, direct_theta, direct_phi = read_complex_field(path)
        if not (np.array_equal(theta, angles[:, 0]) and np.array_equal(phi, angles[:, 1])):
            raise RuntimeError(f"Grid mismatch for {row['case_id']}")
        direct_cache[case_index] = (direct_theta, direct_phi)
        direct = np.concatenate((direct_theta, direct_phi))
        reconstructed = np.concatenate((reconstructed_theta[case_index], reconstructed_phi[case_index]))
        error = reconstructed - direct
        nmse = float(np.sum(np.abs(error) ** 2) / max(float(np.sum(np.abs(direct) ** 2)), 1.0e-20))
        direct_db = 20.0 * np.log10(np.maximum(np.abs(direct), 1.0e-12))
        reconstruction_db = 20.0 * np.log10(np.maximum(np.abs(reconstructed), 1.0e-12))
        visible = direct_db >= float(np.max(direct_db) - 40.0)
        magnitude_rmse = float(np.sqrt(np.mean((reconstruction_db[visible] - direct_db[visible]) ** 2)))
        correlation = float(abs(np.vdot(direct, reconstructed)) / max(float(np.linalg.norm(direct) * np.linalg.norm(reconstructed)), 1.0e-20))
        case_metrics.append(
            {
                **row,
                "complete": 1,
                "complex_nmse": nmse,
                "magnitude_rmse_db_visible40": magnitude_rmse,
                "complex_correlation": correlation,
                "no_scale_pass": int(nmse <= float(args.nmse_max) and magnitude_rmse <= float(args.magnitude_rmse_max_db)),
            }
        )
    write_csv(args.out_dir / "case_reconstruction_metrics.csv", case_metrics)

    candidate_rows: list[dict[str, Any]] = []
    for candidate_index in range(dataset["candidate_indices"].size):
        indices = candidate_case_indices[candidate_index]
        if not indices or any(index not in direct_cache for index in indices):
            continue
        targets_array = dataset["targets_deg"][candidate_index][dataset["task_valid"][candidate_index].astype(bool)]
        targets = [[float(theta_i), float(phi_i)] for theta_i, phi_i in targets_array]
        eep_combined: dict[str, np.ndarray] | None = None
        hfss_combined: dict[str, np.ndarray] | None = None
        eep_tasks: dict[int, dict[str, np.ndarray]] = {}
        hfss_tasks: dict[int, dict[str, np.ndarray]] = {}
        active_rl_values = []
        total_rl_values = []
        for case_index in indices:
            case = case_rows[case_index]
            direct_theta, direct_phi = direct_cache[case_index]
            eep_pattern = field_pattern(theta, phi, reconstructed_theta[case_index], reconstructed_phi[case_index])
            hfss_pattern = field_pattern(theta, phi, direct_theta, direct_phi)
            active_rl_values.append(float(case["worst_active_rl_db"]))
            total_rl_values.append(float(case["total_rl_db"]))
            if case["case_kind"] == "combined":
                eep_combined = eep_pattern
                hfss_combined = hfss_pattern
            else:
                task_index = int(case["task_index"])
                eep_tasks[task_index] = eep_pattern
                hfss_tasks[task_index] = hfss_pattern
        if eep_combined is None or hfss_combined is None:
            continue
        eep_comb = combined_metrics(eep_combined, targets, target_radius_deg=5.0, sidelobe_exclusion_deg=8.0)
        hfss_comb = combined_metrics(hfss_combined, targets, target_radius_deg=5.0, sidelobe_exclusion_deg=8.0)
        eep_iso = isolation_metrics(eep_tasks, targets, target_radius_deg=5.0)
        hfss_iso = isolation_metrics(hfss_tasks, targets, target_radius_deg=5.0)
        eep_pointing = pointing_error(eep_combined, targets)
        hfss_pointing = pointing_error(hfss_combined, targets)
        eep_psll = float(eep_comb["combined_psll_to_weakest_peak_db"])
        hfss_psll = float(hfss_comb["combined_psll_to_weakest_peak_db"])
        eep_nearest = float(eep_iso["isolation_worst_nearest_db"])
        hfss_nearest = float(hfss_iso["isolation_worst_nearest_db"])
        eep_local = float(eep_iso["isolation_worst_local_db"])
        hfss_local = float(hfss_iso["isolation_worst_local_db"])
        eep_peak = float(eep_comb["combined_target_peak_min_db"])
        hfss_peak = float(hfss_comb["combined_target_peak_min_db"])
        all_active_rl = float(min(active_rl_values))
        all_total_rl = float(min(total_rl_values))
        fullwave_complete = bool(hfss_iso["isolation_complete"] == 1)
        gate15 = bool(fullwave_complete and hfss_psll <= 0.0 and hfss_nearest >= 25.0 and hfss_local >= 15.0)
        gate20 = bool(fullwave_complete and hfss_psll <= 0.0 and hfss_nearest >= 25.0 and hfss_local >= 20.0)
        mainlobe_gate = bool(
            hfss_peak >= eep_peak - 0.5
            and float(hfss_comb["combined_target_spread_db"]) <= 3.0
            and hfss_pointing <= 1.5
        )
        active_gate = bool(all_active_rl >= 10.0 and all_total_rl >= 10.0)
        eep_gate15 = bool(eep_psll <= 0.0 and eep_nearest >= 25.0 and eep_local >= 15.0)
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "sample_index": int(dataset["sample_indices"][candidate_index]),
                "sample_id": str(dataset["sample_ids"][candidate_index]),
                "scene_id": str(dataset["scene_ids"][candidate_index]),
                "source_dataset": str(dataset["source_dataset"][candidate_index]),
                "source_sample_index": int(dataset["source_sample_indices"][candidate_index]),
                "selection_role": str(dataset["selection_roles"][candidate_index]),
                "k": int(dataset["k_values"][candidate_index]),
                "ratio": float(dataset["active_ratios_requested"][candidate_index]),
                "active_count": int(dataset["num_active"][candidate_index]),
                "large_scan": int(dataset["large_scan"][candidate_index]),
                "min_target_separation_deg": float(dataset["min_target_separation_deg"][candidate_index]),
                "eep_psll_db": eep_psll,
                "hfss_psll_db": hfss_psll,
                "delta_psll_db": hfss_psll - eep_psll,
                "eep_nearest_iso_db": eep_nearest,
                "hfss_nearest_iso_db": hfss_nearest,
                "delta_nearest_iso_db": hfss_nearest - eep_nearest,
                "eep_local_iso_db": eep_local,
                "hfss_local_iso_db": hfss_local,
                "delta_local_iso_db": hfss_local - eep_local,
                "eep_mainlobe_gain_db": eep_peak,
                "hfss_mainlobe_gain_db": hfss_peak,
                "delta_mainlobe_gain_db": hfss_peak - eep_peak,
                "eep_target_spread_db": float(eep_comb["combined_target_spread_db"]),
                "hfss_target_spread_db": float(hfss_comb["combined_target_spread_db"]),
                "eep_pointing_error_deg": eep_pointing,
                "hfss_pointing_error_deg": hfss_pointing,
                "all_case_worst_active_rl_db": all_active_rl,
                "all_case_worst_total_rl_db": all_total_rl,
                "gate15": int(gate15),
                "strict_gate20": int(gate20),
                "mainlobe_gate": int(mainlobe_gate),
                "active_RL_gate": int(active_gate),
                "strict_engineering_gate": int(gate20 and mainlobe_gate and active_gate),
                "eep_gate15": int(eep_gate15),
                "hard_negative": int(eep_gate15 and not gate15),
                "hard_positive": int(gate20 and (int(dataset["k_values"][candidate_index]) == 6 or int(dataset["large_scan"][candidate_index]) == 1 or float(dataset["active_ratios_requested"][candidate_index]) <= 0.7)),
                "fullwave_complete": 1,
            }
        )
    for row in candidate_rows:
        row["near_boundary"] = int(margins(row) <= 1.0)
        priority = 4 * int(row["hard_negative"]) + 3 * int(row["near_boundary"]) + 2 * int(row["hard_positive"])
        priority += int(row["k"] == 6 and float(row["min_target_separation_deg"]) <= 10.0)
        row["training_priority"] = priority
    write_csv(args.out_dir / "candidate_residual_labels.csv", candidate_rows)

    split, split_payload = build_group_split(candidate_rows, int(args.seed))
    (args.out_dir / "grouped_split_manifest.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    metric = lambda name: np.asarray([float(row[name]) for row in candidate_rows], dtype=np.float32)
    gate = lambda name: np.asarray([int(row[name]) for row in candidate_rows], dtype=np.int8)
    dataset_payload = dict(dataset)
    dataset_payload.setdefault(
        "candidate_index", np.asarray(dataset["candidate_indices"], dtype=np.int64)
    )
    dataset_payload.setdefault(
        "sample_index", np.asarray(dataset["sample_indices"], dtype=np.int64)
    )
    dataset_payload.setdefault("mask", np.asarray(dataset["masks"], dtype=np.int8))
    dataset_payload.setdefault(
        "w_tasks_real_imag",
        np.asarray(dataset["task_weights_real_imag"], dtype=np.float32),
    )
    dataset_payload.setdefault(
        "w_combined_real_imag",
        np.asarray(dataset["combined_weights_real_imag"], dtype=np.float32),
    )
    np.savez_compressed(
        args.out_dir / "residual_critic_dataset.npz",
        **dataset_payload,
        split=split,
        delta_psll_db=metric("delta_psll_db"),
        delta_nearest_iso_db=metric("delta_nearest_iso_db"),
        delta_local_iso_db=metric("delta_local_iso_db"),
        delta_mainlobe_gain_db=metric("delta_mainlobe_gain_db"),
        eep_psll_db=metric("eep_psll_db"),
        eep_nearest_iso_db=metric("eep_nearest_iso_db"),
        eep_local_iso_db=metric("eep_local_iso_db"),
        eep_mainlobe_gain_db=metric("eep_mainlobe_gain_db"),
        hfss_psll_db=metric("hfss_psll_db"),
        hfss_nearest_iso_db=metric("hfss_nearest_iso_db"),
        hfss_local_iso_db=metric("hfss_local_iso_db"),
        hfss_mainlobe_gain_db=metric("hfss_mainlobe_gain_db"),
        worst_active_rl_db=metric("all_case_worst_active_rl_db"),
        total_rl_db=metric("all_case_worst_total_rl_db"),
        gate15=gate("gate15"),
        strict_gate20=gate("strict_gate20"),
        mainlobe_gate=gate("mainlobe_gate"),
        active_RL_gate=gate("active_RL_gate"),
        strict_engineering_gate=gate("strict_engineering_gate"),
        hard_negative=gate("hard_negative"),
        hard_positive=gate("hard_positive"),
        near_boundary=gate("near_boundary"),
        training_priority=metric("training_priority"),
    )
    complete_cases = [row for row in case_metrics if int(row.get("complete", 0)) == 1]
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(candidate_rows),
        "independent_scene_count": int(np.unique([row["sample_index"] for row in candidate_rows]).size),
        "expected_case_count": len(case_rows),
        "complete_case_count": len(complete_cases),
        "all_no_scale_reconstruction_pass": bool(
            len(complete_cases) == len(case_rows) and all(int(row["no_scale_pass"]) == 1 for row in complete_cases)
        ),
        "complex_nmse_max": max((float(row["complex_nmse"]) for row in complete_cases), default=float("nan")),
        "magnitude_rmse_db_max": max((float(row["magnitude_rmse_db_visible40"]) for row in complete_cases), default=float("nan")),
        "gate15_rate": float(np.mean([row["gate15"] for row in candidate_rows])),
        "strict_gate20_rate": float(np.mean([row["strict_gate20"] for row in candidate_rows])),
        "mainlobe_gate_rate": float(np.mean([row["mainlobe_gate"] for row in candidate_rows])),
        "active_RL_gate_rate": float(np.mean([row["active_RL_gate"] for row in candidate_rows])),
        "strict_engineering_gate_rate": float(np.mean([row["strict_engineering_gate"] for row in candidate_rows])),
        "hard_negative_count": int(sum(row["hard_negative"] for row in candidate_rows)),
        "hard_positive_count": int(sum(row["hard_positive"] for row in candidate_rows)),
        "near_boundary_count": int(sum(row["near_boundary"] for row in candidate_rows)),
        "scene_leakage_free": bool(split_payload["scene_leakage_free"]),
        "old_labels_included": False,
        "labels_allowed": bool(len(candidate_rows) == 96 and split_payload["scene_leakage_free"]),
    }
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def status(args: argparse.Namespace) -> dict[str, Any]:
    chunks_path = args.out_dir / "chunk_manifest.csv"
    if not chunks_path.exists():
        return {"prepared": False}
    chunks = read_csv(chunks_path)
    complete = sum(chunk_complete(row) for row in chunks)
    return {
        "prepared": True,
        "chunk_count": len(chunks),
        "complete_chunk_count": complete,
        "run_complete": complete == len(chunks),
        "analyzed": (args.out_dir / "analysis_summary.json").exists(),
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
