#!/usr/bin/env python3
"""Synthesize one finite-Q x/y even-odd correction block on trusted S4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from design_v115_grounded_modal_network import terminate_network
from generate_gate15_boundary_scenes import metric_at
from generate_v09_eep_development_candidates import full_active_metrics, physical_margins
from hfss_task_fullwave_validate import pattern_grid_dirs
from run_v16_robust_drift_oracle import ri_to_complex
from run_v20_three_frequency_mask_weight_joint import operator_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v141_xy_modal_correction_preregistered.json"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v141_xy_modal_correction_20260808_run01"
EPS = 1.0e-15
I4 = np.eye(4, dtype=float)
LX = np.asarray(
    [[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, -1], [0, 0, -1, 1]],
    dtype=float,
)
LY = np.asarray(
    [[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]],
    dtype=float,
)
MODES = np.asarray(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
    dtype=float,
).T / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--maxiter", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parents[1] / path


def blocks() -> np.ndarray:
    return np.asarray(
        [
            [ix * 16 + iy, (ix + 1) * 16 + iy, ix * 16 + iy + 1, (ix + 1) * 16 + iy + 1]
            for ix in range(0, 16, 2)
            for iy in range(0, 16, 2)
        ],
        dtype=np.int64,
    )


def lossy_reactive(value: float, basis: np.ndarray, q_l: float, q_c: float) -> np.ndarray:
    q_value = q_l if value >= 0.0 else q_c
    return (abs(value) / q_value + 1j * value) * basis


def correction_network(
    parameters: np.ndarray,
    s4: np.ndarray,
    q_l: float,
    q_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, xx, xy, b0, bx, by = (float(value) for value in parameters)
    series_z = (
        lossy_reactive(x0, I4, q_l, q_c)
        + lossy_reactive(xx, LX, q_l, q_c)
        + lossy_reactive(xy, LY, q_l, q_c)
    )
    shunt_y = (
        lossy_reactive(b0, I4, q_l, q_c)
        + lossy_reactive(bx, LX, q_l, q_c)
        + lossy_reactive(by, LY, q_l, q_c)
    )
    series_y = np.linalg.inv(series_z)
    network_y = np.block([[series_y, -series_y], [-series_y, series_y + shunt_y]])
    identity = np.eye(8, dtype=complex)
    network_s = (identity - 50.0 * network_y) @ np.linalg.inv(identity + 50.0 * network_y)
    external_s, antenna_map, _reflected_map = terminate_network(network_s, s4)
    return external_s, antenna_map, network_s


def tile(matrix: np.ndarray, block_indices: np.ndarray) -> np.ndarray:
    output = np.zeros((256, 256), dtype=np.complex128)
    for indices in block_indices:
        output[np.ix_(indices, indices)] = matrix
    return output


def map_tasks(matrix: np.ndarray, tasks: np.ndarray, block_indices: np.ndarray) -> np.ndarray:
    output = np.zeros_like(tasks, dtype=np.complex128)
    for indices in block_indices:
        output[indices] = matrix @ tasks[indices]
    return output


def load_commands(
    projection_dir: Path,
    dataset: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    with (projection_dir / "scene_oracle.csv").open(newline="", encoding="utf-8-sig") as handle:
        oracle = list(csv.DictReader(handle))
    tasks: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for scene_index, row in enumerate(oracle):
        with np.load(projection_dir / "scenes" / f"scene_{scene_index:02d}.npz") as source:
            candidates = ri_to_complex(source["tasks_real_imag"])
        k_value = int(row["k_value"])
        candidate_index = int(row["best_candidate_index_in_scene"])
        tasks.append(np.asarray(candidates[candidate_index, :, :k_value], dtype=np.complex128))
        masks.append(np.asarray(dataset["masks"][scene_index], dtype=bool))
        targets.append(np.asarray(dataset["targets"][scene_index, :k_value], dtype=float))
        metadata.append(
            {
                "scene_index": scene_index,
                "sample_index": int(row["sample_index"]),
                "k_value": k_value,
                "ratio": float(row["ratio"]),
                "projection_candidate_index": candidate_index,
            }
        )
    return tasks, masks, targets, metadata


def build_fast_cases(
    tasks: list[np.ndarray],
    masks: list[np.ndarray],
    block_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    sources: list[np.ndarray] = []
    considered: list[np.ndarray] = []
    for scene_tasks, mask in zip(tasks, masks):
        sources.append(np.sum(scene_tasks, axis=1))
        considered.append(mask)
        for task_index in range(scene_tasks.shape[1]):
            value = scene_tasks[:, task_index]
            amplitude = np.abs(value)
            sources.append(value)
            considered.append(mask & (amplitude >= max(float(np.max(amplitude)), EPS) * 0.1))
    source = np.stack(sources, axis=1)[block_indices]
    return {
        "source": source,
        "considered": np.stack(considered, axis=1)[block_indices],
        "incident_power": np.sum(np.abs(source) ** 2, axis=(0, 1)),
    }


def fast_metrics(
    parameters: np.ndarray,
    s4: np.ndarray,
    cases: dict[str, np.ndarray],
    q_l: float,
    q_c: float,
) -> dict[str, float]:
    external_s, antenna_map, _network = correction_network(parameters, s4, q_l, q_c)
    source = cases["source"]
    reflected = np.einsum("ij,bjn->bin", external_s, source)
    antenna_incident = np.einsum("ij,bjn->bin", antenna_map, source)
    antenna_reflected = np.einsum("ij,bjn->bin", s4, antenna_incident)
    gamma = np.where(
        cases["considered"],
        np.abs(reflected) / np.maximum(np.abs(source), EPS),
        0.0,
    )
    total_gamma = np.sqrt(
        np.sum(np.abs(reflected) ** 2, axis=(0, 1)) / cases["incident_power"]
    )
    external_accepted = cases["incident_power"] - np.sum(
        np.abs(reflected) ** 2, axis=(0, 1)
    )
    antenna_accepted = np.sum(np.abs(antenna_incident) ** 2, axis=(0, 1)) - np.sum(
        np.abs(antenna_reflected) ** 2, axis=(0, 1)
    )
    efficiency = antenna_accepted / np.maximum(external_accepted, EPS)
    alpha = np.sum(np.conjugate(source) * antenna_incident, axis=(0, 1)) / cases[
        "incident_power"
    ]
    residual = antenna_incident - source * alpha[None, None, :]
    distortion = np.sqrt(
        np.sum(np.abs(residual) ** 2, axis=(0, 1))
        / np.maximum(np.sum(np.abs(antenna_incident) ** 2, axis=(0, 1)), EPS)
    )
    return {
        "worst_active_gamma": float(np.max(gamma)),
        "worst_total_gamma": float(np.max(total_gamma)),
        "network_efficiency_min": float(np.min(efficiency)),
        "command_map_distortion_max": float(np.max(distortion)),
        "passive_gamma_max": float(np.max(np.abs(np.diag(external_s)))),
    }


def optimize_variant(
    name: str,
    config: dict[str, Any],
    s4: np.ndarray,
    cases: dict[str, np.ndarray],
    maxiter: int,
) -> tuple[np.ndarray, dict[str, float]]:
    topology = config["topology"]
    settings = config["optimization"]
    q_l = float(topology["inductor_q"])
    q_c = float(topology["capacitor_q"])
    rho_active = 10.0 ** (-float(settings["active_design_target_db"]) / 20.0)
    rho_passive = 10.0 ** (-float(settings["passive_target_db"]) / 20.0)
    distortion_limit = float(settings["pattern_map_distortion_limit"])

    def objective(parameters: np.ndarray) -> float:
        metrics = fast_metrics(parameters, s4, cases, q_l, q_c)
        active = max(metrics["worst_active_gamma"], metrics["worst_total_gamma"])
        common = (
            active
            + 100.0 * max(active - rho_active, 0.0) ** 2
            + 100.0 * max(0.95 - metrics["network_efficiency_min"], 0.0) ** 2
            + 50.0 * max(metrics["passive_gamma_max"] - rho_passive, 0.0) ** 2
        )
        if name == "pattern_guarded":
            common += 300.0 * max(
                metrics["command_map_distortion_max"] - distortion_limit, 0.0
            ) ** 2
        return common

    result = differential_evolution(
        objective,
        bounds=[tuple(value) for value in topology["bounds"]],
        seed=int(settings["seed"]) + (0 if name == "active_upper_bound" else 1),
        popsize=int(settings["population_size_multiplier"]),
        maxiter=maxiter,
        tol=1.0e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return np.asarray(result.x, dtype=float), fast_metrics(result.x, s4, cases, q_l, q_c)


def exact_variant(
    name: str,
    parameters: np.ndarray,
    s4: np.ndarray,
    tasks: list[np.ndarray],
    masks: list[np.ndarray],
    targets: list[np.ndarray],
    metadata: list[dict[str, Any]],
    dataset: dict[str, np.ndarray],
    corner: dict[str, Any],
    config: dict[str, Any],
    block_indices: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    topology = config["topology"]
    external_s4, antenna_map4, network_s8 = correction_network(
        parameters,
        s4,
        float(topology["inductor_q"]),
        float(topology["capacitor_q"]),
    )
    external_s256 = tile(external_s4, block_indices)
    rows: list[dict[str, Any]] = []
    gate = config["gates"]
    for scene_index, (external_tasks, mask, scene_targets, meta) in enumerate(
        zip(tasks, masks, targets, metadata)
    ):
        antenna_tasks = map_tasks(antenna_map4, external_tasks, block_indices)
        pattern = metric_at(corner["fast"].evaluate(antenna_tasks, scene_targets), 0)
        active = full_active_metrics(external_tasks, mask, external_s256)
        reference = {
            str(metric): float(dataset["reference_metrics"][scene_index, metric_index])
            for metric_index, metric in enumerate(dataset["reference_metric_names"])
        }
        pattern_margins = physical_margins(
            pattern,
            reference,
            {**active, "active_rl_floor_db": 99.0},
        )[:4]
        efficiency_values = []
        for source in [np.sum(external_tasks, axis=1), *external_tasks.T]:
            external_reflected = external_s256 @ source
            antenna_incident = map_tasks(antenna_map4, source[:, None], block_indices)[:, 0]
            antenna_reflected = tile(s4, block_indices) @ antenna_incident
            external_accepted = float(np.vdot(source, source).real - np.vdot(external_reflected, external_reflected).real)
            antenna_accepted = float(
                np.vdot(antenna_incident, antenna_incident).real
                - np.vdot(antenna_reflected, antenna_reflected).real
            )
            efficiency_values.append(antenna_accepted / max(external_accepted, EPS))
        combined = np.sum(external_tasks, axis=1)
        amplitude = np.abs(combined[mask])
        dynamic_db = 20.0 * np.log10(
            max(float(np.max(amplitude)), EPS) / max(float(np.min(amplitude)), EPS)
        )
        peak_rms_db = 20.0 * np.log10(
            max(float(np.max(amplitude)), EPS)
            / max(float(np.sqrt(np.mean(amplitude**2))), EPS)
        )
        joint_pass = int(
            float(active["active_rl_floor_db"]) >= float(gate["active_rl_min_db"])
            and float(active["combined_total_rl_db"]) >= float(gate["total_rl_min_db"])
            and float(np.min(pattern_margins)) >= 0.0
            and min(efficiency_values) >= float(gate["network_efficiency_min"])
        )
        rows.append(
            {
                "variant": name,
                **meta,
                **{key: float(value) for key, value in pattern.items()},
                **active,
                "pattern_margin_db": float(np.min(pattern_margins)),
                "network_efficiency_min": float(min(efficiency_values)),
                "external_combined_dynamic_range_db": float(dynamic_db),
                "external_combined_peak_to_rms_db": float(peak_rms_db),
                "joint_gate_pass": joint_pass,
            }
        )
    passive_rl = -20.0 * np.log10(np.maximum(np.abs(np.diag(external_s4)), EPS))
    modal_external = MODES.conj().T @ external_s4 @ MODES
    summary = {
        "variant": name,
        "parameters": parameters.tolist(),
        "scene_count": len(rows),
        "joint_pass_count": sum(int(row["joint_gate_pass"]) for row in rows),
        "active11_pass_count": sum(float(row["active_rl_floor_db"]) >= 11.0 for row in rows),
        "pattern_pass_count": sum(float(row["pattern_margin_db"]) >= 0.0 for row in rows),
        "worst_active_rl_db": min(float(row["active_rl_floor_db"]) for row in rows),
        "worst_total_rl_db": min(float(row["combined_total_rl_db"]) for row in rows),
        "worst_pattern_margin_db": min(float(row["pattern_margin_db"]) for row in rows),
        "network_efficiency_min": min(float(row["network_efficiency_min"]) for row in rows),
        "passive_rl_min_db": float(np.min(passive_rl)),
        "reciprocity_error": float(np.max(np.abs(external_s4 - external_s4.T))),
        "passivity_sigma": float(np.linalg.svd(external_s4, compute_uv=False)[0]),
        "modal_rl_db": (-20.0 * np.log10(np.maximum(np.abs(np.diag(modal_external)), EPS))).tolist(),
    }
    return rows, summary, external_s4, antenna_map4, network_s8


def component(value: float, kind: str) -> dict[str, Any]:
    omega = 2.0 * math.pi * 10.0e9
    if kind == "series":
        return (
            {"implementation": "inductive_series_path", "equivalent_nh": value / omega * 1.0e9}
            if value >= 0.0
            else {"implementation": "capacitive_series_gap", "equivalent_pf": -1.0 / (omega * value) * 1.0e12}
        )
    return (
        {"implementation": "capacitive_shunt_or_bridge", "equivalent_pf": value / omega * 1.0e12}
        if value >= 0.0
        else {"implementation": "inductive_shunt_or_bridge", "equivalent_nh": -1.0 / (omega * value) * 1.0e9}
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v1.41 output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    inputs = {
        key: resolve(config_path, value).resolve()
        for key, value in config["inputs"].items()
    }
    write_json(
        out / "preregistration.json",
        {
            **config,
            "config_sha256": sha256(config_path),
            "input_sha256": {
                key: sha256(path) for key, path in inputs.items() if path.is_file()
            },
        },
    )
    with np.load(inputs["physical_s4"]) as source:
        s4 = ri_to_complex(source["s_real_imag"])
        port_order = [str(value) for value in source["port_order"]]
    if port_order != config["topology"]["physical_port_order"]:
        raise RuntimeError(f"S4 port order changed: {port_order}")
    with np.load(inputs["frozen_scenes"]) as source:
        dataset = {key: source[key] for key in source.files}
    tasks, masks, targets, metadata = load_commands(inputs["projection_run"], dataset)
    block_indices = blocks()
    cases = build_fast_cases(tasks, masks, block_indices)
    _base, corner = operator_bundle(inputs["pattern_operator"], 10.0)
    maxiter = args.maxiter or int(config["optimization"]["maximum_iterations"])
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for variant in config["optimization"]["variants"]:
        parameters, fast = optimize_variant(variant, config, s4, cases, maxiter)
        rows, summary, external_s4, antenna_map4, network_s8 = exact_variant(
            variant,
            parameters,
            s4,
            tasks,
            masks,
            targets,
            metadata,
            dataset,
            corner,
            config,
            block_indices,
        )
        summary["fast_upper_bound"] = fast
        summary["components"] = {
            name: component(float(value), "series" if index < 3 else "shunt")
            for index, (name, value) in enumerate(
                zip(config["topology"]["parameters"], parameters)
            )
        }
        all_rows.extend(rows)
        summaries.append(summary)
        np.savez_compressed(
            out / f"{variant}_circuit_operator.npz",
            parameters=parameters,
            external_s4_real_imag=np.stack((external_s4.real, external_s4.imag), axis=-1),
            antenna_incident_map4_real_imag=np.stack((antenna_map4.real, antenna_map4.imag), axis=-1),
            network_s8_real_imag=np.stack((network_s8.real, network_s8.imag), axis=-1),
        )
    write_csv(out / "scene_exact_metrics.csv", all_rows)
    coverage_rows: list[dict[str, Any]] = []
    eligible_variants: list[str] = []
    required_k = [int(value) for value in config["coverage_gate"]["required_k_values"]]
    minimum_per_k = int(config["coverage_gate"]["minimum_joint_pass_per_k"])
    for summary in summaries:
        variant_rows = [row for row in all_rows if row["variant"] == summary["variant"]]
        counts = {
            k_value: sum(
                int(row["joint_gate_pass"])
                for row in variant_rows
                if int(row["k_value"]) == k_value
            )
            for k_value in required_k
        }
        coverage_pass = all(counts[k_value] >= minimum_per_k for k_value in required_k)
        summary["joint_pass_count_by_k"] = {str(key): value for key, value in counts.items()}
        summary["multi_k_coverage_gate_pass"] = coverage_pass
        if coverage_pass:
            eligible_variants.append(str(summary["variant"]))
        for k_value, count in counts.items():
            coverage_rows.append(
                {
                    "variant": summary["variant"],
                    "k_value": k_value,
                    "joint_pass_count": count,
                    "minimum_required": minimum_per_k,
                    "coverage_gate_pass": int(count >= minimum_per_k),
                }
            )
    write_csv(out / "coverage_by_k.csv", coverage_rows)
    write_json(out / "variant_summary.json", summaries)
    active_upper = next(value for value in summaries if value["variant"] == "active_upper_bound")
    decision = {
        "circuit_joint_gate_pass": bool(eligible_variants),
        "eligible_variants": eligible_variants,
        "active_only_upper_bound_reaches_11db": bool(active_upper["worst_active_rl_db"] >= 11.0),
        "authorize_physical_2x2_hfss": bool(eligible_variants),
        "authorize_larger_array": False,
        "authorize_labels_or_critic": False,
        "conclusion": (
            "A single x/y modal block preserves at least one strict task scene; freeze it for physical 2x2 HFSS."
            if eligible_variants
            else "The single block has an active-RL-only upper bound but cannot preserve the frozen task pattern. Keep physical HFSS locked; next test must jointly re-optimize task weights on the corrected S4 and mapped EEP before any CAD build."
        ),
    }
    write_json(out / "stage_decision.json", decision)
    print(json.dumps({"variants": summaries, "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
