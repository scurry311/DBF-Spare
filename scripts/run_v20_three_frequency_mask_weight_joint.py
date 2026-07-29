#!/usr/bin/env python3
"""Run three-frequency structured-mask and common task-weight alternating search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from generate_gate15_boundary_scenes import metric_at
from generate_v09_eep_development_candidates import (
    full_active_metrics,
    matched_steering_tasks,
    structured_masks,
)
from hfss_task_fullwave_validate import pattern_grid_dirs
from run_v16_k6_multifrequency_rescue import Variant, optimize_common_command
from run_v16_robust_drift_oracle import (
    apply_calibration,
    complex_to_ri,
    load_nominal_operator,
    load_npz,
    mask_digest,
    normalize_task_norms,
    read_csv,
    ri_to_complex,
    scene_calibration_states,
    write_csv,
)
from run_v19_nominal_9p96_joint_projection import (
    ROBUST_MARGIN_NAMES,
    evaluate_command,
    identity_state,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "v20_three_frequency_mask_weight_joint.json"
DEFAULT_PROJECTED = (
    ROOT
    / "hfss_outputs"
    / "v19_nominal_9p96_joint_projection_20260729_run01"
    / "projected_commands.npz"
)
DEFAULT_SOURCE = (
    ROOT
    / "hfss_outputs"
    / "v18_perturbed_operator_frequency_low_evaluation_20260729_run01"
    / "dataset_arrays.npz"
)
DEFAULT_POOL = (
    ROOT
    / "hfss_outputs"
    / "v16_robust_drift_oracle_20260727_run01"
    / "pool"
    / "candidate_pool.npz"
)
DEFAULT_PARENT = ROOT / "hfss_outputs" / "v16_robust_drift_oracle_20260727_run01"
DEFAULT_AUDIT = ROOT / "hfss_outputs" / "v20_three_frequency_active_rl_audit_20260729_run01"
DEFAULT_STAGE = (
    ROOT
    / "hfss_outputs"
    / "v19_joint_projection_symmetric_high_stage_summary_20260729_run01"
    / "three_frequency_candidate_gates.csv"
)
DEFAULT_NOMINAL = (
    ROOT
    / "hfss_outputs"
    / "fixed_mesh_eep256_20260723_run05"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_LOW = (
    ROOT
    / "hfss_outputs"
    / "v18_perturbed_operator_frequency_low_20260729_run01"
    / "operator"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_HIGH = (
    ROOT
    / "hfss_outputs"
    / "v19_perturbed_operator_frequency_high_20260729_run01"
    / "operator"
    / "grounded_patch_eep_operator_256port.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "v20_three_frequency_joint_search_20260729_run01"
EPS = 1.0e-12
KMAX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--projected", type=Path, default=DEFAULT_PROJECTED)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--stage-gates", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--nominal-operator", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--low-operator", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--high-operator", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-ratios", type=int, default=0)
    parser.add_argument("--initial-masks", type=int, default=0)
    parser.add_argument("--alternating-masks", type=int, default=0)
    parser.add_argument("--full-refine-top", type=int, default=0)
    parser.add_argument("--include-currently-passing", action="store_true")
    return parser.parse_args()


def command_hash(command: np.ndarray) -> str:
    packed = np.ascontiguousarray(complex_to_ri(command))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def operator_bundle(path: Path, expected_frequency: float) -> tuple[dict[str, Any], dict[str, Any]]:
    base, effective, fast, s_matrix = load_nominal_operator(path)
    frequency = float(base["frequency_ghz"])
    if not np.isclose(frequency, expected_frequency, atol=1.0e-6):
        raise RuntimeError(f"Expected {expected_frequency} GHz operator, got {frequency}")
    return base, {"effective": effective, "fast": fast, "s": s_matrix}


def load_global_port_risk(audit_dir: Path) -> np.ndarray:
    rows = load_csv(audit_dir / "repeated_high_risk_ports.csv")
    risk = np.zeros(256, dtype=np.float64)
    for row in rows:
        port = int(row["port_index"])
        risk[port] = float(row["active_rl_fail11_count"]) + 10.0 * max(
            float(row["worst_gamma"]) - 10.0 ** (-11.0 / 20.0), 0.0
        )
    return normalize01(risk)


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - float(np.min(values))) / max(float(np.ptp(values)), EPS)


def load_parent_records(
    parent_dir: Path,
    pool: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    final_arrays = load_npz(parent_dir / "final" / "robust_arrays.npz")
    final_tasks = ri_to_complex(final_arrays["tasks_real_imag"])
    for row in read_csv(parent_dir / "final" / "robust_candidate_metrics.csv"):
        evaluation = int(row["evaluation_index"])
        candidate = int(row["candidate_index"])
        records.append(
            {
                "sample_index": int(row["sample_index"]),
                "k": int(row["k_value"]),
                "ratio": float(row["ratio"]),
                "mask": np.asarray(pool["masks"][candidate], dtype=bool),
                "command": np.asarray(final_tasks[evaluation, :, : int(row["k_value"])], dtype=np.complex64),
                "target": np.asarray(pool["targets"][candidate, : int(row["k_value"])], dtype=float),
                "margin": float(row["E2_worst_margin_db"]),
                "strict": int(row["E2_strict_pass"]),
                "origin": str(row["candidate_origin"]),
                "source_index": candidate,
            }
        )
    rescue_path = parent_dir / "post_rescue" / "rescue_robust_candidate_metrics.csv"
    if rescue_path.exists():
        rescue_arrays = load_npz(parent_dir / "rescue" / "rescue_candidates.npz")
        rescue_tasks = ri_to_complex(rescue_arrays["tasks_real_imag"])
        for row in read_csv(rescue_path):
            rescue = int(row["rescue_index"])
            k_value = int(row["k_value"])
            records.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "k": k_value,
                    "ratio": float(row["ratio"]),
                    "mask": np.asarray(rescue_arrays["masks"][rescue], dtype=bool),
                    "command": np.asarray(rescue_tasks[rescue, :, :k_value], dtype=np.complex64),
                    "target": np.asarray(rescue_arrays["targets"][rescue, :k_value], dtype=float),
                    "margin": float(row["E2_worst_margin_db"]),
                    "strict": int(row["E2_strict_pass"]),
                    "origin": str(row["candidate_origin"]),
                    "source_index": rescue,
                }
            )
    return records


def select_parents(
    records: list[dict[str, Any]],
    sample: int,
    ratio: float,
    count: int,
) -> list[dict[str, Any]]:
    matching = [
        row
        for row in records
        if int(row["sample_index"]) == sample
        and np.isclose(float(row["ratio"]), ratio, atol=1.0e-5)
    ]
    matching.sort(key=lambda row: float(row["margin"]), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in matching:
        digest = mask_digest(np.asarray(row["mask"], dtype=bool))
        if digest in seen:
            continue
        seen.add(digest)
        selected.append(row)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Not enough parents for sample={sample} ratio={ratio}")
    return selected


def historical_donors(
    records: list[dict[str, Any]],
    k_value: int,
    ratio: float,
    limit: int = 4,
) -> list[np.ndarray]:
    matching = [
        row
        for row in records
        if int(row["k"]) == k_value
        and np.isclose(float(row["ratio"]), ratio, atol=1.0e-5)
    ]
    matching.sort(key=lambda row: (int(row["strict"]), float(row["margin"])), reverse=True)
    output: list[np.ndarray] = []
    seen: set[str] = set()
    for row in matching:
        mask = np.asarray(row["mask"], dtype=bool)
        digest = mask_digest(mask)
        if digest in seen:
            continue
        seen.add(digest)
        output.append(mask)
        if len(output) == limit:
            break
    return output


def scene_states(
    pool: dict[str, np.ndarray],
    pool_scene: np.ndarray,
    element_ixiy: np.ndarray,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    identity = identity_state()
    low_state = scene_calibration_states(
        pool,
        pool_scene,
        {"frequency_low_x0.20": {"profile": "frequency_low", "level": 0.20}},
        element_ixiy,
        seed,
    )["frequency_low_x0.20"]
    high_state = scene_calibration_states(
        pool,
        pool_scene,
        {"frequency_high_x0.20": {"profile": "frequency_high", "level": 0.20}},
        element_ixiy,
        seed,
    )["frequency_high_x0.20"]
    states = {
        "nominal_identity": identity,
        "frequency_low_identity": identity,
        "frequency_low_E2_source": low_state,
        "frequency_high_identity": identity,
        "frequency_high_E2_source": high_state,
    }
    return states, {"low": low_state, "high": high_state}


def scene_port_score(
    command: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    grid_dirs: np.ndarray,
    global_risk: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    unique_physical = [
        corners["nominal_identity"],
        corners["frequency_low_identity"],
        corners["frequency_high_identity"],
    ]
    utility = np.zeros(256, dtype=np.float64)
    burden = np.zeros(256, dtype=np.float64)
    for corner in unique_physical:
        steering = matched_steering_tasks(command, targets, corner["effective"], grid_dirs)
        utility += np.sum(np.abs(steering) ** 2, axis=1)
        s_matrix = np.asarray(corner["s"], dtype=np.complex128)
        burden += np.abs(np.diag(s_matrix))
        burden += np.sqrt(np.sum(np.abs(s_matrix) ** 2, axis=0))
    utility = normalize01(utility)
    burden = normalize01(burden)
    stress = np.zeros(256, dtype=np.float64)
    rho = 10.0 ** (-11.0 / 20.0)
    for name, corner in corners.items():
        actual = apply_calibration(command, mask, states[name])
        sources = [np.sum(actual, axis=1), *[actual[:, index] for index in range(actual.shape[1])]]
        for source_index, source in enumerate(sources):
            amplitude = np.abs(source)
            maximum = max(float(np.max(amplitude)), EPS)
            threshold = maximum * (1.0e-8 if source_index == 0 else 0.1)
            considered = mask & (amplitude >= threshold)
            reflected = np.asarray(corner["s"]) @ source
            gamma = np.zeros(256, dtype=np.float64)
            gamma[considered] = np.abs(reflected[considered]) / np.maximum(
                amplitude[considered], threshold
            )
            stress = np.maximum(stress, gamma / rho)
    stress = normalize01(np.minimum(stress, 4.0))
    score = 0.62 * utility - 0.18 * burden - 0.15 * stress - 0.05 * global_risk
    return score, {
        "utility": utility,
        "burden": burden,
        "stress": stress,
        "global_risk": global_risk,
    }


def guided_masks(
    parent_mask: np.ndarray,
    score: np.ndarray,
    count: int,
    seed: int,
    existing: set[str] | None = None,
) -> list[tuple[np.ndarray, str]]:
    existing = set(existing or set())
    active = np.flatnonzero(parent_mask)
    inactive = np.flatnonzero(~parent_mask)
    remove_order = active[np.argsort(score[active], kind="stable")]
    add_order = inactive[np.argsort(score[inactive], kind="stable")[::-1]]
    rng = np.random.default_rng(seed)
    schedule = (1, 2, 3, 4, 6, 8, 10, 12)
    output: list[tuple[np.ndarray, str]] = []
    attempts = 0
    while len(output) < count and attempts < count * 40:
        swaps = min(schedule[attempts % len(schedule)], len(active), len(inactive))
        offset = (attempts // len(schedule)) % max(1, min(8, len(active) - swaps + 1))
        remove = remove_order[offset : offset + swaps]
        add = add_order[offset : offset + swaps]
        proposal = parent_mask.copy()
        if len(remove) == swaps and len(add) == swaps:
            proposal[remove] = False
            proposal[add] = True
        else:
            remove = rng.choice(active, size=swaps, replace=False)
            add = rng.choice(inactive, size=swaps, replace=False)
            proposal[remove] = False
            proposal[add] = True
        digest = mask_digest(proposal)
        attempts += 1
        if digest in existing:
            continue
        existing.add(digest)
        output.append((proposal, f"three_frequency_guided_swap_{len(output):02d}"))
    if len(output) != count:
        raise RuntimeError(f"Generated only {len(output)}/{count} guided masks")
    return output


def adapt_command(
    parent: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    nominal_effective: Any,
    grid_dirs: np.ndarray,
) -> np.ndarray:
    steering = matched_steering_tasks(parent, targets, nominal_effective, grid_dirs)
    output = 0.72 * np.asarray(parent, dtype=np.complex64) + 0.28 * steering
    output[~mask] = 0.0
    output = normalize_task_norms(output, parent)
    output[~mask] = 0.0
    return output.astype(np.complex64)


def quick_active_floor(
    command: np.ndarray,
    mask: np.ndarray,
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    floors: list[float] = []
    high_floor = float("inf")
    for name, corner in corners.items():
        actual = apply_calibration(command, mask, states[name])
        floor = float(full_active_metrics(actual, mask, corner["s"])["active_rl_floor_db"])
        floors.append(floor)
        if name == "frequency_high_E2_source":
            high_floor = floor
    return min(floors), high_floor


def choose_parent(mask: np.ndarray, parents: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        parents,
        key=lambda row: int(np.sum(mask & np.asarray(row["mask"], dtype=bool))),
    )


def full_selection_key(row: dict[str, Any]) -> tuple[int, int, float, int, int, float, float]:
    return (
        int(row["design_reserve_pass"]),
        int(row["all_corner_strict_pass"]),
        min(float(row["robust_worst_margin_db"]), 3.0),
        int(row["all_corner_pattern_pass"]),
        int(row["all_corner_active_rl_pass"]),
        min(float(row["robust_active_rl_margin_db"]), 3.0),
        float(row["robust_psll_margin_db"]),
    )


def append_full_candidate(
    *,
    rows: list[dict[str, Any]],
    arrays: dict[str, list[Any]],
    command: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    reference: dict[str, float],
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    grid_dirs: np.ndarray,
    gates: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    values, margins = evaluate_command(
        command, mask, targets, reference, corners, states, grid_dirs, gates
    )
    values.update(metadata)
    values.update(
        {
            "evaluation_index": len(rows),
            "mask_hash": mask_digest(mask),
            "command_hash": command_hash(command),
            "design_reserve_pass": int(
                values["all_corner_pattern_pass"]
                and float(values["robust_active_rl_margin_db"]) >= 1.0
                and float(values["robust_hardware_margin_db"]) >= 0.0
            ),
            "robust_psll_margin_db": float(np.min(margins[:, 0])),
        }
    )
    rows.append(values)
    padded = np.zeros((256, KMAX), dtype=np.complex64)
    padded[:, : command.shape[1]] = command
    arrays["sample_index"].append(int(metadata["sample_index"]))
    arrays["k_values"].append(int(metadata["k"]))
    arrays["ratio"].append(float(metadata["ratio"]))
    arrays["mask"].append(mask.astype(np.int8))
    arrays["targets"].append(
        np.pad(targets, ((0, KMAX - len(targets)), (0, 0)), constant_values=np.nan)
    )
    arrays["tasks"].append(padded)
    arrays["margins"].append(margins)
    return values


def append_weight_path(
    *,
    rows: list[dict[str, Any]],
    arrays: dict[str, list[Any]],
    warm: np.ndarray,
    optimized: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    reference: dict[str, float],
    corners: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    grid_dirs: np.ndarray,
    gates: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    evaluated: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        command = ((1.0 - alpha) * warm + alpha * optimized).astype(np.complex64)
        command[~mask] = 0.0
        values = append_full_candidate(
            rows=rows,
            arrays=arrays,
            command=command,
            mask=mask,
            targets=targets,
            reference=reference,
            corners=corners,
            states=states,
            grid_dirs=grid_dirs,
            gates=gates,
            metadata={**metadata, "weight_path_alpha": alpha},
        )
        evaluated.append((values, command, mask))
    return max(evaluated, key=lambda item: full_selection_key(item[0]))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v20 search: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    search = protocol["search"]
    acceptance = protocol["acceptance"]
    gates = protocol["gates"]
    initial_count = int(args.initial_masks or search["initial_masks_per_scene_ratio"])
    alternating_count = int(args.alternating_masks or search["alternating_masks_per_scene_ratio"])
    refine_top = int(args.full_refine_top or search["full_refine_top_per_round"])
    if initial_count < refine_top or alternating_count < refine_top:
        raise ValueError("Mask count must be no smaller than full-refine-top")
    variant = Variant(
        name="v20_three_frequency_rl11",
        target_amplitude_mode="median",
        active_rl_design_min_db=float(search["active_rl_design_min_db"]),
        local_radius_deg=float(search["regional_radius_deg"]),
        regional_ceiling_db=float(search["regional_ceiling_db"]),
        joint_projection_passes=int(search["joint_projection_passes"]),
        corner_sweeps=int(search["corner_sweeps"]),
    )

    projected = load_npz(args.projected)
    source = load_npz(args.source)
    pool = load_npz(args.pool)
    current_gates = {int(row["sample_index"]): int(row["three_frequency_pass"]) for row in load_csv(args.stage_gates)}
    nominal_base, nominal = operator_bundle(args.nominal_operator, 10.0)
    low_base, low = operator_bundle(args.low_operator, 9.96)
    high_base, high = operator_bundle(args.high_operator, 10.04)
    for other in (low_base, high_base):
        if not np.array_equal(nominal_base["element_ixiy"], other["element_ixiy"]):
            raise RuntimeError("Three-frequency operator port ordering differs")
        if not (
            np.array_equal(nominal_base["theta_deg"], other["theta_deg"])
            and np.array_equal(nominal_base["phi_deg"], other["phi_deg"])
        ):
            raise RuntimeError("Three-frequency EEP grids differ")
    corners = {
        "nominal_identity": nominal,
        "frequency_low_identity": low,
        "frequency_low_E2_source": low,
        "frequency_high_identity": high,
        "frequency_high_E2_source": high,
    }
    grid_dirs = pattern_grid_dirs(nominal_base["theta_deg"], nominal_base["phi_deg"])
    element_ixiy = np.asarray(nominal_base["element_ixiy"], dtype=np.int64)
    pool_samples = np.asarray(pool["sample_index"], dtype=np.int64)
    samples = np.asarray(projected["sample_index"], dtype=np.int64)
    source_indices = np.asarray(projected["source_candidate_index"], dtype=np.int64)
    projected_commands = ri_to_complex(projected["selected_task_weights_real_imag"])
    records = load_parent_records(args.parent_dir, pool)
    global_risk = load_global_port_risk(args.audit_dir)
    projected_lookup = {int(sample): index for index, sample in enumerate(samples)}
    target_samples = [
        int(sample)
        for sample in samples
        if args.include_currently_passing or current_gates[int(sample)] == 0
    ]
    if args.max_scenes > 0:
        target_samples = target_samples[: args.max_scenes]
    ratios = [float(value) for value in search["ratios"]]
    if args.max_ratios > 0:
        ratios = ratios[: args.max_ratios]

    generated_rows: list[dict[str, Any]] = []
    generated_masks: list[np.ndarray] = []
    full_rows: list[dict[str, Any]] = []
    full_arrays: dict[str, list[Any]] = defaultdict(list)
    optimizer_runs = 0
    started = time.time()

    # Re-evaluate all frozen v19 commands under the exact five-state v20 definition.
    scene_context: dict[int, dict[str, Any]] = {}
    for sample in samples:
        projected_index = projected_lookup[int(sample)]
        source_index = int(source_indices[projected_index])
        k_value = int(projected["k_values"][projected_index])
        targets = np.asarray(projected["targets_deg"][projected_index, :k_value], dtype=float)
        pool_scene = np.flatnonzero(pool_samples == int(sample))
        states, _source_states = scene_states(
            pool, pool_scene, element_ixiy, int(protocol["seed"])
        )
        original = ri_to_complex(
            source["nominal_external_task_weights_real_imag"][source_index, :, :k_value]
        )
        reference = metric_at(nominal["fast"].evaluate(original, targets), 0)
        scene_context[int(sample)] = {
            "k": k_value,
            "targets": targets,
            "states": states,
            "reference": reference,
        }
        command = np.asarray(projected_commands[projected_index, :, :k_value], dtype=np.complex64)
        mask = np.asarray(projected["masks"][projected_index], dtype=bool)
        append_full_candidate(
            rows=full_rows,
            arrays=full_arrays,
            command=command,
            mask=mask,
            targets=targets,
            reference=reference,
            corners=corners,
            states=states,
            grid_dirs=grid_dirs,
            gates=gates,
            metadata={
                "sample_index": int(sample),
                "k": k_value,
                "ratio": float(projected["active_ratios_requested"][projected_index]),
                "round": "v19_frozen",
                "mask_family": "v19_frozen",
                "parent_origin": "v19_joint_projection",
                "parent_source_index": projected_index,
                "quick_active_rl_floor_db": float("nan"),
                "quick_high_active_rl_floor_db": float("nan"),
            },
        )

    for scene_position, sample in enumerate(target_samples, start=1):
        context = scene_context[sample]
        k_value = int(context["k"])
        targets = np.asarray(context["targets"], dtype=float)
        states = context["states"]
        reference = context["reference"]
        reserve_found = False
        for ratio_position, ratio in enumerate(ratios):
            parents = select_parents(
                records,
                sample,
                ratio,
                int(search["parent_commands_per_ratio"]),
            )
            parent = parents[0]
            score, _score_parts = scene_port_score(
                parent["command"],
                parent["mask"],
                targets,
                corners,
                states,
                grid_dirs,
                global_risk,
            )
            guided_count = min(8, initial_count)
            initial_guided = guided_masks(
                np.asarray(parent["mask"], dtype=bool),
                score,
                guided_count,
                int(protocol["seed"] + sample * 1009 + round(ratio * 1000)),
            )
            continuation = []
            higher = [value for value in search["ratios"] if float(value) > ratio]
            if higher:
                continuation = [
                    np.asarray(row["mask"], dtype=bool)
                    for row in select_parents(records, sample, float(min(higher)), 1)
                ]
            donor = historical_donors(records, k_value, ratio, limit=4)
            masks, families = structured_masks(
                np.asarray(parent["mask"], dtype=bool),
                score,
                element_ixiy,
                int(round(256 * ratio)),
                initial_count,
                np.random.default_rng(int(protocol["seed"] + sample * 7919 + ratio_position)),
                donor_masks=donor,
                continuation_masks=continuation,
                priority_masks=[np.asarray(row["mask"], dtype=bool) for row in parents],
                guided_masks=initial_guided,
            )
            quick: list[tuple[tuple[int, int, float, float], int, np.ndarray, np.ndarray, dict[str, Any]]] = []
            seen_ratio: set[str] = set()
            for local_index, (mask, family) in enumerate(zip(masks, families)):
                selected_parent = choose_parent(mask, parents)
                warm = adapt_command(
                    selected_parent["command"], mask, targets, nominal["effective"], grid_dirs
                )
                floor, high_floor = quick_active_floor(warm, mask, corners, states)
                score_mean = float(np.mean(score[mask]))
                digest = mask_digest(mask)
                seen_ratio.add(digest)
                generated_index = len(generated_rows)
                generated_rows.append(
                    {
                        "generated_index": generated_index,
                        "sample_index": sample,
                        "k": k_value,
                        "ratio": ratio,
                        "round": "initial",
                        "local_index": local_index,
                        "mask_family": family,
                        "mask_hash": digest,
                        "parent_origin": str(selected_parent["origin"]),
                        "parent_source_index": int(selected_parent["source_index"]),
                        "quick_active_rl_floor_db": floor,
                        "quick_high_active_rl_floor_db": high_floor,
                        "active_port_score_mean": score_mean,
                        "selected_for_full_refine": 0,
                    }
                )
                generated_masks.append(mask.astype(np.int8))
                key = (int(floor >= 11.0), int(floor >= 10.0), floor, score_mean)
                quick.append((key, generated_index, mask, warm, selected_parent))
            quick.sort(key=lambda item: item[0], reverse=True)
            refined_round: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
            for _key, generated_index, mask, warm, selected_parent in quick[:refine_top]:
                generated_rows[generated_index]["selected_for_full_refine"] = 1
                optimized, diagnostics = optimize_common_command(
                    warm,
                    mask,
                    targets,
                    corners,
                    states,
                    grid_dirs,
                    variant,
                    int(search["active_rl_nullspace_steps"]),
                    float(search["active_rl_nullspace_step_size"]),
                    bool(search["quantization_aware_selection"]),
                )
                optimizer_runs += 1
                values, selected_command, selected_mask = append_weight_path(
                    rows=full_rows,
                    arrays=full_arrays,
                    warm=warm,
                    optimized=optimized,
                    mask=mask,
                    targets=targets,
                    reference=reference,
                    corners=corners,
                    states=states,
                    grid_dirs=grid_dirs,
                    gates=gates,
                    metadata={
                        "sample_index": sample,
                        "k": k_value,
                        "ratio": ratio,
                        "round": "initial_refined",
                        "mask_family": generated_rows[generated_index]["mask_family"],
                        "generated_index": generated_index,
                        "parent_origin": str(selected_parent["origin"]),
                        "parent_source_index": int(selected_parent["source_index"]),
                        "quick_active_rl_floor_db": float(generated_rows[generated_index]["quick_active_rl_floor_db"]),
                        "quick_high_active_rl_floor_db": float(generated_rows[generated_index]["quick_high_active_rl_floor_db"]),
                        **{f"optimizer_{key}": float(value) for key, value in diagnostics.items()},
                    },
                )
                refined_round.append((values, selected_command, selected_mask))

            best_values, best_command, best_mask = max(
                refined_round, key=lambda item: full_selection_key(item[0])
            )
            alternate_score, _parts = scene_port_score(
                best_command,
                best_mask,
                targets,
                corners,
                states,
                grid_dirs,
                global_risk,
            )
            alternates = guided_masks(
                best_mask,
                alternate_score,
                alternating_count,
                int(protocol["seed"] + sample * 65537 + round(ratio * 1000)),
                seen_ratio,
            )
            alternate_quick: list[tuple[tuple[int, int, float, float], int, np.ndarray, np.ndarray]] = []
            for local_index, (mask, family) in enumerate(alternates):
                warm = adapt_command(best_command, mask, targets, nominal["effective"], grid_dirs)
                floor, high_floor = quick_active_floor(warm, mask, corners, states)
                score_mean = float(np.mean(alternate_score[mask]))
                generated_index = len(generated_rows)
                generated_rows.append(
                    {
                        "generated_index": generated_index,
                        "sample_index": sample,
                        "k": k_value,
                        "ratio": ratio,
                        "round": "alternating",
                        "local_index": local_index,
                        "mask_family": family,
                        "mask_hash": mask_digest(mask),
                        "parent_origin": "best_initial_refined",
                        "parent_source_index": int(best_values["evaluation_index"]),
                        "quick_active_rl_floor_db": floor,
                        "quick_high_active_rl_floor_db": high_floor,
                        "active_port_score_mean": score_mean,
                        "selected_for_full_refine": 0,
                    }
                )
                generated_masks.append(mask.astype(np.int8))
                key = (int(floor >= 11.0), int(floor >= 10.0), floor, score_mean)
                alternate_quick.append((key, generated_index, mask, warm))
            alternate_quick.sort(key=lambda item: item[0], reverse=True)
            ratio_rows = [best_values]
            for _key, generated_index, mask, warm in alternate_quick[:refine_top]:
                generated_rows[generated_index]["selected_for_full_refine"] = 1
                optimized, diagnostics = optimize_common_command(
                    warm,
                    mask,
                    targets,
                    corners,
                    states,
                    grid_dirs,
                    variant,
                    int(search["active_rl_nullspace_steps"]),
                    float(search["active_rl_nullspace_step_size"]),
                    bool(search["quantization_aware_selection"]),
                )
                optimizer_runs += 1
                values, _selected_command, _selected_mask = append_weight_path(
                    rows=full_rows,
                    arrays=full_arrays,
                    warm=warm,
                    optimized=optimized,
                    mask=mask,
                    targets=targets,
                    reference=reference,
                    corners=corners,
                    states=states,
                    grid_dirs=grid_dirs,
                    gates=gates,
                    metadata={
                        "sample_index": sample,
                        "k": k_value,
                        "ratio": ratio,
                        "round": "alternating_refined",
                        "mask_family": generated_rows[generated_index]["mask_family"],
                        "generated_index": generated_index,
                        "parent_origin": "best_initial_refined",
                        "parent_source_index": int(best_values["evaluation_index"]),
                        "quick_active_rl_floor_db": float(generated_rows[generated_index]["quick_active_rl_floor_db"]),
                        "quick_high_active_rl_floor_db": float(generated_rows[generated_index]["quick_high_active_rl_floor_db"]),
                        **{f"optimizer_{key}": float(value) for key, value in diagnostics.items()},
                    },
                )
                ratio_rows.append(values)
            ratio_best = max(ratio_rows, key=full_selection_key)
            reserve_found = bool(int(ratio_best["design_reserve_pass"]))
            print(
                f"v20 scene={scene_position:02d}/{len(target_samples):02d} sample={sample} "
                f"K={k_value} ratio={ratio:.1f} strict={ratio_best['all_corner_strict_pass']} "
                f"reserve11={ratio_best['design_reserve_pass']} "
                f"RL={10.0 + float(ratio_best['robust_active_rl_margin_db']):.3f} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
            write_csv(args.out_dir / "checkpoint_generated_mask_manifest.csv", generated_rows)
            write_csv(args.out_dir / "checkpoint_full_refined_candidate_metrics.csv", full_rows)
            (args.out_dir / "checkpoint_progress.json").write_text(
                json.dumps(
                    {
                        "completed_scene_position": scene_position,
                        "completed_sample_index": sample,
                        "completed_ratio": ratio,
                        "generated_mask_count": len(generated_rows),
                        "full_candidate_count": len(full_rows),
                        "full_optimizer_run_count": optimizer_runs,
                        "elapsed_seconds": time.time() - started,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if reserve_found:
                break

    write_csv(args.out_dir / "generated_mask_manifest.csv", generated_rows)
    np.savez_compressed(
        args.out_dir / "generated_mask_pool.npz",
        generated_index=np.arange(len(generated_masks), dtype=np.int64),
        masks=np.stack(generated_masks) if generated_masks else np.zeros((0, 256), dtype=np.int8),
    )
    write_csv(args.out_dir / "full_refined_candidate_metrics.csv", full_rows)
    np.savez_compressed(
        args.out_dir / "full_refined_candidates.npz",
        evaluation_index=np.arange(len(full_rows), dtype=np.int64),
        sample_index=np.asarray(full_arrays["sample_index"], dtype=np.int64),
        k_values=np.asarray(full_arrays["k_values"], dtype=np.int8),
        ratio=np.asarray(full_arrays["ratio"], dtype=np.float32),
        masks=np.stack(full_arrays["mask"]),
        targets=np.stack(full_arrays["targets"]),
        tasks_real_imag=complex_to_ri(np.stack(full_arrays["tasks"])),
        robust_margins=np.stack(full_arrays["margins"]),
        corner_names=np.asarray(list(corners)),
        robust_margin_names=ROBUST_MARGIN_NAMES,
    )

    scene_rows: list[dict[str, Any]] = []
    for sample in samples:
        members = [row for row in full_rows if int(row["sample_index"]) == int(sample)]
        strict = [row for row in members if int(row["all_corner_strict_pass"]) == 1]
        reserve = [row for row in members if int(row["design_reserve_pass"]) == 1]
        best = max(members, key=full_selection_key)
        best_high = max(float(row["frequency_high_E2_source_active_rl_floor_db"]) for row in members)
        scene_rows.append(
            {
                "sample_index": int(sample),
                "k": int(best["k"]),
                "three_frequency_oracle_pass": int(bool(strict)),
                "three_frequency_reserve11_oracle_pass": int(bool(reserve)),
                "minimum_strict_ratio": min(float(row["ratio"]) for row in strict) if strict else float("nan"),
                "minimum_reserve11_ratio": min(float(row["ratio"]) for row in reserve) if reserve else float("nan"),
                "best_evaluation_index": int(best["evaluation_index"]),
                "best_ratio": float(best["ratio"]),
                "best_worst_margin_db": float(best["robust_worst_margin_db"]),
                "best_active_rl_floor_db": 10.0 + float(best["robust_active_rl_margin_db"]),
                "best_high_E2_active_rl_floor_db": best_high,
                "verified_full_candidate_count": len(members),
                "generated_mask_count": sum(
                    int(row["sample_index"]) == int(sample) for row in generated_rows
                ),
            }
        )
    write_csv(args.out_dir / "scene_oracle.csv", scene_rows)
    group_rows: list[dict[str, Any]] = []
    for k_value in (0, 2, 4, 6):
        members = [row for row in scene_rows if k_value == 0 or int(row["k"]) == k_value]
        group_rows.append(
            {
                "k": "all" if k_value == 0 else k_value,
                "scene_count": len(members),
                "strict_oracle_count": sum(int(row["three_frequency_oracle_pass"]) for row in members),
                "strict_oracle_rate": float(np.mean([int(row["three_frequency_oracle_pass"]) for row in members])),
                "reserve11_oracle_count": sum(int(row["three_frequency_reserve11_oracle_pass"]) for row in members),
                "reserve11_oracle_rate": float(np.mean([int(row["three_frequency_reserve11_oracle_pass"]) for row in members])),
                "high_best_of_n_ge_10p5_count": sum(float(row["best_high_E2_active_rl_floor_db"]) >= 10.5 for row in members),
                "high_best_of_n_ge_10p5_rate": float(np.mean([float(row["best_high_E2_active_rl_floor_db"]) >= 10.5 for row in members])),
            }
        )
    write_csv(args.out_dir / "oracle_by_k.csv", group_rows)
    overall = group_rows[0]
    per_k = {int(row["k"]): row for row in group_rows[1:]}
    stage_pass = bool(
        int(overall["strict_oracle_count"])
        >= int(acceptance["three_frequency_best_of_n_strict_pass_min"])
        and float(overall["strict_oracle_rate"])
        >= float(acceptance["three_frequency_best_of_n_strict_rate_min"])
        and all(
            float(per_k[k]["strict_oracle_rate"]) >= float(acceptance["per_k_strict_rate_min"])
            for k in (2, 4, 6)
        )
        and int(per_k[2]["strict_oracle_count"]) > 0
        and int(per_k[4]["strict_oracle_count"]) > 0
    )
    redesign_groups = [
        k
        for k in (2, 4)
        if float(per_k[k]["high_best_of_n_ge_10p5_rate"])
        < float(acceptance["hardware_redesign_trigger_k2_k4_rate_below_threshold"])
    ]
    summary = {
        "protocol": protocol["protocol"],
        "scene_count": len(scene_rows),
        "searched_failed_scene_count": len(target_samples),
        "generated_mask_count": len(generated_rows),
        "full_refined_candidate_count_including_v19": len(full_rows),
        "full_optimizer_run_count": optimizer_runs,
        "three_frequency_strict_oracle_count": int(overall["strict_oracle_count"]),
        "three_frequency_strict_oracle_rate": float(overall["strict_oracle_rate"]),
        "three_frequency_reserve11_oracle_count": int(overall["reserve11_oracle_count"]),
        "per_k": {str(k): per_k[k] for k in (2, 4, 6)},
        "stage_acceptance_pass": stage_pass,
        "hardware_redesign_trigger_groups": redesign_groups,
        "hfss_smoke_allowed": stage_pass,
        "critic_training_allowed": False,
        "thresholds_changed": False,
        "frequency_specific_commands_saved": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# v20 Three-Frequency Mask-Weight Joint Search",
        "",
        f"- Three-frequency strict oracle: {summary['three_frequency_strict_oracle_count']}/{summary['scene_count']}.",
        f"- Three-frequency 11 dB reserve oracle: {summary['three_frequency_reserve11_oracle_count']}/{summary['scene_count']}.",
        f"- Stage acceptance: {stage_pass}.",
        f"- Hardware redesign trigger groups: {redesign_groups}.",
        f"- Generated masks / fully refined candidates: {len(generated_rows)} / {len(full_rows)}.",
        "- Critic training remains locked in this stage.",
    ]
    (args.out_dir / "STAGE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
