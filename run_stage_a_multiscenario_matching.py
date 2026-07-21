"""Run the Stage-A full-S active-match feasibility study without launching HFSS.

This script intentionally evaluates only ideal, single-frequency circuit models
against the exported 256-port HFSS S matrix.  It does not create full-wave
labels.  The local cross-reactance network is an upper-bound reference for a
small reciprocal decoupling circuit and is not treated as a fabricated design.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from design_eep_port_match import s_to_z, z_to_s


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "hfss_outputs" / "multitask_dataset"
BASE = DATASET_ROOT / "full_s256p_matched_v2_20260714"
DEFAULT_S = BASE / "active_return_analysis_20260714" / "full_s_matrix_256.npz"
DEFAULT_DATASET = DATASET_ROOT / "dataset_arrays.npz"
DEFAULT_CLASSES = BASE / "port_class_matching_20260714" / "port_clusters.csv"
DEFAULT_CLASS_NETWORK = BASE / "port_class_matching_20260714" / "port_class_matched_s256.npz"
DEFAULT_OUT = BASE / "stage_a_multiscenario_matching_20260714"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-npz", type=Path, default=DEFAULT_S)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--class-network", type=Path, default=DEFAULT_CLASS_NETWORK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--return-loss-min-db", type=float, default=10.0)
    parser.add_argument("--large-scan-theta-deg", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--de-maxiter", type=int, default=120)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def db(value: np.ndarray) -> np.ndarray:
    return -20.0 * np.log10(np.maximum(np.abs(value), 1.0e-15))


def active_z_from_s(weights: np.ndarray, masks: np.ndarray, s_matrix: np.ndarray, z0: float) -> tuple[np.ndarray, np.ndarray]:
    reflected = weights @ s_matrix.T
    gamma = np.full(weights.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    active = masks & (np.abs(weights) > 1.0e-10)
    gamma[active] = reflected[active] / weights[active]
    with np.errstate(divide="ignore", invalid="ignore"):
        active_z = z0 * (1.0 + gamma) / (1.0 - gamma)
    return gamma, active_z


def evaluate_network(
    name: str,
    s_matrix: np.ndarray,
    weights: np.ndarray,
    masks: np.ndarray,
    k_values: np.ndarray,
    ratios: np.ndarray,
    large_scan: np.ndarray,
    requirement_db: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    reflected = weights @ s_matrix.T
    active = masks & (np.abs(weights) > 1.0e-10)
    gamma = np.full(weights.shape, np.nan, dtype=np.float64)
    gamma[active] = np.abs(reflected[active] / weights[active])
    rho = 10.0 ** (-requirement_db / 20.0)
    incident = np.sum(np.abs(weights) ** 2, axis=1)
    reflected_power = np.sum(np.abs(reflected) ** 2, axis=1)
    total_rl = -10.0 * np.log10(np.maximum(reflected_power / np.maximum(incident, 1.0e-30), 1.0e-30))
    rows: list[dict[str, Any]] = []
    worst_values = np.empty(weights.shape[0], dtype=np.float64)
    for index in range(weights.shape[0]):
        values = gamma[index, active[index]]
        worst_values[index] = -20.0 * math.log10(max(float(np.max(values)), 1.0e-15))
        rows.append(
            {
                "model": name,
                "sample_index": index,
                "k": int(k_values[index]),
                "active_ratio": float(ratios[index]),
                "large_scan": int(large_scan[index]),
                "active_count": int(np.count_nonzero(active[index])),
                "worst_active_return_loss_db": float(worst_values[index]),
                "total_return_loss_db": float(total_rl[index]),
                "reflected_power_fraction": float(reflected_power[index] / max(incident[index], 1.0e-30)),
                "accepted_power_fraction": float(1.0 - reflected_power[index] / max(incident[index], 1.0e-30)),
                "all_active_10db_pass": int(np.all(values <= rho)),
                "total_10db_pass": int(total_rl[index] >= requirement_db),
                "engineering_10db_gate_pass": int(np.all(values <= rho) and total_rl[index] >= requirement_db),
            }
        )
    port_rl = db(gamma[active])
    return rows, {
        "port_rl_p01_db": float(np.nanquantile(port_rl, 0.01)),
        "case_worst_rl_p05_db": float(np.quantile(worst_values, 0.05)),
        "total_rl_median_db": float(np.median(total_rl)),
    }


def summarize_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        keys = [
            (row["model"], "all", "all", "all"),
            (row["model"], str(row["k"]), "all", "all"),
            (row["model"], str(row["k"]), "all", str(row["large_scan"])),
            (row["model"], str(row["k"]), f"{row['active_ratio']:.3f}", str(row["large_scan"])),
        ]
        for key in keys:
            groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (model, k_value, ratio, large), group in sorted(groups.items()):
        worst = np.asarray([float(item["worst_active_return_loss_db"]) for item in group])
        total = np.asarray([float(item["total_return_loss_db"]) for item in group])
        accepted = np.asarray([float(item["accepted_power_fraction"]) for item in group])
        output.append(
            {
                "model": model,
                "k": k_value,
                "active_ratio": ratio,
                "large_scan": large,
                "case_count": len(group),
                "worst_active_rl_min_db": float(np.min(worst)),
                "worst_active_rl_p05_db": float(np.quantile(worst, 0.05)),
                "worst_active_rl_mean_db": float(np.mean(worst)),
                "total_rl_min_db": float(np.min(total)),
                "total_rl_mean_db": float(np.mean(total)),
                "accepted_power_fraction_mean": float(np.mean(accepted)),
                "all_active_10db_pass_rate": float(np.mean([item["all_active_10db_pass"] for item in group])),
                "total_10db_pass_rate": float(np.mean([item["total_10db_pass"] for item in group])),
                "engineering_10db_gate_pass_rate": float(
                    np.mean([item["engineering_10db_gate_pass"] for item in group])
                ),
            }
        )
    return output


def transform_class_network(
    z_antenna: np.ndarray,
    class_ids: np.ndarray,
    parameters: dict[int, tuple[float, float, float]],
) -> np.ndarray:
    """Apply a lossless transformer plus L section, using impedance matrices."""
    count = z_antenna.shape[0]
    x = np.asarray([parameters[int(label)][1] for label in class_ids], dtype=np.float64)
    b = np.asarray([parameters[int(label)][2] for label in class_ids], dtype=np.float64)
    turns = np.asarray([math.exp(parameters[int(label)][0]) for label in class_ids], dtype=np.float64)
    identity = np.eye(count, dtype=np.complex128)
    shunt_loaded = np.linalg.solve(np.linalg.inv(z_antenna) + 1j * np.diag(b), identity)
    l_matched = shunt_loaded + 1j * np.diag(x)
    return turns[:, None] * l_matched * turns[None, :]


def fit_class_transformer_l(
    impedances: np.ndarray,
    z0: float,
    seed: int,
    maxiter: int,
) -> tuple[float, float, float, dict[str, float]]:
    samples = impedances[np.isfinite(impedances)]
    if samples.size == 0:
        raise ValueError("No finite active impedance samples")

    def objective(vector: np.ndarray) -> float:
        log_turns, series_x, shunt_b = (float(value) for value in vector)
        z_l = 1.0 / (1.0 / samples + 1j * shunt_b) + 1j * series_x
        z_in = math.exp(2.0 * log_turns) * z_l
        gamma = np.abs((z_in - z0) / (z_in + z0))
        return float(np.quantile(gamma, 0.99) + 0.35 * np.quantile(gamma, 0.95) + 0.10 * np.median(gamma))

    result = differential_evolution(
        objective,
        bounds=[(-1.2, 1.2), (-300.0, 300.0), (-0.05, 0.05)],
        seed=seed,
        maxiter=maxiter,
        tol=1.0e-7,
        polish=True,
        workers=1,
    )
    log_turns, series_x, shunt_b = (float(value) for value in result.x)
    z_l = 1.0 / (1.0 / samples + 1j * shunt_b) + 1j * series_x
    gamma = np.abs((math.exp(2.0 * log_turns) * z_l - z0) / (math.exp(2.0 * log_turns) * z_l + z0))
    return log_turns, series_x, shunt_b, {
        "sample_count": int(samples.size),
        "scalar_rl_min_db": float(np.min(db(gamma))),
        "scalar_rl_p01_db": float(np.quantile(db(gamma), 0.01)),
        "scalar_rl_p05_db": float(np.quantile(db(gamma), 0.05)),
        "scalar_rl_median_db": float(np.median(db(gamma))),
        "objective": float(result.fun),
    }


def build_local_cross_reactance() -> tuple[np.ndarray, np.ndarray]:
    """Return fixed horizontal/vertical 2x2-block coupling templates in ohms."""
    horizontal = np.zeros((256, 256), dtype=np.float64)
    vertical = np.zeros((256, 256), dtype=np.float64)
    for ix in range(16):
        for iy in range(0, 16, 2):
            a, b = ix * 16 + iy, ix * 16 + iy + 1
            horizontal[a, b] = horizontal[b, a] = 1.0
    for ix in range(0, 16, 2):
        for iy in range(16):
            a, b = ix * 16 + iy, (ix + 1) * 16 + iy
            vertical[a, b] = vertical[b, a] = 1.0
    return horizontal, vertical


def impedance_distribution_rows(
    active_z: np.ndarray,
    gamma: np.ndarray,
    masks: np.ndarray,
    geometry: np.ndarray,
    k_values: np.ndarray,
    ratios: np.ndarray,
    large_scan: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for geometry_class in ("corner", "edge", "interior"):
        ports = geometry == geometry_class
        for k_value in sorted(np.unique(k_values)):
            for ratio in sorted(np.unique(ratios)):
                for large in (0, 1):
                    cases = (k_values == k_value) & np.isclose(ratios, ratio) & (large_scan == large)
                    selected = cases[:, None] & masks & ports[None, :]
                    z_values = active_z[selected]
                    gamma_values = gamma[selected]
                    finite = np.isfinite(z_values) & np.isfinite(gamma_values)
                    if not np.any(finite):
                        continue
                    z_values = z_values[finite]
                    rl = db(gamma_values[finite])
                    rows.append(
                        {
                            "geometry_class": geometry_class,
                            "k": int(k_value),
                            "active_ratio": float(ratio),
                            "large_scan": int(large),
                            "active_port_observations": int(z_values.size),
                            "active_z_real_p05_ohm": float(np.quantile(z_values.real, 0.05)),
                            "active_z_real_median_ohm": float(np.median(z_values.real)),
                            "active_z_real_p95_ohm": float(np.quantile(z_values.real, 0.95)),
                            "active_z_imag_p05_ohm": float(np.quantile(z_values.imag, 0.05)),
                            "active_z_imag_median_ohm": float(np.median(z_values.imag)),
                            "active_z_imag_p95_ohm": float(np.quantile(z_values.imag, 0.95)),
                            "active_rl_min_db": float(np.min(rl)),
                            "active_rl_p05_db": float(np.quantile(rl, 0.05)),
                            "active_rl_median_db": float(np.median(rl)),
                            "active_rl_10db_port_pass_rate": float(np.mean(rl >= 10.0)),
                        }
                    )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    s_payload = np.load(args.s_npz, allow_pickle=False)
    s_raw = np.asarray(s_payload["s_parameters"], dtype=np.complex128)
    z0 = float(s_payload["reference_impedance_ohm"])
    ports = [str(value) for value in s_payload["port_names"]]
    z_raw = s_to_z(s_raw, z0)

    dataset = np.load(args.dataset, allow_pickle=False)
    dataset_ports = [str(value) for value in dataset["port_names"]]
    reorder = [dataset_ports.index(port) for port in ports]
    raw_weights = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float64)
    weights = (raw_weights[:, :, 0] + 1j * raw_weights[:, :, 1])[:, reorder]
    masks = np.asarray(dataset["masks"], dtype=bool)[:, reorder]
    weights *= masks
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1.0e-15)
    k_values = np.asarray(dataset["k_values"], dtype=int)
    ratios = np.asarray(dataset["active_ratios_actual"], dtype=float)
    targets = np.asarray(dataset["targets_deg"], dtype=float)
    large_scan = np.nanmax(targets[np.arange(targets.shape[0]), :], axis=(1, 2)) >= float(args.large_scan_theta_deg)

    class_rows = list(csv.DictReader(args.classes.open(encoding="utf-8-sig")))
    class_rows.sort(key=lambda row: int(row["port_index"]))
    if [row["port_name"] for row in class_rows] != ports:
        raise RuntimeError("Port class CSV order does not match the S256 port order")
    class_ids = np.asarray([int(row["cluster_id"]) for row in class_rows], dtype=int)
    geometry = np.asarray([row["geometry_class"] for row in class_rows])
    class_names = {int(row["cluster_id"]): row["cluster_name"] for row in class_rows}

    raw_gamma, raw_active_z = active_z_from_s(weights, masks, s_raw, z0)
    write_csv(
        args.out_dir / "raw_active_impedance_distribution.csv",
        impedance_distribution_rows(raw_active_z, raw_gamma, masks, geometry, k_values, ratios, large_scan),
    )

    all_case_rows: list[dict[str, Any]] = []
    model_scores: dict[str, dict[str, float]] = {}
    rows, model_scores["raw_50ohm"] = evaluate_network(
        "raw_50ohm", s_raw, weights, masks, k_values, ratios, large_scan, args.return_loss_min_db
    )
    all_case_rows.extend(rows)

    existing = np.load(args.class_network, allow_pickle=False)
    s_existing = np.asarray(existing["s_parameters"], dtype=np.complex128)
    rows, model_scores["existing_class_l_k1"] = evaluate_network(
        "existing_class_l_k1", s_existing, weights, masks, k_values, ratios, large_scan, args.return_loss_min_db
    )
    all_case_rows.extend(rows)

    parameters: dict[int, tuple[float, float, float]] = {}
    parameter_rows: list[dict[str, Any]] = []
    for class_id in sorted(np.unique(class_ids)):
        class_mask = class_ids == class_id
        samples_all = raw_active_z[:, class_mask][masks[:, class_mask]]
        gamma_all = np.abs(raw_gamma[:, class_mask][masks[:, class_mask]])
        # Negative-resistance active loads and |Gamma| >= 0.95 cannot be
        # robustly corrected by a fixed passive one-port network. They remain
        # in the exact 2400-case acceptance test; only the fitting objective is
        # trimmed so it does not collapse to a parameter bound.
        robust = np.isfinite(samples_all) & np.isfinite(gamma_all) & (samples_all.real > 0.0) & (gamma_all <= 0.95)
        samples = samples_all[robust]
        log_turns, series_x, shunt_b, metrics = fit_class_transformer_l(
            samples, z0, int(args.seed) + int(class_id), int(args.de_maxiter)
        )
        parameters[int(class_id)] = (log_turns, series_x, shunt_b)
        parameter_rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "turns_ratio": math.exp(log_turns),
                "series_reactance_ohm": series_x,
                "shunt_susceptance_siemens": shunt_b,
                "raw_active_impedance_sample_count": int(samples_all.size),
                "robust_fit_sample_count": int(samples.size),
                "excluded_nonpassive_or_near_unity_gamma_fraction": float(
                    1.0 - samples.size / max(samples_all.size, 1)
                ),
                **metrics,
            }
        )
    write_csv(args.out_dir / "multiscenario_transformer_l_parameters.csv", parameter_rows)
    z_tl = transform_class_network(z_raw, class_ids, parameters)
    s_tl = z_to_s(z_tl, z0)
    rows, model_scores["class_transformer_l_minimax"] = evaluate_network(
        "class_transformer_l_minimax", s_tl, weights, masks, k_values, ratios, large_scan, args.return_loss_min_db
    )
    all_case_rows.extend(rows)

    # This is a bounded, ideal circuit reference.  Its inclusion does not
    # constitute a layout-level decoupling-network design or HFSS validation.
    horizontal, vertical = build_local_cross_reactance()
    grid = (-100.0, -50.0, 0.0, 50.0, 100.0)
    decoupling_trials: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_s = s_tl
    best_pair = (0.0, 0.0)
    for x_h in grid:
        for x_v in grid:
            s_trial = z_to_s(z_tl + 1j * (x_h * horizontal + x_v * vertical), z0)
            _trial_rows, score = evaluate_network(
                "trial", s_trial, weights, masks, k_values, ratios, large_scan, args.return_loss_min_db
            )
            robust_score = score["port_rl_p01_db"] + 0.20 * score["case_worst_rl_p05_db"] + 0.05 * score["total_rl_median_db"]
            decoupling_trials.append(
                {
                    "horizontal_cross_reactance_ohm": x_h,
                    "vertical_cross_reactance_ohm": x_v,
                    "robust_score": robust_score,
                    **score,
                }
            )
            if robust_score > best_score:
                best_score, best_s, best_pair = robust_score, s_trial, (x_h, x_v)
    write_csv(args.out_dir / "ideal_local_decoupling_grid.csv", decoupling_trials)
    rows, model_scores["class_tl_plus_ideal_local_decoupling"] = evaluate_network(
        "class_tl_plus_ideal_local_decoupling",
        best_s,
        weights,
        masks,
        k_values,
        ratios,
        large_scan,
        args.return_loss_min_db,
    )
    all_case_rows.extend(rows)

    write_csv(args.out_dir / "stage_a_case_metrics.csv", all_case_rows)
    group_rows = summarize_case_rows(all_case_rows)
    write_csv(args.out_dir / "stage_a_group_summary.csv", group_rows)

    overall = [row for row in group_rows if row["k"] == "all" and row["active_ratio"] == "all"]
    overall_by_model = {row["model"]: row for row in overall}
    large_k_rows = [row for row in group_rows if row["k"] in {"1", "2", "4", "6"} and row["active_ratio"] == "all" and row["large_scan"] == "1"]
    meaningful_large_scan = {
        model: all(
            float(row["engineering_10db_gate_pass_rate"]) >= 0.05
            for row in large_k_rows
            if row["model"] == model
        )
        for model in overall_by_model
    }
    physically_admissible = {"raw_50ohm", "existing_class_l_k1", "class_transformer_l_minimax"}
    admissible_models = [model for model in physically_admissible if meaningful_large_scan.get(model, False)]
    best_model = max(
        overall_by_model,
        key=lambda model: float(overall_by_model[model]["engineering_10db_gate_pass_rate"]),
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "single-frequency S256 circuit feasibility only; no AEDT/HFSS task was launched",
        "case_count": int(weights.shape[0]),
        "port_count": int(weights.shape[1]),
        "return_loss_requirement_db": float(args.return_loss_min_db),
        "models": {
            "raw_50ohm": "raw HFSS S256 at 50 ohm",
            "existing_class_l_k1": "existing seven-class lossless L match fitted on K=1 data",
            "class_transformer_l_minimax": "seven-class ideal transformer plus L match fitted across all active-impedance samples",
            "class_tl_plus_ideal_local_decoupling": "ideal reciprocal local cross-reactance reference; not an embedded hardware design",
        },
        "model_scores": model_scores,
        "overall": overall_by_model,
        "best_ideal_local_decoupling": {
            "horizontal_cross_reactance_ohm": best_pair[0],
            "vertical_cross_reactance_ohm": best_pair[1],
            "robust_score": best_score,
        },
        "meaningful_large_scan_gate": meaningful_large_scan,
        "physically_admissible_models_passing": admissible_models,
        "stage_b_eep_allowed": bool(admissible_models),
        "new_hfss_labels_allowed": bool(admissible_models),
        "decision": (
            "allow_stage_b_eep_smoke_only" if admissible_models
            else "block_stage_b_and_new_hfss_labels_due_to_active_match_infeasibility"
        ),
        "limitations": [
            "All networks are ideal, lossless, and single-frequency circuit models based on the exported S256 matrix.",
            "The local decoupling result is a circuit upper-bound reference and cannot be called a fabricated or HFSS-validated network.",
            "No PSLL, gain, EEP, isolation, bandwidth, loss, or full-wave directional conclusion is made by this stage.",
        ],
    }
    (args.out_dir / "stage_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Stage A: multi-scenario active-match feasibility",
        "",
        "This stage evaluated the full 256-port HFSS S matrix only. No AEDT/HFSS job was launched.",
        "",
        "## Overall gate results",
        "",
    ]
    for model, row in overall_by_model.items():
        report.append(
            f"- {model}: all-active 10 dB={float(row['all_active_10db_pass_rate']):.3%}, "
            f"total 10 dB={float(row['total_10db_pass_rate']):.3%}, "
            f"joint gate={float(row['engineering_10db_gate_pass_rate']):.3%}, "
            f"mean total RL={float(row['total_rl_mean_db']):.2f} dB"
        )
    report.extend(
        [
            "",
            "## Decision",
            "",
            f"- {summary['decision']}",
            "- An ideal local decoupling reference is not sufficient evidence for physical implementation.",
            "- If blocked, the minimal next physical change is a layout-level, bandwidth-aware port-class matching/decoupling network or a revised element/feed geometry, followed by the same S256 gate.",
        ]
    )
    (args.out_dir / "stage_a_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "stage_a_network_matrices.npz",
        raw_s=s_raw.astype(np.complex64),
        class_transformer_l_s=s_tl.astype(np.complex64),
        ideal_local_decoupling_s=best_s.astype(np.complex64),
        horizontal_cross_reactance_ohm=np.asarray(best_pair[0]),
        vertical_cross_reactance_ohm=np.asarray(best_pair[1]),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
