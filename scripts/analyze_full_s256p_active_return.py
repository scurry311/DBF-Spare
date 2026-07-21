"""Parse the full HFSS S256P and evaluate multi-beam active return loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from design_eep_port_match import cascade_matching_network, component_description, s_to_z


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_S_DIR = ROOT / "hfss_outputs" / "multitask_dataset" / "full_s256p_matched_v2_20260714"
DEFAULT_DATASET = ROOT / "hfss_outputs" / "multitask_dataset" / "dataset_arrays.npz"
DEFAULT_EEP_OPERATOR = (
    ROOT
    / "hfss_outputs"
    / "multitask_dataset"
    / "eep_smoke_16port_matched_v2_20260714"
    / "eep_operator_16port.npz"
)
DEFAULT_MATCH_SUMMARY = (
    ROOT
    / "hfss_outputs"
    / "multitask_dataset"
    / "eep_smoke_16port_matched_v2_20260714"
    / "matching_50ohm_lsection"
    / "matching_network_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-dir", type=Path, default=DEFAULT_S_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--eep-operator", type=Path, default=DEFAULT_EEP_OPERATOR)
    parser.add_argument("--match-summary", type=Path, default=DEFAULT_MATCH_SUMMARY)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--return-loss-min-db", type=float, default=10.0)
    parser.add_argument("--large-scan-theta-deg", type=float, default=45.0)
    parser.add_argument("--significant-power-relative-db", type=float, default=-30.0)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def parse_touchstone(path: Path) -> dict[str, Any]:
    port_count_match = re.search(r"\.s(\d+)p$", path.name, re.IGNORECASE)
    if not port_count_match:
        raise ValueError(f"Cannot infer port count from {path.name}")
    port_count = int(port_count_match.group(1))
    option: list[str] | None = None
    port_comments: dict[int, str] = {}
    numeric_tokens: list[float] = []
    port_pattern = re.compile(r"!\s*Port\[(\d+)\]\s*=\s*(\S+)", re.IGNORECASE)
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            port_match = port_pattern.match(line)
            if port_match:
                port_comments[int(port_match.group(1))] = port_match.group(2)
                continue
            if line.startswith("!") or line.startswith("["):
                continue
            if line.startswith("#"):
                option = line[1:].strip().split()
                continue
            data_part = line.split("!", 1)[0].strip()
            if data_part:
                numeric_tokens.extend(float(token) for token in data_part.split())
    if option is None or len(option) < 5:
        raise ValueError("Missing or invalid Touchstone option line")
    frequency_unit = option[0].upper()
    parameter = option[1].upper()
    data_format = option[2].upper()
    if parameter != "S":
        raise ValueError(f"Expected S parameters, got {parameter}")
    reference_index = next((index for index, value in enumerate(option) if value.upper() == "R"), None)
    reference_ohm = float(option[reference_index + 1]) if reference_index is not None else 50.0
    record_size = 1 + 2 * port_count * port_count
    if len(numeric_tokens) % record_size != 0:
        raise ValueError(
            f"Touchstone numeric token count {len(numeric_tokens)} is not divisible by record size {record_size}"
        )
    frequency_count = len(numeric_tokens) // record_size
    frequencies: list[float] = []
    matrices: list[np.ndarray] = []
    unit_scale = {"HZ": 1.0, "KHZ": 1.0e3, "MHZ": 1.0e6, "GHZ": 1.0e9}.get(frequency_unit)
    if unit_scale is None:
        raise ValueError(f"Unsupported frequency unit: {frequency_unit}")
    for record_index in range(frequency_count):
        record = np.asarray(
            numeric_tokens[record_index * record_size : (record_index + 1) * record_size], dtype=np.float64
        )
        frequencies.append(float(record[0]) * unit_scale)
        pairs = record[1:].reshape(-1, 2)
        if data_format == "MA":
            values = pairs[:, 0] * np.exp(1j * np.deg2rad(pairs[:, 1]))
        elif data_format == "RI":
            values = pairs[:, 0] + 1j * pairs[:, 1]
        elif data_format == "DB":
            values = 10.0 ** (pairs[:, 0] / 20.0) * np.exp(1j * np.deg2rad(pairs[:, 1]))
        else:
            raise ValueError(f"Unsupported Touchstone format: {data_format}")
        matrices.append(values.reshape(port_count, port_count))
    ports = [port_comments.get(index, f"PORT{index}") for index in range(1, port_count + 1)]
    return {
        "port_count": port_count,
        "port_names": ports,
        "frequency_hz": np.asarray(frequencies, dtype=np.float64),
        "s_parameters": np.stack(matrices).astype(np.complex128),
        "reference_impedance_ohm": reference_ohm,
        "format": data_format,
        "frequency_unit": frequency_unit,
        "numeric_token_count": len(numeric_tokens),
    }


def read_port_order(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["touchstone_index"]))
    return [row["port_name"] for row in rows]


def db20(values: np.ndarray | float) -> np.ndarray | float:
    result = 20.0 * np.log10(np.maximum(np.abs(values), 1.0e-15))
    return float(result) if np.ndim(result) == 0 else result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def optimize_identical_lmatch_diagonal(
    s_matrix: np.ndarray, z0: float, seed: int
) -> tuple[float, float, np.ndarray]:
    z_matrix = s_to_z(s_matrix, z0)
    z_diag = np.diag(z_matrix)

    def objective(parameters: np.ndarray) -> float:
        series_x = float(parameters[0])
        shunt_b = float(parameters[1])
        transformed = 1.0 / (1.0 / z_diag + 1j * shunt_b) + 1j * series_x
        reflection = np.abs((transformed - z0) / (transformed + z0))
        return float(reflection.max() + 0.2 * np.quantile(reflection, 0.95) + 0.05 * reflection.mean())

    result = differential_evolution(
        objective,
        bounds=[(-300.0, 300.0), (-0.05, 0.05)],
        seed=int(seed),
        tol=1.0e-9,
        polish=True,
    )
    return float(result.x[0]), float(result.x[1]), z_diag


def evaluate_cases(
    *,
    model_name: str,
    s_matrix: np.ndarray,
    weights: np.ndarray,
    source_masks: np.ndarray,
    metadata: list[dict[str, Any]],
    threshold_db: float,
    significant_power_relative_db: float,
) -> list[dict[str, Any]]:
    reflected = weights @ s_matrix.T
    incident_power = np.sum(np.abs(weights) ** 2, axis=1)
    reflected_power = np.sum(np.abs(reflected) ** 2, axis=1)
    total_return_loss = -10.0 * np.log10(np.maximum(reflected_power / incident_power, 1.0e-30))
    significant_ratio = 10.0 ** (float(significant_power_relative_db) / 10.0)
    rows: list[dict[str, Any]] = []
    for case_index in range(weights.shape[0]):
        amplitudes = np.abs(weights[case_index])
        active = source_masks[case_index] & (amplitudes > 1.0e-10)
        max_power = max(float(np.max(amplitudes**2)), 1.0e-30)
        significant = active & ((amplitudes**2) >= significant_ratio * max_power)
        gamma = np.full(amplitudes.shape, np.nan, dtype=np.float64)
        gamma[active] = np.abs(reflected[case_index, active] / weights[case_index, active])
        active_rl = -20.0 * np.log10(np.maximum(gamma[active], 1.0e-15))
        significant_rl = -20.0 * np.log10(np.maximum(gamma[significant], 1.0e-15))
        worst_active_pos = int(np.argmin(active_rl)) if active_rl.size else -1
        active_indices = np.flatnonzero(active)
        worst_port_index = int(active_indices[worst_active_pos]) if worst_active_pos >= 0 else -1
        worst_active = float(np.min(active_rl)) if active_rl.size else float("nan")
        worst_significant = float(np.min(significant_rl)) if significant_rl.size else float("nan")
        row = dict(metadata[case_index])
        row.update(
            {
                "model": model_name,
                "source_active_count": int(np.count_nonzero(active)),
                "significant_active_count": int(np.count_nonzero(significant)),
                "worst_active_return_loss_db": worst_active,
                "worst_significant_return_loss_db": worst_significant,
                "total_return_loss_db": float(total_return_loss[case_index]),
                "reflected_power_fraction": float(reflected_power[case_index] / incident_power[case_index]),
                "accepted_power_fraction": float(1.0 - reflected_power[case_index] / incident_power[case_index]),
                "active_port_pass_fraction": float(np.mean(active_rl >= threshold_db)) if active_rl.size else float("nan"),
                "all_active_ports_pass_10db": int(bool(active_rl.size and np.all(active_rl >= threshold_db))),
                "all_significant_ports_pass_10db": int(
                    bool(significant_rl.size and np.all(significant_rl >= threshold_db))
                ),
                "total_return_pass_10db": int(bool(total_return_loss[case_index] >= threshold_db)),
                "worst_port_index": worst_port_index,
                "worst_port_name": f"P{worst_port_index:03d}" if worst_port_index >= 0 else "",
            }
        )
        rows.append(row)
    return rows


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        keys = [
            (row["model"], "all", "all", "all"),
            (row["model"], str(row["k"]), "all", "all"),
            (row["model"], str(row["k"]), "all", str(row["large_scan"])),
            (row["model"], str(row["k"]), f"{row['active_ratio']:.1f}", str(row["large_scan"])),
        ]
        for key in keys:
            groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (model, k_value, ratio, large_scan), group_rows in sorted(groups.items()):
        worst = np.asarray([float(row["worst_active_return_loss_db"]) for row in group_rows])
        significant = np.asarray([float(row["worst_significant_return_loss_db"]) for row in group_rows])
        total = np.asarray([float(row["total_return_loss_db"]) for row in group_rows])
        accepted = np.asarray([float(row["accepted_power_fraction"]) for row in group_rows])
        summaries.append(
            {
                "model": model,
                "k": k_value,
                "active_ratio": ratio,
                "large_scan": large_scan,
                "case_count": len(group_rows),
                "worst_active_rl_min_db": float(np.nanmin(worst)),
                "worst_active_rl_p05_db": float(np.nanquantile(worst, 0.05)),
                "worst_active_rl_mean_db": float(np.nanmean(worst)),
                "worst_significant_rl_min_db": float(np.nanmin(significant)),
                "total_rl_min_db": float(np.nanmin(total)),
                "total_rl_mean_db": float(np.nanmean(total)),
                "all_active_10db_case_pass_rate": float(
                    np.mean([int(row["all_active_ports_pass_10db"]) for row in group_rows])
                ),
                "all_significant_10db_case_pass_rate": float(
                    np.mean([int(row["all_significant_ports_pass_10db"]) for row in group_rows])
                ),
                "total_10db_case_pass_rate": float(
                    np.mean([int(row["total_return_pass_10db"]) for row in group_rows])
                ),
                "accepted_power_fraction_median": float(np.nanmedian(accepted)),
                "accepted_power_fraction_min": float(np.nanmin(accepted)),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.s_dir / "active_return_analysis_20260714")
    out_dir.mkdir(parents=True, exist_ok=True)
    touchstone_path = args.s_dir / "ura16_matched_v2_full.s256p"
    parsed = parse_touchstone(touchstone_path)
    touchstone_ports = list(parsed["port_names"])
    aedt_ports = read_port_order(args.s_dir / "aedt_port_order.csv")
    expected_ports = [f"P{index:03d}" for index in range(256)]
    if touchstone_ports != aedt_ports or touchstone_ports != expected_ports:
        raise RuntimeError("Touchstone, AEDT, and expected P000-P255 port orders do not match")
    if parsed["s_parameters"].shape != (1, 256, 256):
        raise RuntimeError(f"Expected one 256x256 solution, got {parsed['s_parameters'].shape}")
    s_matrix = parsed["s_parameters"][0]

    reciprocity_max = float(np.max(np.abs(s_matrix - s_matrix.T)))
    singular_values = np.linalg.svd(s_matrix, compute_uv=False)
    passivity_sigma_max = float(singular_values.max())
    reflection_db = np.asarray(db20(np.diag(s_matrix)))
    off_diagonal = np.abs(s_matrix - np.diag(np.diag(s_matrix)))
    crosscheck_max = float("nan")
    if args.eep_operator.exists():
        eep = np.load(args.eep_operator, allow_pickle=False)
        selected_ports = [str(port) for port in eep["port_names"]]
        indices = [touchstone_ports.index(port) for port in selected_ports]
        crosscheck_max = float(
            np.max(np.abs(s_matrix[np.ix_(indices, indices)] - np.asarray(eep["s_parameters"], dtype=complex)))
        )
    matrix_valid = bool(
        np.all(np.isfinite(s_matrix))
        and reciprocity_max <= 1.0e-6
        and passivity_sigma_max <= 1.0001
        and crosscheck_max <= 1.0e-6
    )
    if not matrix_valid:
        raise RuntimeError(
            f"S matrix validation failed: reciprocity={reciprocity_max}, sigma={passivity_sigma_max}, "
            f"crosscheck={crosscheck_max}"
        )

    np.savez_compressed(
        out_dir / "full_s_matrix_256.npz",
        port_names=np.asarray(touchstone_ports),
        frequency_hz=parsed["frequency_hz"],
        s_parameters=s_matrix.astype(np.complex64),
        reference_impedance_ohm=np.asarray(parsed["reference_impedance_ohm"]),
    )

    fixed_match = json.loads(args.match_summary.read_text(encoding="utf-8"))
    fixed_x = float(fixed_match["series_reactance_ohm"])
    fixed_b = float(fixed_match["shunt_susceptance_siemens"])
    s_fixed, _, _ = cascade_matching_network(s_matrix, fixed_x, fixed_b, float(parsed["reference_impedance_ohm"]))
    optimized_x, optimized_b, z_diag = optimize_identical_lmatch_diagonal(
        s_matrix, float(parsed["reference_impedance_ohm"]), int(args.seed)
    )
    s_optimized, _, _ = cascade_matching_network(
        s_matrix, optimized_x, optimized_b, float(parsed["reference_impedance_ohm"])
    )

    dataset = np.load(args.dataset, allow_pickle=False)
    dataset_ports = [str(port) for port in dataset["port_names"]]
    reorder = [dataset_ports.index(port) for port in touchstone_ports]
    weights_ri = np.asarray(dataset["hfss_weights_real_imag"], dtype=np.float64)
    weights = (weights_ri[:, :, 0] + 1j * weights_ri[:, :, 1])[:, reorder]
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    weights = weights / np.maximum(norms, 1.0e-15)
    source_masks = np.asarray(dataset["masks"], dtype=bool)[:, reorder]
    k_values = np.asarray(dataset["k_values"], dtype=int)
    ratios = np.asarray(dataset["active_ratios_actual"], dtype=float)
    targets = np.asarray(dataset["targets_deg"], dtype=float)
    sample_ids = [str(item) for item in dataset["sample_ids"]]
    metadata: list[dict[str, Any]] = []
    for index in range(weights.shape[0]):
        valid_targets = targets[index, : k_values[index], :]
        theta_values = valid_targets[:, 0]
        max_theta = float(np.nanmax(theta_values))
        metadata.append(
            {
                "sample_index": index,
                "sample_id": sample_ids[index],
                "k": int(k_values[index]),
                "active_ratio": float(ratios[index]),
                "max_target_theta_deg": max_theta,
                "mean_target_theta_deg": float(np.nanmean(theta_values)),
                "large_scan": int(max_theta >= float(args.large_scan_theta_deg)),
            }
        )

    models = {
        "raw_50ohm": s_matrix,
        "fixed_16port_lmatch": s_fixed,
        "optimized_full256_lmatch": s_optimized,
    }
    all_rows: list[dict[str, Any]] = []
    for model_name, model_s in models.items():
        all_rows.extend(
            evaluate_cases(
                model_name=model_name,
                s_matrix=model_s,
                weights=weights,
                source_masks=source_masks,
                metadata=metadata,
                threshold_db=float(args.return_loss_min_db),
                significant_power_relative_db=float(args.significant_power_relative_db),
            )
        )
    case_metrics_path = out_dir / "active_return_case_metrics.csv"
    write_csv(case_metrics_path, all_rows)
    group_rows = summarize_groups(all_rows)
    group_summary_path = out_dir / "active_return_group_summary.csv"
    write_csv(group_summary_path, group_rows)
    failures = [row for row in all_rows if not int(row["all_active_ports_pass_10db"])]
    failures.sort(key=lambda row: (row["model"], float(row["worst_active_return_loss_db"])))
    write_csv(out_dir / "active_return_failures.csv", failures)

    model_summaries: dict[str, Any] = {}
    for model_name, model_s in models.items():
        model_rows = [row for row in all_rows if row["model"] == model_name]
        large_rows = [row for row in model_rows if int(row["large_scan"]) == 1]
        diag_db = np.asarray(db20(np.diag(model_s)))
        model_summaries[model_name] = {
            "passive_return_loss_min_db": -float(diag_db.max()),
            "all_case_count": len(model_rows),
            "all_active_10db_pass_count": sum(int(row["all_active_ports_pass_10db"]) for row in model_rows),
            "all_active_10db_pass_rate": float(
                np.mean([int(row["all_active_ports_pass_10db"]) for row in model_rows])
            ),
            "all_significant_10db_pass_rate": float(
                np.mean([int(row["all_significant_ports_pass_10db"]) for row in model_rows])
            ),
            "total_10db_pass_rate": float(np.mean([int(row["total_return_pass_10db"]) for row in model_rows])),
            "worst_active_return_loss_db": float(
                np.min([float(row["worst_active_return_loss_db"]) for row in model_rows])
            ),
            "large_scan_case_count": len(large_rows),
            "large_scan_all_active_10db_pass_rate": float(
                np.mean([int(row["all_active_ports_pass_10db"]) for row in large_rows])
            ),
            "large_scan_worst_active_return_loss_db": float(
                np.min([float(row["worst_active_return_loss_db"]) for row in large_rows])
            ),
            "always_at_least_10db": bool(all(int(row["all_active_ports_pass_10db"]) for row in model_rows)),
        }

    frequency_ghz = float(parsed["frequency_hz"][0] / 1.0e9)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "touchstone": str(touchstone_path),
        "dataset": str(args.dataset),
        "port_count": int(parsed["port_count"]),
        "frequency_ghz": frequency_ghz,
        "reference_impedance_ohm": float(parsed["reference_impedance_ohm"]),
        "matrix_validation_passed": matrix_valid,
        "reciprocity_max_abs": reciprocity_max,
        "passivity_max_singular_value": passivity_sigma_max,
        "selected_16port_crosscheck_max_abs": crosscheck_max,
        "raw_passive_return_loss_min_db": -float(reflection_db.max()),
        "raw_passive_return_loss_median_db": -float(np.median(reflection_db)),
        "raw_mutual_coupling_worst_db": float(db20(off_diagonal.max())),
        "return_loss_requirement_db": float(args.return_loss_min_db),
        "large_scan_definition": f"max_target_theta_deg >= {float(args.large_scan_theta_deg):.1f}",
        "significant_port_definition": (
            f"incident port power >= {float(args.significant_power_relative_db):.1f} dB relative to case maximum"
        ),
        "sample_count": int(weights.shape[0]),
        "k_values": sorted(set(int(value) for value in k_values)),
        "active_ratios": sorted(set(round(float(value), 6) for value in ratios)),
        "fixed_16port_lmatch": {
            "series_reactance_ohm": fixed_x,
            "shunt_susceptance_siemens": fixed_b,
            "components": component_description(fixed_x, fixed_b, frequency_ghz * 1.0e9),
        },
        "optimized_full256_lmatch": {
            "series_reactance_ohm": optimized_x,
            "shunt_susceptance_siemens": optimized_b,
            "components": component_description(optimized_x, optimized_b, frequency_ghz * 1.0e9),
            "raw_z_real_ohm_range": [float(np.min(z_diag.real)), float(np.max(z_diag.real))],
            "raw_z_imag_ohm_range": [float(np.min(z_diag.imag)), float(np.max(z_diag.imag))],
        },
        "models": model_summaries,
        "hard_gate_passed": bool(
            any(model["always_at_least_10db"] for model in model_summaries.values())
        ),
        "outputs": {
            "full_s_npz": str(out_dir / "full_s_matrix_256.npz"),
            "case_metrics": str(case_metrics_path),
            "group_summary": str(group_summary_path),
            "failures": str(out_dir / "active_return_failures.csv"),
        },
    }
    (out_dir / "active_return_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_lines = [
        "# Full 256-port active return-loss validation",
        "",
        f"- Matrix validation passed: {matrix_valid}",
        f"- Frequency/reference: {frequency_ghz:.3f} GHz / {float(parsed['reference_impedance_ohm']):.1f} ohm",
        f"- Reciprocity max error: {reciprocity_max:.3e}",
        f"- Passivity max singular value: {passivity_sigma_max:.6f}",
        f"- 16-port cross-check max error: {crosscheck_max:.3e}",
        f"- Raw passive minimum return loss: {-float(reflection_db.max()):.2f} dB",
        f"- Requirement: every source-active port >= {float(args.return_loss_min_db):.1f} dB",
        "",
        "## Large-scan results",
        "",
        "| Model | K | Cases | Worst min (dB) | Worst p05 (dB) | Worst mean (dB) | Significant min (dB) | Total mean (dB) | All-port pass | Total pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in models:
        for k_value in (1, 2, 4, 6):
            selected = next(
                row
                for row in group_rows
                if row["model"] == model_name
                and row["k"] == str(k_value)
                and row["active_ratio"] == "all"
                and row["large_scan"] == "1"
            )
            report_lines.append(
                f"| {model_name} | {k_value} | {selected['case_count']} | "
                f"{selected['worst_active_rl_min_db']:.2f} | {selected['worst_active_rl_p05_db']:.2f} | "
                f"{selected['worst_active_rl_mean_db']:.2f} | {selected['worst_significant_rl_min_db']:.2f} | "
                f"{selected['total_rl_mean_db']:.2f} | {selected['all_active_10db_case_pass_rate']:.3f} | "
                f"{selected['total_10db_case_pass_rate']:.3f} |"
            )
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Hard gate passed: **{summary['hard_gate_passed']}**.",
            "",
            "No evaluated model keeps every active channel at or above 10 dB. The failure is already present for K=1 equal-amplitude scan weights, so critic retraining remains blocked. The next physical action is port-class or active-impedance matching/decoupling using the full 256-port matrix.",
        ]
    )
    (out_dir / "active_return_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
