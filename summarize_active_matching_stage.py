"""Summarize port clustering, class matching, projection, and label gate results."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "hfss_outputs" / "multitask_dataset" / "full_s256p_matched_v2_20260714"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    class_dir = BASE / "port_class_matching_20260714"
    projection_dir = BASE / "active_return_projection_strict_v2_20260714"
    class_summary = load_json(class_dir / "port_class_matching_summary.json")
    projection = load_json(projection_dir / "active_return_projection_summary.json")
    label_gate = load_json(projection_dir / "fullwave_label_gate_decision.json")
    with (class_dir / "port_class_matching_networks.csv").open(encoding="utf-8-sig") as handle:
        class_rows = list(csv.DictReader(handle))
    with (projection_dir / "projected_active_return_group_summary.csv").open(encoding="utf-8-sig") as handle:
        group_rows = list(csv.DictReader(handle))
    large_scan = {
        int(row["k"]): row
        for row in group_rows
        if row["active_ratio"] == "all" and row["large_scan"] == "1" and row["k"] != "all"
    }
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port_class_matching": class_summary,
        "strict_projection": projection,
        "fullwave_label_gate": label_gate,
        "stage_status": "blocked_before_new_fullwave_labels",
    }
    (BASE / "active_matching_stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Active matching and projection stage summary",
        "",
        "## Port classes",
        "",
        f"- Port count/classes: 256 / {class_summary['class_count']}",
        f"- Class-network passive RL min/median: {class_summary['class_network_passive_return_loss_min_db']:.2f} / {class_summary['class_network_passive_return_loss_median_db']:.2f} dB",
        f"- K=1 all-active 10 dB pass rate before projection: {class_summary['k1_all_active_10db_case_pass_rate']:.4f}",
        "",
        "| Class | Ports | Series X (ohm) | Shunt B (S) | Scalar RL min (dB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in class_rows:
        lines.append(
            f"| {row['class_name']} | {row['port_count']} | {float(row['series_reactance_ohm']):.4f} | "
            f"{float(row['shunt_susceptance_siemens']):.7f} | {float(row['scalar_rl_min_db']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Strict active-return projection",
            "",
            f"- All-active 10 dB pass: {projection['projected']['all_active_10db_pass_count']} / {projection['case_count']}",
            f"- Total-reflection 10 dB pass rate: {projection['projected']['total_10db_pass_rate']:.4f}",
            f"- Combined engineering gate pass: {projection['projected']['engineering_10db_gate_pass_count']} / {projection['case_count']}",
            f"- Mean total return loss: {projection['projected']['total_rl_mean_db']:.2f} dB",
            f"- Maximum target-response error: {projection['projected']['target_response_error_max']:.3e}",
            "",
            "| K | Large-scan cases | Worst min (dB) | Worst mean (dB) | Total mean (dB) | Engineering pass |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k_value in (1, 2, 4, 6):
        row = large_scan[k_value]
        lines.append(
            f"| {k_value} | {row['case_count']} | {float(row['worst_active_rl_min_db']):.2f} | "
            f"{float(row['worst_active_rl_mean_db']):.2f} | {float(row['total_rl_mean_db']):.2f} | "
            f"{float(row['engineering_10db_gate_pass_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate decision",
            "",
            f"- Decision: **{label_gate['decision']}**",
            f"- Eligible cases: {label_gate['eligible_case_count']} / {label_gate['case_count']}",
            "- No new HFSS full-wave labels were generated.",
            "- Next physical step: port-class-specific or multiport decoupling/matching optimization, followed by the same dual 10 dB gate.",
        ]
    )
    (BASE / "active_matching_stage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stage_status": summary["stage_status"], "report": str(BASE / "active_matching_stage_summary.md")}, indent=2))


if __name__ == "__main__":
    main()
