"""Plan and materialize a 1000-variant HFSS full-wave candidate dataset.

Round 6 is intentionally full-wave-data centric: choose difficult K=6 scenes
from the original 16x16 HFSS dataset, generate several mask/weight teacher
variants for each scene, then split the variants into small HFSS export batches.

The default plan creates 250 scenes x 4 candidate strategies = 1000 variants.
For K=6, every variant becomes 7 HFSS exports: one combined pattern and six
single-task patterns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_PLAN_ROOT = DEFAULT_DATASET_DIR / "hfss_candidate_datasets"
DEFAULT_TEACHER_ROOT = DEFAULT_DATASET_DIR / "optimized_teachers"
DEFAULT_HFSS_OUT_ROOT = DEFAULT_DATASET_DIR / "hfss_fullwave_validations"
DEFAULT_BASE_TEACHER_DIR = DEFAULT_TEACHER_ROOT / "greedy_psll_v2_canonical"
NUM_ELEMENTS = 256
K_TARGET = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "materialize"), default="plan")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--plan-name", default="round6_candidate_dataset_1000_20260701")
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--base-teacher-dir", type=Path, default=DEFAULT_BASE_TEACHER_DIR)
    parser.add_argument("--hfss-out-root", type=Path, default=DEFAULT_HFSS_OUT_ROOT)
    parser.add_argument("--ratios", default="0.6,0.7,0.8")
    parser.add_argument("--scene-count", type=int, default=250)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--strategy-set",
        choices=("round6", "advanced"),
        default="round6",
        help=(
            "Candidate strategy family. 'advanced' uses structured masks, "
            "regional LCMV/ZF nulls, and projected PSLL secondary refinement."
        ),
    )
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-hfss-prepare",
        action="store_true",
        help="Only create chunk teacher directories; do not call hfss_task_fullwave_validate.py --mode prepare.",
    )
    parser.add_argument(
        "--only-existing-teachers",
        action="store_true",
        help="In materialize mode, prepare only strategy/chunk teacher dirs that already exist.",
    )
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def ps_quote(path_or_text: Path | str) -> str:
    text = str(path_or_text)
    return "'" + text.replace("'", "''") + "'"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def target_unit_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = math.radians(float(theta_deg))
    phi = math.radians(float(phi_deg))
    return np.asarray(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        dtype=np.float64,
    )


def min_target_separation_deg(targets: np.ndarray, valid: np.ndarray) -> float:
    points = [target_unit_vector(theta, phi) for theta, phi in targets[np.asarray(valid, dtype=bool)]]
    if len(points) < 2:
        return 180.0
    best = 180.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            cosang = float(np.clip(np.dot(points[i], points[j]), -1.0, 1.0))
            best = min(best, math.degrees(math.acos(cosang)))
    return best


def scene_rows(arrays: dict[str, np.ndarray], ratios: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ratio_set = {round(float(ratio), 6) for ratio in ratios}
    for sample_index in range(int(arrays["sample_ids"].shape[0])):
        k_value = int(arrays["k_values"][sample_index])
        ratio = float(arrays["active_ratios_requested"][sample_index])
        if k_value != K_TARGET or round(ratio, 6) not in ratio_set:
            continue
        valid = arrays["task_valid"][sample_index].astype(bool)
        targets = arrays["targets_deg"][sample_index]
        valid_targets = targets[valid]
        max_theta = float(np.max(valid_targets[:, 0])) if valid_targets.size else 0.0
        mean_theta = float(np.mean(valid_targets[:, 0])) if valid_targets.size else 0.0
        min_sep = float(min_target_separation_deg(targets, valid))
        # High max scan angle and tight target spacing are the two failure drivers
        # seen in the HFSS-in-loop rounds. The hinge makes very small spacing stand
        # out without discarding merely large-scan scenes.
        risk_score = max_theta + 2.0 * max(0.0, 25.0 - min_sep) + 0.25 * mean_theta
        rows.append(
            {
                "sample_index": sample_index,
                "sample_id": str(arrays["sample_ids"][sample_index]),
                "k": k_value,
                "active_ratio": ratio,
                "num_active": int(arrays["num_active"][sample_index]),
                "max_theta_deg": max_theta,
                "mean_theta_deg": mean_theta,
                "min_target_sep_deg": min_sep,
                "risk_score": risk_score,
                "targets_json": json.dumps(valid_targets.tolist(), separators=(",", ":")),
            }
        )
    return rows


def select_scenes(rows: list[dict[str, Any]], ratios: list[float], scene_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    base = int(scene_count) // len(ratios)
    extra = int(scene_count) % len(ratios)
    for pos, ratio in enumerate(ratios):
        quota = base + (1 if pos < extra else 0)
        ratio_rows = [
            row for row in rows if abs(float(row["active_ratio"]) - float(ratio)) < 1.0e-6
        ]
        ratio_rows.sort(key=lambda row: (-float(row["risk_score"]), int(row["sample_index"])))
        for row in ratio_rows[:quota]:
            selected.append(row)
            seen.add(int(row["sample_index"]))
    if len(selected) < scene_count:
        leftovers = [row for row in rows if int(row["sample_index"]) not in seen]
        leftovers.sort(key=lambda row: (-float(row["risk_score"]), int(row["sample_index"])))
        for row in leftovers[: scene_count - len(selected)]:
            selected.append(row)
            seen.add(int(row["sample_index"]))
    selected.sort(key=lambda row: (-float(row["risk_score"]), float(row["active_ratio"]), int(row["sample_index"])))
    return selected[:scene_count]


def strategy_specs(plan_name: str, strategy_set: str = "round6") -> list[dict[str, Any]]:
    prefix = plan_name.replace("candidate_dataset_", "")
    if strategy_set == "advanced":
        return [
            {
                "name": "a1_structlocal20_proj",
                "run_name": f"{prefix}_a1_structlocal20_proj",
                "description": (
                    "Structured masks, all-target +/-2/+/-5 deg local null broadening, "
                    "25/20 dB isolation gate, then projected PSLL refinement."
                ),
                "seed": 20260721,
                "args": [
                    "--structured-mask-mode",
                    "advanced",
                    "--target-isolation-db",
                    "25",
                    "--target-local-isolation-db",
                    "20",
                    "--mainlobe-drop-limit-db",
                    "0.5",
                    "--random-candidates",
                    "3",
                    "--local-swap-candidates",
                    "6",
                    "--local-swap-rounds",
                    "2",
                    "--max-random-swaps",
                    "5",
                    "--diagonal-loading",
                    "0.005",
                    "--local-null-broadening",
                    "all",
                    "--local-null-offsets-deg",
                    "2,5",
                    "--local-null-diagonal-loading",
                    "0.05",
                    "--psll-refine-mode",
                    "projected",
                    "--psll-refine-steps",
                    "8",
                    "--psll-refine-topk",
                    "16",
                    "--psll-refine-step-size",
                    "0.10",
                ],
            },
            {
                "name": "a2_strictlocal25_proj",
                "run_name": f"{prefix}_a2_strictlocal25_proj",
                "description": (
                    "Strict local isolation arm for engineering pass rate: "
                    "28 dB nearest, 22 dB local, denser +/-1/+/-2/+/-5 deg null region."
                ),
                "seed": 20260722,
                "args": [
                    "--structured-mask-mode",
                    "advanced",
                    "--target-isolation-db",
                    "28",
                    "--target-local-isolation-db",
                    "22",
                    "--mainlobe-drop-limit-db",
                    "0.5",
                    "--random-candidates",
                    "3",
                    "--local-swap-candidates",
                    "8",
                    "--local-swap-rounds",
                    "2",
                    "--max-random-swaps",
                    "6",
                    "--diagonal-loading",
                    "0.008",
                    "--local-null-broadening",
                    "all",
                    "--local-null-offsets-deg",
                    "1,2,5",
                    "--local-null-diagonal-loading",
                    "0.08",
                    "--psll-refine-mode",
                    "projected",
                    "--psll-refine-steps",
                    "6",
                    "--psll-refine-topk",
                    "12",
                    "--psll-refine-step-size",
                    "0.08",
                ],
            },
            {
                "name": "a3_relaxedlocal15_proj",
                "run_name": f"{prefix}_a3_relaxedlocal15_proj",
                "description": (
                    "Near-boundary learner: relaxed local isolation gate keeps hard positives/negatives "
                    "while still using structured masks and projected PSLL refinement."
                ),
                "seed": 20260723,
                "args": [
                    "--structured-mask-mode",
                    "advanced",
                    "--target-isolation-db",
                    "25",
                    "--target-local-isolation-db",
                    "15",
                    "--mainlobe-drop-limit-db",
                    "0.5",
                    "--random-candidates",
                    "4",
                    "--local-swap-candidates",
                    "6",
                    "--local-swap-rounds",
                    "2",
                    "--max-random-swaps",
                    "5",
                    "--diagonal-loading",
                    "0.005",
                    "--local-null-broadening",
                    "all",
                    "--local-null-offsets-deg",
                    "2,5",
                    "--local-null-diagonal-loading",
                    "0.04",
                    "--psll-refine-mode",
                    "projected",
                    "--psll-refine-steps",
                    "8",
                    "--psll-refine-topk",
                    "20",
                    "--psll-refine-step-size",
                    "0.12",
                ],
            },
            {
                "name": "a4_pointzf_control",
                "run_name": f"{prefix}_a4_pointzf_control",
                "description": (
                    "Point-ZF control/hard-negative arm; no local broadening, no PSLL refine. "
                    "Use as contrast, not as the engineering teacher."
                ),
                "seed": 20260724,
                "args": [
                    "--structured-mask-mode",
                    "advanced",
                    "--target-isolation-db",
                    "25",
                    "--target-local-isolation-db",
                    "0",
                    "--mainlobe-drop-limit-db",
                    "0.5",
                    "--random-candidates",
                    "4",
                    "--local-swap-candidates",
                    "4",
                    "--local-swap-rounds",
                    "1",
                    "--max-random-swaps",
                    "5",
                    "--diagonal-loading",
                    "0.003",
                    "--local-null-broadening",
                    "off",
                    "--psll-refine-mode",
                    "off",
                ],
            },
        ]
    return [
        {
            "name": "s1_local20_l003",
            "run_name": f"{prefix}_s1_local20_l003",
            "description": "LCMV/ZF with all-target +/-2/+/-5 deg local null broadening, local isolation 20 dB.",
            "seed": 20260711,
            "args": [
                "--target-isolation-db",
                "25",
                "--target-local-isolation-db",
                "20",
                "--mainlobe-drop-limit-db",
                "0.5",
                "--random-candidates",
                "4",
                "--local-swap-candidates",
                "6",
                "--local-swap-rounds",
                "2",
                "--max-random-swaps",
                "5",
                "--diagonal-loading",
                "0.003",
                "--local-null-broadening",
                "all",
                "--local-null-offsets-deg",
                "2,5",
                "--local-null-diagonal-loading",
                "0.03",
            ],
        },
        {
            "name": "s2_local15_l005",
            "run_name": f"{prefix}_s2_local15_l005",
            "description": "Slightly looser local isolation gate to capture near-boundary positives and negatives.",
            "seed": 20260712,
            "args": [
                "--target-isolation-db",
                "25",
                "--target-local-isolation-db",
                "15",
                "--mainlobe-drop-limit-db",
                "0.5",
                "--random-candidates",
                "4",
                "--local-swap-candidates",
                "6",
                "--local-swap-rounds",
                "2",
                "--max-random-swaps",
                "5",
                "--diagonal-loading",
                "0.005",
                "--local-null-broadening",
                "all",
                "--local-null-offsets-deg",
                "2,5",
                "--local-null-diagonal-loading",
                "0.05",
            ],
        },
        {
            "name": "s3_pointzf_l003",
            "run_name": f"{prefix}_s3_pointzf_l003",
            "description": "Point-target ZF control arm for learning where AF-good point nulls collapse in HFSS.",
            "seed": 20260713,
            "args": [
                "--target-isolation-db",
                "25",
                "--target-local-isolation-db",
                "0",
                "--mainlobe-drop-limit-db",
                "0.5",
                "--random-candidates",
                "6",
                "--local-swap-candidates",
                "6",
                "--local-swap-rounds",
                "2",
                "--max-random-swaps",
                "5",
                "--diagonal-loading",
                "0.003",
                "--local-null-broadening",
                "off",
            ],
        },
        {
            "name": "s4_stronglocal28_l008",
            "run_name": f"{prefix}_s4_stronglocal28_l008",
            "description": "Stronger isolation target and heavier local-null regularization for difficult close-target scenes.",
            "seed": 20260714,
            "args": [
                "--target-isolation-db",
                "28",
                "--target-local-isolation-db",
                "20",
                "--mainlobe-drop-limit-db",
                "0.5",
                "--random-candidates",
                "4",
                "--local-swap-candidates",
                "8",
                "--local-swap-rounds",
                "2",
                "--max-random-swaps",
                "6",
                "--diagonal-loading",
                "0.008",
                "--local-null-broadening",
                "all",
                "--local-null-offsets-deg",
                "2,5",
                "--local-null-diagonal-loading",
                "0.08",
            ],
        },
    ]


def write_scene_outputs(
    *,
    plan_dir: Path,
    selected: list[dict[str, Any]],
    ratios: list[float],
    chunk_size: int,
    dataset_dir: Path,
    base_teacher_dir: Path,
    teacher_root: Path,
    hfss_out_root: Path,
    python_exe: Path,
    plan_name: str,
    strategy_set: str,
) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = plan_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    selected_indices_path = plan_dir / "selected_indices.txt"
    selected_indices_path.write_text(
        "\n".join(str(int(row["sample_index"])) for row in selected) + "\n",
        encoding="utf-8",
    )

    scene_csv = plan_dir / "scene_selection.csv"
    with scene_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            "sample_index",
            "sample_id",
            "k",
            "active_ratio",
            "num_active",
            "max_theta_deg",
            "mean_theta_deg",
            "min_target_sep_deg",
            "risk_score",
            "targets_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            payload = dict(row)
            payload["rank"] = rank
            writer.writerow(payload)

    chunk_paths: list[Path] = []
    for chunk_id, start in enumerate(range(0, len(selected), chunk_size)):
        chunk_rows = selected[start : start + chunk_size]
        chunk_path = chunks_dir / f"chunk_{chunk_id:02d}_indices.txt"
        chunk_path.write_text(
            "\n".join(str(int(row["sample_index"])) for row in chunk_rows) + "\n",
            encoding="utf-8",
        )
        chunk_paths.append(chunk_path)

    strategies = strategy_specs(plan_name, strategy_set=strategy_set)
    plan_payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plan_name": plan_name,
        "strategy_set": strategy_set,
        "dataset_dir": str(dataset_dir),
        "base_teacher_dir": str(base_teacher_dir),
        "teacher_root": str(teacher_root),
        "hfss_out_root": str(hfss_out_root),
        "k": K_TARGET,
        "ratios": ratios,
        "scene_count": len(selected),
        "strategy_count": len(strategies),
        "variant_count": len(selected) * len(strategies),
        "expected_hfss_case_count": len(selected) * len(strategies) * (K_TARGET + 1),
        "chunk_size": chunk_size,
        "chunk_count_per_strategy": len(chunk_paths),
        "selected_indices_file": str(selected_indices_path),
        "scene_selection_csv": str(scene_csv),
        "chunks_dir": str(chunks_dir),
        "strategies": strategies,
    }
    (plan_dir / "strategy_plan.json").write_text(
        json.dumps(plan_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    generate_lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location {ps_quote(ROOT)}",
    ]
    ratios_text = ",".join(f"{ratio:.1f}" for ratio in ratios)
    for strategy in strategies:
        cmd = [
            "&",
            ps_quote(python_exe),
            ps_quote(ROOT / "generate_iso_lcmv_teacher.py"),
            "--dataset-dir",
            ps_quote(dataset_dir),
            "--base-teacher-dir",
            ps_quote(base_teacher_dir),
            "--run-name",
            ps_quote(strategy["run_name"]),
            "--split",
            "all",
            "--k-values",
            "6",
            "--active-ratios",
            ps_quote(ratios_text),
            "--sample-indices-file",
            ps_quote(selected_indices_path),
            "--samples-per-cell",
            "0",
            "--seed",
            str(int(strategy["seed"])),
            *strategy["args"],
            "--overwrite",
        ]
        generate_lines.append("")
        generate_lines.append(f"Write-Host 'Generating {strategy['name']} -> {strategy['run_name']}'")
        generate_lines.append(" ".join(cmd))
    (plan_dir / "run_generate_round6_candidates.ps1").write_text(
        "\n".join(generate_lines) + "\n",
        encoding="utf-8",
    )

    chunked_lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location {ps_quote(ROOT)}",
    ]
    smoke_lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location {ps_quote(ROOT)}",
    ]
    for chunk_path in chunk_paths:
        chunk_id = chunk_path.stem.split("_")[1]
        for strategy in strategies:
            run_name = f"{strategy['run_name']}_chunk{chunk_id}"
            cmd = [
                "&",
                ps_quote(python_exe),
                ps_quote(ROOT / "generate_iso_lcmv_teacher.py"),
                "--dataset-dir",
                ps_quote(dataset_dir),
                "--base-teacher-dir",
                ps_quote(base_teacher_dir),
                "--run-name",
                ps_quote(run_name),
                "--split",
                "all",
                "--k-values",
                "6",
                "--active-ratios",
                ps_quote(ratios_text),
                "--sample-indices-file",
                ps_quote(chunk_path),
                "--samples-per-cell",
                "0",
                "--seed",
                str(int(strategy["seed"]) + int(chunk_id)),
                *strategy["args"],
                "--overwrite",
            ]
            chunked_lines.append("")
            done_path = teacher_root / run_name / "dataset_arrays.npz"
            chunked_lines.append(f"if (Test-Path {ps_quote(done_path)}) {{")
            chunked_lines.append(f"  Write-Host 'Skipping existing {strategy['name']} {chunk_id} -> {run_name}'")
            chunked_lines.append("} else {")
            chunked_lines.append(f"  Write-Host 'Generating {strategy['name']} {chunk_id} -> {run_name}'")
            chunked_lines.append("  " + " ".join(cmd))
            chunked_lines.append("}")
            if chunk_id == "00":
                smoke_lines.append("")
                smoke_lines.append(f"if (Test-Path {ps_quote(done_path)}) {{")
                smoke_lines.append(f"  Write-Host 'Skipping existing smoke {strategy['name']} {chunk_id} -> {run_name}'")
                smoke_lines.append("} else {")
                smoke_lines.append(f"  Write-Host 'Generating smoke {strategy['name']} {chunk_id} -> {run_name}'")
                smoke_lines.append("  " + " ".join(cmd))
                smoke_lines.append("}")
    (plan_dir / "run_generate_round6_candidates_chunked.ps1").write_text(
        "\n".join(chunked_lines) + "\n",
        encoding="utf-8",
    )
    (plan_dir / "run_generate_round6_chunk00_smoke.ps1").write_text(
        "\n".join(smoke_lines) + "\n",
        encoding="utf-8",
    )

    materialize_lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location {ps_quote(ROOT)}",
        "Write-Host 'Materializing chunk teacher directories and HFSS prepare queues'",
        " ".join(
            [
                "&",
                ps_quote(python_exe),
                ps_quote(Path(__file__).resolve()),
                "--mode",
                "materialize",
                "--dataset-dir",
                ps_quote(dataset_dir),
                "--plan-root",
                ps_quote(plan_dir.parent),
                "--plan-name",
                ps_quote(plan_name),
                "--teacher-root",
                ps_quote(teacher_root),
                "--hfss-out-root",
                ps_quote(hfss_out_root),
                "--ratios",
                ps_quote(ratios_text),
                "--scene-count",
                str(len(selected)),
                "--chunk-size",
                str(chunk_size),
                "--strategy-set",
                ps_quote(strategy_set),
                "--python-exe",
                ps_quote(python_exe),
                "--overwrite",
            ]
        ),
    ]
    (plan_dir / "run_materialize_round6_batches.ps1").write_text(
        "\n".join(materialize_lines) + "\n",
        encoding="utf-8",
    )

    readme = f"""# {plan_name}

This plan builds a 1000-variant HFSS full-wave candidate dataset.

- Scenes: {len(selected)} K=6 samples, stratified over ratios {ratios_text}
- Strategy set: {strategy_set}
- Strategies: {len(strategies)}
- Variants: {len(selected) * len(strategies)}
- Expected HFSS cases: {len(selected) * len(strategies) * (K_TARGET + 1)}
- Chunking: {len(chunk_paths)} chunks per strategy, {chunk_size} variants per chunk

Workflow:

1. Preferred: run `run_generate_round6_candidates_chunked.ps1` to create 40 small strategy/chunk teacher directories.
2. Fast smoke test: run `run_generate_round6_chunk00_smoke.ps1` first, then materialize with `--only-existing-teachers`.
3. Alternative: run `run_generate_round6_candidates.ps1` to create four full strategy teacher directories.
4. Run `run_materialize_round6_batches.ps1` to prepare HFSS queues.
3. Run the generated `run_hfss_round6_batches.ps1` after materialization to export and analyze all chunks.
4. Rebuild `stage1_metric_dataset` so the new full-wave positives/negatives enter critic training.
"""
    (plan_dir / "README.md").write_text(readme, encoding="utf-8")


def plan(args: argparse.Namespace) -> None:
    ratios = parse_float_list(args.ratios)
    arrays = load_npz(args.dataset_dir / "dataset_arrays.npz")
    rows = scene_rows(arrays, ratios)
    if len(rows) < int(args.scene_count):
        raise RuntimeError(f"Only found {len(rows)} eligible K=6 scenes, cannot select {args.scene_count}.")
    selected = select_scenes(rows, ratios, int(args.scene_count))
    plan_dir = args.plan_root / args.plan_name
    if plan_dir.exists() and any(plan_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{plan_dir} exists; pass --overwrite to refresh the plan.")
    if plan_dir.exists() and args.overwrite:
        shutil.rmtree(plan_dir)
    write_scene_outputs(
        plan_dir=plan_dir,
        selected=selected,
        ratios=ratios,
        chunk_size=int(args.chunk_size),
        dataset_dir=args.dataset_dir,
        base_teacher_dir=args.base_teacher_dir,
        teacher_root=args.teacher_root,
        hfss_out_root=args.hfss_out_root,
        python_exe=args.python_exe,
        plan_name=args.plan_name,
        strategy_set=str(args.strategy_set),
    )
    print(json.dumps({"plan_dir": str(plan_dir), "selected": len(selected)}, indent=2))


def load_indices(path: Path) -> np.ndarray:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        values.append(int(line.split(",")[0]))
    return np.asarray(values, dtype=np.int64)


def save_aux_with_selected(source_aux: Path, dest_aux: Path, selected_indices: np.ndarray) -> None:
    data = load_npz(source_aux)
    data["selected_indices"] = selected_indices.astype(np.int64)
    np.savez_compressed(dest_aux, **data)


def materialize(args: argparse.Namespace) -> None:
    plan_dir = args.plan_root / args.plan_name
    payload = json.loads((plan_dir / "strategy_plan.json").read_text(encoding="utf-8"))
    ratios_text = ",".join(f"{float(ratio):.1f}" for ratio in payload["ratios"])
    chunks = sorted((plan_dir / "chunks").glob("chunk_*_indices.txt"))
    if not chunks:
        raise RuntimeError(f"No chunk files found in {plan_dir / 'chunks'}")

    batch_rows: list[dict[str, Any]] = []
    hfss_scripts: list[tuple[str, Path, str, Path]] = []
    for strategy in payload["strategies"]:
        for chunk_path in chunks:
            chunk_id = chunk_path.stem.split("_")[1]
            selected_indices = load_indices(chunk_path)
            batch_run_name = f"{strategy['run_name']}_chunk{chunk_id}"
            batch_teacher = args.teacher_root / batch_run_name
            chunk_source_dir = args.teacher_root / batch_run_name
            full_source_dir = args.teacher_root / strategy["run_name"]
            chunk_arrays = chunk_source_dir / "dataset_arrays.npz"
            chunk_aux = chunk_source_dir / "iso_lcmv_teacher_arrays.npz"
            full_arrays = full_source_dir / "dataset_arrays.npz"
            full_aux = full_source_dir / "iso_lcmv_teacher_arrays.npz"
            if chunk_arrays.exists() and chunk_aux.exists():
                source_dir = chunk_source_dir
                batch_teacher = chunk_source_dir
            elif full_arrays.exists() and full_aux.exists():
                source_dir = full_source_dir
                if batch_teacher.exists() and args.overwrite:
                    shutil.rmtree(batch_teacher)
                batch_teacher.mkdir(parents=True, exist_ok=True)
                shutil.copy2(full_arrays, batch_teacher / "dataset_arrays.npz")
                save_aux_with_selected(full_aux, batch_teacher / "iso_lcmv_teacher_arrays.npz", selected_indices)
            else:
                if args.only_existing_teachers:
                    continue
                raise FileNotFoundError(
                    f"Missing teacher outputs for {strategy['run_name']} chunk {chunk_id}. "
                    f"Run {plan_dir / 'run_generate_round6_candidates_chunked.ps1'} "
                    f"or {plan_dir / 'run_generate_round6_candidates.ps1'} first."
                )
            summary = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_teacher_dir": str(source_dir),
                "batch_teacher_dir": str(batch_teacher),
                "strategy_name": strategy["name"],
                "strategy_run_name": strategy["run_name"],
                "chunk_file": str(chunk_path),
                "selected_count": int(selected_indices.size),
                "expected_hfss_case_count": int(selected_indices.size) * (K_TARGET + 1),
            }
            (batch_teacher / "round6_batch_teacher_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            hfss_run_name = f"round6_fullwave_{batch_run_name}"
            hfss_out_dir = args.hfss_out_root / hfss_run_name
            run_ps1 = hfss_out_dir / "run_hfss_task_export.ps1"
            if not args.skip_hfss_prepare:
                cmd = [
                    str(args.python_exe),
                    str(ROOT / "hfss_task_fullwave_validate.py"),
                    "--dataset-dir",
                    str(args.dataset_dir),
                    "--teacher-dir",
                    str(batch_teacher),
                    "--out-root",
                    str(args.hfss_out_root),
                    "--run-name",
                    hfss_run_name,
                    "--split",
                    "all",
                    "--k-values",
                    "6",
                    "--active-ratios",
                    ratios_text,
                    "--samples-per-cell",
                    "0",
                    "--mode",
                    "prepare",
                    "--overwrite",
                ]
                subprocess.run(cmd, check=True)
            hfss_scripts.append((hfss_run_name, run_ps1, strategy["name"], batch_teacher))
            batch_rows.append(
                {
                    "strategy_name": strategy["name"],
                    "strategy_run_name": strategy["run_name"],
                    "chunk": chunk_id,
                    "batch_teacher_dir": str(batch_teacher),
                    "hfss_run_name": hfss_run_name,
                    "hfss_out_dir": str(hfss_out_dir),
                    "selected_count": int(selected_indices.size),
                    "expected_hfss_case_count": int(selected_indices.size) * (K_TARGET + 1),
                    "run_ps1": str(run_ps1),
                }
            )

    manifest_path = plan_dir / "round6_batch_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "strategy_name",
            "strategy_run_name",
            "chunk",
            "batch_teacher_dir",
            "hfss_run_name",
            "hfss_out_dir",
            "selected_count",
            "expected_hfss_case_count",
            "run_ps1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_rows)

    hfss_lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location {ps_quote(ROOT)}",
        f"$PythonExe = {ps_quote(args.python_exe)}",
    ]
    for hfss_run_name, run_ps1, strategy_name, batch_teacher in hfss_scripts:
        hfss_lines.append("")
        hfss_lines.append(f"Write-Host 'HFSS export/analyze: {hfss_run_name} ({strategy_name})'")
        hfss_lines.append(f"& powershell -ExecutionPolicy Bypass -File {ps_quote(run_ps1)}")
        hfss_lines.append(
            " ".join(
                [
                    "&",
                    "$PythonExe",
                    ps_quote(ROOT / "hfss_task_fullwave_validate.py"),
                    "--dataset-dir",
                    ps_quote(args.dataset_dir),
                    "--teacher-dir",
                    ps_quote(batch_teacher),
                    "--out-root",
                    ps_quote(args.hfss_out_root),
                    "--run-name",
                    ps_quote(hfss_run_name),
                    "--split",
                    "all",
                    "--k-values",
                    "6",
                    "--active-ratios",
                    ps_quote(ratios_text),
                    "--samples-per-cell",
                    "0",
                    "--mode",
                    "analyze",
                ]
            )
        )
    (plan_dir / "run_hfss_round6_batches.ps1").write_text(
        "\n".join(hfss_lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "batch_manifest": str(manifest_path),
                "batch_count": len(batch_rows),
                "expected_hfss_case_count": sum(int(row["expected_hfss_case_count"]) for row in batch_rows),
                "run_all_script": str(plan_dir / "run_hfss_round6_batches.ps1"),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.mode == "plan":
        plan(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()
