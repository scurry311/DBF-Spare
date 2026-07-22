#!/usr/bin/env python3
"""Run guarded DDM passes from pass5 until convergence or the pass6 stop gate."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"D:\codex_workspace\hfss_ura16_quick_model")
RUN = ROOT / "hfss_outputs" / "grounded_patch_direct_16x16_feedsheet_ddm_recovery_20260721_run02"
PROJECT = RUN / "grounded_patch_16x16"
ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
TEMPLATE = RUN / "continue_ddm_recovery_to_pass04.vbs"
STATUS = RUN / "guarded_pass5_pass6_status.json"
ANALYZER = ROOT / "scripts" / "analyze_ddm_recovery_stage.py"
MAX_PASS = 12


def save_status(**values: object) -> None:
    current = {}
    if STATUS.exists():
        current = json.loads(STATUS.read_text(encoding="utf-8"))
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(current, indent=2), encoding="utf-8")


def make_vbs(pass_number: int) -> Path:
    text = TEMPLATE.read_text(encoding="ascii")
    text = text.replace("pass04_messages.txt", f"pass{pass_number:02d}_messages.txt")
    text = text.replace("ddm_pass04.s256p", f"ddm_pass{pass_number:02d}.s256p")
    text = re.sub(
        r'"MaximumPasses:=",\s*4,',
        f'"MaximumPasses:=", {pass_number},',
        text,
        count=1,
    )
    path = RUN / f"continue_ddm_recovery_to_pass{pass_number:02d}.vbs"
    path.write_text(text, encoding="ascii")
    return path


def run_pass(pass_number: int, reference: Path) -> dict[str, object]:
    vbs = make_vbs(pass_number)
    touchstone = PROJECT / f"grounded_patch_16x16_ddm_pass{pass_number:02d}.s256p"
    if touchstone.exists():
        raise FileExistsError(f"Refusing to overwrite {touchstone}")
    stdout = RUN / f"continue_ddm_recovery_to_pass{pass_number:02d}.stdout.log"
    stderr = RUN / f"continue_ddm_recovery_to_pass{pass_number:02d}.stderr.log"
    save_status(state=f"running_pass{pass_number:02d}", current_pass=pass_number)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(
            [str(ANSYS), "-ng", "-RunScriptAndExit", str(vbs)],
            cwd=ROOT,
            stdout=out,
            stderr=err,
            check=False,
        )
    if result.returncode != 0 or not touchstone.exists():
        raise RuntimeError(
            f"pass{pass_number:02d} failed: returncode={result.returncode}, "
            f"touchstone_exists={touchstone.exists()}"
        )
    profiles = list((PROJECT / "grounded_patch_16x16.aedtresults").rglob("*.profile"))
    profile = max(profiles, key=lambda item: item.stat().st_mtime_ns)
    stage = RUN / "stages" / f"pass{pass_number:02d}"
    stage.mkdir(parents=True, exist_ok=True)
    stage_profile = stage / profile.name
    stage_touchstone = stage / touchstone.name
    shutil.copy2(profile, stage_profile)
    shutil.copy2(touchstone, stage_touchstone)
    messages = RUN / f"continue_ddm_recovery_to_pass{pass_number:02d}_messages.txt"
    if messages.exists():
        shutil.copy2(messages, stage / messages.name)
    metrics_path = stage / "stage_metrics.json"
    port_csv = stage / f"port_stability_vs_pass{pass_number - 1:02d}.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "scripts")
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--stage",
            f"pass{pass_number:02d}",
            "--profile",
            str(stage_profile),
            "--touchstone",
            str(stage_touchstone),
            "--reference",
            str(reference),
            "--output",
            str(metrics_path),
            "--port-csv",
            str(port_csv),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    pass4_metrics = json.loads(
        (RUN / "stages" / "pass04" / "stage_metrics.json").read_text(encoding="utf-8")
    )
    previous_delta = float(pass4_metrics["final_delta_s"])
    reference = RUN / "stages" / "pass04" / "grounded_patch_16x16_ddm_pass04.s256p"
    consecutive = 0
    history = [{"pass": 4, "delta_s": previous_delta}]
    save_status(
        state="starting_pass05",
        policy={
            "mesh_mm": 0.18,
            "percent_refinement": 5.0,
            "ddm_residual": 1.0e-6,
            "pass6_stop_delta_s": 0.12,
            "stop_on_rebound": True,
            "required_consecutive_delta_passes": 2,
            "delta_s_limit": 0.05,
            "targeted_port_refinement": False,
        },
        history=history,
        training_labels_locked=True,
    )
    try:
        for pass_number in range(5, MAX_PASS + 1):
            metrics = run_pass(pass_number, reference)
            delta = float(metrics["final_delta_s"])
            valid = bool(metrics["numerical_smatrix_valid"])
            history.append(
                {
                    "pass": pass_number,
                    "delta_s": delta,
                    "matched_rl_min_db": metrics["matched_passive_rl_min_db"],
                    "numerical_smatrix_valid": valid,
                }
            )
            save_status(history=history, latest_metrics=metrics)
            if not valid:
                save_status(state="stopped_invalid_smatrix", training_labels_locked=True)
                return
            if delta >= previous_delta:
                save_status(
                    state="stopped_delta_s_rebound_check_ports_and_mesh",
                    training_labels_locked=True,
                )
                return
            if pass_number >= 6 and delta > 0.12:
                save_status(
                    state="stopped_pass6_delta_above_0p12_check_port_definition",
                    training_labels_locked=True,
                )
                return
            consecutive = consecutive + 1 if delta <= 0.05 else 0
            if consecutive >= 2:
                rl_pass = float(metrics["matched_passive_rl_min_db"]) >= 10.0
                save_status(
                    state=(
                        "completed_strict_smatrix_gate"
                        if rl_pass
                        else "stopped_converged_but_matched_rl_failed"
                    ),
                    training_labels_locked=not rl_pass,
                )
                return
            previous_delta = delta
            reference = RUN / "stages" / f"pass{pass_number:02d}" / touchstone_name(pass_number)
        save_status(state="stopped_max_pass12_without_confirmation", training_labels_locked=True)
    except Exception as exc:
        save_status(state="stopped_execution_error", error=repr(exc), training_labels_locked=True)
        raise


def touchstone_name(pass_number: int) -> str:
    return f"grounded_patch_16x16_ddm_pass{pass_number:02d}.s256p"


if __name__ == "__main__":
    main()
