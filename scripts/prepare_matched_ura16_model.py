"""Create and solve a non-touching URA16 HFSS reference model.

The legacy quick model used a 15 mm dipole length at 15 mm x spacing, so
adjacent collinear PEC arms touched. This utility derives a separate model
from the reviewed builder while preserving the original project and results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "hfss_outputs" / "matched_model_v2_20260714"
DEFAULT_PROJECT = ROOT / "models" / "hfss" / "ura16_quick_10ghz_matched_v2.aedt"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")
DESIGN_NAME = "URA16_Quick_10GHz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run the corrected non-touching URA16 HFSS model.")
    parser.add_argument("--mode", choices=("prepare", "run", "status"), default="prepare")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--project-path", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--dipole-length-mm", type=float, default=14.2)
    parser.add_argument("--spacing-mm", type=float, default=15.0)
    return parser.parse_args()


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old}")
    return text.replace(old, new, 1)


def vbs_path(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.dipole_length_mm < args.spacing_mm:
        raise ValueError("dipole length must be positive and strictly smaller than x spacing")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_builder = ROOT / "scripts" / "build_ura16_quick.vbs"
    builder = source_builder.read_text(encoding="utf-8-sig")
    builder = replace_once(
        builder,
        r'projectPath = rootDir & "\models\hfss\ura16_quick_10ghz.aedt"',
        f'projectPath = "{vbs_path(args.project_path)}"',
    )
    builder = replace_once(builder, "spacingMm = 15.0", f"spacingMm = {args.spacing_mm:.6f}")
    builder = replace_once(builder, "dipoleLengthMm = 15.0", f"dipoleLengthMm = {args.dipole_length_mm:.6f}")
    builder = replace_once(
        builder,
        'Array("NAME:spacing", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "15mm"), _',
        f'Array("NAME:spacing", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{args.spacing_mm:.6f}mm"), _\n'
        f'            Array("NAME:dipole_length", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{args.dipole_length_mm:.6f}mm"), _\n'
        f'            Array("NAME:interelement_tip_gap", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "{args.spacing_mm - args.dipole_length_mm:.6f}mm"), _',
    )

    build_vbs = args.out_dir / "build_ura16_matched_v2.vbs"
    build_vbs.write_text(builder, encoding="ascii")

    solve_vbs = args.out_dir / "solve_ura16_matched_v2.vbs"
    solve_vbs.write_text(
        f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{vbs_path(args.project_path)}"
Set oProject = oDesktop.SetActiveProject("{args.project_path.stem}")
Set oDesign = oProject.SetActiveDesign("{DESIGN_NAME}")
oDesign.Analyze "Setup_10GHz"
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
''',
        encoding="ascii",
    )

    runner = args.out_dir / "run_matched_v2_build_solve.ps1"
    runner.write_text(
        f'''$ErrorActionPreference = "Stop"
$ansys = "{args.ansys_exe.resolve()}"
$build = "{build_vbs.resolve()}"
$solve = "{solve_vbs.resolve()}"
$project = "{args.project_path.resolve()}"
if (-not (Test-Path -LiteralPath $ansys)) {{ throw "ansysedt.exe not found: $ansys" }}
if (Test-Path -LiteralPath $project) {{ throw "Refusing to overwrite existing corrected project: $project" }}
& $ansys -RunScriptAndExit $build *>&1 | Tee-Object -FilePath "{(args.out_dir / 'build.log').resolve()}"
if ($LASTEXITCODE -ne 0) {{ throw "AEDT model build failed: $LASTEXITCODE" }}
if (-not (Test-Path -LiteralPath $project)) {{ throw "Corrected project was not created: $project" }}
& $ansys -ng -RunScriptAndExit $solve *>&1 | Tee-Object -FilePath "{(args.out_dir / 'solve.log').resolve()}"
if ($LASTEXITCODE -ne 0) {{ throw "AEDT model solve failed: $LASTEXITCODE" }}
''',
        encoding="ascii",
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_builder": str(source_builder),
        "project_path": str(args.project_path),
        "design_name": DESIGN_NAME,
        "array_shape": [16, 16],
        "frequency_ghz": 10.0,
        "spacing_mm": float(args.spacing_mm),
        "dipole_length_mm": float(args.dipole_length_mm),
        "feed_gap_mm": 0.5,
        "interelement_tip_gap_mm": float(args.spacing_mm - args.dipole_length_mm),
        "port_reference_ohm": 50.0,
        "source_convention": "driven_modal_incident_power_watt",
        "builder": str(build_vbs),
        "solver": str(solve_vbs),
        "runner": str(runner),
    }
    (args.out_dir / "matched_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def status(args: argparse.Namespace) -> dict[str, Any]:
    results_path = Path(str(args.project_path) + "results")
    build_logs = [args.out_dir / "build.log", args.out_dir / "build_ura16_matched_v2.log"]
    solve_logs = [args.out_dir / "solve.log", args.out_dir / "solve_ura16_matched_v2.log"]
    return {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_exists": args.project_path.exists(),
        "project_size_bytes": args.project_path.stat().st_size if args.project_path.exists() else 0,
        "results_exists": results_path.exists(),
        "results_file_count": sum(1 for item in results_path.rglob("*") if item.is_file()) if results_path.exists() else 0,
        "build_log_exists": any(path.exists() for path in build_logs),
        "solve_log_exists": any(path.exists() for path in solve_logs),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "run":
        if not (args.out_dir / "run_matched_v2_build_solve.ps1").exists():
            prepare(args)
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(args.out_dir / "run_matched_v2_build_solve.ps1")],
            cwd=ROOT,
            check=False,
        )
        result = status(args)
        result["runner_exit_code"] = int(completed.returncode)
    else:
        result = status(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
