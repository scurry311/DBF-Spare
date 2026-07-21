"""Export the complete 256-port HFSS network solution to Touchstone."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = ROOT / "ura16_quick_10ghz_matched_v2.aedt"
DEFAULT_OUT = ROOT / "hfss_outputs" / "multitask_dataset" / "full_s256p_matched_v2_20260714"
DEFAULT_ANSYS = Path(r"D:\v231\Win64\ansysedt.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "status"), default="prepare")
    parser.add_argument("--project-path", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ansys-exe", type=Path, default=DEFAULT_ANSYS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def vbs_path(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def write_vbs(path: Path, *, project_path: Path, out_dir: Path) -> None:
    touchstone = out_dir / "ura16_matched_v2_full.s256p"
    port_order = out_dir / "aedt_port_order.csv"
    status_path = out_dir / "export_status.csv"
    text = f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, fso
Dim projectPath, projectName, designName, solutionName, touchstonePath, portOrderPath, statusPath
Dim sources, variations, variation, i, portFile, statusFile

projectPath = "{vbs_path(project_path)}"
projectName = "{project_path.stem}"
designName = "URA16_Quick_10GHz"
solutionName = "Setup_10GHz:LastAdaptive"
touchstonePath = "{vbs_path(touchstone)}"
portOrderPath = "{vbs_path(port_order)}"
statusPath = "{vbs_path(status_path)}"

Set fso = CreateObject("Scripting.FileSystemObject")
Set statusFile = fso.CreateTextFile(statusPath, True)
statusFile.WriteLine "status,solution,variation,touchstone_path,port_count"

On Error Resume Next
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject projectPath
Set oProject = oDesktop.SetActiveProject(projectName)
Set oDesign = oProject.SetActiveDesign(designName)
Set oSolutions = oDesign.GetModule("Solutions")
If Err.Number <> 0 Then
    statusFile.WriteLine "failed_open," & solutionName & ",," & touchstonePath & ",0"
    statusFile.Close
    WScript.Quit 2
End If
On Error GoTo 0

sources = oSolutions.GetAllSources()
Set portFile = fso.CreateTextFile(portOrderPath, True)
portFile.WriteLine "touchstone_index,source_name,port_name"
For i = LBound(sources) To UBound(sources)
    portFile.WriteLine CStr(i + 1) & "," & CStr(sources(i)) & "," & Split(CStr(sources(i)), ":")(0)
Next
portFile.Close

variations = oSolutions.ListVariations(solutionName)
If Not IsArray(variations) Then
    statusFile.WriteLine "failed_no_variation," & solutionName & ",," & touchstonePath & "," & CStr(UBound(sources) + 1)
    statusFile.Close
    oDesktop.CloseProject oProject.GetName()
    oDesktop.QuitApplication
    WScript.Quit 3
End If
If UBound(variations) < LBound(variations) Then
    statusFile.WriteLine "failed_no_variation," & solutionName & ",," & touchstonePath & "," & CStr(UBound(sources) + 1)
    statusFile.Close
    oDesktop.CloseProject oProject.GetName()
    oDesktop.QuitApplication
    WScript.Quit 3
End If
variation = CStr(variations(LBound(variations)))

On Error Resume Next
oSolutions.ExportNetworkData variation, Array(solutionName), 3, touchstonePath, Array("All"), True, 50, "S", -1, 0, 15, True, False, False
If Err.Number <> 0 Then
    statusFile.WriteLine "failed_export," & solutionName & "," & Replace(variation, ",", ";") & "," & touchstonePath & "," & CStr(UBound(sources) + 1)
    statusFile.Close
    oDesktop.CloseProject oProject.GetName()
    oDesktop.QuitApplication
    WScript.Quit 4
End If
On Error GoTo 0

If fso.FileExists(touchstonePath) Then
    If fso.GetFile(touchstonePath).Size > 1000 Then
        statusFile.WriteLine "complete," & solutionName & "," & Replace(variation, ",", ";") & "," & touchstonePath & "," & CStr(UBound(sources) + 1)
    Else
        statusFile.WriteLine "failed_small_file," & solutionName & "," & Replace(variation, ",", ";") & "," & touchstonePath & "," & CStr(UBound(sources) + 1)
    End If
Else
    statusFile.WriteLine "failed_missing_file," & solutionName & "," & Replace(variation, ",", ";") & "," & touchstonePath & "," & CStr(UBound(sources) + 1)
End If
statusFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''
    path.write_text(text, encoding="ascii")


def write_runner(path: Path, *, ansys: Path, vbs: Path, log_path: Path) -> None:
    path.write_text(
        f'''$ErrorActionPreference = "Stop"
$ansys = "{ansys.resolve()}"
$script = "{vbs.resolve()}"
$log = "{log_path.resolve()}"
if (-not (Test-Path -LiteralPath $ansys)) {{ throw "ansysedt.exe not found: $ansys" }}
& $ansys -ng -RunScriptAndExit $script *>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) {{ throw "AEDT full S256P export failed: $LASTEXITCODE" }}
''',
        encoding="ascii",
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if not args.project_path.exists():
        raise FileNotFoundError(args.project_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    touchstone = args.out_dir / "ura16_matched_v2_full.s256p"
    if touchstone.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Touchstone: {touchstone}")
    vbs = args.out_dir / "export_full_s256p.vbs"
    runner = args.out_dir / "run_export_full_s256p.ps1"
    write_vbs(vbs, project_path=args.project_path, out_dir=args.out_dir)
    write_runner(runner, ansys=args.ansys_exe, vbs=vbs, log_path=args.out_dir / "export_full_s256p.log")
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_path": str(args.project_path),
        "solution": "Setup_10GHz:LastAdaptive",
        "expected_port_count": 256,
        "reference_impedance_ohm": 50.0,
        "touchstone": str(touchstone),
        "port_order": str(args.out_dir / "aedt_port_order.csv"),
        "runner": str(runner),
    }
    (args.out_dir / "export_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def status(args: argparse.Namespace) -> dict[str, Any]:
    touchstone = args.out_dir / "ura16_matched_v2_full.s256p"
    port_order = args.out_dir / "aedt_port_order.csv"
    status_path = args.out_dir / "export_status.csv"
    return {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "touchstone_exists": touchstone.exists(),
        "touchstone_size_bytes": touchstone.stat().st_size if touchstone.exists() else 0,
        "port_order_exists": port_order.exists(),
        "status_exists": status_path.exists(),
        "status_tail": status_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[-1] if status_path.exists() else "",
    }


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "run":
        runner = args.out_dir / "run_export_full_s256p.ps1"
        if not runner.exists():
            prepare(args)
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(runner)],
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
