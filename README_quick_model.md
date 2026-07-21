# URA16 Quick HFSS Model

This folder contains a one-click AEDT/HFSS build script for the current
beam-multitask `ura16` array geometry.

## Files

- `ura16_quick_10ghz.aedt`: generated HFSS project.
- `build_ura16_quick.vbs`: AEDT VBScript model builder.
- `run_build.ps1`: PowerShell runner for `D:\v231\Win64\ansysedt.exe`.
- `build_ura16_quick.log`: AEDT batch log from the latest build.

## Model

- Solver: HFSS Driven Modal.
- Array: centered 16 x 16 URA, 256 elements.
- Frequency: 10 GHz.
- Wavelength: 30 mm.
- Element spacing: 15 mm, equal to lambda/2.
- Element: simplified PEC dipole.
- Port: one lumped-port sheet per element.
- Boundary: air box with radiation boundary.
- Far field: theta 0 to 90 deg, phi 0 to 360 deg.

## Rerun

```powershell
powershell -ExecutionPolicy Bypass -File D:\codex_workspace\hfss_ura16_quick_model\run_build.ps1
```

The script creates the model only. It does not run the HFSS solve, so the
project opens quickly and can be inspected before simulation.
