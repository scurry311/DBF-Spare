# v1.49 Periodic CAD Build Gate Checkpoint

## Evidence Scope

This checkpoint records a real AEDT 2023.1 native-CAD build/import audit for
the v1.49 SIW cavity-backed dual-resonant stacked-patch periodic cell. It is
build-only evidence (Evidence Level B/C), not a solved antenna-performance
baseline and not a 16x16 full-wave result.

## Result

- The immutable `run06` build completed with return code 0 in 85.15 s.
- The saved project inventory contains 67 objects, eight boundary name/type
  entries, one feed mode, and two Floquet modes.
- Required Primary/Secondary boundaries, Floquet port, finite-conductivity
  copper, local probe/patch/via meshes, and the closed SIW cavity top are
  present.
- No critical AEDT error or small-segment warning was found in the build log.
- Minimum available RAM during the build was 7.515 GiB.
- The 16-point constraint-aware Latin-hypercube DOE is generated but not
  authorized for HFSS execution.

## Resource Gate

The frozen nominal 9.8-10.2 GHz broadside solve was prepared but not started.
Available RAM was 8.311 GiB, below the preregistered 13 GiB launch gate. No
active RL, passive RL, input impedance, Floquet efficiency, scan gain, or
Delta-S performance claim is available yet.

## Decision

Continue only with the single frozen nominal periodic solve after available
RAM is at least 13 GiB and no AEDT/HFSS process exists. The 45-state periodic
DOE, finite 1x1/2x2/4x4, 16x16, EEP export, label generation, and critic
retraining remain locked.

The next physical gate requires all 45 preregistered
frequency/theta/phi states, traceable HFSS exports, finite metrics, and scan
gain evidence. It also requires a separately versioned AEDT-reopen attestor;
the current stage-A aggregator is diagnostic-only and cannot open the gate. A
broadside three-frequency export cannot open that gate.
