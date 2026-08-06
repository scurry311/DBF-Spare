# v1.37 Vertical Differential Geometry Audit

## Purpose

Audit whether the strong v1.33 matching result depends on sub-mesh CAD
segments. The engineering thresholds are unchanged. The strict geometry gate
requires zero small-segment messages before rerunning three frequencies or
opening 2x2.

## Controlled Changes

| Stage | Planar conductor | Vertical feed | Substrate hole |
|---|---|---|---|
| v1.32 | 0.035 mm volume copper | round via | Boolean subtract |
| v1.34 | volume copper, larger branch overlap | 16-sided via | Boolean subtract |
| v1.35 | volume copper | square post | Boolean subtract |
| v1.36 | finite-conductivity sheet | square post | Boolean subtract |
| v1.37 | finite-conductivity sheet | square post | partitioned boxes |

The radiator lengths, differential pitch, nominal post radius, bottom pad,
substrate, port, frequency, and engineering gates remain unchanged.

## Results

| Case | RL | Delta S | Total messages | Conductor messages |
|---|---:|---:|---:|---:|
| v1.32 | 19.834 dB | 0.002095 | 98 | 54 |
| v1.34 | 19.886 dB | 0.002223 | 90 | 46 |
| v1.35 | 18.257 dB | 0.002392 | 90 | 56 |
| v1.36 | 18.050 dB | 0.002579 | 60 | 0 |
| v1.37 | 18.044 dB | 0.002518 | 60 | 0 |

Finite-conductivity sheets eliminate every conductive-body small segment.
Replacing the substrate Boolean hole with axis-aligned partitioned boxes
changes RL by only 0.006 dB and does not change the remaining message count.
The residual entries occur only on `Substrate` and `AirRegion`, with segment
lengths from 0.0194 to 0.0223 mm in v1.37.

## Decision

The true differential matching mechanism is repeatable across materially
different CAD representations, and the conductor geometry is clean. However,
the preregistered zero-total-message gate still fails. The finite-sheet v1.37
geometry is the preferred repair candidate, but it is not promoted to a final
engineering baseline. Three-frequency rerun, efficiency export, 2x2, arrays,
EEP, labels, and critic training remain locked.

The next action is a mesher-level audit of why dielectric/air bodies report
small segments despite axis-aligned partitioning. The gate may only be revised
if direct/DDM or repeated-mesh evidence demonstrates that these messages are
benign; it must not be relaxed merely because RL is favorable.
