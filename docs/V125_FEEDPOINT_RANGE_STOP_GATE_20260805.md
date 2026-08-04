# v1.25 Feed-Point Range Stop Gate

## Scope

This stage directly modified the trusted patch antenna feed point after the
v1.24 local network was declared physically unreachable. Patch dimensions,
slots, tongue, substrate, probe radius, coax geometry, mesh, frequencies, and
engineering thresholds remained fixed.

Four independent physical 1x1 HFSS models used feed insets of 1.95, 2.10,
2.50, and 2.65 mm from the patch edge. Each model exported the full
9.96/10.00/10.04 GHz S1 and radiation efficiency.

## Results

| Feed inset | Worst passive RL | Minimum efficiency | Gate |
|---:|---:|---:|---|
| 1.95 mm | 15.29 dB | 100.88% | Pass |
| 2.10 mm | 16.77 dB | 100.34% | Pass |
| 2.50 mm | 12.60 dB | 100.75% | Fail |
| 2.65 mm | 10.69 dB | 100.83% | Fail |

All four cases converged with final Delta S below 0.029, no port-topology
warnings, and low memory use. The efficiency values slightly above one are the
existing HFSS radiation-efficiency reporting behavior and are used only for
the unchanged >= 0.95 gate.

## Decision

Neither preregistered symmetric range (1.95/2.65 or 2.10/2.50 mm) has two
passing endpoints. Checkerboard/x-stripe/y-stripe 2x2 patterns are therefore
not built or solved. This avoids selecting a narrower periodic range after
seeing the HFSS result.

The best passing 1x1 feed is 2.10 mm. It may be tested in a new, independently
preregistered uniform-feed 2x2 S4 smoke. That smoke must use the original
three-frequency and 285-excitation active-RL gates; it cannot inherit a pass
from the 1x1 result.
