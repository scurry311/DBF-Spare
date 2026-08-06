# v1.33 Vertical Differential Three-Frequency Baseline

## Evidence Level

This checkpoint contains physical 1x1 HFSS S1 evidence for a groundless,
two-conductor vertical differential via launch feeding a dual-resonant fork.
It contains the preceding direct-gap and stopped planar-CPS comparisons needed
to distinguish radiator tuning from impedance transformation.

## Result

All six v1.32 via/pad candidates pass 10 dB at 10 GHz; three pass 15 dB. The
selected `via_v0_open_pad` candidate reaches 19.83 dB. Frozen independent
single-frequency solves at 9.96, 10.00, and 10.04 GHz all pass, with minimum
RL 18.87 dB and maximum final Delta S 0.002095.

## Gate State

The 1x1 three-frequency passive matching gate passes. A later `.g3derr` audit
found 98 small-segment messages in the selected volume-copper geometry,
including 54 on conductive bodies. This checkpoint is therefore retained as
matching-feasibility evidence, not as the final geometry-trusted baseline.
The completed v1.37 geometry audit keeps radiation efficiency, 2x2, 4x4,
16x16, EEP, training labels, and critic training locked because its strict
zero-total small-segment gate did not pass.
