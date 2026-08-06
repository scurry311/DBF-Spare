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

The 1x1 three-frequency passive matching gate passes. Radiation efficiency
and an independent repeated solve are authorized next. 2x2, 4x4, 16x16, EEP,
training labels, and critic training remain locked.
