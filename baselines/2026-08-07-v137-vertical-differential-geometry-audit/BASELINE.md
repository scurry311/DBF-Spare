# v1.37 Vertical Differential Geometry Audit Baseline

## Evidence Level

This checkpoint compares five physical 10 GHz 1x1 HFSS CAD representations of
the same true differential input. It separates conductive-body geometry
segments from dielectric/air mesh messages.

## Result

The selected finite-sheet, square-post, partitioned-substrate model reaches
18.044 dB RL with final Delta S 0.002518. Conductive-body small-segment messages
fall from 54 in v1.32 to zero in v1.37. Sixty residual messages remain only on
the substrate and air region, so the strict zero-total-message gate fails.

## Gate State

Matching and conductor-geometry sub-gates pass. The strict total-message gate
fails. Three-frequency rerun, efficiency, 2x2, 4x4, 16x16, EEP, labels, and
critic training remain locked.
