# v1.48 Reference Element Import And Calibration Stop Gate

## Evidence

The Marlin source is audited but not solved. Its original AEDT 2025.2
component is incompatible with AEDT 2023.1, a header-only diagnostic copy is
not an engineering model, and the complete source project is encrypted.

The MyriadRF branch contains one valid PCB-derived, memory-guarded HFSS direct
S11 sweep. It is calibration evidence only, not a Marlin result, array result,
full radiation validation, or training label.

## Result

The MyriadRF solve converges at final Delta S `0.0101665`. Passive RL at
9.96/10.00/10.04 GHz is `19.504/20.474/21.576 dB`, and the contiguous 10 dB
band containing 10 GHz is `3.5-13.4 GHz`. All small-segment reports are on the
air region and have no conductor or port incidence.

Raw measured S11 Touchstone is absent from the source archive, and radiation
efficiency/gain were not exported from the solved project. Exact measured and
full-radiation calibration gates remain false.

## Decision

The physical gate is closed. Keep 2x2, 4x4, 16x16, S256, EEP, training labels,
and critic retraining locked. Resume with AEDT 2025.2 or an unencrypted Marlin
source design containing all physical definitions; obtain raw MyriadRF VNA
data for exact reference-plane calibration.
