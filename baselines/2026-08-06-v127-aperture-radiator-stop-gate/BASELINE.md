# v1.27 Aperture-Coupled Radiator/Input Stop-Gate Baseline

## Evidence Level

This checkpoint contains 18 converged 10 GHz physical 1x1 HFSS candidates and
their S1 files. It tests joint dual-slot/tongue, coupling-aperture, feed-line,
open-stub, and launch changes. It contains no 2x2 S4, nearest-neighbor Sij,
active-RL, EEP, array, label, or critic evidence.

## Result

All 18 candidates pass the Delta S and topology gates, but none reaches the
10 dB passive-RL prerequisite. The best result is 4.148 dB at
19.61 - j39.11 ohm. A controlled tongue-length comparison changes RL by less
than 0.008 dB, while the stub crosses reactance sign only through a high-
resistance resonance.

## Gate State

The present aperture/tongue/open-stub topology is stopped. Three-frequency
1x1, 2x2, 4x4, 16x16, EEP export, training labels, and critic training remain
locked. The next physical branch must add an independent balanced or
dual-resonant impedance transformation before coupling can be audited.
