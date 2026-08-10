# v1.47 Literature 10 GHz Balun-Dipole Stop Gate

## Evidence Level

This checkpoint contains one complete physical 8-12 GHz HFSS 1x1 frequency
sweep with discrete 9.96/10.00/10.04 GHz field solves. It contains no
independent direct/DDM acceptance evidence and no authorized 2x2 S4, array,
EEP, dataset, label, or critic evidence.

## Result

The memory-safe direct solve converges to final Delta S `0.01105` with 138,360
tetrahedra, 7.88 GiB peak solver memory, and zero CAD topology warnings. The
reconstructed element fails physically: three-frequency worst passive RL is
`0.04443 dB`, 8-12 GHz RL is `0.03895-0.05021 dB`, contiguous 10 dB bandwidth
is zero, and radiation efficiency is `2.02%`.

## Evidence Boundary

Published dimensions and the integrated microstrip/slotline topology were
used, but the measured improved unit's complete Gerber/CAD, launch, back-ground
width, and reference-plane details are unavailable in the public article.
This baseline rejects the v1.47 reconstruction only; it does not contradict
the measured literature result.

## Decision

Stop before independent repetition and physical 2x2 active-RL. Keep 4x4,
16x16, S256/EEP, labels, and critic training locked. Resume only from complete
author/vendor geometry or another fully specified measured 10 GHz 50 ohm
element and balun.
