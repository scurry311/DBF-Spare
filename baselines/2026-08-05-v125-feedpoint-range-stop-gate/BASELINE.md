# v1.25 Feed-Point Range Stop-Gate Baseline

## Evidence Level

This checkpoint contains four independent three-frequency physical HFSS 1x1
patch/coax models. Only feed inset changes. It contains no 2x2 active-load,
EEP, array, label, or critic evidence.

## Results

| Feed inset | Worst passive RL | Final Delta S | Decision |
|---:|---:|---:|---|
| 1.95 mm | 15.29 dB | 0.02696 | Passed |
| 2.10 mm | 16.77 dB | 0.02760 | Passed |
| 2.50 mm | 12.60 dB | 0.02856 | Failed |
| 2.65 mm | 10.69 dB | 0.02496 | Failed |

All cases have complete S1 data, no topology warnings, and minimum observed
free memory above 11.1 GiB.

## Gate State

The preregistered 0.20 and 0.35 mm periodic feed ranges both fail because
their high-inset endpoints do not preserve 15 dB passive RL. Periodic 2x2
patterns are prohibited. The best passing feed is 2.10 mm and may enter one
new independently preregistered uniform-feed 2x2 S4 smoke.
