# v1.40/v1.41 S4 Projection And Modal Correction Stop Gate

## Question

Can task-level complex weights alone satisfy the 11 dB active-RL reserve on
the trusted physical S4 while preserving the frozen multi-task EEP pattern? If
not, can one local x/y even-odd correction block create the required joint
feasible set without another network stage?

## Method

v1.40 tiles the physical S4 over 64 nonoverlapping 2x2 cells using the fixed
port order `P00, P10, P01, P11`. A sequential convex projection preserves task
target equalities, dense nearest/local leakage bounds, combined mainlobes,
weight norms, and the fixed mask while raising combined and significant-task
active RL. Final metrics always use the complete dense EEP constraints.

v1.41 then synthesizes a single finite-Q modal block. Its six degrees of
freedom are common/x/y series correction and common/x/y shunt correction. The
two preregistered objectives measure an active-RL upper bound and a
pattern-guarded tradeoff.

## Results

| Result | Value | Gate |
|---|---:|---:|
| Frozen scenes | 20 | 20 |
| v1.40 best active RL | 9.9983 dB | >=11 dB |
| v1.40 engineering strict oracle | 0/20 | nonempty |
| v1.40 reserve-11 oracle | 0/20 | nonempty |
| v1.41 active-only worst active RL | 11.4598 dB | >=11 dB |
| v1.41 active-only worst total RL | 13.1976 dB | >=11 dB |
| v1.41 active-only passive RL | 13.4836 dB | >=12 dB |
| v1.41 minimum efficiency | 96.68% | >=95% |
| v1.41 map distortion | 39.19% | <=8% |
| Active-only joint pass, K=2/4/6 | 3/0/0 | >=1 each |
| Pattern-guarded joint pass, K=2/4/6 | 2/0/0 | >=1 each |

## Interpretation

The hardware S4 is not hopeless: the six-parameter modal circuit has enough
freedom to satisfy active and passive matching with finite-Q efficiency. The
failure is now a joint operator problem. The correction that fixes matching
substantially rotates/scales antenna-plane task weights, and the frozen K=4/K=6
patterns no longer meet isolation/mainlobe gates.

No physical HFSS build is authorized from a K=2-only feasible subset. The next
test must jointly optimize task weights on the corrected external S4 and the
mapped EEP, with the same masks and thresholds. If K=4/K=6 remain empty, stop
this single-block topology and alter the radiator/feed input impedance rather
than adding another correction stage.
