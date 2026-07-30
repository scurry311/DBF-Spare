# v1.15 Grounded X-Modal Feed Network

This baseline separates four source-facing PRE reference planes from four
antenna-facing POST reference planes. Network insertion efficiency is computed
as antenna-plane accepted power divided by PRE-plane accepted power; transducer
efficiency additionally includes input mismatch.

## Circuit Synthesis

The selected finite-Q grounded-lowpass modal circuit reaches a minimum passive
RL of 15.663 dB, representative active RL of
12.190 dB, total RL of
13.580 dB, and actual-load insertion efficiency
of 97.57%. Its
reference-plane cascade error is 2.727e-16.

## Physical S8 Evidence

| Stage | Passive RL | Active RL | Total RL | Insertion efficiency | Delta S |
|---|---:|---:|---:|---:|---:|
| Initial fixture | 6.744 dB | 4.590 dB | 5.861 dB | 95.71% | 0.00079 |
| Physical-aware near gate | 13.867 dB | 10.932 dB | 12.024 dB | 97.64% | 0.00139 |
| Reserve-target confirmation | 12.055 dB | 9.359 dB | 10.775 dB | 98.22% | 0.00227 |

The best physical result misses the unchanged 11 dB active-RL gate by
0.068 dB. The subsequent reserve-target
geometry does not confirm the surrogate prediction. Both physical networks are
reciprocal, passive, converged, and above 95% insertion efficiency, so the
remaining failure is joint active matching rather than dissipative loss.

## Decision

The circuit concept is feasible, but this centralized lumped-layout realization
is not physically qualified. No independent repeat, integrated antenna-network
full-wave model, 4x4/16x16 expansion, training labels, or critic retraining is
authorized. Further work must use parameterized full-wave S8 optimization or a
distributed even/odd hybrid; low-order surrogate extrapolation is stopped.
