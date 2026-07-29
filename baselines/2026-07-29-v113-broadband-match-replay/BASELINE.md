# v1.13 Three-Frequency Broadband-Match Replay

This baseline freezes the twenty v1.12 scene-level masks and task commands,
then replaces the common single series inductor with finite-Q S-L-S circuit
networks evaluated on the physical 9.96, 10.00, and 10.04 GHz raw S256/EEP
operators. No mask, command, ratio, E2 state, or threshold is changed.

## Matching And Operator Results

| Result | Old network | Three-class S-L-S |
|---|---:|---:|
| 9.96 GHz minimum passive RL | 15.728 dB | 15.557 dB |
| 10.00 GHz minimum passive RL | 13.535 dB | 13.607 dB |
| 10.04 GHz minimum passive RL | 10.616 dB | 10.748 dB |
| Minimum network efficiency | 98.05% | 98.09% |

The rebuilt external-port operators pass reciprocity and passivity checks. The
maximum no-scale complex EEP map NMSE is 1.92e-13. These are circuit-cascade
operators on physical raw HFSS data, not embedded-layout HFSS validation.

## Frozen Replay

| Result | Old network | Three-class S-L-S | Required |
|---|---:|---:|---:|
| Overall strict | 8/20 | 8/20 | Diagnostic |
| K=2 strict | 4/7 | 4/7 | >= 6/7 |
| K=4 strict | 2/6 | 2/6 | >= 5/6 |
| K=6 strict | 2/7 | 2/7 | Diagnostic |
| 11 dB reserve | 0/20 | 0/20 | >= 1 |

The network slightly improves passive high-corner matching and preserves the
patterns, but it does not enlarge the strict feasible intersection. Active-RL
gains are mixed across frozen commands.

## Decision

The replay gate fails. Candidate optimization, candidate-array HFSS smoke,
bulk HFSS, and critic training remain locked. Further independent one-port
S-L-S tuning is not justified. Any resumed hardware work should first test a
dual-resonant or coupled-feed matching concept on a small physical cell/array
frequency sweep, without treating it as a training-label run.
