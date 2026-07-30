# v1.20 Joint Feed/Fanout Sparse-Graph Baseline

The v1.19 full correction is projected onto the physical port chain
`0-2-3-1`. Route length, per-port loading, and only the three adjacent graph
edges are jointly synthesized against the same 285 frozen representative
excitations. No additional decoupling stage is introduced.

## Sparse-Graph Circuit Gate

| Metric | Result | Gate |
|---|---:|---:|
| Active RL | 13.205 dB | >= 11 dB |
| Total RL | 13.670 dB | >= 11 dB |
| Passive RL | 16.308 dB | >= 10 dB |
| Corrected versus target max abs Delta S | 0.03271 | <= 0.05 |
| 1000-sample tolerance joint pass | 98.5% | >= 90% |

This result is a finite-Q route/circuit surrogate, not physical HFSS. It
proves that the required correction can be represented on the adjacent graph,
but it does not prove that the selected distributed geometry realizes it.

## Physical S8 Front Gates

| Candidate | Solver Delta S | Peak memory | Efficiency | Active RL | Total RL | Physical-target Delta S | Corrected Delta S |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nonuniform trace width | 0.00279 | 9.51 GiB | 95.62% | 4.681 dB | 8.034 dB | 0.464 | 0.434 |
| Uniform near-50-ohm width | 0.00240 | 14.92 GiB | 93.01% | 3.252 dB | 7.598 dB | 0.478 | 0.415 |

Both HFSS S8 solutions are converged, reciprocal, and passive. The second
candidate changes only trace width and shunt reference-plane placement. It
does not recover matching and reduces network efficiency below 95%.

Across the three frequencies, at least 98.02% of the first-run residual and
96.10% of the second-run residual energy lies on the existing diagonal and
adjacent graph entries. The topology therefore captures the location of the
error, while the circuit-to-geometry parameter map is not predictive enough.

## Decision

The physical front gate fails. Independent repeat, integrated 2x2, 4x4,
16x16, training labels, and critic training remain locked. Additional
decoupling stages and blind width/gap sweeps are stopped. The next hardware
experiment must parameterize the radiator feed point/launch and the existing
single-stage POST transition in one HFSS sensitivity model, then optimize
those physical variables directly against S8 and active-RL margins.
