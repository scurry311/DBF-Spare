# v1.19 Full-Matrix Multiport POST Decoupler

The v1.18 integrated S4 is deembedded through the trusted v1.16 feed S8. A
reciprocal four-port pi correction network is synthesized with full
off-diagonal series coupling, input/output bridge terms, finite component Q,
E96 quantization, and the same 285 frozen representative excitations.

## Circuit Upper Bound

| Metric | Result | Gate |
|---|---:|---:|
| Active RL | 12.515 dB | >= 11 dB |
| Total RL | 13.057 dB | >= 11 dB |
| Passive RL | 14.774 dB | >= 10 dB |
| Corrected versus target max abs Delta S | 0.03867 | <= 0.05 |
| Matched-load network efficiency | 98.14% | >= 95% |
| 1000-sample tolerance joint pass | 100.0% | >= 90% |

This is a circuit upper bound, not physical HFSS evidence. Twelve of twenty
equivalent discrete values are outside the declared 10 GHz package range. An
exact Givens implementation requires twelve transform stages and has only
88.64% estimated transform
efficiency, so the compact realization gate fails.

## Physical S8 Tests

| Candidate | Solver Delta S | Peak memory | Efficiency | Max off-diagonal transmission | Modal phase span | Active RL | Corrected Delta S |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5.4 mm one section | 0.00241 | 3.63 GiB | 98.54% | 0.110 | 7.9 deg | 0.346 dB | 0.408 |
| 10.8 mm one section | 0.00166 | 7.37 GiB | 96.82% | 0.117 | 22.7 deg | 5.904 dB | 0.278 |
| 9+9 mm noncommuting sections | 0.00399 | 8.09 GiB | 94.91% | 0.123 | 24.1 deg | -1.115 dB | 0.323 |

All three S8 solutions are converged, reciprocal, and passive. The longer
single section improves active RL but remains far below 11 dB. The two-section
network does not create the required approximately 53 degree modal spread,
and its efficiency drops below 95%. The phase alignment used in this screen
is optimistic because it is fitted at each frequency; failure therefore
cannot be blamed on an overly strict reference-plane convention.

## Decision

No independent repeat, integrated 2x2, array expansion, HFSS labels, or critic
training is authorized. Additional coupled sections are stopped because they
increase size and loss without supplying the required full coupling matrix.
The next hardware revision must reduce the required correction at its source:
jointly redesign the antenna feed and POST fanout while constraining the
network to a realizable sparse adjacency graph.
