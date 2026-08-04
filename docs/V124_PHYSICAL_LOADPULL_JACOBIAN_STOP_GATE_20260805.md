# v1.24 Physical Load-Pull/Modal Jacobian Stop Gate

## Scope

This stage tested whether manufacturable geometry changes inside the frozen
v1.23 local block could reproduce the circuit target. All capacitor,
inductor, ESR, Q, substrate, trace width, differential pitch, and PRE/POST
reference-plane values were frozen. No component-value optimization was run.

Five normalized geometry variables were evaluated with independent physical
HFSS center differences at 10 GHz:

- common ground-load position;
- common air-bridge position;
- N/P bridge half-stagger;
- air-bridge height;
- bridge and via-post width.

The ten cases were solved serially with independent adaptive meshes. All ten
exported complete S8 files. Minimum observed free memory was 3.81 GiB, above
the 3 GiB abort threshold.

## Numerical Audit

All cases converged with final Delta S between 0.00369 and 0.01603 and passed
reciprocity/passivity checks. Only three passed the preregistered 0.005 mm
minimum small-segment length; the other seven are retained as diagnostic
sensitivity evidence and cannot authorize a design confirmation.

All five variables produced pair signals above max |Delta S| = 0.01. The
active-response Jacobian has rank 5 and condition number 15.41. The result is
therefore not a rank-collapse artifact. However, three geometry directions
are materially nonlinear, and the maximum center nonlinearity ratio is 1.371.

## Reachability

| Target view | Bounded residual energy explained | Gate | Status |
|---|---:|---:|---|
| S11/S22 diagonal plus local adjacency | 9.55% | >= 60% | Fail |
| Trusted-S4 terminated operator | 5.04% | >= 60% | Fail |
| Frozen active-response samples | 6.64% | >= 60% | Fail |
| Modal full S8 | 1.14% | Diagnostic | Fail |
| Full S8 | 1.14% | Diagnostic | Fail |

The bounded nonlinear search drove all five normalized coordinates to their
limits. It predicted active RL of 3.52 dB, total RL of 6.64 dB, matched-load
efficiency of 94.50%, and actual-load insertion efficiency of 93.12%. These
remain far below the preregistered gates.

## Decision

The current local bridge/ground-load block is physically unreachable for the
required correction direction. No predicted-geometry confirmation, frequency
extension, integrated model, array expansion, label generation, or critic
training is authorized.

The next hardware branch must directly change the patch feed point or antenna
input impedance. It should first preserve the three-frequency 1x1 passive and
efficiency gate, then compare periodic feed-offset patterns in a physical 2x2
S4 under the same frozen representative excitations.
