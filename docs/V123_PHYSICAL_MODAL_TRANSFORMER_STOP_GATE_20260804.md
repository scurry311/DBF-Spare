# v1.23 Physical Modal Transformer Stop Gate

## Scope

This stage froze the selected run06 balanced launch and mapped the v1.23
finite-Q circuit solution into one local physical correction block. It used
eight balanced ground-capacitor branches and four staggered air-bridge
inductor branches. No second correction stage, nonlocal connection, antenna
integration, or training-label generation was allowed.

## Circuit Evidence

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| Three-frequency worst active RL | 12.368 dB | 12 dB | Pass |
| Three-frequency worst total RL | 14.963 dB | 12 dB | Pass |
| Correction-block efficiency | 98.553% | 97% | Pass |
| Requested tolerance pass rate | 99.7% | 95% | Pass |
| Frozen launch plus block efficiency | 93.320% | 95% | Fail |

The correction network had a useful circuit upper bound, but the complete
frozen chain already predicted an efficiency failure. This authorized one
10 GHz network-only HFSS smoke as a diagnostic, not a three-frequency run.

## Physical 10 GHz Result

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| Final Delta S | 0.017436 | <= 0.05 | Pass |
| Reciprocity error | 1.47e-6 | <= 1e-4 | Pass |
| Passivity sigma | 0.9771 | <= 1.001 | Pass |
| Passive RL | 11.831 dB | >= 12 dB | Fail |
| Worst active RL | 3.149 dB | >= 11 dB | Fail |
| Worst total RL | 6.537 dB | >= 11 dB | Fail |
| Matched-load network efficiency | 94.259% | >= 95% | Fail |
| Actual-load insertion efficiency | 92.508% | >= 95% | Fail |
| Actual-load transducer efficiency | 71.974% | >= 90% | Fail |
| Physical-to-target max abs Delta S | 0.5329 | <= 0.10 | Fail |

Peak solver memory was 9.05 GiB. The external memory guard observed a minimum
of 4.24 GiB free, above the 3 GiB abort line.

## Paired Run06 Comparison

The comparison used the same trusted antenna S4 and the same 57 frozen 10 GHz
stimuli. Relative to run06, v1.23 changed:

- worst active RL by -3.078 dB;
- worst total RL by -4.499 dB;
- matched-load efficiency by -1.019 percentage points;
- actual-load insertion efficiency by -1.726 percentage points.

Only 1 of 57 stimuli improved in active RL; 56 degraded. The median active-RL
change was -4.170 dB. The physical S8 moved by 0.2238 from run06 but remained
0.5329 away from the circuit target.

## Decision

Stop the current fixed lumped-component mapping. Do not run 9.96/10.04 GHz,
an independent repeat, integrated 2x2, 4x4/16x16, HFSS labels, or critic
training.

The next useful experiment is a load-pull/modal sensitivity study performed
inside the physical HFSS S8 model, with the antenna S4 embedded in the objective.
It should identify a realizable local geometry Jacobian before another
optimization. If that Jacobian cannot span the required S11/S22 and local
off-diagonal correction directions, stop local-network compensation and move
the antenna feed point or input impedance instead.
