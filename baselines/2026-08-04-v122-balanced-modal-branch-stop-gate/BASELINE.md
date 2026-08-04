# v1.22 Balanced Launch and Local Modal Branch Stop-Gate Baseline

## Evidence Level

This checkpoint contains physical HFSS network-only S2 and S8 evidence. The
launch is a ground-referenced symmetric differential microstrip pair with
explicit 50-ohm PRE/POST reference planes. The 2x2 model adds one symmetric
floating-copper correction branch for each local x-neighbor pair. There are no
ideal lumped components, nonlocal links, or additional correction stages.

The trusted antenna S4 termination is a numerical screening cascade. This is
not an integrated feed-antenna, EEP, array, or training-label result.

## Reproducible Result

The finite-conductivity sheet run06 is the selected launch. Its three-frequency
1x1 gate passes.

| 1x1 metric | Worst value | Gate | Decision |
|---|---:|---:|---|
| Final Delta S | 0.00323 | <= 0.05 | Passed |
| Input / output RL | 24.99 / 24.98 dB | >= 15 dB | Passed |
| Insertion efficiency | 95.27% | >= 95% | Passed |
| Transducer efficiency | 94.99% | >= 92% | Passed |
| Reciprocity error | 1.35e-6 | <= 1e-4 | Passed |
| Passivity sigma | 0.9786 | <= 1.001 | Passed |

The first 2x2 discrete sweep completed adaptive meshing but was stopped when
concurrent frequency workers reduced free memory to 1.87 GiB. No S8 was
exported from that attempt. The unchanged geometry was then solved by three
strictly serial exact LastAdaptive cases. All completed normally; minimum free
memory remained 4.76 GiB.

| 2x2 metric | Worst value | Gate | Decision |
|---|---:|---:|---|
| Final Delta S | 0.02405 | <= 0.05 | Passed |
| Reciprocity error | 1.37e-5 | <= 1e-4 | Passed |
| Passivity sigma | 0.9792 | <= 1.001 | Passed |
| Matched-load passive RL | 20.74 dB | >= 12 dB | Passed |
| Matched-load network efficiency | 95.27% | >= 95% | Passed |
| Frozen-source active RL | 5.74 dB | >= 11 dB | Failed |
| Frozen-source total RL | 10.05 dB | >= 11 dB | Failed |
| Actual-load insertion efficiency | 93.97% | >= 95% | Failed |
| Actual-load transducer efficiency | 84.69% | >= 90% | Failed |

The worst loading occurs at 10.04 GHz. Across 9.96, 10.00, and 10.04 GHz,
active RL decreases from 6.54 to 6.23 to 5.74 dB. Good passive matching with
poor active matching shows that the single floating branch does not correct
the non-diagonal antenna/load interaction under the 285 frozen excitations.

## Gate State

The physical S8 gate fails. Independent repeat, integrated 2x2, 4x4/16x16,
training labels, and critic retraining remain prohibited. The next minimum
hardware experiment must replace the correction mechanism with a genuinely
load-aware local modal transformer or change the antenna/feed transition. It
must not add another cascaded branch to this topology.
