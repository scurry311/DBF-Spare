# v1.23 Load-Aware Modal Transformer Stop-Gate Baseline

## Evidence Level

This checkpoint contains a three-frequency finite-Q circuit/modal upper bound
and one preregistered 10 GHz network-only HFSS S8. The physical model freezes
the run06 balanced differential launch and adds one output-loaded local modal
block. The trusted antenna S4 termination is post-processing, not integrated
feed-antenna HFSS or EEP evidence.

## Circuit Upper Bound

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Worst active RL | 12.368 dB | >= 12 dB | Passed |
| Worst total RL | 14.963 dB | >= 12 dB | Passed |
| Correction-block efficiency | 98.553% | >= 97% | Passed |
| Requested 1000-trial pass rate | 99.7% | >= 95% | Passed |
| Frozen launch plus block efficiency | 93.320% | >= 95% | Failed |

The complete-chain efficiency warning authorized one diagnostic 10 GHz S8 and
kept all later HFSS stages locked.

## Physical 10 GHz Result

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Final Delta S | 0.017436 | <= 0.05 | Passed |
| Reciprocity error | 1.47e-6 | <= 1e-4 | Passed |
| Passivity sigma | 0.9771 | <= 1.001 | Passed |
| Passive RL | 11.831 dB | >= 12 dB | Failed |
| Active RL | 3.149 dB | >= 11 dB | Failed |
| Total RL | 6.537 dB | >= 11 dB | Failed |
| Matched-load network efficiency | 94.259% | >= 95% | Failed |
| Actual-load insertion efficiency | 92.508% | >= 95% | Failed |
| Actual-load transducer efficiency | 71.974% | >= 90% | Failed |
| Physical-to-target max abs Delta S | 0.5329 | <= 0.10 | Failed |

The solve used at most 9.05 GiB solver memory. The memory guard observed at
least 4.24 GiB free, above its 3 GiB abort threshold.

## Paired Decision

Under the same trusted S4 and 57 frozen 10 GHz stimuli, the physical v1.23
block degraded active RL in 56 cases and improved it in one. Median active-RL
change was -4.170 dB. Relative to run06, worst active RL changed by -3.078 dB
and total RL by -4.499 dB.

The local circuit topology has a mathematical upper bound, but its assumed
modal operator is not reproduced by the physical launch and bridge geometry.
The current component mapping is stopped. Three-frequency HFSS, independent
repeat, integrated 2x2, 4x4/16x16, labels, and critic training are prohibited.
