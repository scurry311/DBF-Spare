# v1.7 K=6 Multifrequency Rescue

This baseline completes Stage B with frozen E1/E2/E3 operators, calibration
realizations, masks, matching network, and strict engineering thresholds. It
adds one common set of K=6 task weights across all corners; no profile-specific
weights are stored or evaluated.

## Evidence Scope

All results remain a 4x4-HFSS-calibrated 16x16 EEP/S256 proxy. This is valid
candidate-space and frozen-gate evidence, not perturbed 16x16 HFSS evidence and
not a new critic training label set.

## Frozen Policy

- E2 intensity remains 0.20 with seed 20260827 and six frequency/geometry corners.
- PSLL, nearest/local isolation, mainlobe, active-RL, and hardware gates are unchanged.
- One mask and one common `w1...w6` command are used across all E2 corners.
- Ratio 1.0 remains control-only and is absent from the sparse search.
- The first round was preregistered; the quantization-aware boundary round is
  explicitly marked adaptive after observing the first-round failures.
- The executed second-round JSON used a legacy boolean alias for the
  `K=6, ratio<=0.7` requirement. The normalized config restores the v1.6
  numeric key; both executions apply the same unconditional code-level check.

## Oracle Progression

| Stage | Added candidates | E1 new | E2 overall | E2 K=2 | E2 K=4 | E2 K=6 |
|---|---:|---:|---:|---:|---:|---:|
| v1.6 mask rescue | 0 | 86.67% | 82.67% | 100% | 96% | 52% |
| Multifrequency equality/region | 384 | 96.67% | 90.67% | 100% | 96% | 76% |
| Quantization-aware boundary rescue | 192 | 100% | 97.33% | 100% | 96% | 96% |

The final E2 oracle passes 73/75 independent scenes. Minimum feasible sparse
ratios are 0.5 for 51 scenes, 0.6 for 15, 0.7 for 4, and 0.8 for 3.

## Active-RL Margin

The first-round strict candidates have a minimum/median active-RL margin of
0.00065/0.429 dB. Quantization-aware strict candidates improve this to
0.486/1.577 dB after the frozen 7-bit amplitude/phase mapping and compression.

The 12.5-13.5 dB optimization targets are not hard-feasible for every generated
candidate: zero variants satisfy their nominal design target across all E2
corners. They act as optimization targets, while the unchanged engineering gate
remains 10 dB. This baseline therefore claims improved accepted-candidate margin,
not a universal 12.5 dB active-RL guarantee.

## Remaining Failures

- K=6 sample 421023 remains 0.09797 dB below the active-RL gate and is also near
  the nearest-isolation boundary.
- K=4 sample 423018 remains 0.00287 dB below the active-RL gate.
- E3 stress coverage remains 0%; E3 is diagnostic and is not a Stage-B acceptance gate.

## Decision

Stage B passes its preregistered E1/E2 K-stratified oracle criteria. A frozen,
small 16x16 perturbed HFSS smoke is now permitted. Automatic large HFSS batches
remain prohibited, and no engineering critic is promoted by this proxy result.
The next run must freeze the selected masks, `w1...wK`, E2 states, and thresholds
before HFSS results are inspected.
