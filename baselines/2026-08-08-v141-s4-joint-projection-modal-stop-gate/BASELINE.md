# v1.40/v1.41 Frozen-S4 Joint Projection And Modal Stop Gate

## Evidence Level

The matching operator is the trusted physical 10 GHz HFSS S4 from v1.39,
tiled into 64 independent 2x2 blocks. Pattern constraints use the frozen
nominal 256-port EEP associated with the 20 K=2/K=4/K=6 task scenes. The modal
network is a finite-Q circuit upper bound, not an integrated 2x2 or 16x16 HFSS
result.

## Task-Weight Projection

- Frozen scenes: `20` (`K=2/4/6`: `7/6/7`).
- Fixed masks, ratios, targets, pattern gates, and active-RL gates.
- Sequential SOCP stages: `5, 8, 9, 10, 10.5, 11, 11.25 dB`.
- Best exact active-RL floor: `9.9983 dB`.
- Engineering strict oracle: `0/20`.
- 11 dB reserve oracle: `0/20`.

The active-set solver uses at most 128 initially tight leakage rows per task,
but every candidate is accepted or rejected with the complete dense local-5deg
constraint set. Solver failures are recorded as search failures rather than
global infeasibility proofs.

## Single Modal Block

The authorized follow-up is one local block with common/x/y series modal
reactance and common/x/y shunt modal susceptance. It adds no second cascade
stage or nonlocal connection.

The active-only circuit upper bound reaches:

- Worst active RL: `11.4598 dB` over all 20 frozen scenes.
- Worst total RL: `13.1976 dB`.
- Minimum passive RL: `13.4836 dB`.
- Minimum finite-Q network efficiency: `96.68%`.

This requires a maximum command-to-antenna map distortion of `39.19%` and
does not preserve the multi-task pattern. Exact joint passes are `3/7` for
K=2 and `0/6`, `0/7` for K=4 and K=6. The pattern-guarded design similarly
has no K=4/K=6 joint pass.

## Decision

The multi-K circuit coverage gate fails. Physical 2x2 HFSS, larger arrays,
EEP labels, and critic training remain locked. The next authorized algorithmic
test is a joint task-weight projection using the selected corrected S4 and its
mapped EEP operator. Only a nonempty K=2/K=4/K=6 joint set can authorize a
physical CAD smoke.
