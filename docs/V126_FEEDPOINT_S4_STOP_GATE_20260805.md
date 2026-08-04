# v1.26 Physical Feed-Point S4 Stop Gate

## Scope

This stage closes the v1.24 physical S8 load-pull/Jacobian audit and the v1.25
feed-point screen. The local bridge geometry remained frozen. The best passing
1x1 feed inset, 2.10 mm, was built as one physical 2x2 patch/coax antenna and
tested at 9.96, 10.00, and 10.04 GHz against the same 285 frozen excitations.

The first discrete three-frequency run was stopped by the memory guard at
0.84 GiB free memory and produced no S4. The independent run02 used three exact
single-frequency adaptive solves in strict serial order. All three completed;
their minimum free-memory values were 6.97, 6.63, and 6.42 GiB.

## Physical Results

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Maximum final Delta S | 0.03862 | <= 0.05 | Pass |
| Reciprocity error | 0 | <= 1e-4 | Pass |
| Passivity sigma | 0.3065 | <= 1.001 | Pass |
| Minimum radiation efficiency | 99.24% | >= 95% | Pass |
| Worst passive RL | 15.10 dB | >= 12 dB | Pass |
| Worst total RL | 11.31 dB | >= 11 dB | Pass |
| Worst active RL | 5.08 dB | >= 11 dB | Fail |
| Active-RL pass count | 113/285 | all frozen excitations | Fail |

The worst excitation is sample 426113, K=4, ratio=0.8, at 9.96 GHz under the
E2 task-source state. K=4 at 9.96 GHz passes only 14/60 active-RL checks.

Relative to the trusted 2.30 mm control, the 2.10 mm feed loses 1.56 dB passive
RL and 0.47 dB active RL, while total RL improves by 0.96 dB. Total reflected
power alone therefore hides a severe per-active-port mismatch.

## Load-Pull Upper Bound

The measured physical off-diagonal S4 entries were frozen and the diagonal
entries were optimized directly against the 285 excitations. This is an
optimistic bound because each frequency may choose a different ideal Sii.

| Allowed diagonal correction | Worst-frequency best active RL | Gate |
|---|---:|---|
| One shared ideal Sii | 8.81 dB | Fail |
| Four independent ideal Sii | 9.49 dB | Fail |

Even the per-port ideal bound remains 1.51 dB below the 11 dB gate. The missing
correction direction is therefore non-diagonal; feed inset or S11/S22 tuning
alone cannot close it.

## Decision

Stop the current bridge and feed-point-only branches. Do not run 4x4, 16x16,
EEP export, label generation, or critic training from this hardware state.

The next minimum physical branch must change the radiator/input structure so
that Sii and nearest-neighbor coupling move together. Suitable variables are
the dual-slot electrical length/separation, tongue current path, and one
balanced or aperture-coupled input transition. Candidate selection must be
based on a physical S4 Jacobian spanning both diagonal and adjacent off-diagonal
targets, not on passive S11 alone.
