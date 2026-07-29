# v1.9 Physical Perturbed-Operator HFSS Smoke

This baseline executes the first real 16x16 physical-operator corner opened by
v1.8. The 9.96 GHz E2 frequency-low corner reuses the trusted pass5 tetrahedral
mesh, disables adaptive refinement, and solves with one DDM task and four cores.
Masks, task weights, source perturbations, matching network, and engineering
thresholds remain frozen.

## Resource Guard

- The solve used an 80% AEDT memory limit with 20-second resource polling.
- Minimum observed free physical RAM was 0.796 GB, above the 0.75 GB hard stop.
- Minimum commit headroom was 10.666 GB.
- Minimum free D-drive space was 31.966 GB, above the 25 GB hard stop.
- The solver profile reports 15.8 GB maximum memory per process, a 5,576,454
  matrix, and 42.5 GB off-core matrix storage.
- The solve completed without a resource abort and released its scratch space.

The pre-solve physical-RAM gate was reduced from 15.5 GB to 15.0 GB before
launch because the trusted nominal field solve had started safely at 14.75 GB.
No numerical, physical, or engineering gate was relaxed.

## Physical Operator

| Result | Value | Decision |
|---|---:|---|
| Frequency | 9.96 GHz | E2 frequency-low corner |
| S-matrix reciprocity error | 2.47e-11 | Passed |
| Maximum singular value | 0.98126 | Passive |
| Matched passive minimum RL | 15.73 dB | Passed 10 dB |
| Complete complex EEP ports | 256/256 | Passed |
| EEP grid directions | 16,471 | Complete |
| Nominal versus physical max abs Delta S | 0.08295 | Physical response |

The old `Delta S <= 0.05` gate compares direct and DDM solves of the same
model. It is not applicable when frequency is intentionally changed. The
9.96 GHz S-matrix is numerically valid; its 0.08295 difference from nominal is
the physical perturbation this smoke was designed to expose.

## Frozen Candidate Evaluation

The same 20 independent v1.8 candidates were evaluated without changing masks,
weights, source states, calibration seed, or thresholds.

| Evaluation | Strict pass | Rate |
|---|---:|---:|
| Physical operator, nominal commands | 5/20 | 25% |
| Physical operator plus frozen source perturbation | 4/20 | 20% |

All 16 failures are active-RL failures. Direction-pattern constraints remain
satisfied across the full set: worst PSLL is -0.858 dB, minimum nearest
isolation is 25.265 dB, and minimum local-5-degree isolation is 20.584 dB. The
minimum active-RL margin falls to 8.623 dB. The four surviving candidates are
all K=6; ratio 0.5 and 0.6 have no surviving candidate in this frozen sample.

## Direct HFSS Check

One predicted positive K=6 candidate and one predicted active-RL-negative K=4
candidate were frozen for direct saved-field HFSS export. All 12 combined/task
cases completed. The maximum no-scale complex NMSE is 5.29e-12 and the maximum
magnitude RMSE is 2.67e-5 dB. The significant-task active-RL decision agrees
for 2/2 candidates: the K=6 case passes at 10.061 dB, while the K=4 case fails
at 9.872 dB.

## Decision

The physical 9.96 GHz operator is accepted as Level-A development evidence,
and labels derived from this operator may be used for robust candidate design.
Critic retraining and a large HFSS batch remain locked: one asymmetric physical
corner and two direct candidates are not enough to identify a general
full-wave residual distribution.

The next step is joint nominal/9.96 GHz active-RL-constrained weight projection,
followed by the symmetric 10.04 GHz physical operator smoke under the same
resource guard. The critic may be reconsidered only after multiple independent
physical corners and threshold-crossing examples are available.
