# v1.8 Frozen 16x16 Perturbed HFSS Smoke

This baseline executes the small 16x16 HFSS smoke opened by v1.7. Twenty
independent, previously unseen target sets are frozen before HFSS: K=2/4/6 are
represented by 7/6/7 scenes and ratio 0.5/0.6/0.7/0.8 by 7/5/4/4 scenes. The
100 combined/task cases use unchanged masks, task weights, thresholds, and E2
source/calibration states.

## Evidence Scope

The run uses the trusted nominal 16x16 saved-field basis and direct AEDT field
export with frozen source/calibration perturbations. It does not solve new
16x16 frequency, geometry, dielectric, or S-parameter corners. The result is
therefore Level-A evidence for the nominal saved-field and excitation chain,
but not a hidden-physics residual label set.

## Numerical Validation

- All 100/100 HFSS cases and 5/5 chunks completed.
- No-scale complex-field NMSE is at most 5.87e-12.
- No-scale magnitude RMSE is at most 2.99e-5 dB.
- HFSS versus EEP absolute residual maxima are 0.172 dB PSLL, 1.011 dB nearest
  isolation, 0.928 dB local isolation, and 0.075 dB mainlobe gain.
- No threshold or frozen weight was changed after HFSS results were inspected.

## Engineering Gates

| Group | Pass | Worst PSLL | Min nearest iso | Min local-5deg iso | Min combined RL | Min significant-task RL |
|---|---:|---:|---:|---:|---:|---:|
| K=2 | 7/7 | -4.65 dB | 27.18 dB | 22.95 dB | 11.03 dB | 11.02 dB |
| K=4 | 6/6 | -3.10 dB | 25.66 dB | 21.32 dB | 10.88 dB | 10.62 dB |
| K=6 | 7/7 | -0.91 dB | 26.43 dB | 20.95 dB | 10.99 dB | 10.89 dB |

All 20 candidates pass PSLL <= 0 dB, nearest isolation >= 25 dB, local-5deg
isolation >= 20 dB, the frozen mainlobe gate, and the combined plus -20 dB
significant-task active-RL gate. The legacy all-nonzero task diagnostic passes
11/20 because it treats near-zero decomposition coefficients as driven ports;
the disagreement is retained in the snapshots and is not used as the hardware
gate.

## Decision

The frozen small 16x16 source-perturbed HFSS smoke passes with 100% accepted
candidate precision and seven sparse K=6 positives. These results confirm the
weight mapping, saved-field reconstruction, and frozen candidate quality.

Residual-critic retraining on this batch remains blocked: source perturbations
are explicitly represented in the command, while the unobserved physical
frequency/geometry/dielectric/S-parameter corners were not solved. The next
physics step is a small perturbed-operator 16x16 HFSS smoke on a machine state
with at least 14 GB available RAM; the current nominal smoke must not be
relabelled as physical-corner evidence.
