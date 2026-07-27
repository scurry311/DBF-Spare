# Frozen 16x16 Perturbed HFSS Smoke

- Frozen candidates: 20 independent scenes, 100 complete HFSS cases.
- No-scale reconstruction: NMSE max 5.867e-12; magnitude RMSE max 2.988e-05 dB.
- Strict pattern gate: 20/20.
- Combined plus significant-task active-RL gate: 20/20.
- Final engineering smoke gate: 20/20 (100.0%).
- K=6 positives: 7; all sparse multibeam positives: 20.
- Legacy all-nonzero task diagnostic: 11/20; disagreements: 9.

## Decision

The frozen small HFSS smoke passes.
It validates the trusted nominal 16x16 saved-field basis under frozen source and calibration perturbations.
It does not contain perturbed 16x16 frequency, geometry, dielectric, or S-parameter operators, so it is not a new hidden-physics critic label set.
