# v1.10 Three-Frequency Active-RL Validation

This baseline first projects one common task-weight command against the trusted
nominal and physical 9.96 GHz EEP/S256 operators, then freezes the resulting
masks and commands before solving and evaluating the symmetric physical
10.04 GHz corner. No critic is trained or used in this stage.

## Joint Projection

Each of the 20 independent scenes retains its original mask. One common
`w1...wK` command is evaluated under nominal/identity, nominal/E2-source,
9.96/identity, and 9.96/E2-source states. PSLL, nearest/local isolation,
mainlobe, active-RL, and hardware thresholds remain unchanged.

| Result | Frozen baseline | Selected projection |
|---|---:|---:|
| Four-state strict pass | 3/20 | 12/20 |
| E2-source pair strict pass | 4/20 | 13/20 |
| Four-state pattern pass | 20/20 | 20/20 |
| Four-state active-RL pass | 3/20 | 12/20 |

The selected K=2/4/6 strict counts are 6/3/3. All eight remaining failures are
active-RL failures. Fixed-mask continuous projection therefore provides a
large improvement, but does not close the low-ratio K=4/K=6 feasibility gap.

## Physical 10.04 GHz Operator

- The trusted pass5 mesh is reused without adaptive refinement.
- The DDM solve uses one task, four cores, and the validated 80% RAM path.
- All 256 complex EEP ports and 16,471 grid directions are complete.
- S reciprocity error is 7.45e-12 and maximum singular value is 0.97906.
- Matched passive minimum RL is 10.616 dB, compared with 15.728 dB at 9.96 GHz.
- Nominal-to-high max abs Delta S is 0.08328; low-to-high matched-S max abs
  difference is 0.20824.

The old same-model `Delta S <= 0.05` direct/DDM consistency gate is not
applicable to an intentional frequency change. The high operator is
numerically valid and passes its structural and passive matching gates.

## Resource Evidence

- Solver elapsed time: 00:38:13.
- Maximum memory per process: 15.9 GB.
- Matrix size and off-core storage: 5,576,630 and 42.5 GB.
- Minimum physical RAM, commit headroom, and D-drive free space were
  0.715 GB, 10.324 GB, and 30.983 GB.
- One 20-second sample fell below 0.75 GB, then recovered to 1.40 GB; the
  two-consecutive-sample hard stop was not triggered.

## Prospective High-Corner Result

The 20 joint-projected commands are frozen before applying the independently
seeded `frequency_high_x0.20` source/calibration state.

| Result | Value |
|---|---:|
| High identity strict pass | 8/20 |
| High E2-source strict pass | 6/20 |
| High E2-source pattern pass | 19/20 |
| High E2-source active-RL pass | 6/20 |
| Common nominal/9.96/10.04 strict pass | 2/20 |

All 14 high-corner strict failures have active-RL as their worst root cause.
The two three-frequency positives are K=6; K=2 and K=4 have no common strict
positive in this frozen set.

## Direct HFSS Check

One predicted K=6 positive and one near-boundary K=6 negative are frozen for
14 combined/task direct saved-field cases. All cases complete. Maximum
no-scale complex NMSE is 5.49e-12 and magnitude RMSE is 2.95e-5 dB. The
significant-task engineering gate agrees for 2/2 candidates: 10.447 dB passes,
while 9.945 dB fails. The all-nonzero diagnostic remains excluded because it
tests near-zero decomposition coefficients as driven ports.

## Decision

The symmetric physical corner is accepted as Level-A development evidence,
but the fixed low-corner projection is not band-robust. Critic retraining and
bulk HFSS remain locked. The next optimization must use nominal, 9.96, and
10.04 GHz physical EEP/S256 operators simultaneously, add at least 1 dB
active-RL design reserve, and permit structured mask swaps for failed scenes.
Only after that three-frequency oracle improves should another physical corner
or new critic labels be generated.
