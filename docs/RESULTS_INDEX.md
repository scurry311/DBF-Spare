# Result Index

Current baseline: `v0.5.0-implementation-residual`, frozen on 2026-07-24.

## Evidence Levels

| Level | Meaning | Permitted use |
|---|---|---|
| A | Trusted HFSS result with numerical provenance and stated gates | Engineering or pattern evidence within the passed gates |
| B | Numerically valid HFSS result with an incomplete physical gate | Diagnostic evidence only |
| C | Validated interface from a blocked physical baseline | Pipeline validation only |
| D | AF, synthesized S256, or learned proxy | Proposal, screening, and pretraining |
| X | Invalid, incomplete, stale, or empty result | Excluded |

## Trusted 16x16 Baseline

| Result | Value | Decision |
|---|---:|---|
| Field-enabled DDM vs trusted direct max Delta S | 4.76e-11 | Passed |
| S reciprocity max error | 2.51e-11 | Passed |
| S passivity maximum singular value | 0.98009 | Passed |
| Matched passive minimum RL | 13.53 dB | Passed |
| Matched passive 10 dB port rate | 100% | Passed |
| Complex EEP completeness | 256/256 ports | Passed |

The fixed-mesh field solve is the current physical reference. Large raw AEDT,
EEP CSV, and operator files remain under `hfss_outputs/` and are not committed.

## EEP/HFSS Validation

| Result | Value | Decision |
|---|---:|---|
| Candidates / independent scenes | 96 / 76 | New labels only |
| Combined plus task-level HFSS cases | 474/474 complete | Passed |
| Maximum no-scale complex NMSE | 7.93e-12 | Passed |
| Maximum magnitude RMSE | 6.83e-5 dB | Passed |
| Basic gate15 rate | 33.33% | Pattern gate |
| Strict local-20 rate | 21.88% | Pattern gate |
| Mainlobe gate rate | 30.21% | Pattern gate |
| Active-RL gate rate | 0% | Failed |
| Strict engineering gate rate | 0% | Failed |

The no-scale result demonstrates that the EEP operator reproduces HFSS fields
for the same linear electromagnetic basis. It does not establish that the
current sparse weights satisfy active matching or low-power hardware gates.

## Active-RL Joint Optimization

Baseline `v0.3.0-active-rl-joint` audits and projects the same 96 candidates
with task-level weights constrained to satisfy `w = sum(w_k)`.

| Result | Before | After |
|---|---:|---:|
| Combined strict active-RL gate | 0/96 | 85/96 |
| Combined plus significant task active-RL gate | 0/96 | 62/96 |
| Strict local-20 pattern gate | 21/96 original | 27/96 optimized |
| Mainlobe gate | 29/96 original | 30/96 optimized |
| Strict engineering intersection | 0/96 | 11/96 |
| Sparse strict engineering positives | 0 | 8 |
| Sparse multibeam strict positives | 0 | 1 |
| K=6 strict engineering positives | 0 | 0 |

The mean improvement in combined worst active RL is 10.33 dB. The projection
therefore solves most of the matching failure, but sparse multibeam coverage is
not sufficient for new HFSS training labels. Full-EEP local-5-degree leakage
is roughly 5-10 dB worse than the sparse regional constraints in hard scenes.
The next solver gate is a dense local-5-degree EEP operator with explicit
combined-target equalities, followed by another 96-candidate smoke.

## Dense Local EEP and Gated HFSS

Baseline `v0.4.0-dense-local-hfss` replaces the discrete regional offsets with
every EEP grid point inside the local 5-degree neighborhoods and adds explicit
complex combined-pattern equalities at all target directions.

| Result | Value | Decision |
|---|---:|---|
| Dense EEP candidates | 96 | Complete |
| Strict engineering positives | 27/96 | Passed |
| Sparse multibeam positives | 15 | Exceeds required 5 |
| Sparse K=6 positives | 2 | Exceeds required 1 |
| Gated HFSS candidates / cases | 15 / 65 | Complete |
| HFSS strict engineering positives | 15/15 | Passed |
| Maximum no-scale complex NMSE | 6.03e-12 | Passed |
| Maximum magnitude RMSE | 3.07e-5 dB | Passed |

The 15 records are accepted as trusted sparse positive and near-boundary HFSS
labels. Residual-critic training remains held because the run contains no hard
negative and all EEP-to-HFSS residuals remain at numerical-noise scale.

## Implementation-Residual Critic

Baseline `v0.5.0-implementation-residual` conditions the residual on known
implementation errors. Nominal command weights are evaluated by EEP, while
quantized/gain-phase-perturbed or failed-channel weights are returned to HFSS.
Actual-weight EEP remains the numerical mapping audit and is not used as the
critic prediction baseline.

| Result | Value | Decision |
|---|---:|---|
| Boundary candidates / HFSS cases | 21 / 95 | Complete |
| EEP-pass / HFSS-fail hard negatives | 15 | Passed data gate |
| Lower-ratio paired candidates | 6 | K=2/4/6 covered |
| Actual EEP to direct-HFSS maximum NMSE | 5.85e-12 | Mapping passed |
| Nominal to direct-HFSS maximum magnitude RMSE | 3.77 dB | Learnable residual |
| Training candidates / independent scenes | 36 / 15 | Scene-grouped |
| Five-seed gate15/gate20 test AUROC | 1.00 / 1.00 | Preliminary pass |
| Five-seed gate15/gate20 mean ECE | 0.175 / 0.169 | Failed 0.08 gate |
| Scene-test support | 3 scenes | Insufficient |

Five experimental checkpoints were trained. They may be used for conservative
boundary ranking but are not promoted to the engineering critic: calibration
is insufficient, the test set is small, and mainlobe-failure negatives are
missing.

## Expanded Independent-Scene Critic

Baseline `v0.6.0-expanded-residual` adds 45 unique target-direction scenes,
including medium implementation errors, lower-ratio counterfactuals, and
targeted mainlobe failures.

| Result | Value | Decision |
|---|---:|---|
| New candidates / HFSS cases | 105 / 455 | Complete |
| K=2 / K=4 / K=6 new scenes | 21 / 18 / 6 | Covered |
| Reoptimized lower-ratio pairs | 15 | Strict full-wave pass |
| Full-wave mainlobe failures | 37 | Support gate passed |
| Actual EEP to direct-HFSS maximum NMSE | 6.13e-12 | Mapping passed |
| Combined candidates / independent scenes | 141 / 60 | Scene-grouped |
| Train / validation / test scenes | 38 / 11 / 11 | No leakage |
| Gate15 / gate20 AUROC | 0.836 / 0.893 | Gate15 failed 0.88 |
| Gate15 / gate20 ECE | 0.138 / 0.125 | Failed 0.08 |
| Mainlobe / strict AUROC | 1.00 / 1.00 | Passed |

The checkpoint now has the intended mainlobe-specific bias and reliable
candidate-group ranking, but it remains experimental. The next increment must
target independent PSLL and isolation threshold crossings rather than adding
more mainlobe failures.

## Nominal Residual Critic (v0.2)

| Signal | Observed |
|---|---:|
| Delta PSLL standard deviation | 1.19e-5 dB |
| Delta nearest-isolation standard deviation | 3.49e-5 dB |
| Delta local-isolation standard deviation | 1.55e-5 dB |
| Delta mainlobe-gain standard deviation | 7.21e-6 dB |
| Hard negatives | 0 |
| Hard positives | 17 |
| Near-boundary candidates | 87 |

Neural residual-critic training is held. The nominal EEP and HFSS labels are
identical to numerical precision, so a high-capacity model would fit export
noise. The versioned null checkpoint is only a reconstruction sanity baseline.

## Label Policy

- `trusted_pattern_label`: allowed for EEP/HFSS pattern optimization and gates.
- `trusted_engineering_label`: requires active-RL and strict engineering gates;
  EEP/S256 optimization now has 11 candidates, but they are not new HFSS
  training labels and do not cover K=6.
- `proxy_only`: allowed for proposal generation, warm starts, and rejection.
- `legacy_shape_only`: excluded unless explicitly selected for pretraining.
- `invalid`: never included in train/validation/test data.

The split is grouped by independent `sample_index` scenes with no leakage, and
old labels are not mixed automatically. The machine-readable inventories are
`baselines/2026-07-24/artifact_manifest.csv` and the retained historical
`baselines/2026-07-21/artifact_manifest.csv`. The active-RL joint projection
inventory is `baselines/2026-07-24-active-rl-joint/artifact_manifest.csv`.
The dense-local gated HFSS inventory is
`baselines/2026-07-24-dense-local-hfss/artifact_manifest.csv`.
The implementation-residual critic inventory is
`baselines/2026-07-24-implementation-residual/artifact_manifest.csv`.
The expanded independent-scene inventory is
`baselines/2026-07-24-expanded-residual/artifact_manifest.csv`.
