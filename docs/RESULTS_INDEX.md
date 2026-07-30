# Result Index

Current stage baseline: `v1.14.0-small-cell-broadband-feed`, frozen on 2026-07-30.

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

## Gate15 Boundary Critic

Baseline `v0.7.0-gate15-boundary` adds 30 independent full-wave scenes aimed
only at PSLL, nearest-isolation, and local-isolation threshold crossings. Each
scene contains a strict control, a just-inside candidate, and a just-outside
candidate; no mainlobe-failure scene was added.

| Result | Value | Decision |
|---|---:|---|
| New candidates / independent scenes | 90 / 30 | Complete |
| PSLL / nearest / local scenes | 10 / 10 / 10 | Balanced |
| Full-wave cases | 444/444 | Complete |
| Inside pass / isolated outside failure | 30/30 / 30/30 | Passed |
| New mainlobe failures | 0 | Requirement met |
| Combined candidates / scenes | 231 / 90 | Scene-grouped |
| Train / validation / test scenes | 62 / 14 / 14 | No leakage |
| Gate15 / gate20 AUROC | 0.883 / 0.910 | Passed 0.88 |
| Gate15 / gate20 pooled ECE | 0.050 / 0.071 | Passed 0.08 |
| Strict ranking rate | 1.000 | Passed |

The best checkpoint is seed `20260727`, but it is promoted only together with
the pooled scene-grouped regularized-isotonic calibrator. Test labels were not
used to select the calibrator. This is a retrospective stage-one promotion;
prospective HFSS validation on unseen scenes is still required before the
critic may automatically admit candidates to HFSS.

## Prospective Frozen-Critic HFSS Validation

Baseline `v0.8.0-prospective-hfss` freezes the v0.7 checkpoint, pooled
calibrator, feature substitutions, uncertainty factor, and thresholds before
running HFSS on 24 unseen target-direction sets. The prospective labels are
evaluation-only and were not used for retraining or recalibration.

| Result | Value | Decision |
|---|---:|---|
| Unseen scenes / candidates / HFSS cases | 24 / 72 / 354 | Complete |
| Training target-hash overlap | 0 | Passed |
| PSLL / nearest / local scenes | 8 / 8 / 8 | Balanced |
| Inside pass / isolated outside failure | 24/24 / 24/24 | Passed |
| Mainlobe failures | 0 | Requirement met |
| Maximum no-scale complex NMSE | 6.19e-12 | Passed |
| Gate15 AUROC / ECE | 0.795 / 0.041 | AUROC failed |
| Gate20 AUROC / ECE | 0.895 / 0.084 | ECE failed |
| Pattern15 admission precision / recall | 0.853 / 0.604 | Diagnostic |
| Strict admission precision / recall | 1.000 / 0.885 | Diagnostic |

The critic orders every just-inside candidate above its paired just-outside
candidate, but the mean probability margin is only 0.0218 and control-free
gate15 AUROC is 0.590. It has useful scene-relative ranking behavior but does
not provide reliable absolute feasibility probabilities across unseen scenes.
All top-one methods selected the nominal control, so their 24/24 strict pass
rate is not evidence that near-boundary sparse variants generalize.

The pre-registered prospective protocol failed. Automatic HFSS admission stays
disabled, and this prospective set must not be used to tune the frozen model or
calibrator in the same development cycle.

## Physical-Margin Adaptive Critic v0.9

Baseline `v0.9.0-physical-margin-adaptive` removes nominal controls from sparse
ranking and predicts five uncertainty-aware physical-margin residuals.  The
development set contains 60 independent scenes, 1,920 EEP/S256 candidates, and
a 420-candidate scene-grouped subset.

| Result | Value | Decision |
|---|---:|---|
| Development scenes / candidates | 60 / 420 | Leakage-free 36/12/12 split |
| Strict positives / hard negatives | 79 / 70 | Sufficient support |
| gate15 AUROC / ECE | 0.989 / 0.055 | Passed |
| strict AUROC / ECE | 0.935 / 0.073 | Passed |
| No-control top-1 / fixed ratio-0.6 | 66.7% / 41.7% | Ranking improved |
| Held-out HFSS candidates / cases | 84 / 420 | Complete |
| HFSS strict AUROC | 0.918 | Passed |
| HFSS strict ECE before / after val calibration | 0.091 / 0.066 | Calibrated pass |
| Second prospective admitted / strict pass | 5/12 / 5/5 | Conservative pass |
| Prospective K=6 positive | ratio 0.6 | Passed |
| Prospective scene coverage | 41.7% | Final 80% target not met |

The selector may now automatically route only conservative admissions to HFSS.
Fallback candidates remain explicitly infeasible/unconfirmed.  The prospective
admitted ratios are 0.5, 0.6, and 0.7, corresponding to a 44% mean channel
reduction among admitted scenes.  This is not a measured same-EIRP RF power
reduction because no new paired ratio-1 full-wave control was included.

## Targeted K=2/K=6 EEP Oracle Audit v1.0

Baseline `v1.0.0-targeted-eep-oracle-audit` expands the structured mask search
to 24 candidates per ratio and adds eight best-neighbor rescue masks for each
failed scene.  It keeps the v0.9 second prospective directions frozen and
evaluation-only.

| Result | Value | Decision |
|---|---:|---|
| Independent scenes / initial candidates | 48 / 4,608 | Complete |
| K=2 / K=6 scenes | 24 / 24 | Targeted |
| Initial strict candidates / hard negatives | 452 / 202 | Candidate support |
| Additional failed-scene rescue candidates | 832 | Complete |
| Target-hash overlap with excluded sets | 0 | Passed |
| Overall scene oracle | 22/48 (45.8%) | Failed 90% gate |
| K=2 / K=6 scene oracle | 62.5% / 29.2% | K=6 bottleneck |
| Non-large / large-scan scene oracle | 77.3% / 19.2% | Scan bottleneck |

The 24-to-32 mask expansion did not rescue another scene.  Residual critic
retraining and HFSS admission remain locked.  The failure is a coupled
active-RL, mainlobe, and isolation feasibility problem rather than a shortage
of random masks; candidate-level positives must not be reported as scene-level
coverage.

## EEP/S256 Operating Envelope v1.2

Baseline `v1.2.0-eep-operating-envelope` pre-registers and independently
validates a narrower engineering support domain, while retaining the broader
domain as a stress set. No HFSS labels were generated in this stage.

| Result | Value | Decision |
|---|---:|---|
| v1.1 independent scenes / candidates | 40 / 3,840 | Complete |
| v1.1 K=2 / K=6 oracle | 70.0% / 95.0% | K=2 failed |
| Active-RL joint rescue candidates | 672 | Complete |
| Rescued scenes / oracle after rescue | 2 / 87.5% | Still below 90% |
| New independent K=2 scenes / candidates | 20 / 1,920 | Complete |
| New K=2 oracle at scan <=48 deg | 19/20 (95.0%) | Passed |
| Combined supported K=2 / K=6 oracle | 97.1% / 95.0% | Passed |
| Combined supported total | 52/54 (96.3%) | Passed |
| Excluded target-hash overlap | 0 | Passed |

The supported EEP/S256 envelope is K=2 with maximum target scan 48 degrees
and minimum separation 16 degrees, and K=6 with maximum target scan 58 degrees
and minimum separation 13 degrees. K=2 at 50 degrees remains a pressure set:
the joint discrete-mask/continuous-weight rescue improved it only to 80%.
Within the validated envelope, a frozen 15-20-candidate HFSS smoke may proceed;
bulk HFSS and full-wave performance claims remain disabled.

## K=4, Frozen HFSS, and Critic Gate v1.3

Baseline `v1.3.0-k4-hfss-critic-gate` adds K=4 at scan <=48 degrees and
minimum target separation >=16 degrees, then executes the pre-registered
frozen HFSS smoke and one targeted full-wave label batch.

| Result | Value | Decision |
|---|---:|---|
| New K=4 scenes / candidates | 20 / 1,920 | Complete |
| K=4 scene oracle | 20/20 (100%) | Passed 90% |
| Supported K=2 / K=4 / K=6 oracle | 97.1% / 100% / 95.0% | Passed |
| Frozen smoke scenes K=2/4/6 | 7 / 6 / 7 | Pre-registered |
| Frozen HFSS smoke cases | 100/100 complete | Passed |
| Pattern, mainlobe, operating active-RL gate | 20/20 | Passed |
| Targeted labels / independent scenes | 50 / 41 | Complete |
| Targeted HFSS cases | 256/256 complete | Passed |
| Critic strict AUROC / ECE | 0.760 / 0.234 | Failed |
| Critic precision / top-one | 0.900 / 0.467 | Top-one failed |
| Actual-weight EEP/HFSS strict agreement | 100% | Deterministic gate |

The residual critic is not promoted and the final adaptive-ratio prospective
HFSS stage remains locked. The known actual implementation weights are already
evaluated by EEP/S256 with numerical agreement to HFSS; asking a nominal-only
critic to infer those known perturbations reduced reliability. Future critic
labels must contain operator or hardware drift not already represented by the
actual weights and fixed EEP/S256 basis.

## Operator-Drift Residual Critic v1.4

Baseline `v1.4.0-operator-drift-critic` introduces frequency, patch geometry,
dielectric, complex S-matrix, calibration, quantization, soft-failure, and
temperature residuals that are absent from the fixed nominal EEP/S256 operator.

| Result | Value | Decision |
|---|---:|---|
| Valid 4x4 HFSS operator profiles | 7/7 | Calibration passed |
| Non-nominal field NMSE range | 0.00858-0.07381 | Nonzero physical residual |
| Non-nominal maximum S16 drift | 0.1342-0.3927 | Nonzero physical residual |
| Base scenes / proxy candidates | 45 / 4,320 | K=2/4/6 balanced |
| Strict positives / hard negatives | 966 / 1,897 | Training support passed |
| Train/val/test target-hash overlap | 0 | Passed |
| Strict AUROC / AUPRC | 0.9887 / 0.9554 | Proxy pass |
| Strict ECE / precision | 0.0330 / 0.9064 | Proxy pass |
| Active-RL AUROC / ECE | 0.9867 / 0.0423 | Proxy pass |
| Critic top-1 / fixed ratio-0.7 | 30.0% / 24.5% | Improved, below 80% target |
| Candidate oracle | 34.3% | Stress-set ceiling |
| Drift-feature strict AUPRC delta | +0.0201 | Ablation gain |
| Drift-feature strict recall delta | +0.1503 | Ablation gain |
| Drift-feature top-1 delta | +0.0028 | Candidate-space limited |

The five checkpoints are accepted for proxy screening, ablation, and pretraining
only. The 16x16 labels are generated by a 4x4-HFSS-calibrated EEP/S256 mapping,
not by perturbed 16x16 HFSS solves. Engineering promotion and the final adaptive
ratio HFSS stage remain locked until an independent frozen 16x16 operator-drift
smoke passes without threshold or model changes.

## Feature Observability Audit v1.5

Baseline `v1.5.0-feature-observability` separates directly deployable features,
measurement-assisted features, and unavailable simulation truth on the same
v1.4 scene-grouped split.

| Result | Model A | Model B | Model C |
|---|---:|---:|---:|
| Feature tier | Deployable | Measurement-assisted | Oracle truth |
| Strict AUROC | 0.9829 | 0.9899 | 0.9887 |
| Strict ECE | 0.0318 | 0.0311 | 0.0330 |
| Strict precision | 0.9343 | 0.9071 | 0.9064 |
| Strict recall | 0.7672 | 0.8723 | 0.8983 |
| Top-1 strict rate | 29.17% | 30.09% | 30.00% |

Models A and B pass the aggregate pre-registered Stage-A gate, and the target
hash overlap remains zero. Measurement-assisted inputs improve recall by 10.5
percentage points relative to deployable-only inputs. However, strict precision
inside the E2 intensity-0.20 subgroup is only 59.98%/69.83%/71.36% for Models
A/B/C. Stage B robust-candidate optimization may proceed, but E2 automatic
critic admission, engineering promotion, and 16x16 HFSS remain disabled.

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
The dedicated gate15-boundary inventory is
`baselines/2026-07-25-gate15-boundary/artifact_manifest.csv`.
The frozen prospective HFSS inventory is
`baselines/2026-07-25-prospective-hfss/artifact_manifest.csv`.
The v0.9 physical-margin adaptive HFSS inventory is
`baselines/2026-07-26-v09-adaptive-hfss/artifact_manifest.csv`.
The v1.0 targeted K=2/K=6 EEP oracle audit is
`baselines/2026-07-27-v10-targeted-eep-oracle/BASELINE.md`.
The validated v1.2 EEP/S256 operating envelope is
`baselines/2026-07-27-v12-operating-envelope/BASELINE.md`.
The v1.3 K=4, frozen HFSS, and critic gate baseline is
`baselines/2026-07-27-v13-k4-hfss-critic-gate/BASELINE.md`.
The v1.4 operator-drift residual critic baseline is
`baselines/2026-07-27-v14-operator-drift-critic/BASELINE.md`.
The v1.5 feature observability audit baseline is
`baselines/2026-07-27-v15-feature-observability/BASELINE.md`.

## Robust Drift Oracle v1.6

The preregistered common-mask/common-command search uses 75 independent
K=2/4/6 scenes and 9,600 initial candidates. Common-weight projection raises
the E2 oracle from 1.33% to 69.33%; active-RL-guided mask rescue raises it to
82.67%. K=2 and K=4 reach 100% and 96%, while K=6 remains at 52%. E1 coverage
on the 30 new scenes is 86.67%, and E3 stress coverage is 0%.

Stage B therefore remains failed. Critic retraining and perturbed 16x16 HFSS
are still disabled. The authoritative snapshot is
`baselines/2026-07-27-v16-robust-drift-oracle/BASELINE.md`.

## K=6 Multifrequency Rescue v1.7

E2, its calibration seed, all strict thresholds, and the common-command rule
remain frozen. The first round adds 384 K=6 candidates with multifrequency
combined-mainlobe equalities and dense regional inequalities. A declared
adaptive second round adds 192 quantization-aware boundary candidates.

| Stage | E1 new | E2 overall | E2 K=2 | E2 K=4 | E2 K=6 |
|---|---:|---:|---:|---:|---:|
| v1.6 baseline | 86.67% | 82.67% | 100% | 96% | 52% |
| Multifrequency rescue | 96.67% | 90.67% | 100% | 96% | 76% |
| Quantization-aware rescue | 100% | 97.33% | 100% | 96% | 96% |

Stage B passes. The final E2 feasible-ratio distribution is 51/15/4/3 scenes
at ratio 0.5/0.6/0.7/0.8. Two active-RL failures remain, and E3 stress coverage
is 0%. The result is still a 4x4-HFSS-calibrated 16x16 EEP/S256 proxy. It opens
only a frozen small 16x16 perturbed HFSS smoke; it does not create new HFSS
labels or authorize an automatic large batch. The authoritative snapshot is
`baselines/2026-07-27-v17-k6-multifrequency-rescue/BASELINE.md`.

## Frozen 16x16 Perturbed HFSS Smoke v1.8

Twenty independent E2-positive candidates were frozen before HFSS, with
K=2/4/6 represented by 7/6/7 scenes and sparse ratios 0.5/0.6/0.7/0.8 by
7/5/4/4 scenes. All 100 combined/task cases completed. The no-scale complex
NMSE and magnitude RMSE maxima are 5.87e-12 and 2.99e-5 dB.

All 20 candidates pass the strict pattern gate and the combined plus -20 dB
significant-task active-RL gate. Worst full-wave PSLL is -0.91 dB; minimum
nearest/local isolation is 25.66/20.95 dB; minimum combined and significant-task
active RL is 10.88/10.62 dB. The old all-nonzero task diagnostic passes 11/20
and is retained only to expose the near-zero coefficient semantic mismatch.

This smoke reuses the trusted nominal 16x16 saved-field basis and applies frozen
source/calibration perturbations. It does not include new 16x16 physical
frequency, geometry, dielectric, or S-parameter corner solves, so it validates
the execution chain but does not open hidden-physics residual-critic training.
The authoritative snapshot is
`baselines/2026-07-28-v18-frozen-16x16-hfss-smoke/BASELINE.md`.

## Physical Perturbed-Operator HFSS Smoke v1.9

The first real 16x16 physical corner changes the fixed-mesh operator from the
nominal frequency to the preregistered E2 frequency-low point at 9.96 GHz. The
solve is numerically valid and completes all 256 complex EEP exports under a
hard resource monitor.

| Result | Value | Decision |
|---|---:|---|
| Matched passive minimum RL | 15.73 dB | Passed |
| Complete EEP ports / directions | 256 / 16,471 | Passed |
| Nominal versus physical max abs Delta S | 0.08295 | Expected physical drift |
| Frozen operator-only strict pass | 5/20 | 25% |
| Physical plus source strict pass | 4/20 | 20% |
| Pattern-gate failures | 0 | Passed |
| Active-RL failures | 16 | Hardware-margin bottleneck |
| Direct EEP/HFSS gate agreement | 2/2 | Passed |
| Minimum free RAM / disk | 0.796 / 31.966 GB | No resource abort |

The `Delta S <= 0.05` same-model direct/DDM consistency gate does not apply to
an intentional frequency change. The physical operator is accepted for robust
development, but critic retraining and a large HFSS batch remain locked. Joint
nominal/9.96 GHz active-RL projection and a symmetric 10.04 GHz physical corner
must come next. The authoritative snapshot is
`baselines/2026-07-29-v19-perturbed-operator-hfss-smoke/BASELINE.md`.

## Three-Frequency Active-RL Validation v1.10

One common task-weight command per scene is first projected jointly against
the nominal and physical 9.96 GHz operators. Masks, thresholds, and source
states remain fixed. The selected commands are then frozen before the new
physical 10.04 GHz operator is solved and evaluated.

| Result | Value | Decision |
|---|---:|---|
| Nominal/9.96 strict pass before projection | 3/20 | Baseline |
| Nominal/9.96 strict pass after projection | 12/20 | Improved |
| 10.04 GHz high-source strict pass | 6/20 | Insufficient |
| Common three-frequency strict pass | 2/20 | Failed coverage |
| High-corner pattern pass | 19/20 | Pattern mostly robust |
| High-corner active-RL pass | 6/20 | Dominant bottleneck |
| Direct high-corner EEP/HFSS agreement | 2/2 | Passed |
| Direct maximum complex NMSE | 5.49e-12 | Passed |

The physical 10.04 GHz S256/EEP operator is structurally valid, with matched
passive minimum RL 10.616 dB. Its response is materially less tolerant than
the 9.96 GHz operator, whose matched passive minimum RL is 15.728 dB. All 14
prospective strict failures have active-RL as their worst root cause.

The next stage must jointly optimize structured masks and common weights across
nominal, 9.96, and 10.04 GHz with at least 1 dB active-RL reserve. Critic
retraining and bulk HFSS remain locked. The authoritative snapshot is
`baselines/2026-07-29-v110-three-frequency-active-rl/BASELINE.md`.

## Three-Frequency Mask-Weight Joint Search v1.11

The v1.11 stage audits 500 combined/significant-task active-RL cases and then
searches 18 failed scenes with 24 structured plus eight alternating masks per
explored ratio. One common task-weight command is used at nominal, 9.96, and
10.04 GHz; frequency-specific weights and threshold changes remain prohibited.

| Result | Value | Decision |
|---|---:|---|
| Generated masks | 2,304 | Completed |
| Full optimizer runs | 144 | Completed |
| Three-frequency strict oracle | 7/20 | Improved, insufficient |
| K=2 / K=4 / K=6 strict oracle | 3/7 / 2/6 / 2/7 | Below 80% strata gate |
| Active-RL reserve >= 11 dB | 0/20 | Failed |
| K=2/K=4 high-corner best-of-N >= 10.5 dB | 100% / 100% | No immediate hardware stop |
| Remaining root causes | 9 active-RL, 4 mainlobe | Joint-feasibility gap |

HFSS smoke and critic training remain locked. The next step is a targeted
same-mask joint-feasibility rescue on the recorded pattern/RL bridge masks.
The authoritative snapshot is
`baselines/2026-07-29-v111-three-frequency-joint-search/BASELINE.md`.

## Pareto Joint-Feasibility Rescue v1.12

The v20.1 stage first validates two exact alpha neighborhoods, then reevaluates
all 2,304 existing masks with the five-state warm EEP operator. Each of 72
scene-ratio groups retains pattern-best, active-RL-best, max-min, and
Pareto-knee candidates. The eight failed K=2/K=4 scenes receive progressive
10/10.5/11 dB target-equality-nullspace optimization with a common command at
9.96, 10.00, and 10.04 GHz.

| Result | Value | Decision |
|---|---:|---|
| Exact alpha neighborhoods | 2/2 strict | Passed, no 11 dB reserve |
| Warm masks / selected roles | 2,304 / 288 | Completed |
| Progressive paths / commands | 128 / 2,048 | Completed |
| K=2 strict oracle | 4/7 | Below 6/7 gate |
| K=4 strict oracle | 2/6 | Below 5/6 gate |
| Overall strict oracle | 8/20 | Below 18/20 gate |
| Active-RL reserve >= 11 dB | 0/20 | Failed |
| Failed root causes | 5 active-RL, 2 mainlobe | Six at 10.04 GHz |

The preregistered stop condition triggers. K=6 rescue, HFSS smoke, and critic
training are not executed. Further algorithm-only expansion stops; the next
physical stage must broaden 10.04 GHz active matching and replay the frozen
v20.1 candidates on a newly validated S256/EEP operator. The authoritative
snapshot is
`baselines/2026-07-29-v112-pareto-joint-feasibility/BASELINE.md`.

## Three-Frequency Broadband-Match Replay v1.13

The twenty v1.12 masks, task commands, ratios, E2 states, and thresholds are
frozen with a SHA-256 replay package. Uniform and corner/edge/interior finite-Q
S-L-S networks are cascaded with the three physical raw S256/EEP operators.

| Result | Old series match | Three-class S-L-S | Decision |
|---|---:|---:|---|
| 10.04 GHz passive RL | 10.616 dB | 10.748 dB | Small improvement |
| Minimum network efficiency | 98.05% | 98.09% | Passed 95% gate |
| Maximum EEP map complex NMSE | 1.95e-13 | 1.92e-13 | Passed |
| K=2 strict | 4/7 | 4/7 | Below 6/7 |
| K=4 strict | 2/6 | 2/6 | Below 5/6 |
| Overall strict | 8/20 | 8/20 | No feasible-set gain |
| 11 dB reserve | 0/20 | 0/20 | Failed |

The new circuit network preserves pattern behavior but produces mixed
active-RL changes and no new strict scene. Candidate optimization, HFSS smoke,
bulk labels, and critic training remain locked. The authoritative snapshot is
`baselines/2026-07-29-v113-broadband-match-replay/BASELINE.md`.

## Small-Cell Broadband-Feed Feasibility v1.14

The physical feed is developed and gated on 1x1 and 2x2 only. Frequencies,
frozen representative excitations, and engineering thresholds remain fixed.

| Result | Value | Decision |
|---|---:|---|
| 1x1 three-frequency passive RL | 15.914 dB | Passed 15 dB gate |
| 1x1 cross-mesh max Delta S | 0.02125 | Passed 0.05 gate |
| Trusted 2x2 passive RL | 16.656 dB | Passed 12 dB gate |
| Trusted 2x2 final Delta S | 0.02101 | Passed 0.05 gate |
| Trusted 2x2 minimum efficiency | 99.08% | Passed 95% gate |
| Trusted 2x2 representative active RL | 5.553 dB | Failed 11 dB gate |
| Trusted 2x2 representative total RL | 10.348 dB | Failed 11 dB gate |
| 10.04 GHz four-port modal active RL | 9.788 dB | Below reserve |
| Isolated x-strip candidates passing 2x2 | 0/3 | Topology rejected |

The 1x1 feed and lower-memory surface mesh are credible, but the 2x2 active
matching gate fails. No 4x4 or 16x16 rebuild and no HFSS training labels were
started. The next physical experiment is a grounded or capacitively loaded
x-pair even/odd-mode network on the trusted S4. The authoritative snapshot is
`baselines/2026-07-30-v114-small-cell-broadband-feed/BASELINE.md`.
