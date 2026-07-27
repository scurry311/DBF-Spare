# v1.5 Feature Observability Audit

This baseline completes Stage A of the operator-drift program. It audits every
critic input by inference-time availability and trains three five-seed models
on the identical scene-grouped v1.4 split.

## Evidence Scope

All labels remain a 4x4-HFSS-calibrated 16x16 EEP/S256 proxy. This stage does
not add perturbed 16x16 HFSS evidence and does not promote an engineering
critic.

## Input Tiers

| Tier | Meaning | Current input groups/features | Deployment use |
|---|---|---:|---|
| A | Commanded scene and nominal EEP/S256 information | 35 | Directly deployable |
| B | VNA, calibration, telemetry, and fault-state information | 11 | Measurement-assisted |
| C | True geometry, material, or synthetic drift truth | 3 | Oracle ablation only |

Five future/actual-label signals are explicitly prohibited. The full inventory
and inference source are recorded in `snapshots/feature_observability_matrix.csv`.

## Split Audit

| Result | Value |
|---|---:|
| Candidates / base direction scenes | 4,320 / 45 |
| Train / validation / test candidates | 2,880 / 576 / 864 |
| Train/validation target-hash overlap | 0 |
| Train/test target-hash overlap | 0 |
| Validation/test target-hash overlap | 0 |

## Five-Seed Results

| Test metric | Model A: deployable | Model B: measured | Model C: oracle |
|---|---:|---:|---:|
| Strict AUROC | 0.9829 | 0.9899 | 0.9887 |
| Strict AUPRC | 0.9406 | 0.9566 | 0.9554 |
| Strict Brier | 0.0411 | 0.0334 | 0.0343 |
| Strict ECE | 0.0318 | 0.0311 | 0.0330 |
| Strict precision | 0.9343 | 0.9071 | 0.9064 |
| Strict recall | 0.7672 | 0.8723 | 0.8983 |
| Top-1 strict rate | 0.2917 | 0.3009 | 0.3000 |
| Candidate oracle | 0.3426 | 0.3426 | 0.3426 |

Model A passes by being conservative. Model B adds 10.5 percentage points of
strict recall while retaining mean precision above 90%. Model C does not offer
a meaningful ranking gain over Model B, so unavailable geometry/material truth
is not required for the aggregate proxy gate.

## Envelope Diagnostics

| Drift group | Model A precision | Model B precision | Model C precision |
|---|---:|---:|---:|
| E1 proxy intensity 0.05 | 0.9903 | 0.9887 | 0.9856 |
| E2 proxy intensity 0.20 | 0.5998 | 0.6983 | 0.7136 |

The pre-registered aggregate Stage-A gate passes for Models A and B. The E2
subgroup precision gate fails for all models. Stage B drift-envelope
preregistration and robust candidate optimization may proceed, but the current
critic must not automatically admit E2 candidates or launch HFSS.

## Decision

- `stage_a_gate_pass`: true.
- `stage_b_allowed`: true.
- `e2_critic_auto_acceptance_allowed`: false.
- `engineering_critic_promoted`: false.
- `automatic_hfss_admission_allowed`: false.

The next stage must improve the common-mask/common-command-weight robust oracle
inside the fixed E2 envelope. No additional critic training or 16x16 HFSS is
allowed until that candidate-space gate is evaluated.

