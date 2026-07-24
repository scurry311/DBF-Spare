# Baseline v0.5.0-implementation-residual

## Decision

This baseline freezes the first implementation-conditioned full-wave residual
critic. Training is scientifically open because the dataset now contains real
nominal-EEP-pass/direct-HFSS-fail pairs with nontrivial residual variance. The
trained checkpoint remains experimental and is not promoted to the engineering
gate.

## Label Construction

Each trusted sparse positive supplies a nominal command. The paired HFSS case
uses documented phase/amplitude quantization, gain/phase error, or failed
channels. Lower-ratio masks remain paired under the same `sample_index`.

| Evidence | Result |
|---|---:|
| New candidates / HFSS cases | 21 / 95 |
| Complete cases | 95/95 |
| Hard negatives | 15 |
| Lower-ratio pairs | 6 |
| Actual EEP to direct-HFSS maximum NMSE | 5.85e-12 |
| Nominal to direct-HFSS maximum magnitude RMSE | 3.77 dB |

The actual-weight EEP/direct-HFSS agreement verifies the source mapping. The
nominal-command difference is the implementation residual learned by the
critic.

## Dataset and Training

| Result | Value |
|---|---:|
| Candidates / independent scenes | 36 / 15 |
| Train / validation / test scenes | 10 / 2 / 3 |
| Train / validation / test hard negatives | 10 / 2 / 3 |
| Model parameters | 188343 |
| Random seeds | 5 |
| Gate15 / gate20 AUROC | 1.00 / 1.00 |
| Gate15 / gate20 mean ECE | 0.175 / 0.169 |
| Nearest / local isolation residual RMSE | 1.37 / 1.14 dB |

## Promotion Gate

The checkpoint is restricted to uncertainty-aware boundary ranking. It fails
engineering promotion because mean ECE exceeds 0.08, only three independent
scenes are in the test split, and the mainlobe gate has no negative test
support. Add independent scenes and mainlobe-failure perturbations before the
next promotion attempt.

All compact snapshots are listed in `artifact_manifest.csv`. Verify them with:

```powershell
python tools/build_result_index.py --tag 2026-07-24-implementation-residual --verify-only
```
