# Baseline v0.6.0-expanded-residual

## Decision

This baseline freezes the independent-scene expansion requested after v0.5.
The new full-wave labels are allowed for training, and the critic is useful for
candidate-group ranking. The checkpoint remains experimental because gate15
discrimination and probability calibration do not pass the engineering gates.

## Scene And HFSS Expansion

| Evidence | Result |
|---|---:|
| New independent target scenes | 45 |
| New candidates / HFSS cases | 105 / 455 |
| K=2 / K=4 / K=6 scenes | 21 / 18 / 6 |
| Large-scan scenes | 21 |
| Nominal controls | 45 |
| Reoptimized lower-ratio pairs | 15 |
| Targeted mainlobe failures | 24 |
| Total full-wave mainlobe failures | 37 |
| Complete HFSS cases | 455/455 |
| Actual EEP to direct-HFSS maximum NMSE | 6.13e-12 |

All 45 nominal controls and all 15 lower-ratio pairs pass the strict full-wave
engineering gate. The 45 implementation variants supply 18 gate15 hard
negatives and 37 mainlobe failures. No ratio-1 optimization result is included.

## Dataset And Training

The v0.5 evidence and this expansion form 141 candidates from 60 independent
scenes. The grouped split contains 38/11/11 train/validation/test scenes with no
`sample_index` leakage. The test split contains six gate15 negatives, nine
mainlobe negatives, and eleven strict-engineering negatives.

| Five-seed test result | Mean |
|---|---:|
| Gate15 AUROC / AUPRC | 0.836 / 0.904 |
| Gate20 AUROC / AUPRC | 0.893 / 0.944 |
| Gate15 / gate20 ECE | 0.138 / 0.125 |
| Mainlobe / strict-engineering AUROC | 1.000 / 1.000 |
| Strict candidate-group ranking pass rate | 1.000 |
| PSLL residual RMSE | 0.509 dB |
| Nearest / local isolation residual RMSE | 1.467 / 0.474 dB |

## Promotion Gate

Mainlobe failure support and scene-level test support now pass. Engineering
promotion still fails because gate15 AUROC is below 0.88 and calibrated ECE is
above 0.08. The next data increment should target independent PSLL and
nearest/local-isolation threshold crossings, not more mainlobe failures.

Verify the compact snapshots with:

```powershell
python tools/build_result_index.py --tag 2026-07-24-expanded-residual --verify-only
```
