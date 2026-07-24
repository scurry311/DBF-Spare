# Baseline v0.7.0-gate15-boundary

## Decision

This baseline freezes the dedicated gate15 boundary expansion requested after
v0.6. The new labels target PSLL, nearest isolation, and local isolation just
inside and outside their thresholds. No mainlobe-failure scene was added.

The residual critic passes the retrospective stage-one discrimination and
calibration gates only when the checkpoint and pooled calibrator are used as a
single inference package. Prospective HFSS validation on unseen scenes remains
mandatory before automatic candidate admission.

## Boundary Dataset

| Evidence | Result |
|---|---:|
| Independent boundary scenes | 30 |
| PSLL / nearest / local scenes | 10 / 10 / 10 |
| Control / just-inside / just-outside candidates | 30 / 30 / 30 |
| Full-wave cases | 444/444 complete |
| Just-inside gate15 pass | 30/30 |
| Just-outside isolated failure | 30/30 |
| New mainlobe failures | 0 |
| Maximum no-scale complex NMSE | 6.03e-12 |
| Maximum magnitude RMSE | 2.97e-5 dB |

The mean inside/outside values are `-0.323/+0.227 dB` for PSLL,
`25.322/24.393 dB` for nearest isolation, and `15.449/14.349 dB` for local
isolation. These are pattern-residual labels; only 35.6% of the new candidates
pass the active-RL gate, so they do not by themselves prove final hardware
feasibility.

## Dataset And Training

The four-source dataset contains 231 candidates from 90 independent scenes.
The grouped split uses 62/14/14 train/validation/test scenes with no
`sample_index` leakage. Both validation and test contain all three new boundary
types and seven retained mainlobe negatives; no new mainlobe negatives were
generated.

| Five-seed held-out result | Mean | 95% CI half-width |
|---|---:|---:|
| Gate15 AUROC | 0.883 | 0.006 |
| Gate15 AUPRC | 0.957 | 0.003 |
| Gate15 pooled-calibrated ECE | 0.050 | 0.014 |
| Gate20 AUROC | 0.910 | 0.015 |
| Gate20 AUPRC | 0.959 | 0.006 |
| Gate20 pooled-calibrated ECE | 0.071 | 0.017 |
| Mainlobe / strict-engineering AUROC | 0.999 / 1.000 | - |
| Strict candidate-group ranking rate | 1.000 | 0.000 |

The pooled regularized-isotonic calibrator is fitted only on validation
predictions. Its regularization coefficient is selected by five-fold
`sample_index`-grouped out-of-fold NLL; test labels are not used for calibrator
selection. The standalone checkpoint's temperature-scaled ECE remains above
the gate and must not be reported as calibrated.

## Next Gate

Freeze the checkpoint and calibrator. Run prospective HFSS validation on new
target-direction sets without changing weights, split, thresholds, or
calibration. Only after that run confirms discrimination, ECE, active-RL, and
candidate-admission precision should the critic control automatic HFSS entry.

Verify the compact snapshots with:

```powershell
python tools\build_result_index.py --tag 2026-07-25-gate15-boundary --verify-only
```
