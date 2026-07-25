# Baseline v0.9.0-physical-margin-adaptive

## Decision

The v0.9 physical-margin critic and calibrated adaptive-ratio selector are
promoted for conservative HFSS admission.  Promotion applies only when all
five lower-confidence physical margins pass and the calibrated strict
probability is at least 0.5.  The second independent prospective run admitted
5 of 12 scenes and all 5 passed the strict HFSS engineering gate.

This is not completion of the final coverage objective.  Prospective admission
coverage is 41.7%, below the requested 80% adaptive-ratio scene pass rate.  The
result supports high-precision rejection and candidate routing, not universal
scene feasibility.  Ratio 1.0 was excluded from optimization and no new paired
same-EIRP ratio-1 HFSS power comparison was run, so channel reduction must not
be presented as measured RF power reduction.

## Development Evidence

| Evidence | Result |
|---|---:|
| Independent development scenes | 60 |
| Full EEP/S256 candidates | 1,920 |
| Ratios / masks per ratio | 0.5, 0.6, 0.7, 0.8 / 8 |
| Selected development candidates | 420 |
| Train / validation / test scenes | 36 / 12 / 12 |
| Strict positives / hard negatives | 79 / 70 |
| Nominal controls / ratio-1 candidates | 0 / 0 |

The critic predicts five residual margins: PSLL, nearest isolation,
local-20 isolation, mainlobe preservation, and active return loss.  Gate
probabilities and ranking scores are derived from these margins; there is no
independent opaque gate head.

| Five-seed test metric | Result |
|---|---:|
| gate15 AUROC / ECE | 0.989 / 0.055 |
| strict AUROC / ECE | 0.935 / 0.073 |
| No-control top-1 strict rate | 66.7% |
| Fixed ratio-0.6 nominal-margin rate | 41.7% |
| Oracle strict rate | 66.7% |
| Mean top-1 ratio | 0.565 |

## Full-Wave Evidence

The 18-candidate smoke completed 82/82 cases and authorized the larger batch.
The non-overlapping 66-candidate batch completed 338/338 cases.  Together they
cover all 84 candidates in the held-out 12-scene development test split.

| Held-out HFSS metric | Result |
|---|---:|
| Candidates / cases | 84 / 420 |
| Strict engineering positives | 18 |
| EEP/HFSS strict-label agreement | 94.0% |
| Maximum no-scale complex NMSE | 6.10e-12 |
| Maximum magnitude RMSE | 4.04e-5 dB |
| Strict AUROC / AUPRC | 0.918 / 0.732 |
| Strict ECE before HFSS-val calibration | 0.091 |
| Critic top-1 / fixed ratio-0.6 / oracle | 50.0% / 33.3% / 66.7% |
| Adaptive admission / admitted precision | 41.7% / 80.0% |

A regularized Platt calibrator was fit on 12 independent validation scenes,
one candidate per scene.  It reduced held-out ECE to 0.066 without changing
AUROC, while Brier increased slightly from 0.0995 to 0.1013.  This tradeoff is
retained explicitly; the calibrator is accepted for the prospective precision
test but is not described as uniformly better.

## Second Prospective Validation

The ensemble, calibrator, probability threshold, margin confidence rule, and
checkpoint hashes were frozen before HFSS.  The 12 target-direction sets were
excluded from the v0.9 development pool, the v0.8 prospective set, and earlier
boundary datasets.

| Prospective metric | Result |
|---|---:|
| Unseen scenes / candidates / HFSS cases | 12 / 12 / 60 |
| Completed cases | 60/60 |
| Admitted scenes | 5/12 |
| Admitted strict HFSS pass | 5/5 |
| K=6 admitted positive | ratio 0.6 |
| Admitted ratios | 0.5 x3, 0.6 x1, 0.7 x1 |
| Mean admitted ratio / channel reduction | 0.56 / 44% |
| Maximum no-scale complex NMSE | 5.85e-12 |
| Post-HFSS tuning | None |

The admitted K=4/K=6 cases have PSLL from -5.10 to -6.77 dB, nearest
isolation from 25.49 to 27.67 dB, local-5-degree isolation from 21.72 to
22.39 dB, and worst active return loss from 10.34 to 11.25 dB.

## Use Policy

- Use the five-checkpoint ensemble and frozen Platt calibrator together.
- Admit a candidate only when calibrated probability is at least 0.5 and all
  five conservative margins pass; otherwise retain it as a fallback, not a
  feasible result.
- Keep ratio 1.0 as a separate engineering control and do not train on it as a
  sparse candidate.
- Lock the second prospective labels as evaluation-only evidence.
- The next data increment must target feasible K=2/K=6 fallback scenes and
  improve coverage without reducing admitted precision below 80%.

Verify the compact snapshots with:

```powershell
python tools\build_result_index.py --tag 2026-07-26-v09-adaptive-hfss --verify-only
```
