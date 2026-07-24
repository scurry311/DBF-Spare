# Baseline v0.4.0-dense-local-hfss

## Decision

This baseline freezes the dense local-5-degree EEP projection and its gated
HFSS verification. The requested opening conditions were met before the batch
was launched: 15 sparse multibeam strict positives and 2 sparse K=6 positives
were found in the same 96-candidate smoke.

## Dense EEP Result

| Gate | Result |
|---|---:|
| Candidate count | 96 |
| Dense local-5-degree strict gate20 | 65/96 |
| Mainlobe gate | 29/96 |
| Strict engineering gate | 27/96 |
| Sparse multibeam strict positives | 15 |
| Sparse K=6 strict positives | 2 |

The operator uses every EEP angular-grid point inside each non-own target's
5-degree neighborhood and explicit complex combined-pattern target equalities.
The task variables preserve `w = sum(w_k)`.

## HFSS Verification

| Result | Value |
|---|---:|
| Candidates / cases | 15 / 65 |
| K=2 / K=4 / K=6 candidates | 7 / 6 / 2 |
| Ratio-1 controls | 0 |
| Complete cases | 65/65 |
| Maximum no-scale complex NMSE | 6.03e-12 |
| Maximum magnitude RMSE | 3.07e-5 dB |
| Strict engineering positives | 15/15 |
| Sparse K=6 positives | 2/2 |

These 15 records are accepted as trusted sparse positive and near-boundary
HFSS labels. The generic validator reports `labels_allowed=false` only because
its legacy policy requires exactly 96 candidates; the experiment-specific
decision removes that unrelated row-count assumption.

## Critic Gate

Residual-critic training remains locked. There are no hard negatives and the
largest residual standard deviation is only 3.10e-5 dB, so a learned residual
model would fit numerical export noise. Add paired near-boundary perturbations
and EEP-pass/HFSS-fail cases before retraining the critic.

All compact snapshots are listed in `artifact_manifest.csv`. Verify them with:

```powershell
python tools/build_result_index.py --tag 2026-07-24-dense-local-hfss --verify-only
```
