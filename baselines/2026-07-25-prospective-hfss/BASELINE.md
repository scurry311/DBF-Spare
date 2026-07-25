# Baseline v0.8.0-prospective-hfss

## Decision

The v0.7 checkpoint, pooled calibrator, feature substitutions, uncertainty
factor, probability threshold, and engineering thresholds were frozen before
this HFSS run. The prospective set contains 24 target-direction sets absent
from the 90 training scenes. No post-HFSS retraining, threshold adjustment, or
mainlobe-failure expansion was performed.

The physical validation passed, but the frozen critic failed the inherited
prospective discrimination and calibration protocol. Automatic HFSS candidate
admission therefore remains disabled. These 72 labels are locked as
prospective evaluation evidence and must not be used to tune this checkpoint
or calibrator.

## Physical Evidence

| Evidence | Result |
|---|---:|
| Independent unseen direction sets | 24 |
| PSLL / nearest / local scenes | 8 / 8 / 8 |
| Candidates / HFSS cases | 72 / 354 |
| Completed HFSS cases | 354/354 |
| Training target-hash overlap | 0 |
| Just-inside gate15 pass | 24/24 |
| Isolated just-outside failure | 24/24 |
| Mainlobe failures | 0 |
| Maximum no-scale complex NMSE | 6.19e-12 |
| Maximum magnitude RMSE | 3.22e-5 dB |

## Frozen Critic

| Prospective metric | Result | Gate |
|---|---:|---:|
| gate15 AUROC | 0.795 | Failed 0.88 |
| gate15 AUPRC / ECE | 0.908 / 0.041 | ECE passed |
| gate20 AUROC | 0.895 | Passed 0.88 |
| gate20 AUPRC / ECE | 0.926 / 0.084 | ECE failed 0.08 |
| strict-engineering AUROC / ECE | 0.978 / 0.017 | Diagnostic |
| pattern15 admission precision / recall | 0.853 / 0.604 | Diagnostic |
| strict admission precision / recall | 1.000 / 0.885 | Diagnostic |

Every inside candidate received a higher gate15 probability than its paired
outside candidate, but the mean probability gap was only 0.0218. Removing the
easy nominal controls reduces gate15 AUROC to 0.590. The critic therefore has
useful scene-grouped ranking bias but insufficient cross-scene absolute
feasibility calibration.

All four frozen top-one methods passed 24/24 scenes because each selected the
nominal control. This is valid evidence for conservative control selection,
not evidence that near-boundary sparse variants are universally reliable.

## Use Policy

- Keep the v0.7 checkpoint and calibrator unchanged for reproducibility.
- Do not enable unconditional automatic HFSS admission.
- The strict compound admission may be studied as a high-precision rejector,
  but its 23/23 precision does not supersede the failed primary protocol.
- Do not train on this prospective set in the same model-development cycle.

Verify the compact snapshots with:

```powershell
python tools\build_result_index.py --tag 2026-07-25-prospective-hfss --verify-only
```
