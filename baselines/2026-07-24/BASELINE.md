# Baseline v0.2.0-trusted-eep

## Decision

This baseline freezes the first trusted 16x16 field-enabled S256 and complete
256-port complex Etheta/Ephi EEP workflow. Pattern reconstruction labels are
valid for numerical and algorithmic studies. Engineering-feasibility critic
training remains locked because no candidate passes the active-RL gate.

## Acceptance Matrix

| Gate | Requirement | Observed | Pass |
|---|---:|---:|---:|
| Fieldsolve vs trusted direct Delta S | <= 0.05 | 4.76e-11 | Yes |
| S reciprocity error | <= 1e-4 | 2.51e-11 | Yes |
| Passivity sigma | <= 1.001 | 0.98009 | Yes |
| Matched passive minimum RL | >= 10 dB | 13.53 dB | Yes |
| Complete complex EEP | 256/256 ports | 256/256 | Yes |
| No-scale EEP/HFSS task reconstruction | all 474 cases | 474/474 | Yes |
| Maximum complex-field NMSE | <= 1e-8 | 7.93e-12 | Yes |
| Maximum visible magnitude RMSE | <= 0.05 dB | 6.83e-5 dB | Yes |
| Scene leakage | none | none | Yes |
| Active-RL engineering positives | nonzero | 0/96 | No |

## Dataset

- 96 candidates from 76 independent scenes.
- K = 1/2/4/6, ratio = 0.5/0.6/0.7/0.8/0.9/1.0, including large scans.
- 474 HFSS task cases: one combined case plus K task cases per candidate.
- The dataset stores mask, task-level weights, combined weights, EEP metrics,
  HFSS metrics, S256 matching metrics, residuals, gates, and split labels.
- The canonical `dataset_v2_20260724` package exposes explicit `sample_index`,
  `candidate_index`, `mask`, `w_tasks_real_imag`, and
  `w_combined_real_imag` keys without modifying the original run output.
- Split is grouped by `sample_index`: 52/12/12 train/validation/test scenes.
- Legacy labels are excluded.

## Critic Decision

The four HFSS-minus-EEP residual standard deviations are only 7.21e-6 to
3.49e-5 dB. There are zero EEP-pass/HFSS-fail hard negatives, zero active-RL
positives, and zero strict engineering positives. A neural residual critic
would therefore learn numerical export noise rather than a physical bias.

The included null checkpoint is a reproducibility baseline only. Neural critic
training resumes after active-RL-constrained candidates and genuine physical
perturbation labels create both positive and negative support.

All copied artifacts are listed in `artifact_manifest.csv`; their SHA-256
digests must pass `python tools/build_result_index.py --tag 2026-07-24
--verify-only` before use.
