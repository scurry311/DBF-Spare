# Baseline v0.1.0-physics-gated

## Decision

This baseline freezes the project after the first numerically valid 16x16 DDM
pass pair. The benchmark is not yet eligible for new EEP/HFSS training labels.

## Acceptance matrix

| Gate | Requirement | Observed | Pass |
|---|---:|---:|---:|
| Pass2 Delta S | <= 0.05 | 0.29495 | No |
| Consecutive Delta S | two passes <= 0.05 | zero | No |
| Reciprocity | <= 1e-4 | 2.50e-6 | Yes |
| Passivity sigma | <= 1.001 | 0.97779 | Yes |
| Matched passive min RL | >= 10 dB | 5.13 dB | No |
| Fresh engineering labels | all physical gates | locked | No |

## Included evidence

- Converged 1x1, 4x4, and 8x8 grounded-patch stage summaries.
- DDM pass1/pass2 S256, profile, and audited metrics.
- EEP operator and superposition summaries.
- HFSS joint-smoke and label decisions.
- AF/proxy optimization, active-return projection, dataset, critic, and
  Stage-1 acceptance summaries.

All copied artifacts are listed in `artifact_manifest.csv`; SHA-256 must match
before the baseline is used for comparison.

## Next gate

Continue staged 16x16 DDM refinement with the same deterministic 0.18 mm feed
mesh and 5% refinement. Do not retrain the full-wave critic until the benchmark
has two consecutive converged passes, a physically valid S256, and a meaningful
10 dB passive/active-RL feasible set.

If passive RL remains far below 10 dB after convergence, redesign matching
against the converged full-array S256 before generating new labels.
