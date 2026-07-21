# Result Index

Baseline: `v0.1.0-physics-gated`, frozen on 2026-07-21.

## Evidence levels

| Level | Meaning | Permitted use |
|---|---|---|
| A | Converged HFSS result with numerical and engineering gates | Engineering evidence |
| B | Numerically valid HFSS result with incomplete convergence or RF gate | Diagnostic evidence only |
| C | Validated EEP/full-wave interface sourced from a blocked baseline | Pipeline validation only |
| D | AF, synthesized S256, or learned proxy | Proposal, screening, and pretraining |
| X | Invalid, incomplete, stale, or empty result | Excluded |

## Physical baseline

| Model | Evidence | Delta S | Matched min RL | Status |
|---|---:|---:|---:|---|
| Grounded patch 1x1 | A | 0.02343 | 15.96 dB | Passed |
| Grounded patch 4x4 | A | 0.03670 | 11.87 dB | Passed |
| Grounded patch 8x8 | A | 0.03453 | 10.93 dB | Passed |
| Grounded patch 16x16 DDM pass1 | B | N/A | 3.71 dB | Reciprocal/passive; blocked |
| Grounded patch 16x16 DDM pass2 | B | 0.29495 | 5.13 dB | Reciprocal/passive; not converged |

The 16x16 pass2 matrix has reciprocity error `2.50e-6`, maximum singular value
`0.97779`, and 32.03% of matched passive ports at or above 10 dB. Training
labels remain locked because neither convergence nor minimum RL passes.

## EEP and full-wave checks

| Result | Value | Level | Decision |
|---|---:|---:|---|
| 256-port EEP field completeness | 256/256 ports | C | Interface complete |
| Three-case direct superposition NMSE | <= 3.02e-12 | C | Linear mapping verified |
| HFSS smoke basic gate | 10/26 | B | Diagnostic only |
| HFSS smoke strict gate | 8/26 | B | Diagnostic only |
| K=6 HFSS smoke strict gate | 0/10 | B | Hard-case failure |
| Close 5-10 degree paired joint pass | 0% | B | Candidate/physics gap |

## Algorithm and data baseline

| Result | Value | Evidence level |
|---|---:|---:|
| AF + local-kernel S256 joint gate | 64.29% of 2400 | D |
| Active-return projection engineering gate | 39.25% of 2400 | D |
| Mean effective ratio after projection | 0.639 | D |
| Full-wave residual dataset | 2019 variants / 265 scenes | B |
| Dataset strict engineering positive rate | 4.85% | B |
| Critic strict AUROC | 0.9133 | B |
| Critic strict scene top-1 | 5.13% | B |
| Best fixed strict strategy | 7.69% | B |
| Observed strict best-of-N oracle | 10.26% | B |

High AUROC has not translated into useful candidate ranking. The current
bottleneck is the small physically feasible candidate set, especially for
K=6, large scans, and closely spaced targets.

## Label policy

- `trusted_fullwave`: accepted only after convergence, S-matrix validity, RF
  gate, and complete pattern/isolation output.
- `proxy_only`: allowed for proposal generation, warm starts, and rejection.
- `legacy_shape_only`: allowed only for pattern-shape pretraining after source
  convention audit.
- `invalid`: never included in train/validation/test data.

The machine-readable inventory is
[`baselines/2026-07-21/artifact_manifest.csv`](baselines/2026-07-21/artifact_manifest.csv).
