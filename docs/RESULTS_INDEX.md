# Result Index

Current baseline: `v0.2.0-trusted-eep`, frozen on 2026-07-24.

## Evidence Levels

| Level | Meaning | Permitted use |
|---|---|---|
| A | Trusted HFSS result with numerical provenance and stated gates | Engineering or pattern evidence within the passed gates |
| B | Numerically valid HFSS result with an incomplete physical gate | Diagnostic evidence only |
| C | Validated interface from a blocked physical baseline | Pipeline validation only |
| D | AF, synthesized S256, or learned proxy | Proposal, screening, and pretraining |
| X | Invalid, incomplete, stale, or empty result | Excluded |

## Trusted 16x16 Baseline

| Result | Value | Decision |
|---|---:|---|
| Field-enabled DDM vs trusted direct max Delta S | 4.76e-11 | Passed |
| S reciprocity max error | 2.51e-11 | Passed |
| S passivity maximum singular value | 0.98009 | Passed |
| Matched passive minimum RL | 13.53 dB | Passed |
| Matched passive 10 dB port rate | 100% | Passed |
| Complex EEP completeness | 256/256 ports | Passed |

The fixed-mesh field solve is the current physical reference. Large raw AEDT,
EEP CSV, and operator files remain under `hfss_outputs/` and are not committed.

## EEP/HFSS Validation

| Result | Value | Decision |
|---|---:|---|
| Candidates / independent scenes | 96 / 76 | New labels only |
| Combined plus task-level HFSS cases | 474/474 complete | Passed |
| Maximum no-scale complex NMSE | 7.93e-12 | Passed |
| Maximum magnitude RMSE | 6.83e-5 dB | Passed |
| Basic gate15 rate | 33.33% | Pattern gate |
| Strict local-20 rate | 21.88% | Pattern gate |
| Mainlobe gate rate | 30.21% | Pattern gate |
| Active-RL gate rate | 0% | Failed |
| Strict engineering gate rate | 0% | Failed |

The no-scale result demonstrates that the EEP operator reproduces HFSS fields
for the same linear electromagnetic basis. It does not establish that the
current sparse weights satisfy active matching or low-power hardware gates.

## Residual Critic

| Signal | Observed |
|---|---:|
| Delta PSLL standard deviation | 1.19e-5 dB |
| Delta nearest-isolation standard deviation | 3.49e-5 dB |
| Delta local-isolation standard deviation | 1.55e-5 dB |
| Delta mainlobe-gain standard deviation | 7.21e-6 dB |
| Hard negatives | 0 |
| Hard positives | 17 |
| Near-boundary candidates | 87 |

Neural residual-critic training is held. The nominal EEP and HFSS labels are
identical to numerical precision, so a high-capacity model would fit export
noise. The versioned null checkpoint is only a reconstruction sanity baseline.

## Label Policy

- `trusted_pattern_label`: allowed for EEP/HFSS pattern optimization and gates.
- `trusted_engineering_label`: requires active-RL and strict engineering gates;
  the current new dataset has no positives.
- `proxy_only`: allowed for proposal generation, warm starts, and rejection.
- `legacy_shape_only`: excluded unless explicitly selected for pretraining.
- `invalid`: never included in train/validation/test data.

The split is grouped by independent `sample_index` scenes with no leakage, and
old labels are not mixed automatically. The machine-readable inventories are
`baselines/2026-07-24/artifact_manifest.csv` and the retained historical
`baselines/2026-07-21/artifact_manifest.csv`.
