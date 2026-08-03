# v1.21 Parametric Feed/POST Build-Smoke Baseline

## Evidence Level

This checkpoint contains a verified parameterized HFSS CAD build, not solved
S-parameter or antenna performance evidence. Circuit, cascade, and EEP results
from earlier versions are not promoted to integrated v1.21 HFSS claims.

## Frozen Parent

- Commit: `94aae2f694b406b777d3eaa50cb7b792dc2a2cb0`
- Tag: `v1.20.0-joint-feed-fanout-sparse-graph`
- Sparse physical chain: `0-2-3-1`
- Decoupling stages: one
- Non-neighbor connections: forbidden

## Reproducible Result

The network-only S8 and integrated 2x2 S4 builders share the same eleven
physical parameters and v1.20 finite-Q sparse-graph values. Nominal,
constrained-lower, and upper parameter sets all build in AEDT 2023 R1.

Preregistration amendment 01 was made before any S-parameter solve. It adds
the route-length feasibility constraint and projects only `doe12_lhs` common
POST correction from -0.9587 mm to -0.9287 mm. The original preregistration
and DOE remain preserved beside the effective files; no engineering threshold
or parameter range changed.

| Audit | Result |
|---|---:|
| Network-only builds | 3/3 passed |
| Integrated 2x2 builds | 3/3 passed |
| Port-set audits | 6/6 passed |
| Graph/single-stage audits | 6/6 passed |
| Small-segment or geometry warnings | 0 |
| Prepared 10 GHz LHS candidates | 16 |

The initial integrated attempt is intentionally retained in the snapshots. It
failed because the mesh operation referenced patch/probe names removed by a
Boolean union. The corrected attempt assigns the feed mesh to the surviving
united conductor without changing physical dimensions or thresholds.

## Gate State

The CAD gate passes. The formal solve gate remains closed because free memory
at the final audit was about 12.53 GiB, below the preregistered 13 GiB launch
threshold. There is no v1.21 active-RL, efficiency, Delta-S, direct/DDM, or
integrated solved result in this baseline.

The following remain prohibited:

- three-frequency promotion before the 10 GHz physical DOE;
- integrated solving before a three-frequency S8 pass;
- 4x4 or 16x16 expansion;
- full-wave label generation;
- residual critic retraining.

## Files

`snapshots/preregistration.json` preserves the original protocol and
`snapshots/preregistration_amendment01.json` records the pre-solve correction.
The effective protocol and DOE use the corresponding `effective` filenames.
The two build-audit pairs preserve both the failed bookkeeping attempt and the
successful corrected attempt.
