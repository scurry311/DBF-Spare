# Baseline v1.0 Targeted EEP Oracle Audit

## Decision

The targeted K=2/K=6 candidate-space expansion is complete but is not promoted
to critic retraining or HFSS validation.  The combined 24+8 masks per ratio
produce a scene-level EEP/S256 strict oracle rate of 22/48 (45.8%), below the
required 90% gate.  Candidate-level positives must not be reported as scene
coverage.

## Independent Development Pool

| Evidence | Result |
|---|---:|
| Independent scenes | 48 |
| K=2 / K=6 scenes | 24 / 24 |
| Ratios | 0.5, 0.6, 0.7, 0.8 |
| Initial masks per ratio | 24 |
| Initial candidates | 4,608 |
| Local rescue masks per failed-scene ratio | 8 |
| Rescue candidates | 832 |
| Initial strict candidates / hard negatives | 452 / 202 |
| Frozen/excluded target-hash overlap | 0 |

The targeted search adds historical hard-positive mask neighborhoods,
swap/add/remove-equivalent fixed-count mutations, high-to-low ratio
continuation, quadrant/ring/center/edge/spacing structures, regional EEP LCMV
warm starts, S256-aware mask scoring, dense local-5-degree leakage projection,
and active-return projection.  The second prospective v0.9 scenes remain
evaluation-only and were not used as targets or donors.

## Scene-Level Oracle

| Group | Strict scenes | Total | Oracle rate |
|---|---:|---:|---:|
| Overall | 22 | 48 | 45.8% |
| K=2 | 15 | 24 | 62.5% |
| K=6 | 7 | 24 | 29.2% |
| Non-large scan | 17 | 22 | 77.3% |
| Large scan | 5 | 26 | 19.2% |

The additional local search did not rescue another scene.  Ten failed scenes
are within 0.5 dB aggregate margin violation and 15 are within 2 dB, but the
remaining large-scan and K=6 cases show coupled mainlobe, isolation, PSLL, and
active-RL failures.  A single mask swap often destroys active-RL feasibility.

## Use Policy

- Do not retrain or promote the residual critic from this pool.
- Do not launch HFSS based on the 452 candidate-level EEP positives.
- Keep the v0.9 frozen prospective labels evaluation-only.
- Treat 6-8 degree separation with a local-5-degree 20 dB null requirement as
  an out-of-domain feasibility stress case for the current aperture.
- The next physical step must change the joint active-RL formulation or the
  supported scan/separation envelope; lowering the gate is not allowed.
