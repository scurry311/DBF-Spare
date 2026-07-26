# Baseline v1.2 EEP/S256 Operating Envelope

## Decision

The current EEP/S256-supported development envelope is:

- K=2: maximum target scan angle 48 deg and minimum target separation 16 deg.
- K=6: maximum target scan angle 58 deg and minimum target separation 13 deg.

This envelope passed the pre-registered scene-level best-of-N strict oracle
gate. It permits a frozen 15-20-candidate HFSS smoke inside the envelope. It
does not establish an HFSS full-wave pass rate, and it does not permit bulk
HFSS or claims about normalized RF power reduction.

## v1.1 Envelope Audit

| Group | Strict scenes | Total | Oracle rate | Decision |
|---|---:|---:|---:|---|
| K=2, scan <=50 deg, separation >=16 deg | 14 | 20 | 70.0% | Failed |
| K=6, scan <=58 deg, separation >=13 deg | 19 | 20 | 95.0% | Passed |
| Overall | 33 | 40 | 82.5% | Failed |

The failure was concentrated at the K=2 50-degree boundary. Tightening a
threshold after seeing these labels was not treated as validation; it produced
the separate v1.2 hypothesis below and required new unseen K=2 scenes.

## Discrete Mask-Continuous Weight Rescue

The rescue solver maps per-port active-reflection stress from the S256 operator
back to the mask. It swaps high-reflection, low-task-contribution active ports
for lower-coupling inactive ports, then reruns dense regional LCMV, combined
target equalities, local-5-degree leakage projection, and active-RL projection.

| Result | Value |
|---|---:|
| Failed scenes searched | 7 |
| Added candidates | 672 |
| Active-RL-guided candidates | 448 |
| Guided active-RL passes | 272/448 (60.7%) |
| Guided strict passes | 4/448 (0.9%) |
| Rescued independent scenes | 2 |
| Overall oracle before / after | 82.5% / 87.5% |
| K=2 50-degree envelope after rescue | 16/20 (80.0%) |

The joint solver has engineering value as a boundary rescue operator, but it
does not expand the K=2 supported scan limit to 50 degrees. The remaining
failures are coupled nearest-isolation and mainlobe constraints, not an
active-RL-only failure.

## Independent v1.2 Validation

The K=2 scan limit was frozen at 48 degrees before generating 20 new direction
sets. Historical, v1.0, v1.1, and frozen prospective target hashes were
excluded.

| Result | Value | Decision |
|---|---:|---|
| New K=2 scenes / candidates | 20 / 1,920 | Complete |
| New K=2 scene oracle | 19/20 (95.0%) | Passed 90% |
| New K=2 target-hash overlap | 0 | Passed |
| Combined in-envelope K=2 | 33/34 (97.1%) | Passed |
| Frozen in-envelope K=6 | 19/20 (95.0%) | Passed |
| Combined in-envelope total | 52/54 (96.3%) | Passed |

Among the 19 new K=2 feasible scenes, the minimum ratios were 0.5 for 16
scenes, 0.6 for two scenes, and 0.7 for one scene. These are EEP/S256 minimum
ratios and still require full-wave confirmation.

## Use Policy

- Keep K=2 50 degrees and all directions outside the limits as a pressure set.
- Use only the supported-scene list for the next frozen HFSS smoke.
- Do not change gates after viewing HFSS smoke labels.
- Do not mix the frozen v0.9 prospective labels into training automatically.
- Permit critic redevelopment only inside this envelope and only with
  scene-grouped splits.
- Require a successful 15-20-candidate HFSS smoke before launching 50-100
  full-wave candidates.
- Continue to report EEP/S256 and HFSS evidence separately.
