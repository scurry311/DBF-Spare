# Baseline v0.3.0-active-rl-joint

## Decision

This baseline freezes the first trusted matched-S256 and EEP joint task-weight
projection on all 96 candidates. Active matching is no longer an all-negative
gate, but new HFSS training labels remain locked because sparse multibeam and
K=6 direction-pattern coverage is insufficient.

## Four-Step Result

1. Active-RL semantics audit: combined-only, task-level, and -20/-30/-40 dB
   driven-port definitions all had zero 10 dB positives before optimization.
2. Joint projection: task variables preserve `w = sum(w_k)`, EEP target and
   regional constraints, matched-S256 active RL, and total reflected power.
3. Full smoke: all 96 candidates were evaluated on the complete EEP grid.
4. HFSS decision: no new training-label batch is allowed at this checkpoint.

## Acceptance Matrix

| Gate | Before | After | Decision |
|---|---:|---:|---|
| Combined active RL | 0/96 | 85/96 | Improved |
| Combined plus significant task active RL | 0/96 | 62/96 | Improved |
| Strict local-20 pattern | 21/96 original | 27/96 | Limited improvement |
| Mainlobe | 29/96 original | 30/96 | Nearly unchanged |
| Strict engineering intersection | 0/96 | 11/96 | Nonzero |
| Sparse strict positives | 0 | 8 | Nonzero |
| Sparse multibeam strict positives | 0 | 1 | Insufficient |
| K=6 strict positives | 0 | 0 | Failed |

Mean combined worst-active-RL improvement is 10.33 dB. The optimizer therefore
finds useful S256-feasible weight directions, but sparse local-null sampling
underestimates the complete EEP local-5-degree leakage by approximately 5-10
dB in hard scenes. Adding an 8 dB scalar margin did not recover K=6 positives.

## Next Gate

Replace the sparse offset rows with a dense local-5-degree EEP operator and add
explicit combined-target equalities. Repeat the same 96-candidate smoke and
require at least one sparse K=6 strict positive plus at least five sparse
multibeam strict positives before preparing a 50-100-case HFSS shortlist.

All snapshots are listed in `artifact_manifest.csv`. Verify them with:

```powershell
python tools/build_result_index.py --tag 2026-07-24-active-rl-joint --verify-only
```
