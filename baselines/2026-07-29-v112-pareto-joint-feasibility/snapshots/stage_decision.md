# v20.1 Pareto Joint-Feasibility Rescue Decision

The engineering thresholds, E2 perturbation, three physical frequencies, and
common-command requirement remained frozen. No HFSS solve or critic training
was performed in this stage.

## Exact Alpha Validation

Two preregistered interpolation neighborhoods were evaluated on a 0.001 alpha
grid with the full five-state EEP/S256 operator.

| Scene | K | Ratio | Best alpha | Worst margin | Active-RL floor | Result |
|---|---:|---:|---:|---:|---:|---|
| 426107 | 2 | 0.8 | 0.812 | 0.078 dB | 10.078 dB | Strict pass |
| 426115 | 4 | 0.7 | 0.843 | 0.126 dB | 10.128 dB | Strict pass |

Both are narrow 10 dB feasibility points and neither reaches the 11 dB design
reserve. Scene 426107 adds one new strict scene; scene 426115 lowers the
minimum feasible ratio from 0.8 to 0.7 but was already scene-feasible.

## Warm Pareto Screening

All 2,304 existing masks were reevaluated with the same five-state warm EEP
operator. The 72 scene-ratio groups retained four distinct roles: pattern
best, active-RL best, max-min best, and Pareto knee. This produced 288 selected
candidates. None of the warm candidates passed the strict or reserve gate, so
active-RL-only top-one selection was not used for the rescue.

## K=2/K=4 Rescue

The eight failed K=2/K=4 scenes produced 128 selected mask paths. Each path
was optimized at active-RL design targets 10, 10.5, and 11 dB, with common
weights across 9.96, 10.00, and 10.04 GHz, target-equality-nullspace updates,
and five-point backtracking. In total, 2,048 commands were evaluated.

| Stratum | Required | Achieved | Rate | 11 dB reserve |
|---|---:|---:|---:|---:|
| K=2 | 6/7 | 4/7 | 57.1% | 0/7 |
| K=4 | 5/6 | 2/6 | 33.3% | 0/6 |

The seven remaining K=2/K=4 failures are limited by active RL in five scenes
and mainlobe preservation in two scenes. Four limiting events occur at the
10.04 GHz identity state, two at the 10.04 GHz E2-source state, and one at the
9.96 GHz E2-source state. Five failed scenes already have a positive robust
pattern margin; adding more pattern candidates cannot fix their active-match
failure.

## Decision

The preregistered K=2/K=4 gate fails and the 11 dB reserve set remains empty.
Therefore:

- K=6 Pareto rescue is not executed.
- No HFSS smoke list is frozen or launched.
- Critic training remains locked.
- Engineering thresholds are unchanged.
- Further algorithm-only candidate expansion stops.

The next physical step is to broaden the matched active-RL bandwidth at 10.04
GHz while preserving the trusted nominal and 9.96 GHz operators. After a new
S256/EEP operator passes the same consistency checks, the frozen v20.1 masks
and commands must be replayed before any new optimization is allowed.
