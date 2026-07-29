# v1.12 Pareto Joint-Feasibility Rescue

This baseline executes the preregistered v20.1 decision sequence without
changing E2, physical frequencies, common-command semantics, or engineering
thresholds. HFSS and critic training remain locked throughout.

## Results

| Result | Value | Decision |
|---|---:|---|
| Exact dense-alpha validations | 2/2 strict | Passed at 10 dB only |
| Existing warm masks reevaluated | 2,304 | Completed |
| Scene-ratio groups / selected roles | 72 / 4 | Completed |
| K=2/K=4 optimized Pareto paths | 128 | Completed |
| Evaluated progressive commands | 2,048 | Completed |
| K=2 strict oracle | 4/7 | Below 6/7 gate |
| K=4 strict oracle | 2/6 | Below 5/6 gate |
| Overall K=2/4/6 strict oracle | 8/20 | Below 18/20 gate |
| Active-RL reserve >= 11 dB | 0/20 | Failed |

Scene 426107 becomes newly feasible at ratio 0.8 with alpha 0.812 and a
0.078 dB worst margin. Scene 426115 remains feasible and its minimum ratio
improves from 0.8 to 0.7 at alpha 0.843 with a 0.126 dB worst margin. Neither
candidate has enough margin for the 11 dB design gate.

## Root Cause

Seven K=2/K=4 scenes remain infeasible. Five are limited by active RL and two
by mainlobe preservation. Six limiting states occur at 10.04 GHz and one at
9.96 GHz. Five of the failed scenes already have positive robust pattern
margins, which isolates high-frequency active matching as the dominant
physical bottleneck rather than PSLL or candidate-pool coverage.

## Decision

The preregistered K=2/K=4 gate fails. K=6 rescue is skipped, no HFSS smoke is
started, and critic training remains locked. Algorithm-only candidate
expansion stops. The next stage must broaden the matched active-RL bandwidth
at 10.04 GHz and then replay the frozen masks and commands on a newly validated
S256/EEP operator before any optimization or labeling is reopened.
