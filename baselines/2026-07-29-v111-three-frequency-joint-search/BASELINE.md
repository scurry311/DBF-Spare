# v1.11 Three-Frequency Mask-Weight Joint Search

This baseline audits per-port active return at nominal, 9.96 GHz, and 10.04
GHz, then permits structured mask changes and one common set of task weights
across all physical operators. Frequency-specific commands, threshold changes,
HFSS labels, and critic training are prohibited.

## Active-RL Sensitivity

The 20 frozen v1.10 candidates produce 500 combined/significant-task cases
across five physical/source states. Events whose worst active RL is below the
11 dB design target are concentrated on interior ports.

| Worst-port class | Below-11-dB events |
|---|---:|
| Interior | 198 |
| Edge | 58 |
| Corner | 3 |

Repeated high-risk paths are dominated by short-range coupling, including
interior nearest-neighbor channels. The mask score therefore combines target
direction utility, active-reflection stress, cross-frequency variation, and
passive coupling burden.

## Joint Search

Eighteen v1.10 failures are searched at ratio 0.5/0.6/0.7/0.8. Each explored
scene-ratio generates 24 structured masks followed by eight active-RL-guided
alternating masks. All 32 are screened with the three-frequency S256 model;
the top mask in each round receives the full regional-constraint projection.
The optimized path is evaluated at five Pareto continuation points.

| Result | v1.10 | v1.11 |
|---|---:|---:|
| Three-frequency strict oracle | 2/20 | 7/20 |
| Strict K=2 | 0/7 | 3/7 |
| Strict K=4 | 0/6 | 2/6 |
| Strict K=6 | 2/7 | 2/7 |
| Active-RL reserve >= 11 dB | 0/20 | 0/20 |

The minimum strict-ratio distribution is three scenes at 0.6, two at 0.7,
and two at 0.8. Ratio 1.0 remains a control and is not optimized.

## Failure Analysis

Thirteen scenes remain infeasible. Their max-min selected root constraints are
nine active-RL failures and four mainlobe failures. Nevertheless, every failed
scene has separate pattern-feasible and active-RL-feasible candidates. The
same mask contains separated pattern/RL feasible weight points 14 times for
K=2, 12 times for K=4, and twice for K=6.

The 10.04 GHz best-of-N active-RL floor reaches at least 10.5 dB for 100% of
K=2 and K=4 scenes and 85.7% of K=6 scenes. The preregistered immediate
matching-redesign stop condition therefore does not trigger. The remaining
problem is primarily the missing intersection of pattern and active-RL
feasible weights, not absence of high-corner active-RL candidates.

## Decision

The 18/20 acceptance gate fails and the 11 dB reserve set is empty. No HFSS
smoke list is frozen, no full-wave labels are generated, and critic training
remains locked.

The next algorithmic step is a targeted same-mask feasibility rescue on the
recorded bridge masks: fine Pareto continuation plus augmented active-RL
projection constrained by all three frequency target equalities. If that step
cannot create 11 dB reserve candidates, the 10.04 GHz matching bandwidth must
be broadened before HFSS.
