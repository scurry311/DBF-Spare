# v1.42/v1.43 Mapped-EEP Joint Optimization And Input Redesign

## Question

Can task weights recover a nonempty K=4/K=6 strict set after the v1.41 modal
correction is included in both S4 and the EEP command map? If K=6 remains
empty, can a y-balanced element input reduce the physical modal split and
active reflection without another correction-network stage?

## v1.42 Method

The finite-Q v1.41 external S4 and antenna-incident map are tiled over 64
nonoverlapping 2x2 cells in `P00, P10, P01, P11` order. The nominal 256-port
EEP is mapped through the same operator. Each frozen scene uses direct,
block-LS, and blended task-weight seeds. Two Pareto starts are refined by a
sequential complex SOCP.

Solver-only leakage slacks prevent an infeasible warm start from terminating
the search. They do not change acceptance: every candidate is re-evaluated
with the complete dense local-5deg constraints, original pattern thresholds,
hardware limits, and 11 dB active-RL reserve.

## v1.42 Result

| K | Scenes | Engineering strict | Reserve 11 dB |
|---:|---:|---:|---:|
| 2 | 7 | 4 | 4 |
| 4 | 6 | 1 | 1 |
| 6 | 7 | 0 | 0 |

K=6 failures are joint failures: six of seven selected candidates fail the
dense task constraints, five fail hardware limits, three fall below 11 dB
active RL, and three fail the pattern margin. Since K=6 is empty, the v1.41
network is stopped.

## v1.43 Physical Redesign

A convex modal target synthesis over 1,566 frozen stimuli places all four
desired mode impedances around 41-50 ohm with small capacitive reactance. The
new true-differential element mirrors the secondary branch about y and keeps
via/pad dimensions as an independent input transformer.

The 1x1 physical HFSS prescreen reaches `37.67-j7.97 ohm`, passive RL
`15.56 dB`, and final Delta S `0.003179`. This validates the self-impedance
direction only.

Strict 2x2 meshes at 0.10 and 0.12 mm exceed the memory stop line. A final
preregistered 0.18 mm/5% direct solve completes with valid S4 and final Delta
S `0.006754`. Its physical performance is worse than v1.39 in the metrics that
matter: worst active RL `-6.16 dB`, total RL `6.94 dB`, and minimum system
efficiency `80.15%`. Y-neighbor isolation improves only `0.13 dB`; even and
x-odd modes worsen.

## Decision

The mirrored-secondary input topology is also stopped. It proves that fixing
1x1 self impedance does not solve coherent active matching when the 2x2
coupling modes remain separated. The next hardware experiment must use a
ground-backed/cavity-backed radiator or element-level neutralization with an
independent impedance transform. EEP export, labels, critic training, 4x4,
and 16x16 remain locked.
