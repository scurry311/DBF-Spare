# v1.14 Small-Cell Broadband-Feed Feasibility

This baseline evaluates physical broadband-feed feasibility only. It does not
rebuild the 16x16 array, generate HFSS training labels, alter engineering
thresholds, search masks, or retrain the residual critic.

## One-Cell Gate

The final stepped-tongue dual-slot feed (`st_h`) reaches a three-frequency
minimum passive RL of 15.914 dB on the volumetric
0.18 mm feed mesh. The lower-memory surface-refined realization reaches
16.168 dB. Its cross-mesh maximum |Delta S| is
0.02125; the independent direct repeat
has maximum |Delta S| 1.881e-13. Radiation efficiency is approximately
100.00%.

The intermediate tongue candidates were chosen after inspecting preceding
impedance results and are therefore engineering development evidence, not an
independent candidate benchmark.

## Two-Cell Gate

| Metric | Trusted base feed | Required |
|---|---:|---:|
| Minimum passive RL | 16.656 dB | >= 12 dB |
| Representative worst active RL | 5.553 dB | >= 11 dB |
| Representative worst total RL | 10.348 dB | >= 11 dB |
| Minimum radiation efficiency | 99.08% | >= 95% |
| Final Delta S | 0.02101 | <= 0.05 |
| Peak solver memory | 8.55 GiB | Diagnostic |

The passive, efficiency, and convergence gates pass, but representative active
matching fails. At 10.04 GHz the trusted base x-neighbor coupling is reported in
`modal_coupling_comparison.csv`; its worst four-port even/odd-mode active RL is
9.788 dB. The x-neighbor coupling drives the weak mode.
An independent 2x2 repeat is not opened after this active-RL stop condition.

## Local Decoupler Screen

3 isolated x-strip candidates were solved. None passes the 2x2 base
gate. Even the best high-corner four-port modal result among these candidates is
8.480 dB, so this ungrounded
parasitic-strip topology is rejected rather than expanded to 4x4.

## Decision

The 1x1 physical feed and deterministic surface-mesh method are credible, but
the 2x2 representative active-RL gate fails. Therefore 4x4, 16x16 S256/EEP,
candidate HFSS labels, mask search, and critic training remain locked. The next
hardware experiment must synthesize a controlled grounded or capacitively
loaded x-pair even/odd-mode decoupling network on the trusted S4, then validate
one physical 2x2 realization against the same frozen stimuli and unchanged
thresholds.
