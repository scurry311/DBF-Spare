# v1.22 Balanced Differential Launch and Local Modal Branch

## Purpose

v1.22 replaces the stopped v1.21 single-ended POST/local-loading mapping with
a genuinely differential physical launch. It contains no ideal RLC sheet and
no second decoupling stage.

The minimum physical sequence is:

1. a 1x1 two-reference-plane differential S2 launch;
2. a 2x2 four-channel differential S8 fixture;
3. one local correction section formed by symmetric floating copper strips
   between the two physical x-neighbor channel pairs.

The floating strips couple differently to even and odd adjacent-channel field
states without galvanically shorting ports. Four PRE ports and four POST ports
are explicit 50-ohm differential power-wave reference planes.

Thin copper is represented by two-sided finite-conductivity sheets with an
explicit DC thickness. This retains copper surface loss while avoiding
volumetric meshing of thin conductor edge fragments.

## Evidence Boundary

The network-only S8 may be numerically terminated by the trusted 2x2 antenna
S4 to screen representative active RL, total RL, insertion efficiency, and
transducer efficiency. That cascade is not an integrated HFSS antenna result.
An integrated 2x2 model remains locked until the S8 gate and an independent
repeat both pass.

No 4x4/16x16 build, training label, or critic update is permitted in this
stage.

## Gates

The 1x1 launch must converge with Delta S <= 0.05 and remain reciprocal/passive.
Mesh-fragment diagnostics are normalized to the independently trusted v1.14
2x2 S4 baseline: at most 100 records per differential channel and no reported
segment shorter than 0.005 mm. Any candidate passing this screen still requires
an independent repeat before integrated modeling. The launch must also
provide three-frequency input and output RL >= 15 dB, insertion efficiency >=
95%, and transducer efficiency >= 92%.

Only then may the 2x2 S8 run. Its three-frequency requirements are passive RL
>= 12 dB, representative active and total RL >= 11 dB, matched-load network
efficiency >= 95%, actual-load insertion efficiency >= 95%, and actual-load
transducer efficiency >= 90%.

The independent repeat must satisfy max absolute Delta S <= 0.05 before an
integrated 2x2 antenna/feed model is authorized.

## Memory-Safe Three-Frequency Execution

The first 2x2 discrete sweep completed adaptive meshing but exceeded the 3 GiB
free-memory stop line when multiple frequency workers ran concurrently. Its
incomplete output is not evidence. The frozen geometry is therefore solved as
three exact, independent LastAdaptive cases at 9.96, 10.00, and 10.04 GHz,
strictly serially. No interpolation, geometry change, or threshold change is
introduced by this execution amendment.

## Final Gate Result

The run06 1x1 launch passes all gates: three-frequency input/output RL is at
least 24.98 dB, insertion efficiency is 95.27%, transducer efficiency is
94.99%, and final Delta S is 0.00323.

All three serial 2x2 S8 cases converge with worst final Delta S 0.02405,
reciprocity error 1.37e-5, passivity sigma 0.9792, passive RL 20.74 dB, and
matched-load network efficiency 95.27%. Under the trusted antenna S4 and 285
frozen representative excitations, however, worst active RL is 5.74 dB,
total RL is 10.05 dB, actual-load insertion efficiency is 93.97%, and
transducer efficiency is 84.69%. These four physical loading gates fail.

The single local floating modal branch is therefore numerically credible but
not an engineering-feasible active matching network. Independent repeat,
integrated 2x2, array expansion, label generation, and critic training remain
locked. The next hardware model must change the correction mechanism itself,
not add another cascaded branch to this topology.
