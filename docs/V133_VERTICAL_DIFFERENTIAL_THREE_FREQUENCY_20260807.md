# v1.33 True-Balanced Vertical Differential Input

## Objective

Replace the stopped aperture/tongue and long planar-CPS branches with a true
two-conductor differential input whose radiator resonance and impedance
transformation controls are physically separable. This stage is restricted to
physical 1x1 HFSS evidence. Array, EEP, label generation, and critic training
remain locked.

## Evidence Chain

1. v1.28 established a groundless two-conductor differential port, but its
   mixed 8.5 mm feed/transformer path returned 406 - j382 ohm and only
   1.13 dB RL. The nominal transformer length was not a genuinely independent
   CAD degree of freedom.
2. v1.29 moved the reference plane directly to the radiator gap. All six
   dual-resonant forks converged, and `direct_d1_both_short` reached
   43.73 - j0.51 ohm and 23.47 dB RL.
3. v1.30 kept the reference plane and total planar-feed path fixed while
   varying only CPS transformer width and length. The variables produced
   distinct physical sensitivities, but the best candidate retained
   +j58.59 ohm and reached only 4.92 dB RL.
4. v1.31 froze that best CPS and changed only the dual-resonant load. No one of
   seven candidates reached 10 dB; the planar CPS branch was stopped.
5. v1.32 replaced the long planar line with a 0.787 mm vertical differential
   via pair and independent bottom pad capacitance. All six candidates passed
   10 dB; three exceeded 15 dB. The best candidate reached 41.05 - j2.46 ohm
   and 19.83 dB RL.
6. v1.33 froze the best v1.32 geometry and independently solved 9.96, 10.00,
   and 10.04 GHz. All three points passed, with minimum RL 18.87 dB.

## Frozen Geometry

| Parameter | Value |
|---|---:|
| Primary arm length | 5.20 mm |
| Secondary arm length | 4.50 mm |
| Secondary offset | 1.00 mm |
| Differential via pitch | 1.10 mm |
| Via radius | 0.18 mm |
| Bottom pad width | 0.50 mm |
| Bottom pad length | 0.80 mm |
| Substrate | 0.787 mm, er 2.2 |

## Three-Frequency Result

| Frequency | Input impedance | Passive RL | Final Delta S |
|---:|---:|---:|---:|
| 9.96 GHz | 40.33 - j3.54 ohm | 18.87 dB | 0.002008 |
| 10.00 GHz | 41.05 - j2.46 ohm | 19.83 dB | 0.002095 |
| 10.04 GHz | 41.78 - j1.38 ohm | 20.84 dB | 0.002006 |

The six-candidate v1.32 sweep also shows the expected independent pad trend:
increasing pad area moves the input reactance from -j2.46 toward -j15.97 ohm.
This is direct HFSS evidence that the matching control is physically active,
not merely a declared parameter.

## Decision

The true-balanced vertical differential branch passes the 1x1 three-frequency
passive matching gate with more than 8.8 dB margin over the 10 dB threshold.
It supersedes the long planar CPS branch as the preferred input architecture.

This is not yet an array engineering conclusion. Before 2x2, the same frozen
candidate must pass radiation-efficiency export and an independent repeated
solve. Before 16x16 or new labels, a physical 2x2 must demonstrate acceptable
passive and active RL, coupling, efficiency, and solver agreement. EEP export,
HFSS label generation, and critic retraining remain locked.
