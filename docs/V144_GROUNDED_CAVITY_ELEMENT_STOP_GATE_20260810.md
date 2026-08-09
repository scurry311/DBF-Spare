# v1.44 Grounded-Cavity Element And Input Stop Gate

## Objective

v1.44 replaces the stopped v1.43 free-standing differential element with a
ground-backed patch branch.  A grounded wall is assigned to each 15 mm cell
to suppress substrate coupling, while the radiator and feed retain separate
input-impedance controls.  Only a passing physical 1x1 is allowed to open the
2x2 frozen-stimulus coupling and active-RL gate.

## Physical Models Evaluated

All accepted solves use copper conductors, RO5880, one physical 50 ohm port,
0.18 mm local feed/radiator mesh, 5% adaptive refinement, and the unchanged
10 GHz gate.  No AF, EEP, or postprocessed S matrix is used as a physical
pass.

1. A 1 mm above-aperture shallow cavity with direct probe tuning.
2. Patch width/length and orthogonal probe-offset retuning.
3. A ceramic-filled element-integrated coaxial transformer.
4. A flush grounded substrate wall with direct-probe retuning.
5. A slotted inset high-impedance feed tongue.
6. A split feed post containing a finite-Q 0.533 nH element-level inductor.

The high cavity wall created a low-radiation-resistance load.  Lowering the
wall to the patch plane recovered 97.9%-98.2% efficiency, but its resonant
input resistance remained about 13 ohm.  The integrated coax transformer and
finite-Q inductor did not preserve a stable reference load and either missed
the match or reduced efficiency below 95%.

## Key Results

| Candidate | Physical change | RL (dB) | Input impedance (ohm) | Efficiency | Delta S |
|---|---|---:|---:|---:|---:|
| run01 c00 | 1 mm cavity, old patch | 6.390 | 74.05-j61.93 | 97.97% | 0.00858 |
| run03 | near-resonant high-wall edge probe | 2.758 | 7.93+j4.27 | 96.38% | 0.00300 |
| run06 | 19.34 ohm ceramic coax section | 4.280 | 20.17+j38.89 | 95.88% | 0.00490 |
| run07 | closed-loop 6.73 ohm coax section | 2.521 | 8.25+j18.82 | 92.04% | 0.00520 |
| run08 c00 | flush wall, best direct-probe RL | 8.098 | 26.04-j19.53 | 98.23% | 0.00513 |
| run08 c02 | flush-wall resonant point | 4.628 | 13.02-j0.74 | 97.98% | 0.00057 |
| run10 c01 | 1.25 mm inset tongue | 6.430 | 71.33-j61.21 | 98.38% | 0.00308 |
| run12 | 0.533 nH, Q=50 element match | 8.026 | 92.07-j40.91 | 96.71% | 0.00437 |
| run13 | retuned finite-Q element match | 6.656 | 27.23+j31.33 | 93.04% | 0.00998 |

The best new physical 1x1 RL is 8.098 dB.  The preregistered 15 dB gate and
even the 10 dB stop line are both missed.  All listed solves are converged and
have valid topology, so this is a physical input-architecture failure rather
than an adaptive-mesh failure.

## Decision

The rectangular direct-probe grounded patch, high cavity wall, integrated
single-section coax transformer, inset tongue, and split-post lumped match are
stopped.  No v1.44 2x2 solve is authorized; consequently no 4x4/16x16 model,
EEP export, training label, or critic update is authorized.

The next hardware branch must change the input class rather than tune these
parameters again.  The preferred order is:

1. Aperture-coupled patch with a below-ground feed line and independent slot,
   stub, and patch-resonance controls.
2. True balanced differential radiator above a continuous reflector with a
   reference-plane-stable balun.
3. Only after a 15 dB 1x1 pass, add one x/y element-neutralization layer and
   run the frozen 2x2 active-RL test.

The v1.43 2x2 failure remains the active array-level evidence.  v1.44 does not
replace it because v1.44 did not pass the element gate.
