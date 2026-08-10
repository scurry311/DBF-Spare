# v1.45/v1.46 Element Topology Stop-Gate Report

## Objective

Replace the stopped direct-probe cavity element with either a standard
aperture-coupled patch or a true balanced differential radiator over a
continuous reflecting ground, then repeat the physical 1x1 to 2x2 gate.

## What Was Built

### Standard Aperture-Coupled Patch

The v1.45 CAD has two dielectric layers separated by a continuous finite-
conductivity ground.  A complete top patch is coupled through one transverse
ground aperture by a lower microstrip and open stub.  Patch length, aperture
size, and stub length are independent.  A later diagnostic adds exactly one
integrated impedance-transforming section; no second stage is used.

The old v1.27 dual-slot radiator and coplanar GSG launch are not reused.

### Ground-Backed Differential Patch

The v1.46 CAD has two symmetric top patch halves, two physical probes, two
small clearances in an otherwise continuous ground, and one bottom
differential lumped port.  One external array channel therefore still maps to
one physical element port and one complex beamforming weight.

## Resource Correction

Applying 0.18 mm to every large conductor generated 216k-313k tetrahedra and
triggered the 3 GiB memory stop.  The trusted replacement keeps the same
0.18 mm local scale at the port and uses field-adaptive refinement elsewhere.
Complete cases use at most 9,151 tetrahedra and retain at least 13.565 GiB
free memory.  No concurrent AEDT instances were used.

## Measured Result

All 37 retained cases are physical 10 GHz HFSS results with valid S1 output,
final Delta S at or below 0.05, and zero port-topology warnings.

| Branch | Cases | Best passive RL | Best impedance | Efficiency at best |
|---|---:|---:|---:|---:|
| Aperture, direct | 13 | 3.200 dB | 11.44-j24.78 ohm | 99.47% |
| Aperture, one transformer | 14 | 0.379 dB | 1.22+j16.84 ohm | 92.61% |
| Ground-backed differential | 10 | 0.630 dB | 23.69+j172.24 ohm | 98.16% |

The aperture family remains low-resistance.  Enlarging the aperture does not
form a 50 ohm feasible region, and the single physical transformer worsens
both matching and efficiency.  The differential family is dominated by probe
inductance; lowering the profile reduces reactance but collapses resistance
and efficiency rather than producing a match.

## Gate Decision

- Physical 1x1 RL >= 15 dB: 0/37, failed.
- Physical 2x2: not authorized and not executed.
- 4x4/16x16, EEP, labels, and critic: locked.
- No result in this stage is a training label or an array-level engineering
  conclusion.

The appropriate next move is not another local geometry sweep.  Use a
published or commercial 10 GHz array element with a validated 50 ohm launch,
then preserve the current mask/weight/S-parameter interfaces around that
cell.  A new cell must pass a frequency sweep and independent 1x1 repeat
before any 2x2 active-RL experiment.
