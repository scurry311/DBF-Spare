# v1.27 Aperture-Coupled Radiator/Input Stop Gate

## Scope

This stage replaces the stopped v1.26 feed-inset-only branch with a new
physical 1x1 radiator/input model. The model combines a dual-slot patch, the
central tongue current path, a ground coupling aperture, a bottom microstrip
open stub, and a local horizontal GSG launch. The array spacing remains 15 mm.

The work is deliberately limited to a 10 GHz 1x1 prerequisite. A 2x2 model is
not authorized until at least one physical candidate reaches 10 dB passive
return loss. Consequently this stage contains no nearest-neighbor Sij, active
RL, EEP, 4x4, 16x16, label, or critic evidence.

## Numerical Qualification

The first valid center solve used four deterministic 0.18 mm local mesh
regions around the slots, coupling aperture, bottom aperture, and launch. It
completed with final Delta S 0.01274 and zero topology warnings, but its
10 GHz passive RL was only 0.76 dB.

Three preregistered six-candidate screens were then completed. A retained mesh
volume overlap in the first run11 attempt was diagnosed before solving and was
fixed by partitioning the two same-material local mesh regions. No copper,
dielectric, slot, aperture, or feed dimension changed in the corrected run12.

| Screen | Candidates | Delta S pass | 10 dB RL pass | Best RL |
|---|---:|---:|---:|---:|
| Joint slot/aperture/feed screen | 6 | 6 | 0 | 4.148 dB |
| Complex-impedance targeted screen | 6 | 6 | 0 | 2.397 dB |
| Tongue-resonance recovery | 6 | 6 | 0 | 4.036 dB |
| **Combined** | **18** | **18** | **0** | **4.148 dB** |

The combined impedance envelope is broad: resistance spans 6.04 to 476.87
ohm and reactance spans -278.36 to +291.10 ohm. The geometry therefore changes
S11 substantially, but the accessible trajectory does not approach the 50-ohm
origin closely enough to pass the engineering gate.

## Physical Interpretation

The best candidate is `joint_b_long_high`, with input impedance
19.61 - j39.11 ohm, |S11| 0.6203, and 4.148 dB return loss.

The controlled tongue-length comparison is more decisive. Keeping the E-like
aperture and feed fixed while increasing slot length from 3.0 to 3.4 and 3.65
mm changes resistance by at most 0.38 ohm, reactance by at most 0.20 ohm, and
RL by at most 0.008 dB. In this input configuration, the tested tongue path is
not an independent 10 GHz matching control.

Shortening the otherwise identical E-case stub from 4.0 to 2.8 mm moves the
input from 58.54 - j100.84 ohm to 239.65 + j262.97 ohm. The stub reverses the
reactance only by traversing a high-resistance resonance. It cannot separately
set resistance and reactance.

An ideal, lossless cancellation of candidate O's reactance would produce a
27.33 dB diagnostic upper bound. This is not a physical result. It shows that
the next feed must add a genuinely independent, manufacturable impedance
transformation rather than another correlated slot/stub adjustment.

## Resource Audit

All 18 valid candidates finished without a memory abort. Several solves made
short excursions below 3 GiB free memory; the minimum recorded value was
2.365 GiB. The 1x1 model is therefore already near the workstation resource
limit and does not authorize a direct 2x2 expansion with this mesh strategy.

## Decision

Stop the current aperture/tongue/open-stub topology. The 10 dB 1x1 prerequisite
failed 0/18, so three-frequency 1x1 verification and physical 2x2 S4 are not
authorized. Without S4, no claim is made that nearest-neighbor coupling was
improved.

Keep 4x4, 16x16, EEP export, full-wave labels, and critic training locked. The
next hardware branch should use a true balanced differential or dual-resonant
input with an independent impedance-transforming degree of freedom. It must
first pass 1x1 three-frequency matching and efficiency gates before any 2x2
coupling Jacobian is attempted.
