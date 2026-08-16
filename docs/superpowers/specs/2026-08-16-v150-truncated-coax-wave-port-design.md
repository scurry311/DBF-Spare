# v1.50 Truncated-Coax Wave-Port Design

**Status:** Approved design direction; implementation remains gated by this
written specification.

**Objective:** Replace the invalid radial lumped-port launch used by v1.49
with an auditable external-boundary coaxial TEM wave port and a continuous
finite-thickness ground transition, while holding the radiator, SIW cavity,
feed location, solver thresholds, and downstream locks fixed.

## Evidence And Root Cause

The sealed v1.49 run21 completed with final Delta S `0.006485`, 66,881 solved
tetrahedra, 6.402 GiB peak solver memory, and no lumped-port conductor-count
warning. Its three-frequency return loss was only `0.00617-0.00658 dB`, with
input impedance near `0.019+j11.2 ohm`. The port was a radial vertical sheet
whose plane included the coax propagation direction instead of a transverse
coax cross-section. It therefore did not represent a clean TEM reference
plane. The 24-sided coax solids also generated most of the 54 remaining
small-segment flags.

The v1.49 run directories and results are immutable. v1.50 is a new physical
branch and may not backfill or relabel any v1.49 output.

## Frozen Scope

The following values remain identical to the v1.49 nominal geometry:

- 15 mm square period;
- driven and stacked patch dimensions and layer heights;
- SIW via diameter, pitch, inset, and all via centers;
- feed center at `(0.0, -2.45) mm`;
- feed-probe radius `0.25 mm` and its patch connection;
- RO5880 material, copper conductivity, and loss tangent;
- 9.8-10.2 GHz sweep and 9.96/10.00/10.04 GHz checks;
- `MaxDeltaS=0.05`, 5% adaptive refinement, and serial execution;
- the 13 GiB launch threshold and 3 GiB runtime-abort threshold;
- all DOE, finite-array, EEP, label-generation, and critic-training locks.

Only the coaxial reference section, wave-port boundary, and ground transition
may change in this branch.

## Selected Topology

The coax is terminated at `z=-1.0 mm`. Its bottom dielectric face is an
external solution boundary and carries one modal wave port. The port normal is
parallel to `+Z`, and its integration line runs radially from the center
conductor to the outer conductor. The raw reference plane is the physical
bottom face; de-embedding is disabled for the first smoke so that no hidden
reference-plane transformation can mask the launch behavior.

The vacuum-filled coax dimensions are:

- center-conductor radius `a=0.25 mm`;
- outer-conductor inner radius `b=0.575 mm`;
- outer-conductor outer radius `1.05 mm`;
- coax length `1.0 mm`.

The ideal vacuum-coax estimate `60 ln(b/a)` is `49.97 ohm`. HFSS must
still solve the actual modal impedance; the analytic value is only a static
geometry sanity check.

All coax cylinders use analytic circular geometry (`NumSides=0`). The ground
is a `0.035 mm` thick copper solid from `z=-0.035 mm` to `z=0`. It has one
coaxial aperture of radius `0.575 mm`. The outer conductor extends through the
ground thickness and is united with the ground solid into one connected copper
body. The dielectric annulus and center conductor remain disjoint from that
body. No radial `PortSheet`, lumped port, contact overlap, or artificial axial
port height remains.

## CAD And Boundary Audit

The generated HFSS model must expose these independently checkable facts:

- one connected `GroundOuterConductor` copper solid;
- one `FeedProbe` copper solid;
- one coax dielectric annulus ending on the bottom reference plane;
- one wave-port face selected from the dielectric annulus bottom face;
- one `FeedPort` wave-port mode and two `FloquetTop` modes;
- no object named `PortSheet` and no lumped-port assignment;
- no solid overlap between feed probe, coax dielectric, or ground/outer body;
- exact circular coax and aperture construction rather than faceted circles;
- unchanged patch, SIW, lattice, and feed-center inventory.

The build smoke runs `ValidateDesign` but performs no solve. Any ambiguous face
selection, missing TEM mode, geometry intersection, or port-conductor warning
stops the branch before continuation creation. Small-segment evidence is
evaluated after meshing in the nominal solve.

## Mesh And Resource Design

The v1.49 memory-safe policy is retained:

- feed/coax interface maximum length `0.30 mm`;
- patch-edge maximum length `0.55 mm`;
- SIW-via maximum length `0.50 mm`;
- no global `0.18 mm` mesh;
- no `0.10 mm` volumetric port-sheet mesh;
- no conductor-volume refinement.

The coax dielectric receives a surface-focused local operation at `0.30 mm`;
it must not force refinement throughout the cell. A nominal
solve is rejected as non-memory-safe if it exceeds 120,000 adaptive
tetrahedra, reaches less than 3 GiB free host memory, or leaves an AEDT process
after termination.

## Execution And Evidence Flow

1. Create a new v1.50 preregistration and immutable output prefix.
2. Generate and unit-test the VBS without launching AEDT.
3. Allocate a build-only run and perform import, inventory, port-face, and
   `ValidateDesign` audits.
4. Seal the passing build with SHA-256 evidence.
5. Allocate one new hash-bound, single-use continuation from that build.
6. Launch only when at least 13 GiB is free and no AEDT/HFSS process exists.
7. Solve one broadside 9.8-10.2 GHz sweep and export the feed plus two Floquet
   modes.
8. Parse convergence, memory, mesh, warning, impedance, reflection, accepted
   power, and Floquet power evidence.
9. Freeze the result before deciding whether any periodic DOE is permitted.

Failed build or solve runs remain immutable and are indexed as diagnostic
evidence. A retry always receives a new run number and authorization.

## Test Strategy

Control-plane tests are written before production changes. They verify:

- the v1.50 config freezes all radiator and SIW values against v1.49;
- the generated VBS contains an external transverse wave port and no lumped
  port or `PortSheet`;
- analytic-circle, finite-ground, aperture, and ground/outer union commands;
- deterministic face selection at the bottom coax plane;
- inventory and excitation-mode expectations;
- unchanged memory guards, single-use authorization, and immutable finalization;
- analysis rejects incomplete source modes, hashes, convergence, or power;
- downstream stages remain locked unless every nominal gate passes.

The complete v1.49 test suite remains green to prove the new branch does not
rewrite historical behavior.

## Gates And Decisions

The build gate requires `ValidateDesign=1`, correct inventory, exactly one feed
wave mode, two Floquet modes, and zero critical topology warning.

The nominal numerical gate requires:

- HFSS return code zero and no memory abort;
- final Delta S at most `0.05`;
- at most 120,000 adaptive tetrahedra;
- zero critical port/contact and small-segment warning;
- complete, hash-verified Touchstone and source-mode exports;
- reciprocal/passive/power-consistent exported behavior where applicable.

The nominal physical gate requires all three frequencies to meet:

- passive/active broadside RL at least `15 dB`;
- accepted-power efficiency at least `97%`;
- physically plausible finite input impedance and power balance.

Passing the numerical gate alone proves only that the new reference plane is
credible. It does not prove that the antenna is matched. The periodic DOE may
be considered only after both nominal gates pass; 1x1, 2x2, 4x4, 16x16, EEP,
labels, and critic retraining remain separately locked.

If the clean wave-port result converges but remains below 15 dB RL, the port
topology is frozen and the next branch must tune the antenna input match, not
the mesh. If the result remains near a short or open, the ground union, annular
dielectric face, and wave-port orientation are audited before any further
solve. No learning model may compensate for either failure.

## Deliverables

- v1.50 preregistered configuration;
- parameterized builder, runner, analyzer, and finalizer;
- unit and regression tests;
- build inventory and port-face audit;
- single-use continuation evidence;
- nominal three-frequency S-parameter and power CSV;
- convergence, tetrahedron, memory, and warning summary;
- SHA-256 manifest, stage report, result-index update, and Git commits.
