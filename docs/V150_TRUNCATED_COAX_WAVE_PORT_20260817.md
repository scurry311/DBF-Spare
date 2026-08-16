# v1.50 Truncated-Coax Wave-Port Stop Gate

## Scope

v1.50 implements scheme A as a clean external truncated-coax TEM launch with
a finite ground transition. The v1.49 radiator, SIW fence, feed location,
materials, three frequencies, solver settings, and downstream locks remain
frozen. This stage authorizes one build smoke and one nominal broadside
continuation only; it does not authorize DOE, finite arrays, EEP export,
training labels, or critic retraining.

## Implementation And TDD

- Replaced the former lumped-port sheet with one 50 ohm modal wave port on the
  external coax dielectric reference plane.
- Built exact inner conductor, dielectric annulus, outer conductor, ground
  aperture, and a single outer-conductor/ground Boolean union.
- Preserved the 0.30 mm local launch mesh but confined it to `FeedProbe` and
  `CoaxDielectric`. Applying it to the united ground body caused the sealed
  run02 memory abort.
- Made the final stage decision require both numerical and physical nominal
  gates, so a memory, mesh, or export failure cannot be reported as a pass.
- Added regression tests for the wave-port topology, forbidden legacy port
  objects, memory-safe mesh scope, immutable controls, and evidence gates.

Verification before HFSS:

- v1.50 tests: 15/15 passed.
- v1.49 regression tests: 43/43 passed.
- Python compilation and `git diff --check`: passed.

## HFSS Evidence

### Build smoke: run03

The immutable build run passed `ValidateDesign` and the model inventory gate:

| Metric | Result |
|---|---:|
| AEDT return code | 0 |
| Objects / boundaries / excitation modes | 65 / 16 / 3 |
| Feed excitation | One modal Wave Port |
| Legacy `PortSheet` or lumped port | Absent |
| Critical build warning hits | 0 |
| Minimum free host memory | 13.093 GiB |
| Build gate | PASS |

Evidence: `hfss_outputs/v150_truncated_coax_wave_port_20260817_run03/`.

### Single continuation: run04

The hash-bound continuation completed in 405.47 s. Touchstone, source names,
the solved project, convergence profile, and run audit were exported. It used
the same frozen geometry at 9.96, 10.00, and 10.04 GHz.

| Metric | Result | Gate |
|---|---:|---:|
| Final Delta S | 0.0177504 | <= 0.05 |
| Maximum tetrahedra | 113,872 | <= 120,000 |
| Minimum free host memory | 3.666 GiB | >= 3 GiB |
| Small-segment records | 104 | 0 |
| Delta-S comparisons / adaptive passes | 6 / 7 | Complete |

The local-mesh correction therefore solved the run02 resource failure and the
resource sub-gate passed. The full numerical gate still fails because the
small-segment count is nonzero. Those records occur on `MainSubstrate` and
selected `SIWVia_*` bodies, not on the new coax/wave-port objects.

A read-only post-run audit of the sealed S3P gives maximum reciprocity error
`5.75e-15` and maximum singular value `0.999535`; both pass the newly explicit
`1e-4` reciprocity and `1.001` passivity checks. This audit does not alter the
sealed run04 files or rescue the failed small-segment gate.

## Physical Result

| Frequency | Passive/active RL | Input impedance | Accepted-power efficiency |
|---:|---:|---:|---:|
| 9.96 GHz | 0.937 dB | 26.23 + j145.70 ohm | 55.30% |
| 10.00 GHz | 1.110 dB | 60.50 + j203.51 ohm | 52.46% |
| 10.04 GHz | 1.328 dB | 245.44 + j314.77 ohm | 49.47% |

The requested broadside passive RL is at least 15 dB and accepted-power
efficiency is at least 97%. Both physical gates fail by a large margin. The
large positive input reactance and rapid impedance variation show that the
clean TEM reference plane exposes a strongly off-resonant frozen antenna/feed
load; further port-sheet or mesh tuning cannot supply the missing impedance
transformation.

## Decision

`STOP_AFTER_NOMINAL_DIAGNOSTIC_FAILURE`.

The CAD/build gate and runtime resource guard pass. The complete numerical
gate and physical gate fail, so scheme A with the frozen v1.49
radiator/SIW/feed location is not a viable engineering element. Do not start
the 16-sample DOE, 1x1/2x2/4x4/16x16 builds, EEP export, label generation, or
critic training from v1.50.

The sealed run04 summary contains a historical metadata error:
`periodic_physical_gate_evaluated=true` refers only to the nominal broadside
three-frequency check, not the full 45-state periodic gate. Current control
code now reports `nominal_physical_gate_evaluated=true` and keeps
`periodic_physical_gate_evaluated=false`; run04 remains immutable.

The next physical branch must introduce an independent input-impedance degree
of freedom while retaining the verified external coax reference plane. The
minimum useful experiment is a periodic-cell input redesign that jointly:

1. removes SIW-via/substrate sliver contacts so the small-segment count is zero;
2. tunes probe inset/transition plus one real impedance-transforming feature
   such as an aperture-coupled or dual-resonant feed;
3. uses a 10 GHz circuit/modal pre-screen before at most 3-5 three-frequency
   HFSS Pareto candidates;
4. requires nominal RL >= 15 dB and efficiency >= 97% before any scan DOE.

Evidence: `hfss_outputs/v150_truncated_coax_wave_port_20260817_run04/`.
