# v1.49 Nominal Periodic Continuation

Date: 2026-08-16

## Scope

This record covers one nominal 9.8-10.2 GHz periodic-unit full-wave attempt.
It does not authorize DOE, finite arrays, EEP export, labels, or critic
training.

## Evidence Chain

- Run08 copied the sealed run06 build project and started HFSS, but solver
  validation rejected three solid intersections. It produced no Touchstone.
- Run09 removed the solid overlaps. AEDT 2023.1 then rejected a multi-face
  Primary assignment because the boundary accepted exactly one selection.
- Run10 split the three material layers into six single-face linked-boundary
  pairs. Native CAD build returned 0, `ValidateDesign` returned 1, and the
  inventory contained 67 objects, 16 boundaries, and 3 excitation modes.
- Run10 was sealed with 30 verified files. Its project SHA-256 is
  `1aa8dd8a8b9ba471e753b4ae1f01abdbb4b0dd1a05228475bf66259e9442d0bc`.
- Run11 copied that exact project under a single-use solve authorization and
  launched the nominal sweep with 13.479 GiB free memory.

## Run11 Outcome

- AEDT return code: 1.
- Memory guard: triggered.
- Minimum free memory: 2.802 GiB.
- Elapsed time: 320.838 s.
- Initial/manual-refined mesh: approximately 356,493 tetrahedra.
- Matching-boundary final MCT diagnostic: 0 unmatched points.
- Lumped-port diagnostic: only one conductor touches `FeedPort`.
- Touchstone export: absent.
- Source-name export: absent.
- HFSS antenna metrics: unavailable.

The attempt is a resource-and-port-definition failure, not a measured antenna
performance failure. No result from run11 may enter a baseline metric table or
training dataset.

## Next Permitted Action

Create a new immutable build run after both changes are preregistered:

1. Extend the lumped-port sheet/integration line slightly into the coax outer
   conductor within the port plane, then require a two-conductor port check.
2. Replace the 356k-tetrahedron manual mesh profile with a memory-safe local
   profile. Keep refinement local to the launch, patch edges, and via surfaces;
   preserve `MaxDeltaS=0.05` and the 3 GiB runtime abort gate.
3. Run build plus `ValidateDesign` only. If valid, launch one new continuation
   solve. Do not reuse run11's consumed authorization.

## Port And Mesh Repair Continuations

The repair sequence was executed in new immutable runs; no failed result was
overwritten:

- `run13` reduced the manual mesh to 137,886 tetrahedra but still reached the
  memory stop and retained the one-conductor port warning.
- `run15` reduced the solved mesh to 66,525 tetrahedra and stayed above the
  runtime memory floor. Its finite-overlap port removed the conductor-count
  warning but introduced 108 duplicated small-segment hits and a near-short.
- `run17` tested an annular port and was stopped after HFSS reported too many
  touching conductors. `run18` and `run19` are build-only failures retained for
  provenance; neither produced a physical result.
- `run20` uses a radial model-sheet lumped port between the coax inner and
  outer conductors. It passed `ValidateDesign` with zero critical build warning
  and was sealed as the trusted source for one continuation.
- `run21` is that single-use continuation. The first launch was blocked at
  11.83 GiB without consuming authorization; the later launch began with
  13.078 GiB and completed normally.

## Run21 Verified Outcome

| Metric | Result | Gate decision |
|---|---:|---|
| HFSS return code / memory abort | 0 / false | Passed |
| Solved tetrahedra | 66,881 | Memory-safe |
| Maximum adaptive tetrahedra | 83,938 | 76.5% below run11 |
| Peak solver memory | 6.402 GiB | Diagnostic |
| Minimum host free memory | 4.865 GiB | Passed 3 GiB stop |
| Adaptive passes / final Delta S | 3 / 0.006485 | Passed 0.05 |
| Port conductor-count warning | None | Passed |
| Small-segment flags | 54 | Failed zero-warning gate |
| Worst three-frequency passive RL | 0.00617 dB | Failed 15 dB |
| 10 GHz input impedance | 0.0192+j11.2348 ohm | Near-short |
| 10 GHz accepted power | 0.1458% | Failed physical intent |
| Periodic DOE and downstream stages | Locked | Required |

The 54 small-segment flags are localized to `CoaxDielectric` (38), `CoaxOuter`
(15), and `FeedProbe` (1). Power accounting and convergence evidence are
internally consistent, so the poor RL is not being reported as an incomplete
solver export. It is a physical feed/input-topology failure in the present CAD.

## Decision

The immediate request is complete in the narrow numerical sense: the lumped
port no longer produces a one- or multi-conductor contact warning, the mesh is
well below the approximately 356k-tetrahedron attempt, the solve fits the host
memory guard, and a new single-use continuation completed and was sealed.

The antenna model has not passed its physical gate. The next branch must repair
the coax/dielectric/ground intersection and input reference geometry, preferably
with a clean truncated coax wave-port/reference face or an equivalently
auditable transition, then retune the driven input. Additional mesh coarsening
or continuation runs on this near-short geometry are not authorized.

Run21 is finalized with 82 SHA-256 entries. Its stage decision is
`STOP_AFTER_NOMINAL_DIAGNOSTIC_FAILURE`; periodic DOE, finite arrays, EEP,
training labels, and critic retraining remain locked.
