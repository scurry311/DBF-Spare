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

