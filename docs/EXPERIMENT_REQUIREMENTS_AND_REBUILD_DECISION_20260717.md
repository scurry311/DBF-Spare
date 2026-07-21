# 16x16 Sparse Multi-Beam Array: Experiment Requirements and Rebuild Decision

## Fixed research requirements

- Array: 16x16, 256 independently controllable RF channels, nominal 0.5-lambda pitch at 10 GHz.
- Search: ratio 0.5 -> 0.6 -> 0.7 -> 0.8; return the minimum feasible ratio. Ratio 1.0 is a control only.
- Tasks: K=1/2/4/6, including K=6, large scan angles, and closely spaced targets.
- Direction-pattern gate: full-wave PSLL <= 0 dB for screening, then optimize toward -3 dB and -6 dB.
- Isolation gate: nearest-target isolation >= 25 dB and local +/-5 degree isolation >= 20 dB.
- Main-lobe gate: weakest target gain loss <= 0.5 dB, beam imbalance <= 3 dB, and pointing error within the sampling grid.
- RF gate: every active port and total reflected-power return loss >= 10 dB.
- Power comparison: normalize all candidates to equal weakest-target gain or equal EIRP.
- Data integrity: only complete full-wave results; deduplicate by scene, mask, and weights; split by sample_index with no scene leakage.
- Optimization order: hardware feasibility -> EEP operator -> regional LCMV/SOCP -> structured mask search -> HFSS validation -> residual critic.

## Why the original dipole route was stopped

The original simplified PEC dipole array had poor baseline passive matching and unstable embedded-feed CAD. Deeper matching/decoupling cascades did not create a meaningful all-port 10 dB feasible set, and several 4x4/8x8 variants failed mesh or port-topology validation. Continuing to tune end loads or network depth would not address the physical bottleneck.

## Selected rebuild

The replacement is a shared-ground rectangular microstrip-patch URA with one shielded probe port per element:

- RO5880-like substrate: er=2.2, tan(delta)=0.0009, h=0.787 mm.
- Pitch: 15 mm in x and y.
- Patch: 11.8 mm x 9.35 mm.
- Feed offset: 3.5 mm along the resonant dimension.
- Explicit probe and coax shield geometry with a valid two-conductor lumped port.
- Embedded finite-Q series match: L=0.533 nH, Q=50, represented explicitly as a circuit-level S-matrix cascade.

This topology preserves the existing 256-entry mask, complex weight, EEP, S256, and critic interfaces. It is materially easier to replicate than the balanced-dipole feed because each element uses the same planar radiator and local shielded feed.

## Stage gates

1. 1x1: no port-topology errors, Delta S <= 0.05, and matched passive RL >= 10 dB.
2. 4x4: valid converged S16, class-wise corner/edge/interior RL, and all-port matched passive RL >= 10 dB.
3. 16x16: valid converged S256, then all 2400 existing K/ratio/scan scenarios must be re-evaluated for active RL.
4. EEP/HFSS labels and model training may start only after the RF gate has a nonzero engineering-feasible set.

No AF, EEP, circuit projection, or unconverged S-matrix result may be reported as a full-wave engineering pass.

## Results obtained on 2026-07-17

- Coax-fed 1x1: converged, Delta S=0.0190, matched RL=31.24 dB.
- Coax-fed 4x4: converged, Delta S=0.0355, all 16 matched ports pass 10 dB, minimum RL=14.77 dB.
- Direct-reference-plane 1x1: converged, Delta S=0.0234, matched RL=15.96 dB.
- Direct-reference-plane 4x4: converged, Delta S=0.0367, all 16 matched ports pass 10 dB, minimum RL=11.87 dB.
- The direct model reduced maximum matrix size from 1,147,536 to 323,974 and reduced the 4x4 result directory from about 0.76 GB to 0.22 GB. It is the selected training model.

The workstation has 23.6 GB RAM and only 18.4 GB free on the HFSS drive. A direct monolithic 256-port adaptive solution was therefore not launched. A reciprocal passive S256 proxy was built from the converged local S16 impedance kernel and is explicitly labeled as a proxy, not full-array HFSS.

## Proxy pretraining and projection results

- Local-kernel S256 proxy: reciprocity error 3.6e-16; maximum singular value 0.916.
- Original 2400 cases: strict proxy active-RL plus total-RL pass rate 1.42%.
- Five-seed proxy critic: mean AUROC 0.995, AUPRC 0.766, ECE 0.063. This checkpoint is for pretraining and rejection only.
- Strong active-RL projection with -12 dB channel pruning reduced mean active ratio from 0.750 to 0.639.
- Strict proxy RF pass rate increased to 39.25%; K=6 pass rate increased to 30.33%.
- Target-point complex response error remained below 3e-10.
- Paired combined-AF check: mean PSLL improved by 2.24 dB, but the 95th-percentile PSLL change was +7.17 dB and only 10/2400 cases passed both proxy RF and combined-AF screening.

## Current decision

The rebuilt patch array is a better and more tractable physical basis than the original dipole array. The RF feasibility critic and active-return projection are useful, but their outputs are not final training labels. The next training stage must reconstruct per-task weights, repeat regional LCMV/ZF null broadening and projected PSLL optimization, and then validate a small oracle-selected set with true full-wave HFSS/DDM or HPC resources.
