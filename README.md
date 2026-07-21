# DBF-Spare

Physics-gated sparse-array research code for a 16x16, 256-channel, 10 GHz
multi-task phased array. The project searches for the minimum feasible active
ratio while jointly controlling sidelobes, task isolation, main-lobe fidelity,
active return loss, and normalized power.

## Current baseline

The first versioned baseline is `v0.1.0-physics-gated` (2026-07-21). It records
the current model, scripts, compact evidence snapshots, and the reasons that
new engineering training labels remain locked.

- 1x1, 4x4, and 8x8 grounded-patch models pass their convergence and matched
  passive-return-loss gates.
- The 16x16 DDM S-matrix is reciprocal and passive, but Adaptive Pass 2 has
  `Delta S = 0.29495` and matched minimum passive RL of `5.13 dB`.
- The 256-port EEP superposition interface is numerically correct, but its
  current full-array source solution is not an accepted engineering baseline.
- AF/proxy optimization and the existing critic are retained for proposal and
  pretraining only. They are not HFSS-certified final results.

See [RESULTS_INDEX.md](RESULTS_INDEX.md) and
[baselines/2026-07-21/BASELINE.md](baselines/2026-07-21/BASELINE.md).

## Research pipeline

```text
scene and task encoding
  -> adaptive ratio and structured masks
  -> regional LCMV/SOCP task weights
  -> active-return and power projection
  -> EEP/full-wave gate
  -> residual critic ranking
  -> local mask search
  -> final HFSS validation
```

The search order is ratio `0.5 -> 0.6 -> 0.7 -> 0.8`; ratio `1.0` is a control.

## Repository policy

Source scripts and compact reference models are versioned. Raw AEDT result
trees, generated training datasets, checkpoints, field exports, and scratch
files are intentionally excluded because the local workspace is over 60 GiB.
Selected result evidence is copied into `baselines/<date>/snapshots` and
verified with SHA-256 hashes.

Rebuild the current result index from an intact local workspace with:

```powershell
python tools/build_result_index.py --tag 2026-07-21
```

The script never changes source HFSS results.

## Engineering gates

- Two consecutive adaptive passes with `Delta S <= 0.05`.
- S-matrix reciprocity error `<= 1e-4` and maximum singular value `<= 1.001`.
- Matched passive RL and active-port/total reflected-power RL `>= 10 dB`.
- PSLL screening `<= 0 dB`, then targets of `-3 dB` and `-6 dB`.
- Nearest-target isolation `>= 25 dB`; local +/-5 degree isolation `>= 20 dB`.
- Weakest-target gain loss `<= 0.5 dB`; beam imbalance `<= 3 dB`.

No AF, proxy, EEP, or unconverged S-matrix metric is presented as a final
full-wave engineering claim.
