# 9.96 GHz Perturbed-Operator HFSS Smoke

- Physical operator gate: PASS.
- Resource minima: 0.796 GB free RAM, 10.666 GB commit headroom, 31.966 GB free disk.
- Complete EEP ports/grid: 256/16471 directions.
- Frozen candidate strict pass: operator only 5/20; operator plus source perturbation 4/20.
- All 16 failures are active-RL failures; the frozen pattern gates remain satisfied.
- Direct HFSS agreement: 2/2, no-scale NMSE max 5.289e-12.

## Decision

The physical 9.96 GHz operator is valid for development evaluation. Critic retraining remains locked.
The next algorithmic step is joint nominal/9.96 GHz active-RL projection before another HFSS corner.
