# v1.39 Physical 2x2 Active-RL Stop-Gate Baseline

## Evidence Level

Physical 10 GHz HFSS direct S4 from a four-port 2x2 true-differential array,
with frozen K=2/K=4 replay, real K=6 task weights, coherent efficiency
calibration, and active-RL modal/Jacobian diagnosis.

## Result

- Final Delta S: `0.0045651` in two adaptive passes.
- Maximum tetrahedra: `157,557`; peak solver memory: `8.173 GiB`.
- Reciprocity error: `1.08e-16`; passivity sigma: `0.4151`.
- Minimum passive RL: `17.162 dB`; conductor warning count: `0`.
- Minimum active RL: `-6.032 dB`; minimum total RL: `7.637 dB`.
- Minimum system efficiency: `83.14%`.
- Active-RL failures below 10 dB: `408/484`.
- K=2/K=4/K=6 minimum active RL: `3.286/-2.521/-6.032 dB`.

The valid physical S4 passes all numerical and passive gates but fails the
10 dB active-RL stop line. Y-neighbor coupling and the even/x-odd modes are the
dominant repeatable failure mechanisms.

## Gate State

DDM, three-frequency, 4x4/16x16, EEP, labels, and critic training are locked.
Only frozen-S4 task-weight feasibility projection or one local x/y modal
correction study is authorized next.

