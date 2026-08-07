# v1.39 Physical 2x2 Differential Array Stop Gate

## Scope

v1.39 copied the independently validated v1.38 true-differential element into
a physical 2x2 array with 15 mm x/y spacing and four logical ports ordered as
`P00, P10, P01, P11`. The test replays the 285 frozen K=2/K=4 excitations,
adds task and combined weights from 20 existing K=6 supported scenes, and adds
canonical even/odd modes and 48 degree phase gradients. No weights were
invented from target directions.

The first build failed because a many-body substrate Boolean did not
regenerate. The replacement uses deterministic touching same-material
partitions and changes no physical dimensions. Two subsequent 0.10 mm direct
attempts exceeded the workstation memory limit before adaptive pass 1. A
hash-locked numerical amendment changed only the eight primary-sheet local
mesh operations to 0.18 mm and adaptive refinement to 5%. Geometry, direct
solver, `MaxDeltaS=0.05`, ports, materials, and engineering gates stayed fixed.

## Valid HFSS Result

The amended 10 GHz direct case converged in two passes:

| Metric | Result | Gate |
|---|---:|---:|
| Final Delta S | 0.004565 | <= 0.05 |
| Maximum tetrahedra | 157,557 | Reported |
| Peak solver memory | 8.173 GiB | Reported |
| Minimum free memory | 7.075 GiB | > 3 GiB |
| Reciprocity error | 1.08e-16 | <= 1e-4 |
| Passivity sigma | 0.4151 | <= 1.001 |
| Minimum passive RL | 17.162 dB | >= 12 dB |
| Conductor small-segment messages | 0 | 0 |
| Minimum active RL | -6.032 dB | >= 11 dB design; 10 dB stop |
| Minimum total RL | 7.637 dB | >= 11 dB |
| Minimum system efficiency | 83.14% | >= 95% |

HFSS exported ports as `P00, P01, P10, P11`; analysis explicitly reordered the
S4 to the preregistered physical order before any replay. Sixteen coherent
HFSS efficiency calibration states reconstruct the 4x4 radiated-power
operator, so system efficiency includes mismatch and mutual-coupling loss.

## Failure Diagnosis

Of 484 frozen 10 GHz excitations, 408 (84.30%) fall below 10 dB active RL and
only 59 (12.19%) reach the 11 dB design line. K=2, K=4, and K=6 minima are
3.286, -2.521, and -6.032 dB respectively.

The physical coupling levels are approximately -15.0 dB in x, -13.1 dB in y,
and -23.9 dB diagonally. The x-odd and even modal RL values are only 7.64 and
9.46 dB. Across failed excitations, y-neighbor terms have the largest average
aligned reflected-wave contribution (0.1178) and the largest mean active-RL
Jacobian magnitude (24.11 dB per unit S perturbation). The worst K=6 corner
case produces a reflected wave at `P10` equal to 2.003 times its significant
incident wave.

The hardware therefore has good single-port Sii but does not match the
coherent array modes used by the task weights. This is a physical active-match
failure, not a critic-label problem.

## Decision

Independent repeat, DDM, three-frequency solving, 4x4/16x16 expansion, EEP
export, training-label generation, and critic retraining remain locked.

The next low-cost authorized step is an S4-aware fixed-mask task-weight
projection against this frozen operator. It must first demonstrate a nonempty
10 GHz 11 dB reserve while retaining task equalities and pattern margins. If
that oracle remains empty, use one local x/y even-odd modal correction block;
do not continue passive-S11 tuning or launch more HFSS cases.

