# v1.42/v1.43 Mapped-EEP Joint Oracle And Input-Redesign Stop Gate

## Evidence Level

v1.42 uses the frozen v1.41 finite-Q corrected circuit S4 and antenna-incident
map tiled over the nominal 256-port EEP. It is an algorithmic circuit/mapped-
EEP oracle, not integrated 16x16 HFSS evidence. v1.43 adds physical 10 GHz
HFSS evidence from one y-balanced 1x1 and one four-port 2x2 array.

## Corrected-S4 Mapped-EEP Oracle

- Frozen scenes: `20` (`K=2/4/6`: `7/6/7`).
- Masks, ratios, target directions, gates, and correction-network parameters
  remain frozen.
- Seed families: direct, block least-squares precompensation, and three blends.
- Final acceptance uses complete dense local-5deg constraints without slack.
- Engineering strict passes: `5/20`.
- 11 dB reserve passes by K: `4/7`, `1/6`, `0/7`.

The preregistered K=6 nonempty requirement fails. The v1.41 correction-network
topology is stopped. HFSS scaling, EEP labels, and critic training remain
locked.

## Input-Impedance Target

The closest symmetric modal S4 satisfying 11 dB active RL for 1,566 frozen
stimuli requires modal impedances near:

- even: `45.74-j8.14 ohm`;
- x-odd: `40.97-j6.35 ohm`;
- y-odd: `49.56-j9.14 ohm`;
- checker: `41.38-j4.48 ohm`.

The physical candidate mirrors the secondary resonant branch about the
element y center and retains the via/pad pair as an independent self-impedance
transformer. It adds no external correction stage.

## Physical Results

The 1x1 input prescreen passes with final Delta S `0.003179`, passive RL
`15.56 dB`, and input impedance `37.67-j7.97 ohm`, only `6.72 ohm` from the
synthesized self-impedance target. Its reported 100.54% radiation efficiency
is retained as a 0.54% HFSS power-balance error.

The strict 2x2 solves at 0.10 and 0.12 mm are preserved as memory aborts. The
final preregistered 0.18 mm/5% solve succeeds with two converged passes, final
Delta S `0.006754`, peak memory `9.29 GiB`, reciprocity error `0`, and valid
S4. Compared with v1.39:

| Metric | v1.39 | v1.43 |
|---|---:|---:|
| Passive RL | 17.162 dB | 15.004 dB |
| Worst active RL | -6.032 dB | -6.160 dB |
| Worst total RL | 7.637 dB | 6.936 dB |
| Minimum system efficiency | 83.14% | 80.15% |
| Mean y-neighbor coupling | -13.107 dB | -13.238 dB |

The y-neighbor isolation improves only `0.13 dB`. Y-odd modal RL improves
from `11.91` to `13.18 dB`, but even and x-odd degrade to `8.72` and
`6.94 dB`. The bottleneck therefore remains and the physical strict gate
fails.

## Decision

Both the v1.41 correction network and the v1.43 mirrored-secondary input
topology are stopped. Do not tune the mirrored branch further, add another
correction stage, export EEP, create labels, or retrain the critic.

The next authorized hardware branch must change the element class: a
ground-backed/cavity-backed radiator or element-level neutralization topology
with independent self-impedance tuning and several-dB x/y coupling reduction.
It must pass 1x1 and memory-feasible 2x2 physical gates before any array work.
