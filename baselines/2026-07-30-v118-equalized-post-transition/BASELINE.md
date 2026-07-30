# v1.18 Equalized POST Transition

The v1.16 RLC values, Q factors, substrate, copper, trace width, channel
locations, and modal bridge values are frozen. The compact network is moved as
a rigid body and four non-crossing routes are made equal at
`23.3928 mm`. A common stepped probe launch
is included and the integrated physical 2x2 is solved at 9.96/10/10.04 GHz.

## Numerical Credibility

| Metric | Result | Gate |
|---|---:|---:|
| Adaptive passes | 7 | converged |
| Final Delta S | 0.02449 | <= 0.05 |
| Tetrahedra | 253,408 | resource audit |
| Peak solver memory | 9.11 GiB | resource audit |
| Minimum free memory | 6.13 GiB | > 1.5 GiB abort line |
| Reciprocity error | 9.790e-17 | <= 1e-4 |
| Passivity sigma | 0.396 | <= 1.001 |

## Pre-Repeat Gate

| Metric | v1.17 | v1.18 | Gate | Decision |
|---|---:|---:|---:|---|
| Active RL | 1.601 dB | 7.226 dB | >= 11 dB | Failed |
| Integrated versus cascade max abs Delta S | 0.307 | 0.277 | <= 0.05 | Failed |
| Passive RL | 8.251 dB | 9.552 dB | diagnostic | Improved |
| Total RL | 8.215 dB | 9.353 dB | diagnostic | Improved |

Equalization is beneficial but insufficient. A common transition search and
per-port uncoupled stepped-line/L-section synthesis have no joint feasible
point. The best per-port stepped-line proxy reaches about 9.97 dB active RL
and 0.096 max abs Delta S, still outside both gates.

The EEP report export produced no files. This is recorded but is not a reason
to rerun the field solve because the S4 pre-repeat gate already failed.

## Decision

No independent repeat, 4x4/16x16 expansion, HFSS labels, or critic training is
authorized. The next physical topology must integrate a multiport POST
transition/decoupler; further common-length or uncoupled local launch sweeps
are stopped.
