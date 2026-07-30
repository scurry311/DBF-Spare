# v1.16 Exact-HFSS Physical S8 Optimization

This baseline replaces low-order circuit extrapolation with exact HFSS solves
of a parameterized physical eight-port feed fixture. Four PRE and four POST
reference planes remain explicit, and all 285 frozen representative sources
are replayed after cascading each physical S8 with the trusted antenna S4.

## Search

Seven one-at-a-time coarse points and six local refinement points were solved
sequentially. The selected finite-Q component values are:

- series inductor: `0.617728 nH`;
- grounded shunt capacitor: `0.138748 pF`;
- x-pair bridge inductor: `3.099437 nH`.

## Result

| Metric | v1.15 nominal | v1.16 selected | Independent repeat |
|---|---:|---:|---:|
| Passive RL | 13.867 dB | 15.402 dB | 15.402 dB |
| Active RL | 10.932 dB | 11.937 dB | 11.937 dB |
| Total RL | 12.024 dB | 13.048 dB | 13.048 dB |
| Actual-load insertion efficiency | 97.64% | 97.66% | 97.66% |
| Transducer efficiency | 92.01% | 93.33% | 93.33% |
| Final Delta S | 0.00139 | 0.00127 | 0.00127 |

The selected design improves minimum active RL by
`1.005 dB`. The independent repeat
differs by only `1.062e-13` in
maximum absolute S and passes the unchanged 11.5 dB design gate.

## Decision

Retain the grounded x-modal topology. One integrated physical 2x2
antenna-network smoke is now authorized. A distributed even/odd hybrid is kept
only as a fallback. The 4x4/16x16 expansions, HFSS labels, and critic training
remain locked because this evidence is still an S8 plus trusted-S4 cascade,
not an integrated full-wave antenna-network validation.
