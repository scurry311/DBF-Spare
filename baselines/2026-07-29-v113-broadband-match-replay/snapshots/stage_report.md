# v21 Frozen Broadband-Matching Replay

This stage freezes the v1.12 masks, task commands, ratios, E2 states, and
engineering thresholds. No candidate is reoptimized and no HFSS job or critic
training is launched.

## Provenance

The final run03 freeze package contains 20 independent scenes and reproduces
the v1.12 distribution K=2/4/6 = 7/6/7. Its package SHA-256 is
`754dfa94aaa76e717f0b579ff5f8960758bdf20a88175c7c8bbc9233180092e4`.
The package is byte-identical to the corrected run02 package; run01 is
invalidated because it used the new optimizer seed instead of the frozen E2
state seed.

## Matching Design

The old hardware proxy is one common 0.533 nH, Q=50 series inductor. New
finite-Q S-L-S networks are evaluated directly on the three physical raw
S256 matrices. Uniform and corner/edge/interior parameterizations are tested.
The selected three-class circuit remains a circuit-cascade model and is not an
embedded-layout HFSS validation.

| Frequency | Old passive RL | Three-class passive RL |
|---:|---:|---:|
| 9.96 GHz | 15.728 dB | 15.557 dB |
| 10.00 GHz | 13.535 dB | 13.607 dB |
| 10.04 GHz | 10.616 dB | 10.748 dB |

The minimum replayed network efficiency is 98.09%. The three-class network
therefore does not obtain its return-loss result through excessive dissipation.

## S256/EEP Validation

All three rebuilt external-port operators are reciprocal and passive. The
maximum reciprocity error is 4.17e-9, the maximum passivity singular value is
0.97002, and the maximum no-scale complex EEP map NMSE is 1.92e-13.

## Frozen Replay

| Result | Old network | Three-class network | Gate |
|---|---:|---:|---:|
| Overall strict | 8/20 | 8/20 | 18/20 later-stage target |
| K=2 strict | 4/7 | 4/7 | >= 6/7 |
| K=4 strict | 2/6 | 2/6 | >= 5/6 |
| K=6 strict | 2/7 | 2/7 | Diagnostic only in this stage |
| 11 dB reserve | 0/20 | 0/20 | >= 1 |

K=2 pattern coverage remains 7/7, while active-RL coverage remains 4/7. K=4
has pattern coverage 3/6 and active-RL coverage 4/6; their intersection is
only 2/6. Pattern changes from the new network are small, but active-RL gains
are mixed and do not expand the feasible intersection.

## Decision

The replay gate fails. Candidate optimization, small candidate HFSS smoke,
bulk HFSS, and critic training remain locked. The present independent
three-class S-L-S network is not sufficient to broaden the high-frequency
active-match feasibility set.

If hardware work resumes, the next experiment must be a small-cell physical
matching redesign, such as a dual-resonant feed/stub or coupled-feed patch,
validated first on 1x1/4x4 frequency sweeps. It must not be presented as a
candidate-array HFSS validation or training-label run.
