# v1.47 Literature 10 GHz Balun-Dipole Stop Gate

## Objective

Replace the previous locally invented element branches with a published 10 GHz
printed dipole and integrated balun, then require a broadband 1x1 frequency
sweep and an independent direct/DDM repeat before opening physical 2x2
active-RL evaluation.

## Reference Basis And Evidence Boundary

The reconstruction uses the conventional 10 GHz dimensions reported by Huang
et al.: a 30 mm reflector, 0.5 mm RO5880 substrate (`epsilon_r=2.2`), 13 mm by
2 mm dipole, 0.3 mm by 6.83 mm slotline, 0.87 mm microstrip lines, 1.8 mm leg
separation, and 7.5/5.0 mm balun branches. The front U-shaped microstrip and
back slotline/dipole topology is cross-checked against Li et al.

- [Huang et al., 2014](https://onlinelibrary.wiley.com/doi/10.1155/2014/765891)
- [Li et al., IEEE TAP 2009](https://tentzeris.ece.gatech.edu/TAP2009_RL.pdf)
- [Chen et al., measured X-band 4x4 array, 2025](https://doi.org/10.1109/LSENS.2025.3575781)

The public Huang text does not provide a complete manufacturing CAD or Gerber
for the measured improved slotted-cavity unit. In particular, the exact back
ground width and all coax/launch/reference-plane details are not tabulated.
v1.47 therefore records those quantities as reconstructed assumptions. It is
a literature-guided reconstruction, not a claim that the measured article
unit has been reproduced exactly.

## Preregistered Gates

- Final adaptive `Delta S <= 0.05`, preferred `<= 0.02`.
- Passive RL at 9.96/10.00/10.04 GHz `>= 15 dB`.
- Contiguous 10 dB bandwidth `>= 1 GHz`.
- Radiation efficiency `>= 95%`.
- Topology warning count `= 0`.
- Independent direct/DDM `max |Delta S| <= 0.05`.
- Physical 2x2 remains locked until all nominal and independent 1x1 gates pass.

## Execution

Runs 05-07 established the memory-safe solver configuration. Run08 used a
0.18 mm local balun/slot/port mesh, 5% adaptive refinement, four solver cores,
and explicit exclusion of frequency distribution so the three discrete
frequencies could not solve concurrently. The batch and export processes both
returned zero.

| Numerical metric | Run08 result | Gate |
|---|---:|---:|
| Final adaptive Delta S | 0.01105 | <= 0.05 |
| Maximum tetrahedra | 138,360 | recorded |
| Peak solver memory | 7.88 GiB | memory-safe |
| Minimum free system memory | 7.16 GiB | > 3 GiB abort line |
| CAD topology warnings | 0 | 0 |

The solve is numerically credible, but the reconstructed antenna is not a
usable 50 ohm element:

| Physical metric | Run08 result | Gate |
|---|---:|---:|
| RL at 9.96 GHz | 0.04443 dB | >= 15 dB |
| RL at 10.00 GHz | 0.04454 dB | >= 15 dB |
| RL at 10.04 GHz | 0.04465 dB | >= 15 dB |
| RL range over 8-12 GHz | 0.03895-0.05021 dB | 10 dB band required |
| Contiguous 10 dB bandwidth | 0 GHz | >= 1 GHz |
| Radiation efficiency | 2.02% | >= 95% |

At 10 GHz, `S11=-0.8062+j0.5830` and `|Gamma|` is approximately 0.9949. The
flat near-open response across 8-12 GHz demonstrates a feed/topology mapping
failure, not a narrow resonance that should be rescued by local dimension
tuning.

## Decision

The nominal frequency-sweep gate fails. Independent direct/DDM repetition is
not run because it would only repeat a physically incorrect nominal model.
Physical 2x2 active-RL, 4x4/16x16, S256/EEP export, training labels, and critic
training remain locked.

This result does not invalidate the published antenna. It rejects only the
incomplete public-data reconstruction used here. The next admissible input is
one of the following:

1. Author/vendor Gerber, STEP, or AEDT/CST geometry with complete stackup,
   launch, reference plane, and measured S11 for the exact 10 GHz unit.
2. A commercial or open reference X-band 50 ohm element whose complete PCB
   artwork and balun/connector transition are available.

After acquiring that artifact, the same frozen sequence must be repeated:
8-12 GHz nominal scan, independent direct/DDM 1x1 validation, and only then
physical 2x2 active-RL. No blind tuning of the reconstructed back-ground width
or port location is authorized.
