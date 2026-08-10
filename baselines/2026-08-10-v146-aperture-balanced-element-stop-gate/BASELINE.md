# v1.45/v1.46 Aperture And Balanced Element Stop Gate

## Evidence Level

This checkpoint contains 37 complete physical 10 GHz HFSS 1x1 cases.  It
contains no authorized 2x2 S4, array, EEP, dataset, label, or critic evidence.

## Physical Models

- v1.45 uses a complete patch, a continuous slotted reflecting ground, a
  lower microstrip/open stub, and a vertical ground-referenced port.  It also
  tests one integrated single-section impedance transformer.
- v1.46 uses two symmetric patch halves, two physical probes, one true
  differential port, and a continuous reflecting ground with two local probe
  clearances.
- Copper sheets retain finite conductivity and physical thickness.  The
  0.18 mm deterministic mesh is confined to the local port sheet; aperture,
  patch, and ground fields are adaptively refined.

## Result

| Family | Complete | Numerical gate | Efficiency gate | RL >= 10 dB | Best RL |
|---|---:|---:|---:|---:|---:|
| Aperture, no transformer | 13 | 13 | 13 | 0 | 3.200 dB |
| Aperture, one transformer | 14 | 14 | 0 | 0 | 0.379 dB |
| Ground-backed differential | 10 | 10 | 4 | 0 | 0.630 dB |

The best aperture case is `acp11_l13_w10_stub35` at
`11.44-j24.78 ohm`, 99.47% radiation efficiency, and final Delta S 0.03813.
The best ground-backed differential case is `gbd04_gap_narrow` at
`23.69+j172.24 ohm`, 98.16% efficiency, and final Delta S 0.04009.

Two early whole-conductor 0.18 mm mesh attempts reached 1.89-2.15 GiB free
memory and were aborted without S1 output.  The local-port mesh representation
reduced complete cases to at most 9,151 tetrahedra and retained at least
13.565 GiB free memory.

## Decision

Zero of 37 complete physical candidates reaches the 15 dB 1x1 gate.  Stop
both element families before 2x2.  Do not export S256/EEP, create training
labels, or retrain the critic from these runs.

The next hardware branch must start from an independently validated 50 ohm
commercial or literature-backed radiator/feed cell, or add a physically
measured differential balun/matching transition as part of the element.  It
must first pass a frequency sweep and an independent 1x1 repeat; it must not
continue by adding another local matching or decoupling stage to v1.45/v1.46.
