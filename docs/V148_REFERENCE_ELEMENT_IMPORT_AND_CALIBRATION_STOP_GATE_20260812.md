# v1.48 Reference Element Import And Calibration Stop Gate

## Scope

This stage audits the Marlin standalone U-slot `a3dcomp`, attempts a frozen
1x1 import path on AEDT 2023.1, and independently reconstructs the open
MyriadRF Vivaldi PCB as a port/material/reference-plane calibration fixture.
The physical gate is evaluated before any 2x2, 16x16, EEP, label, or critic
work.

## Marlin Audit

The source component is a 459,678-byte AEDT 2025.2 design-derived component.
Static inspection finds the patch, microstrip, grounds, pads, and vias, but
does not expose a verifiable excitation or radiation boundary. The original
component is rejected by AEDT 2023.1 as a future-version file.

A header-only 2023.1 compatibility copy can be inserted as an encapsulated
diagnostic component. It remains non-engineering evidence: the top-level
project exposes no auditable port or boundary, and the copy is not the
original artifact. The full `MARLIN.aedt` project also cannot be audited
because AEDT reports `Unable to decrypt encrypted project file.`

Consequently no trustworthy Marlin 1x1 exists in this environment. The
nominal sweep, independent direct/DDM, and 2x2 frozen active-RL stages were not
run.

## MyriadRF Calibration Fixture

The calibration fixture is generated from commit
`fdeb99138e04a63137a0aea7820244ee39cc2398`. It uses the exact four filled
KiCad copper polygons (180 vertices), two plated vias, the report-specified
0.5 mm RO4350B substrate, relative permittivity 3.6, loss tangent 0.0032, and
a frozen 50-ohm reference plane at the SMA center-pad inner board edge. The
SMA body is not modeled.

The valid serial direct solve gives:

| Metric | Result |
|---|---:|
| Final Delta S | 0.0101665 |
| 9.96/10.00/10.04 GHz passive RL | 19.504 / 20.474 / 21.576 dB |
| Contiguous 10 dB band containing 10 GHz | 3.5-13.4 GHz |
| Peak numerical mesh | 48,856 tetrahedra |
| Minimum free memory during valid run | 12.20 GiB |

The report states a calculated `S11 < -10 dB` band of 3.5-14 GHz, so the
reconstructed port/material/reference-plane choice is qualitatively
consistent in S11. The ten small-segment records are all on `AirRegion`; none
is on a conductor, port, or dielectric. Their reported S-matrix-only cutoff is
4.21 MHz, well below the modeled band.

The public report contains only a measured S11 plot, not raw VNA Touchstone
data. Existing field postprocessing also produced no efficiency or gain table.
Therefore exact measured calibration and full radiation calibration are not
passed. This fixture cannot generate engineering or training labels.

## Resource Audit

Two invalid attempts are retained and excluded from metrics. The distributed
run stalled at initial coarsening. A serial run with 0.18 mm mesh applied to
all artwork expanded to 2.136 million tetrahedra and was stopped by the memory
guard at 0.58 GiB free. The accepted run keeps 0.18 mm deterministic mesh only
at the port reference plane and lets lambda/adaptive refinement resolve the
full artwork.

## Decision

The v1.48 physical gate fails because Marlin has no trustworthy 1x1 model in
AEDT 2023.1 and the MyriadRF result is calibration-only evidence. Keep 2x2,
4x4, 16x16, S256, EEP, training labels, and critic retraining locked.

Resume the Marlin branch only with AEDT 2025.2 or an unencrypted source design
that includes dielectric, excitation, reference plane, boundaries, and setup.
Raw MyriadRF VNA data are additionally required for exact reference-plane
regression.
