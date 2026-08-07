# v1.38 Vertical Differential Independent Validation

## Scope

This stage tests whether the 60 residual v1.37 small-segment messages affect
the solved input response. The finite-conductivity sheets, square differential
posts, partitioned substrate, port, mesh controls, and 10 GHz thresholds are
frozen. Three new projects are built from source: two direct-solver repeats and
one domain-decomposition repeat. No mesh or solved project is copied between
cases.

The three-frequency and efficiency projects are created only after the
independent consistency gate passes. Physical 2x2, arrays, EEP, labels, and
critic training remain outside this stage.

## Independent 10 GHz Result

| Case | Solver | RL | Final Delta S | Tetrahedra | Total messages | Conductor messages |
|---|---|---:|---:|---:|---:|---:|
| direct repeat A | Direct | 18.044 dB | 0.002518 | 94,991 | 60 | 0 |
| direct repeat B | Direct | 18.044 dB | 0.002518 | 94,991 | 60 | 0 |
| DDM repeat | Domain Decomposition | 18.044 dB | 0.002518 | 94,991 | 60 | 0 |

The maximum pairwise absolute S difference is `2.2123e-14`, the RL span is
`1.5277e-12 dB`, and the maximum input-impedance difference is
`1.8482e-12 ohm`. Every run reports exactly `AirRegion=48` and `Substrate=12`;
no conductive body or unexpected body is reported. The generated DDM setup
explicitly contains `DrivenSolverType = Domain Decomposition`.

For this exact frozen model, the residual dielectric/air messages are therefore
classified as numerically benign. This evidence does not relax the geometry
gate for a changed feed, substrate partition, or array model.

## Three-Frequency And Power Result

| Frequency | RL | Final Delta S | HFSS radiation efficiency | System efficiency | Accepted power | Radiated power |
|---:|---:|---:|---:|---:|---:|---:|
| 9.96 GHz | 17.070 dB | 0.002601 | 1.001313 | 0.981655 | 0.980368 W | 0.981655 W |
| 10.00 GHz | 18.044 dB | 0.002518 | 1.001964 | 0.986245 | 0.984311 W | 0.986245 W |
| 10.04 GHz | 19.100 dB | 0.002661 | 1.002054 | 0.989727 | 0.987698 W | 0.989727 W |

`RadiationEfficiency` agrees with `RadiatedPower / AcceptedPower` within
`3.34e-15`. Its 0.131%-0.205% excess above unity is retained as HFSS numerical
power-balance error and is not reported as physical efficiency above 100%.
The conservative `RadiatedPower / IncidentPower` system efficiency is
98.17%-98.97%, independently supporting the 95% efficiency lower bound.

All three frequencies retain zero conductor messages and the same 60
substrate/air messages.

## Decision

The v1.38 gate passes. The frozen true differential, dual-resonant,
finite-sheet 1x1 input is now an engineering-trusted small-cell reference for
matching, solver repeatability, and efficiency. One physical 2x2 coupling and
active-RL smoke is authorized next.

4x4, 16x16, EEP export, training-label generation, and critic retraining remain
locked until that physical 2x2 passes.
