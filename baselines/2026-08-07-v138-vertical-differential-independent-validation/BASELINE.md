# v1.38 Vertical Differential Independent Validation Baseline

## Evidence Level

Physical HFSS 1x1 evidence from two independently built direct projects, one
independently built DDM project, and three independently solved frequency
projects with antenna-power reports.

## Result

- Direct/direct/DDM maximum absolute S difference: `2.2123e-14`.
- Minimum three-frequency passive RL: `17.070 dB`.
- Maximum final Delta S: `0.002661`.
- Conductive-body small-segment messages: `0` in every run.
- Residual messages: `AirRegion=48`, `Substrate=12` in every run.
- Conservative system efficiency: `98.17%-98.97%`.

The raw HFSS radiation-efficiency excess above unity is at most 0.205% and
closes against radiated/accepted power to numerical precision. It is retained
as a power-balance error, not a physical efficiency above 100%.

## Gate State

The frozen 1x1 geometry passes independent mesh/solver, three-frequency RL,
and 95% efficiency gates. Physical 2x2 is authorized. 4x4, 16x16, EEP, labels,
and critic training remain locked.
