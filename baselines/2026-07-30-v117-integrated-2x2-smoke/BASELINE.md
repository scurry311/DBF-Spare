# v1.17 Integrated Physical 2x2 Smoke

This baseline physically connects the v1.16 grounded x-modal network to four
dual-slot probe-fed patches. The compact network is placed below the shared
ground, followed by routed microstrip fanout and four physical probes. Only
the four source-facing PRE ports remain as excitations.

## Numerical Credibility

| Metric | Result | Gate |
|---|---:|---:|
| Adaptive passes | 7 | converged |
| Final Delta S | 0.02889 | <= 0.05 |
| Tetrahedra | 254,705 | resource audit |
| Peak solver memory | 9.50 GiB | resource audit |
| Reciprocity error | 1.076e-15 | <= 1e-4 |
| Passivity sigma | 0.466 | <= 1.001 |
| EEP power relative error | 0.86% | <= 5% |

The electromagnetic solve completed normally. A postprocessing variable-name
error was recovered by reopening the saved fields without rerunning HFSS; all
12 EEP files were exported.

## Engineering Gate

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Passive RL | 8.251 dB | >= 12 dB | Failed |
| Active RL | 1.601 dB | >= 11 dB | Failed |
| Total RL | 8.215 dB | >= 11 dB | Failed |
| Accepted-to-radiated efficiency | 94.12% | >= 95% | Failed |
| Transducer efficiency | 81.34% | >= 85% | Failed |
| Integrated versus S8+S4 max abs Delta S | 0.307 | <= 0.05 | Failed |

## Decision

No independent repeat, 4x4/16x16 expansion, HFSS labels, or critic training is
authorized. The validated S8 network itself is not the failed element; the
new POST-to-probe fanout changes electrical lengths and port impedances. The
next hardware experiment must equalize routed electrical lengths and optimize
the POST transition using this integrated S4, while keeping the v1.16 network
components and all engineering thresholds frozen.
