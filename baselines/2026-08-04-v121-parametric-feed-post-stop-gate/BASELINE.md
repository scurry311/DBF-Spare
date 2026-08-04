# v1.21 Parametric Feed/POST Physical Stop-Gate Baseline

## Evidence Level

This checkpoint contains solved 10 GHz network-only HFSS S8 evidence for 20
physical parameter sets. It is not an integrated 2x2, array, EEP, or full-wave
training-label result.

## Reproducible Result

The original 16-point Latin-hypercube DOE and four frozen Jacobian/Pareto local
refinements all converged and exported valid S8. The local cases ran serially;
the minimum remaining system memory was 7.62 GiB, above the 3 GiB abort guard.

| Gate component | Passing cases |
|---|---:|
| Final Delta S <= 0.05 | 20/20 |
| Reciprocity error <= 1e-4 | 19/20 |
| Passivity sigma <= 1.001 | 20/20 |
| Network efficiency >= 95% | 15/20 |
| Passive RL >= 10 dB | 10/20 |
| Active RL >= 11 dB | 0/20 |
| Total RL >= 11 dB | 0/20 |
| Physical-to-target S8 <= 0.10 | 0/20 |
| Complete network gate | 0/20 |

Best active RL remains 5.076 dB (`doe09_lhs`). The local gradient corner gives
the best physical-to-target error, 0.293, and the best total RL, 9.657 dB, but
its active RL is only 2.566 dB. It also has the best observed efficiency,
96.596%.

The first-order response surface predicted 8.124 dB active RL for that corner.
The 5.558 dB prediction error demonstrates strong nonlinear multiport/modal
interaction outside the sampled interior. Only the HFSS values are used for
the gate decision.

## Gate State

The preregistered 20-candidate review is reached. The current single-stage
POST/local-loading topology is stopped because:

- best active RL is below the 10 dB stop threshold;
- best physical-to-target S8 error is above the 0.15 stop threshold;
- no candidate passes the complete 10 GHz gate.

Three-frequency optimization, direct/DDM promotion, integrated 2x2 solving,
4x4/16x16 expansion, label generation, and critic training are prohibited.

## Next Model

Replace the feed body with a true balanced/differential launch and one local
2x2 even/odd-mode correction branch. Validate PRE/POST reference planes and
transducer efficiency on 1x1/2x2 before considering aperture-coupled or
dual-resonant coupled-feed alternatives. Do not add another decoupling stage.
