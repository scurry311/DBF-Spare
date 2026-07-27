# v1.6 Robust Drift Oracle

This baseline preregisters E1/E2/E3 hardware-drift envelopes and evaluates a
common-mask, common-command-weight robust candidate search. It does not use
corner-specific weights, critic labels, or perturbed 16x16 HFSS results.

## Evidence Scope

All results are a 4x4-HFSS-calibrated 16x16 EEP/S256 proxy. They are valid for
candidate-space development and failure diagnosis only. They do not promote
the critic and do not open the frozen 16x16 HFSS smoke.

## Frozen Envelopes

- E1 commissioning: drift intensity 0.05.
- E2 engineering: intensity 0.20, representing frequency +/-0.04 GHz, patch
  length +/-0.02 mm, relative permittivity +/-0.008, and the frozen
  calibration/quantization/temperature model.
- E3 stress: intensities 0.50 and 1.00; diagnostic only.
- E2 was not reduced after observing results.

## Candidate Pool

| Item | Value |
|---|---:|
| Existing / new independent scenes | 45 / 30 |
| K=2 / K=4 / K=6 new scenes | 10 / 10 / 10 |
| Initial candidates | 9,600 |
| Unique initial masks per scene/ratio | 32 |
| Dense multi-corner candidates | 1,200 |
| Common-weight refinements | 300 |
| Targeted rescue candidates | 736 |
| Final unique masks per scene/ratio | 32-40 |
| Frozen target-set overlap | 0 |
| ratio=1.0 candidates | 0 |

## Robust Oracle

| Stage | E2 overall | K=2 | K=4 | K=6 |
|---|---:|---:|---:|---:|
| Initial ranking | 1.33% | 4% | 0% | 0% |
| Common-weight projection | 69.33% | 88% | 76% | 44% |
| Active-RL-guided mask rescue | 82.67% | 100% | 96% | 52% |

The E1 oracle on the 30 new scenes is 86.67%, below the preregistered 90%
gate. E3 is 0%. Sixty-two of 75 E2 scenes pass: 50 first pass at ratio 0.5,
9 at 0.6, and 3 at 0.7. A sparse K=6 positive at ratio <=0.7 exists.

## Failure Decision

Stage B fails because E1 new-scene coverage, E2 overall coverage, and E2 K=6
coverage remain below 90%. The 13 remaining E2 scene failures are led by eight
mainlobe failures, four active-RL failures, and one nearest-isolation failure.
Most mainlobe failures are K=6 frequency-corner pointing shifts at the fixed
sampling-grid gate.

- `stage_b_gate_pass`: false.
- `critic_retraining_allowed`: false.
- `hfss_smoke_allowed`: false.
- `engineering_critic_promoted`: false.

The next experiment must keep E2 fixed and add explicit multi-frequency
combined-beam constraints for K=6, together with active-RL margin reserve. It
must not train a critic to hide the remaining physical infeasibility.
