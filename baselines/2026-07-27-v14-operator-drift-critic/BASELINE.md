# v1.4 Operator-Drift Residual Critic

This baseline introduces electromagnetic-operator and hardware residuals that
the nominal 10 GHz EEP/S256 operator cannot observe. It does not replace the
trusted v1.3 16x16 nominal array baseline.

## Added Physics

- HFSS frequency profiles at 9.8, 10.0, and 10.2 GHz.
- Patch-length profiles at 9.25, 9.35, and 9.45 mm.
- Relative-permittivity profiles at 2.16, 2.20, and 2.24.
- Full complex S-matrix and Etheta/Ephi operator drift.
- Gain/phase calibration error, quadrant phase bias, quantization, compression,
  soft channel failures, and temperature-offset proxies.

All seven 4x4 HFSS profiles converged with final Delta S below 0.05 and complete
16-port complex EEP exports. Non-nominal full-wave field NMSE is 0.00858-0.07381,
and maximum S16 drift is 0.1342-0.3927. These residuals are physically material,
not nominal EEP/HFSS export noise.

## Proxy Dataset

The 4x4 full-wave sensitivities are mapped to the trusted nominal 16x16
EEP/S256 basis at four nonzero drift strengths: 0.05, 0.20, 0.50, and 1.00.
This mapping is a physics-calibrated proxy and must not be reported as a 16x16
HFSS result.

| Result | Value |
|---|---:|
| Base direction scenes | 45 |
| Drift scenes / candidates | 1,080 / 4,320 |
| K=2 / K=4 / K=6 base scenes | 15 / 15 / 15 |
| Strict positives / negatives | 966 / 3,354 |
| Hard negatives | 1,897 |
| Train / validation / test candidates | 2,880 / 576 / 864 |
| Test strict positives | 177 |
| Target-hash split overlap | 0 |

Strict positive rates decrease from 59.7% at drift strength 0.05 to 29.3% at
0.20, 0.46% at 0.50, and 0% at 1.00. This produces a usable feasibility
boundary while retaining the full-drift failures.

## Five-Seed Critic

The scene-conditioned critic predicts five heteroscedastic physical-margin
residuals from target sets, masks, task-level complex weights, nominal EEP/S256
margins, and the new drift descriptors.

| Test result | Mean | 95% half-width |
|---|---:|---:|
| Strict AUROC | 0.9887 | 0.0032 |
| Strict AUPRC | 0.9554 | 0.0052 |
| Strict ECE | 0.0330 | 0.0025 |
| Strict precision | 0.9064 | 0.0248 |
| Active-RL AUROC | 0.9867 | 0.0032 |
| Active-RL ECE | 0.0423 | 0.0167 |
| Critic top-1 strict rate | 0.3000 | 0.0150 |
| Fixed ratio-0.7 strict rate | 0.2454 | 0 |
| Candidate oracle strict rate | 0.3426 | 0 |

The proxy discrimination, calibration, and 90% precision gates pass. The
project top-1 target of 80% does not pass because this stress dataset contains
many hardware states with no feasible candidate. The critic is therefore
limited to candidate screening, proxy ablation, and pretraining.

## Drift-Feature Ablation

The same five seeds were retrained on the identical split after removing only
the new operator/hardware drift descriptors.

| Metric | With drift | Without drift | Delta |
|---|---:|---:|---:|
| Strict AUROC | 0.9887 | 0.9827 | +0.0060 |
| Strict AUPRC | 0.9554 | 0.9353 | +0.0201 |
| Strict recall | 0.8983 | 0.7480 | +0.1503 |
| Active-RL AUROC | 0.9867 | 0.9765 | +0.0103 |
| Active-RL recall | 0.9171 | 0.7930 | +0.1240 |
| Top-1 strict rate | 0.3000 | 0.2972 | +0.0028 |

The added physics improves residual discrimination and failure recall. Its
ranking gain is negligible because the candidate oracle is only 34.3%; further
top-1 improvement requires better mask/weight candidates, not a larger critic.

## Evidence Decision

- `operator_drift_calibration_gate_pass`: true.
- `proxy_critic_gate_pass`: true for discrimination/calibration/precision.
- `drift_feature_ablation_gate_pass`: true for discrimination and recall.
- `engineering_critic_promoted`: false.
- `adaptive_ratio_final_hfss_allowed`: false.

Promotion requires a frozen, independent 16x16 HFSS smoke containing frequency,
geometry, matching, and hardware perturbations. No threshold or model update may
use that prospective set.
