# Baseline v1.3 K=4 HFSS and Critic Gate

## Decision

The supported EEP/S256 operating envelope now includes K=4 at maximum target
scan angle 48 deg and minimum target separation 16 deg. A frozen 20-scene HFSS
smoke passed the full pattern, mainlobe, and operating-state active-RL gates.
This authorized one targeted 50-candidate full-wave label batch.

The residual critic did not meet its pre-registered strict AUROC, calibration,
or top-one requirements. It is not promoted, and adaptive-ratio prospective
HFSS remains locked. The actual implementation-weight EEP/S256 gate is retained
as the deterministic selector inside the validated fixed electromagnetic basis.

## K=4 Operating Envelope

| Result | Value | Decision |
|---|---:|---|
| New K=4 scenes / candidates | 20 / 1,920 | Complete |
| Masks per ratio | 24 | Complete |
| K=4 scene oracle | 20/20 (100%) | Passed 90% |
| Excluded target-hash overlap | 0 | Passed |
| Supported K=2 oracle | 33/34 (97.1%) | Passed |
| Supported K=4 oracle | 20/20 (100%) | Passed |
| Supported K=6 oracle | 19/20 (95.0%) | Passed |

K=1 remains a matching and power-calibration control and is not included in
multitask isolation claims.

## Frozen HFSS Smoke

Twenty independent scenes were frozen before HFSS: K=2/4/6 counts were 7/6/7,
and ratio 0.5/0.6/0.7/0.8 counts were 9/5/3/3. The set included strict-positive,
PSLL-boundary, isolation-boundary, and active-RL-boundary candidates. Masks,
task weights, combined weights, hashes, and thresholds were not changed after
viewing full-wave results.

| Result | Value | Decision |
|---|---:|---|
| HFSS cases | 100/100 complete | Passed |
| Maximum no-scale complex NMSE | 6.04e-12 | Passed |
| Maximum magnitude RMSE | 3.14e-5 dB | Passed |
| Full-wave pattern and mainlobe gate | 20/20 | Passed |
| Combined operating-state active-RL gate | 20/20 | Passed |
| Combined plus significant-task active-RL gate | 20/20 | Passed |
| All-nonzero-task active-RL diagnostic | 12/20 | Diagnostic only |

The physical hardware excitation is the simultaneous combined weight
`w = sum(w_k)`. Per-task weights are a mathematical decomposition used for
isolation metrics. Task coefficients below -20 dB relative amplitude are not
treated as independently driven hardware states; the unfiltered all-nonzero
result is retained as a conservative diagnostic. No RL threshold was changed,
and no HFSS output was modified.

## Targeted Full-Wave Labels

After the smoke passed, 50 candidates from 41 new scenes were frozen and run.
The planned composition was 20 near-boundary, 13 hard-positive, 12
hard-negative, and five paired-ratio candidates. Known quantization,
gain/phase, failed-channel, and matching perturbations were included.

| Result | Value | Decision |
|---|---:|---|
| HFSS cases | 256/256 complete | Passed |
| Maximum no-scale complex NMSE | 5.94e-12 | Passed |
| Maximum magnitude RMSE | 5.18e-5 dB | Passed |
| Full-wave pattern plus mainlobe positives | 28/50 | Label support |
| Combined operating-state engineering positives | 25/50 | Label support |
| Combined plus significant-task positives | 18/50 | Critic label |
| Independent scenes / train-val-test | 41 / 29-6-6 | No leakage |

## Residual Critic Gate

Five CPU seeds were trained on the 50-candidate scene-grouped dataset. The
strict target used full-wave pattern, mainlobe, combined active RL, and the
pre-existing -20 dB significant-task active-RL diagnostic.

| Metric | Observed | Required | Decision |
|---|---:|---:|---|
| Strict AUROC | 0.760 | >=0.88 | Failed |
| Strict ECE | 0.234 | <=0.08 | Failed |
| Accepted-candidate precision | 0.900 | >=0.90 | Passed |
| Scene top-one strict rate | 0.467 | >=0.80 | Failed |
| Test scene oracle | 0.500 | - | Insufficient support |

The nominal-command critic is not promoted. In contrast, evaluating the known
actual implementation weights with the EEP/S256 operator agreed with HFSS on
all 50 strict labels. Maximum absolute HFSS-minus-EEP margin residuals were
5.72e-5 dB or smaller for pattern metrics and 9.89e-7 dB for active RL.

## Use Policy

- Use actual implementation weights with EEP/S256 for deterministic physics
  gating inside the validated K=2/4/6 envelope.
- Do not use the v1.3 critic for automatic HFSS admission or final selection.
- Do not start adaptive-ratio prospective HFSS from this checkpoint.
- Do not tune thresholds on the frozen smoke or targeted-label results.
- A non-trivial critic requires labels with unobserved operator drift, such as
  frequency, geometry, calibration, S-parameter, temperature, or hardware
  variation that is not already supplied to deterministic EEP evaluation.
- Resume adaptive-ratio validation only after the critic protocol is revised
  and independently passes, or after a separately pre-registered deterministic
  EEP/S256 selector protocol is approved.
