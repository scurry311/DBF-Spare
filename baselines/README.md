# Baselines

Each dated directory is an immutable, compact checkpoint of the research
state. It contains a human-readable decision, selected result snapshots, and a
SHA-256 manifest pointing back to the original local artifacts.

Large raw solver trees and generated datasets remain outside Git. A snapshot
is evidence of the recorded result, not permission to treat a blocked result
as an engineering label.

Current checkpoints:

- `2026-07-21`: blocked 16x16 adaptive-mesh physics gate.
- `2026-07-24`: trusted fixed-mesh S256 and EEP reconstruction; active-RL and
  strict engineering critic labels remain blocked.
- `2026-07-24-active-rl-joint`: trusted EEP/S256 task-weight projection smoke;
  active matching improves, while K=6 regional-pattern coverage keeps new
  HFSS training labels locked.
- `2026-07-24-dense-local-hfss`: dense local-5-degree EEP constraints and 15
  trusted sparse full-wave positives open implementation-residual training.
- `2026-07-24-implementation-residual`: first paired implementation-error
  critic; useful for ranking but limited to 15 independent scenes.
- `2026-07-24-expanded-residual`: 45 independent scenes and 455 complete HFSS
  cases; mainlobe and ranking support pass, while gate15 AUROC and calibration
  keep the residual critic experimental.
- `2026-07-27-v16-robust-drift-oracle`: preregistered E1/E2/E3 common-command
  robust search; K=2/K=4 pass the E2 oracle gate, while K=6 and E1 keep critic
  retraining and perturbed 16x16 HFSS locked.
- `2026-07-27-v17-k6-multifrequency-rescue`: frozen-E2 K=6 multifrequency and
  quantization-aware boundary rescue; Stage B passes with 97.33% overall and
  96% K=6 oracle coverage, opening only the small frozen 16x16 HFSS smoke.
- `2026-07-28-v18-frozen-16x16-hfss-smoke`: twenty frozen independent sparse
  candidates and 100 direct saved-field HFSS cases pass the strict pattern and
  semantic active-RL gates; physical-corner critic labels remain locked.
- `2026-07-29-v19-perturbed-operator-hfss-smoke`: first real 16x16 9.96 GHz
  physical-operator corner; EEP/direct-HFSS mapping passes, while frozen strict
  coverage falls to 20% solely because active matching loses margin.
- `2026-07-29-v110-three-frequency-active-rl`: nominal/9.96 joint projection
  improves strict coverage to 60%, but prospective 10.04 GHz coverage is 30%
  and only 10% remains feasible across all three frequencies; active matching
  is the limiting physical mechanism.
- `2026-07-29-v111-three-frequency-joint-search`: three-frequency port
  sensitivity and structured mask/common-weight alternating search improve the
  strict oracle from 2/20 to 7/20, but the 18/20 gate and 11 dB reserve fail;
  HFSS smoke and critic training remain locked pending same-mask feasibility
  rescue.
