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
