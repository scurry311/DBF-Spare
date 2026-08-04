# Baselines

Each dated directory is an immutable, compact checkpoint of the research
state. It contains a human-readable decision, selected result snapshots, and a
SHA-256 manifest pointing back to the original local artifacts.

Large raw solver trees and generated datasets remain outside Git. A snapshot
is evidence of the recorded result, not permission to treat a blocked result
as an engineering label.

Current checkpoints:

- `2026-08-05-v124-physical-loadpull-jacobian-stop-gate`: ten independent
  physical S8 center differences show that five manufacturable local-block
  geometry directions explain only 6.64% of the required active-response
  residual; the block is stopped and the feed-point/input-impedance branch is
  authorized.
- `2026-08-04-v123-load-aware-modal-transformer-stop-gate`: a finite-Q local
  modal circuit passes its upper-bound gates, but its physical S8 mapping
  reaches 3.15 dB active RL and degrades 56/57 paired stimuli.
- `2026-08-04-v122-balanced-modal-branch-stop-gate`: the true differential
  1x1 launch passes its three-frequency S2 gate, while the single local 2x2
  floating modal branch reaches only 5.74 dB active RL and fails loaded
  efficiency gates; independent repeat, integration, array expansion, labels,
  and critic training remain locked.
- `2026-08-04-v121-parametric-feed-post-stop-gate`: twenty converged 10 GHz
  physical S8 candidates produce no complete gate pass; best active RL is
  5.08 dB and best physical-to-target S8 error is 0.293, so the single-stage
  POST/local-loading topology stops before three-frequency, integrated 2x2,
  array expansion, labels, or critic training.
- `2026-08-03-v121-parametric-feed-post-build-smoke`: shared network-only S8
  and integrated 2x2 CAD builders pass 6/6 construction audits; this checkpoint
  contains no solved electromagnetic performance.
- `2026-07-31-v120-joint-feed-fanout-sparse-graph`: the adjacent-chain
  finite-Q surrogate passes its matching, Delta-S, and tolerance gates, while
  two physical HFSS S8 mappings fail active-RL and physical-correlation gates;
  further decoupling stages, integrated 2x2, array expansion, and labels are
  locked pending direct feed-point/POST physical sensitivity optimization.
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
- `2026-07-29-v112-pareto-joint-feasibility`: exact alpha validation rescues
  one additional scene, but four-role Pareto reranking and 2,048 progressive
  command evaluations reach only K=2 4/7 and K=4 2/6 with no 11 dB reserve;
  algorithm-only expansion stops pending broader 10.04 GHz active matching.
- `2026-07-29-v113-broadband-match-replay`: frozen v1.12 masks and commands are
  replayed through uniform and three-class finite-Q broadband circuit networks;
  rebuilt three-frequency S256/EEP operators pass structural checks, but K=2
  4/7, K=4 2/6, and an empty 11 dB reserve keep algorithms and HFSS locked.
