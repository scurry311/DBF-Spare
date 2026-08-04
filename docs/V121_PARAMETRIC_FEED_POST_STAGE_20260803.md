# v1.21 Parametric Feed/POST Physical Calibration

## Scope

v1.21 replaces the unreliable circuit-to-geometry extrapolation in v1.20 with
one shared physical parameter vector for two HFSS models:

- network-only S8 with four PRE and four POST reference ports;
- integrated 2x2 S4 with the same feed, launch, sparse graph, and single POST
  stage connected to the trusted small-cell antenna geometry.

The physical neighbor chain remains `0-2-3-1`. No second decoupling stage,
non-neighbor connection, 4x4/16x16 expansion, label generation, or critic
training is authorized.

## Preregistered Parameters

Eleven dimensions are frozen. Five are observable in the low-cost network S8
DOE: common trace width, outer gap, center gap, shunt reference offset, and
common POST length correction. Six integrated-only dimensions remain at their
nominal values during that DOE: feed inset, transverse feed offset, probe
radius, ground clearance radius, launch pad radius, and launch taper length.

The simultaneous all-lower corner was rejected before HFSS because the port-0
target electrical length (22.123 mm) was shorter than its minimum Manhattan
route (23.360 mm). This failed construction is retained as an audit artifact;
the lower smoke point uses the preregistered upper POST-length bound and is
labeled `lower_safe`.

This route constraint was added as preregistration amendment 01 before any
S-parameter solve. The original files remain unchanged. In the original LHS,
`doe12_lhs` was short by about 0.033 mm, so only its common POST correction was
projected from -0.9587 mm to -0.9287 mm. All sixteen effective DOE candidates
now map to both shared CAD models; no threshold or parameter range changed.

## Stage A/B Results

The parent commit and tag both resolve to
`94aae2f694b406b777d3eaa50cb7b792dc2a2cb0`. No unrelated worktree changes or
running AEDT process were found at preregistration.

The first integrated build attempt failed after the patch/probe solids had
been united into `TraceRight`: a later mesh operation referred to the removed
pre-union names. This was a CAD bookkeeping failure, not an electromagnetic
result. The archived attempt remains under the local v1.21 output directory.

After assigning the 0.18 mm feed mesh to the surviving united conductor, all
six independent build cases passed:

| Candidate | Network S8 | Integrated 2x2 | Geometry warnings |
|---|---:|---:|---:|
| nominal | pass | pass | 0 |
| lower_safe | pass | pass | 0 |
| upper_bound | pass | pass | 0 |

The three network projects are approximately 226 KB. The integrated projects
are 1.32-1.35 MB. Port sets, six input/output neighbor-loading sheets, and the
single-stage constraint are valid in every generated project.

## Stage C Physical Results

The first nominal direct-solver resource probe produced no S8. Its initial
mesh had 356,315 tetrahedra, its matrix dimension reached 1,885,474, and HFSS
estimated about 14.55 GiB total memory. The preregistered 3 GiB free-memory
guard stopped the solve. Amendment 02 therefore changed only the low-cost
10 GHz DOE solver to the HFSS iterative solver with residual `1e-6`; geometry,
mesh, ranges, and engineering thresholds were unchanged. Independent direct
and DDM validation remained mandatory for any promoted candidate.

All 16 original DOE cases then converged and exported valid S8 files. None
passed the complete 10 GHz physical gate. The observed Pareto set contained
four candidates, but its best worst-case active RL was only 5.08 dB and its
best physical-to-target S8 error remained far above 0.10.

Four additional candidates were frozen from the observed Jacobian and Pareto
set before solving. They brought the physical review count to the
preregistered 20-case checkpoint. All four converged without a memory abort;
the lowest remaining system memory was 7.62 GiB.

| Local candidate | Passive RL (dB) | Active RL (dB) | Total RL (dB) | Efficiency | Target S8 error |
|---|---:|---:|---:|---:|---:|
| gradient corner | 12.93 | 2.57 | 9.66 | 96.60% | 0.293 |
| efficiency balance | 12.72 | 1.82 | 9.52 | 96.55% | 0.294 |
| DOE09 shunt fix | 13.44 | 3.08 | 8.95 | 96.54% | 0.335 |
| conservative knee | 13.40 | 3.71 | 9.15 | 96.47% | 0.335 |

Across all 20 cases, convergence/Delta-S and passivity were generally sound,
15 passed the 95% efficiency threshold, and 10 passed passive RL >= 10 dB.
However, zero passed active RL >= 11 dB, zero passed total RL >= 11 dB, and
zero achieved physical-to-target S8 error <= 0.10.

The linear response model predicted 8.12 dB active RL for the gradient corner,
whereas HFSS produced 2.57 dB. This is evidence that the first-order
sensitivity extrapolation is invalid at the multi-variable boundary: coupled
phase and modal changes dominate the independent trends. The linear model was
used only to select physical tests and is not promoted as performance
evidence.

## Stop Decision

The 20-case review gate is complete and the current single-stage POST/local
loading topology is stopped:

- best active RL: 5.08 dB, below the 10 dB stop threshold and 11 dB gate;
- best physical-to-target S8 error: 0.293, above the 0.15 stop threshold;
- complete 10 GHz network-gate passes: 0/20.

Three-frequency optimization, direct/DDM promotion, integrated 2x2 solving,
4x4/16x16 expansion, training-label generation, and critic retraining remain
locked. No v1.21 network-only result is described as an integrated antenna or
array result.

## Next Physical Model

The next minimum model should replace the feed body instead of adding another
decoupling stage. Start with a true balanced/differential launch plus one local
2x2 even/odd-mode correction branch, because the frozen failures are driven by
active modal reflection rather than passive loss. Keep PRE/POST reference
planes and transducer efficiency explicit.

Screen that model on 1x1 and one physical 2x2 at 9.96/10.00/10.04 GHz. If it
cannot provide passive RL >= 12 dB, representative active RL >= 11 dB,
efficiency >= 95%, and repeat-solve Delta-S <= 0.05, stop it before any larger
array. Aperture-coupled or dual-resonant coupled feeds are the next alternatives
when the balanced local-mode model cannot create the required bandwidth and
active-RL reserve.
