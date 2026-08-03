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

## Gate Decision

The CAD build gate passes 6/6. The formal HFSS solve gate remains closed
because free memory after the build audit was about 12.53 GiB, below the
preregistered 13 GiB launch threshold. The 16-point 10 GHz Latin-hypercube DOE
is prepared but has not been solved.

Consequently, v1.21 currently provides no new S8, active-RL, efficiency,
direct/DDM, or integrated full-wave performance claim. Three-frequency
optimization, integrated solved validation, array expansion, labels, and
critic training remain locked.

## Resume Order

1. Re-audit that no AEDT process is running and free memory is at least 13 GiB.
2. Run the 16 serial 10 GHz network-only cases with the 3 GiB abort guard.
3. Analyze at least 12 complete cases, compute the physical Jacobian, and rank
   the Pareto set using RL, efficiency, and physical-to-target S8 error.
4. Run 3-5 Pareto candidates at 9.96/10.00/10.04 GHz.
5. Only after one candidate passes every three-frequency S8 gate, freeze it
   for independent direct/DDM validation and then one integrated 2x2 solve.

The resume commands are:

```powershell
python scripts\run_v121_parametric_feed_post.py --mode refresh-resource-gate
python scripts\run_v121_parametric_feed_post.py --mode run-doe
```
