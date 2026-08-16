# v1.50 Truncated-Coax Wave-Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v1.49 radial lumped port with a transverse external coaxial TEM wave port and a continuous finite-thickness ground transition, then run one immutable build smoke and, only after it passes, one memory-gated nominal continuation.

**Architecture:** Derive a separate v1.50 control script and configuration from the proven v1.49 orchestration so old runs remain readable and immutable. Change only the geometry, port, inventory, and nominal-gate logic required by the approved design; retain hash-bound continuation, single-use authorization, memory monitoring, Touchstone analysis, and finalization behavior.

**Tech Stack:** Python 3.9, standard-library `unittest`, NumPy, AEDT/HFSS 2023.1 VBS automation, Touchstone S3P, JSON/CSV/SHA-256 evidence.

## Global Constraints

- Work in `D:\codex_workspace\hfss_ura16_quick_model` on branch `codex/v150-truncated-coax-wave-port`.
- Do not modify or overwrite any v1.49 run, baseline, metric, or training artifact.
- Freeze the v1.49 radiator, SIW geometry, lattice, material, feed center, frequencies, scan grid, engineering thresholds, and downstream locks.
- Use one vacuum-filled 50-ohm truncated coax with `a=0.25 mm`, `b=0.575 mm`, outer radius `1.05 mm`, and length `1.0 mm`.
- Use analytic circular cylinders, a `0.035 mm` finite-thickness ground, and one ground/outer-conductor Boolean union.
- Use one transverse external wave port at `z=-1.0 mm`; no lumped port, radial `PortSheet`, contact overlap, axial port height, or de-embedding.
- Keep local mesh lengths `0.30/0.55/0.50 mm`, 5% adaptive refinement, `MaxDeltaS=0.05`, and no global `0.18 mm` mesh.
- Reject nominal evidence above 120,000 adaptive tetrahedra or below 3 GiB free host memory.
- Launch AEDT serially only with at least 13 GiB free memory and no AEDT/HFSS process.
- DOE, 1x1, 2x2, 4x4, 16x16, EEP, labels, and critic training stay locked throughout this plan.
- Never add the existing untracked `references/` or `tmp/` directories to Git.

---

### Task 1: Freeze v1.50 Configuration And Regression Contract

**Files:**
- Create: `configs/v150_truncated_coax_wave_port_preregistered.json`
- Create: `tests/test_v150_truncated_coax_wave_port.py`
- Create: `scripts/run_v150_truncated_coax_wave_port.py`

**Interfaces:**
- Consumes: immutable values from `configs/v149_siw_cavity_stacked_patch_preregistered.json`.
- Produces: `load_config()`, `validate_config()`, `validate_geometry()`, `geometry_audit()`, and the v1.50 CLI entry point.

- [ ] **Step 1: Write failing config tests**

```python
def test_v150_freezes_radiator_and_declares_transverse_wave_port(self):
    v150.validate_config(v150.load_config(CONFIG150))
    assert config["wave_port"]["type"] == "coaxial_modal_wave_port"
    assert config["wave_port"]["deembed"] is False
    assert config["nominal_geometry"]["coax_dielectric_outer_radius_mm"] == 0.575

def test_v150_rejects_changed_patch_or_legacy_lumped_port(self):
    changed = deepcopy(config)
    changed["nominal_geometry"]["driven_patch_length_mm"] += 0.1
    with self.assertRaises(ValueError):
        v150.validate_config(changed)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_v150_truncated_coax_wave_port -v`

Expected: import/file failure because the v1.50 files do not exist.

- [ ] **Step 3: Add the minimal v1.50 config and control scaffold**

Create a separate v1.50 script derived mechanically from v1.49, rename protocol/default paths/lock names, and implement strict comparisons against the v1.49 frozen radiator and SIW fields. Do not modify the v1.49 script.

- [ ] **Step 4: Verify GREEN and legacy compatibility**

Run:

```powershell
python -m unittest tests.test_v150_truncated_coax_wave_port -v
python -m unittest tests.test_v149_siw_cavity_stacked_patch -q
python -m py_compile scripts/run_v150_truncated_coax_wave_port.py
```

- [ ] **Step 5: Commit**

```powershell
git add configs/v150_truncated_coax_wave_port_preregistered.json scripts/run_v150_truncated_coax_wave_port.py tests/test_v150_truncated_coax_wave_port.py
git commit -m "feat: freeze v1.50 wave-port control plane"
```

---

### Task 2: Generate The Coax Wave Port And Ground Transition

**Files:**
- Modify: `scripts/run_v150_truncated_coax_wave_port.py`
- Modify: `tests/test_v150_truncated_coax_wave_port.py`

**Interfaces:**
- Consumes: `config["wave_port"]` and frozen `nominal_geometry`.
- Produces: `_element_geometry_text()`, `_mesh_and_setup_text()`, `periodic_builder_text()`, and deterministic inventory expectations.

- [ ] **Step 1: Write failing CAD-text tests**

Assert that generated VBS contains `NumSides:=0`, a finite ground solid,
`GroundOuterConductor`, deterministic bottom-face selection, one modal wave-port
assignment, and `DoDeembed:=False`. Assert that it contains neither
`AssignLumpedPort` nor `PortSheet` nor `Mesh_PortSheet`.

- [ ] **Step 2: Run the focused CAD tests and verify RED**

Run: `python -m unittest tests.test_v150_truncated_coax_wave_port.V150CadGenerationTests -v`

- [ ] **Step 3: Implement the minimal CAD change**

Generate exact coax shells and annuli, subtract the ground aperture, overlap
the outer conductor through the ground thickness, unite both into
`GroundOuterConductor`, select the annular dielectric bottom face at
`z=-1.0 mm`, and assign one wave mode with a radial integration line. Apply
the `0.30 mm` feed/coax object mesh without conductor-volume refinement.

- [ ] **Step 4: Verify generated topology**

Run the v1.50 focused tests, the complete v1.49 suite, `py_compile`, and
`git diff --check`.

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_v150_truncated_coax_wave_port.py tests/test_v150_truncated_coax_wave_port.py
git commit -m "feat: generate v1.50 truncated coax wave port"
```

---

### Task 3: Harden Build, Mesh, And Nominal Evidence Gates

**Files:**
- Modify: `scripts/run_v150_truncated_coax_wave_port.py`
- Modify: `tests/test_v150_truncated_coax_wave_port.py`

**Interfaces:**
- Consumes: AEDT inventory, validation log, run audit, profile, S3P, and source names.
- Produces: build gate, port audit, nominal numerical gate, nominal physical gate, and immutable stage decision.

- [ ] **Step 1: Write failing audit tests**

Add fixtures proving that build audit rejects a lumped feed, missing wave mode,
wrong face/inventory, or `PortSheet`; nominal audit rejects more than 120,000
tetrahedra, any small segment, RL below 15 dB, efficiency below 97%, incomplete
mode binding, or a changed artifact hash.

- [ ] **Step 2: Run focused audit tests and verify RED**

Run: `python -m unittest tests.test_v150_truncated_coax_wave_port.V150EvidenceGateTests -v`

- [ ] **Step 3: Implement the evidence gates**

Record `port_type`, selected face, analytic coax impedance, de-embed state,
mesh lengths, adaptive tetrahedra, minimum memory, warnings, three-frequency
RL/impedance/power, and separate numerical/physical pass flags. Ensure neither
flag unlocks a downstream stage in v1.50.

- [ ] **Step 4: Verify all tests**

Run both v1.50 and v1.49 suites, compile checks, and `git diff --check`.

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_v150_truncated_coax_wave_port.py tests/test_v150_truncated_coax_wave_port.py
git commit -m "feat: gate v1.50 wave-port evidence"
```

---

### Task 4: Execute One Build-Only HFSS Smoke

**Files:**
- Create: `hfss_outputs/v150_truncated_coax_wave_port_20260817_run01/**`

**Interfaces:**
- Consumes: committed v1.50 script/config and AEDT 2023.1.
- Produces: immutable preregistration, build project, validation, inventory, port audit, build gate, logs, and SHA-256 manifest.

- [ ] **Step 1: Verify execution preconditions**

Run tests, confirm the control script is tracked at HEAD, confirm zero AEDT
processes, and require at least 8 GiB free memory for build.

- [ ] **Step 2: Allocate and preregister a new run**

Run the v1.50 `preregister` command. Record Git, executable, input, config, and
resource hashes without touching prior output.

- [ ] **Step 3: Prepare and run build smoke**

Generate the VBS, run AEDT once without `Analyze`, and capture validation plus
inventory. Do not create a continuation if the AEDT return code or static gate
fails.

- [ ] **Step 4: Audit and seal the build**

Require `ValidateDesign=1`, one feed wave mode, two Floquet modes, correct
ground/coax objects, no `PortSheet`, and zero critical topology warning.

- [ ] **Step 5: Commit build evidence controls and report**

Commit only tracked script/config/test/docs changes. Raw `.aedtresults` remain
outside Git unless an existing baseline pattern explicitly includes a compact
evidence subset.

---

### Task 5: Create And Run One Single-Use Nominal Continuation

**Files:**
- Create: `hfss_outputs/v150_truncated_coax_wave_port_20260817_run02/**`
- Create: `docs/V150_TRUNCATED_COAX_WAVE_PORT_20260817.md`
- Modify: `docs/RESULTS_INDEX.md`

**Interfaces:**
- Consumes: sealed passing build and its SHA-256 manifest.
- Produces: one authorized nominal solve, three-frequency metrics, stage summary, immutable manifest, and project decision.

- [ ] **Step 1: Create a hash-bound continuation**

Copy the exact passing project into a new run, bind source project/build-gate
hashes, and authorize one nominal broadside sweep only.

- [ ] **Step 2: Apply launch and runtime resource guards**

If free memory is below 13 GiB, record a non-consuming preflight block and
wait for memory recovery. During the solve, terminate and preserve evidence if
free memory drops below 3 GiB. Never start a second AEDT instance.

- [ ] **Step 3: Analyze the completed solve**

Export and bind one feed mode plus two Floquet modes. Report return code,
tetrahedra, memory, Delta S, warnings, three-frequency RL, impedance,
reflection/accepted/Floquet power, and numerical/physical gates.

- [ ] **Step 4: Finalize without threshold changes**

Seal `stage_summary.json`, `stage_summary.md`, and `sha256_manifest.csv`. Keep
all downstream stages locked whether the nominal result passes or fails; only
record the next permitted physical action.

- [ ] **Step 5: Document and commit**

Append the measured outcome to the v1.50 report and result index. Commit and
push the branch without adding `references/` or `tmp/`.

---

### Task 6: Final Verification And Review

**Files:**
- Verify all files changed by Tasks 1-5.

**Interfaces:**
- Consumes: Git branch, test suites, sealed run manifest, process table.
- Produces: final verification record and concise user-facing decision.

- [ ] **Step 1: Run complete verification**

```powershell
python -m unittest tests.test_v150_truncated_coax_wave_port -q
python -m unittest tests.test_v149_siw_cavity_stacked_patch -q
python -m py_compile scripts/run_v150_truncated_coax_wave_port.py
git diff --check
```

- [ ] **Step 2: Verify evidence and process state**

Recompute the final run SHA-256 manifest, require zero AEDT/HFSS processes,
and verify the branch is synchronized with origin.

- [ ] **Step 3: Review scope and claims**

Confirm no v1.49 artifact changed, no downstream stage opened, and no
build/EEP/proxy result is described as a 16x16 full-wave conclusion.

- [ ] **Step 4: Commit and push final documentation**

Use a scoped commit message and push `codex/v150-truncated-coax-wave-port`.

- [ ] **Step 5: Report measured outcome**

State separately whether port topology, numerical convergence, memory safety,
and physical matching passed. Include absolute links to the report, build gate,
nominal analysis, and stage summary.
