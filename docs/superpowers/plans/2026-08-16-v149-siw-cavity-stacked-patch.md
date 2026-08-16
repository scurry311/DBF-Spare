# v1.49 SIW Cavity Stacked Patch Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and validate a native, auditable HFSS 2023.1 workflow for a 10 GHz, 15 mm-period SIW cavity-backed dual-resonant stacked-patch element, while keeping 2x2, 4x4, 16x16, label generation, and critic training locked behind explicit physical gates.

**Architecture:** A single versioned Python orchestrator owns preregistration, CAD/VBS generation, serial AEDT execution, result parsing, DOE/Pareto ranking, stage gates, resource locks, and immutable run manifests. Pure calculation and gate functions are unit-tested independently from AEDT; generated VBS uses only native HFSS geometry and named boundaries/ports.

**Tech Stack:** Python 3, standard library, NumPy/Pandas/Matplotlib where already available, HFSS/AEDT 2023.1 VBS automation, Touchstone/CSV exports, unittest.

---

### Task 1: Freeze v1.49 Configuration And Gate Semantics

**Files:**
- Create: configs/v149_siw_cavity_stacked_patch_preregistered.json
- Create: tests/test_v149_siw_cavity_stacked_patch.py
- Create: scripts/run_v149_siw_cavity_stacked_patch.py

**Step 1:** Write failing tests for allowed geometry ranges, scan-state enumeration, stage locks, memory gates, and periodic gate evaluation.

**Step 2:** Run the focused test file and confirm the expected failures.

**Step 3:** Implement the preregistration loader, immutable run allocation, scan-state generator, gate evaluator, and resource checks.

**Step 4:** Re-run tests and compile the script.

**Step 5:** Commit the configuration and control-plane implementation.

### Task 2: Generate Native Periodic/1x1 HFSS CAD

**Files:**
- Modify: scripts/run_v149_siw_cavity_stacked_patch.py
- Modify: tests/test_v149_siw_cavity_stacked_patch.py

**Step 1:** Add failing tests for generated HFSS 2023.1 VBS features: RO5880 material, finite-conductivity copper, SIW vias, driven/parasitic patches, explicit feed, linked boundaries, Floquet port, local mesh, and named exports.

**Step 2:** Implement native parameterized periodic-unit and finite 1x1 VBS builders.

**Step 3:** Add a build-only smoke mode that does not solve and writes logs plus a geometry audit manifest.

**Step 4:** Run unit tests, Python compile checks, and AEDT build/import smoke serially.

**Step 5:** Commit the CAD generation implementation.

### Task 3: Add DOE, Analysis, Pareto Ranking, And Stage Gates

**Files:**
- Modify: scripts/run_v149_siw_cavity_stacked_patch.py
- Modify: tests/test_v149_siw_cavity_stacked_patch.py

**Step 1:** Add failing tests for deterministic 12-20 point DOE generation, active-RL calculations, scan-blindness detection, Pareto dominance, and no-evidence/no-pass behavior.

**Step 2:** Implement DOE manifests, Touchstone/CSV analysis, scan heatmaps, convergence/resource summaries, and Pareto selection limited to 3-5 candidates.

**Step 3:** Implement strict periodic and finite-array gate reports that cannot infer 16x16 conclusions.

**Step 4:** Run focused and repository compile tests.

**Step 5:** Commit analysis and gate logic.

### Task 4: Execute v1.49 Stage-A Preparation And Build Smoke

**Files:**
- Create: hfss_outputs/v149_siw_cavity_stacked_patch_20260816_runXX/preregistration.json
- Create: hfss_outputs/v149_siw_cavity_stacked_patch_20260816_runXX/manifests/*
- Create: hfss_outputs/v149_siw_cavity_stacked_patch_20260816_runXX/logs/*
- Create: hfss_outputs/v149_siw_cavity_stacked_patch_20260816_runXX/stage_summary.md

**Step 1:** Allocate a new run directory without modifying prior results.

**Step 2:** Record Git state, hashes, AEDT version/path, process state, RAM/disk state, variables, thresholds, and stop rules.

**Step 3:** Generate DOE and all VBS/build artifacts.

**Step 4:** Run one non-solving AEDT build/import smoke if the process and memory preflight passes.

**Step 5:** Audit the saved project/log for topology, ports, boundary names, and critical warnings.

### Task 5: Decide Whether A Real Periodic Solve May Start

**Files:**
- Modify: hfss_outputs/v149_siw_cavity_stacked_patch_20260816_runXX/stage_summary.md
- Create when eligible: hfss_outputs/v149_siw_cavity_stacked_patch_20260816_runXX/periodic/*

**Step 1:** Recheck available memory and AEDT process count immediately before solving.

**Step 2:** If free memory is at least 13 GiB and build smoke passed, run only the nominal periodic candidate serially.

**Step 3:** Abort and preserve logs if free memory falls below 3 GiB.

**Step 4:** Analyze real HFSS evidence and issue a strict pass/fail decision before any DOE batch or 1x1 progression.

**Step 5:** Keep 2x2, 4x4, 16x16, HFSS label generation, and critic training locked unless their preceding gates pass.

### Task 6: Version And Verify

**Files:**
- Create: baselines/2026-08-16-v149-siw-cavity-stacked-patch/* only after a physical gate passes
- Modify: project result index/readme files only if an established local pattern requires it

**Step 1:** Run unit tests, Python compile checks, hash verification, and immutable-output checks.

**Step 2:** Generate SHA-256 manifests for new scripts, config, and run evidence.

**Step 3:** Commit implementation and evidence without adding unrelated untracked directories.

**Step 4:** Create a baseline/tag only when backed by passing real HFSS metrics.

**Step 5:** Report measured results, gate status, resource blocker if any, and the next permitted action.
