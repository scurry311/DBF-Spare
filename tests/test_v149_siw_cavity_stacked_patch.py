import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v149_siw_cavity_stacked_patch.py"
CONFIG = ROOT / "configs" / "v149_siw_cavity_stacked_patch_preregistered.json"


def load_module():
    spec = importlib.util.spec_from_file_location("v149", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class V149ControlPlaneTests(unittest.TestCase):
    def setUp(self):
        self.v149 = load_module()
        self.config = self.v149.load_config(CONFIG)

    def test_preregistered_geometry_and_scope_are_frozen(self):
        self.v149.validate_config(self.config)
        geometry = self.config["nominal_geometry"]
        self.assertEqual(geometry["period_x_mm"], 15.0)
        self.assertEqual(geometry["period_y_mm"], 15.0)
        total_height = (
            geometry["main_substrate_thickness_mm"]
            + geometry["stack_spacer_thickness_mm"]
            + geometry["copper_thickness_mm"]
        )
        self.assertLessEqual(total_height, 3.5)
        self.assertFalse(self.config["scope"]["allow_2x2"])
        self.assertFalse(self.config["scope"]["allow_4x4"])
        self.assertFalse(self.config["scope"]["allow_16x16"])
        self.assertFalse(self.config["scope"]["allow_training_labels"])
        self.assertFalse(self.config["scope"]["allow_critic_training"])

    def test_resource_and_engineering_gates_cannot_be_lowered(self):
        changed = json.loads(json.dumps(self.config))
        changed["resources"]["minimum_free_memory_before_solve_gib"] = 0.0
        with self.assertRaises(ValueError):
            self.v149.validate_config(changed)
        changed = json.loads(json.dumps(self.config))
        changed["scan"]["theta_deg"] = [0.0, 10.0, 20.0, 30.0, 40.0]
        with self.assertRaises(ValueError):
            self.v149.validate_config(changed)
        changed = json.loads(json.dumps(self.config))
        changed["gates"]["minimum_periodic_active_rl_db"] = 0.0
        with self.assertRaises(ValueError):
            self.v149.validate_config(changed)

    def test_scan_states_cover_all_preregistered_corners(self):
        states = self.v149.scan_states(self.config)
        self.assertEqual(len(states), 45)
        self.assertIn(
            {"frequency_ghz": 10.04, "theta_deg": 48.0, "phi_deg": 90.0},
            states,
        )
        self.assertEqual(
            {state["frequency_ghz"] for state in states},
            {9.96, 10.0, 10.04},
        )

    def test_memory_and_single_instance_guard(self):
        self.assertTrue(
            self.v149.memory_allows_stage(
                self.config, free_memory_gib=13.0, aedt_instance_count=0
            )
        )
        self.assertFalse(
            self.v149.memory_allows_stage(
                self.config, free_memory_gib=12.99, aedt_instance_count=0
            )
        )
        self.assertFalse(
            self.v149.memory_allows_stage(
                self.config, free_memory_gib=24.0, aedt_instance_count=1
            )
        )

    def test_periodic_gate_requires_complete_real_hfss_evidence(self):
        metrics = {
            "evidence_source": "HFSS_periodic_fullwave",
            "evidence_complete": True,
            "provenance_complete": True,
            "physical_attestation_complete": True,
            "physical_gate_armed": True,
            "scan_state_count": 45,
            "minimum_active_rl_db": 13.2,
            "broadside_passive_rl_db": 16.0,
            "minimum_efficiency": 0.975,
            "maximum_scan_gain_drop_db": 2.7,
            "maximum_final_delta_s": 0.04,
            "scan_blindness_detected": False,
            "critical_warning_count": 0,
        }
        decision = self.v149.evaluate_periodic_gate(metrics, self.config["gates"])
        self.assertFalse(decision["passed"])
        self.assertIn("physical_attestation", decision["failed_checks"])
        self.assertIn("stage_a_gate_armed", decision["failed_checks"])

        incomplete = dict(metrics, evidence_complete=False)
        self.assertFalse(
            self.v149.evaluate_periodic_gate(
                incomplete, self.config["gates"]
            )["passed"]
        )

        proxy = dict(metrics, evidence_source="surrogate")
        self.assertFalse(
            self.v149.evaluate_periodic_gate(proxy, self.config["gates"])[
                "passed"
            ]
        )
        nan_metrics = dict(metrics, minimum_active_rl_db=math.nan)
        self.assertFalse(
            self.v149.evaluate_periodic_gate(
                nan_metrics, self.config["gates"]
            )["passed"]
        )

    def test_failed_periodic_gate_keeps_every_later_stage_locked(self):
        metrics = {
            "evidence_source": "HFSS_periodic_fullwave",
            "evidence_complete": True,
            "provenance_complete": True,
            "physical_attestation_complete": True,
            "physical_gate_armed": True,
            "scan_state_count": 45,
            "minimum_active_rl_db": 12.99,
            "broadside_passive_rl_db": 18.0,
            "minimum_efficiency": 0.99,
            "maximum_scan_gain_drop_db": 2.0,
            "maximum_final_delta_s": 0.03,
            "scan_blindness_detected": False,
            "critical_warning_count": 0,
        }
        decision = self.v149.stage_decision_after_periodic(
            metrics, self.config["gates"]
        )
        self.assertFalse(decision["allow_1x1"])
        self.assertFalse(decision["allow_2x2"])
        self.assertFalse(decision["allow_4x4"])
        self.assertFalse(decision["allow_16x16"])
        self.assertFalse(decision["allow_training_labels"])
        self.assertFalse(decision["allow_critic_training"])


class V149CadGenerationTests(unittest.TestCase):
    def setUp(self):
        self.v149 = load_module()
        self.config = self.v149.load_config(CONFIG)
        self.geometry = self.config["nominal_geometry"]

    def test_periodic_builder_contains_native_auditable_features(self):
        source = self.v149.periodic_builder_text(
            Path("D:/scratch/v149_periodic.aedt"),
            self.config,
            self.geometry,
            {"frequency_ghz": 10.0, "theta_deg": 0.0, "phi_deg": 0.0},
        )
        required = [
            "RO5880_V149",
            "DrivenPatch",
            "StackedPatch",
            "SIWCavityTop",
            "CavityPatchAperture",
            "SIWVia_",
            "FeedProbe",
            "CoaxOuter",
            "PortSheet",
            "AssignLumpedPort",
            "AssignFiniteCond",
            "AssignPrimary",
            "AssignSecondary",
            "AssignFloquetPort",
            "Mesh_ProbeLaunch",
            "Mesh_PatchEdges",
            "Mesh_SIWVias",
        ]
        for token in required:
            self.assertIn(token, source)
        self.assertIn("Sweep_9p8_10p2", source)
        self.assertIn("model_inventory.txt", source)
        self.assertIn("GetMatchedObjectName", source)
        self.assertIn("GetBoundaries", source)
        self.assertIn("GetExcitations", source)
        self.assertIn(
            '"NAME:SecondaryX", Array("NAME:CoordSysVector"',
            source,
        )
        self.assertIn(
            '"NAME:SecondaryY", Array("NAME:CoordSysVector"',
            source,
        )
        secondary_lines = [
            line
            for line in source.splitlines()
            if "AssignSecondary" in line
        ]
        self.assertEqual(len(secondary_lines), 2)
        self.assertTrue(
            all('"ReverseV:=", True' in line for line in secondary_lines)
        )
        self.assertNotIn(
            'UniteSelection oEditor, "DrivenPatch,FeedProbe"', source
        )
        self.assertNotIn(
            'UniteSelection oEditor, "Ground,CoaxOuter"', source
        )
        self.assertIn(
            'SubtractObject oEditor, "SIWCavityTop", "CavityPatchAperture"',
            source,
        )
        self.assertNotIn(
            'SubtractObject oEditor, "MainSubstrate,StackSpacer", '
            '"SIWViaCut_',
            source,
        )
        self.assertNotIn("Global0p18mm", source)
        self.assertNotIn("a3dcomp", source.lower())

    def test_saved_model_inventory_must_contain_real_objects_and_boundaries(self):
        lines = [
            "OBJECT|MainSubstrate",
            "OBJECT|StackSpacer",
            "OBJECT|Ground",
            "OBJECT|SIWCavityTop",
            "OBJECT|DrivenPatch",
            "OBJECT|StackedPatch",
            "OBJECT|FeedProbe",
            "OBJECT|CoaxOuter",
            "OBJECT|CoaxDielectric",
            "OBJECT|PortSheet",
            "OBJECT|AirCell",
            *[
                f"OBJECT|SIWVia_{index:03d}"
                for index in range(
                    self.v149.geometry_audit(
                        self.config, self.geometry
                    )["siw_via_count"]
                )
            ],
            "BOUNDARY|CopperSheetFiniteConductivity",
            "BOUNDARY|PrimaryX",
            "BOUNDARY|SecondaryX",
            "BOUNDARY|PrimaryY",
            "BOUNDARY|SecondaryY",
            "EXCITATION|FeedPort:1|Lumped Port",
            "EXCITATION|FloquetTop:1|Floquet Port",
            "EXCITATION|FloquetTop:2|Floquet Port",
        ]
        result = self.v149.validate_model_inventory(
            lines, self.config, self.geometry
        )
        self.assertTrue(result["verified"])
        missing = self.v149.validate_model_inventory(
            [line for line in lines if "FloquetTop" not in line],
            self.config,
            self.geometry,
        )
        self.assertFalse(missing["verified"])

    def test_finite_one_by_one_uses_radiation_boundary_not_floquet(self):
        source = self.v149.finite_one_by_one_builder_text(
            Path("D:/scratch/v149_1x1.aedt"),
            self.config,
            self.geometry,
            solver_type="direct",
        )
        self.assertIn("AssignRadiation", source)
        self.assertIn("DrivenPatch", source)
        self.assertIn("StackedPatch", source)
        self.assertNotIn("AssignFloquetPort", source)
        self.assertNotIn("AssignPrimary", source)

    def test_geometry_audit_reports_single_feed_and_via_fence(self):
        audit = self.v149.geometry_audit(self.config, self.geometry)
        self.assertEqual(audit["feed_port_count"], 1)
        self.assertEqual(audit["driven_patch_count"], 1)
        self.assertEqual(audit["stacked_patch_count"], 1)
        self.assertGreaterEqual(audit["siw_via_count"], 24)
        self.assertTrue(audit["patches_clear_via_fence"])
        self.assertTrue(audit["feed_clears_siw_vias"])
        self.assertTrue(audit["total_height_gate"])

    def test_invalid_patch_or_via_geometry_is_rejected(self):
        oversized = dict(self.geometry, driven_patch_width_mm=14.8)
        with self.assertRaises(ValueError):
            self.v149.validate_geometry(self.config, oversized)
        bad_vias = dict(
            self.geometry,
            siw_via_diameter_mm=0.6,
            siw_via_pitch_mm=1.5,
        )
        with self.assertRaises(ValueError):
            self.v149.validate_geometry(self.config, bad_vias)

    def test_prepare_build_smoke_is_immutable_and_keeps_finite_stages_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run01"
            root.mkdir()
            result = self.v149.prepare_periodic_build_smoke(
                root, self.config
            )
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_scope"], "periodic_unit_only")
            self.assertFalse(manifest["authorizes_real_solve"])
            self.assertTrue(Path(manifest["builder_path"]).exists())
            self.assertFalse((root / "finite_1x1").exists())
            with self.assertRaises(FileExistsError):
                self.v149.prepare_periodic_build_smoke(root, self.config)

    def test_build_smoke_preflight_blocks_below_memory_threshold(self):
        decision = self.v149.build_smoke_preflight(
            self.config,
            free_memory_gib=7.99,
            aedt_instance_count=0,
        )
        self.assertFalse(decision["allowed"])
        self.assertIn("memory", decision["reason"].lower())
        allowed = self.v149.build_smoke_preflight(
            self.config,
            free_memory_gib=10.0,
            aedt_instance_count=0,
        )
        self.assertTrue(allowed["allowed"])

    def test_nominal_solve_preflight_keeps_thirteen_gib_gate(self):
        blocked = self.v149.solve_preflight(
            self.config,
            free_memory_gib=12.99,
            aedt_instance_count=0,
        )
        self.assertFalse(blocked["allowed"])
        allowed = self.v149.solve_preflight(
            self.config,
            free_memory_gib=13.0,
            aedt_instance_count=0,
        )
        self.assertTrue(allowed["allowed"])

    def test_nominal_solver_exports_touchstone_without_optimization(self):
        source = self.v149.periodic_solver_text(
            Path("D:/scratch/v149_periodic.aedt"),
            Path("D:/scratch/v149_periodic.s3p"),
            Path("D:/scratch/sources.txt"),
            self.config,
        )
        self.assertIn('oDesign.Analyze "Setup_10GHz"', source)
        self.assertIn("ExportNetworkData", source)
        self.assertIn('15, True, True, False', source)
        self.assertIn("Sweep_9p8_10p2", source)
        self.assertNotIn("Optimetrics", source)
        self.assertNotIn("critic", source.lower())

    def test_nominal_solve_preparation_requires_passing_build_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            build = root / "periodic" / "build_smoke"
            build.mkdir(parents=True)
            project = build / "built.aedt"
            project.write_bytes(b"audited-build")
            (build / "case_manifest.json").write_text(
                json.dumps({"project_path": str(project)}),
                encoding="utf-8",
            )
            (build / "build_gate.json").write_text(
                json.dumps({"build_smoke_passed": False}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                self.v149.prepare_nominal_periodic_solve(
                    root, self.config
                )
            (build / "build_gate.json").write_text(
                json.dumps(
                    {
                        "build_smoke_passed": True,
                        "model_inventory_verified": True,
                        "project_sha256": self.v149.sha256(project),
                    }
                ),
                encoding="utf-8",
            )
            result = self.v149.prepare_nominal_periodic_solve(
                root, self.config
            )
            self.assertTrue(Path(result["project_path"]).exists())
            self.assertTrue(Path(result["solver_path"]).exists())
            self.assertFalse(result["solve_started"])

    def test_nominal_solve_rejects_mutated_project_or_solver_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "periodic" / "nominal_solve"
            folder.mkdir(parents=True)
            project = folder / "nominal.aedt"
            solver = folder / "solve.vbs"
            project.write_bytes(b"frozen-project")
            solver.write_text("frozen-solver", encoding="ascii")
            manifest = {
                "case_id": "periodic_nominal_broadside",
                "project_path": str(project),
                "solver_path": str(solver),
                "project_sha256_before_solve": self.v149.sha256(project),
                "solver_sha256": self.v149.sha256(solver),
            }
            (folder / "case_manifest.json").write_text(
                json.dumps(manifest), encoding="ascii"
            )
            solver.write_text("mutated-solver", encoding="ascii")
            with self.assertRaises(RuntimeError):
                self.v149.run_nominal_periodic_solve(root, self.config)

    def test_finalized_run_cannot_be_mutated_by_nominal_solve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stage_summary.json").write_text("{}", encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "finalized and immutable"):
                self.v149.run_nominal_periodic_solve(root, self.config)

    def test_run_config_hash_detects_post_preregistration_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = root / "preregistration.json"
            prereg.write_text(
                json.dumps(self.config, sort_keys=True), encoding="utf-8"
            )
            (root / "baseline_audit.json").write_text(
                json.dumps(
                    {
                        "preregistration_sha256": self.v149.sha256(prereg)
                    }
                ),
                encoding="utf-8",
            )
            loaded = self.v149.load_run_config(root)
            self.assertEqual(loaded["protocol"], self.config["protocol"])
            changed = json.loads(prereg.read_text(encoding="utf-8"))
            changed["gates"]["minimum_periodic_active_rl_db"] = 0.0
            prereg.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self.v149.load_run_config(root)

    def test_aedt_execution_lock_is_atomic_and_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "aedt.lock"
            with self.v149.aedt_execution_lock(lock):
                self.assertTrue(lock.exists())
                with self.assertRaises(RuntimeError):
                    with self.v149.aedt_execution_lock(lock):
                        pass
            self.assertFalse(lock.exists())


class V149DoeAndAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.v149 = load_module()
        self.config = self.v149.load_config(CONFIG)

    def test_doe_is_deterministic_unique_and_physically_valid(self):
        first = self.v149.generate_periodic_doe(self.config)
        second = self.v149.generate_periodic_doe(self.config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        signatures = {
            tuple(
                row[name]
                for name in sorted(self.config["manufacturing_ranges"])
            )
            for row in first
        }
        self.assertEqual(len(signatures), 16)
        for row in first:
            geometry = dict(self.config["nominal_geometry"])
            geometry.update(
                {
                    name: row[name]
                    for name in self.config["manufacturing_ranges"]
                }
            )
            self.v149.validate_geometry(self.config, geometry)

    def test_active_rl_uses_significant_port_semantics_and_total_power(self):
        s = np.diag([0.1 + 0j, 0.1 + 0j])
        excitation = np.array([1.0 + 0j, 0.01 + 0j])
        metrics = self.v149.active_return_metrics(
            s, excitation, significant_relative_amplitude=0.05
        )
        self.assertAlmostEqual(metrics["minimum_significant_active_rl_db"], 20.0)
        self.assertAlmostEqual(metrics["total_rl_db"], 20.0)
        self.assertEqual(metrics["significant_port_count"], 1)
        self.assertEqual(metrics["all_nonzero_port_count"], 2)

    def test_touchstone_parser_and_nominal_periodic_analysis_are_physical(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            ansys_executable = folder / "ansysedt.exe"
            ansys_executable.write_bytes(b"MZ-v149-test-executable")
            analysis_config = json.loads(json.dumps(self.config))
            analysis_config["ansys_executable"] = str(ansys_executable)
            touchstone = folder / "nominal.s3p"
            source_names = folder / "source_names.txt"
            project = folder / "nominal.aedt"
            solver = folder / "solve_export.vbs"
            solver_log = folder / "solve_export.log"
            project.write_bytes(b"hfss-project")
            solver.write_text("frozen-solver", encoding="ascii")
            solver_log.write_text("HFSS completed", encoding="ascii")
            source_names.write_text(
                "FeedPort:1\nFloquetTop:1\nFloquetTop:2\n",
                encoding="ascii",
            )
            # Touchstone ordering is column-major: S11,S21,S31,S12,...
            rows = []
            for frequency in (9.96, 10.00, 10.04):
                values = [
                    (0.00, 0.0),
                    (0.00, 0.0),
                    (0.00, 0.0),
                    (0.70, 0.0),
                    (0.10, 0.0),
                    (0.70, 0.0),
                    (0.00, 0.0),
                    (0.00, 0.0),
                    (0.00, 0.0),
                ]
                rows.append(
                    " ".join(
                        [str(frequency)]
                        + [str(token) for pair in values for token in pair]
                    )
                )
            touchstone.write_text(
                "! Port[1] = FloquetTop:1\n"
                "! Port[2] = FeedPort:1\n"
                "! Port[3] = FloquetTop:2\n"
                "# GHz S MA R 50\n" + "\n".join(rows) + "\n",
                encoding="ascii",
            )
            (folder / "solve.profile").write_text(
                "Adaptive Passes converged\nMax Mag. Delta S', 0.04\n",
                encoding="ascii",
            )
            manifest = {
                "case_id": "periodic_nominal_broadside",
                "scan_state": {"theta_deg": 0.0, "phi_deg": 0.0},
                "project_path": str(project),
                "touchstone_path": str(touchstone),
                "source_names_path": str(source_names),
                "solver_path": str(solver),
                "solver_sha256": self.v149.sha256(solver),
            }
            audit = {
                "schema": "v149_nominal_periodic_run_audit_v1",
                "aedt_version": "2023.1",
                "ansys_executable_sha256": self.v149.sha256(
                    ansys_executable
                ),
                "started": True,
                "blocked": False,
                "return_code": 0,
                "memory_aborted": False,
                "touchstone_exists": True,
                "source_names_exist": True,
                "project_sha256_after_solve": self.v149.sha256(project),
                "touchstone_sha256": self.v149.sha256(touchstone),
                "source_names_sha256": self.v149.sha256(source_names),
                "solver_sha256_executed": self.v149.sha256(solver),
                "solver_log_sha256": self.v149.sha256(solver_log),
                "convergence_artifacts": (
                    self.v149.convergence_artifact_manifest(folder)
                ),
                "result_directory": str(folder),
            }
            frequencies, matrices, reference, port_names = (
                self.v149.parse_touchstone(touchstone, 3)
            )
            self.assertEqual(list(frequencies), [9.96, 10.0, 10.04])
            self.assertEqual(reference, 50.0)
            self.assertEqual(
                port_names,
                ["FloquetTop:1", "FeedPort:1", "FloquetTop:2"],
            )
            self.assertAlmostEqual(abs(matrices[0, 1, 1]), 0.1)
            self.assertAlmostEqual(
                abs(matrices[0, 0, 1]) ** 2
                + abs(matrices[0, 2, 1]) ** 2,
                0.98,
            )
            # Synthetic files can validate parser math but cannot pass a
            # real-HFSS evidence gate.
            with self.assertRaises(RuntimeError):
                self.v149.analyze_nominal_periodic_exports(
                    manifest, audit, analysis_config, solver_log
                )

    def test_nominal_periodic_analysis_rejects_hash_or_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            touchstone = folder / "nominal.s3p"
            touchstone.write_text(
                "# GHz S RI R 50\n10 0.1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
                encoding="ascii",
            )
            source_names = folder / "source_names.txt"
            source_names.write_text("FeedPort:1\nFloquetTop:1\n", encoding="ascii")
            project = folder / "nominal.aedt"
            project.write_bytes(b"hfss-project")
            solver_log = folder / "solve_export.log"
            solver_log.write_text("HFSS completed", encoding="ascii")
            manifest = {
                "case_id": "periodic_nominal_broadside",
                "scan_state": {"theta_deg": 0.0, "phi_deg": 0.0},
                "project_path": str(project),
                "touchstone_path": str(touchstone),
                "source_names_path": str(source_names),
            }
            audit = {
                "started": True,
                "blocked": False,
                "return_code": 0,
                "memory_aborted": False,
                "touchstone_exists": True,
                "source_names_exist": True,
                "project_sha256_after_solve": "0" * 64,
                "touchstone_sha256": self.v149.sha256(touchstone),
                "source_names_sha256": self.v149.sha256(source_names),
                "solver_log_sha256": self.v149.sha256(solver_log),
            }
            with self.assertRaises(RuntimeError):
                self.v149.analyze_nominal_periodic_exports(
                    manifest, audit, self.config, solver_log
                )

    def test_pareto_selection_does_not_collapse_to_active_rl_top_one(self):
        rows = [
            {
                "candidate_id": "active_best",
                "minimum_active_rl_db": 15.0,
                "minimum_efficiency": 0.970,
                "maximum_scan_gain_drop_db": 3.0,
                "maximum_final_delta_s": 0.04,
            },
            {
                "candidate_id": "balanced",
                "minimum_active_rl_db": 14.0,
                "minimum_efficiency": 0.985,
                "maximum_scan_gain_drop_db": 2.0,
                "maximum_final_delta_s": 0.03,
            },
            {
                "candidate_id": "dominated",
                "minimum_active_rl_db": 12.0,
                "minimum_efficiency": 0.960,
                "maximum_scan_gain_drop_db": 4.0,
                "maximum_final_delta_s": 0.06,
            },
        ]
        selected = self.v149.select_pareto_candidates(rows, 4)
        ids = {row["candidate_id"] for row in selected}
        self.assertEqual(len(selected), 3)
        self.assertIn("active_best", ids)
        self.assertIn("balanced", ids)
        self.assertIn("dominated", ids)

    def test_scan_aggregation_requires_all_real_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            ansys_executable = Path(self.config["ansys_executable"])
            rows = []
            for index, state in enumerate(
                self.v149.scan_states(self.config)
            ):
                case = folder / f"case_{index:02d}"
                case.mkdir()
                project = case / "project.aedt"
                touchstone = case / "state.s3p"
                source_names = case / "source_names.txt"
                solver_script = case / "solve.vbs"
                solver_log = case / "solve.log"
                model_state_audit = case / "model_state.json"
                gain_report = case / "realized_gain.csv"
                convergence_profile = case / "solve.profile"
                geometry_error = case / "solve.g3derr"
                project.write_text(
                    "$begin 'AnsoftProject'\n"
                    "$begin 'V149_PeriodicUnit'\n"
                    "PrimaryX SecondaryX PrimaryY SecondaryY "
                    "FeedPort FloquetTop\n"
                    f"state={index}\n"
                    + " " * 100_000,
                    encoding="ascii",
                )
                source_names.write_text(
                    "FeedPort:1\nFloquetTop:1\nFloquetTop:2\n",
                    encoding="ascii",
                )
                solver_script.write_text(
                    f"' frozen HFSS state solver {index}", encoding="ascii"
                )
                solver_log.write_text(
                    f"HFSS state {index} completed", encoding="ascii"
                )
                model_state_audit.write_text(
                    json.dumps(state, sort_keys=True), encoding="ascii"
                )
                gain_value = 7.0 - 2.5 * state["theta_deg"] / 48.0
                gain_report.write_text(
                    "frequency_ghz,theta_deg,phi_deg,realized_gain_db\n"
                    f"{state['frequency_ghz']},{state['theta_deg']},"
                    f"{state['phi_deg']},{gain_value}\n",
                    encoding="ascii",
                )
                convergence_profile.write_text(
                    "Adaptive Passes converged\n"
                    f"State {index} Max Mag. Delta S', 0.04\n",
                    encoding="ascii",
                )
                geometry_error.write_text("", encoding="ascii")
                gamma = 10.0 ** (-16.0 / 20.0)
                accepted = 1.0 - gamma**2
                outgoing = math.sqrt(accepted * 0.98 / 2.0)
                s_values = [
                    (gamma, 0.0),
                    (outgoing, 0.0),
                    (outgoing, 0.0),
                    *((0.0, 0.0),) * 6,
                ]
                touchstone.write_text(
                    "! Port[1] = FeedPort:1\n"
                    "! Port[2] = FloquetTop:1\n"
                    "! Port[3] = FloquetTop:2\n"
                    "# GHz S MA R 50\n"
                    + " ".join(
                        [str(state["frequency_ghz"])]
                        + [
                            f"{token:.15g}"
                            for pair in s_values
                            for token in pair
                        ]
                    )
                    + "\n",
                    encoding="ascii",
                )
                artifact_files = {
                    "project": project,
                    "touchstone": touchstone,
                    "source_names": source_names,
                    "solver_script": solver_script,
                    "solver_log": solver_log,
                    "model_state_audit": model_state_audit,
                    "gain_report": gain_report,
                    "convergence_profile": convergence_profile,
                    "geometry_error": geometry_error,
                }
                run_audit = case / "run_audit.json"
                run_audit.write_text(
                    json.dumps(
                        {
                            "schema": "v149_periodic_state_run_audit_v1",
                            "aedt_version": "2023.1",
                            "ansys_executable_sha256": self.v149.sha256(
                                ansys_executable
                            ),
                            "started": True,
                            "blocked": False,
                            "return_code": 0,
                            "memory_aborted": False,
                            "solver_sha256_executed": self.v149.sha256(
                                solver_script
                            ),
                            "result_directory": str(case),
                            "artifact_sha256": {
                                name: self.v149.sha256(path)
                                for name, path in artifact_files.items()
                            },
                            "convergence_artifacts": (
                                self.v149.convergence_artifact_manifest(case)
                            ),
                        },
                        sort_keys=True,
                    ),
                    encoding="ascii",
                )
                artifact_files["run_audit"] = run_audit
                artifacts = {
                    name: {
                        "path": str(path),
                        "sha256": self.v149.sha256(path),
                    }
                    for name, path in artifact_files.items()
                }
                row = {
                    **state,
                    "case_id": f"periodic_state_{index:02d}",
                    "evidence_source": "HFSS_periodic_fullwave",
                    "solve_complete": True,
                    "provenance_verified": True,
                    "active_rl_db": 16.0,
                    "passive_rl_db": 16.0,
                    "efficiency": 0.98,
                    "realized_gain_db": gain_value,
                    "final_delta_s": 0.04,
                    "scan_blindness": False,
                    "critical_warning_count": 0,
                }
                evidence = {
                    "case_id": row["case_id"],
                    "scan_state": state,
                    "evidence_source": row["evidence_source"],
                    "solve_complete": True,
                    "prepared_solver_sha256": self.v149.sha256(
                        solver_script
                    ),
                    "solver_sha256_executed": self.v149.sha256(
                        solver_script
                    ),
                    "artifacts": artifacts,
                    "metrics": {
                        key: row[key]
                        for key in (
                            "active_rl_db",
                            "passive_rl_db",
                            "efficiency",
                            "realized_gain_db",
                            "final_delta_s",
                            "scan_blindness",
                            "critical_warning_count",
                        )
                    },
                }
                evidence_path = case / "state_evidence.json"
                evidence_path.write_text(
                    json.dumps(evidence, sort_keys=True), encoding="ascii"
                )
                row["evidence_manifest_path"] = str(evidence_path)
                row["evidence_manifest_sha256"] = self.v149.sha256(
                    evidence_path
                )
                rows.append(row)
            metrics = self.v149.aggregate_periodic_scan(rows, self.config)
            self.assertFalse(metrics["evidence_complete"])
            self.assertFalse(metrics["provenance_complete"])
            self.assertFalse(metrics["physical_attestation_complete"])
            self.assertFalse(metrics["physical_gate_armed"])
            self.assertEqual(metrics["scan_state_count"], 45)
            self.assertLessEqual(metrics["maximum_scan_gain_drop_db"], 2.5)
            self.assertFalse(
                self.v149.evaluate_periodic_gate(
                    metrics, self.config["gates"]
                )["passed"]
            )
            incomplete = self.v149.aggregate_periodic_scan(
                rows[:-1], self.config
            )
            self.assertFalse(incomplete["evidence_complete"])
            contaminated = [dict(row) for row in rows]
            contaminated[-1]["efficiency"] = math.nan
            metrics = self.v149.aggregate_periodic_scan(
                contaminated, self.config
            )
            self.assertFalse(metrics["evidence_complete"])
            self.assertFalse(metrics["provenance_complete"])
            copied = [dict(row) for row in rows]
            copied[-1]["evidence_manifest_path"] = rows[0][
                "evidence_manifest_path"
            ]
            copied[-1]["evidence_manifest_sha256"] = rows[0][
                "evidence_manifest_sha256"
            ]
            self.assertFalse(
                self.v149.aggregate_periodic_scan(copied, self.config)[
                    "evidence_complete"
                ]
            )
            duplicated_artifact = [dict(row) for row in rows]
            first_evidence = json.loads(
                Path(rows[0]["evidence_manifest_path"]).read_text(
                    encoding="ascii"
                )
            )
            last_manifest = Path(rows[-1]["evidence_manifest_path"])
            last_evidence = json.loads(
                last_manifest.read_text(encoding="ascii")
            )
            first_project = Path(
                first_evidence["artifacts"]["project"]["path"]
            )
            last_project = Path(
                last_evidence["artifacts"]["project"]["path"]
            )
            last_project.write_bytes(first_project.read_bytes())
            last_evidence["artifacts"]["project"]["sha256"] = (
                self.v149.sha256(last_project)
            )
            last_manifest.write_text(
                json.dumps(last_evidence, sort_keys=True), encoding="ascii"
            )
            duplicated_artifact[-1]["evidence_manifest_sha256"] = (
                self.v149.sha256(last_manifest)
            )
            self.assertFalse(
                self.v149.aggregate_periodic_scan(
                    duplicated_artifact, self.config
                )["evidence_complete"]
            )


if __name__ == "__main__":
    unittest.main()
