import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v150_truncated_coax_wave_port.py"
CONFIG = ROOT / "configs" / "v150_truncated_coax_wave_port_preregistered.json"
CONFIG149 = ROOT / "configs" / "v149_siw_cavity_stacked_patch_preregistered.json"


def load_module():
    spec = importlib.util.spec_from_file_location("v150", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class V150ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.v150 = load_module()
        self.config = self.v150.load_config(CONFIG)
        self.v149_config = json.loads(CONFIG149.read_text(encoding="utf-8"))

    def test_v150_freezes_radiator_siw_and_solver_contract(self):
        self.v150.validate_config(self.config)
        frozen_geometry = (
            "period_x_mm",
            "period_y_mm",
            "driven_patch_length_mm",
            "driven_patch_width_mm",
            "stacked_patch_length_mm",
            "stacked_patch_width_mm",
            "main_substrate_thickness_mm",
            "stack_spacer_thickness_mm",
            "feed_offset_x_mm",
            "feed_offset_y_mm",
            "probe_radius_mm",
            "siw_via_diameter_mm",
            "siw_via_pitch_mm",
            "siw_via_inset_mm",
            "siw_top_aperture_clearance_mm",
            "local_mesh_probe_mm",
            "local_mesh_patch_edge_mm",
            "local_mesh_via_mm",
            "adaptive_refinement_percent",
            "maximum_passes",
        )
        for name in frozen_geometry:
            self.assertEqual(
                self.config["nominal_geometry"][name],
                self.v149_config["nominal_geometry"][name],
                name,
            )
        self.assertEqual(
            self.config["frequencies_ghz"],
            self.v149_config["frequencies_ghz"],
        )
        self.assertEqual(self.config["scan"], self.v149_config["scan"])
        self.assertEqual(self.config["gates"], self.v149_config["gates"])
        self.assertEqual(
            self.config["resources"], self.v149_config["resources"]
        )

    def test_v150_declares_external_transverse_wave_port(self):
        wave_port = self.config["wave_port"]
        geometry = self.config["nominal_geometry"]
        self.assertEqual(wave_port["type"], "coaxial_modal_wave_port")
        self.assertEqual(wave_port["reference_plane_z_mm"], -1.0)
        self.assertFalse(wave_port["deembed"])
        self.assertEqual(wave_port["mode_count"], 1)
        self.assertEqual(wave_port["renormalization_ohm"], 50.0)
        self.assertEqual(geometry["probe_radius_mm"], 0.25)
        self.assertEqual(geometry["coax_dielectric_outer_radius_mm"], 0.575)
        self.assertEqual(geometry["coax_outer_radius_mm"], 1.05)
        self.assertEqual(geometry["coax_drop_mm"], 1.0)
        self.assertEqual(geometry["ground_thickness_mm"], 0.035)
        self.assertEqual(geometry["maximum_adaptive_tetrahedra"], 120000)
        self.assertAlmostEqual(
            self.v150.analytic_vacuum_coax_impedance_ohm(geometry),
            49.97,
            places=2,
        )

    def test_v150_rejects_radiator_mutation(self):
        changed = copy.deepcopy(self.config)
        changed["nominal_geometry"]["driven_patch_length_mm"] += 0.1
        with self.assertRaisesRegex(ValueError, "frozen v1.49 geometry"):
            self.v150.validate_config(changed)

    def test_v150_rejects_legacy_lumped_port_fields(self):
        changed = copy.deepcopy(self.config)
        changed["port_definition"] = {
            "contact_overlap_mm": 0.0,
            "reference_plane_offset_mm": 0.05,
            "axial_height_mm": 0.10,
            "local_mesh_mm": 0.10,
        }
        with self.assertRaisesRegex(ValueError, "legacy lumped-port"):
            self.v150.validate_config(changed)

    def test_v150_keeps_all_downstream_stages_locked(self):
        self.v150.validate_config(self.config)
        for name in (
            "allow_periodic_doe_batch",
            "allow_1x1",
            "allow_2x2",
            "allow_4x4",
            "allow_16x16",
            "allow_eep_export",
            "allow_training_labels",
            "allow_critic_training",
        ):
            self.assertFalse(self.config["scope"][name], name)


class V150CadGenerationTests(unittest.TestCase):
    def setUp(self):
        self.v150 = load_module()
        self.config = self.v150.load_config(CONFIG)
        self.geometry = copy.deepcopy(self.config["nominal_geometry"])

    def _builder_text(self, root: Path) -> str:
        return self.v150.periodic_builder_text(
            root / "v150_periodic_build_smoke.aedt",
            self.config,
            self.geometry,
            {"frequency_ghz": 10.0, "theta_deg": 0.0, "phi_deg": 0.0},
        )

    def test_builder_uses_exact_coax_and_continuous_ground_union(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._builder_text(Path(temporary))
        self.assertIn('CreateBox oEditor, "GroundOuterConductor"', source)
        self.assertIn('CreateCylinderZExact oEditor, "FeedProbe"', source)
        self.assertIn('CreateCylinderZExact oEditor, "CoaxOuter"', source)
        self.assertIn('CreateCylinderZExact oEditor, "CoaxDielectric"', source)
        self.assertIn('"NumSides:=", "0"', source)
        self.assertIn(
            'UniteSelection oEditor, "GroundOuterConductor,CoaxOuter"',
            source,
        )
        self.assertIn('SubtractObject oEditor, "GroundOuterConductor"', source)

    def test_builder_assigns_one_transverse_external_wave_port(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._builder_text(Path(temporary))
        self.assertIn("coaxPortFace = oEditor.GetFaceByPosition", source)
        self.assertIn('"ZPosition:=", Mm(-1)', source)
        self.assertIn(
            'AssignCoaxWavePort oBoundary, "FeedPort", CLng(coaxPortFace)',
            source,
        )
        self.assertIn('"NumModes:=", 1', source)
        self.assertIn('"DoDeembed:=", False', source)
        self.assertIn('"DoRenorm:=", True', source)
        self.assertIn('"RenormValue:=", "50ohm"', source)
        self.assertNotIn("AssignLumpedPort", source)
        self.assertNotIn("PortSheet", source)

    def test_builder_uses_memory_safe_object_mesh_without_port_sheet(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._builder_text(Path(temporary))
        self.assertIn('"NAME:Mesh_CoaxLaunch"', source)
        self.assertIn(
            '"Objects:=", Array("FeedProbe", "CoaxDielectric")',
            source,
        )
        self.assertNotIn(
            '"Objects:=", Array("FeedProbe", "GroundOuterConductor", "CoaxDielectric")',
            source,
        )
        self.assertIn('"MaxLength:=", "0.3000000mm"', source)
        self.assertNotIn("Mesh_PortSheet", source)
        self.assertNotIn('"RefineInside:=", True', source)

    def test_geometry_audit_identifies_wave_port_and_frozen_radiator(self):
        audit = self.v150.geometry_audit(self.config, self.geometry)
        self.assertEqual(audit["feed_port_type"], "coaxial_modal_wave_port")
        self.assertEqual(audit["wave_port_reference_plane_z_mm"], -1.0)
        self.assertEqual(audit["coax_analytic_impedance_ohm"], 49.97)
        self.assertTrue(audit["ground_outer_conductor_united"])
        self.assertTrue(audit["analytic_circular_coax"])
        self.assertFalse(audit["radial_vertical_lumped_port"])
        self.assertFalse(audit["global_0p18mm_mesh_used"])


class V150EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.v150 = load_module()
        self.config = self.v150.load_config(CONFIG)

    def _inventory(self, feed_type: str = "Wave Port") -> list[str]:
        geometry = self.config["nominal_geometry"]
        lines = [
            "OBJECT|MainSubstrate",
            "OBJECT|StackSpacer",
            "OBJECT|GroundOuterConductor",
            "OBJECT|SIWCavityTop",
            "OBJECT|DrivenPatch",
            "OBJECT|StackedPatch",
            "OBJECT|FeedProbe",
            "OBJECT|CoaxDielectric",
            "OBJECT|AirCell",
            "BOUNDARY|CopperSheetFiniteConductivity",
            "BOUNDARY|PrimaryX",
            "BOUNDARY|SecondaryX",
            "BOUNDARY|PrimaryY",
            "BOUNDARY|SecondaryY",
            "BOUNDARY|PrimaryX_Main",
            "BOUNDARY|SecondaryX_Main",
            "BOUNDARY|PrimaryX_Stack",
            "BOUNDARY|SecondaryX_Stack",
            "BOUNDARY|PrimaryY_Main",
            "BOUNDARY|SecondaryY_Main",
            "BOUNDARY|PrimaryY_Stack",
            "BOUNDARY|SecondaryY_Stack",
            f"EXCITATION|FeedPort:1|{feed_type}",
            "EXCITATION|FloquetTop:1|Floquet Port",
            "EXCITATION|FloquetTop:2|Floquet Port",
        ]
        lines.extend(
            f"OBJECT|SIWVia_{index:03d}"
            for index in range(len(self.v150.siw_via_centers(geometry)))
        )
        return lines

    def _passing_analysis(self) -> dict:
        return {
            "nominal_export_evidence_complete": True,
            "power_consistency_passed": True,
            "convergence_evidence_complete": True,
            "profile": {
                "converged": True,
                "final_delta_s": 0.02,
                "maximum_tetrahedra": 100000,
                "small_segment_count": 0,
            },
            "critical_warning_hits": {},
            "rows": [
                {
                    "frequency_ghz": frequency,
                    "active_rl_db": 16.0,
                    "passive_rl_db": 16.0,
                    "accepted_power_efficiency": 0.98,
                    "input_impedance_real_ohm": 50.0,
                    "input_impedance_imag_ohm": 0.0,
                }
                for frequency in self.config["frequencies_ghz"]
            ],
        }

    def _passing_audit(self) -> dict:
        return {
            "return_code": 0,
            "memory_aborted": False,
            "minimum_free_memory_gib": 4.0,
        }

    def test_inventory_requires_wave_port_and_forbids_legacy_objects(self):
        result = self.v150.validate_model_inventory(
            self._inventory(), self.config, self.config["nominal_geometry"]
        )
        self.assertTrue(result["verified"])
        self.assertTrue(result["checks"]["feed_is_wave_port"])
        self.assertTrue(result["checks"]["legacy_port_objects_absent"])

        lumped = self.v150.validate_model_inventory(
            self._inventory("Lumped Port"),
            self.config,
            self.config["nominal_geometry"],
        )
        self.assertFalse(lumped["verified"])
        self.assertFalse(lumped["checks"]["feed_is_wave_port"])

        legacy = self._inventory() + ["OBJECT|PortSheet"]
        result = self.v150.validate_model_inventory(
            legacy, self.config, self.config["nominal_geometry"]
        )
        self.assertFalse(result["verified"])
        self.assertFalse(result["checks"]["legacy_port_objects_absent"])

    def test_nominal_gate_separates_numerical_and_physical_pass(self):
        result = self.v150.evaluate_nominal_gates(
            self._passing_analysis(), self._passing_audit(), self.config
        )
        self.assertTrue(result["numerical_gate_passed"])
        self.assertTrue(result["physical_gate_passed"])
        self.assertFalse(result["authorizes_periodic_doe_batch"])
        self.assertTrue(all(result["locked_stages"].values()))

    def test_nominal_gate_rejects_mesh_and_small_segments(self):
        analysis = self._passing_analysis()
        analysis["profile"]["maximum_tetrahedra"] = 120001
        result = self.v150.evaluate_nominal_gates(
            analysis, self._passing_audit(), self.config
        )
        self.assertFalse(result["numerical_gate_passed"])
        self.assertIn("adaptive_tetrahedra", result["failed_numerical_checks"])

        analysis = self._passing_analysis()
        analysis["profile"]["small_segment_count"] = 1
        result = self.v150.evaluate_nominal_gates(
            analysis, self._passing_audit(), self.config
        )
        self.assertFalse(result["numerical_gate_passed"])
        self.assertIn("no_small_segments", result["failed_numerical_checks"])

    def test_nominal_gate_rejects_low_rl_or_efficiency(self):
        analysis = self._passing_analysis()
        analysis["rows"][0]["passive_rl_db"] = 14.9
        result = self.v150.evaluate_nominal_gates(
            analysis, self._passing_audit(), self.config
        )
        self.assertTrue(result["numerical_gate_passed"])
        self.assertFalse(result["physical_gate_passed"])
        self.assertIn("three_frequency_rl", result["failed_physical_checks"])

        analysis = self._passing_analysis()
        analysis["rows"][1]["accepted_power_efficiency"] = 0.969
        result = self.v150.evaluate_nominal_gates(
            analysis, self._passing_audit(), self.config
        )
        self.assertFalse(result["physical_gate_passed"])
        self.assertIn("accepted_efficiency", result["failed_physical_checks"])

    def test_stage_summary_gate_requires_numerical_and_physical_pass(self):
        self.assertTrue(
            self.v150.nominal_analysis_passes_summary_gate(
                {
                    "nominal_export_evidence_complete": True,
                    "numerical_gate_passed": True,
                    "physical_gate_passed": True,
                }
            )
        )
        self.assertFalse(
            self.v150.nominal_analysis_passes_summary_gate(
                {
                    "nominal_export_evidence_complete": True,
                    "numerical_gate_passed": False,
                    "physical_gate_passed": True,
                }
            )
        )

    def test_wave_port_warning_terms_are_critical(self):
        hits = self.v150.critical_log_hits(
            "Wave port assignment failed\nToo many conductors touch wave port"
        )
        self.assertIn("wave port assignment failed", hits)
        self.assertIn("conductors touch wave port", hits)


if __name__ == "__main__":
    unittest.main()
