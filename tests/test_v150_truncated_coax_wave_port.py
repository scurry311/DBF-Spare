import copy
import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
