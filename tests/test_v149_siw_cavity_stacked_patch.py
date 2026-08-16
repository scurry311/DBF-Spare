import importlib.util
import json
import math
import unittest
from pathlib import Path


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
        self.assertTrue(decision["passed"])

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

    def test_failed_periodic_gate_keeps_every_later_stage_locked(self):
        metrics = {
            "evidence_source": "HFSS_periodic_fullwave",
            "evidence_complete": True,
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


if __name__ == "__main__":
    unittest.main()
