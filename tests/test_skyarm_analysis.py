"""Structural design-rule tests: every load path keeps its safety factor."""

import unittest

from skyarm import analysis, spec


class TestStructuralChecks(unittest.TestCase):
    def test_all_safety_factors_met(self):
        for check in analysis.run_all():
            min_sf = analysis.MIN_SF.get(check.name, analysis.DEFAULT_MIN_SF)
            self.assertGreaterEqual(
                check.sf, min_sf,
                f"{check.name}: SF {check.sf:.2f} < {min_sf} "
                f"({check.demand:.1f}/{check.capacity:.1f} {check.unit})")

    def test_tip_sag_under_limit(self):
        c = analysis.check_tip_deflection()
        self.assertLess(c.demand, 5.0)

    def test_bridge_deflection_under_limit(self):
        c = analysis.check_bridge_deflection()
        self.assertLess(c.demand, 3.0)

    def test_output_bearing_margin(self):
        c = analysis.check_output_bearing()
        self.assertGreaterEqual(c.sf, 2.0)

    def test_tube_choice_consistent(self):
        # the spec tube must actually be the one the analysis validates
        self.assertEqual(spec.TUBE_OD_MM, 50.8)
        self.assertGreaterEqual(spec.TUBE_WALL_MM, 2.0)


if __name__ == "__main__":
    unittest.main()
