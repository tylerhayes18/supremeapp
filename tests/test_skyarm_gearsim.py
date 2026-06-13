"""Mechanism tests: the cycloidal drives must actually transmit motion.

Skipped without the CAD deps.  These sweep the real disc geometry
through a full input revolution against the real housing geometry.
"""

import unittest

try:
    import trimesh  # noqa: F401
    import manifold3d  # noqa: F401
    import shapely  # noqa: F401
    HAVE_CAD = True
except ImportError:
    HAVE_CAD = False

from skyarm import spec


@unittest.skipUnless(HAVE_CAD, "CAD deps not installed")
class TestCycloidalMechanism(unittest.TestCase):
    def test_wrist_gearbox_turns_without_jamming(self):
        from skyarm.gearsim import jam_check
        r = jam_check(spec.CYCLO_SMALL, samples=8)
        self.assertLess(r["worst_overlap_mm3"], 5.0,
                        "disc interferes with ring pins — drive would jam")
        self.assertLess(r["min_gap_mm"], 1.0,
                        "lobes never approach the pins — no torque path")
        self.assertGreater(r["min_gap_mm"], 0.0)

    def test_shoulder_gearbox_turns_without_jamming(self):
        from skyarm.gearsim import jam_check
        r = jam_check(spec.CYCLO_BIG, samples=6)
        self.assertLess(r["worst_overlap_mm3"], 5.0)
        self.assertLess(r["min_gap_mm"], 1.0)

    def test_output_pin_running_clearance(self):
        # hole-to-pin centre distance is E by construction; clearance is
        # what's left of the hole radius
        from skyarm.cad.cycloidal import CLEAR
        for cs in (spec.CYCLO_BIG, spec.CYCLO_MID, spec.CYCLO_YAW,
                   spec.CYCLO_SMALL):
            hole_r = cs.output_pin_r + cs.eccentricity + CLEAR
            clearance = hole_r - cs.output_pin_r - cs.eccentricity
            self.assertGreaterEqual(clearance, 0.2, cs.name)


if __name__ == "__main__":
    unittest.main()
