"""PyBullet rigid-body dynamics tests — skipped when pybullet is absent.

These are the mechanical proof tests: real masses and inertias under
gravity, motors limited to their spec'd gearbox output capacity.
"""

import math
import unittest

try:
    import pybullet  # noqa: F401
    HAVE_PB = True
except ImportError:
    HAVE_PB = False

from skyarm import spec


@unittest.skipUnless(HAVE_PB, "pybullet not installed")
class TestDynamics(unittest.TestCase):
    def test_holds_rated_payload_horizontal(self):
        from skyarm.sim.dynamics import CAPACITY, hold_horizontal
        r = hold_horizontal(spec.PAYLOAD_KG, seconds=4.0)
        self.assertLess(abs(r.droop_tip_mm), 2.0, "arm sagged under hold")
        # steady torque must clear capacity with dynamic headroom
        for joint in ("shoulder", "elbow", "wrist"):
            self.assertLess(r.steady_torque[joint], 0.95 * CAPACITY[joint],
                            f"{joint} saturating at steady state")

    def test_steady_torques_match_static_spec(self):
        # cross-validation: the physics engine's gravity torques should
        # agree with the closed-form budget in spec.py within ~10%
        from skyarm.sim.dynamics import hold_horizontal
        r = hold_horizontal(spec.PAYLOAD_KG, seconds=4.0)
        self.assertAlmostEqual(r.steady_torque["shoulder"],
                               spec.SHOULDER.gravity_torque_nm(),
                               delta=0.1 * spec.SHOULDER.gravity_torque_nm())
        self.assertAlmostEqual(r.steady_torque["elbow"],
                               spec.ELBOW.gravity_torque_nm(),
                               delta=0.1 * spec.ELBOW.gravity_torque_nm())

    def test_mission_tracks_within_tolerance(self):
        from skyarm.sim.dynamics import mission
        r = mission(seconds_per_leg=2.5)
        worst_joint = max(v for k, v in r.max_err.items()
                          if k not in ("gx", "gy"))
        self.assertLess(math.degrees(worst_joint), 2.0)
        self.assertLess(max(r.max_err["gx"], r.max_err["gy"]), 0.005)  # 5 mm


if __name__ == "__main__":
    unittest.main()
