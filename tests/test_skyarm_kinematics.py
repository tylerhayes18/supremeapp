"""FK/IK and spec sanity tests for the SkyArm design (no heavy deps)."""

import math
import random
import unittest

from skyarm import kinematics as kin
from skyarm import spec


class TestSpec(unittest.TestCase):
    def test_total_reach_is_exactly_five_feet(self):
        total = spec.CARRIAGE_STACK_MM + spec.ARM_REACH_MM
        self.assertAlmostEqual(total, 1524.0)
        self.assertAlmostEqual(spec.TOTAL_REACH_MM, 1524.0)

    def test_every_joint_meets_torque_margin(self):
        for joint in spec.ARM_JOINTS:
            self.assertGreaterEqual(
                joint.safety_margin(), spec.MIN_SAFETY_MARGIN,
                f"{joint.name}: margin {joint.safety_margin():.2f}")

    def test_gantry_axes_meet_margin(self):
        for axis in spec.GANTRY_AXES:
            self.assertGreaterEqual(axis.safety_margin(), 2.0, axis.name)

    def test_cyclo_reductions_match_joints(self):
        for joint in (spec.YAW, spec.SHOULDER, spec.ELBOW, spec.WRIST_PITCH):
            box = spec.JOINT_GEARBOX[joint.name]
            self.assertEqual(box.reduction, joint.ratio,
                             f"{joint.name} gearbox/joint ratio mismatch")

    def test_gearboxes_fit_print_volume(self):
        for box in (spec.CYCLO_BIG, spec.CYCLO_MID, spec.CYCLO_YAW,
                    spec.CYCLO_SMALL):
            self.assertLess(box.housing_od, spec.PRINT_VOLUME_MM[0], box.name)

    def test_tip_resolution_under_tenth_mm(self):
        self.assertLess(spec.tip_resolution_mm(), 0.1)


class TestForward(unittest.TestCase):
    def test_straight_down_reaches_five_feet_below_ceiling(self):
        pose = kin.Pose(gx=1000, gy=1000)
        chain = kin.forward(pose)
        self.assertAlmostEqual(chain.tip[0], 1000.0)
        self.assertAlmostEqual(chain.tip[1], 1000.0)
        self.assertAlmostEqual(chain.tip[2], kin.CEILING_MM - 1524.0)

    def test_horizontal_arm(self):
        pose = kin.Pose(gx=500, gy=500, yaw=0, shoulder=90)
        chain = kin.forward(pose)
        self.assertAlmostEqual(chain.tip[0], 500 + spec.ARM_REACH_MM, places=6)
        self.assertAlmostEqual(chain.tip[2], kin.SHOULDER_Z_MM, places=6)

    def test_yaw_rotates_plane(self):
        pose = kin.Pose(gx=0, gy=0, yaw=90, shoulder=90)
        chain = kin.forward(pose)
        self.assertAlmostEqual(chain.tip[0], 0.0, places=6)
        self.assertAlmostEqual(chain.tip[1], spec.ARM_REACH_MM, places=6)


class TestInverse(unittest.TestCase):
    def test_roundtrip_random_targets(self):
        rng = random.Random(42)
        solved = 0
        for _ in range(300):
            gx = rng.uniform(200, kin.X_TRAVEL_MM - 200)
            gy = rng.uniform(200, kin.Y_TRAVEL_MM - 200)
            # targets inside a comfortable cone below the carriage
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(0, 700)
            depth = rng.uniform(400, 1300)
            target = (gx + rad * math.cos(ang), gy + rad * math.sin(ang),
                      kin.SHOULDER_Z_MM - depth)
            # with a vertical approach the wrist sits 164 mm above the tip;
            # close-in targets need an elbow fold past the travel limit
            wrist_d = math.hypot(rad, depth - spec.WRIST_TO_TIP_MM)
            if wrist_d > 0.98 * (spec.UPPER_ARM_MM + spec.FOREARM_MM):
                continue
            if wrist_d < 740.0:    # |elbow| > ~105 deg
                continue
            try:
                pose = kin.inverse(target, psi_deg=0.0, gantry=(gx, gy))
            except kin.Unreachable:
                continue
            solved += 1
            tip = kin.forward(pose).tip
            for got, want in zip(tip, target):
                self.assertAlmostEqual(got, want, delta=1e-6)
        self.assertGreater(solved, 120)

    def test_vertical_approach_keeps_gripper_down(self):
        target = (1200, 900, 1250)
        pose = kin.inverse(target, psi_deg=0.0)
        total = pose.shoulder + pose.elbow + pose.wrist_pitch
        self.assertAlmostEqual(total, 0.0, places=6)  # tip segment vertical

    def test_out_of_reach_raises(self):
        with self.assertRaises(kin.Unreachable):
            kin.inverse((kin.X_TRAVEL_MM / 2, kin.Y_TRAVEL_MM / 2, 0.0),
                        gantry=(kin.X_TRAVEL_MM / 2, kin.Y_TRAVEL_MM / 2))

    def test_gantry_clamps_and_arm_leans_for_outside_targets(self):
        # target beyond X travel: carriage parks at edge, arm leans out
        target = (kin.X_TRAVEL_MM + 400, kin.Y_TRAVEL_MM / 2, 1600)
        pose = kin.inverse(target, psi_deg=45.0)
        self.assertAlmostEqual(pose.gx, kin.X_TRAVEL_MM)
        tip = kin.forward(pose).tip
        for got, want in zip(tip, target):
            self.assertAlmostEqual(got, want, delta=1e-6)

    def test_joint_limits_enforced(self):
        # tip at shoulder height, 100 mm out, gripper down: the arm would
        # have to fold to ~169 deg at the elbow — beyond limits either way
        with self.assertRaises(kin.Unreachable):
            kin.inverse((600, 500, kin.SHOULDER_Z_MM),
                        psi_deg=0.0, gantry=(500, 500))


if __name__ == "__main__":
    unittest.main()
