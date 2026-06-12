"""Tests for the SkyArm simulation world (pure stdlib + spec/kinematics)."""

import unittest

from skyarm import kinematics as kin
from skyarm import spec
from skyarm.sim.world import Axis, SafetyError, World, demo_mission


class TestAxis(unittest.TestCase):
    def test_reaches_target_and_settles(self):
        ax = Axis("t", 0.0, vmax=100.0, amax=400.0, lo=0.0, hi=1000.0)
        ax.set_target(500.0)
        for _ in range(int(12 / 0.01)):
            ax.step(0.01)
        self.assertTrue(ax.settled)
        self.assertAlmostEqual(ax.pos, 500.0)

    def test_velocity_limit_respected(self):
        ax = Axis("t", 0.0, vmax=100.0, amax=400.0, lo=0.0, hi=1000.0)
        ax.set_target(1000.0)
        vmax_seen = 0.0
        for _ in range(2000):
            ax.step(0.01)
            vmax_seen = max(vmax_seen, abs(ax.vel))
        self.assertLessEqual(vmax_seen, 100.0 + 1e-6)

    def test_target_clamped_to_travel(self):
        ax = Axis("t", 0.0, vmax=100.0, amax=400.0, lo=-10.0, hi=10.0)
        ax.set_target(99.0)
        self.assertEqual(ax.target, 10.0)


class TestWorld(unittest.TestCase):
    def test_demo_mission_completes_without_faults(self):
        world = demo_mission(World())
        world.run()
        self.assertTrue(world.settled)
        self.assertEqual(world.faults, [])
        self.assertEqual(world.mission, [])

    def test_demo_reaches_full_five_foot_drop(self):
        world = demo_mission(World())
        world.run()
        # the first waypoint commands a drop to 10 mm above the full
        # 5 ft reach — the trace must pass through it
        want = (kin.X_TRAVEL_MM / 2, kin.Y_TRAVEL_MM / 2,
                kin.CEILING_MM - 1524.0 + 10.0)
        best = min(sum((a - b) ** 2 for a, b in zip(p, want)) ** 0.5
                   for _, p in world.trace)
        self.assertLess(best, 1.0)
        # and the tip never dips below the spec'd 5 ft envelope
        zmin = min(p[2] for _, p in world.trace)
        self.assertGreaterEqual(zmin, kin.CEILING_MM - 1524.0 - 1e-6)

    def test_floor_target_rejected(self):
        world = World()
        with self.assertRaises((SafetyError, kin.Unreachable)):
            world.goto((kin.X_TRAVEL_MM / 2, kin.Y_TRAVEL_MM / 2, 0.0))

    def test_goto_converges_to_ik_solution(self):
        world = World()
        target = (1500.0, 1200.0, 1400.0)
        world.goto(target)
        world.run()
        tip = world.chain().tip
        for got, want in zip(tip, target):
            self.assertAlmostEqual(got, want, delta=0.5)

    def test_gripper_takes_half_second(self):
        world = World()
        world.grip(True)
        world.run(seconds=0.4)
        self.assertGreater(world.gripper, 0.0)   # still closing
        world.run(seconds=0.2)
        self.assertAlmostEqual(world.gripper, 0.0, delta=1e-6)

    def test_trace_recorded(self):
        world = World()
        world.run(seconds=1.0)
        self.assertEqual(len(world.trace), 120)


if __name__ == "__main__":
    unittest.main()
