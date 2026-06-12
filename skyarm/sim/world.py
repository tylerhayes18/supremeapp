"""Headless simulation model of the SkyArm — the testable core.

Models every axis with velocity+acceleration-limited motion (limits from
skyarm.spec), enforces the room safety envelope, and executes waypoint
missions.  The OpenGL viewer is just a window onto this object, so any
behaviour you can see can also be asserted in a unit test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import spec
from .. import kinematics as kin


@dataclass
class Axis:
    """One velocity/accel-limited servo axis."""
    name: str
    pos: float
    vmax: float
    amax: float
    lo: float
    hi: float
    target: float = None  # type: ignore[assignment]
    vel: float = 0.0

    def __post_init__(self):
        if self.target is None:
            self.target = self.pos

    def set_target(self, value: float):
        self.target = min(max(value, self.lo), self.hi)

    def step(self, dt: float):
        err = self.target - self.pos
        # braking distance check: decelerate if we'd overshoot
        stop = (self.vel * self.vel) / (2 * self.amax) if self.amax else 0.0
        if abs(err) <= 1e-9 and abs(self.vel) < 1e-9:
            self.vel = 0.0
            return
        want = math.copysign(self.vmax, err)
        if abs(err) < stop:
            want = 0.0
        dv = want - self.vel
        max_dv = self.amax * dt
        self.vel += min(max(dv, -max_dv), max_dv)
        self.pos += self.vel * dt
        if (err > 0 and self.pos > self.target) or (err < 0 and self.pos < self.target):
            self.pos = self.target
            self.vel = 0.0

    @property
    def settled(self) -> bool:
        return abs(self.target - self.pos) < 1e-6 and abs(self.vel) < 1e-6


class SafetyError(RuntimeError):
    pass


class World:
    """The room + machine.  Drive it with goto()/step()."""

    def __init__(self):
        s = spec
        self.axes = {
            "gx": Axis("gx", kin.X_TRAVEL_MM / 2, s.X_AXIS.max_speed_mm_s,
                       s.X_AXIS.max_accel_mm_s2, 0.0, kin.X_TRAVEL_MM),
            "gy": Axis("gy", kin.Y_TRAVEL_MM / 2, s.Y_AXIS.max_speed_mm_s,
                       s.Y_AXIS.max_accel_mm_s2, 0.0, kin.Y_TRAVEL_MM),
        }
        for joint, attr in ((s.YAW, "yaw"), (s.SHOULDER, "shoulder"),
                            (s.ELBOW, "elbow"), (s.WRIST_PITCH, "wrist_pitch"),
                            (s.WRIST_ROLL, "wrist_roll")):
            self.axes[attr] = Axis(attr, 0.0, s.JOINT_SPEED_DPS[joint.name],
                                   s.JOINT_ACCEL_DPS2[joint.name],
                                   joint.travel_deg[0], joint.travel_deg[1])
        self.gripper = 1.0          # 1 open .. 0 closed
        self.gripper_target = 1.0
        self.time = 0.0
        self.mission: list = []     # queue of (kind, payload)
        self.trace: list = []       # (t, tip_xyz) samples
        self.faults: list = []

    # -- state ----------------------------------------------------------

    def pose(self) -> kin.Pose:
        a = self.axes
        return kin.Pose(gx=a["gx"].pos, gy=a["gy"].pos, yaw=a["yaw"].pos,
                        shoulder=a["shoulder"].pos, elbow=a["elbow"].pos,
                        wrist_pitch=a["wrist_pitch"].pos,
                        wrist_roll=a["wrist_roll"].pos)

    def chain(self) -> kin.Chain:
        return kin.forward(self.pose())

    @property
    def settled(self) -> bool:
        return all(ax.settled for ax in self.axes.values()) \
            and abs(self.gripper - self.gripper_target) < 1e-6

    # -- commands ---------------------------------------------------------

    def goto(self, target, psi_deg: float = 0.0):
        """IK move: tip to target (x, y, z mm), approach angle psi."""
        pose = kin.inverse(target, psi_deg)
        self._check_target_safe(pose)
        for attr in ("gx", "gy", "yaw", "shoulder", "elbow", "wrist_pitch"):
            self.axes[attr].set_target(getattr(pose, attr))
        return pose

    def goto_joints(self, **joints):
        for name, value in joints.items():
            self.axes[name].set_target(value)

    def grip(self, closed: bool):
        self.gripper_target = 0.0 if closed else 1.0

    def queue(self, *missions):
        """mission items: ("goto", target, psi) | ("grip", closed) |
        ("dwell", seconds)"""
        self.mission.extend(missions)

    # -- safety -----------------------------------------------------------

    def _check_target_safe(self, pose: kin.Pose):
        chain = kin.forward(pose)
        for label, p in (("elbow", chain.elbow), ("wrist", chain.wrist),
                         ("tip", chain.tip)):
            if p[2] < spec.FLOOR_CLEARANCE_MM - 1e-6:
                raise SafetyError(f"{label} would hit the floor: z={p[2]:.0f}")
            if not (-spec.ARM_REACH_MM <= p[0] <= spec.ROOM["x_mm"] + 1):
                raise SafetyError(f"{label} outside room x: {p[0]:.0f}")
            if not (-spec.ARM_REACH_MM <= p[1] <= spec.ROOM["y_mm"] + 1):
                raise SafetyError(f"{label} outside room y: {p[1]:.0f}")

    def _runtime_safety(self):
        chain = self.chain()
        if chain.tip[2] < 0 or chain.elbow[2] < 0 or chain.wrist[2] < 0:
            self.faults.append((self.time, "floor collision"))
            raise SafetyError(f"floor collision at t={self.time:.2f}s")

    # -- integration -------------------------------------------------------

    def step(self, dt: float = 1 / 120):
        if self.mission and self.settled:
            kind, *payload = self.mission.pop(0)
            if kind == "goto":
                self.goto(*payload)
            elif kind == "grip":
                self.grip(payload[0])
            elif kind == "dwell":
                self._dwell_until = self.time + payload[0]
                self.mission.insert(0, ("_dwelling",))
            elif kind == "_dwelling":
                if self.time < getattr(self, "_dwell_until", 0.0):
                    self.mission.insert(0, ("_dwelling",))
        for ax in self.axes.values():
            ax.step(dt)
        g_err = self.gripper_target - self.gripper
        g_step = 2.0 * dt          # full open->close in 0.5 s
        self.gripper += min(max(g_err, -g_step), g_step)
        self.time += dt
        self._runtime_safety()
        self.trace.append((self.time, self.chain().tip))

    def run(self, seconds: float = None, dt: float = 1 / 120):
        """Step until mission + motion complete (or for fixed seconds)."""
        if seconds is not None:
            for _ in range(int(seconds / dt)):
                self.step(dt)
            return
        guard = 0.0
        while (self.mission or not self.settled) and guard < 600.0:
            self.step(dt)
            guard += dt
        if guard >= 600.0:
            raise RuntimeError("mission did not settle within 600 s")


# ---------------------------------------------------------------------------
# Demo mission used by the viewer, snapshot mode and tests
# ---------------------------------------------------------------------------

def demo_mission(world: World):
    cx, cy = kin.X_TRAVEL_MM / 2, kin.Y_TRAVEL_MM / 2
    low_z = kin.CEILING_MM - 1524.0 + 10.0      # near max reach, straight down
    world.queue(
        ("goto", (cx, cy, low_z), 0.0),                       # full 5 ft drop
        ("grip", True), ("dwell", 0.3),                        # pick
        ("goto", (cx, cy, low_z + 600), 0.0),                  # lift
        ("goto", (400.0, 400.0, low_z + 600), 0.0),            # carry to corner
        ("goto", (400.0, 400.0, low_z + 50), 0.0),
        ("grip", False), ("dwell", 0.3),                       # place
        ("goto", (kin.X_TRAVEL_MM - 300, kin.Y_TRAVEL_MM - 300,
                  low_z + 300), 30.0),                         # angled approach
        ("goto", (cx, cy, low_z + 500), 0.0),                  # home-ish
    )
    return world
