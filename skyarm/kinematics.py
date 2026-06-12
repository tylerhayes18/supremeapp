"""Forward/inverse kinematics for the SkyArm gantry + arm.

Pure math (stdlib only, like handsense's gestures.py) so it is trivially
unit-testable and reusable by both the simulator and real firmware.

Room frame: x/y horizontal (gantry axes), z up, floor at z=0.

State vector (the "pose"):
    gx, gy            gantry carriage position, mm
    yaw               J1, deg, rotation of the arm plane about vertical
    shoulder          J2, deg, 0 = upper arm straight down,
                      positive swings the arm toward the yaw heading
    elbow             J3, deg, relative to upper arm
    wrist_pitch       J4, deg, relative to forearm
    wrist_roll        J5, deg (does not move the tip point)

FK returns the chain of points: shoulder, elbow, wrist, tip.
IK solves gantry + arm for a tip target and an approach angle psi
(angle of the final wrist segment from straight-down, in the arm plane;
psi = 0 means the gripper points at the floor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .spec import (ARM_REACH_MM, CARRIAGE_STACK_MM, FOREARM_MM, ROOM,
                   SHOULDER, ELBOW, WRIST_PITCH, UPPER_ARM_MM,
                   WRIST_TO_TIP_MM, X_TRAVEL_MM, Y_TRAVEL_MM,
                   YAW_SHOULDER_OFFSET_MM)

CEILING_MM = ROOM["ceiling_mm"]
SHOULDER_Z_MM = CEILING_MM - CARRIAGE_STACK_MM
E_OFF = YAW_SHOULDER_OFFSET_MM      # shoulder offset from the yaw axis


@dataclass(frozen=True)
class Pose:
    gx: float = X_TRAVEL_MM / 2
    gy: float = Y_TRAVEL_MM / 2
    yaw: float = 0.0
    shoulder: float = 0.0
    elbow: float = 0.0
    wrist_pitch: float = 0.0
    wrist_roll: float = 0.0

    def joints(self) -> tuple:
        return (self.yaw, self.shoulder, self.elbow,
                self.wrist_pitch, self.wrist_roll)


@dataclass(frozen=True)
class Chain:
    """World-space joint positions, mm."""
    shoulder: tuple
    elbow: tuple
    wrist: tuple
    tip: tuple


class Unreachable(ValueError):
    pass


def _dir(yaw_rad: float, phi_rad: float) -> tuple:
    """Unit vector at angle phi from straight-down, in the yaw plane."""
    s = math.sin(phi_rad)
    return (s * math.cos(yaw_rad), s * math.sin(yaw_rad), -math.cos(phi_rad))


def forward(pose: Pose) -> Chain:
    yaw = math.radians(pose.yaw)
    # shoulder sits E_OFF to the side of the yaw axis (rotates with yaw)
    p0 = (pose.gx + E_OFF * math.sin(yaw),
          pose.gy - E_OFF * math.cos(yaw), SHOULDER_Z_MM)

    a2 = math.radians(pose.shoulder)
    a3 = a2 + math.radians(pose.elbow)
    a4 = a3 + math.radians(pose.wrist_pitch)

    def step(p, length, phi):
        d = _dir(yaw, phi)
        return (p[0] + length * d[0], p[1] + length * d[1], p[2] + length * d[2])

    elbow = step(p0, UPPER_ARM_MM, a2)
    wrist = step(elbow, FOREARM_MM, a3)
    tip = step(wrist, WRIST_TO_TIP_MM, a4)
    return Chain(p0, elbow, wrist, tip)


def _clamp(v, lo, hi):
    return min(max(v, lo), hi)


def inverse(target: tuple, psi_deg: float = 0.0, elbow_up: bool = True,
            gantry: tuple | None = None) -> Pose:
    """Solve for a tip position ``target`` (x, y, z in mm).

    The gantry is placed directly over the target when travel allows
    (vertical reach, stiffest configuration); when the target lies outside
    gantry travel, the carriage parks at the nearest edge and the arm
    leans out to cover the residual.  Pass ``gantry`` to pin the carriage.
    Tries the preferred elbow configuration first and falls back to the
    other one if joint limits reject it.
    """
    try:
        return _solve(target, psi_deg, elbow_up, gantry)
    except Unreachable:
        return _solve(target, psi_deg, not elbow_up, gantry)


def _solve(target: tuple, psi_deg: float, elbow_up: bool,
           gantry: tuple | None) -> Pose:
    tx, ty, tz = target
    if gantry is None:
        # park the yaw axis so the (offset) shoulder lands over the
        # target at yaw = 0 — the stiffest configuration
        gx = _clamp(tx, 0.0, X_TRAVEL_MM)
        gy = _clamp(ty + E_OFF, 0.0, Y_TRAVEL_MM)
    else:
        gx, gy = gantry

    # horizontal solve with the lateral shoulder offset:
    # target - gantry = E_OFF * u(yaw) + r * t(yaw),  u perpendicular t
    dx, dy = tx - gx, ty - gy
    v = math.hypot(dx, dy)
    if v < E_OFF - 1e-9:
        raise Unreachable(
            f"target {target} horizontally inside the {E_OFF:.0f} mm "
            "yaw-shoulder offset circle of the pinned gantry")
    r = math.sqrt(max(v * v - E_OFF * E_OFF, 0.0))
    yaw = math.degrees(math.atan2(dy, dx) + math.atan2(E_OFF, r)) \
        if v > 1e-9 else 0.0
    depth = SHOULDER_Z_MM - tz          # distance below shoulder, +down

    # Wrist point: back off from the tip along the approach direction.
    psi = math.radians(psi_deg)
    rw = r - WRIST_TO_TIP_MM * math.sin(psi)
    dw = depth - WRIST_TO_TIP_MM * math.cos(psi)

    L1, L2 = UPPER_ARM_MM, FOREARM_MM
    d2 = rw * rw + dw * dw
    cos_el = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if cos_el > 1.0:
        if cos_el < 1.0 + 1e-9:
            cos_el = 1.0
        else:
            raise Unreachable(
                f"target {target} needs {math.sqrt(d2):.0f} mm from shoulder; "
                f"arm reaches {L1 + L2:.0f} mm (to wrist)")
    if cos_el < -1.0:
        raise Unreachable(f"target {target} too close to shoulder")

    el = math.acos(cos_el)
    if elbow_up:
        el = -el
    # angle of the shoulder->wrist line from straight-down, then correct
    # for the elbow bend
    shoulder = math.atan2(rw, dw) - math.atan2(L2 * math.sin(el),
                                               L1 + L2 * math.cos(el))
    wrist_pitch = psi - shoulder - el

    pose = Pose(gx=gx, gy=gy, yaw=yaw,
                shoulder=math.degrees(shoulder),
                elbow=math.degrees(el),
                wrist_pitch=math.degrees(wrist_pitch))
    _check_limits(pose)
    return pose


def _check_limits(pose: Pose) -> None:
    for value, joint in ((pose.shoulder, SHOULDER), (pose.elbow, ELBOW),
                         (pose.wrist_pitch, WRIST_PITCH)):
        lo, hi = joint.travel_deg
        if not (lo - 1e-6 <= value <= hi + 1e-6):
            raise Unreachable(
                f"{joint.name} = {value:.1f} deg outside travel {lo}..{hi}")


def reachable(target: tuple, psi_deg: float = 0.0) -> bool:
    try:
        inverse(target, psi_deg)
        return True
    except Unreachable:
        try:
            inverse(target, psi_deg, elbow_up=False)
            return True
        except Unreachable:
            return False


def workspace_floor_ok() -> bool:
    """The arm must reach the floor anywhere under the gantry (5 ft from
    a 2.7 m ceiling leaves ~1.18 m... it does NOT touch the floor — the
    spec'd reach puts the tip at ceiling-1524 mm).  Helper for tests."""
    chain = forward(inverse((X_TRAVEL_MM / 2, Y_TRAVEL_MM / 2,
                             CEILING_MM - CARRIAGE_STACK_MM - ARM_REACH_MM)))
    return abs(chain.tip[2] - (CEILING_MM - 1524.0)) < 1e-6
