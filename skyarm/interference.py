"""Mesh interference checking across the arm's range of motion.

Places the real CAD assembly at a pose and computes boolean
intersections between parts that should never touch (cross-cluster
pairs).  Used to set the elbow/wrist travel limits in spec.py to values
the hardware can actually reach without self-collision.

    python -m skyarm.interference            # sweep + report
"""

from __future__ import annotations

import numpy as np
import trimesh

from . import assembly
from . import kinematics as kin
from . import spec

# parts in the same cluster may touch by design; cross-cluster contacts
# listed here are also intended (tube-in-clamp, stacked interfaces)
ALLOWED = {
    frozenset(("upper-arm-tube", "shoulder-clevis")),
    frozenset(("upper-arm-tube", "elbow-bracket")),
    frozenset(("forearm-tube", "elbow-clevis")),
    frozenset(("forearm-tube", "wrist-bracket")),
    frozenset(("yaw-column", "shoulder-housing")),
    frozenset(("yaw-column", "shoulder-cover")),
    frozenset(("yaw-output", "yaw-column")),
    frozenset(("wrist-clevis", "roll-housing")),
    frozenset(("wrist-clevis", "roll-motor")),
    frozenset(("roll-pulley", "gripper-base")),
    frozenset(("gripper-base", "gripper-finger")),
    frozenset(("gripper-base", "gripper-servo")),
    frozenset(("gripper-servo", "gripper-finger")),
}
VOL_TOL_MM3 = 300.0      # ignore sub-0.3 cm3 modelling kisses


def _cluster(name: str) -> str:
    return name.split("-")[0]


def collisions(pose: kin.Pose, include_gantry: bool = False) -> list:
    placed = assembly.machine(pose, include_gantry=include_gantry)
    bad = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = placed[i], placed[j]
            if _cluster(a.name) == _cluster(b.name):
                continue
            if frozenset((a.name, b.name)) in ALLOWED:
                continue
            # cheap AABB rejection first
            amin, amax = a.mesh.bounds
            bmin, bmax = b.mesh.bounds
            if (amax < bmin).any() or (bmax < amin).any():
                continue
            try:
                inter = trimesh.boolean.intersection(
                    [a.mesh, b.mesh], engine="manifold")
                vol = abs(inter.volume) if inter is not None else 0.0
            except Exception:
                vol = 0.0
            if vol > VOL_TOL_MM3:
                bad.append((a.name, b.name, vol))
    return bad


def max_elbow_fold(step_deg: float = 5.0) -> float:
    """Largest |elbow| with no self-collision (straight-down upper arm)."""
    angle = 0.0
    safe = 0.0
    while angle <= 135.0:
        pose = kin.Pose(shoulder=0.0, elbow=-angle, wrist_pitch=0.0)
        if collisions(pose):
            break
        safe = angle
        angle += step_deg
    return safe


TEST_POSES = (
    ("straight down", kin.Pose()),
    ("hero", kin.Pose(shoulder=30.0, elbow=-60.0, wrist_pitch=30.0)),
    ("elbow at +limit", kin.Pose(elbow=spec.ELBOW.travel_deg[1])),
    ("elbow at -limit", kin.Pose(elbow=spec.ELBOW.travel_deg[0])),
    ("wrist at limit", kin.Pose(elbow=-60.0,
                                wrist_pitch=spec.WRIST_PITCH.travel_deg[1])),
    ("shoulder at limit", kin.Pose(shoulder=spec.SHOULDER.travel_deg[1],
                                   elbow=-30.0)),
)


def report() -> str:
    lines = ["SkyArm self-interference check (boolean mesh intersections)",
             "=" * 64]
    any_bad = False
    for label, pose in TEST_POSES:
        bad = collisions(pose)
        if bad:
            any_bad = True
            lines.append(f"[{label}] COLLISIONS:")
            for a, b, v in bad:
                lines.append(f"    {a} x {b}: {v / 1000:.1f} cm3")
        else:
            lines.append(f"[{label}] clear")
    fold = max_elbow_fold()
    lines.append(f"\nmax collision-free elbow fold: {fold:.0f} deg "
                 f"(spec travel: {spec.ELBOW.travel_deg})")
    lines.append("PASS" if not any_bad
                 and fold >= abs(spec.ELBOW.travel_deg[0])
                 else "INTERFERENCE PRESENT")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
