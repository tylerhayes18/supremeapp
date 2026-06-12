"""Builds render geometry for the machine + room from a World state.

Returns (vertices, faces, color) tuples consumed by both the headless
rasterizer (skyarm.render) and exportable as a GLB for inspection.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .. import kinematics as kin
from .. import spec
from .world import World

COL_RAIL = (110, 110, 120)
COL_BRIDGE = (90, 130, 200)
COL_CARRIAGE = (235, 137, 52)
COL_UPPER = (90, 170, 235)
COL_FOREARM = (140, 220, 140)
COL_WRIST = (230, 110, 110)
COL_GRIP = (220, 220, 90)
COL_FLOOR = (45, 48, 55)
COL_TARGET = (255, 60, 60)


def _seg(a, b, d, color):
    m = trimesh.creation.cylinder(radius=d / 2, height=1.0, sections=24,
                                  segment=(a, b))
    return (m.vertices, m.faces, color)


def _box(extents, center, color):
    m = trimesh.creation.box(extents)
    m.apply_translation(center)
    return (m.vertices, m.faces, color)


def build(world: World, target=None) -> list:
    s = spec
    ceil = kin.CEILING_MM
    chain = world.chain()
    pose = world.pose()
    meshes = []

    # floor + room hint
    meshes.append(_box((s.ROOM["x_mm"], s.ROOM["y_mm"], 8),
                       (s.ROOM["x_mm"] / 2, s.ROOM["y_mm"] / 2, -4), COL_FLOOR))
    # ceiling X rails (2040)
    for ry in (150.0, s.ROOM["y_mm"] - 150.0):
        meshes.append(_box((s.ROOM["x_mm"], 40, 20),
                           (s.ROOM["x_mm"] / 2, ry, ceil - 10), COL_RAIL))
    # bridge beam (2060) at carriage x
    meshes.append(_box((60, s.ROOM["y_mm"] - 200, 20),
                       (pose.gx + 150, s.ROOM["y_mm"] / 2, ceil - 30), COL_BRIDGE))
    # trucks
    for ry in (150.0, s.ROOM["y_mm"] - 150.0):
        meshes.append(_box((130, 120, 14),
                           (pose.gx + 150, ry, ceil - 47), COL_CARRIAGE))
    # carriage + yaw stack (gantry coords offset: rails inset 150)
    cx, cy = pose.gx + 150, pose.gy + 200
    meshes.append(_box((160, 150, 16), (cx, cy, ceil - 48), COL_CARRIAGE))
    meshes.append(_seg((cx, cy, ceil - 50), (cx, cy, chain.shoulder[2] + 30),
                       90, COL_CARRIAGE))

    # NOTE: kinematics uses gantry coords; scene shifts by rail inset so the
    # machine sits inside the room box.
    off = np.array([150.0, 200.0, 0.0])
    p_sh = np.array(chain.shoulder) + off
    p_el = np.array(chain.elbow) + off
    p_wr = np.array(chain.wrist) + off
    p_tip = np.array(chain.tip) + off

    # shoulder gearbox blob + upper arm + elbow + forearm + wrist + gripper
    meshes.append(_seg(p_sh + (0, 0, 40), p_sh - (0, 0, 40),
                       s.CYCLO_BIG.housing_od * 0.66, COL_CARRIAGE))
    meshes.append(_seg(p_sh, p_el, s.TUBE_OD_MM, COL_UPPER))
    meshes.append(_seg(p_el + (0, 0, 30), p_el - (0, 0, 30), 70, COL_WRIST))
    meshes.append(_seg(p_el, p_wr, s.TUBE_OD_MM, COL_FOREARM))
    meshes.append(_seg(p_wr, p_tip, 40, COL_WRIST))

    # gripper jaws: open/close visual
    jaw_gap = 18 + 30 * world.gripper
    axis = p_tip - p_wr
    n = np.linalg.norm(axis)
    if n > 1e-9:
        axis = axis / n
    side = np.cross(axis, (0, 0, 1.0))
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0, 0])
    side = side / np.linalg.norm(side)
    base = p_tip - axis * 70
    for sgn in (-1, 1):
        a = base + side * sgn * jaw_gap / 2
        b = p_tip + side * sgn * jaw_gap / 4
        meshes.append(_seg(a, b, 12, COL_GRIP))

    if target is not None:
        t = np.array(target) + off
        m = trimesh.creation.icosphere(subdivisions=2, radius=25)
        m.apply_translation(t)
        meshes.append((m.vertices, m.faces, COL_TARGET))

    return meshes


def snapshot(world: World, out_png: str, target=None, size=(1100, 800),
             eye_dir=(0.8, -1.0, 0.45)):
    from ..render import render_meshes
    meshes = build(world, target)
    chain = world.chain()
    off = np.array([150.0, 200.0, 0.0])
    center = (np.array(chain.shoulder) + np.array(chain.tip)) / 2 + off
    d = 1.05 * max(spec.ROOM["x_mm"], spec.ROOM["y_mm"])
    e = np.asarray(eye_dir, float)
    eye = center + e / np.linalg.norm(e) * d
    img = render_meshes(meshes, size=size, eye=eye, target=center, fov_deg=38)
    img.save(out_png)
    return out_png
