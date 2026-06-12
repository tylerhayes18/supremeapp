"""Full-machine assembly mockups from the real CAD part meshes.

Places every generated part (plus stock items: rails, tubes, motors,
wheels, beam) at its assembled position for a given arm pose.  Each part
carries an explode vector so the same model renders as an assembled
mockup (factor 0) or a blown-apart view (factor 1) with every piece
keeping its assembled orientation.

    python -m skyarm.assembly            # writes previews/assembled.png,
                                         #        previews/exploded.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from . import kinematics as kin
from . import spec
from .cad import cycloidal, parts as cad_parts
from .cad.cycloidal import BEARINGS, NEMA_OF
from .cad.primitives import NEMA

OX, OY = 150.0, 200.0           # gantry origin inside the room (as in sim.scene)
CEIL = kin.CEILING_MM

# colours
C_PRINT = (235, 137, 52)        # printed structural parts
C_HOUSING = (90, 130, 200)
C_DISC = (225, 200, 80)
C_CAM = (220, 90, 90)
C_OUTPUT = (120, 200, 120)
C_COVER = (170, 175, 185)
C_ALU = (185, 190, 198)
C_MOTOR = (70, 72, 82)
C_WHEEL = (50, 52, 58)
C_GRIP = (220, 220, 90)

MAX_FACES = 3500                # decimation budget per cached part


@dataclass
class Placed:
    name: str
    mesh: trimesh.Trimesh       # world position at explode factor 0
    color: tuple
    explode: np.ndarray         # world displacement at explode factor 1


_cache: dict = {}


def _part(name, builder, decimate=True):
    if name not in _cache:
        m = builder()
        if decimate and len(m.faces) > MAX_FACES:
            m = m.simplify_quadric_decimation(face_count=MAX_FACES)
        _cache[name] = m
    return _cache[name].copy()


def _basis(origin, z, x):
    """4x4 transform: local z->``z``, local x->``x`` (orthonormalised)."""
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x, float)
    x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2], t[:3, 3] = x, y, z, origin
    return t


def _place(out, name, mesh, frame, local_xyz, color, explode_local):
    m = mesh
    m.apply_translation(local_xyz)
    m.apply_transform(frame)
    rot = frame[:3, :3]
    out.append(Placed(name, m, color, rot @ np.asarray(explode_local, float)))


def _box(extents):
    return trimesh.creation.box(extents)


def _cyl(d, h, seg=32):
    m = trimesh.creation.cylinder(radius=d / 2, height=h, sections=seg)
    m.apply_translation((0, 0, h / 2))     # sit on local z=0 like cad parts
    return m


def _motor_block(nema_size, length):
    n = NEMA[nema_size]
    m = _box((n["face"], n["face"], length))
    m.apply_translation((0, 0, -length / 2))   # extends along local -z
    return m


# ---------------------------------------------------------------------------
# Cycloidal gearbox cluster (internals stacked along local +z = output dir)
# ---------------------------------------------------------------------------

def _gearbox(out, label, cs, frame):
    t = cs.disc_thickness
    e = cs.eccentricity
    nema_size = NEMA_OF[cs.name]
    motor_len = {17: 48, 23: 81, 24: 100, 34: 156}[nema_size]

    _place(out, f"{label}-motor", _motor_block(nema_size, motor_len),
           frame, (0, 0, 0), C_MOTOR, (0, 0, -90))
    _place(out, f"{label}-housing",
           _part(f"{cs.name}-housing", lambda: cycloidal.housing(cs)),
           frame, (0, 0, 0), C_HOUSING, (0, 0, 0))
    _place(out, f"{label}-cam",
           _part(f"{cs.name}-cam", lambda: cycloidal.cam(cs)),
           frame, (0, 0, 2), C_CAM, (0, 0, 45))
    disc = _part(f"{cs.name}-disc-coarse",
                 lambda: cycloidal.disc(cs, samples=360))
    _place(out, f"{label}-disc1", disc.copy(), frame, (e, 0, 7.4),
           C_DISC, (0, 0, 85))
    _place(out, f"{label}-disc2", disc, frame, (-e, 0, 7.4 + t + 0.4),
           C_DISC, (0, 0, 120))
    output = _part(f"{cs.name}-output", lambda: cycloidal.output_flange(cs))
    output.apply_transform(trimesh.transformations.rotation_matrix(
        math.pi, (1, 0, 0)))                       # pins now point local -z
    zo = cycloidal.output_face_z(cs)
    _place(out, f"{label}-output", output, frame, (0, 0, zo),
           C_OUTPUT, (0, 0, 185))
    _place(out, f"{label}-cover",
           _part(f"{cs.name}-cover", lambda: cycloidal.cover(cs)),
           frame, (0, 0, cycloidal.cover_z(cs)), C_COVER, (0, 0, 245))
    return zo                                      # output face local z


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------

def machine(pose: kin.Pose | None = None, include_gantry: bool = True) -> list:
    pose = pose or kin.Pose(shoulder=30.0, elbow=-60.0, wrist_pitch=30.0)
    chain = kin.forward(pose)
    off = np.array([OX, OY, 0.0])
    p_sh = np.array(chain.shoulder) + off
    p_el = np.array(chain.elbow) + off
    p_wr = np.array(chain.wrist) + off
    p_tip = np.array(chain.tip) + off
    t_up = (p_el - p_sh) / np.linalg.norm(p_el - p_sh)       # upper-arm dir
    t_fo = (p_wr - p_el) / np.linalg.norm(p_wr - p_el)
    t_ti = (p_tip - p_wr) / np.linalg.norm(p_tip - p_wr)
    yaw = math.radians(pose.yaw)
    a = np.array([-math.sin(yaw), math.cos(yaw), 0.0])       # pitch axes dir

    out: list[Placed] = []
    cx, cy = pose.gx + OX, pose.gy + OY
    room_x, room_y = spec.ROOM["x_mm"], spec.ROOM["y_mm"]

    if include_gantry:
        for ry in (150.0, room_y - 150.0):
            m = _box((room_x, 40, 20))
            m.apply_translation((room_x / 2, ry, CEIL - 10))
            out.append(Placed("x-rail", m, C_ALU, np.array([0, 0, 260.0])))
            # ceiling brackets, upside-down, every ~700 mm
            n_br = max(2, int(room_x // 700))
            for i in range(n_br + 1):
                bx = 120 + i * (room_x - 240) / n_br
                br = _part("ceiling_bracket", cad_parts.ceiling_bracket)
                f = _basis((bx, ry + 50, CEIL), (0, 0, -1), (1, 0, 0))
                _place(out, "ceiling-bracket", br, f, (0, 0, 0),
                       C_PRINT, (0, 0, -380))
            # truck plate + 4 V wheels
            tr = _part("x_truck", cad_parts.x_truck)
            f = _basis((cx, ry, CEIL - 29), (0, 0, 1), (1, 0, 0))
            _place(out, "x-truck", tr, f, (0, 0, 0), C_PRINT, (0, 0, -160))
            for sx in (-1, 1):
                for sy in (-1, 1):
                    w = _cyl(24.4, 11)
                    w.apply_translation((cx + sx * 32.2, ry + sy * 32.2,
                                         CEIL - 16))
                    out.append(Placed("v-wheel", w, C_WHEEL,
                                      np.array([0, sy * 90.0, -80.0])))
            # bridge endcap
            ec = _part("bridge_endcap", cad_parts.bridge_endcap)
            ey = ry + (32 if ry < room_y / 2 else -42)
            f = _basis((cx, ey, CEIL - 70), (0, 1, 0), (1, 0, 0))
            _place(out, "bridge-endcap", ec, f, (0, 0, 0), C_PRINT,
                   (0, (1 if ry < room_y / 2 else -1) * -220.0, 0))

        m = _box((60, room_y - 320, 20))
        m.apply_translation((cx, room_y / 2, CEIL - 30))
        out.append(Placed("bridge-2060", m, C_HOUSING, np.array([0, 0, 0.0])))

        # Y carriage (rotated so wheel rows straddle the bridge) + wheels
        yc = _part("y_carriage", cad_parts.y_carriage)
        f = _basis((cx, cy, CEIL - 52), (0, 0, 1), (0, 1, 0))
        _place(out, "y-carriage", yc, f, (0, 0, 0), C_PRINT, (0, 0, -200))
        half_span = spec.CARRIAGE_WHEEL_SPAN_MM / 2
        for sx in (-1, 1):
            for dy in (-half_span, 0, half_span):
                w = _cyl(24.4, 11)
                w.apply_translation((cx + sx * 42.2, cy + dy, CEIL - 40))
                out.append(Placed("v-wheel", w, C_WHEEL,
                                  np.array([sx * 90.0, 0, -120.0])))
        # gantry drive motors (2 X rail ends + 1 bridge end)
        for pos, zdir, xdir in (
                ((room_x - 60, 150.0, CEIL - 60), (-1, 0, 0), (0, 0, 1)),
                ((room_x - 60, room_y - 150.0, CEIL - 60), (-1, 0, 0), (0, 0, 1)),
                ((cx, room_y - 230, CEIL - 60), (0, 1, 0), (0, 0, 1))):
            mm = _part("gantry_motor_mount", cad_parts.gantry_motor_mount)
            f = _basis(pos, zdir, xdir)
            _place(out, "gantry-motor-mount", mm, f, (0, 0, 0), C_PRINT,
                   (np.array(zdir, float) * -180).tolist())
            mo = _motor_block(23, 81)
            _place(out, "gantry-motor", mo, f, (0, 0, 0), C_MOTOR,
                   (np.array(zdir, float) * -300).tolist())

    # ---- yaw module under the carriage (offset -a so the big shoulder
    # gearbox below clears it) ------------------------------------------
    yaw_org = np.array([cx, cy, CEIL - 54.0]) - a * 40.0
    f_yaw = _basis(yaw_org, (0, 0, -1), (1, 0, 0))
    _gearbox(out, "yaw", spec.CYCLO_YAW, f_yaw)

    # column: yaw output -> shoulder gearbox housing band
    d_face_sh = (cad_parts.W_OUT["shoulder"]
                 + cycloidal.output_face_z(spec.CYCLO_BIG))
    col_y = cad_parts.W_OUT["shoulder"] + cycloidal.output_face_z(
        spec.CYCLO_BIG) / 2
    col_top, col_bot = CEIL - 122.0, p_sh[2] + spec.CYCLO_BIG.housing_od / 2 + 4
    m = _box((100, 170, col_top - col_bot))
    col_at = p_sh + a * (col_y - 35.0)
    m.apply_translation((col_at[0], col_at[1], (col_top + col_bot) / 2))
    out.append(Placed("yaw-column", m, C_PRINT, np.array([0, 0, 0.0])))

    # ---- arm joints (gearboxes output toward the arm plane) -------------
    for label, cs, joint, t_out in (
            ("shoulder", spec.CYCLO_BIG, p_sh, t_up),
            ("elbow", spec.CYCLO_MID, p_el, t_fo),
            ("wrist", spec.CYCLO_SMALL, p_wr, t_ti)):
        d_face = (cad_parts.W_OUT[label] + cycloidal.output_face_z(cs))
        f = _basis(joint + a * d_face, -a, t_out)
        _gearbox(out, label, cs, f)

    # ---- printed link beams (local: z = beam axis, y = -a) --------------
    def place_link(prefix, joint, t_dir, segs, builders):
        x_dir = np.cross(-a, t_dir)
        z0 = 0.0
        for i, (seg_len, builder) in enumerate(zip(segs, builders)):
            name, fn = builder
            mesh = _part(name, fn)
            fl = _basis(joint + t_dir * z0, t_dir, x_dir)
            _place(out, f"{prefix}-{name}", mesh, fl, (0, 0, 0), C_PRINT,
                   (0, 0, 60 + 70 * i))
            z0 += seg_len

    place_link("upperlink", p_sh, t_up, cad_parts.UPPER_SEGS,
               [("upper_root", cad_parts.upper_root),
                ("upper_mid", cad_parts.upper_mid),
                ("upper_mid2", lambda: cad_parts.upper_mid()),
                ("upper_tip", cad_parts.upper_tip)])
    place_link("forearmlink", p_el, t_fo, cad_parts.FOREARM_SEGS,
               [("forearm_root", cad_parts.forearm_root),
                ("forearm_mid", cad_parts.forearm_mid),
                ("forearm_tip", cad_parts.forearm_tip)])

    # ---- wrist roll + leadscrew gripper ---------------------------------
    f_roll = _basis(p_wr + t_ti * 55.0, t_ti, a)
    rh = _part("roll_housing", cad_parts.roll_housing)
    _place(out, "roll-housing", rh, f_roll, (-32, 0, 0), C_PRINT, (0, 0, 120))
    _place(out, "roll-motor", _motor_block(23, 81), f_roll, (-70, 0, 0),
           C_MOTOR, (0, 0, 60))
    rp = _part("roll_pulley", cad_parts.roll_pulley)
    _place(out, "roll-pulley", rp, f_roll, (0, 0, 11), C_PRINT, (0, 0, 190))
    gb = _part("gripper_body", cad_parts.gripper_body)
    _place(out, "gripper-body", gb, f_roll, (0, 0, 32), C_GRIP, (0, 0, 260))
    # gripper jaws slide along the body's long axis (= a direction)
    _place(out, "gripper-motor", _motor_block(17, 48),
           _basis(p_wr + t_ti * 93.0 - a * 80.0, a, t_ti),
           (0, 0, 0), C_MOTOR, tuple((-a * 80).tolist()))
    for side, mirrored in ((-1, False), (1, True)):
        jaw = _part(f"jaw{side}", lambda m=mirrored: cad_parts.gripper_jaw(m))
        fj = _basis(p_wr + t_ti * 99.0 + a * (side * 40.0), t_ti, a)
        _place(out, "gripper-jaw", jaw, fj, (0, 0, 0), C_GRIP,
               tuple((a * side * 120).tolist()))
    return out


def to_render(placed: list, factor: float = 0.0) -> list:
    res = []
    for p in placed:
        v = p.mesh.vertices + p.explode * factor
        res.append((v, p.mesh.faces, p.color))
    return res


def render_views(out_dir: str | None = None):
    from .render import render_meshes
    out = Path(out_dir or Path(__file__).resolve().parent / "previews")
    out.mkdir(exist_ok=True)
    placed = machine()
    chain = kin.forward(kin.Pose(shoulder=30.0, elbow=-60.0, wrist_pitch=30.0))
    off = np.array([OX, OY, 0.0])
    center = (np.array(chain.shoulder) + np.array(chain.tip)) / 2 + off
    written = []
    for name, factor, dist, c in (
            ("assembled", 0.0, 2700, center + (0, 0, 150)),
            ("exploded", 1.0, 3400, center + (0, 0, 100))):
        eye = c + np.array([0.85, -1.0, 0.35]) / 1.36 * dist
        img = render_meshes(to_render(placed, factor), size=(1280, 980),
                            eye=eye, target=c, fov_deg=38)
        path = out / f"{name}.png"
        img.save(path)
        written.append(str(path))
    return written


if __name__ == "__main__":
    for p in render_views():
        print(p)
