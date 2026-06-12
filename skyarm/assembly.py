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
    ring_h = 2 * t + 1.2
    e = cs.eccentricity
    nema_size = NEMA_OF[cs.name]
    motor_len = {17: 48, 23: 81, 24: 100}[nema_size]

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
    _place(out, f"{label}-output", output, frame,
           (0, 0, 21.2 + 2 * t), C_OUTPUT, (0, 0, 175))
    _place(out, f"{label}-cover",
           _part(f"{cs.name}-cover", lambda: cycloidal.cover(cs)),
           frame, (0, 0, 7 + ring_h), C_COVER, (0, 0, 230))
    return 21.2 + 2 * t                            # output face local z


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
        for sx in (-1, 1):
            for dy in (-55, 0, 55):
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

    # ---- yaw module under the carriage --------------------------------
    f_yaw = _basis((cx, cy, CEIL - 54), (0, 0, -1), (1, 0, 0))
    _gearbox(out, "yaw", spec.CYCLO_YAW, f_yaw)

    # column: yaw output -> shoulder gearbox ring
    col_top, col_bot = CEIL - 105.0, p_sh[2] + 52.0
    m = _box((90, 90, col_top - col_bot))
    m.apply_translation(((p_sh + a * 55)[0], (p_sh + a * 55)[1],
                         (col_top + col_bot) / 2))
    out.append(Placed("yaw-column", m, C_PRINT, np.array([0, 0, 0.0])))

    # ---- arm joints -----------------------------------------------------
    def pitch_cluster(label, cs, joint, t_out, clevis_name, clevis_builder,
                      bracket=None, t_in=None):
        # gearbox: output face toward the arm plane
        f = _basis(joint + a * 68.7, -a, t_out)
        zo = _gearbox(out, label, cs, f)
        clev = _part(clevis_name, clevis_builder)
        _place(out, f"{label}-clevis", clev, f, (0, 0, zo + 0.2),
               C_PRINT, (0, 0, 290))
        if bracket is not None:
            name, builder = bracket
            br = _part(name, builder)
            fb = _basis(joint - 60.0 * t_in - a * 30.7, a, t_in)
            _place(out, f"{label}-bracket", br, fb, (0, 0, 0),
                   C_PRINT, (0, 0, -120))

    pitch_cluster("shoulder", spec.CYCLO_BIG, p_sh, t_up,
                  "shoulder_clevis", cad_parts.shoulder_clevis)
    pitch_cluster("elbow", spec.CYCLO_MID, p_el, t_fo,
                  "forearm_clevis", cad_parts.forearm_clevis,
                  bracket=("elbow_clevis", cad_parts.elbow_clevis), t_in=t_up)
    pitch_cluster("wrist", spec.CYCLO_SMALL, p_wr, t_ti,
                  "wrist_clevis", cad_parts.forearm_clevis,
                  bracket=("wrist_bracket", cad_parts.wrist_bracket), t_in=t_fo)

    # tubes (stock aluminium)
    for name, p0, p1 in (("upper-arm-tube", p_sh + t_up * 60, p_el - t_up * 60),
                         ("forearm-tube", p_el + t_fo * 60, p_wr - t_fo * 60)):
        seg = trimesh.creation.cylinder(radius=spec.TUBE_OD_MM / 2, height=1.0,
                                        sections=28, segment=(p0, p1))
        out.append(Placed(name, seg, C_ALU, np.array([0.0, 0, 0])))

    # ---- wrist roll + gripper ------------------------------------------
    f_roll = _basis(p_wr + t_ti * 55.0, t_ti, a)
    rh = _part("roll_housing", cad_parts.roll_housing)
    _place(out, "roll-housing", rh, f_roll, (-28, 0, 0), C_PRINT, (0, 0, 120))
    _place(out, "roll-motor", _motor_block(17, 48), f_roll, (-60, 0, 0),
           C_MOTOR, (0, 0, 60))
    rp = _part("roll_pulley", cad_parts.roll_pulley)
    _place(out, "roll-pulley", rp, f_roll, (0, 0, 9), C_PRINT, (0, 0, 190))
    gb = _part("gripper_base", cad_parts.gripper_base)
    _place(out, "gripper-base", gb, f_roll, (0, 0, 27), C_GRIP, (0, 0, 260))
    servo = _box((40, 20, 38))
    _place(out, "gripper-servo", servo, f_roll, (0, 10, 18), C_MOTOR,
           (0, 0, 225))
    for side, mirrored in ((-1, False), (1, True)):
        fi = _part(f"finger{side}", lambda m=mirrored: cad_parts.gripper_finger(m))
        ff = _basis(p_wr + t_ti * 92.0 + a * (side * 15.0)
                    - np.cross(t_ti, a) * 18.0, -a, t_ti)
        rot = trimesh.transformations.rotation_matrix(
            math.radians(-10 * side), (0, 0, 1))
        fi.apply_transform(rot)
        _place(out, "gripper-finger", fi, ff, (0, 0, side * -4.5), C_GRIP,
               (0, 0, side * 330))
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
