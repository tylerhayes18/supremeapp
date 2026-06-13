"""All non-gearbox printable parts for the SkyArm (fully printed arm).

The arm links are printed box-section beams (no metal tube): segments
<=230 mm tall, printed standing, joined by bolted register flanges.
Beams taper to a narrow neck at the distal joint so links clear each
other when folding.  Section sizes come from spec.UPPER_BEAM /
spec.FOREARM_BEAM and are validated by skyarm/analysis.py.

Print: PETG-CF or PA-CF, 6 walls, 50% gyroid for beams and joint
structure; PETG for gearbox internals.
"""

from __future__ import annotations

import math

import trimesh

from .. import spec
from . import cycloidal
from . import primitives as P
from .primitives import NEMA, bolt_circle, box, cyl, cyl_x, cyl_y, difference, holes, rounded_plate, tube, union

WHEEL_HOLE = 5.2                  # M5 axles for V wheels
M5, M4, M3 = 5.2, 4.4, 3.4
TNUT_PITCH = 20.0                 # V-slot channel spacing
TRUCK_WHEEL_SPAN = 170.0          # along-rail wheelbase (moment check)

# lateral offsets: arm-plane -> gearbox output face, per joint
W_OUT = {"shoulder": 60.0, "elbow": 50.0, "wrist": 45.0}

NECK_LEN = 80.0                   # taper length at the distal beam end
FLANGE_T = 8.0                    # segment-joint flange thickness
FLANGE_RIM = 12.0


def motor_face_offset(joint: str, cs) -> float:
    """Arm-plane -> gearbox motor-mount plane distance for a joint."""
    return W_OUT[joint] + cycloidal.output_face_z(cs)


# ---------------------------------------------------------------------------
# Gantry
# ---------------------------------------------------------------------------

def ceiling_bracket() -> trimesh.Trimesh:
    """L-bracket: lag-screws into a ceiling joist, M5 T-nuts into the 2040
    X rail.  Print 8+ (one every ~45 cm of rail)."""
    t = 8.0
    base = rounded_plate(90, 60, t)                       # against ceiling
    wall = box(90, t, 52, (0, 30 - t / 2, 0))
    ribs = [box(8, 38, 38, (dx, 4, t)) for dx in (-37, 37)]
    body = union(base, wall, *ribs)
    base_cut = [cyl(6.5, t, (dx, -12, 0)) for dx in (-25, 25)]
    wall_cut = [cyl_y(M5, t, (dx, 30 - t, 14 + TNUT_PITCH * i))
                for dx in (-25, 25) for i in (0, 1)]
    return difference(body, *base_cut, *wall_cut)


def x_truck() -> trimesh.Trimesh:
    """Wheeled truck riding a ceiling 2040 rail.  Print 2.  Wheelbase set
    by the V-wheel moment check in skyarm/analysis.py."""
    t = 10.0
    plate = rounded_plate(TRUCK_WHEEL_SPAN + 60, 150, t)
    cut = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cut.append(cyl(WHEEL_HOLE, t, (sx * TRUCK_WHEEL_SPAN / 2,
                                           sy * 32.2, 0)))
    cut += holes(M5, t, [(sx * 40, sy * 25) for sx in (-1, 1) for sy in (-1, 1)])
    cut += [box(16, 3, t, (dx, 0, 0)) for dx in (-14, 14)]   # belt clamps
    return difference(plate, *cut)


def bridge_endcap() -> trimesh.Trimesh:
    """Joins the C-beam 4080 bridge end to an X truck.  Print 2."""
    t = 12.0
    face = rounded_plate(90, 130, t)
    cut = holes(M5, t, [(sx * 20, dy) for sx in (-1, 1)
                        for dy in (-40, -20, 0, 20, 40)])    # beam T-nuts
    cut += holes(M5, t, [(sx * 30, sy * 52) for sx in (-1, 1) for sy in (-1, 1)])
    return difference(face, *cut)


def y_carriage() -> trimesh.Trimesh:
    """Main carriage: bolts to two MGN15H blocks on the bridge rail (the
    V-wheel option failed the 15 lb moment check), carries the yaw
    gearbox on a 6-bolt circle.  Print 1."""
    t = 12.0
    plate = rounded_plate(230, 150, t)
    cut = []
    # 2x MGN15H blocks, 150 mm apart, 4x M3 each on 45 x 26 pattern
    for by in (-75, 75):
        cut += holes(M3, t, [(sx * 22.5, by + sy * 13)
                             for sx in (-1, 1) for sy in (-1, 1)])
    box_r = spec.CYCLO_YAW.housing_od / 2 - 6
    cut += holes(M5, t, bolt_circle(6, box_r))
    cut.append(cyl(50, t, (0, 0, 0)))                   # cable pass-through
    cut += [box(16, 3, t, (dx, 55, 0)) for dx in (-30, 30)]  # belt clamps
    return difference(plate, *cut)


def gantry_motor_mount() -> trimesh.Trimesh:
    """NEMA 23 bracket for the rail-end belt drives.  Print 3."""
    t = 8.0
    n = NEMA[23]
    face = rounded_plate(n["face"] + 16, n["face"] + 16, t)
    cut = [cyl(n["boss_d"], t)]
    cut += holes(n["bolt_d"], t, P.nema_bolt_centers(23))
    wall = box(n["face"] + 16, t, 60, (0, (n["face"] + 16) / 2 - t / 2, 0))
    body = union(face, wall)
    cut += [cyl_y(M5, t, (dx, (n["face"] + 16) / 2 - t, 40)) for dx in (-20, 20)]
    return difference(body, *cut)


def belt_tensioner() -> trimesh.Trimesh:
    """Sliding idler block, M5 idler bolt + M4 tension screw.  Print 3."""
    body = box(40, 30, 18)
    cut = [cyl(M5, 18, (6, 0, 0))]
    cut.append(cyl_x(M4, 41, (-21, 0, 9), seg=24))
    cut.append(box(20, 6.4, 18, (-8, 0, 0)))
    return difference(body, *cut)


# ---------------------------------------------------------------------------
# Printed link beams
# ---------------------------------------------------------------------------

def _beam(length, depth, width, wall, neck=None, neck_lo=None,
          flange_lo=False, flange_hi=False) -> trimesh.Trimesh:
    """Box beam along +z from 0..length, depth along x, width along y.
    ``neck``/``neck_lo`` taper the top/bottom NECK_LEN of depth down
    (solid there — it sits at the joint where bending moment is lowest,
    and the narrow profile is what lets folded links clear each other)."""
    hi = [(depth / 2, length), (-depth / 2, length)]
    hollow_hi = length - wall
    if neck is not None and neck < depth:
        hi = [(depth / 2, length - NECK_LEN),
              (neck / 2, length - NECK_LEN / 4), (neck / 2, length),
              (-neck / 2, length), (-neck / 2, length - NECK_LEN / 4),
              (-depth / 2, length - NECK_LEN)]
        hollow_hi = length - NECK_LEN - wall
    lo = [(-depth / 2, 0), (depth / 2, 0)]
    hollow_lo = wall
    if neck_lo is not None and neck_lo < depth:
        lo = [(-neck_lo / 2, 0), (neck_lo / 2, 0),
              (neck_lo / 2, NECK_LEN / 4), (depth / 2, NECK_LEN * 0.75)]
        hi = hi + [(-depth / 2, NECK_LEN * 0.75)]
        hollow_lo = NECK_LEN * 0.75 + wall
    prof = lo + hi
    outer = P.extrude(prof, width)
    inner = P.extrude([(-depth / 2 + wall, hollow_lo),
                       (depth / 2 - wall, hollow_lo),
                       (depth / 2 - wall, hollow_hi),
                       (-depth / 2 + wall, hollow_hi)], width - 2 * wall,
                      at=(0, 0, wall))
    beam = difference(outer, inner)
    # map: profile y (length) -> world z, extrude z (width) -> world y
    beam.apply_transform(trimesh.transformations.rotation_matrix(
        math.pi / 2, (1, 0, 0)))
    beam.apply_translation((0, width / 2, 0))
    parts = [beam]
    for at_top, present in ((False, flange_lo), (True, flange_hi)):
        if not present:
            continue
        z = length - FLANGE_T if at_top else 0.0
        fl = box(depth + 2 * FLANGE_RIM, width + 2 * FLANGE_RIM, FLANGE_T,
                 (0, 0, z))
        fl = difference(fl, *holes(M5, FLANGE_T, _flange_bolts(depth, width),
                                   z=z - 1))
        parts.append(fl)
    return union(*parts)


def _flange_bolts(depth, width):
    dx, dy = depth / 2 + FLANGE_RIM / 2, width / 2 + FLANGE_RIM / 2
    return [(sx * dx, sy * dy * 0.5) for sx in (-1, 1) for sy in (-1, 1)] + \
           [(sx * dx * 0.4, sy * dy) for sx in (-1, 1) for sy in (-1, 1)]


def _root_mount(joint: str, cs, beam_width) -> trimesh.Trimesh:
    """Side plate bolting to the gearbox output journal + gusset reaching
    the beam.  Local frame matches the beam (z = beam axis, y = joint
    axis); the joint axis passes through the origin."""
    w = W_OUT[joint]
    plate_d = cycloidal.head_d(cs) * 0.9
    plate = cyl_y(plate_d, 10.0, (0, -w, 0), seg=64)
    # gusset reaches the beam on the beam side of the joint only, so the
    # opposing link's beam can fold past
    gusset = box(plate_d * 0.6, w - beam_width / 2 + 2, 70,
                 (0, -(w + beam_width / 2) / 2 + 1, 0))
    body = union(plate, gusset)
    cut = [cyl_y(M5, 12, (cx, -w - 1, cy), seg=24)
           for cx, cy in bolt_circle(6, cs.output_pin_circle_r * 0.55, 45)]
    return difference(body, *cut)


def _motor_plate(joint: str, cs, beam_width) -> trimesh.Trimesh:
    """Plate the next joint's motor bolts through, plus the web reaching
    back across the gearbox to the beam.  Plate normal = joint axis."""
    nema = NEMA[cycloidal.NEMA_OF[cs.name]]
    wm = motor_face_offset(joint, cs)
    pd = nema["face"] + 24
    plate = box(pd, 10.0, pd, (0, -wm - 5, -pd / 2))   # centred on the axis
    # the web routes BELOW the gearbox envelope (clear of housing radius)
    # then a strap rises outside the motor-face plane to meet the plate
    clear_r = cs.housing_od / 2 + 6.0
    web = box(pd, wm - beam_width / 2 + 2, 38,
              (0, -(wm + beam_width / 2) / 2 + 1, -clear_r - 38))
    strap = box(pd, 10.0, clear_r + 6 - pd / 2,
                (0, -wm - 5, -clear_r - 2))
    body = union(plate, web, strap)
    cut = [cyl_y(nema["boss_d"], 12, (0, -wm - 11, 0))]
    cut += [cyl_y(nema["bolt_d"], 12, (cx, -wm - 11, cy))
            for cx, cy in P.nema_bolt_centers(cycloidal.NEMA_OF[cs.name])]
    return difference(body, *cut)


UB, FB = spec.UPPER_BEAM, spec.FOREARM_BEAM
# segment splits keep every part (incl. mounts) inside the print volume
UPPER_SEGS = (150.0, 180.0, 180.0, 135.0)  # = UPPER_ARM_MM, 2 identical mids
FOREARM_SEGS = (160.0, 180.0, 160.0)       # = FOREARM_MM


JOINT_GAP = 40.0   # beams stop this short of joint axes so links can fold


def upper_root() -> trimesh.Trimesh:
    """Shoulder end of the upper link: output-journal mount plate +
    beam segment with a joint flange on top.  Print 1, standing."""
    b = _beam(UPPER_SEGS[0] - JOINT_GAP, UB["depth"], UB["width"],
              UB["wall"], flange_hi=True)
    b.apply_translation((0, 0, JOINT_GAP))
    mount = _root_mount("shoulder", spec.CYCLO_BIG, UB["width"])
    return union(b, mount)


def upper_mid() -> trimesh.Trimesh:
    """Middle segment of the upper link, flanges both ends.  Print 2."""
    return _beam(UPPER_SEGS[1], UB["depth"], UB["width"], UB["wall"],
                 flange_lo=True, flange_hi=True)


def upper_tip() -> trimesh.Trimesh:
    """Elbow end of the upper link: tapered neck + elbow motor plate.
    Print 1."""
    L = UPPER_SEGS[3]
    b = _beam(L - JOINT_GAP, UB["depth"], UB["width"], UB["wall"],
              neck=UB["neck_depth"], flange_lo=True)
    mp = _motor_plate("elbow", spec.CYCLO_MID, UB["width"])
    mp.apply_translation((0, 0, L))
    return union(b, mp)


def forearm_root() -> trimesh.Trimesh:
    """Elbow end of the forearm: output mount + beam with a lead-in neck
    so it clears the upper link's neck when the elbow folds.  Print 1."""
    b = _beam(FOREARM_SEGS[0] - JOINT_GAP, FB["depth"], FB["width"],
              FB["wall"], neck_lo=FB["neck_depth"], flange_hi=True)
    b.apply_translation((0, 0, JOINT_GAP))
    mount = _root_mount("elbow", spec.CYCLO_MID, FB["width"])
    return union(b, mount)


def forearm_mid() -> trimesh.Trimesh:
    """Middle segment of the forearm, flanges both ends.  Print 1."""
    return _beam(FOREARM_SEGS[1], FB["depth"], FB["width"], FB["wall"],
                 flange_lo=True, flange_hi=True)


def forearm_tip() -> trimesh.Trimesh:
    """Wrist end of the forearm: tapered neck + wrist motor plate."""
    L = FOREARM_SEGS[2]
    b = _beam(L - JOINT_GAP, FB["depth"], FB["width"], FB["wall"],
              neck=FB["neck_depth"], flange_lo=True)
    mp = _motor_plate("wrist", spec.CYCLO_SMALL, FB["width"])
    mp.apply_translation((0, 0, L))
    return union(b, mp)


# ---------------------------------------------------------------------------
# Off-the-shelf arm integration (see docs/ALTERNATIVES.md)
# ---------------------------------------------------------------------------

# VERIFY against the OpenArm STEP (enactic/openarm_hardware) before
# printing — the base plate is documented as "evenly spaced M6 taps".
OPENARM_BASE_GRID_MM = 40.0
OPENARM_GRID_N = 4                 # 4x4 M6 grid assumed


def openarm_adapter() -> trimesh.Trimesh:
    """Adapter: Z-stage gantry plate (20 mm M5 grid, C-beam standard) on
    one face -> OpenArm base M6 grid on the other, for hanging the arm
    inverted from the drop stage.  Print 1 in PETG-CF, 100% perimeters."""
    t = 14.0
    span = OPENARM_BASE_GRID_MM * (OPENARM_GRID_N - 1)
    plate = rounded_plate(span + 60, span + 60, t)
    cut = []
    # OpenArm base: M6 clearance, counterbored from the gantry side
    g0 = -span / 2
    m6 = [(g0 + OPENARM_BASE_GRID_MM * i, g0 + OPENARM_BASE_GRID_MM * j)
          for i in range(OPENARM_GRID_N) for j in range(OPENARM_GRID_N)]
    cut += holes(6.4, t, m6)
    cut += [cyl(11.5, 7.0, (cx, cy, t - 7.0 + 1)) for cx, cy in m6]
    # C-beam gantry plate pattern: M5 on a 20 mm grid ring
    ring = [(sx * 70, sy * 70) for sx in (-1, 1) for sy in (-1, 1)]
    ring += [(sx * 70, 0) for sx in (-1, 1)] + [(0, sy * 70) for sy in (-1, 1)]
    cut += holes(M5, t, ring)
    cut.append(cyl(36, t))             # cable pass-through
    return difference(plate, *cut)


# ---------------------------------------------------------------------------
# Wrist roll + leadscrew gripper
# ---------------------------------------------------------------------------

def roll_housing() -> trimesh.Trimesh:
    """Wrist-roll module: NEMA 23 drives a 5:1 GT3 belt to the roll pulley
    riding in two 6806 bearings.  Bolts to the wrist-pitch output."""
    t = 10.0
    n = NEMA[23]
    body = box(130, 70, t)
    cut = [cyl(n["boss_d"], t, (-38, 0, 0))]
    cut += holes(n["bolt_d"], t, [(-38 + cx, cy) for cx, cy in P.nema_bolt_centers(23)])
    cut.append(cyl(42.2, t, (32, 0, 0)))                # 6806 bearing seat
    cut += holes(M5, t, bolt_circle(6,
                 spec.CYCLO_SMALL.output_pin_circle_r * 0.55, 45))
    return difference(body, *cut)


def roll_pulley() -> trimesh.Trimesh:
    """100T GT3 output pulley + gripper mount flange.  Print 1."""
    teeth, pitch = 100, 2.0
    r = teeth * pitch / (2 * math.pi)
    body = cyl(2 * r, 14)
    cut = []
    for i in range(teeth):
        a = 2 * math.pi * i / teeth
        cut.append(cyl(1.4, 14, (r * math.cos(a), r * math.sin(a), 0), seg=16))
    flange = cyl(2 * r + 10, 4, (0, 0, 14))
    body = union(body, flange)
    cut.append(cyl(30.2, 19, (0, 0, -0.5)))             # bore over bearing sleeve
    cut += holes(M4, 18, bolt_circle(4, 24, 45))        # gripper mount bolts
    return difference(body, *cut)


def gripper_body() -> trimesh.Trimesh:
    """Leadscrew gripper frame: NEMA 17 turns a T8 LH/RH leadscrew that
    pulls both jaws together — self-locking, so holding 15 lb costs no
    motor current.  Jaws slide on printed rails.  Print 1."""
    t = 12.0
    body = box(160, 70, t)
    rails = [box(160, 8, 8, (0, dy, t)) for dy in (-22, 22)]   # jaw guides
    motor_wall = box(t, 70, 54, (-80 + t / 2, 0, 0))
    end_wall = box(t, 70, 40, (80 - t / 2, 0, 0))
    body = union(body, *rails, motor_wall, end_wall)
    n = NEMA[17]
    cut = [cyl_x(n["boss_d"], t + 2, (-81, 0, 30))]
    cut += [cyl_x(n["bolt_d"], t + 2, (-81, cy, 30 + cx), seg=24)
            for cx, cy in P.nema_bolt_centers(17)]
    cut.append(cyl_x(8.4, t + 2, (73, 0, 30), seg=24))   # screw end bearing
    cut += holes(M4, t, bolt_circle(4, 24, 45))          # to roll pulley
    return difference(body, *cut)


def gripper_jaw(mirrored: bool = False) -> trimesh.Trimesh:
    """Sliding jaw with T8 nut pocket and grooved grip face.  Print 1 of
    each hand; add TPU pads."""
    body = box(30, 64, 60)
    finger = box(30, 12, 70, (0, -26, 60))
    body = union(body, finger)
    cut = [cyl_x(10.4, 32, (-16, 0, 30), seg=24)]        # T8 screw clearance
    cut.append(box(22.4, 22.4, 12, (0, 0, 24)))          # T8 nut pocket
    cut += [box(32, 8.4, 8.4, (0, dy, -0.2)) for dy in (-22, 22)]  # rail slots
    # grip grooves
    cut += [box(32, 4, 5, (0, -33, 70 + dz)) for dz in range(0, 50, 12)]
    body = difference(body, *cut)
    if mirrored:
        body.apply_scale((1, -1, 1))
        if body.volume < 0:
            body.invert()
    return body
