"""All non-gearbox printable parts for the SkyArm.

Print-quantity and material notes live in the manifest emitted by
``generate.py``; geometry is parametric off skyarm.spec so resizing the
room, tube, or motors regenerates everything consistently.

Structural parts should be printed in PETG-CF or ABS, 6 walls, 40-60%
gyroid infill.  Gearbox internals (discs/cams) like PETG or nylon.
"""

from __future__ import annotations

import math

import trimesh

from .. import spec
from . import cycloidal
from . import primitives as P
from .primitives import NEMA, bolt_circle, box, cyl, cyl_x, cyl_y, difference, holes, rounded_plate, tube, union

TUBE_D = spec.TUBE_OD_MM          # arm tube OD
CLAMP_OFFSET = 30.0               # output clamp offset past the joint axis
WHEEL_HOLE = 5.2                  # M5 axles for V wheels
M5, M4, M3 = 5.2, 4.4, 3.4
TNUT_PITCH = 20.0                 # V-slot channel spacing


# ---------------------------------------------------------------------------
# Gantry
# ---------------------------------------------------------------------------

def ceiling_bracket() -> trimesh.Trimesh:
    """L-bracket: lag-screws into a ceiling joist, M5 T-nuts into the 2040
    X rail.  Print 8+ (one every ~45 cm of rail).  Print laying on its side."""
    t = 8.0
    base = rounded_plate(90, 60, t)                       # against ceiling
    wall = box(90, t, 52, (0, 30 - t / 2, 0))
    # gusset ribs tying wall to base
    ribs = [box(8, 38, 38, (dx, 4, t)) for dx in (-37, 37)]
    body = union(base, wall, *ribs)
    # 2x lag screw holes in base
    base_cut = [cyl(6.5, t, (dx, -12, 0)) for dx in (-25, 25)]
    # 2x M5 holes into rail T-nuts on the wall, on 20 mm channel pitch
    wall_cut = [cyl_y(M5, t, (dx, 30 - t, 14 + TNUT_PITCH * i))
                for dx in (-25, 25) for i in (0, 1)]
    return difference(body, *base_cut, *wall_cut)


def x_truck() -> trimesh.Trimesh:
    """Wheeled truck riding a ceiling 2040 rail (wheels on the 40 mm faces).
    Print 2.  Holds one bridge endcap and the X belt clamp."""
    t = 9.0
    plate = rounded_plate(120, 150, t)
    body = plate
    cut = []
    # 4 V-wheel axles: spacing across the 40 mm extrusion = 64.4 mm
    for sx in (-1, 1):
        for sy in (-1, 1):
            cut.append(cyl(WHEEL_HOLE, t, (sx * 32.2, sy * 55, 0)))
    # bridge endcap bolt pattern (4x M5)
    cut += holes(M5, t, [(sx * 40, sy * 25) for sx in (-1, 1) for sy in (-1, 1)])
    # belt clamp slots (belt loops through and locks on teeth)
    cut += [box(12, 3, t, (dx, 0, 0)) for dx in (-12, 12)]
    return difference(body, *cut)


def bridge_endcap() -> trimesh.Trimesh:
    """Joins the 2060 bridge beam end to an X truck.  Print 2."""
    t = 10.0
    face = rounded_plate(70, 130, t)                    # against 2060 end
    cut = holes(M5, t, [(0, dy) for dy in (-40, -20, 0, 20, 40)])  # beam T-nuts
    cut += holes(M5, t, [(sx * 22, sy * 52) for sx in (-1, 1) for sy in (-1, 1)])
    return difference(face, *cut)


def y_carriage() -> trimesh.Trimesh:
    """Main carriage riding the 2060 bridge; the yaw gearbox housing bolts
    to its underside on a 6-bolt circle.  Print 1."""
    t = 10.0
    span = spec.CARRIAGE_WHEEL_SPAN_MM
    plate = rounded_plate(span + 60, 150, t)
    cut = []
    # 6 V wheels straddling the bridge beam; wheelbase from spec (the
    # V-wheel load check in skyarm/analysis.py sizes it)
    for sy in (-1, 1):
        for dx in (-span / 2, 0, span / 2):
            cut.append(cyl(WHEEL_HOLE, t, (dx, sy * 42.2, 0)))
    # belt clamp slots
    cut += [box(12, 3, t, (dx, 0, 0)) for dx in (-30, 30)]
    # yaw gearbox mount: bolt circle matching cyclo-yaw housing OD flange
    box_r = spec.CYCLO_YAW.housing_od / 2 - 6
    cut += holes(M5, t, bolt_circle(6, box_r))
    cut.append(cyl(40, t, (0, 0, 0)))                   # cable pass-through
    return difference(plate, *cut)


def gantry_motor_mount() -> trimesh.Trimesh:
    """NEMA 23 bracket for the rail-end belt drives.  Print 3 (2x X, 1x Y)."""
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
    body = box(40, 24, 16)
    cut = [cyl(M5, 16, (6, 0, 0))]                       # idler axle
    cut.append(cyl_x(M4, 41, (-21, 0, 8), seg=24))       # tension screw
    cut.append(box(20, 6.4, 16, (-8, 0, 0)))             # slide slot
    return difference(body, *cut)


# ---------------------------------------------------------------------------
# Arm structure
# ---------------------------------------------------------------------------

def _tube_clamp_block(length: float = 46.0,
                      width: float | None = None) -> trimesh.Trimesh:
    """Solid block with a horizontal tube bore + pinch slot + 2 M4 bolts.
    The bore runs along x at mid-height; width tracks the tube size."""
    width = width if width is not None else TUBE_D + 16
    h = TUBE_D + 18
    body = box(length, width, h)
    bore = cyl_x(TUBE_D + 0.3, length + 2, (-length / 2 - 1, 0, h / 2))
    slot = box(length + 2, 2.4, h / 2, (0, 0, h / 2))
    body = difference(body, bore, slot)
    bolt_cuts = [P.cyl(M4, h + 2, (dx, dy, -1), seg=24)
                 for dx in (-length / 2 + 9, length / 2 - 9)
                 for dy in (-width / 2 + 8, width / 2 - 8)]
    return difference(body, *bolt_cuts)


def shoulder_clevis() -> trimesh.Trimesh:
    """Bolts to the shoulder gearbox output flange (4x M5 on the spec'd
    circle); clamps the upper-arm tube.  Print 1 + 1 mirrored at slicer."""
    g = spec.CYCLO_BIG
    plate_t = 8.0
    plate = cyl(g.output_pin_circle_r * 1.4, plate_t)
    cut = holes(M5, plate_t, bolt_circle(4, g.output_pin_circle_r * 0.55, 45))
    plate = difference(plate, *cut)
    clamp = _tube_clamp_block()
    # clamp sits 30 mm past the joint axis along the tube so the incoming
    # and outgoing links clear each other when the joint folds
    clamp.apply_translation((CLAMP_OFFSET, 0, plate_t))
    web = box(36, 30, plate_t, (CLAMP_OFFSET / 2, 0, 0))
    return union(plate, web, clamp)


def _input_standoff(cs, plate_t: float) -> float:
    """Web height putting the tube-clamp bore exactly on the arm plane.

    The motor face sits ``output_face_z + 0.2 + clevis plate + half
    clamp`` from the arm plane; the bracket plate (against the motor
    face) plus this web plus half the clamp must span the same distance,
    so the input and output of a joint grip coaxial tubes."""
    half_clamp = (TUBE_D + 18) / 2
    motor_face = cycloidal.output_face_z(cs) + 0.2 + 8.0 + half_clamp
    return motor_face - plate_t - half_clamp


def elbow_clevis() -> trimesh.Trimesh:
    """Upper-arm tube -> elbow gearbox.  The elbow motor bolts through the
    side plate into the gearbox housing; the web drops the tube clamp to
    the arm plane so the upper and forearm tubes stay coaxial.  Print 1."""
    side_t = 9.0
    n = NEMA[23]
    side = rounded_plate(n["face"] + 20, n["face"] + 20, side_t)
    cut = [cyl(n["boss_d"], side_t)]
    cut += holes(n["bolt_d"], side_t, P.nema_bolt_centers(23))
    side = difference(side, *cut)
    s = _input_standoff(spec.CYCLO_MID, side_t)
    web = box(40, 34, s, (0, 0, side_t))
    clamp = _tube_clamp_block()
    clamp.apply_translation((0, 0, side_t + s))
    return union(side, web, clamp)


def forearm_clevis() -> trimesh.Trimesh:
    """Elbow gearbox output flange -> forearm tube.  Same interface as the
    shoulder clevis but sized for the mid gearbox.  Print 1."""
    g = spec.CYCLO_MID
    plate_t = 8.0
    plate = cyl(g.output_pin_circle_r * 1.5, plate_t)
    cut = holes(M5, plate_t, bolt_circle(4, g.output_pin_circle_r * 0.55, 45))
    plate = difference(plate, *cut)
    clamp = _tube_clamp_block(length=40)
    clamp.apply_translation((CLAMP_OFFSET, 0, plate_t))
    web = box(36, 28, plate_t, (CLAMP_OFFSET / 2, 0, 0))
    return union(plate, web, clamp)


def wrist_bracket() -> trimesh.Trimesh:
    """Forearm tube -> wrist-pitch gearbox (NEMA 17 cyclo-small); same
    standoff-web construction as the elbow clevis.  Print 1."""
    side_t = 8.0
    n = NEMA[17]
    side = rounded_plate(n["face"] + 18, n["face"] + 18, side_t)
    cut = [cyl(n["boss_d"], side_t)]
    cut += holes(n["bolt_d"], side_t, P.nema_bolt_centers(17))
    side = difference(side, *cut)
    s = _input_standoff(spec.CYCLO_SMALL, side_t)
    web = box(32, 32, s, (0, 0, side_t))
    clamp = _tube_clamp_block(length=36)
    clamp.apply_translation((0, 0, side_t + s))
    return union(side, web, clamp)


def roll_housing() -> trimesh.Trimesh:
    """Wrist-roll module: NEMA 17 drives a 5:1 GT2 belt to the roll pulley
    riding in two 6806 bearings.  Bolts to the wrist-pitch output. Print 1."""
    t = 8.0
    n = NEMA[17]
    body = box(110, 56, t)
    cut = [cyl(n["boss_d"], t, (-32, 0, 0))]
    cut += holes(n["bolt_d"], t, [(-32 + cx, cy) for cx, cy in P.nema_bolt_centers(17)])
    cut.append(cyl(42.2, t, (28, 0, 0)))                # 6806 bearing seat
    cut += holes(M4, t, bolt_circle(4, spec.CYCLO_SMALL.output_pin_circle_r * 0.55, 45))
    return difference(body, *cut)


def roll_pulley() -> trimesh.Trimesh:
    """100T GT2 output pulley + gripper mount flange (teeth approximated;
    print at 0.12 mm layers or substitute an aluminium pulley).  Print 1."""
    teeth, pitch = 100, 2.0
    r = teeth * pitch / (2 * math.pi)                   # ~31.8 mm
    body = cyl(2 * r, 12)
    cut = []
    for i in range(teeth):
        a = 2 * math.pi * i / teeth
        cut.append(cyl(1.4, 12, (r * math.cos(a), r * math.sin(a), 0), seg=16))
    flange = cyl(2 * r + 8, 3, (0, 0, 12))
    body = union(body, flange)
    cut.append(cyl(30.2, 16, (0, 0, -0.5)))             # bore over bearing sleeve
    cut += holes(M4, 15, bolt_circle(4, 22, 45))        # gripper mount bolts
    return difference(body, *cut)


# ---------------------------------------------------------------------------
# Gripper (servo, gear-synced fingers)
# ---------------------------------------------------------------------------

def gripper_base() -> trimesh.Trimesh:
    """Holds the DS3225 servo + two finger pivots.  Print 1."""
    t = 10.0
    body = box(90, 60, t)
    cut = [box(40.5, 20.5, t, (0, 10, 0))]              # servo pocket
    cut += holes(M3, t, [(sx * 24.5, 10 + sy * 5) for sx in (-1, 1) for sy in (-1, 1)])
    cut += holes(M5, t, [(-15, -18), (15, -18)])        # finger pivots
    cut += holes(M4, t, bolt_circle(4, 22, 45))         # to roll pulley flange
    return difference(body, *cut)


def gripper_finger(mirrored: bool = False) -> trimesh.Trimesh:
    """Finger with integral gear segment; the two fingers mesh so one servo
    drives both.  Print 1 normal + 1 mirrored."""
    t = 9.0
    # gear segment at pivot
    seg_r = 16.0
    hub = cyl(2 * seg_r, t)
    teeth = []
    for i in range(-3, 4):
        a = math.radians(i * 14)
        teeth.append(cyl(4.0, t, ((seg_r + 1.2) * math.cos(a),
                                  (seg_r + 1.2) * math.sin(a), 0), seg=16))
    hub = union(hub, *teeth)
    # finger beam with a curved grip face
    beam = P.extrude([(0, -8), (70, -4), (78, 6), (70, 8), (30, 10), (0, 10)], t)
    pads = [cyl(6, t, (x, 11, 0), seg=24) for x in (45, 60, 74)]
    body = union(hub, beam, *pads)
    body = difference(body, cyl(M5, t, (0, 0, 0)))      # pivot bore
    if mirrored:
        body.apply_scale((1, -1, 1))
        if body.volume < 0:
            body.invert()
    return body
