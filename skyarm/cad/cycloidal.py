"""Printable cycloidal reducer generator.

One `CycloSpec` (from skyarm.spec) yields a five-part gearbox:

    *_housing.stl   ring with integral pins + motor flange (print flange down)
    *_disc.stl      cycloid disc — PRINT TWO, they run 180 deg out of phase
    *_cam.stl       eccentric input cam, D-bore for the motor shaft
    *_output.stl    output flange with drive pins
    *_cover.stl     front cover holding the output bearing

Why cycloidal: zero-backlash-class precision from a printer.  The disc
rolls on ring pins; reduction = pin_count - 1; load is shared across
~1/3 of the pins simultaneously, so printed parts carry surprising
torque.  Pair with the closed-loop steppers in the spec and the joint
output resolution lands in the milli-degree range.

Bearings (per size, see spec.CYCLO_BEARING):
    big   : cam lobes carry 6705ZZ (25x32x4), output rides 6810ZZ (50x65x7)
    mid/yaw: same bearing set as big (shared tooling)
    small : cam lobes carry 6803ZZ (17x26x5), output rides 6806ZZ (30x42x7)
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from ..spec import CycloSpec
from . import primitives as P

CLEAR = 0.25          # printed running clearance
DISC_GAP = 0.4        # axial gap around discs

# (cam_lobe_od = bearing id, disc_bore = bearing od) per gearbox size
BEARINGS = {
    "cyclo-big": dict(cam_od=25.0, disc_bore=32.0, out_id=50.0, out_od=65.0, out_w=7.0),
    "cyclo-mid": dict(cam_od=25.0, disc_bore=32.0, out_id=50.0, out_od=65.0, out_w=7.0),
    "cyclo-yaw": dict(cam_od=25.0, disc_bore=32.0, out_id=50.0, out_od=65.0, out_w=7.0),
    "cyclo-small": dict(cam_od=17.0, disc_bore=26.0, out_id=30.0, out_od=42.0, out_w=7.0),
}

NEMA_OF = {"cyclo-big": 24, "cyclo-mid": 23, "cyclo-yaw": 23, "cyclo-small": 17}


def cycloid_profile(spec: CycloSpec, samples: int = 1200) -> np.ndarray:
    """2D outline of the cycloid disc (classic epitrochoid offset curve)."""
    R, Rr, E, N = spec.pin_circle_r, spec.pin_r, spec.eccentricity, spec.pin_count
    t = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    psi = np.arctan2(np.sin((1 - N) * t), (R / (E * N)) - np.cos((1 - N) * t))
    x = R * np.cos(t) - Rr * np.cos(t + psi) - E * np.cos(N * t)
    y = -R * np.sin(t) + Rr * np.sin(t + psi) + E * np.sin(N * t)
    return np.column_stack([x, y])


def disc(spec: CycloSpec) -> trimesh.Trimesh:
    b = BEARINGS[spec.name]
    outline = cycloid_profile(spec)
    # shrink by running clearance (offset towards centre)
    from shapely.geometry import Polygon
    poly = Polygon(outline).buffer(-CLEAR, quad_segs=4)
    body = trimesh.creation.extrude_polygon(poly, spec.disc_thickness)
    cutters = [P.cyl(b["disc_bore"] + 0.1, spec.disc_thickness + 2, (0, 0, -1))]
    # output pin holes: pin_d + 2*eccentricity + clearance
    hole_d = 2 * (spec.output_pin_r + spec.eccentricity) + 2 * CLEAR
    cutters += P.holes(hole_d, spec.disc_thickness,
                       P.bolt_circle(spec.output_pin_count, spec.output_pin_circle_r))
    return P.difference(body, *cutters)


def housing(spec: CycloSpec) -> trimesh.Trimesh:
    """Motor-side flange + pin ring.  Print flange-down, no supports."""
    b = BEARINGS[spec.name]
    nema = P.NEMA[NEMA_OF[spec.name]]
    od = spec.housing_od
    flange_t = 7.0
    ring_h = 2 * spec.disc_thickness + 3 * DISC_GAP
    cavity_r = spec.pin_circle_r + spec.pin_r * 0.5   # pins half-embedded

    base = P.cyl(od, flange_t)
    ring = P.tube(od, 2 * cavity_r, ring_h, (0, 0, flange_t))
    # ring pins, printed integral, half-buried in the wall
    pins = [P.cyl(2 * spec.pin_r, ring_h, (cx, cy, flange_t), seg=48)
            for cx, cy in P.bolt_circle(spec.pin_count, spec.pin_circle_r)]
    body = P.union(base, ring, *pins)

    cutters = [P.cyl(nema["boss_d"], flange_t + 2, (0, 0, -1))]          # motor boss
    cutters += P.holes(nema["bolt_d"], flange_t, P.nema_bolt_centers(NEMA_OF[spec.name]))
    # cover bolt bosses: 6x M4 through the ring wall top
    cover_r = (cavity_r + od / 2) / 2 + spec.pin_r * 0.25
    cutters += [P.cyl(3.4, 14, (cx, cy, flange_t + ring_h - 12), seg=24)
                for cx, cy in P.bolt_circle(6, cover_r, start_deg=30)]
    return P.difference(body, *cutters)


def cam(spec: CycloSpec) -> trimesh.Trimesh:
    """Eccentric input cam: two lobes 180 deg apart, D-bore."""
    b = BEARINGS[spec.name]
    nema = P.NEMA[NEMA_OF[spec.name]]
    E, t = spec.eccentricity, spec.disc_thickness
    hub_h = 4.0
    hub = P.cyl(b["cam_od"] + 8, hub_h)
    lobe1 = P.cyl(b["cam_od"], t + DISC_GAP, (E, 0, hub_h))
    lobe2 = P.cyl(b["cam_od"], t + DISC_GAP, (-E, 0, hub_h + t + DISC_GAP))
    total_h = hub_h + 2 * (t + DISC_GAP)
    bore = P.dshaft_cutter(nema["shaft_d"] + 0.2,
                           0.5 if nema["shaft_d"] <= 5 else 1.0,
                           total_h + 2, (0, 0, -1))
    body = P.union(hub, lobe1, lobe2)
    # M3 grub screw into the flat, through the hub
    grub = P.cyl_x(2.9, b["cam_od"], (-b["cam_od"], 0, hub_h / 2), seg=24)
    return P.difference(body, bore, grub)


def output_flange(spec: CycloSpec) -> trimesh.Trimesh:
    """Drive-pin carrier; the pins pass through the disc holes and convert
    the disc's wobble into clean output rotation.  Print pins-up."""
    b = BEARINGS[spec.name]
    plate_t = 6.0
    pin_len = 2 * spec.disc_thickness + 2 * DISC_GAP
    plate = P.cyl(b["out_id"] - 0.2, plate_t + b["out_w"])  # rides in output bearing
    pins = [P.cyl(2 * spec.output_pin_r, pin_len, (cx, cy, plate_t + b["out_w"]), seg=48)
            for cx, cy in P.bolt_circle(spec.output_pin_count, spec.output_pin_circle_r)]
    body = P.union(plate, *pins)
    # output bolt pattern: 4x M5 threaded inserts on half the pin circle radius
    cutters = P.holes(4.6, plate_t + b["out_w"],
                      P.bolt_circle(4, spec.output_pin_circle_r * 0.55, 45))
    return P.difference(body, *cutters)


def cover(spec: CycloSpec) -> trimesh.Trimesh:
    """Front cover; captures the output bearing, bolts to the housing."""
    b = BEARINGS[spec.name]
    od = spec.housing_od
    t = b["out_w"] + 5.0
    cavity_r = spec.pin_circle_r + spec.pin_r * 0.5
    cover_r = (cavity_r + od / 2) / 2 + spec.pin_r * 0.25

    body = P.cyl(od, t)
    cutters = [
        P.cyl(b["out_od"] + 0.2, b["out_w"] + 0.2, (0, 0, t - b["out_w"] - 0.2)),
        P.cyl(b["out_od"] - 6, t + 2, (0, 0, -1)),       # central opening
    ]
    cutters += P.holes(4.4, t, P.bolt_circle(6, cover_r, start_deg=30))
    return P.difference(body, *cutters)


def parts(spec: CycloSpec) -> dict:
    return {
        f"{spec.name}_housing": housing(spec),
        f"{spec.name}_disc": disc(spec),          # print 2
        f"{spec.name}_cam": cam(spec),
        f"{spec.name}_output": output_flange(spec),
        f"{spec.name}_cover": cover(spec),
    }
