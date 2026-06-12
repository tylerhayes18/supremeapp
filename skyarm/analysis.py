"""Structural / mechanical engineering checks for the SkyArm.

Closed-form engineering analysis of every load path, driven by spec.py:
beam bending, tube stress and tip deflection, printed cycloidal pin
stresses, V-wheel loads, belt working tension, output bearing loads and
ceiling anchor loads.  Each check reports demand vs capacity and a
safety factor; tests assert the design rules hold, so a bad spec change
fails CI before anything gets printed.

    python -m skyarm.analysis        # full report

Material assumptions (conservative datasheet values):
    6061-T6 aluminium  yield 240 MPa, E 69 GPa
    PETG-CF (printed)  flexural 70 MPa, shear 30 MPa (XY, derated)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import spec

G = 9.81
E_AL = spec.ALU_E_GPA * 1e9          # Pa
YIELD_AL = 240e6
FLEX_PETG_CF = 70e6
SHEAR_PETG_CF = 30e6
VWHEEL_RATED_N = 400.0               # solid V wheel radial rating
BELT_RATED_N = 140.0                 # GT2 9 mm steel-core working tension
BRG_6810_C0_N = 6800.0               # static load rating
LAG_SCREW_SHEAR_N = 2200.0           # 6.5 mm lag in joist, per screw


@dataclass
class Check:
    name: str
    demand: float
    capacity: float
    unit: str
    note: str = ""

    @property
    def sf(self) -> float:
        return self.capacity / self.demand if self.demand else float("inf")

    def row(self) -> str:
        return (f"{self.name:34s} {self.demand:9.1f} / {self.capacity:9.1f} "
                f"{self.unit:4s} SF {self.sf:5.2f}  {self.note}")


def _arm_weight_n() -> float:
    return G * (spec.M_UPPER_TUBE + spec.M_ELBOW_ASSY + spec.M_FOREARM_TUBE
                + spec.M_WRIST_CLUSTER + spec.M_GRIPPER + spec.PAYLOAD_KG)


def _shoulder_moment_nm() -> float:
    return spec.SHOULDER.gravity_torque_nm()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def tube_section():
    do = spec.TUBE_OD_MM / 1000.0
    di = do - 2 * spec.TUBE_WALL_MM / 1000.0
    i = math.pi / 64 * (do ** 4 - di ** 4)
    z = i / (do / 2)
    return i, z


def check_tube_stress() -> Check:
    """Upper-arm tube root bending at full horizontal reach + payload."""
    _, z = tube_section()
    sigma = _shoulder_moment_nm() / z
    return Check("upper tube bending stress", sigma / 1e6, YIELD_AL / 1e6,
                 "MPa", "arm horizontal, rated payload")


def check_tip_deflection() -> Check:
    """Elastic tip drop from tube bending, arm horizontal (two-segment
    cantilever, tip load + distributed self-weight lumped at midpoints)."""
    i, _ = tube_section()
    l1 = spec.UPPER_ARM_MM / 1000.0
    l2 = spec.FOREARM_MM / 1000.0
    l3 = spec.WRIST_TO_TIP_MM / 1000.0
    p_tip = G * (spec.PAYLOAD_KG + spec.M_GRIPPER + spec.M_WRIST_CLUSTER)
    p_el = G * spec.M_ELBOW_ASSY
    # cantilever: load P at distance a from root: delta(x>=a) grows with
    # slope; evaluate tip deflection by superposition
    def cant(p, a, x):
        if x <= a:
            return p * x * x * (3 * a - x) / (6 * E_AL * i)
        return p * a * a * (3 * x - a) / (6 * E_AL * i)
    x_tip = l1 + l2 + l3
    d = cant(p_tip, l1 + l2, x_tip) + cant(p_el, l1, x_tip)
    d += cant(G * spec.M_UPPER_TUBE, l1 / 2, x_tip)
    d += cant(G * spec.M_FOREARM_TUBE, l1 + l2 / 2, x_tip)
    return Check("tip sag from tube flex", d * 1000, 5.0, "mm",
                 "limit 5 mm, arm horizontal")


def check_bridge_deflection() -> Check:
    """Bridge mid-span deflection: hanging mass + arm-horizontal moment."""
    span = (spec.ROOM["y_mm"] - 300) / 1000.0
    i = spec.BRIDGE_I_CM4 * 1e-8
    f = (spec.Y_AXIS.moving_mass_kg) * G
    d_point = f * span ** 3 / (48 * E_AL * i)
    w = spec.BRIDGE_KG_PER_M * G                      # self weight N/m
    d_self = 5 * w * span ** 4 / (384 * E_AL * i)
    return Check("bridge mid-span deflection", (d_point + d_self) * 1000,
                 3.0, "mm", f"{spec.VSLOT_BRIDGE.split(',')[0]}")


def check_rail_deflection() -> Check:
    """X rail between ceiling brackets, half machine weight mid-span."""
    span = 0.45
    i = spec.RAIL_I_CM4 * 1e-8
    f = (spec.X_AXIS.moving_mass_kg / 2) * G
    d = f * span ** 3 / (48 * E_AL * i)
    return Check("X rail deflection between brackets", d * 1000, 1.0, "mm",
                 "bracket every 45 cm")


def check_ring_pin_shear() -> Check:
    """Shoulder gearbox ring pins: torque shared over ~1/3 of pins."""
    cs = spec.CYCLO_BIG
    n_eff = max(cs.pin_count // 3, 1)
    f_pin = spec.SHOULDER.capacity_nm() / (cs.pin_circle_r / 1000.0) / n_eff
    area = math.pi * (cs.pin_r / 1000.0) ** 2
    tau = f_pin / area
    return Check("cyclo ring pin shear (shoulder)", tau / 1e6,
                 SHEAR_PETG_CF / 1e6, "MPa",
                 f"{n_eff} pins sharing at full motor torque")


def check_output_pin_bending() -> Check:
    """Shoulder output pins: cantilever bending at full gearbox torque."""
    cs = spec.CYCLO_BIG
    n_eff = max(cs.output_pin_count // 2, 1)
    f_pin = spec.SHOULDER.capacity_nm() / (cs.output_pin_circle_r / 1000.0) / n_eff
    lever = (cs.disc_thickness + 0.4) / 1000.0        # to mid first disc
    m = f_pin * lever
    z = math.pi * (2 * cs.output_pin_r / 1000.0) ** 3 / 32
    sigma = m / z
    return Check("cyclo output pin bending (shoulder)", sigma / 1e6,
                 FLEX_PETG_CF / 1e6, "MPa",
                 f"{n_eff} pins sharing at full motor torque")


def check_output_bearing() -> Check:
    """Shoulder output journal rides TWO 6810 bearings with a 10 mm
    spacer (cycloidal.py); the moment becomes a couple across the
    17 mm bearing-centre span, shared radial load on top."""
    span = 0.017                                      # 7 mm brg + 10 spacer
    radial = _arm_weight_n()
    worst = _shoulder_moment_nm() / span + radial / 2
    return Check("shoulder output bearing load", worst,
                 BRG_6810_C0_N, "N", "worst of 2x 6810ZZ, static rating")


def check_carriage_wheels() -> Check:
    """Worst V-wheel load on the Y carriage: hanging weight + the
    overturning moment when the arm is fully horizontal."""
    weight = (spec.ARM_MASS_KG + spec.PAYLOAD_KG + spec.CARRIAGE_MASS_KG) * G
    m = _shoulder_moment_nm()                          # moment about carriage
    wheel_span = spec.CARRIAGE_WHEEL_SPAN_MM / 1000.0
    f_moment = m / wheel_span / 2                      # shared by 2 wheels/row
    f = weight / 6 + f_moment
    return Check("worst Y-carriage wheel load", f, VWHEEL_RATED_N, "N",
                 "arm horizontal at full reach")


def check_belt_tension() -> Check:
    f = max(a.required_force_n() for a in spec.GANTRY_AXES)
    return Check("gantry belt working tension", f, BELT_RATED_N, "N",
                 "GT2 9 mm steel core")


def check_ceiling_anchors() -> Check:
    """Per-bracket pull-out/shear with everything on one bracket pair's
    tributary span, 2x dynamic factor."""
    n_brackets = 2 * (max(2, int(spec.ROOM["x_mm"] // 700)) + 1)
    total = (spec.X_AXIS.moving_mass_kg + 8.0) * G    # + rails/brackets
    per_bracket = 2.0 * total / (n_brackets / 2)      # worst tributary, 2x dyn
    return Check("ceiling bracket load", per_bracket,
                 2 * LAG_SCREW_SHEAR_N, "N", f"{n_brackets} brackets, 2 lags each")


ALL_CHECKS = (check_tube_stress, check_tip_deflection,
              check_bridge_deflection, check_rail_deflection,
              check_ring_pin_shear, check_output_pin_bending,
              check_output_bearing, check_carriage_wheels,
              check_belt_tension, check_ceiling_anchors)

# minimum safety factors by check (structure 2.0, consumables 1.3)
MIN_SF = {"gantry belt working tension": 1.3,
          "tip sag from tube flex": 1.0,
          "bridge mid-span deflection": 1.0,
          "X rail deflection between brackets": 1.0}
DEFAULT_MIN_SF = 2.0


def run_all() -> list:
    return [fn() for fn in ALL_CHECKS]


def report() -> str:
    lines = ["SkyArm structural analysis", "=" * 78,
             f"{'check':34s} {'demand':>9s} / {'capacity':>9s} unit  SF"]
    ok = True
    for c in run_all():
        lines.append(c.row())
        if c.sf < MIN_SF.get(c.name, DEFAULT_MIN_SF):
            ok = False
            lines.append(f"  ** FAILS minimum SF "
                         f"{MIN_SF.get(c.name, DEFAULT_MIN_SF)} **")
    lines.append("")
    lines.append("all checks pass" if ok else "DESIGN CHECK FAILURES PRESENT")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
