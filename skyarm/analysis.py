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
E_PRINT = spec.PETG_CF_E_GPA * 1e9
FLEX_PETG_CF = spec.PETG_CF_FLEX_MPA * 1e6
FLEX_PETG_CF_Z = 0.45 * FLEX_PETG_CF   # across layers (printed standing)
SHEAR_PETG_CF = 30e6
VWHEEL_RATED_N = 620.0               # Xtreme solid V wheel radial rating
BELT_RATED_N = 250.0                 # GT3 15 mm steel-core working tension
BRG_6815_C0_N = 8450.0               # shoulder output pair, static each
BRG_6810_C0_N = 6800.0               # elbow/yaw output pair
MGN15_C0_N = 16800.0                 # MGN15H block static rating
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
    return G * (spec.M_UPPER_LINK + spec.M_ELBOW_ASSY + spec.M_FOREARM_LINK
                + spec.M_WRIST_CLUSTER + spec.M_GRIPPER + spec.PAYLOAD_KG)


def _shoulder_moment_nm() -> float:
    return spec.SHOULDER.gravity_torque_nm()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def beam_section(beam):
    """Second moment + section modulus of a printed box beam (m^4, m^3)."""
    d, w, t = (beam["depth"] / 1000.0, beam["width"] / 1000.0,
               beam["wall"] / 1000.0)
    i = (w * d ** 3 - (w - 2 * t) * (d - 2 * t) ** 3) / 12.0
    return i, i / (d / 2)


def check_beam_stress() -> Check:
    """Upper-link root bending at full horizontal reach + payload.
    Beams print standing, so bending loads the LAYER BONDS — capacity is
    the derated across-layer flexural strength."""
    _, z = beam_section(spec.UPPER_BEAM)
    sigma = _shoulder_moment_nm() / z
    return Check("upper beam bending stress", sigma / 1e6,
                 FLEX_PETG_CF_Z / 1e6, "MPa",
                 "across-layer strength, arm horizontal")


def check_tip_deflection() -> Check:
    """Elastic tip drop from printed-beam bending, arm horizontal
    (two-segment cantilever superposition, E = printed composite)."""
    i1, _ = beam_section(spec.UPPER_BEAM)
    i2, _ = beam_section(spec.FOREARM_BEAM)
    l1 = spec.UPPER_ARM_MM / 1000.0
    l2 = spec.FOREARM_MM / 1000.0
    l3 = spec.WRIST_TO_TIP_MM / 1000.0
    x_tip = l1 + l2 + l3

    p_tip = G * (spec.PAYLOAD_KG + spec.M_GRIPPER + spec.M_WRIST_CLUSTER)
    p_el = G * spec.M_ELBOW_ASSY

    # upper beam: end force (p_tip + p_el) plus the end moment the
    # overhanging forearm applies; carry slope through to the tip
    f_end = p_tip + p_el
    m_end = p_tip * (l2 + l3)
    d1 = (f_end * l1 ** 3 / 3 + m_end * l1 ** 2 / 2) / (E_PRINT * i1)
    th1 = (f_end * l1 ** 2 / 2 + m_end * l1) / (E_PRINT * i1)
    d1 += G * spec.M_UPPER_LINK * l1 ** 3 / (8 * E_PRINT * i1)  # self weight
    # forearm: own cantilever flexure under the tip load
    d2 = (p_tip * l2 ** 2 * (3 * (l2 + l3) - l2) / 6
          + G * spec.M_FOREARM_LINK * l2 ** 3 / 8) / (E_PRINT * i2)
    d = d1 + th1 * (l2 + l3) + d2
    return Check("tip sag from printed-beam flex", d * 1000, 8.0, "mm",
                 "limit 8 mm at 15 lb, arm horizontal")


def check_flange_bolts() -> Check:
    """Segment-joint flange: worst tension in the outer bolt row at the
    upper link root moment (M5 class 8.8 proof ~7 kN; limit by the
    printed flange bearing instead: ~1.5 kN per bolt)."""
    m = _shoulder_moment_nm()
    d_eff = (spec.UPPER_BEAM["depth"] + 12.0) / 1000.0   # bolt row spread
    f_per_bolt = m / d_eff / 3.0                          # 3 bolts per side
    return Check("link flange bolt tension", f_per_bolt, 1500.0, "N",
                 "printed flange bearing limit per M5")


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
    """Shoulder output journal rides TWO 6815 bearings with a 40 mm
    printed spacer ledge (cycloidal.py); the moment becomes a couple
    across the 50 mm bearing-centre span, shared radial load on top."""
    from .cad import cycloidal as _c
    b = _c.BEARINGS["cyclo-big"]
    span = (b["out_w"] + b["spacing"]) / 1000.0
    radial = _arm_weight_n()
    worst = _shoulder_moment_nm() / span + radial / 2
    return Check("shoulder output bearing load", worst,
                 BRG_6815_C0_N, "N", "worst of 2x 6815ZZ, static rating")


def check_carriage_rail() -> Check:
    """Y carriage rides 2x MGN15H blocks 150 mm apart (V wheels failed
    the 15 lb overturning-moment check); worst block load = moment
    couple + half the hanging weight."""
    weight = (spec.ARM_MASS_KG + spec.PAYLOAD_KG + spec.CARRIAGE_MASS_KG) * G
    f = _shoulder_moment_nm() / 0.150 + weight / 2
    return Check("worst MGN15 block load", f, MGN15_C0_N, "N",
                 "arm horizontal at full reach")


def check_truck_wheels() -> Check:
    """Worst V-wheel load on an X truck: half the machine weight + half
    the arm-horizontal moment across the truck wheelbase."""
    from .cad.parts import TRUCK_WHEEL_SPAN
    weight = spec.X_AXIS.moving_mass_kg * G
    f = (_shoulder_moment_nm() / 2) / (TRUCK_WHEEL_SPAN / 1000.0) / 2 \
        + weight / 8
    return Check("worst X-truck wheel load", f, VWHEEL_RATED_N, "N",
                 "Xtreme solid V wheel rating")


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


ALL_CHECKS = (check_beam_stress, check_tip_deflection, check_flange_bolts,
              check_bridge_deflection, check_rail_deflection,
              check_ring_pin_shear, check_output_pin_bending,
              check_output_bearing, check_carriage_rail, check_truck_wheels,
              check_belt_tension, check_ceiling_anchors)

# minimum safety factors by check (structure 2.0, consumables 1.3)
MIN_SF = {"gantry belt working tension": 1.3,
          "tip sag from printed-beam flex": 1.0,
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
