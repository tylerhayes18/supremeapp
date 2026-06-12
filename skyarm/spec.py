"""SkyArm engineering specification — the single source of truth.

Every dimension, motor choice, gear ratio, mass budget and torque
calculation for the ceiling-mounted gantry arm lives in this module.
The CAD generator, kinematics, simulator and docs all import from here,
so changing a number here propagates through the whole design.

Units: millimetres, kilograms, newton-metres, degrees (unless noted).

Coordinate convention (room frame):
    x, y  — horizontal, along the two gantry axes
    z     — up; floor at z = 0, ceiling at z = ROOM["ceiling_mm"]

System overview
---------------
1. Two parallel 2040 V-slot rails bolt to ceiling joists (X axis).
2. A 2060 V-slot bridge beam spans them on wheeled trucks, driven by
   two electronically-synced closed-loop NEMA 23 belt drives.
3. A carriage rides the bridge (Y axis, one more NEMA 23 belt drive).
4. The arm hangs from the carriage:
       J1 yaw  (vertical axis)   NEMA 23 + 15:1 printed cycloidal
       J2 shoulder pitch         NEMA 24 + 25:1 printed cycloidal
       J3 elbow pitch            NEMA 23 + 20:1 printed cycloidal
       J4 wrist pitch            NEMA 17 + 11:1 printed cycloidal
       J5 wrist roll             NEMA 17 + 5:1 GT2 belt
       Gripper                   DS3225 25 kg·cm servo, linkage fingers
5. Long members are off-the-shelf 25.4 mm (1") OD aluminium tube;
   everything else (joints, gearboxes, trucks, gripper) is printed.

All five joints use printed cycloidal reducers because they are
near-zero-backlash — combined with closed-loop steppers (4000 count
encoders) the tip repeatability budget works out to ~0.1 mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

G = 9.81  # m/s^2
MM_PER_FT = 304.8

# ---------------------------------------------------------------------------
# Reach budget — must total exactly 5 ft from the ceiling plane to tip.
# ---------------------------------------------------------------------------

TOTAL_REACH_MM = 5 * MM_PER_FT  # 1524.0

CARRIAGE_STACK_MM = 200.0   # ceiling plane -> shoulder pitch axis
UPPER_ARM_MM = 660.0        # shoulder axis -> elbow axis
FOREARM_MM = 500.0          # elbow axis -> wrist pitch axis
WRIST_TO_TIP_MM = 164.0     # wrist pitch axis -> gripper tip (jaws closed)

ARM_REACH_MM = UPPER_ARM_MM + FOREARM_MM + WRIST_TO_TIP_MM  # from shoulder

# ---------------------------------------------------------------------------
# Room / gantry envelope (parametric — set to your room at build time).
# ---------------------------------------------------------------------------

ROOM = {
    "x_mm": 3600.0,          # room length covered by X rails
    "y_mm": 3000.0,          # bridge span direction
    "ceiling_mm": 2700.0,    # floor -> ceiling
}

X_TRAVEL_MM = ROOM["x_mm"] - 300.0   # truck end margins
Y_TRAVEL_MM = ROOM["y_mm"] - 400.0   # carriage + endcap margins

PAYLOAD_KG = 1.0            # rated payload at full reach
FLOOR_CLEARANCE_MM = 25.0   # softest z-stop above the floor

# ---------------------------------------------------------------------------
# Stock hardware (not printed)
# ---------------------------------------------------------------------------

TUBE_OD_MM = 25.4           # 1" aluminium round tube, 1.5 mm wall
TUBE_WALL_MM = 1.5
VSLOT_X_RAIL = "2040 V-slot, 2 pcs, length = room x"
VSLOT_BRIDGE = "2060 V-slot, 1 pc, length = room y"
BELT = "GT2, 9 mm wide, steel-core"
PULLEY_TEETH = 20
BELT_PITCH_MM = 2.0
MM_PER_REV = PULLEY_TEETH * BELT_PITCH_MM  # 40 mm of travel per motor rev
WHEEL = "Solid V wheel, 24.39 mm OD, with 625ZZ bearings"
CYCLO_BEARING = "6705ZZ thin-section (25x32x4) on eccentric cam"
OUTPUT_BEARING = "6810ZZ thin-section (50x65x7) on output flange"


# ---------------------------------------------------------------------------
# Motors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Motor:
    name: str
    example_model: str
    holding_torque_nm: float
    mass_kg: float
    body_mm: float            # square faceplate size
    length_mm: float
    closed_loop: bool
    encoder_counts: int       # counts/rev (0 if open loop)


NEMA17 = Motor("NEMA 17 closed-loop", "StepperOnline CL42T + 17E1.8",
               0.59, 0.40, 42.3, 48.0, True, 4000)
NEMA23 = Motor("NEMA 23 closed-loop", "StepperOnline CL57T + 23E2.2 (2.2 Nm)",
               2.20, 1.05, 57.0, 81.0, True, 4000)
NEMA24 = Motor("NEMA 24 closed-loop", "StepperOnline CL57T-V41 + 24E4.0 (4.0 Nm)",
               4.00, 1.70, 60.0, 100.0, True, 4000)
SERVO_DS3225 = Motor("DS3225 servo", "25 kg·cm waterproof digital servo",
                     2.45, 0.06, 40.0, 20.0, False, 0)


# ---------------------------------------------------------------------------
# Arm joints with worst-case static torque analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Joint:
    name: str
    motor: Motor
    ratio: float                  # gearbox reduction
    efficiency: float             # gearbox efficiency
    # (mass_kg, lever_mm) pairs measured from this joint's axis with the
    # distal arm stretched horizontal — the worst gravity case.
    gravity_loads: tuple = ()
    # pure-inertia joints (yaw) size on angular acceleration instead
    inertia_loads: tuple = ()     # (mass_kg, radius_mm)
    design_accel_rad_s2: float = 0.0
    travel_deg: tuple = (-135.0, 135.0)

    def gravity_torque_nm(self) -> float:
        return sum(m * G * (d / 1000.0) for m, d in self.gravity_loads)

    def inertia_torque_nm(self) -> float:
        i = sum(m * (r / 1000.0) ** 2 for m, r in self.inertia_loads)
        return i * self.design_accel_rad_s2

    def required_torque_nm(self) -> float:
        return max(self.gravity_torque_nm(), self.inertia_torque_nm())

    def capacity_nm(self) -> float:
        return self.motor.holding_torque_nm * self.ratio * self.efficiency

    def safety_margin(self) -> float:
        return self.capacity_nm() / self.required_torque_nm()

    def output_resolution_deg(self) -> float:
        counts = self.motor.encoder_counts or 200 * 16  # 1/16 µstep fallback
        return 360.0 / (counts * self.ratio)


# Mass budget (printed parts in PETG-CF + motors + tube), measured from
# each joint axis with everything distal held horizontal.
M_UPPER_TUBE = 0.80      # tube + clamps, CG at mid upper arm
M_ELBOW_ASSY = 1.60      # elbow motor + gearbox + housing at elbow axis
M_FOREARM_TUBE = 0.50
M_WRIST_CLUSTER = 0.90   # wrist pitch+roll motors, housings, near wrist axis
M_GRIPPER = 0.40

SHOULDER = Joint(
    "J2 shoulder pitch", NEMA24, ratio=29, efficiency=0.65,
    gravity_loads=(
        (PAYLOAD_KG, ARM_REACH_MM),                       # payload at tip
        (M_GRIPPER, UPPER_ARM_MM + FOREARM_MM + 80.0),
        (M_WRIST_CLUSTER, UPPER_ARM_MM + FOREARM_MM),
        (M_FOREARM_TUBE, UPPER_ARM_MM + FOREARM_MM / 2),
        (M_ELBOW_ASSY, UPPER_ARM_MM),
        (M_UPPER_TUBE, UPPER_ARM_MM / 2),
    ),
    travel_deg=(-110.0, 110.0),
)

ELBOW = Joint(
    "J3 elbow pitch", NEMA23, ratio=20, efficiency=0.65,
    gravity_loads=(
        (PAYLOAD_KG, FOREARM_MM + WRIST_TO_TIP_MM),
        (M_GRIPPER, FOREARM_MM + 80.0),
        (M_WRIST_CLUSTER, FOREARM_MM),
        (M_FOREARM_TUBE, FOREARM_MM / 2),
    ),
    travel_deg=(-135.0, 135.0),
)

WRIST_PITCH = Joint(
    "J4 wrist pitch", NEMA17, ratio=11, efficiency=0.70,
    gravity_loads=(
        (PAYLOAD_KG, WRIST_TO_TIP_MM),
        (M_GRIPPER, 80.0),
    ),
    travel_deg=(-120.0, 120.0),
)

YAW = Joint(
    "J1 yaw", NEMA23, ratio=15, efficiency=0.65,
    inertia_loads=(
        (PAYLOAD_KG, ARM_REACH_MM),
        (M_GRIPPER, UPPER_ARM_MM + FOREARM_MM + 80.0),
        (M_WRIST_CLUSTER, UPPER_ARM_MM + FOREARM_MM),
        (M_FOREARM_TUBE, UPPER_ARM_MM + FOREARM_MM / 2),
        (M_ELBOW_ASSY, UPPER_ARM_MM),
        (M_UPPER_TUBE, UPPER_ARM_MM / 2),
    ),
    design_accel_rad_s2=2.0,      # 0 -> 90°/s in ~0.8 s, arm horizontal
    travel_deg=(-180.0, 180.0),
)

WRIST_ROLL = Joint(
    "J5 wrist roll", NEMA17, ratio=5, efficiency=0.90,   # GT2 belt stage
    inertia_loads=((PAYLOAD_KG, 60.0), (M_GRIPPER, 30.0)),
    design_accel_rad_s2=20.0,
    travel_deg=(-180.0, 180.0),
)

ARM_JOINTS = (YAW, SHOULDER, ELBOW, WRIST_PITCH, WRIST_ROLL)

MIN_SAFETY_MARGIN = 1.5   # design rule: every joint keeps >=1.5x torque


# ---------------------------------------------------------------------------
# Gantry axes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GantryAxis:
    name: str
    motor: Motor
    motor_count: int
    moving_mass_kg: float
    max_speed_mm_s: float
    max_accel_mm_s2: float
    friction_n: float

    def required_force_n(self) -> float:
        return self.moving_mass_kg * self.max_accel_mm_s2 / 1000.0 + self.friction_n

    def capacity_n(self) -> float:
        pulley_radius_m = MM_PER_REV / (2 * math.pi) / 1000.0
        return self.motor_count * self.motor.holding_torque_nm / pulley_radius_m

    def safety_margin(self) -> float:
        return self.capacity_n() / self.required_force_n()

    def resolution_mm(self) -> float:
        return MM_PER_REV / self.motor.encoder_counts


# Moving-mass roll-up
ARM_MASS_KG = (M_UPPER_TUBE + M_ELBOW_ASSY + M_FOREARM_TUBE + M_WRIST_CLUSTER
               + M_GRIPPER + NEMA24.mass_kg + NEMA23.mass_kg + 2.2)  # +yaw module
CARRIAGE_MASS_KG = 1.5
BRIDGE_MASS_KG = 2.0 * ROOM["y_mm"] / 1000.0 + 1.5 + NEMA23.mass_kg  # beam+trucks+Y motor

Y_AXIS = GantryAxis("Y bridge", NEMA23, 1,
                    moving_mass_kg=ARM_MASS_KG + CARRIAGE_MASS_KG + PAYLOAD_KG,
                    max_speed_mm_s=500.0, max_accel_mm_s2=2000.0, friction_n=25.0)
X_AXIS = GantryAxis("X rails (dual)", NEMA23, 2,
                    moving_mass_kg=Y_AXIS.moving_mass_kg + BRIDGE_MASS_KG,
                    max_speed_mm_s=500.0, max_accel_mm_s2=2000.0, friction_n=40.0)

GANTRY_AXES = (X_AXIS, Y_AXIS)

# Arm joint speed limits used by the motion planner (deg/s at the joint)
JOINT_SPEED_DPS = {"J1 yaw": 90.0, "J2 shoulder pitch": 60.0,
                   "J3 elbow pitch": 90.0, "J4 wrist pitch": 120.0,
                   "J5 wrist roll": 180.0}
JOINT_ACCEL_DPS2 = {"J1 yaw": 180.0, "J2 shoulder pitch": 120.0,
                    "J3 elbow pitch": 180.0, "J4 wrist pitch": 240.0,
                    "J5 wrist roll": 360.0}


# ---------------------------------------------------------------------------
# Printed cycloidal gearbox parameters (three sizes, reused per joint)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CycloSpec:
    """Geometry of one printed cycloidal reducer.

    reduction = pin_count - 1.  Two discs run 180 degrees out of phase to
    cancel the eccentric imbalance.  Ring pins are printed into the
    housing; output pins are printed on the flange and pass through
    oversized holes in the discs.
    """
    name: str
    motor: Motor
    pin_count: int            # ring pins; reduction = pin_count - 1
    pin_circle_r: float       # radius of ring-pin centre circle
    pin_r: float              # ring pin radius
    eccentricity: float
    disc_thickness: float
    output_pin_count: int
    output_pin_r: float
    output_pin_circle_r: float
    housing_wall: float = 6.0

    @property
    def reduction(self) -> int:
        return self.pin_count - 1

    @property
    def housing_od(self) -> float:
        return 2 * (self.pin_circle_r + self.pin_r + self.housing_wall)


CYCLO_BIG = CycloSpec(      # shoulder, 29:1, NEMA 24
    "cyclo-big", NEMA24, pin_count=30, pin_circle_r=62.0, pin_r=3.0,
    eccentricity=1.5, disc_thickness=9.0,
    output_pin_count=6, output_pin_r=5.0, output_pin_circle_r=38.0)

CYCLO_MID = CycloSpec(      # elbow 20:1 / yaw 15:1 share the housing size
    "cyclo-mid", NEMA23, pin_count=21, pin_circle_r=50.0, pin_r=3.0,
    eccentricity=1.5, disc_thickness=8.0,
    output_pin_count=6, output_pin_r=4.0, output_pin_circle_r=30.0)

CYCLO_YAW = CycloSpec(      # yaw 15:1
    "cyclo-yaw", NEMA23, pin_count=16, pin_circle_r=50.0, pin_r=3.5,
    eccentricity=1.8, disc_thickness=8.0,
    output_pin_count=6, output_pin_r=4.0, output_pin_circle_r=30.0)

CYCLO_SMALL = CycloSpec(    # wrist pitch, 11:1, NEMA 17
    "cyclo-small", NEMA17, pin_count=12, pin_circle_r=34.0, pin_r=2.5,
    eccentricity=1.2, disc_thickness=6.0,
    output_pin_count=4, output_pin_r=3.0, output_pin_circle_r=20.0)

JOINT_GEARBOX = {"J1 yaw": CYCLO_YAW, "J2 shoulder pitch": CYCLO_BIG,
                 "J3 elbow pitch": CYCLO_MID, "J4 wrist pitch": CYCLO_SMALL}


# ---------------------------------------------------------------------------
# Print envelope rule (Bambu X1C / Prusa XL class printer)
# ---------------------------------------------------------------------------

PRINT_VOLUME_MM = (250.0, 250.0, 250.0)


# ---------------------------------------------------------------------------
# Tip accuracy roll-up
# ---------------------------------------------------------------------------

def tip_resolution_mm() -> float:
    """Worst single-source tip motion per encoder count, arm fully out."""
    ang = math.radians(SHOULDER.output_resolution_deg())
    arm = ang * ARM_REACH_MM
    return max(arm, X_AXIS.resolution_mm(), Y_AXIS.resolution_mm())


def design_report() -> str:
    lines = ["SkyArm design check", "=" * 55]
    lines.append(f"Total reach from ceiling: {CARRIAGE_STACK_MM + ARM_REACH_MM:.0f} mm"
                 f"  (target {TOTAL_REACH_MM:.0f} mm = 5 ft)")
    lines.append(f"Rated payload at full reach: {PAYLOAD_KG:.1f} kg")
    lines.append("")
    for j in ARM_JOINTS:
        lines.append(
            f"{j.name:18s} {j.motor.name:22s} {j.ratio:>4.0f}:1"
            f"  need {j.required_torque_nm():5.1f} Nm"
            f"  have {j.capacity_nm():5.1f} Nm"
            f"  margin {j.safety_margin():4.2f}x"
            f"  res {j.output_resolution_deg():.4f} deg")
    lines.append("")
    for a in GANTRY_AXES:
        lines.append(
            f"{a.name:18s} {a.motor_count}x {a.motor.name:20s}"
            f"  need {a.required_force_n():5.1f} N"
            f"  have {a.capacity_n():6.1f} N"
            f"  margin {a.safety_margin():5.2f}x"
            f"  res {a.resolution_mm():.3f} mm")
    lines.append("")
    lines.append(f"Worst-case tip resolution: {tip_resolution_mm():.3f} mm")
    return "\n".join(lines)


if __name__ == "__main__":
    print(design_report())
