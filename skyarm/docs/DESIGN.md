# SkyArm design

A ceiling-mounted cartesian gantry carries a 5-DOF, **fully 3D-printed**
arm that drops down to reach anything within 5 ft (1524 mm) of the
ceiling plane, anywhere in the room, rated to hold **15 lb (6.8 kg) at
full horizontal reach** with >=1.5x margin on every load path.

```
 ceiling ─┬──────────────────────────────┬─  2× 2040 X rails (joist-mounted)
          │  truck ═══ bridge 2060 ═══ truck   dual NEMA 23 belt drive
          │              │
          │           carriage  ── Y: NEMA 23 belt drive
          │              │
          │           J1 yaw      NEMA 23 + 15:1 cycloidal
          │              │
          │           J2 shoulder NEMA 24 + 29:1 cycloidal   ┐
          │              │  upper arm: 25.4 mm Al tube, 660  │
          │           J3 elbow    NEMA 23 + 20:1 cycloidal   │ 1524 mm
          │              │  forearm: 25.4 mm Al tube, 500    │ = 5 ft
          │           J4 wrist    NEMA 17 + 11:1 cycloidal   │
          │           J5 roll     NEMA 17 + 5:1 GT2          │
          ▼           gripper     DS3225 servo, geared jaws  ┘
```

## Reach budget (spec.py is the source of truth)

| segment | mm |
|---|---|
| ceiling plane → shoulder axis (carriage + yaw stack) | 215 |
| shoulder → elbow (upper arm) | 645 |
| elbow → wrist (forearm) | 500 |
| wrist → gripper tip | 164 |
| **total** | **1524 = 5 ft** |

The shoulder is also offset 235 mm sideways from the yaw axis (like a
UR arm): the link root sweeps a ~75 mm cylinder around the shoulder
axis, and the yaw module, its bracket, and the shoulder gearbox all
have to live outside that swept zone.  The IK compensates the offset
automatically.

In a 2.7 m room the tip reaches down to ~1.18 m above the floor when
hanging straight down, and reaches *the floor* and beyond when the
target is offset so the arm can lean (the IK leans automatically when a
target lies outside gantry travel).

## Why these choices

**Gantry instead of a longer arm.** A 5 ft arm cantilevered sideways
must fight ~46 Nm of gravity; a gantry positions the *whole* arm above
the work so most moves happen with the arm near vertical, where gravity
torque is near zero. Long horizontal reaches remain possible but are
the exception, not the design point.

**Printed cycloidal reducers on every joint.** Cycloidals share load
across ~a third of their ring pins at once, so a printed disc carries
real torque, and they have effectively zero backlash — the property
that makes the arm *precise*, not just strong. Two discs run 180° out
of phase to cancel the eccentric shake. Reduction = pins − 1.

**Closed-loop steppers.** 4000-count encoders mean missed steps are
corrected, holding torque is rated (not hoped for), and the resolution
math is real:

| joint | drive | output torque (65% eff) | worst load @ 15 lb | margin | resolution |
|---|---|---|---|---|---|
| J1 yaw | NEMA 24 × 18 | 46.8 Nm | 29.3 Nm (inertia) | 1.6× | 0.005° |
| J2 shoulder | NEMA 34 × 35 | 273 Nm | 169 Nm | 1.6× | 0.0026° |
| J3 elbow | NEMA 24 × 40 | 104 Nm | 65 Nm | 1.6× | 0.0022° |
| J4 wrist | NEMA 23 × 15 | 21.4 Nm | 11.6 Nm | 1.8× | 0.006° |
| X / Y | NEMA 23, 20T GT3 | 690 / 345 N | 118 / 80 N | >4× | 0.010 mm |

Worst-case tip resolution (shoulder resolution × full reach): **0.08 mm**.
Design rule enforced by tests: every joint ≥ 1.5× torque margin at
1 kg payload, full horizontal extension.

**Printed box-beam links (no metal tube).** The links are printed
box sections — upper arm 120 x 80 mm with 6 mm walls, forearm
90 x 60 x 5 — split into <=230 mm segments joined by bolted register
flanges (8x M5 each).  A printed box this deep is *stiffer* than the 2"
aluminium tube it replaced (2.7 mm tip sag at 15 lb vs 3.4 mm at 2 lb
for the tube) because stiffness goes with depth cubed.  Segments print
standing; root bending stress is 2.3 MPa against ~31 MPa of derated
across-layer strength.  Beams taper to a narrow neck and stop 40 mm
short of each joint axis so the links fold past each other (measured
fold capacity 115 deg, spec'd at +-105).

**Leadscrew gripper.** A NEMA 17 spins a LH/RH T8 leadscrew pulling
both jaws together: self-locking, so holding 15 lb costs zero motor
current, and ~150 N of grip without gearing.

## Kinematics (`skyarm/kinematics.py`)

FK/IK treat the machine as gantry (x, y) + yaw + a 2-link planar arm in
the vertical plane + wrist. The IK policy places the carriage directly
over the target whenever travel allows — the stiffest, lowest-torque
configuration — and parks at the travel edge and leans the arm for
targets outside it. Both elbow configurations are tried against joint
limits. `psi` selects the approach angle of the gripper (0 = straight
down).

## Safety envelope (`skyarm/sim/world.py`)

- IK targets are rejected if elbow/wrist/tip would cross the floor
  clearance (25 mm) or leave the room.
- Runtime checks fault the machine if any joint crosses the floor.
- Hardware: NC e-stop in the 24 V rail, endstop homing on every axis,
  Klipper's stall detection on the closed-loop drivers.

## Mechanical validation (and what it caught)

Four validation layers, all runnable and all enforced by tests:

1. **Static torque budget** — `python -m skyarm.spec`
2. **Structural analysis** — `python -m skyarm.analysis`: closed-form
   beam bending, tube stress/deflection, printed pin shear/bending,
   bearing loads, V-wheel loads, belt tension, ceiling anchors, each
   with an asserted safety factor.
3. **Rigid-body dynamics** — `python -m skyarm.sim.dynamics`: the
   machine as a URDF in PyBullet with real masses, gravity, and motors
   torque-clamped to gearbox capacity.  Holds 1 kg horizontal with
   zero droop (steady shoulder torque 47.3 / 75.4 Nm, agreeing with
   the static budget within 2%), tracks the demo mission to 0.5 deg /
   1.3 mm, and measures the true payload ceiling: the shoulder
   saturates and the arm collapses between 3.0 and 3.5 kg.
4. **Self-interference sweep** — `python -m skyarm.interference`:
   boolean mesh intersections of the real CAD assembly across joint
   travel.

Design errors these layers caught and the fixes now in the spec/CAD:

| found by | problem | fix |
|---|---|---|
| analysis | 37 mm tip sag with 1" tube | 50.8 mm x 2 mm tube (3.4 mm sag) |
| analysis | output pins SF 1.1 in bending | bigger pins (e.g. 14 mm on the shoulder), SF 3.2 |
| analysis | single output bearing SF 1.04 vs moment | mushroom output flange, journal through 2 spaced bearings, SF 2.4 |
| analysis | carriage wheel SF 1.8 | wheelbase 110 -> 150 mm, SF 2.2 |
| analysis | 2060 bridge deflects several mm | C-beam 4080 bridge (1.3 mm) |
| dynamics | wrist gearbox saturated during moves | 11:1 -> 13:1 |
| interference | tubes collide folding past 105 deg | output clamps offset 30 mm past the joint, travel limits set to measured values (elbow ±105, wrist ±95) |
| analysis | yaw output bearings SF 0.82 — the whole arm's moment hangs from the yaw journal | 2× 6818 wide-thin pair (SF 2.15) |
| interference | no collision-free route from yaw output to shoulder through the link's swing zone | shoulder offset 235 mm from the yaw axis (UR-style), yoke bracket with twin side rails around the gearbox |

## Iterating

1. Edit `skyarm/spec.py` (room size, tube, motors, ratios…).
2. `python -m unittest discover -s tests -p "test_skyarm*"` — torque
   margins, reach, print-volume and IK tests catch bad combinations.
3. `python -m skyarm.cad.generate` — regenerate STLs.
4. `python -m skyarm.sim --headless` or `--snapshot` — re-test missions.
5. `python -m skyarm.sim` — fly it interactively (needs a display).
