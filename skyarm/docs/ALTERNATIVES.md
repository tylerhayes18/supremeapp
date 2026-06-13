# Off-the-shelf open-source arm on the SkyArm gantry

Evaluation of mounting an existing open-source arm on our (validated)
ceiling X/Y gantry instead of the custom SkyArm manipulator.

## Candidates surveyed (June 2026)

| | OpenArm v1 (Enactic) | PAROL6 (Source Robotics) | AR4-MK4 (Annin) |
|---|---|---|---|
| DOF | 7 | 6 | 6 |
| Reach | 633 mm | 400 mm | ~600 mm |
| Payload | **6.0 kg peak / 4.1 kg nominal** | 1 kg near base, 0.5 kg full workspace | ~1–2 kg |
| Arm mass | 5.5 kg | 5.5 kg | ~12 kg |
| Drives | Damiao QDD (CAN-FD, backdrivable, torque control) | steppers + planetary/belt | steppers |
| Structure | aluminium + stainless | 3D-printed PETG | aluminium |
| Openness | CERN-OHL-S v2 hardware + Apache-2.0 software, STEP/STL on GitHub | GPLv3 — STLs, BOM, firmware, GUI | free plans; control software is non-commercial |
| Cost | ~$3.3k/arm (bimanual $6.5k) | €3,570 assembled; kit/DIY-printed cheaper | ~$2k |
| Repeatability | (QDD, research-grade) | 0.1–0.2 mm | ~0.05 mm class |

**Recommendation: OpenArm v1**, with **PAROL6 + SSG-48** as the
budget/printed alternative.  OpenArm is the only one in payload range
(6.0 kg peak = 13.2 lb; nominal 4.1 kg = 9 lb), it is genuinely fully
open (CERN-OHL-S, full STEP), the QDD joints are backdrivable and
torque-controlled (smooth, compliant motion — and safe around people,
which matters for a machine flying around a room), and it looks like a
modern humanoid arm rather than a hobby project.

**Honest caveat:** nothing off the shelf "easily" holds 15 lb — OpenArm
peaks at 13.2 lb and holds 9 lb sustained.  The custom SkyArm
(273 Nm shoulder) remains the only 15 lb+ option in this repo.

## Gripper

The **SSG-48 adaptive electric gripper** (Source Robotics, open
source): BLDC-driven with current sensing and closed-loop force control
from 5–80 N, parallel jaws, force feedback, and a Python API that is
arm-agnostic.  80 N of controlled grip with TPU-padded jaws holds
several kg by friction without crushing delicate objects.  OpenArm's
stock parallel gripper (with in-hand camera) is the teleop/ML option;
the Yale OpenHand designs are the 3-finger adaptive alternative for
irregular objects.

## Integration architecture

Keep everything the gantry analysis already validated; replace the arm
stack below the carriage:

```
ceiling rails / trucks / C-beam bridge / MGN15 carriage   (unchanged)
        |
   Z drop stage  — C-beam 4080, 1500 mm travel, ballscrew + NEMA 23
        |          (recovers the floor reach a 633 mm arm cannot give)
   adapter plate — printed, Z-gantry-plate pattern -> arm base M6 grid
        |
   OpenArm v1 (inverted) + SSG-48 gripper
```

- **Reach budget:** arm base travels ceiling−300 … ceiling−1800 mm;
  with 633 mm of arm the tip covers floor+~270 mm everywhere and the
  full room volume above it.  (Full floor touch needs 1850 mm of Z
  travel — a 2 m C-beam fits a 2.7 m room.)
- **Loads:** arm+payload+Z-carriage ≈ 13 kg, arm-horizontal moment
  ≈ 71 Nm at the carriage — inside the 169 Nm envelope the MGN15
  carriage, bridge, and trucks were already validated against.
- **Electronics:** gantry stays on Klipper (closed-loop steppers);
  OpenArm runs its own CAN-FD bus with a Python/ROS2 SDK; PAROL6 ships
  its own open controller.  The mission planner (skyarm/sim/world.py)
  becomes the top-level coordinator commanding both.
- **Validation carries over:** OpenArm publishes a URDF — it can be
  loaded into our PyBullet harness (sim/dynamics.py) hanging inverted
  from the gantry prismatic joints to re-run the hold/mission/payload
  sweeps before committing to hardware.
- The printed part `openarm_adapter` (cad/parts.py) is the mechanical
  bridge; verify the M6 grid spacing against the published STEP before
  printing (constant `OPENARM_BASE_GRID_MM`).

## Sources

- https://github.com/enactic/openarm / https://github.com/enactic/openarm_hardware
- https://docs.openarm.dev/
- https://github.com/PCrnjak/PAROL6-Desktop-robot-arm
- https://source-robotics.github.io/PAROL-docs/page2_2/
- https://source-robotics.com/products/parol6-robotic-arm
- https://github.com/PCrnjak/SSG-48-adaptive-electric-gripper
- https://anninrobotics.com/ / https://robodk.com/robot/Annin-Robotics/AR4
