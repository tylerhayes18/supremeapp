# SkyArm assembly guide

Build order is bottom-up per subsystem, then top-down for install.
Print everything per `stl/MANIFEST.md` first; you'll want all four
gearboxes bench-tested before anything goes on the ceiling.

## 1. Cycloidal gearboxes (×4: big, mid, yaw, small)

Per gearbox: `*_housing`, 2× `*_disc`, `*_cam`, `*_output`, `*_cover`.

1. Press the two cam bearings onto the cam lobes (6705ZZ for
   big/mid/yaw, 6803ZZ for small).
2. Slide the cam onto the motor shaft, flat to the grub screw; bolt the
   motor to the housing flange.
3. Drop disc 1 over the lower lobe with its lobes meshing the ring
   pins; disc 2 over the upper lobe — rotate it half a lobe so the two
   discs sit 180° out of phase.
4. Feed the output flange pins through the disc holes (the wide head
   stays inside the ring).  Press BOTH output bearings into the cover —
   one from each face; the spacer ledge between them is printed in —
   then slide the cover over the journal and bolt it to the housing
   (6× M4).  The spaced bearing pair is what carries the joint's
   bending moment; do not substitute a single bearing.
5. Bench test: drive the motor; output must turn smoothly 1/reduction
   per motor turn with no perceptible backlash. A drop of PTFE-safe
   grease on pins and lobes.

## 2. Ceiling rails and gantry

1. Find joists; lag-screw `ceiling_bracket` every ~45 cm along both
   rail lines (rails parallel, spacing = bridge length minus endcaps).
2. Hang the two 2040 X rails in the brackets (M5 + T-nuts).
3. Assemble both `x_truck` plates with 4 V-wheels each; adjust
   eccentric spacers until they roll without slop.
4. Bolt a `bridge_endcap` to each truck, then the 2060 bridge beam
   between them.
5. Assemble `y_carriage` with 6 V-wheels on the bridge.
6. Mount the three `gantry_motor_mount` + NEMA 23s at the rail ends;
   run GT2 belts rail-end → truck clamp → idler → back, and through the
   carriage clamps on the bridge. Tension with `belt_tensioner` blocks
   until a plucked belt gives a low musical note (~30–50 Hz).
   The two X motors are wired as one axis in Klipper (`stepper_x` +
   `stepper_x1`).

## 3. The arm

1. Bolt the **yaw** gearbox housing to the carriage underside (6× M5);
   its output flange carries the shoulder bracket.
2. Bolt the **shoulder** (big) gearbox to the yaw output via
   `shoulder_clevis` pair; clamp the 660 mm upper-arm tube.
3. `elbow_clevis` clamps the other end of the tube and carries the
   **elbow** (mid) gearbox; `forearm_clevis` on its output clamps the
   500 mm forearm tube.
4. `wrist_bracket` clamps the forearm end and carries the **small**
   gearbox; `roll_housing` bolts to its output with the roll NEMA 17
   and 5:1 GT2 belt to `roll_pulley`.
5. `gripper_base` bolts to the roll pulley flange; servo into its
   pocket, fingers on M5 pivots with their gear segments meshed, servo
   horn linked to the driven finger. Glue TPU pads to the finger faces.
6. Route wiring along the tubes (spiral wrap), into the drag chain at
   the carriage, across the bridge, and along one X rail to the
   controller box.

## 4. Electronics + software

- Octopus Pro: X, X1, Y on three drivers; J1–J5 on five more; servo on
  a PWM pin. 24 V rail through the e-stop mushroom.
- Klipper config: `rotation_distance = 40` for X/Y (20T GT2);
  joints are `rotation_distance = 360 / ratio` degrees-as-distance.
- Home X/Y/yaw to endstops, shoulder/elbow/wrist to their hall marks,
  then run the same mission scripts the simulator uses — the
  `skyarm.sim.world` mission format is deliberately g-code-shaped so
  the planner can emit it.

## 5. First-power checklist

1. E-stop within arm's reach of *you*, tested.
2. Joints homed with the arm hanging straight down, nothing below.
3. Run the simulator demo (`python -m skyarm.sim --headless`) and
   confirm the same waypoints in air with speed limits at 25%.
4. Only then load the gripper.
