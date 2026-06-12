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

## 3. The arm (fully printed links)

1. Bolt the **yaw** gearbox housing to the carriage underside (6× M5);
   its output carries the shoulder column.
2. Bolt the **shoulder** (big, NEMA 34) gearbox into the column band;
   `upper_root` bolts to its output journal (6× M5).
3. Stack `upper_root` + 2× `upper_mid` + `upper_tip` with the bolted
   register flanges (8× M5 each, thread-locked).  The `upper_tip`
   motor plate carries the **elbow** (NEMA 24, 40:1) gearbox.
4. `forearm_root` bolts to the elbow output; add `forearm_mid` and
   `forearm_tip`, which carries the **wrist** (NEMA 23, 15:1) gearbox.
5. `roll_housing` bolts to the wrist output with the roll NEMA 23 and
   5:1 GT3 belt to `roll_pulley`.
6. `gripper_body` bolts to the roll pulley flange.  Slide both jaws
   onto the printed rails, thread the LH/RH T8 leadscrew through both
   nut pockets, couple the NEMA 17.  The screw is self-locking — the
   grip holds 15 lb with the motor unpowered.  Glue TPU pads on.
7. Route wiring inside the hollow beams (that's what the box section
   is for), into the drag chain at the carriage, across the bridge,
   and along one X rail to the controller box.

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
