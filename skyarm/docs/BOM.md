# SkyArm bill of materials

Printed parts are listed in `skyarm/stl/MANIFEST.md` (regenerate with
`python -m skyarm.cad.generate`). Everything below is purchased.
Prices are rough 2026 USD.

## Motion — motors and drives

Closed-loop steppers everywhere: the encoder (4000 counts/rev) plus the
zero-backlash printed cycloidal reducers is what makes the arm precise
*and* means a stall is detected instead of silently losing position
under load.  Sized for **15 lb (6.8 kg) payload at full horizontal
reach** with >=1.5x margin (validated in PyBullet — the arm holds 27 lb
before the shoulder saturates).

| # | item | qty | use | ~$ |
|---|---|---|---|---|
| 1 | NEMA 34 closed-loop stepper 12 Nm + driver (e.g. StepperOnline CL86T + 34E12.0) | 1 | J2 shoulder (273 Nm through 35:1) | 160 |
| 2 | NEMA 24 closed-loop stepper 4.0 Nm + driver (CL57T + 24E4.0) | 2 | J1 yaw, J3 elbow | 190 |
| 3 | NEMA 23 closed-loop stepper 2.2 Nm + driver (CL57T + 23E2.2) | 5 | 2× X, 1× Y gantry, J4 wrist, J5 roll | 425 |
| 4 | NEMA 17 stepper + driver | 1 | leadscrew gripper (self-locking) | 25 |
| 5 | T8 leadscrew, LH/RH combo, 150 mm + 2 nuts | 1 | gripper jaws | 12 |

## Motion — mechanical

| # | item | qty | use | ~$ |
|---|---|---|---|---|
| 6 | 2040 V-slot extrusion, room length (~3.6 m) | 2 | ceiling X rails | 80 |
| 7 | C-beam 4080 V-slot, room width (~3.0 m) | 1 | Y bridge beam (2060 deflected too much — see analysis.py) | 95 |
| 8 | GT3 belt, 15 mm steel-core, 10 m | 1 | X + Y drives (9 mm GT2 failed the 15 lb margin) | 45 |
| 9 | GT3 pulley 20T / 8 mm bore | 3 | gantry drives | 15 |
| 10 | GT3 idler 15 mm, smooth | 6 | belt returns + tensioners | 15 |
| 11 | Xtreme solid V wheel kits (620 N rated) | 8 | X trucks (regular wheels failed the moment check) | 30 |
| 12 | MGN15 rail 400 mm + 2× MGN15H blocks | 1 | Y carriage (V wheels failed the 15 lb overturning moment) | 45 |
| 13 | 6905ZZ bearing (25×42×9) | 2 | cyclo-big cam lobes | 8 |
| 14 | 6705ZZ thin bearing (25×32×4) | 6 | mid/yaw/small cam lobes | 18 |
| 15 | 6815ZZ bearing (75×95×10) | 2 | shoulder output, spaced pair reacts 169 Nm | 24 |
| 16 | 6810ZZ bearing (50×65×7) | 4 | mid/yaw output pairs | 28 |
| 17 | 6806ZZ bearing (30×42×7) | 4 | wrist output pair + roll | 16 |
| 18 | GT3 belt closed loop 300 mm + 16T pulley 8 mm | 1 | wrist roll 5:1 stage | 10 |

## Electronics

| # | item | qty | use | ~$ |
|---|---|---|---|---|
| 17 | BIGTREETECH Octopus Pro (or any 8-driver 32-bit board running Klipper) | 1 | motion controller | 60 |
| 18 | Raspberry Pi 5 | 1 | Klipper host, mission planner, camera | 80 |
| 19 | Mean Well LRS-350-24 (24 V 14.6 A) | 1 | motor power | 40 |
| 20 | Mean Well LRS-50-5 | 1 | logic/servo power | 15 |
| 21 | Emergency-stop mushroom switch, NC | 1 | cuts 24 V rail — **mandatory** | 10 |
| 22 | Inductive/mechanical endstops | 4 | X×2, Y, yaw homing | 10 |
| 23 | Hall endstop or index mark | 3 | shoulder/elbow/wrist homing | 8 |
| 24 | Cable drag chain 15×30 mm, 4 m + 18 AWG silicone wire | 1 | X + Y runs | 35 |
| 25 | Pi Camera v3 wide | 1 | optional: watch the workspace | 35 |
| 26 | Fuse blocks, ferrules, XT60s, heat-shrink | — | wiring | 25 |

**Total ≈ $1,290** (motors dominate — they are the "beefy and extremely
precise" line items: the shoulder alone delivers ~75 Nm through its 29:1
reducer with 0.003° output resolution).

## Fasteners (hardware-store bag)

- M5×10/16/20 + T-nuts for all V-slot joints (~80)
- M5×35 + nyloc as wheel axles (14) and finger pivots (2)
- M4×16/25 for tube clamps and covers (~40), M4 threaded inserts (~20)
- M3×8/12 for NEMA 17 faces, servo, grub screws (~30)
- 6.5 mm lag screws ×2 per ceiling bracket — **into joists, not drywall**;
  each bracket pair must hold ~25 kg working load with 4× margin.

## Structural note on the ceiling

Total hanging mass is ≈ 18 kg plus dynamic loads. Mount every ceiling
bracket into a joist with two lag screws, one bracket every ~45 cm per
rail. If your joists run the wrong way, add two 2040 cross-rails first.
