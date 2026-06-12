# SkyArm bill of materials

Printed parts are listed in `skyarm/stl/MANIFEST.md` (regenerate with
`python -m skyarm.cad.generate`). Everything below is purchased.
Prices are rough 2026 USD.

## Motion — motors and drives

Closed-loop steppers everywhere: the encoder (4000 counts/rev) plus the
zero-backlash printed cycloidal reducers is what makes the arm precise
*and* means a stall is detected instead of silently losing position
under load.

| # | item | qty | use | ~$ |
|---|---|---|---|---|
| 1 | NEMA 24 closed-loop stepper 4.0 Nm + driver (e.g. StepperOnline CL57T + 24E4.0) | 1 | J2 shoulder | 95 |
| 2 | NEMA 23 closed-loop stepper 2.2 Nm + driver (CL57T + 23E2.2) | 5 | 2× X gantry, 1× Y gantry, J1 yaw, J3 elbow | 425 |
| 3 | NEMA 17 closed-loop stepper 0.59 Nm + driver (CL42T + 17E) | 2 | J4 wrist pitch, J5 wrist roll | 120 |
| 4 | DS3225 25 kg·cm digital servo | 1 | gripper | 15 |

## Motion — mechanical

| # | item | qty | use | ~$ |
|---|---|---|---|---|
| 5 | 2040 V-slot extrusion, room length (~3.6 m) | 2 | ceiling X rails | 80 |
| 6 | C-beam 4080 V-slot, room width (~3.0 m) | 1 | Y bridge beam (2060 deflected too much — see analysis.py) | 95 |
| 7 | GT2 belt, 9 mm steel-core, 10 m | 1 | X + Y drives | 30 |
| 8 | GT2 pulley 20T / 8 mm bore | 3 | gantry drives | 12 |
| 9 | GT2 idler 9 mm, smooth | 6 | belt returns + tensioners | 12 |
| 10 | Solid V wheel kits (wheel + 625ZZ + spacers + M5) | 14 | 8 on X trucks, 6 on Y carriage | 35 |
| 11 | Aluminium tube 50.8 mm (2") OD × 2 mm wall × 1 m | 2 | upper arm + forearm (1" tube sagged 37 mm — see analysis.py) | 50 |
| 12 | 6705ZZ thin bearing (25×32×4) | 6 | cyclo cam lobes (big/mid/yaw) | 18 |
| 13 | 6803ZZ bearing (17×26×5) | 2 | cyclo-small cam lobes | 6 |
| 14 | 6810ZZ bearing (50×65×7) | 6 | big/mid/yaw output: 2 per joint, spaced pair reacts the bending moment | 42 |
| 15 | 6806ZZ bearing (30×42×7) | 4 | cyclo-small output (2) + wrist roll (2) | 16 |
| 16 | GT2 belt closed loop 200 mm + 16T pulley 5 mm | 1 | wrist roll 5:1 stage | 8 |

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
