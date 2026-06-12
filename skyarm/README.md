# SkyArm — ceiling-gantry robotic arm

A room-scale robot: an X/Y gantry on the ceiling carries a 5-DOF arm
with a 5 ft (1524 mm) reach, printed cycloidal reducers on every joint,
closed-loop steppers, and a servo gripper. Designed, generated and
tested entirely from this package.

| | |
|---|---|
| `spec.py` | every dimension/motor/ratio + torque & resolution math |
| `kinematics.py` | FK/IK for gantry + arm (pure stdlib) |
| `cad/` | parametric part generators → 35 printable STLs |
| `sim/` | 3D test environment: headless world + OpenGL viewer |
| `render.py` | headless preview renderer (no GPU needed) |
| `stl/` | generated parts + print manifest |
| `previews/` | rendered images of parts and missions |
| `docs/` | DESIGN.md, BOM.md, ASSEMBLY.md |

## Quick start

```bash
pip install numpy trimesh manifold3d shapely pillow pygame PyOpenGL pybullet

python -m skyarm.spec                 # torque/resolution design report
python -m skyarm.analysis             # structural checks w/ safety factors
python -m skyarm.sim.dynamics         # PyBullet torque-limited physics tests
python -m skyarm.interference         # self-collision sweep (real meshes)
python -m skyarm.cad.generate         # regenerate all STLs + manifest
python -m skyarm.sim                  # interactive 3D test environment
python -m skyarm.sim --headless       # run the demo mission, print report
python -m skyarm.sim --snapshot f.png # render mission keyframes (--real for CAD meshes)
python -m skyarm.assembly             # assembled + exploded machine renders
python -m unittest discover -s tests -p "test_skyarm*" -v
```

Numbers at a glance: 1 kg payload at full reach, ≥1.5× torque margin on
every joint (shoulder: NEMA 24 through a printed 29:1 cycloidal = 75 Nm),
0.01 mm gantry resolution, ~0.08 mm worst-case tip resolution.

Change the room, the tube, a motor or a ratio in `spec.py`, rerun the
tests, regenerate the STLs — the design stays consistent because
everything derives from the spec.
