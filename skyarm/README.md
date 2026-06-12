# SkyArm — ceiling-gantry robotic arm

A room-scale robot: an X/Y gantry on the ceiling carries a 5-DOF,
fully 3D-printed arm with a 5 ft (1524 mm) reach, rated for a 15 lb
(6.8 kg) payload at full horizontal extension.  Printed box-beam links,
printed cycloidal reducers on every joint, closed-loop steppers, and a
self-locking leadscrew gripper.  Designed, generated, and physics-
validated (PyBullet dynamics + structural analysis + mesh interference
sweeps) entirely from this package.

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

Numbers at a glance: 15 lb payload at full reach with ≥1.5× margin on
every load path (shoulder: NEMA 34 through a printed 35:1 cycloidal =
273 Nm vs 169 Nm worst load; PyBullet holds 27 lb before saturating),
2.7 mm tip sag at rated load, 0.01 mm gantry resolution, ~0.06 mm
worst-case tip resolution.

Change the room, the tube, a motor or a ratio in `spec.py`, rerun the
tests, regenerate the STLs — the design stays consistent because
everything derives from the spec.
