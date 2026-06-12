# CLAUDE.md

Guidance for AI assistants working in this repository.

## Project

HandSense — a Python webcam app that tracks hands and full body with
MediaPipe, classifies finger poses into gestures (thumbs up/down, middle
finger, peace, OK, etc.), and maps them to interactive modes (commands,
air-drawing, rock-paper-scissors, a pinch-driven dial). A separate body
tracking mode mirrors your full-body pose onto a 3D humanoid avatar
rendered in OpenGL.

Entry points: `python -m handsense` (hand gestures), `python -m handsense.body`
(full-body 3D avatar).

## Layout

```
handsense/
  main.py               capture loop, key handling, mode switching, top HUD bar
  detector.py            MediaPipe Tasks HandLandmarker wrapper -> HandObservation
  gestures.py            pure-Python gesture classification + GestureStabilizer
  modes.py               CommandMode, DrawMode, RPSMode, DialMode (ALL_MODES registry)
  hud.py                 drawing helpers: panels, outlined text, banners, EventLog, FPS
  holistic_detector.py   MediaPipe HolisticLandmarker -> body(33)+face(478)+hands(21×2)
  pose_detector.py       MediaPipe PoseLandmarker wrapper (body-only, legacy)
  mesh_loader.py         GLB/VRM model loader, body-part segmentation, OpenGL VBOs
  avatar.py              OpenGL 3D avatar renderer (mesh or geometric, face, fingers)
  body.py                body tracking entry point (camera + 3D avatar)
tests/
  test_gestures.py       synthetic-landmark tests for classifier + stabilizer
  test_pose.py           coordinate conversion, body structure, face orientation tests
```

## Key conventions

- **`gestures.py` must stay dependency-free** (no cv2/mediapipe imports) so
  classification logic is unit-testable with synthetic landmarks. Camera and
  ML deps live only in `detector.py`, `hud.py`, `modes.py`, `main.py`.
- Landmarks are MediaPipe-normalized: 21 `(x, y, z)` tuples, y grows
  **downward**. Index constants (WRIST, THUMB_TIP, ...) are defined in
  `gestures.py` — use them, never bare integers.
- Classification is orientation-agnostic: compare joint distances normalized
  by `hand_size()`, not absolute screen coordinates, so rotated hands work.
- New gestures: add to the `Gesture` enum, extend `classify()` (mind the
  ordering — OK sign is checked first because its curled index defeats the
  extension test), add a label in `hud.GESTURE_EMOJI`, and add a synthetic
  test in `tests/test_gestures.py` using `make_hand()`.
- New modes: implement `name`, `help`, and `update(frame, hands, now)`;
  register in `ALL_MODES`. Use `GestureStabilizer` for any gesture-triggered
  action — never act on a single frame's classification.
- Frames are BGR (OpenCV) and mutated in place by modes; colors in `hud.py`
  are BGR tuples.
- `detector.py` uses the MediaPipe **Tasks** API (`vision.HandLandmarker`,
  VIDEO running mode with strictly increasing timestamps). Do not reintroduce
  `mp.solutions.*` — it no longer exists in current MediaPipe releases. The
  `.task` model auto-downloads to `~/.cache/handsense/` on first run.

## Development

```bash
pip install -r requirements.txt          # opencv-python, mediapipe, numpy, pygame, PyOpenGL
python -m handsense                      # hand gesture modes (needs webcam + display)
python -m handsense.body                 # full-body 3D avatar (needs webcam + display + OpenGL)
python -m unittest discover -s tests -v  # tests run anywhere, no camera/deps
```

There is no camera in CI/cloud environments — verify changes via the unit
tests and `python -m py_compile handsense/*.py`. Anything touching the live
loop needs a human with a webcam to confirm.

## SkyArm (second project in this repo)

`skyarm/` is a ceiling-gantry robotic arm design: parametric CAD that
generates printable STLs, FK/IK, and a 3D simulation test environment.
All dimensions/motors/ratios live in `skyarm/spec.py` — change it, run
`python -m unittest discover -s tests -p "test_skyarm*"`, then
`python -m skyarm.cad.generate` to regenerate `skyarm/stl/`. CAD needs
`trimesh manifold3d shapely numpy`; the kinematics/sim core is
stdlib-only and its tests run anywhere. See `skyarm/README.md`.

## Git

- Work on `claude/...` feature branches; push with `git push -u origin <branch>`.
- `snapshots/` and `.venv/` are gitignored output — never commit them.
