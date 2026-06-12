# CLAUDE.md

Guidance for AI assistants working in this repository.

## Project

HandSense — a Python webcam app that tracks hands with MediaPipe, classifies
finger poses into gestures (thumbs up/down, middle finger, peace, OK, etc.),
and maps them to interactive modes (commands, air-drawing, rock-paper-scissors,
a pinch-driven dial). Entry point: `python -m handsense`.

## Layout

```
handsense/
  main.py       capture loop, key handling, mode switching, top HUD bar
  detector.py   MediaPipe Tasks HandLandmarker wrapper -> HandObservation
  gestures.py   pure-Python gesture classification + GestureStabilizer
  modes.py      CommandMode, DrawMode, RPSMode, DialMode (ALL_MODES registry)
  hud.py        drawing helpers: panels, outlined text, banners, EventLog, FPS
tests/
  test_gestures.py  synthetic-landmark tests for classifier + stabilizer
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
pip install -r requirements.txt          # opencv-python, mediapipe, numpy
python -m handsense                      # run (needs a webcam + display)
python -m unittest discover -s tests -v  # tests run anywhere, no camera/deps
```

There is no camera in CI/cloud environments — verify changes via the unit
tests and `python -m py_compile handsense/*.py`. Anything touching the live
loop needs a human with a webcam to confirm.

## Git

- Work on `claude/...` feature branches; push with `git push -u origin <branch>`.
- `snapshots/` and `.venv/` are gitignored output — never commit them.
