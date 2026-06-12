# HandSense

Live hand-gesture recognition and full-body motion tracking from your webcam.
HandSense tracks 21 hand landmarks and 33 body landmarks in real time
(MediaPipe), classifies finger poses into gestures, maps them to interactive
modes — and can mirror your entire body onto a 3D humanoid avatar rendered
in OpenGL.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python -m handsense
```

A window opens showing your camera (mirrored, like a selfie). Hold a gesture
steady for a fraction of a second to trigger it. On first run the hand
landmark model (~8 MB) is downloaded once to `~/.cache/handsense/`.

If the camera doesn't open, try `--camera 1` (external webcams often aren't
index 0) and check your OS camera permissions.

## Recognized gestures

| Gesture | Hand sign |
|---|---|
| `thumbs_up` / `thumbs_down` | fist with thumb up / down |
| `middle_finger` | you know the one |
| `pointing` | index finger only |
| `peace` | index + middle |
| `three` / `four` | finger counting |
| `fist` / `open_palm` | all fingers curled / extended |
| `ok_sign` | thumb-index circle, other fingers up |
| `rock_on` | index + pinky horns |
| `shaka` | thumb + pinky, hang loose |

Pinch distance (thumb tip to index tip) is also tracked continuously as an
analog input.

## Modes

Switch with keys `1`–`4` or cycle with `m`.

1. **COMMAND** — gestures fire one-shot actions into an on-screen event log.
   Thumbs up/down keep a like score, the middle finger gets auto-pixelated
   and racks up a rudeness counter, *peace* starts a 3-2-1 photo countdown
   (saved to `snapshots/`), and *open palm* pauses/resumes processing.
2. **DRAW** — air-drawing. Point to draw with your index fingertip, pinch to
   change brush size, *peace* cycles colors, hold an open palm for one second
   to clear the canvas.
3. **RPS** — rock-paper-scissors vs the computer. Thumbs up starts a round;
   after the countdown show fist (rock), open palm (paper), or peace
   (scissors). First to 3 takes the match.
4. **DIAL** — a 270° virtual dial driven by your pinch distance, smoothed
   with an EMA. A template for wiring gestures to anything continuous
   (volume, brightness, zoom...).

## Full-body 3D avatar

A separate mode tracks your full body and mirrors your pose onto a lit,
geometric 3D humanoid in an OpenGL window:

```bash
python -m handsense.body
```

The avatar is built from spheres (joints) and cylinders (limbs) with
per-body-part coloring and OpenGL lighting. Your camera feed appears as a
picture-in-picture in the corner.

| Key | Action |
|---|---|
| arrow keys | orbit the 3D camera around the avatar |
| `+` / `-` | zoom in / out |
| `q` / `Esc` | quit |

```
python -m handsense.body [--camera N] [--width W] [--height H]
                         [--no-mirror] [--avatar-size N]
```

## Keys (hand gesture modes)

| Key | Action |
|---|---|
| `1`–`4` | jump to mode |
| `m` | next mode |
| `k` | toggle hand-skeleton overlay |
| `f` | toggle mirror flip |
| `q` / `Esc` | quit |

## CLI options

```
python -m handsense [--camera N] [--width W] [--height H]
                    [--max-hands N] [--mode 1-4] [--no-mirror] [--no-skeleton]
```

## Tests

The gesture classifier and debouncer are pure Python and tested with
synthetic landmarks — no camera or MediaPipe install needed:

```bash
python -m unittest discover -s tests -v
```

## How it works

- `handsense/detector.py` wraps the MediaPipe Tasks `HandLandmarker` (the
  modern API — legacy `mp.solutions` was removed from recent MediaPipe
  releases) and emits 21 normalized landmarks plus handedness per hand. It
  auto-downloads the model on first use.
- `handsense/gestures.py` classifies poses with orientation-agnostic
  geometry (joint distances relative to hand size, not screen axes), so
  sideways and upside-down hands still read correctly. A `GestureStabilizer`
  debounces classifications across frames and enforces per-gesture cooldowns
  so a held pose fires once.
- `handsense/modes.py` implements the four interactive modes; each consumes
  per-frame `(observation, gesture)` pairs and draws its own UI.
- `handsense/main.py` runs the capture loop, mode switching, and HUD.
- `handsense/pose_detector.py` wraps MediaPipe's `PoseLandmarker` for
  full-body tracking (33 landmarks in both image and world coordinates).
- `handsense/avatar.py` renders a geometric 3D humanoid in a pygame/OpenGL
  window, positioned from the world-coordinate landmarks. Supports orbiting
  the camera, zoom, and a camera-feed picture-in-picture.
- `handsense/body.py` ties the pose detector and avatar together into a
  real-time body-mirroring loop.
