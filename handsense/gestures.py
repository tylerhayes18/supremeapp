"""Gesture classification from MediaPipe hand landmarks.

This module is intentionally dependency-free (no cv2/mediapipe imports) so the
classification logic can be unit-tested with synthetic landmarks. A landmark
set is a sequence of 21 (x, y, z) tuples in MediaPipe's normalized image
coordinates: x grows right, y grows DOWN, origin at the top-left of the frame.

Landmark indices (MediaPipe Hands):
    0 wrist
    1-4   thumb  (CMC, MCP, IP, TIP)
    5-8   index  (MCP, PIP, DIP, TIP)
    9-12  middle (MCP, PIP, DIP, TIP)
    13-16 ring   (MCP, PIP, DIP, TIP)
    17-20 pinky  (MCP, PIP, DIP, TIP)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


class Gesture(str, Enum):
    FIST = "fist"
    OPEN_PALM = "open_palm"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    MIDDLE_FINGER = "middle_finger"
    POINTING = "pointing"
    PEACE = "peace"
    THREE = "three"
    FOUR = "four"
    OK_SIGN = "ok_sign"
    ROCK_ON = "rock_on"
    SHAKA = "shaka"
    UNKNOWN = "unknown"


@dataclass
class GestureResult:
    gesture: Gesture
    fingers: tuple[bool, bool, bool, bool, bool]  # thumb..pinky extended
    pinch: float  # thumb-index tip distance, normalized by hand size
    hand_size: float  # wrist -> middle MCP distance (normalized units)

    @property
    def finger_count(self) -> int:
        return sum(self.fingers)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_size(lm) -> float:
    """Characteristic scale of the hand: wrist to middle-finger knuckle."""
    return max(_dist(lm[WRIST], lm[MIDDLE_MCP]), 1e-6)


def finger_extended(lm, tip: int, pip: int) -> bool:
    """A (non-thumb) finger is extended when its tip is meaningfully farther
    from the wrist than its PIP joint. This is orientation-agnostic, so it
    works for upside-down or sideways hands (e.g. thumbs-down)."""
    return _dist(lm[tip], lm[WRIST]) > _dist(lm[pip], lm[WRIST]) * 1.08


def thumb_extended(lm) -> bool:
    """The thumb is extended when its tip is farther from the pinky knuckle
    than its IP joint is — a curled thumb folds across the palm toward it."""
    return _dist(lm[THUMB_TIP], lm[PINKY_MCP]) > _dist(lm[THUMB_IP], lm[PINKY_MCP]) * 1.05


def finger_states(lm) -> tuple[bool, bool, bool, bool, bool]:
    return (
        thumb_extended(lm),
        finger_extended(lm, INDEX_TIP, INDEX_PIP),
        finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP),
        finger_extended(lm, RING_TIP, RING_PIP),
        finger_extended(lm, PINKY_TIP, PINKY_PIP),
    )


def pinch_amount(lm) -> float:
    """Thumb-tip to index-tip distance, normalized by hand size.
    ~0.2 when touching, ~2.0+ when spread wide."""
    return _dist(lm[THUMB_TIP], lm[INDEX_TIP]) / hand_size(lm)


def classify(lm) -> GestureResult:
    """Classify a single hand's landmarks into a discrete gesture."""
    size = hand_size(lm)
    fingers = finger_states(lm)
    thumb, index, middle, ring, pinky = fingers
    pinch = pinch_amount(lm)

    gesture = Gesture.UNKNOWN

    # OK sign first: the curled index defeats the simple extension test, so it
    # must be recognized by the thumb-index circle plus three open fingers.
    if pinch < 0.3 and middle and ring and pinky:
        gesture = Gesture.OK_SIGN
    elif fingers == (False, False, False, False, False):
        gesture = Gesture.FIST
    elif fingers == (True, True, True, True, True):
        gesture = Gesture.OPEN_PALM
    elif thumb and not (index or middle or ring or pinky):
        # Direction of the thumb (image y grows downward) decides up vs down.
        dy = (lm[THUMB_MCP][1] - lm[THUMB_TIP][1]) / size
        if dy > 0.35:
            gesture = Gesture.THUMBS_UP
        elif dy < -0.35:
            gesture = Gesture.THUMBS_DOWN
    elif middle and not (thumb or index or ring or pinky):
        gesture = Gesture.MIDDLE_FINGER
    elif index and not (middle or ring or pinky):
        gesture = Gesture.POINTING  # thumb may be out or tucked
    elif index and middle and not (ring or pinky or thumb):
        gesture = Gesture.PEACE
    elif index and middle and ring and not pinky and not thumb:
        gesture = Gesture.THREE
    elif index and middle and ring and pinky and not thumb:
        gesture = Gesture.FOUR
    elif index and pinky and not (middle or ring):
        gesture = Gesture.ROCK_ON  # thumb out or in both read as horns
    elif thumb and pinky and not (index or middle or ring):
        gesture = Gesture.SHAKA

    return GestureResult(gesture=gesture, fingers=fingers, pinch=pinch, hand_size=size)


class GestureStabilizer:
    """Debounces noisy per-frame classifications.

    A gesture only becomes "stable" after being seen for `hold_frames`
    consecutive frames, and `trigger()` enforces a per-gesture cooldown so a
    held pose fires its command once, not sixty times a second.
    """

    def __init__(self, hold_frames: int = 8, cooldown_s: float = 1.5):
        self.hold_frames = hold_frames
        self.cooldown_s = cooldown_s
        self._current = Gesture.UNKNOWN
        self._streak = 0
        self._stable = Gesture.UNKNOWN
        self._last_fired: dict[Gesture, float] = {}

    def update(self, gesture: Gesture) -> Gesture:
        """Feed one frame's classification; returns the current stable gesture."""
        if gesture == self._current:
            self._streak += 1
        else:
            self._current = gesture
            self._streak = 1
        if self._streak >= self.hold_frames:
            self._stable = self._current
        return self._stable

    @property
    def stable(self) -> Gesture:
        return self._stable

    def trigger(self, now: float) -> Gesture | None:
        """Return the stable gesture if it should fire a command now, else None."""
        g = self._stable
        if g == Gesture.UNKNOWN:
            return None
        last = self._last_fired.get(g, -1e9)
        if now - last < self.cooldown_s:
            return None
        self._last_fired[g] = now
        return g
