"""Unit tests for gesture classification using synthetic landmark sets.

The synthetic hand is an upright right hand facing the camera: wrist near the
bottom of the frame, knuckles in a row above it, thumb on the left side.
Coordinates follow MediaPipe conventions (normalized, y grows downward).
"""

import unittest

from handsense.gestures import Gesture, GestureStabilizer, classify

WRIST = (0.50, 0.85, 0.0)
THUMB_CMC = (0.44, 0.78, 0.0)
THUMB_MCP = (0.40, 0.72, 0.0)

# Finger knuckle columns: (index, middle, ring, pinky)
MCP_X = (0.42, 0.48, 0.54, 0.60)
MCP_Y = 0.60


def _finger(x: float, extended: bool):
    """(PIP, DIP, TIP) joints for one non-thumb finger."""
    if extended:
        return [(x, 0.50, 0.0), (x, 0.45, 0.0), (x, 0.38, 0.0)]
    return [(x, 0.55, 0.0), (x, 0.62, 0.0), (x, 0.68, 0.0)]


def _thumb(state: str):
    """(IP, TIP) for the thumb: 'curled', 'side' (extended), or 'up'."""
    if state == "curled":
        return [(0.45, 0.70, 0.0), (0.50, 0.66, 0.0)]
    if state == "side":
        return [(0.36, 0.68, 0.0), (0.28, 0.68, 0.0)]
    if state == "up":
        return [(0.38, 0.62, 0.0), (0.36, 0.50, 0.0)]
    raise ValueError(state)


def make_hand(thumb="curled", index=False, middle=False, ring=False, pinky=False):
    lm = [WRIST, THUMB_CMC, THUMB_MCP, *_thumb(thumb)]
    for x, ext in zip(MCP_X, (index, middle, ring, pinky)):
        lm.append((x, MCP_Y, 0.0))
        lm.extend(_finger(x, ext))
    assert len(lm) == 21
    return lm


def flip_vertical(lm):
    """Mirror a hand top-to-bottom (turns thumbs-up into thumbs-down)."""
    return [(x, 1.2 - y, z) for x, y, z in lm]


class TestClassify(unittest.TestCase):
    def test_fist(self):
        self.assertEqual(classify(make_hand()).gesture, Gesture.FIST)

    def test_open_palm(self):
        lm = make_hand(thumb="side", index=True, middle=True, ring=True, pinky=True)
        self.assertEqual(classify(lm).gesture, Gesture.OPEN_PALM)

    def test_thumbs_up(self):
        self.assertEqual(classify(make_hand(thumb="up")).gesture, Gesture.THUMBS_UP)

    def test_thumbs_down(self):
        lm = flip_vertical(make_hand(thumb="up"))
        self.assertEqual(classify(lm).gesture, Gesture.THUMBS_DOWN)

    def test_middle_finger(self):
        self.assertEqual(classify(make_hand(middle=True)).gesture, Gesture.MIDDLE_FINGER)

    def test_pointing(self):
        self.assertEqual(classify(make_hand(index=True)).gesture, Gesture.POINTING)

    def test_peace(self):
        lm = make_hand(index=True, middle=True)
        self.assertEqual(classify(lm).gesture, Gesture.PEACE)

    def test_three(self):
        lm = make_hand(index=True, middle=True, ring=True)
        self.assertEqual(classify(lm).gesture, Gesture.THREE)

    def test_four(self):
        lm = make_hand(index=True, middle=True, ring=True, pinky=True)
        self.assertEqual(classify(lm).gesture, Gesture.FOUR)

    def test_rock_on(self):
        lm = make_hand(index=True, pinky=True)
        self.assertEqual(classify(lm).gesture, Gesture.ROCK_ON)

    def test_shaka(self):
        lm = make_hand(thumb="side", pinky=True)
        self.assertEqual(classify(lm).gesture, Gesture.SHAKA)

    def test_ok_sign(self):
        # Index curls down to meet the thumb tip; other three fingers extended.
        lm = make_hand(middle=True, ring=True, pinky=True)
        lm[3], lm[4] = (0.42, 0.64, 0.0), (0.435, 0.605, 0.0)   # thumb IP, TIP
        lm[6], lm[7], lm[8] = (0.42, 0.52, 0.0), (0.43, 0.56, 0.0), (0.43, 0.60, 0.0)
        result = classify(lm)
        self.assertEqual(result.gesture, Gesture.OK_SIGN)
        self.assertLess(result.pinch, 0.3)

    def test_finger_count(self):
        lm = make_hand(thumb="side", index=True, middle=True, ring=True, pinky=True)
        self.assertEqual(classify(lm).finger_count, 5)
        self.assertEqual(classify(make_hand()).finger_count, 0)


class TestStabilizer(unittest.TestCase):
    def test_requires_consecutive_frames(self):
        stab = GestureStabilizer(hold_frames=3, cooldown_s=1.0)
        stab.update(Gesture.PEACE)
        stab.update(Gesture.PEACE)
        self.assertEqual(stab.stable, Gesture.UNKNOWN)
        stab.update(Gesture.PEACE)
        self.assertEqual(stab.stable, Gesture.PEACE)

    def test_flicker_resets_streak(self):
        stab = GestureStabilizer(hold_frames=3, cooldown_s=1.0)
        stab.update(Gesture.PEACE)
        stab.update(Gesture.FIST)
        stab.update(Gesture.PEACE)
        stab.update(Gesture.PEACE)
        self.assertEqual(stab.stable, Gesture.UNKNOWN)

    def test_cooldown(self):
        stab = GestureStabilizer(hold_frames=1, cooldown_s=1.0)
        stab.update(Gesture.THUMBS_UP)
        self.assertEqual(stab.trigger(now=10.0), Gesture.THUMBS_UP)
        self.assertIsNone(stab.trigger(now=10.5))  # still cooling down
        self.assertEqual(stab.trigger(now=11.1), Gesture.THUMBS_UP)

    def test_unknown_never_fires(self):
        stab = GestureStabilizer(hold_frames=1, cooldown_s=0.0)
        stab.update(Gesture.UNKNOWN)
        self.assertIsNone(stab.trigger(now=0.0))


if __name__ == "__main__":
    unittest.main()
