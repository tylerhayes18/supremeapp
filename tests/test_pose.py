"""Tests for pose/holistic coordinate conversion, body structure, and face orientation."""

import math
import unittest

import numpy as np


class TestMediaPipeToGL(unittest.TestCase):
    def test_coordinate_conversion(self):
        from handsense.avatar import _mp_to_gl
        gx, gy, gz = _mp_to_gl(1.0, 2.0, 3.0)
        self.assertAlmostEqual(gx, 1.0)
        self.assertAlmostEqual(gy, -2.0)
        self.assertAlmostEqual(gz, -3.0)

    def test_origin_unchanged(self):
        from handsense.avatar import _mp_to_gl
        self.assertEqual(_mp_to_gl(0, 0, 0), (0, 0, 0))


class TestVectorMath(unittest.TestCase):
    def test_vec_len(self):
        from handsense.avatar import _vec_len
        self.assertAlmostEqual(_vec_len((3, 4, 0)), 5.0)
        self.assertAlmostEqual(_vec_len((0, 0, 0)), 0.0)

    def test_vec_normalize(self):
        from handsense.avatar import _vec_normalize, _vec_len
        n = _vec_normalize((3, 0, 0))
        self.assertAlmostEqual(n[0], 1.0)
        self.assertAlmostEqual(_vec_len(n), 1.0)

    def test_vec_normalize_zero(self):
        from handsense.avatar import _vec_normalize
        n = _vec_normalize((0, 0, 0))
        self.assertEqual(n, (0, 0, 1))

    def test_vec_cross(self):
        from handsense.avatar import _vec_cross
        c = _vec_cross((1, 0, 0), (0, 1, 0))
        self.assertAlmostEqual(c[0], 0)
        self.assertAlmostEqual(c[1], 0)
        self.assertAlmostEqual(c[2], 1)


class TestBodyDefinitions(unittest.TestCase):
    def test_all_bone_indices_valid(self):
        from handsense.avatar import BODY_BONES
        for start, end, r1, r2, color in BODY_BONES:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start, 32)
            self.assertGreaterEqual(end, 0)
            self.assertLessEqual(end, 32)
            self.assertGreater(r1, 0)
            self.assertGreater(r2, 0)
            self.assertEqual(len(color), 4)

    def test_all_joint_indices_valid(self):
        from handsense.avatar import BODY_JOINTS
        for idx, radius, color in BODY_JOINTS:
            self.assertGreaterEqual(idx, 0)
            self.assertLessEqual(idx, 32)
            self.assertGreater(radius, 0)
            self.assertEqual(len(color), 4)

    def test_pose_connections_valid(self):
        from handsense.holistic_detector import POSE_CONNECTIONS
        for a, b in POSE_CONNECTIONS:
            self.assertGreaterEqual(a, 0)
            self.assertLessEqual(a, 32)
            self.assertGreaterEqual(b, 0)
            self.assertLessEqual(b, 32)

    def test_hand_finger_indices_valid(self):
        from handsense.holistic_detector import HAND_FINGERS
        self.assertEqual(len(HAND_FINGERS), 5)
        for finger in HAND_FINGERS:
            self.assertEqual(len(finger), 4)
            for idx in finger:
                self.assertGreaterEqual(idx, 0)
                self.assertLessEqual(idx, 20)


class TestFaceOrientation(unittest.TestCase):
    def _make_face_landmarks(self, yaw_shift=0.0, pitch_shift=0.0, roll_shift=0.0):
        """Create a synthetic 478-point face landmark set with controllable orientation."""
        lm = [(0.5, 0.5, 0.0)] * 478
        from handsense.holistic_detector import (
            FACE_NOSE_TIP, FACE_FOREHEAD, FACE_CHIN,
            FACE_LEFT_EYE_OUTER, FACE_RIGHT_EYE_OUTER,
            FACE_LEFT_EYE_INNER, FACE_RIGHT_EYE_INNER,
            FACE_LEFT_MOUTH, FACE_RIGHT_MOUTH,
            FACE_UPPER_LIP, FACE_LOWER_LIP,
            FACE_LEFT_EYE_TOP, FACE_LEFT_EYE_BOTTOM,
            FACE_RIGHT_EYE_TOP, FACE_RIGHT_EYE_BOTTOM,
        )
        lm = list(lm)
        # Centered face looking straight.
        lm[FACE_NOSE_TIP] = (0.5 + yaw_shift, 0.5 + pitch_shift, 0.0)
        lm[FACE_FOREHEAD] = (0.5, 0.35, 0.0)
        lm[FACE_CHIN] = (0.5, 0.65, 0.0)
        lm[FACE_LEFT_EYE_OUTER] = (0.42, 0.46 + roll_shift, 0.0)
        lm[FACE_RIGHT_EYE_OUTER] = (0.58, 0.46 - roll_shift, 0.0)
        lm[FACE_LEFT_EYE_INNER] = (0.46, 0.46, 0.0)
        lm[FACE_RIGHT_EYE_INNER] = (0.54, 0.46, 0.0)
        lm[FACE_LEFT_MOUTH] = (0.45, 0.58, 0.0)
        lm[FACE_RIGHT_MOUTH] = (0.55, 0.58, 0.0)
        lm[FACE_UPPER_LIP] = (0.5, 0.57, 0.0)
        lm[FACE_LOWER_LIP] = (0.5, 0.59, 0.0)
        lm[FACE_LEFT_EYE_TOP] = (0.44, 0.44, 0.0)
        lm[FACE_LEFT_EYE_BOTTOM] = (0.44, 0.48, 0.0)
        lm[FACE_RIGHT_EYE_TOP] = (0.56, 0.44, 0.0)
        lm[FACE_RIGHT_EYE_BOTTOM] = (0.56, 0.48, 0.0)
        return lm

    def test_neutral_face_near_zero(self):
        from handsense.holistic_detector import _compute_face_orientation
        fo = _compute_face_orientation(self._make_face_landmarks())
        self.assertAlmostEqual(fo.yaw, 0.0, delta=5.0)
        self.assertAlmostEqual(fo.roll, 0.0, delta=5.0)

    def test_yaw_right(self):
        from handsense.holistic_detector import _compute_face_orientation
        fo = _compute_face_orientation(self._make_face_landmarks(yaw_shift=0.05))
        self.assertGreater(fo.yaw, 5.0)

    def test_eyes_open(self):
        from handsense.holistic_detector import _compute_face_orientation
        fo = _compute_face_orientation(self._make_face_landmarks())
        self.assertGreater(fo.left_eye_open, 0.3)
        self.assertGreater(fo.right_eye_open, 0.3)


if __name__ == "__main__":
    unittest.main()
