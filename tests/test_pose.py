"""Tests for pose-related coordinate conversion and body structure."""

import math
import unittest


class TestMediaPipeToGL(unittest.TestCase):
    def test_coordinate_conversion(self):
        from handsense.avatar import _mediapipe_to_gl
        # MediaPipe: x right, y down, z toward camera
        # OpenGL:    x right, y up,   z toward viewer
        gx, gy, gz = _mediapipe_to_gl(1.0, 2.0, 3.0)
        self.assertAlmostEqual(gx, 1.0)
        self.assertAlmostEqual(gy, -2.0)
        self.assertAlmostEqual(gz, -3.0)

    def test_origin_unchanged(self):
        from handsense.avatar import _mediapipe_to_gl
        self.assertEqual(_mediapipe_to_gl(0, 0, 0), (0, 0, 0))


class TestBodyDefinitions(unittest.TestCase):
    def test_all_bone_indices_valid(self):
        from handsense.avatar import BONES
        for start, end, radius, color in BONES:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start, 32)
            self.assertGreaterEqual(end, 0)
            self.assertLessEqual(end, 32)
            self.assertGreater(radius, 0)
            self.assertEqual(len(color), 4)

    def test_all_joint_indices_valid(self):
        from handsense.avatar import JOINTS
        for idx, radius in JOINTS:
            self.assertGreaterEqual(idx, 0)
            self.assertLessEqual(idx, 32)
            self.assertGreater(radius, 0)

    def test_pose_connections_valid(self):
        from handsense.pose_detector import POSE_CONNECTIONS
        for a, b in POSE_CONNECTIONS:
            self.assertGreaterEqual(a, 0)
            self.assertLessEqual(a, 32)
            self.assertGreaterEqual(b, 0)
            self.assertLessEqual(b, 32)


if __name__ == "__main__":
    unittest.main()
