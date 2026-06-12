"""Tests for mesh skinning: bone weights, LBS deformation, and coordinate transforms."""

import unittest

import numpy as np


def _make_test_mesh():
    """Create a synthetic humanoid GLB with distinct limb parts and load it."""
    import trimesh
    import tempfile
    import os

    R90z = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])

    torso = trimesh.creation.cylinder(radius=0.12, height=0.6, sections=16)
    torso.apply_translation([0, 1.15, 0])
    head = trimesh.creation.uv_sphere(radius=0.10, count=[12, 6])
    head.apply_translation([0, 1.7, 0])
    parts = [torso, head]

    for side in [-1, 1]:
        upper_arm = trimesh.creation.cylinder(radius=0.04, height=0.22, sections=8)
        upper_arm.apply_transform(R90z)
        upper_arm.apply_translation([side * 0.33, 1.47, 0])
        parts.append(upper_arm)

        forearm = trimesh.creation.cylinder(radius=0.035, height=0.22, sections=8)
        forearm.apply_transform(R90z)
        forearm.apply_translation([side * 0.56, 1.47, 0])
        parts.append(forearm)

        thigh = trimesh.creation.cylinder(radius=0.05, height=0.36, sections=10)
        thigh.apply_translation([side * 0.09, 0.65, 0])
        parts.append(thigh)

        shin = trimesh.creation.cylinder(radius=0.04, height=0.36, sections=10)
        shin.apply_translation([side * 0.085, 0.29, 0])
        parts.append(shin)

        foot = trimesh.creation.box(extents=[0.06, 0.04, 0.12])
        foot.apply_translation([side * 0.085, 0.02, 0.03])
        parts.append(foot)

    mesh = trimesh.util.concatenate(parts)
    fd, path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)
    mesh.export(path)

    from handsense.mesh_loader import AvatarMesh
    avatar = AvatarMesh(path)
    os.unlink(path)
    return avatar


class TestBoneWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.avatar = _make_test_mesh()

    def test_weights_sum_to_one(self):
        sums = self.avatar._bone_weights.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_weights_nonnegative(self):
        self.assertTrue(np.all(self.avatar._bone_weights >= 0))

    def test_bone_indices_in_range(self):
        from handsense.mesh_loader import N_BONES
        self.assertTrue(np.all(self.avatar._bone_idx >= 0))
        self.assertTrue(np.all(self.avatar._bone_idx < N_BONES))

    def test_all_verts_assigned(self):
        from handsense.mesh_loader import N_BONES
        total = sum(np.sum(self.avatar._bone_idx[:, 0] == b) for b in range(N_BONES))
        self.assertEqual(total, self.avatar._n_verts)


class TestSkinning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.avatar = _make_test_mesh()

    def test_identity_skinning(self):
        rest = self.avatar._rest_landmarks[:33].copy()
        self.avatar.skin(rest)
        diff = np.abs(self.avatar._skinned_verts - self.avatar._vertices).max()
        self.assertLess(diff, 0.001)

    def test_raise_left_arm(self):
        from handsense.holistic_detector import LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        rest = self.avatar._rest_landmarks[:33].copy()
        ls = rest[LEFT_SHOULDER].copy()
        rest[LEFT_ELBOW] = ls + [0, 0.25, 0]
        rest[LEFT_WRIST] = ls + [0, 0.50, 0]
        self.avatar.skin(rest)

        # Left arm bones: 2 (upper) and 3 (forearm).
        arm_mask = np.isin(self.avatar._bone_idx[:, 0], [2, 3])
        if np.any(arm_mask):
            dy = (self.avatar._skinned_verts[arm_mask, 1] -
                  self.avatar._vertices[arm_mask, 1]).mean()
            self.assertGreater(dy, 0.03)

    def test_independent_limbs(self):
        from handsense.holistic_detector import LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        rest = self.avatar._rest_landmarks[:33].copy()
        ls = rest[LEFT_SHOULDER].copy()
        rest[LEFT_ELBOW] = ls + [0, 0.25, 0]
        rest[LEFT_WRIST] = ls + [0, 0.50, 0]
        self.avatar.skin(rest)

        # Right arm bones: 4 (upper) and 5 (forearm).
        r_arm_mask = np.isin(self.avatar._bone_idx[:, 0], [4, 5])
        if np.any(r_arm_mask):
            r_diff = np.abs(self.avatar._skinned_verts[r_arm_mask] -
                            self.avatar._vertices[r_arm_mask]).max()
            self.assertLess(r_diff, 0.05)

    def test_no_nan_or_inf(self):
        rest = self.avatar._rest_landmarks[:33].copy()
        self.avatar.skin(rest)
        self.assertFalse(np.any(np.isnan(self.avatar._skinned_verts)))
        self.assertFalse(np.any(np.isinf(self.avatar._skinned_verts)))
        self.assertFalse(np.any(np.isnan(self.avatar._skinned_norms)))

    def test_normals_unit_length(self):
        rest = self.avatar._rest_landmarks[:33].copy()
        self.avatar.skin(rest)
        lengths = np.linalg.norm(self.avatar._skinned_norms, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=0.01)


class TestRotationBetween(unittest.TestCase):
    def test_identity(self):
        from handsense.mesh_loader import _rotation_between
        v = np.array([0, 1, 0], dtype=np.float32)
        R = _rotation_between(v, v)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-5)

    def test_180_degrees(self):
        from handsense.mesh_loader import _rotation_between
        v1 = np.array([0, 1, 0], dtype=np.float32)
        v2 = np.array([0, -1, 0], dtype=np.float32)
        R = _rotation_between(v1, v2)
        result = R @ v1
        np.testing.assert_allclose(result, v2, atol=1e-4)

    def test_90_degrees(self):
        from handsense.mesh_loader import _rotation_between
        v1 = np.array([1, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0], dtype=np.float32)
        R = _rotation_between(v1, v2)
        result = R @ v1
        np.testing.assert_allclose(result, v2, atol=1e-4)

    def test_orthogonal(self):
        from handsense.mesh_loader import _rotation_between
        v1 = np.array([0.6, 0.8, 0], dtype=np.float32)
        v2 = np.array([0, 0, 1], dtype=np.float32)
        R = _rotation_between(v1, v2)
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=4)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)


class TestPointSegmentDist(unittest.TestCase):
    def test_point_on_segment(self):
        from handsense.mesh_loader import _point_segment_dist
        pts = np.array([[0.5, 0, 0]], dtype=np.float32)
        a = np.array([0, 0, 0], dtype=np.float32)
        b = np.array([1, 0, 0], dtype=np.float32)
        d = _point_segment_dist(pts, a, b)
        self.assertAlmostEqual(d[0], 0.0, places=5)

    def test_perpendicular(self):
        from handsense.mesh_loader import _point_segment_dist
        pts = np.array([[0.5, 1, 0]], dtype=np.float32)
        a = np.array([0, 0, 0], dtype=np.float32)
        b = np.array([1, 0, 0], dtype=np.float32)
        d = _point_segment_dist(pts, a, b)
        self.assertAlmostEqual(d[0], 1.0, places=5)

    def test_past_endpoint(self):
        from handsense.mesh_loader import _point_segment_dist
        pts = np.array([[2, 0, 0]], dtype=np.float32)
        a = np.array([0, 0, 0], dtype=np.float32)
        b = np.array([1, 0, 0], dtype=np.float32)
        d = _point_segment_dist(pts, a, b)
        self.assertAlmostEqual(d[0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
