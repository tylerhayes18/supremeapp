"""CAD generation tests — skipped when trimesh/manifold3d are absent."""

import math
import unittest

try:
    import numpy as np
    import trimesh  # noqa: F401
    import manifold3d  # noqa: F401
    import shapely  # noqa: F401
    HAVE_CAD = True
except ImportError:
    HAVE_CAD = False

from skyarm import spec


@unittest.skipUnless(HAVE_CAD, "CAD deps (trimesh/manifold3d/shapely) not installed")
class TestCycloidProfile(unittest.TestCase):
    def test_profile_radius_bounds(self):
        from skyarm.cad.cycloidal import cycloid_profile
        for cs in (spec.CYCLO_BIG, spec.CYCLO_MID, spec.CYCLO_SMALL):
            pts = cycloid_profile(cs)
            r = np.linalg.norm(pts, axis=1)
            # disc must stay inside the pin circle and outside a sane core
            self.assertLess(r.max(), cs.pin_circle_r + cs.pin_r, cs.name)
            self.assertGreater(r.min(), cs.pin_circle_r / 2, cs.name)

    def test_lobe_count_is_reduction(self):
        from skyarm.cad.cycloidal import cycloid_profile
        cs = spec.CYCLO_SMALL
        pts = cycloid_profile(cs, samples=4400)
        r = np.linalg.norm(pts, axis=1)
        # count local maxima (lobes) — equals pin_count - 1
        peaks = 0
        n = len(r)
        for i in range(n):
            if r[i] > r[i - 1] and r[i] >= r[(i + 1) % n]:
                peaks += 1
        self.assertEqual(peaks, cs.reduction)

    def test_no_undercut_condition(self):
        # eccentricity must satisfy E < R/N for a valid cycloid
        for cs in (spec.CYCLO_BIG, spec.CYCLO_MID, spec.CYCLO_YAW,
                   spec.CYCLO_SMALL):
            self.assertLess(cs.eccentricity, cs.pin_circle_r / cs.pin_count,
                            cs.name)


@unittest.skipUnless(HAVE_CAD, "CAD deps (trimesh/manifold3d/shapely) not installed")
class TestPrintedParts(unittest.TestCase):
    def test_small_gearbox_parts_watertight_and_fit_bed(self):
        from skyarm.cad import cycloidal
        from skyarm.cad.primitives import check_part
        for name, mesh in cycloidal.parts(spec.CYCLO_SMALL).items():
            check_part(mesh, name, spec.PRINT_VOLUME_MM)  # raises on failure
            self.assertGreater(mesh.volume, 1000.0, name)  # > 1 cm3 of plastic

    def test_structural_parts_watertight(self):
        from skyarm.cad import parts
        from skyarm.cad.primitives import check_part
        for fn in (parts.ceiling_bracket, parts.x_truck, parts.y_carriage,
                   parts.upper_root, parts.upper_mid, parts.upper_tip,
                   parts.forearm_root, parts.forearm_tip,
                   parts.gripper_body):
            check_part(fn(), fn.__name__, spec.PRINT_VOLUME_MM)

    def test_link_segments_sum_to_link_lengths(self):
        from skyarm.cad import parts
        self.assertAlmostEqual(sum(parts.UPPER_SEGS), spec.UPPER_ARM_MM)
        self.assertAlmostEqual(sum(parts.FOREARM_SEGS), spec.FOREARM_MM)

    def test_mirrored_jaw_is_valid_solid(self):
        from skyarm.cad import parts
        m = parts.gripper_jaw(mirrored=True)
        self.assertTrue(m.is_watertight)
        self.assertGreater(m.volume, 0)


if __name__ == "__main__":
    unittest.main()
