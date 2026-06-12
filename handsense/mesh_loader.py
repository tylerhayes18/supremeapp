"""Load a GLB/VRM humanoid mesh and pose it via linear blend skinning.

Each vertex is assigned weighted influence from the nearest skeleton bones
(line segments between MediaPipe landmark pairs). At runtime, per-bone
rigid transforms (rotation + translation) are computed from rest -> tracked
pose and blended per-vertex, producing smooth articulated movement without
joint tearing.

An optional T-pose calibration step captures the user's actual body
proportions, mapping them onto the model for much better bone weights.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

from OpenGL.GL import *

from .holistic_detector import (
    NOSE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    LEFT_INDEX, RIGHT_INDEX,
    LEFT_PINKY, RIGHT_PINKY,
    LEFT_THUMB, RIGHT_THUMB,
)

DEFAULT_MODEL_URL = "https://arweave.net/gwG7w4bY-A5c3R6A6GOz3xBCgbPvkFQmqPIDtvnNsYI"
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "handsense")
DEFAULT_MODEL_PATH = os.path.join(MODEL_CACHE, "avatar.glb")

K_WEIGHTS = 3

# Virtual landmark indices (appended after MediaPipe's 33).
V_HIP_CENTER = 33
V_SHOULDER_CENTER = 34
N_LANDMARKS = 35

# Skeleton bones: (parent_landmark, child_landmark).
# Torso has three bones (center spine + two sides) to prevent arm bones
# from capturing torso vertices.
SKEL_BONES = [
    (V_HIP_CENTER, V_SHOULDER_CENTER),   # 0  spine center
    (LEFT_HIP, LEFT_SHOULDER),            # 1  left torso side
    (RIGHT_HIP, RIGHT_SHOULDER),          # 2  right torso side
    (V_SHOULDER_CENTER, NOSE),            # 3  head/neck
    (LEFT_SHOULDER, LEFT_ELBOW),          # 4  left upper arm
    (LEFT_ELBOW, LEFT_WRIST),             # 5  left forearm
    (RIGHT_SHOULDER, RIGHT_ELBOW),        # 6  right upper arm
    (RIGHT_ELBOW, RIGHT_WRIST),           # 7  right forearm
    (LEFT_HIP, LEFT_KNEE),               # 8  left thigh
    (LEFT_KNEE, LEFT_ANKLE),             # 9  left shin
    (RIGHT_HIP, RIGHT_KNEE),             # 10 right thigh
    (RIGHT_KNEE, RIGHT_ANKLE),           # 11 right shin
    (LEFT_ANKLE, LEFT_FOOT_INDEX),       # 12 left foot
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),     # 13 right foot
]

N_BONES = len(SKEL_BONES)

# Bones that should only capture vertices far from the body center.
# These get a distance penalty for central vertices.
_LIMB_BONES = {4, 5, 6, 7}    # arm bones
_LEG_BONES = {8, 9, 10, 11}   # leg bones


def _ensure_model(model_path: str | None = None) -> str:
    path = model_path or DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"downloading avatar model -> {path}")
        try:
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, path)
        except OSError as exc:
            raise SystemExit(
                f"could not download avatar model ({exc}).\n"
                f"Download manually from\n  {DEFAULT_MODEL_URL}\nand save as {path}"
            ) from exc
    return path


def _add_virtual_landmarks(lm: np.ndarray) -> np.ndarray:
    """Extend (33, 3) landmarks with computed midpoints to (35, 3)."""
    hip_center = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2
    shoulder_center = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2
    return np.vstack([lm, [hip_center], [shoulder_center]])


def _rotation_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix rotating unit vector v1 onto v2 (Rodrigues)."""
    c = float(np.dot(v1, v2))
    if c > 0.9999:
        return np.eye(3, dtype=np.float32)
    if c < -0.9999:
        perp = np.array([1, 0, 0], dtype=np.float32)
        if abs(v1[0]) > 0.9:
            perp = np.array([0, 1, 0], dtype=np.float32)
        axis = np.cross(v1, perp)
        axis /= np.linalg.norm(axis)
        return (2 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    axis = np.cross(v1, v2)
    s = np.linalg.norm(axis)
    axis /= s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + K * s + K @ K * (1 - c))


def _point_segment_dist(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point (N,3) to segment a->b. Returns (N,)."""
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return np.linalg.norm(points - a, axis=1)
    t = np.dot(points - a, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    proj = a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def _estimate_rest_landmarks(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    """Place 35 landmarks (33 MediaPipe + 2 virtual) in the model's T-pose."""
    h = float(bounds_max[1] - bounds_min[1])
    lo = float(bounds_min[1])
    cx = float((bounds_min[0] + bounds_max[0]) / 2)
    cz = float((bounds_min[2] + bounds_max[2]) / 2)

    arm_span = float(bounds_max[0] - bounds_min[0])

    shoulder_y = lo + h * 0.81
    hip_y = lo + h * 0.47
    knee_y = lo + h * 0.26
    ankle_y = lo + h * 0.05
    foot_y = lo + h * 0.01
    nose_y = lo + h * 0.94
    ear_y = lo + h * 0.93

    shoulder_w = h * 0.12
    hip_w = h * 0.10
    elbow_x = arm_span * 0.32
    wrist_x = arm_span * 0.44
    hand_x = arm_span * 0.48

    lm = np.zeros((N_LANDMARKS, 3), dtype=np.float32)

    lm[NOSE] = [cx, nose_y, cz + h * 0.04]
    lm[LEFT_EAR] = [cx - h * 0.06, ear_y, cz - h * 0.02]
    lm[RIGHT_EAR] = [cx + h * 0.06, ear_y, cz - h * 0.02]
    for i in range(1, 7):
        lm[i] = [cx, nose_y, cz + h * 0.03]
    lm[7] = lm[LEFT_EAR]
    lm[8] = lm[RIGHT_EAR]
    lm[9] = [cx - h * 0.02, lo + h * 0.91, cz + h * 0.04]
    lm[10] = [cx + h * 0.02, lo + h * 0.91, cz + h * 0.04]

    lm[LEFT_SHOULDER] = [cx - shoulder_w, shoulder_y, cz]
    lm[RIGHT_SHOULDER] = [cx + shoulder_w, shoulder_y, cz]
    lm[LEFT_ELBOW] = [cx - elbow_x, shoulder_y, cz]
    lm[RIGHT_ELBOW] = [cx + elbow_x, shoulder_y, cz]
    lm[LEFT_WRIST] = [cx - wrist_x, shoulder_y, cz]
    lm[RIGHT_WRIST] = [cx + wrist_x, shoulder_y, cz]
    lm[LEFT_PINKY] = [cx - hand_x, shoulder_y, cz - h * 0.01]
    lm[RIGHT_PINKY] = [cx + hand_x, shoulder_y, cz - h * 0.01]
    lm[LEFT_INDEX] = [cx - hand_x, shoulder_y, cz + h * 0.01]
    lm[RIGHT_INDEX] = [cx + hand_x, shoulder_y, cz + h * 0.01]
    lm[LEFT_THUMB] = [cx - hand_x * 0.95, shoulder_y - h * 0.01, cz + h * 0.02]
    lm[RIGHT_THUMB] = [cx + hand_x * 0.95, shoulder_y - h * 0.01, cz + h * 0.02]

    lm[LEFT_HIP] = [cx - hip_w, hip_y, cz]
    lm[RIGHT_HIP] = [cx + hip_w, hip_y, cz]
    lm[LEFT_KNEE] = [cx - hip_w * 0.95, knee_y, cz]
    lm[RIGHT_KNEE] = [cx + hip_w * 0.95, knee_y, cz]
    lm[LEFT_ANKLE] = [cx - hip_w * 0.90, ankle_y, cz]
    lm[RIGHT_ANKLE] = [cx + hip_w * 0.90, ankle_y, cz]
    lm[LEFT_HEEL] = [cx - hip_w * 0.90, lo + h * 0.02, cz - h * 0.03]
    lm[RIGHT_HEEL] = [cx + hip_w * 0.90, lo + h * 0.02, cz - h * 0.03]
    lm[LEFT_FOOT_INDEX] = [cx - hip_w * 0.90, foot_y, cz + h * 0.05]
    lm[RIGHT_FOOT_INDEX] = [cx + hip_w * 0.90, foot_y, cz + h * 0.05]

    lm[V_HIP_CENTER] = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2
    lm[V_SHOULDER_CENTER] = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2

    return lm


def _compute_bone_weights(vertices: np.ndarray,
                          rest_landmarks: np.ndarray,
                          model_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Assign each vertex weighted influence from the K nearest bones.

    Applies a distance penalty to arm/leg bones for central-body vertices,
    preventing limb bones from capturing torso geometry.
    """
    n = len(vertices)
    cx = float((rest_landmarks[LEFT_HIP][0] + rest_landmarks[RIGHT_HIP][0]) / 2)
    shoulder_w = abs(rest_landmarks[LEFT_SHOULDER][0] - cx)
    hip_y = float(rest_landmarks[V_HIP_CENTER][1])
    shoulder_y = float(rest_landmarks[V_SHOULDER_CENTER][1])

    dists = np.full((n, N_BONES), np.inf, dtype=np.float32)
    for b, (lm_a, lm_b) in enumerate(SKEL_BONES):
        dists[:, b] = _point_segment_dist(vertices, rest_landmarks[lm_a], rest_landmarks[lm_b])

    # Penalize arm bones for vertices inside the shoulder width.
    torso_guard = model_height * 0.10
    x_dist = np.abs(vertices[:, 0] - cx)
    inside_torso = x_dist < shoulder_w * 1.3
    for b in _LIMB_BONES:
        dists[inside_torso, b] += torso_guard

    # Penalize leg bones for vertices above hip level.
    above_hip = vertices[:, 1] > (hip_y + shoulder_w * 0.5)
    for b in _LEG_BONES:
        dists[above_hip, b] += torso_guard

    # Penalize arm bones for vertices below hip level.
    below_hip = vertices[:, 1] < hip_y
    for b in _LIMB_BONES:
        dists[below_hip, b] += torso_guard

    bone_idx = np.argpartition(dists, K_WEIGHTS, axis=1)[:, :K_WEIGHTS].astype(np.int32)
    row_idx = np.arange(n)[:, None]
    k_dists = dists[row_idx, bone_idx]

    avg_bone_len = np.mean([
        np.linalg.norm(rest_landmarks[b] - rest_landmarks[a])
        for a, b in SKEL_BONES
    ])
    sigma = max(avg_bone_len * 0.35, 0.01)
    w = np.exp(-(k_dists ** 2) / (sigma ** 2))
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum < 1e-12] = 1.0
    weights = (w / w_sum).astype(np.float32)

    return bone_idx, weights


def is_tpose(landmarks_33: np.ndarray) -> bool:
    """Check if the person is roughly in a T-pose (arms horizontal)."""
    ls = landmarks_33[LEFT_SHOULDER]
    rs = landmarks_33[RIGHT_SHOULDER]
    le = landmarks_33[LEFT_ELBOW]
    re = landmarks_33[RIGHT_ELBOW]
    lw = landmarks_33[LEFT_WRIST]
    rw = landmarks_33[RIGHT_WRIST]

    shoulder_width = abs(rs[0] - ls[0])
    if shoulder_width < 0.01:
        return False

    # Arms should extend well past shoulders.
    left_span = abs(lw[0] - ls[0])
    right_span = abs(rw[0] - rs[0])
    if left_span < shoulder_width * 0.8 or right_span < shoulder_width * 0.8:
        return False

    # Elbows and wrists should be near shoulder height (not hanging down).
    shoulder_y = (ls[1] + rs[1]) / 2
    for pt in [le, re, lw, rw]:
        if abs(pt[1] - shoulder_y) > shoulder_width * 0.6:
            return False

    return True


class AvatarMesh:
    """A humanoid mesh with bone-based linear blend skinning."""

    def __init__(self, model_path: str | None = None):
        import trimesh

        path = _ensure_model(model_path)
        loaded = trimesh.load(path)
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded

        self._vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(self._vertices):
            self._normals = np.array(mesh.vertex_normals, dtype=np.float32)
        else:
            self._normals = self._compute_normals(self._vertices, faces)

        self.bounds_min = self._vertices.min(axis=0)
        self.bounds_max = self._vertices.max(axis=0)
        self.model_height = float(self.bounds_max[1] - self.bounds_min[1])
        self.model_center = (self.bounds_min + self.bounds_max) / 2
        self.hip_y = float(self.bounds_min[1] + self.model_height * 0.47)

        colors = self._extract_colors(mesh, len(self._vertices))

        self._face_idx = faces.flatten()
        self._vertex_count = len(self._face_idx)

        self._draw_colors = None
        if colors is not None:
            self._draw_colors = np.ascontiguousarray(
                colors[self._face_idx], dtype=np.float32)

        # Initial bone weights from bounding-box estimates.
        self._rest_landmarks = _estimate_rest_landmarks(self.bounds_min, self.bounds_max)
        self._setup_bones(self._rest_landmarks)

        self._calibrated = False
        self._skinned_verts = self._vertices.copy()
        self._skinned_norms = self._normals.copy()

        self._n_verts = len(self._vertices)
        self._n_faces = len(faces)

    def _setup_bones(self, rest_landmarks: np.ndarray) -> None:
        """Compute bone weights and precompute per-bone vertex lists."""
        self._rest_landmarks = rest_landmarks
        self._bone_idx, self._bone_weights = _compute_bone_weights(
            self._vertices, rest_landmarks, self.model_height)

        self._bone_vert_lists: list[list[np.ndarray]] = []
        for b in range(N_BONES):
            per_k = []
            for k in range(K_WEIGHTS):
                per_k.append(np.where(self._bone_idx[:, k] == b)[0])
            self._bone_vert_lists.append(per_k)

        self._rest_dirs = np.zeros((N_BONES, 3), dtype=np.float32)
        self._rest_anchors = np.zeros((N_BONES, 3), dtype=np.float32)
        for b, (lm_a, lm_b) in enumerate(SKEL_BONES):
            a = rest_landmarks[lm_a]
            d = rest_landmarks[lm_b] - a
            length = np.linalg.norm(d)
            self._rest_anchors[b] = a
            self._rest_dirs[b] = d / max(length, 1e-6)

    def calibrate(self, gl_landmarks_33: np.ndarray,
                  hip_center: np.ndarray, scale: float) -> None:
        """Recalculate bone weights from a captured T-pose.

        gl_landmarks_33: (33, 3) landmarks in GL space from the T-pose capture.
        hip_center: GL-space hip center used for coordinate conversion.
        scale: model-to-GL scale factor.
        """
        model_hip = np.array([self.model_center[0], self.hip_y, self.model_center[2]])
        tracked_model = (gl_landmarks_33 - hip_center) / scale + model_hip
        rest_35 = _add_virtual_landmarks(tracked_model)
        self._setup_bones(rest_35)
        self._calibrated = True
        print("T-pose calibration complete")

    def skin(self, tracked_landmarks: np.ndarray) -> None:
        """Deform vertices via LBS to match tracked landmarks in model space."""
        if len(tracked_landmarks) == 33:
            tracked_landmarks = _add_virtual_landmarks(tracked_landmarks)

        Rs = np.zeros((N_BONES, 3, 3), dtype=np.float32)
        offsets = np.zeros((N_BONES, 3), dtype=np.float32)

        for b, (lm_a, lm_b) in enumerate(SKEL_BONES):
            track_a = tracked_landmarks[lm_a]
            track_d = tracked_landmarks[lm_b] - track_a
            track_len = np.linalg.norm(track_d)
            if track_len < 1e-6:
                Rs[b] = np.eye(3, dtype=np.float32)
                offsets[b] = track_a - self._rest_anchors[b]
                continue
            track_dir = track_d / track_len
            if b <= 2:
                track_dir = 0.6 * self._rest_dirs[b] + 0.4 * track_dir
                dn = np.linalg.norm(track_dir)
                if dn > 1e-6:
                    track_dir = track_dir / dn
                else:
                    track_dir = self._rest_dirs[b].copy()
            R = _rotation_between(self._rest_dirs[b], track_dir)
            Rs[b] = R
            offsets[b] = track_a - R @ self._rest_anchors[b]

        result = np.zeros_like(self._vertices)
        norm_result = np.zeros_like(self._normals)

        for b in range(N_BONES):
            R_T = Rs[b].T
            off = offsets[b]
            for k in range(K_WEIGHTS):
                idx = self._bone_vert_lists[b][k]
                if len(idx) == 0:
                    continue
                w = self._bone_weights[idx, k:k + 1]
                result[idx] += w * (self._vertices[idx] @ R_T + off)
                norm_result[idx] += w * (self._normals[idx] @ R_T)

        norms = np.linalg.norm(norm_result, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        self._skinned_norms = (norm_result / norms).astype(np.float32)
        self._skinned_verts = result

    def draw(self) -> None:
        """Render the skinned mesh."""
        draw_verts = np.ascontiguousarray(
            self._skinned_verts[self._face_idx], dtype=np.float32)
        draw_norms = np.ascontiguousarray(
            self._skinned_norms[self._face_idx], dtype=np.float32)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, draw_verts)
        glNormalPointer(GL_FLOAT, 0, draw_norms)
        if self._draw_colors is not None:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(4, GL_FLOAT, 0, self._draw_colors)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        if self._draw_colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

    @staticmethod
    def _extract_colors(mesh, n_verts):
        """Return (n_verts, 4) float32 RGBA array, or None."""
        try:
            visual = mesh.visual
            if hasattr(visual, "to_color"):
                visual = visual.to_color()
            vc = visual.vertex_colors
            if vc is None or len(vc) != n_verts:
                return None
            colors = np.array(vc, dtype=np.float32)
            if colors.max() > 1.0:
                colors /= 255.0
            if colors.shape[1] == 3:
                alpha = np.ones((len(colors), 1), dtype=np.float32)
                colors = np.hstack([colors, alpha])
            if np.allclose(colors[:, :3], colors[0, :3], atol=0.01):
                return None
            r, g, b_ch = colors[:, 0], colors[:, 1], colors[:, 2]
            max_c = np.maximum(np.maximum(r, g), b_ch)
            min_c = np.minimum(np.minimum(r, g), b_ch)
            extreme = ((max_c - min_c) > 0.5) & (max_c > 0.6)
            n_extreme = int(np.sum(extreme))
            if 0 < n_extreme < int(0.05 * len(colors)):
                median_color = np.median(colors[~extreme, :3], axis=0)
                colors[extreme, :3] = median_color
            return colors
        except Exception:
            pass
        return None

    @staticmethod
    def _compute_normals(vertices, faces):
        normals = np.zeros_like(vertices)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for i in range(3):
            np.add.at(normals, faces[:, i], fn)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return (normals / norms).astype(np.float32)
