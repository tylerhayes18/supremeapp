"""Load a GLB/VRM humanoid mesh and deform it per-vertex from pose landmarks.

Instead of segmenting the mesh into rigid body parts (which tears at joints),
this module keeps the mesh as a single unit and applies smooth per-vertex
deformation: each vertex is displaced by a weighted blend of the nearest
pose-landmark offsets, giving organic-looking body movement.
"""

from __future__ import annotations

import math
import os
import urllib.request

import numpy as np

from OpenGL.GL import *

from .holistic_detector import (
    NOSE, LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_PINKY, RIGHT_PINKY,
    LEFT_INDEX, RIGHT_INDEX,
    LEFT_THUMB, RIGHT_THUMB,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

DEFAULT_MODEL_URL = "https://arweave.net/gwG7w4bY-A5c3R6A6GOz3xBCgbPvkFQmqPIDtvnNsYI"
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "handsense")
DEFAULT_MODEL_PATH = os.path.join(MODEL_CACHE, "avatar.glb")

K_NEAREST = 4  # number of landmarks influencing each vertex


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


def _estimate_rest_landmarks(bounds_min: np.ndarray, bounds_max: np.ndarray,
                             height_axis: int = 1) -> np.ndarray:
    """Estimate where MediaPipe pose landmarks would be in the model's T-pose.

    Returns (33, 3) array of rest-pose landmark positions in model space.
    Assumes model is standing, centered, in a T-pose (arms horizontal).
    """
    center = (bounds_min + bounds_max) / 2
    h = bounds_max[height_axis] - bounds_min[height_axis]
    lo = bounds_min[height_axis]
    # For T-pose, the horizontal extent includes outstretched arms.
    arm_extent = (bounds_max[0] - bounds_min[0]) / 2

    cx, cz = center[0], center[2]

    def pt(x, y_frac, z=0.0):
        p = [0.0, 0.0, 0.0]
        p[0] = cx + x
        p[height_axis] = lo + h * y_frac
        p[2] = cz + z
        return p

    shoulder_w = h * 0.12
    hip_w = h * 0.10
    # In T-pose, arms extend to model bounds.
    elbow_x = (shoulder_w + arm_extent) / 2
    wrist_x = arm_extent * 0.88
    hand_x = arm_extent * 0.96

    lm = np.zeros((33, 3), dtype=np.float32)
    lm[NOSE] = pt(0, 0.94, 0.04)
    lm[LEFT_EYE] = pt(-0.03, 0.95, 0.03)
    lm[RIGHT_EYE] = pt(0.03, 0.95, 0.03)
    lm[1] = pt(-0.02, 0.95, 0.03)   # left eye inner
    lm[3] = pt(-0.04, 0.95, 0.02)   # left eye outer
    lm[4] = pt(0.02, 0.95, 0.03)    # right eye inner
    lm[6] = pt(0.04, 0.95, 0.02)    # right eye outer
    lm[LEFT_EAR] = pt(-0.07, 0.93, -0.02)
    lm[RIGHT_EAR] = pt(0.07, 0.93, -0.02)
    lm[9] = pt(-0.02, 0.91, 0.04)   # mouth left
    lm[10] = pt(0.02, 0.91, 0.04)   # mouth right

    lm[LEFT_SHOULDER] = pt(-shoulder_w, 0.82, 0)
    lm[RIGHT_SHOULDER] = pt(shoulder_w, 0.82, 0)
    lm[LEFT_ELBOW] = pt(-elbow_x, 0.82, 0)
    lm[RIGHT_ELBOW] = pt(elbow_x, 0.82, 0)
    lm[LEFT_WRIST] = pt(-wrist_x, 0.82, 0)
    lm[RIGHT_WRIST] = pt(wrist_x, 0.82, 0)
    lm[LEFT_PINKY] = pt(-hand_x, 0.82, -0.01)
    lm[RIGHT_PINKY] = pt(hand_x, 0.82, -0.01)
    lm[LEFT_INDEX] = pt(-hand_x, 0.82, 0.02)
    lm[RIGHT_INDEX] = pt(hand_x, 0.82, 0.02)
    lm[LEFT_THUMB] = pt(-hand_x * 0.95, 0.81, 0.03)
    lm[RIGHT_THUMB] = pt(hand_x * 0.95, 0.81, 0.03)

    lm[LEFT_HIP] = pt(-hip_w, 0.47, 0)
    lm[RIGHT_HIP] = pt(hip_w, 0.47, 0)
    lm[LEFT_KNEE] = pt(-hip_w * 0.9, 0.26, 0)
    lm[RIGHT_KNEE] = pt(hip_w * 0.9, 0.26, 0)
    lm[LEFT_ANKLE] = pt(-hip_w * 0.85, 0.05, 0)
    lm[RIGHT_ANKLE] = pt(hip_w * 0.85, 0.05, 0)
    lm[LEFT_HEEL] = pt(-hip_w * 0.85, 0.02, -0.03)
    lm[RIGHT_HEEL] = pt(hip_w * 0.85, 0.02, -0.03)
    lm[LEFT_FOOT_INDEX] = pt(-hip_w * 0.85, 0.01, 0.06)
    lm[RIGHT_FOOT_INDEX] = pt(hip_w * 0.85, 0.01, 0.06)

    return lm


def _compute_weights(vertices: np.ndarray, rest_landmarks: np.ndarray,
                     k: int = K_NEAREST) -> tuple[np.ndarray, np.ndarray]:
    """For each vertex, find the K nearest rest-pose landmarks and weights.

    Returns:
        indices: (N, K) int array of landmark indices
        weights: (N, K) float array of normalized inverse-distance-squared weights
    """
    n = len(vertices)
    # (N, 33) distance matrix.
    dists = np.linalg.norm(vertices[:, None, :] - rest_landmarks[None, :, :], axis=2)

    indices = np.argpartition(dists, k, axis=1)[:, :k]
    # Gather the actual distances for the K nearest.
    row_idx = np.arange(n)[:, None]
    k_dists = dists[row_idx, indices]
    k_dists = np.maximum(k_dists, 1e-6)

    w = 1.0 / (k_dists ** 2)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum < 1e-12] = 1.0
    weights = (w / w_sum).astype(np.float32)

    return indices.astype(np.int32), weights


class AvatarMesh:
    """A humanoid mesh that can be deformed per-vertex from pose landmarks."""

    def __init__(self, model_path: str | None = None):
        import trimesh

        path = _ensure_model(model_path)
        loaded = trimesh.load(path)
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded

        self.rest_vertices = np.array(mesh.vertices, dtype=np.float32)
        self.faces = np.array(mesh.faces, dtype=np.int32)
        self.vertex_count = len(self.faces) * 3

        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(self.rest_vertices):
            self.rest_normals = np.array(mesh.vertex_normals, dtype=np.float32)
        else:
            self.rest_normals = self._compute_normals(self.rest_vertices, self.faces)

        self.bounds_min = self.rest_vertices.min(axis=0)
        self.bounds_max = self.rest_vertices.max(axis=0)
        self.model_height = self.bounds_max[1] - self.bounds_min[1]
        self.model_center = (self.bounds_min + self.bounds_max) / 2

        # Estimate rest-pose landmarks and compute per-vertex weights.
        self.rest_landmarks = _estimate_rest_landmarks(self.bounds_min, self.bounds_max)
        self._bone_indices, self._bone_weights = _compute_weights(
            self.rest_vertices, self.rest_landmarks, K_NEAREST)

        # Flatten face indices for rendering.
        self._face_idx = self.faces.flatten()

        # Deformed vertex buffer (updated each frame).
        self._deformed = self.rest_vertices.copy()
        self._deformed_normals = self.rest_normals.copy()

        self._n_verts = len(self.rest_vertices)
        self._n_faces = len(self.faces)

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

    def deform(self, tracked_landmarks_model: np.ndarray) -> None:
        """Deform the mesh to match tracked landmark positions (in model space).

        tracked_landmarks_model: (33, 3) array in the model's coordinate space.
        """
        # Compute per-landmark deltas from rest pose.
        deltas = tracked_landmarks_model - self.rest_landmarks  # (33, 3)

        # Gather deltas for each vertex's K nearest landmarks.
        # bone_indices: (N, K), bone_weights: (N, K)
        nearest_deltas = deltas[self._bone_indices]  # (N, K, 3)
        weights = self._bone_weights[:, :, None]     # (N, K, 1)

        displacement = (nearest_deltas * weights).sum(axis=1)  # (N, 3)
        self._deformed = self.rest_vertices + displacement

    def draw(self) -> None:
        """Render the (potentially deformed) mesh with glVertexPointer."""
        idx = self._face_idx
        verts = np.ascontiguousarray(self._deformed[idx], dtype=np.float32)
        norms = np.ascontiguousarray(self._deformed_normals[idx], dtype=np.float32)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, verts)
        glNormalPointer(GL_FLOAT, 0, norms)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
