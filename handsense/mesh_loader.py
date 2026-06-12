"""Load and render GLB/VRM humanoid meshes with OpenGL.

Loads a glTF-binary mesh via trimesh and uploads it to OpenGL as vertex
buffer objects for efficient rendering. Supports posing the mesh by
repositioning body segments based on MediaPipe landmark positions.

The mesh is segmented by vertical body zone and rendered per-segment
with independent transforms, giving a "sectional mannequin" style that
tracks real body movement.
"""

from __future__ import annotations

import math
import os
import urllib.request

import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *

# Default avatar model (CC0 VRM from open-source-avatars).
DEFAULT_MODEL_URL = "https://arweave.net/gwG7w4bY-A5c3R6A6GOz3xBCgbPvkFQmqPIDtvnNsYI"
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "handsense")
DEFAULT_MODEL_PATH = os.path.join(MODEL_CACHE, "avatar.glb")


def _ensure_model(model_path: str | None = None) -> str:
    path = model_path or DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        url = DEFAULT_MODEL_URL
        print(f"downloading avatar model -> {path}")
        try:
            urllib.request.urlretrieve(url, path)
        except OSError as exc:
            raise SystemExit(
                f"could not download avatar model ({exc}).\n"
                f"Download manually from\n  {url}\nand save as {path}"
            ) from exc
    return path


# ── Body segment classification ────────────────────────────────────────

# We classify every vertex into a body segment based on its position in the
# model's rest pose (T-pose, Y up, ~1.7m tall).  Each segment gets its own
# OpenGL transform so it can follow the corresponding MediaPipe landmark.

SEGMENT_NAMES = [
    "head",        # 0
    "torso_upper", # 1
    "torso_lower", # 2
    "left_upper_arm",  # 3
    "left_forearm",    # 4
    "left_hand",       # 5
    "right_upper_arm", # 6
    "right_forearm",   # 7
    "right_hand",      # 8
    "left_thigh",      # 9
    "left_shin",       # 10
    "left_foot",       # 11
    "right_thigh",     # 12
    "right_shin",      # 13
    "right_foot",      # 14
]


def _classify_vertex(v, bounds_min, bounds_max) -> int:
    """Assign a body-segment index to vertex v based on its rest-pose position.

    Works for models in a T-pose (arms horizontal) or A-pose. Uses X distance
    from center to classify arm sub-segments (shoulder vs elbow vs hand) since
    they all sit at a similar Y height in the rest pose.
    """
    size = bounds_max - bounds_min
    size[size < 1e-6] = 1.0
    n = (v - bounds_min) / size
    nx, ny, nz = n[0], n[1], n[2]

    center_x = 0.5
    arm_thresh = 0.20

    if ny > 0.88:
        return 0  # head

    # Arms: far from center-x in the upper body.  Sub-segment by how far
    # outward the vertex is (distance from center-x), not height.
    dx = abs(nx - center_x)
    if dx > arm_thresh and ny > 0.50:
        is_left = nx < center_x
        if dx < 0.34:
            return 3 if is_left else 6   # upper arm (near shoulder)
        elif dx < 0.46:
            return 4 if is_left else 7   # forearm
        else:
            return 5 if is_left else 8   # hand

    # Legs: below hip line.
    if ny < 0.46:
        is_left = nx < center_x
        if ny > 0.26:
            return 9 if is_left else 12   # thigh
        elif ny > 0.10:
            return 10 if is_left else 13  # shin
        else:
            return 11 if is_left else 14  # foot

    if ny > 0.65:
        return 1  # upper torso
    return 2  # lower torso


class MeshSegment:
    """One body-part's geometry, uploaded to OpenGL."""

    def __init__(self, vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray,
                 centroid: np.ndarray):
        self.centroid = centroid  # rest-pose center of this segment
        self.vertex_count = len(faces) * 3

        # Flatten for glDrawArrays.
        idx = faces.flatten()
        self._verts = np.ascontiguousarray(vertices[idx], dtype=np.float32)
        self._norms = np.ascontiguousarray(normals[idx], dtype=np.float32)

        self._vbo_v = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_v)
        glBufferData(GL_ARRAY_BUFFER, self._verts.nbytes, self._verts, GL_STATIC_DRAW)

        self._vbo_n = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_n)
        glBufferData(GL_ARRAY_BUFFER, self._norms.nbytes, self._norms, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self) -> None:
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_v)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_n)
        glNormalPointer(GL_FLOAT, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


class AvatarMesh:
    """A humanoid mesh split into poseable body segments."""

    def __init__(self, model_path: str | None = None):
        import trimesh

        path = _ensure_model(model_path)
        loaded = trimesh.load(path)
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        # Compute or use vertex normals.
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
            normals = np.array(mesh.vertex_normals, dtype=np.float32)
        else:
            normals = np.zeros_like(vertices)
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                n = np.cross(v1 - v0, v2 - v1)
                ln = np.linalg.norm(n)
                if ln > 1e-8:
                    n /= ln
                normals[face[0]] += n
                normals[face[1]] += n
                normals[face[2]] += n
            norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
            norms_len[norms_len < 1e-8] = 1.0
            normals /= norms_len

        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        self._model_height = bounds_max[1] - bounds_min[1]
        self._model_center = (bounds_min + bounds_max) / 2

        # Classify each vertex into a body segment.
        seg_ids = np.array([_classify_vertex(v, bounds_min, bounds_max) for v in vertices])

        # Classify each face by majority vote of its vertices.
        face_segs = np.zeros(len(faces), dtype=np.int32)
        for i, face in enumerate(faces):
            counts = np.bincount(seg_ids[face], minlength=len(SEGMENT_NAMES))
            face_segs[i] = counts.argmax()

        # Build per-segment geometry.
        self.segments: list[MeshSegment | None] = []
        self.segment_names = SEGMENT_NAMES

        for seg_id in range(len(SEGMENT_NAMES)):
            mask = face_segs == seg_id
            seg_faces = faces[mask]
            if len(seg_faces) == 0:
                self.segments.append(None)
                continue

            seg_vert_ids = np.unique(seg_faces)
            centroid = vertices[seg_vert_ids].mean(axis=0)
            self.segments.append(MeshSegment(vertices, normals, seg_faces, centroid))

        self._total_verts = len(vertices)
        self._total_faces = len(faces)

    @property
    def model_height(self) -> float:
        return self._model_height

    @property
    def model_center(self) -> np.ndarray:
        return self._model_center
