"""Load a GLB/VRM humanoid mesh for OpenGL rendering.

Loads the mesh as a single intact piece — no segmentation, no per-vertex
deformation. The mesh is rendered with a global transform (position,
rotation, scale) computed from the tracked body landmarks.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

from OpenGL.GL import *

DEFAULT_MODEL_URL = "https://arweave.net/gwG7w4bY-A5c3R6A6GOz3xBCgbPvkFQmqPIDtvnNsYI"
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "handsense")
DEFAULT_MODEL_PATH = os.path.join(MODEL_CACHE, "avatar.glb")


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


class AvatarMesh:
    """A humanoid mesh uploaded to OpenGL as vertex arrays.

    The mesh is rendered as one solid piece — no segmentation, no per-vertex
    deformation. Callers apply a global transform (translate/rotate/scale)
    before calling draw().
    """

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

        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
            normals = np.array(mesh.vertex_normals, dtype=np.float32)
        else:
            normals = self._compute_normals(vertices, faces)

        self.bounds_min = vertices.min(axis=0)
        self.bounds_max = vertices.max(axis=0)
        self.model_height = float(self.bounds_max[1] - self.bounds_min[1])
        self.model_center = (self.bounds_min + self.bounds_max) / 2
        self.hip_y = float(self.bounds_min[1] + self.model_height * 0.47)

        # Extract per-vertex colors if the model provides them.
        colors = self._extract_colors(mesh, len(vertices))

        # Flatten for GL_TRIANGLES rendering.
        idx = faces.flatten()
        self._draw_verts = np.ascontiguousarray(vertices[idx], dtype=np.float32)
        self._draw_norms = np.ascontiguousarray(normals[idx], dtype=np.float32)
        self._vertex_count = len(idx)

        self._draw_colors = None
        if colors is not None:
            self._draw_colors = np.ascontiguousarray(colors[idx], dtype=np.float32)

        self._n_verts = len(vertices)
        self._n_faces = len(faces)

    @staticmethod
    def _extract_colors(mesh, n_verts):
        """Return (n_verts, 4) float32 RGBA array, or None."""
        try:
            visual = mesh.visual
            # TextureVisuals (UV-mapped models) can bake textures to vertex colors.
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
            # Skip if all vertices got the same trimesh default gray.
            if np.allclose(colors[:, :3], colors[0, :3], atol=0.01):
                return None
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

    def draw(self) -> None:
        """Render the mesh. Caller must set up the modelview transform first."""
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self._draw_verts)
        glNormalPointer(GL_FLOAT, 0, self._draw_norms)
        if self._draw_colors is not None:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(4, GL_FLOAT, 0, self._draw_colors)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        if self._draw_colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
