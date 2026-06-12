"""Headless software renderer for STL previews and sim snapshots.

No GPU/display needed: a small numpy z-buffer rasterizer with Lambert
shading.  Used by ``python -m skyarm.render`` to make part previews and
by the simulator's --snapshot mode.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BG = np.array([24, 26, 32], dtype=np.float64)


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f


def render_meshes(meshes, size=(900, 700), eye=None, target=None,
                  fov_deg=40.0, light=(0.4, -0.6, 0.8)):
    """meshes: list of (vertices Nx3, faces Mx3, rgb tuple). Returns PIL Image."""
    w, h = size
    all_v = np.vstack([v for v, _, _ in meshes])
    center = (all_v.min(0) + all_v.max(0)) / 2
    radius = float(np.linalg.norm(all_v - center, axis=1).max())
    if target is None:
        target = center
    if eye is None:
        d = radius / math.tan(math.radians(fov_deg / 2)) * 1.15
        eye = center + np.array([0.55, -1.0, 0.55]) / 1.27 * d

    eye, r, u, f = look_at(eye, target)
    light = np.asarray(light, float)
    light = light / np.linalg.norm(light)

    img = np.tile(BG, (h, w, 1))
    zbuf = np.full((h, w), np.inf)
    focal = (h / 2) / math.tan(math.radians(fov_deg / 2))

    for verts, faces, color in meshes:
        color = np.asarray(color, float)
        rel = verts - eye
        cx = rel @ r
        cy = rel @ u
        cz = rel @ f
        cz = np.maximum(cz, 1e-6)
        sx = w / 2 + focal * cx / cz
        sy = h / 2 - focal * cy / cz

        tri = faces
        v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(n, axis=1)
        ok = nl > 1e-12
        n[ok] = n[ok] / nl[ok, None]
        shade = 0.25 + 0.75 * np.clip(n @ light, 0, None)
        # backface midpoint depth ordering handled by z-buffer per pixel

        p0 = np.stack([sx[tri[:, 0]], sy[tri[:, 0]], cz[tri[:, 0]]], 1)
        p1 = np.stack([sx[tri[:, 1]], sy[tri[:, 1]], cz[tri[:, 1]]], 1)
        p2 = np.stack([sx[tri[:, 2]], sy[tri[:, 2]], cz[tri[:, 2]]], 1)

        for i in range(len(tri)):
            a, b, c = p0[i], p1[i], p2[i]
            xmin = max(int(min(a[0], b[0], c[0])), 0)
            xmax = min(int(max(a[0], b[0], c[0])) + 1, w)
            ymin = max(int(min(a[1], b[1], c[1])), 0)
            ymax = min(int(max(a[1], b[1], c[1])) + 1, h)
            if xmin >= xmax or ymin >= ymax:
                continue
            xs, ys = np.meshgrid(np.arange(xmin, xmax) + 0.5,
                                 np.arange(ymin, ymax) + 0.5)
            d = ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
            if abs(d) < 1e-12:
                continue
            w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
            w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            z = w0 * a[2] + w1 * b[2] + w2 * c[2]
            sub_z = zbuf[ymin:ymax, xmin:xmax]
            upd = inside & (z < sub_z)
            sub_z[upd] = z[upd]
            img[ymin:ymax, xmin:xmax][upd] = color * shade[i]

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


PALETTE = [(235, 137, 52), (90, 170, 235), (140, 220, 140), (230, 110, 110),
           (200, 180, 90), (170, 130, 220), (120, 210, 200), (220, 220, 220)]


def render_stl(path, out_png, color=(235, 137, 52), size=(700, 560)):
    import trimesh
    m = trimesh.load(path)
    img = render_meshes([(m.vertices, m.faces, color)], size=size)
    img.save(out_png)
    return out_png


def contact_sheet(stl_dir, out_png, cols=6, cell=360):
    import trimesh
    from PIL import ImageDraw
    paths = sorted(Path(stl_dir).glob("*.stl"))
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), tuple(BG.astype(int)))
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        m = trimesh.load(p)
        img = render_meshes([(m.vertices, m.faces, PALETTE[i % len(PALETTE)])],
                            size=(cell, cell - 24))
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.paste(img, (x, y))
        draw.text((x + 8, y + cell - 20), p.stem, fill=(230, 230, 230))
    sheet.save(out_png)
    return out_png


def exploded_gearbox(out_png, spec_name="cyclo-big"):
    """Exploded view of one cycloidal reducer along +z."""
    from . import spec as S
    from .cad import cycloidal
    cs = {"cyclo-big": S.CYCLO_BIG, "cyclo-mid": S.CYCLO_MID,
          "cyclo-yaw": S.CYCLO_YAW, "cyclo-small": S.CYCLO_SMALL}[spec_name]
    parts = cycloidal.parts(cs)
    offsets = {"_housing": 0, "_cam": 55, "_disc": 100, "_output": 165, "_cover": 235}
    meshes = []
    for i, (name, mesh) in enumerate(parts.items()):
        for suffix, dz in offsets.items():
            if name.endswith(suffix):
                mesh.apply_translation((0, 0, dz))
                if suffix == "_disc":   # show both discs, phased
                    m2 = mesh.copy()
                    m2.apply_translation((0, 0, 30))
                    meshes.append((m2.vertices, m2.faces, PALETTE[(i + 3) % 8]))
        meshes.append((mesh.vertices, mesh.faces, PALETTE[i % 8]))
    img = render_meshes(meshes, size=(900, 1100), fov_deg=35)
    img.save(out_png)
    return out_png


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "previews"
    out.mkdir(exist_ok=True)
    stl = Path(__file__).resolve().parent / "stl"
    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        print(contact_sheet(stl, out / "all_parts.png"))
    elif len(sys.argv) > 1 and sys.argv[1] == "gearbox":
        print(exploded_gearbox(out / "cyclo_big_exploded.png"))
    else:
        print(contact_sheet(stl, out / "all_parts.png"))
        print(exploded_gearbox(out / "cyclo_big_exploded.png"))
