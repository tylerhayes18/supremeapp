"""Small CSG toolkit on top of trimesh + manifold3d.

All parts are built from these helpers so every mesh goes through the
manifold boolean backend and comes out watertight and printable.

Conventions: dimensions in mm, parts modelled in their natural print
orientation (z = build direction) whenever possible.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

# shapely is only needed when BUILDING parts (2D profile extrusion); the
# viewer loads pre-generated STLs instead, so keep the import lazy

ENGINE = "manifold"
SEG = 96          # default circle segments


def union(*meshes: trimesh.Trimesh) -> trimesh.Trimesh:
    meshes = [m for m in meshes if m is not None]
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.boolean.union(meshes, engine=ENGINE)


def difference(base: trimesh.Trimesh, *cutters: trimesh.Trimesh) -> trimesh.Trimesh:
    if not cutters:
        return base
    cut = union(*cutters) if len(cutters) > 1 else cutters[0]
    return trimesh.boolean.difference([base, cut], engine=ENGINE)


def box(sx: float, sy: float, sz: float, at=(0, 0, 0)) -> trimesh.Trimesh:
    """Axis-aligned box centred in x/y, sitting on z=0, then moved by ``at``."""
    m = trimesh.creation.box((sx, sy, sz))
    m.apply_translation((at[0], at[1], at[2] + sz / 2))
    return m


def cyl(d: float, h: float, at=(0, 0, 0), seg: int = SEG) -> trimesh.Trimesh:
    """Z-axis cylinder of diameter ``d`` sitting on z=0, moved by ``at``."""
    m = trimesh.creation.cylinder(radius=d / 2, height=h, sections=seg)
    m.apply_translation((at[0], at[1], at[2] + h / 2))
    return m


def tube(od: float, id_: float, h: float, at=(0, 0, 0), seg: int = SEG):
    return difference(cyl(od, h, at, seg), cyl(id_, h + 2, (at[0], at[1], at[2] - 1), seg))


def cyl_x(d: float, h: float, at=(0, 0, 0), seg: int = SEG) -> trimesh.Trimesh:
    """Cylinder along +x, starting at ``at``."""
    m = trimesh.creation.cylinder(radius=d / 2, height=h, sections=seg)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0)))
    m.apply_translation((at[0] + h / 2, at[1], at[2]))
    return m


def cyl_y(d: float, h: float, at=(0, 0, 0), seg: int = SEG) -> trimesh.Trimesh:
    """Cylinder along +y, starting at ``at``."""
    m = trimesh.creation.cylinder(radius=d / 2, height=h, sections=seg)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))
    m.apply_translation((at[0], at[1] + h / 2, at[2]))
    return m


def rounded_plate(sx: float, sy: float, t: float, r: float = 6.0,
                  at=(0, 0, 0)) -> trimesh.Trimesh:
    """Rounded-corner plate centred in x/y on z=0."""
    from shapely.geometry import Polygon
    poly = Polygon([(-sx / 2, -sy / 2), (sx / 2, -sy / 2),
                    (sx / 2, sy / 2), (-sx / 2, sy / 2)])
    poly = poly.buffer(-r).buffer(r, quad_segs=8)
    m = trimesh.creation.extrude_polygon(poly, t)
    m.apply_translation(at)
    return m


def extrude(points, t: float, holes=(), at=(0, 0, 0)) -> trimesh.Trimesh:
    from shapely.geometry import Polygon
    poly = Polygon(points, holes=[list(h) for h in holes])
    m = trimesh.creation.extrude_polygon(poly, t)
    m.apply_translation(at)
    return m


def holes(d: float, t: float, centers, z: float = -1.0, seg: int = 48):
    """Cylindrical cutters of diameter d, length t+2, at xy centers."""
    return [cyl(d, t + 2, (cx, cy, z), seg) for cx, cy in centers]


def bolt_circle(n: int, r: float, start_deg: float = 0.0):
    return [(r * math.cos(math.radians(start_deg + i * 360 / n)),
             r * math.sin(math.radians(start_deg + i * 360 / n)))
            for i in range(n)]


def counterbored_holes(base, centers, t, d_thread=4.4, d_head=8.4, head_h=4.0):
    """Through holes with counterbores sunk from the top face (z = t)."""
    cutters = holes(d_thread, t, centers)
    cutters += [cyl(d_head, head_h + 1, (cx, cy, t - head_h)) for cx, cy in centers]
    return difference(base, *cutters)


# Motor interface dimensions: faceplate bolt square, bolt thread, pilot
# boss diameter, shaft diameter.
NEMA = {
    17: dict(square=31.0, bolt_d=3.4, boss_d=22.3, shaft_d=5.0, face=42.3),
    23: dict(square=47.14, bolt_d=5.2, boss_d=38.4, shaft_d=8.0, face=57.0),
    24: dict(square=47.14, bolt_d=5.2, boss_d=38.4, shaft_d=8.0, face=60.0),
    34: dict(square=69.6, bolt_d=6.6, boss_d=73.4, shaft_d=14.0, face=86.0),
}


def nema_bolt_centers(size: int):
    s = NEMA[size]["square"] / 2
    return [(-s, -s), (s, -s), (s, s), (-s, s)]


def dshaft_cutter(d: float, flat_depth: float, h: float, at=(0, 0, 0)):
    """Cutter for a D-profile shaft bore (flat on -x side)."""
    shaft = cyl(d, h, at)
    if flat_depth <= 0:
        return shaft
    flat = box(flat_depth * 2, d + 2, h, (at[0] - d / 2, at[1], at[2]))
    return difference(shaft, flat)


def shaft_flat_depth(d: float) -> float:
    """Typical D-shaft flat depth for a given shaft diameter."""
    if d <= 5.5:
        return 0.5
    if d >= 12.0:
        return 1.5
    return 1.0


def check_part(mesh: trimesh.Trimesh, name: str, max_xyz=(250.0, 250.0, 250.0)):
    """Validate a part is printable; raises on failure."""
    if not mesh.is_watertight:
        raise ValueError(f"{name}: mesh is not watertight")
    if mesh.volume <= 0:
        raise ValueError(f"{name}: non-positive volume")
    ext = mesh.extents
    for e, lim, ax in zip(ext, max_xyz, "xyz"):
        if e > lim:
            raise ValueError(f"{name}: {ax} extent {e:.1f} exceeds {lim} mm bed")
    return mesh
