"""SkyArm 3D test environment entry point.

    python -m skyarm.sim                     interactive OpenGL viewer
    python -m skyarm.sim --demo              viewer, demo mission running
    python -m skyarm.sim --headless          run demo mission, print report
    python -m skyarm.sim --snapshot PNG      render mission keyframes to files
"""

from __future__ import annotations

import argparse
import sys

from .. import kinematics as kin
from .world import World, demo_mission


def headless_report() -> int:
    world = demo_mission(World())
    world.run()
    tips = [t for _, t in world.trace]
    zmin = min(p[2] for p in tips)
    print(f"demo mission complete in {world.time:.1f} s simulated")
    print(f"trace samples: {len(world.trace)}, faults: {len(world.faults)}")
    print(f"lowest tip point: {zmin:.0f} mm above floor "
          f"(5 ft reach => {kin.CEILING_MM - 1524.0:.0f} mm)")
    print(f"final tip: {tuple(round(v) for v in world.chain().tip)}")
    return 0 if not world.faults else 1


def snapshots(base: str, real: bool = False) -> int:
    from . import scene
    world = demo_mission(World())
    stem = base[:-4] if base.endswith(".png") else base
    frames = []
    i = 0
    every = 4.0 if real else 2.0   # real CAD geometry renders are slower
    size = (960, 720) if real else (1100, 800)
    while (world.mission or not world.settled) and world.time < 600:
        world.run(seconds=every)
        path = f"{stem}_{i:02d}.png"
        scene.snapshot(world, path, size=size, real_geometry=real)
        frames.append(path)
        i += 1
    print("\n".join(frames))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--snapshot", metavar="PNG")
    ap.add_argument("--real", action="store_true",
                    help="snapshot with real CAD assembly meshes")
    args = ap.parse_args()
    if args.snapshot:
        return snapshots(args.snapshot, real=args.real)
    if args.headless:
        return headless_report()
    from .viewer import run_viewer
    return run_viewer(demo=args.demo)


if __name__ == "__main__":
    sys.exit(main())
