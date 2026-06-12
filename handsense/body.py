"""Body tracking entry point: full-body + face + hands mirrored onto a 3D avatar.

Usage:
    python -m handsense.body [--camera N] [--width W] [--height H]

Opens a 3D OpenGL window showing a humanoid avatar that mirrors your body,
face orientation, and finger movements in real time. Camera feed appears as
a picture-in-picture with the detected skeleton overlaid.

Controls:
    arrow keys   orbit the 3D camera around the avatar
    +/-          zoom in/out
    r            reset camera angle
    q / Esc      quit
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from .avatar import AvatarRenderer
from .holistic_detector import HolisticDetector


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="handsense.body",
        description="Full-body + face + hand tracking with a real-time 3D avatar.",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--no-mirror", action="store_true", help="disable selfie-style mirror")
    parser.add_argument("--avatar-size", type=int, default=900,
                        help="3D window size in pixels (default 900)")
    parser.add_argument("--model", type=str, default=None,
                        help="path to a .glb/.vrm humanoid model (downloads a default if omitted)")
    parser.add_argument("--no-mesh", action="store_true",
                        help="use geometric avatar instead of a 3D mesh")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"error: could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = HolisticDetector()
    avatar = AvatarRenderer(width=args.avatar_size, height=args.avatar_size,
                            model_path=args.model, use_mesh=not args.no_mesh)
    mirror = not args.no_mirror

    print("Body tracking running.")
    print("  Stand in T-POSE (arms horizontal) to calibrate the mesh.")
    print("  arrow keys = orbit | +/- = zoom | r = reset view | q = quit")
    print("  Tracking: body (33), face (478), hands (21 each)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            obs = detector.process(frame)

            if obs is not None:
                detector.draw_overlay(frame, obs)

            if not avatar.handle_events():
                break

            avatar.render(obs=obs, camera_frame=frame)
    finally:
        detector.close()
        cap.release()
        avatar.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
