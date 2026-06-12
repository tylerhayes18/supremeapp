"""Body tracking entry point: full-body pose mirrored onto a 3D avatar.

Usage:
    python -m handsense.body [--camera N] [--width W] [--height H]

Opens two views: a camera feed (picture-in-picture in the corner) and a 3D
OpenGL window showing a geometric humanoid that mirrors your movements in
real time.

Controls:
    arrow keys   orbit the 3D camera around the avatar
    +/-          zoom in/out
    q / Esc      quit
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from .avatar import AvatarRenderer
from .pose_detector import PoseDetector


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="handsense.body",
        description="Full-body tracking with a real-time 3D avatar.",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--no-mirror", action="store_true", help="disable selfie-style mirror")
    parser.add_argument("--avatar-size", type=int, default=800,
                        help="3D window size in pixels (default 800)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"error: could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = PoseDetector()
    avatar = AvatarRenderer(width=args.avatar_size, height=args.avatar_size)
    mirror = not args.no_mirror

    print("Body tracking running. Arrow keys orbit camera, +/- zoom, q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            pose = detector.process(frame)

            # Draw pose skeleton on the camera frame (for the PIP).
            if pose is not None:
                detector.draw_skeleton(frame, pose)

            if not avatar.handle_events():
                break

            avatar.render(
                world_landmarks=pose.world_landmarks if pose else None,
                camera_frame=frame,
            )
    finally:
        detector.close()
        cap.release()
        avatar.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
