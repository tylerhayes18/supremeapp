"""HandSense application entry point: camera loop, mode switching, HUD."""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from . import hud
from .detector import HandDetector
from .gestures import classify
from .modes import ALL_MODES

WINDOW = "HandSense"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="handsense",
        description="Live hand-gesture recognition and commands from your webcam.",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--max-hands", type=int, default=2, help="hands to track (default 2)")
    parser.add_argument("--mode", type=int, default=1, choices=range(1, len(ALL_MODES) + 1),
                        help="starting mode: 1=command 2=draw 3=rps 4=dial")
    parser.add_argument("--no-mirror", action="store_true",
                        help="disable the selfie-style horizontal flip")
    parser.add_argument("--no-skeleton", action="store_true",
                        help="hide the hand landmark overlay")
    return parser.parse_args(argv)


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"error: could not open camera index {args.camera}. "
                 f"Try --camera 1 (or check OS camera permissions).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def main(argv=None) -> int:
    args = parse_args(argv)
    cap = open_camera(args)
    detector = HandDetector(max_hands=args.max_hands)
    modes = [cls() for cls in ALL_MODES]
    mode_idx = args.mode - 1
    fps = hud.FPSMeter()
    show_skeleton = not args.no_skeleton
    mirror = not args.no_mirror

    print("HandSense running. Keys: 1-4 switch mode, m next mode, "
          "k toggle skeleton, f flip mirror, q/ESC quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("warning: camera frame grab failed, retrying...", file=sys.stderr)
                time.sleep(0.05)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()
            observations = detector.process(frame)
            hands = [(obs, classify(obs.landmarks)) for obs in observations]

            if show_skeleton:
                for obs, _ in hands:
                    detector.draw_skeleton(frame, obs)

            mode = modes[mode_idx]
            mode.update(frame, hands, now)

            # Top status bar.
            hud.panel(frame, 0, 0, frame.shape[1], 68)
            primary = hands[0][1].gesture.value if hands else "no hand"
            label = hud.GESTURE_EMOJI.get(primary, primary)
            hud.text(frame, f"[{mode.name}]  {label}", (12, 26), 0.75, hud.GREEN, 2)
            hud.text(frame, mode.help, (12, 52), 0.5, hud.WHITE)
            hud.text(frame, f"{fps.tick():4.0f} fps", (frame.shape[1] - 110, 26), 0.6, hud.YELLOW)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 = ESC
                break
            elif key == ord("m"):
                mode_idx = (mode_idx + 1) % len(modes)
            elif ord("1") <= key <= ord(str(len(modes))):
                mode_idx = key - ord("1")
            elif key == ord("k"):
                show_skeleton = not show_skeleton
            elif key == ord("f"):
                mirror = not mirror
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
