"""Hand tracking via the MediaPipe Tasks HandLandmarker.

Uses the Tasks API (the legacy `mp.solutions` API was removed from recent
MediaPipe releases). The landmark model (~8 MB) is downloaded once to
~/.cache/handsense/ on first run.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "handsense")
MODEL_PATH = os.path.join(MODEL_CACHE, "hand_landmarker.task")

# Landmark index pairs forming the hand skeleton.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
)


@dataclass
class HandObservation:
    landmarks: list[tuple[float, float, float]]  # 21 normalized (x, y, z)
    handedness: str  # "Left" or "Right" (after mirror flip, matches the user)

    def pixel(self, idx: int, width: int, height: int) -> tuple[int, int]:
        x, y, _ = self.landmarks[idx]
        return int(x * width), int(y * height)

    def bbox(self, width: int, height: int, pad: float = 0.1) -> tuple[int, int, int, int]:
        """Pixel bounding box (x1, y1, x2, y2) around the hand, padded."""
        xs = [p[0] for p in self.landmarks]
        ys = [p[1] for p in self.landmarks]
        x1 = max(0, int((min(xs) - pad) * width))
        y1 = max(0, int((min(ys) - pad) * height))
        x2 = min(width - 1, int((max(xs) + pad) * width))
        y2 = min(height - 1, int((max(ys) + pad) * height))
        return x1, y1, x2, y2


def _ensure_model(model_path: str | None) -> str:
    if model_path:
        return model_path
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_CACHE, exist_ok=True)
        print(f"downloading hand landmark model -> {MODEL_PATH}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except OSError as exc:
            raise SystemExit(
                f"could not download the hand landmark model ({exc}).\n"
                f"Download it manually from\n  {MODEL_URL}\nand save it as {MODEL_PATH}"
            ) from exc
    return MODEL_PATH


class HandDetector:
    def __init__(self, max_hands: int = 2, detection_conf: float = 0.6,
                 tracking_conf: float = 0.5, model_path: str | None = None):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def process(self, frame_bgr) -> list[HandObservation]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        result = self._landmarker.detect_for_video(image, ts)

        observations: list[HandObservation] = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            observations.append(
                HandObservation(
                    landmarks=[(p.x, p.y, p.z) for p in lms],
                    handedness=handed[0].category_name,
                )
            )
        return observations

    def draw_skeleton(self, frame_bgr, observation: HandObservation) -> None:
        h, w = frame_bgr.shape[:2]
        points = [observation.pixel(i, w, h) for i in range(21)]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[a], points[b], (200, 200, 200), 2, cv2.LINE_AA)
        for i, pt in enumerate(points):
            color = (60, 60, 230) if i in (4, 8, 12, 16, 20) else (80, 220, 100)
            cv2.circle(frame_bgr, pt, 4, color, -1, cv2.LINE_AA)

    def close(self) -> None:
        self._landmarker.close()
