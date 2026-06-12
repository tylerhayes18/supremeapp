"""Full-body pose estimation via the MediaPipe Tasks PoseLandmarker.

Returns 33 body landmarks in both image-normalized and world (meters)
coordinate systems. The world landmarks use a hip-centered origin with
y pointing down and z pointing toward the camera.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task")
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "handsense")
MODEL_PATH = os.path.join(MODEL_CACHE, "pose_landmarker_full.task")

# Landmark indices (MediaPipe Pose, 33 points).
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Bone connections for drawing the skeleton overlay on the camera feed.
POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT_INDEX), (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_FOOT_INDEX), (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
]


@dataclass
class PoseObservation:
    image_landmarks: list[tuple[float, float, float]]   # 33 normalized (x,y,z)
    world_landmarks: list[tuple[float, float, float]]    # 33 meters, hip-centered

    def pixel(self, idx: int, w: int, h: int) -> tuple[int, int]:
        x, y, _ = self.image_landmarks[idx]
        return int(x * w), int(y * h)


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_CACHE, exist_ok=True)
        print(f"downloading pose landmark model -> {MODEL_PATH}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except OSError as exc:
            raise SystemExit(
                f"could not download the pose model ({exc}).\n"
                f"Download manually from\n  {MODEL_URL}\nand save as {MODEL_PATH}"
            ) from exc
    return MODEL_PATH


class PoseDetector:
    def __init__(self, detection_conf: float = 0.5, tracking_conf: float = 0.5):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def process(self, frame_bgr) -> PoseObservation | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        result = self._landmarker.detect_for_video(image, ts)

        if not result.pose_landmarks:
            return None

        img_lms = [(p.x, p.y, p.z) for p in result.pose_landmarks[0]]
        world_lms = [(p.x, p.y, p.z) for p in result.pose_world_landmarks[0]]
        return PoseObservation(image_landmarks=img_lms, world_landmarks=world_lms)

    def draw_skeleton(self, frame, pose: PoseObservation) -> None:
        h, w = frame.shape[:2]
        pts = [pose.pixel(i, w, h) for i in range(33)]
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            color = (80, 220, 100)
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

    def close(self) -> None:
        self._landmarker.close()
