"""Holistic body+face+hands detection via MediaPipe HolisticLandmarker.

Returns pose (33 body), face (478 mesh), and per-hand (21 each) landmarks
in a single pass, enabling full-body + finger + face-orientation tracking.
"""

from __future__ import annotations

import math
import os
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "holistic_landmarker/holistic_landmarker/float16/latest/"
             "holistic_landmarker.task")
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "handsense")
MODEL_PATH = os.path.join(MODEL_CACHE, "holistic_landmarker.task")

# Pose landmark indices (same as pose_detector.py).
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

# Hand landmark indices (same as gestures.py).
HAND_WRIST = 0
HAND_THUMB = [1, 2, 3, 4]
HAND_INDEX = [5, 6, 7, 8]
HAND_MIDDLE = [9, 10, 11, 12]
HAND_RING = [13, 14, 15, 16]
HAND_PINKY = [17, 18, 19, 20]
HAND_FINGERS = [HAND_THUMB, HAND_INDEX, HAND_MIDDLE, HAND_RING, HAND_PINKY]

# Key face landmark indices for orientation (from the 478-point mesh).
FACE_NOSE_TIP = 1
FACE_FOREHEAD = 10
FACE_CHIN = 152
FACE_LEFT_EYE_OUTER = 33
FACE_RIGHT_EYE_OUTER = 263
FACE_LEFT_EYE_INNER = 133
FACE_RIGHT_EYE_INNER = 362
FACE_LEFT_MOUTH = 61
FACE_RIGHT_MOUTH = 291
FACE_UPPER_LIP = 13
FACE_LOWER_LIP = 14
FACE_LEFT_EYE_TOP = 159
FACE_LEFT_EYE_BOTTOM = 145
FACE_RIGHT_EYE_TOP = 386
FACE_RIGHT_EYE_BOTTOM = 374

# Pose skeleton connections for drawing on camera frame.
POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
    (LEFT_WRIST, LEFT_INDEX), (LEFT_WRIST, LEFT_PINKY), (LEFT_WRIST, LEFT_THUMB),
    (RIGHT_WRIST, RIGHT_INDEX), (RIGHT_WRIST, RIGHT_PINKY), (RIGHT_WRIST, RIGHT_THUMB),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class FaceOrientation:
    yaw: float = 0.0    # left/right rotation (degrees)
    pitch: float = 0.0  # up/down tilt (degrees)
    roll: float = 0.0   # head tilt (degrees)

    # Eye and mouth openness (0=closed, 1=open).
    left_eye_open: float = 1.0
    right_eye_open: float = 1.0
    mouth_open: float = 0.0


@dataclass
class HolisticObservation:
    pose_image: list[tuple[float, float, float]]     # 33 normalized
    pose_world: list[tuple[float, float, float]]      # 33 meters, hip-centered
    face_landmarks: list[tuple[float, float, float]] | None = None   # 478 normalized
    left_hand: list[tuple[float, float, float]] | None = None        # 21 normalized
    right_hand: list[tuple[float, float, float]] | None = None       # 21 normalized
    left_hand_world: list[tuple[float, float, float]] | None = None  # 21 meters
    right_hand_world: list[tuple[float, float, float]] | None = None # 21 meters
    face_orientation: FaceOrientation = field(default_factory=FaceOrientation)

    @property
    def has_face(self) -> bool:
        return self.face_landmarks is not None and len(self.face_landmarks) > 0

    @property
    def has_left_hand(self) -> bool:
        return self.left_hand is not None and len(self.left_hand) > 0

    @property
    def has_right_hand(self) -> bool:
        return self.right_hand is not None and len(self.right_hand) > 0


def _compute_face_orientation(face_lm: list[tuple[float, float, float]]) -> FaceOrientation:
    """Estimate head yaw/pitch/roll from key face mesh landmarks."""
    nose = np.array(face_lm[FACE_NOSE_TIP])
    forehead = np.array(face_lm[FACE_FOREHEAD])
    chin = np.array(face_lm[FACE_CHIN])
    left_eye = np.array(face_lm[FACE_LEFT_EYE_OUTER])
    right_eye = np.array(face_lm[FACE_RIGHT_EYE_OUTER])

    # Yaw: horizontal offset of nose from eye midpoint.
    eye_mid = (left_eye + right_eye) / 2
    eye_dist = max(np.linalg.norm(left_eye[:2] - right_eye[:2]), 1e-6)
    yaw = float(np.degrees(np.arcsin(np.clip((nose[0] - eye_mid[0]) / (eye_dist * 0.8), -1, 1))))

    # Pitch: vertical position of nose relative to forehead-chin axis.
    face_height = max(np.linalg.norm(forehead[:2] - chin[:2]), 1e-6)
    mid_y = (forehead[1] + chin[1]) / 2
    pitch = float(np.degrees(np.arcsin(np.clip((nose[1] - mid_y) / (face_height * 0.4), -1, 1))))

    # Roll: angle of the line between the eyes.
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll = float(np.degrees(np.arctan2(dy, max(abs(dx), 1e-6) * np.sign(dx + 1e-9))))

    # Eye openness: vertical distance between top and bottom eyelid.
    def _eye_openness(top_idx, bot_idx, ref_dist):
        top = np.array(face_lm[top_idx])
        bot = np.array(face_lm[bot_idx])
        return float(np.clip(np.linalg.norm(top[:2] - bot[:2]) / max(ref_dist, 1e-6) * 5, 0, 1))

    le_open = _eye_openness(FACE_LEFT_EYE_TOP, FACE_LEFT_EYE_BOTTOM, eye_dist)
    re_open = _eye_openness(FACE_RIGHT_EYE_TOP, FACE_RIGHT_EYE_BOTTOM, eye_dist)

    # Mouth openness.
    upper = np.array(face_lm[FACE_UPPER_LIP])
    lower = np.array(face_lm[FACE_LOWER_LIP])
    mouth_open = float(np.clip(np.linalg.norm(upper[:2] - lower[:2]) / max(eye_dist, 1e-6) * 3, 0, 1))

    return FaceOrientation(yaw=yaw, pitch=pitch, roll=roll,
                           left_eye_open=le_open, right_eye_open=re_open,
                           mouth_open=mouth_open)


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_CACHE, exist_ok=True)
        print(f"downloading holistic landmark model -> {MODEL_PATH}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except OSError as exc:
            raise SystemExit(
                f"could not download the holistic model ({exc}).\n"
                f"Download manually from\n  {MODEL_URL}\nand save as {MODEL_PATH}"
            ) from exc
    return MODEL_PATH


class HolisticDetector:
    def __init__(self, detection_conf: float = 0.5, tracking_conf: float = 0.5):
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=detection_conf,
            min_pose_landmarks_confidence=tracking_conf,
            min_hand_landmarks_confidence=0.4,
            min_face_landmarks_confidence=0.4,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def process(self, frame_bgr) -> HolisticObservation | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        result = self._landmarker.detect_for_video(image, ts)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        pose_img = [(p.x, p.y, p.z) for p in result.pose_landmarks]
        pose_world = [(p.x, p.y, p.z) for p in result.pose_world_landmarks]

        face_lm = None
        face_orient = FaceOrientation()
        if result.face_landmarks and len(result.face_landmarks) > 0:
            face_lm = [(p.x, p.y, p.z) for p in result.face_landmarks]
            face_orient = _compute_face_orientation(face_lm)

        def _hand_lms(lms):
            if lms and len(lms) > 0:
                return [(p.x, p.y, p.z) for p in lms]
            return None

        return HolisticObservation(
            pose_image=pose_img,
            pose_world=pose_world,
            face_landmarks=face_lm,
            left_hand=_hand_lms(result.left_hand_landmarks),
            right_hand=_hand_lms(result.right_hand_landmarks),
            left_hand_world=_hand_lms(result.left_hand_world_landmarks),
            right_hand_world=_hand_lms(result.right_hand_world_landmarks),
            face_orientation=face_orient,
        )

    def draw_overlay(self, frame, obs: HolisticObservation) -> None:
        """Draw pose skeleton + hand skeletons + face mesh on camera frame."""
        h, w = frame.shape[:2]

        def px(lm):
            return int(lm[0] * w), int(lm[1] * h)

        # Pose skeleton.
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame, px(obs.pose_image[a]), px(obs.pose_image[b]),
                     (180, 180, 180), 2, cv2.LINE_AA)
        for i in range(33):
            cv2.circle(frame, px(obs.pose_image[i]), 3, (80, 220, 100), -1, cv2.LINE_AA)

        # Hands.
        for hand_lm in [obs.left_hand, obs.right_hand]:
            if hand_lm is None:
                continue
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, px(hand_lm[a]), px(hand_lm[b]),
                         (100, 200, 255), 1, cv2.LINE_AA)
            for lm in hand_lm:
                cv2.circle(frame, px(lm), 2, (60, 60, 230), -1, cv2.LINE_AA)

        # Face mesh (sparse — just key contours).
        if obs.has_face:
            face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                            288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                            150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                            67, 109, 10]
            for i in range(len(face_contour) - 1):
                cv2.line(frame, px(obs.face_landmarks[face_contour[i]]),
                         px(obs.face_landmarks[face_contour[i + 1]]),
                         (200, 180, 100), 1, cv2.LINE_AA)

        # Face orientation text.
        fo = obs.face_orientation
        cv2.putText(frame, f"yaw:{fo.yaw:+.0f} pitch:{fo.pitch:+.0f} roll:{fo.roll:+.0f}",
                    (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def close(self) -> None:
        self._landmarker.close()
