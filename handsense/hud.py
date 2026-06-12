"""On-screen HUD rendering helpers (panels, text, event log, banners)."""

from __future__ import annotations

import time

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (80, 220, 100)
RED = (60, 60, 230)
YELLOW = (60, 200, 255)
CYAN = (230, 200, 60)
MAGENTA = (200, 80, 220)

GESTURE_EMOJI = {
    "thumbs_up": "THUMBS UP",
    "thumbs_down": "THUMBS DOWN",
    "middle_finger": "MIDDLE FINGER",
    "peace": "PEACE",
    "fist": "FIST",
    "open_palm": "OPEN PALM",
    "pointing": "POINTING",
    "ok_sign": "OK",
    "rock_on": "ROCK ON",
    "shaka": "SHAKA",
    "three": "THREE",
    "four": "FOUR",
    "unknown": "--",
}


def panel(frame, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    """Translucent dark rectangle for text backdrops."""
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x >= x2 or y >= y2:
        return
    roi = frame[y:y2, x:x2]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, alpha, roi, 1 - alpha, 0, dst=roi)


def text(frame, msg: str, org: tuple[int, int], scale: float = 0.6,
         color=WHITE, thickness: int = 1) -> None:
    cv2.putText(frame, msg, org, FONT, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, msg, org, FONT, scale, color, thickness, cv2.LINE_AA)


def banner(frame, msg: str, color=GREEN) -> None:
    """Large centered announcement (round results, countdowns...)."""
    h, w = frame.shape[:2]
    scale = 1.6
    (tw, th), _ = cv2.getTextSize(msg, FONT, scale, 3)
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(frame, msg, org, FONT, scale, BLACK, 7, cv2.LINE_AA)
    cv2.putText(frame, msg, org, FONT, scale, color, 3, cv2.LINE_AA)


class EventLog:
    """Rolling log of triggered actions, rendered bottom-left."""

    def __init__(self, max_items: int = 5, ttl_s: float = 6.0):
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._items: list[tuple[float, str, tuple]] = []

    def add(self, msg: str, color=WHITE) -> None:
        self._items.append((time.time(), msg, color))
        self._items = self._items[-self.max_items:]

    def render(self, frame) -> None:
        now = time.time()
        live = [it for it in self._items if now - it[0] < self.ttl_s]
        if not live:
            return
        h = frame.shape[0]
        panel(frame, 8, h - 24 * len(live) - 16, 360, 24 * len(live) + 8)
        for i, (_, msg, color) in enumerate(reversed(live)):
            text(frame, msg, (16, h - 16 - 24 * i), 0.55, color)


class FPSMeter:
    def __init__(self, smoothing: float = 0.9):
        self._smoothing = smoothing
        self._fps = 0.0
        self._last = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = max(now - self._last, 1e-6)
        self._last = now
        inst = 1.0 / dt
        self._fps = self._fps * self._smoothing + inst * (1 - self._smoothing) if self._fps else inst
        return self._fps
