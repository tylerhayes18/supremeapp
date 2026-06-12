"""Interactive modes. Each mode consumes per-frame hand observations and
renders its own UI onto the frame.

Mode interface:
    update(frame, hands, now) -> None
        frame: BGR image (mutated in place)
        hands: list of (HandObservation, GestureResult) pairs
        now:   time.time() timestamp for this frame
"""

from __future__ import annotations

import math
import os
import random
import time

import cv2
import numpy as np

from . import hud
from .gestures import (
    INDEX_TIP,
    Gesture,
    GestureStabilizer,
)


class CommandMode:
    """Gestures fire one-shot commands, shown in a rolling event log.

    thumbs up      -> "like" (with a brief green flash)
    thumbs down    -> "dislike" (red flash)
    middle finger  -> hand gets pixelated + rudeness counter increments
    peace          -> 3-2-1 countdown, then saves a snapshot to ./snapshots
    ok sign        -> friendly acknowledgement
    rock on        -> crowd goes wild
    shaka          -> hang loose
    open palm (hold) -> pause/resume command processing
    """

    name = "COMMAND"
    help = "thumbs up/down, middle finger, OK, rock-on, shaka | peace=photo | palm=pause"

    def __init__(self, snapshot_dir: str = "snapshots"):
        self.stabilizer = GestureStabilizer(hold_frames=8, cooldown_s=1.5)
        self.log = hud.EventLog()
        self.snapshot_dir = snapshot_dir
        self.rudeness = 0
        self.likes = 0
        self.paused = False
        self._flash: tuple[float, tuple] | None = None  # (until, color)
        self._countdown_until: float | None = None

    def update(self, frame, hands, now: float) -> None:
        gesture = hands[0][1].gesture if hands else Gesture.UNKNOWN
        self.stabilizer.update(gesture)

        # Censor a rude hand every frame it is visible, even while paused.
        if hands and gesture == Gesture.MIDDLE_FINGER:
            self._pixelate_hand(frame, hands[0][0])

        # Snapshot countdown runs to completion regardless of gesture changes.
        if self._countdown_until is not None:
            remaining = self._countdown_until - now
            if remaining > 0:
                hud.banner(frame, str(math.ceil(remaining)), hud.YELLOW)
            else:
                self._save_snapshot(frame)
                self._countdown_until = None
        else:
            fired = self.stabilizer.trigger(now)
            if fired is not None:
                self._handle(fired, now)

        if self._flash and now < self._flash[0]:
            overlay = np.full_like(frame, self._flash[1], dtype=np.uint8)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)

        status = f"likes {self.likes}  |  rudeness {self.rudeness}"
        if self.paused:
            status += "  |  PAUSED (open palm to resume)"
        hud.text(frame, status, (12, 92), 0.55, hud.CYAN)
        self.log.render(frame)

    def _handle(self, gesture: Gesture, now: float) -> None:
        if gesture == Gesture.OPEN_PALM:
            self.paused = not self.paused
            self.log.add("paused" if self.paused else "resumed", hud.YELLOW)
            return
        if self.paused:
            return

        if gesture == Gesture.THUMBS_UP:
            self.likes += 1
            self._flash = (now + 0.4, hud.GREEN)
            self.log.add("+1 like — thanks!", hud.GREEN)
        elif gesture == Gesture.THUMBS_DOWN:
            self.likes -= 1
            self._flash = (now + 0.4, hud.RED)
            self.log.add("-1 ... noted.", hud.RED)
        elif gesture == Gesture.MIDDLE_FINGER:
            self.rudeness += 1
            self.log.add(f"RUDE! (offense #{self.rudeness}) — censored", hud.RED)
        elif gesture == Gesture.PEACE:
            self._countdown_until = now + 3.0
            self.log.add("photo in 3...", hud.YELLOW)
        elif gesture == Gesture.OK_SIGN:
            self.log.add("OK! Acknowledged.", hud.CYAN)
        elif gesture == Gesture.ROCK_ON:
            self.log.add("\\m/ the crowd goes wild", hud.MAGENTA)
        elif gesture == Gesture.SHAKA:
            self.log.add("hang loose, brother", hud.CYAN)
        elif gesture == Gesture.POINTING:
            self.log.add("hey, it's rude to point", hud.YELLOW)

    def _pixelate_hand(self, frame, observation) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = observation.bbox(w, h, pad=0.05)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        small = cv2.resize(roi, (max(1, (x2 - x1) // 16), max(1, (y2 - y1) // 16)))
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    def _save_snapshot(self, frame) -> None:
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(path, frame)
        self._flash = (time.time() + 0.25, hud.WHITE)
        self.log.add(f"saved {path}", hud.GREEN)


class DrawMode:
    """Air-drawing: point with your index finger to draw.

    pointing   -> draw a stroke following the fingertip
    pinch      -> brush size follows thumb-index distance
    peace      -> cycle brush color
    open palm  -> hold to clear the canvas
    """

    name = "DRAW"
    help = "point=draw | pinch=brush size | peace=color | hold palm=clear"

    COLORS = [hud.GREEN, hud.CYAN, hud.MAGENTA, hud.YELLOW, hud.WHITE, hud.RED]

    def __init__(self):
        self.stabilizer = GestureStabilizer(hold_frames=5, cooldown_s=0.8)
        self.canvas = None  # lazily sized to the frame
        self.color_idx = 0
        self.brush = 6
        self._last_point: tuple[int, int] | None = None
        self._clear_started: float | None = None

    def update(self, frame, hands, now: float) -> None:
        h, w = frame.shape[:2]
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros_like(frame)

        gesture = hands[0][1].gesture if hands else Gesture.UNKNOWN
        stable = self.stabilizer.update(gesture)

        if hands and gesture == Gesture.POINTING:
            point = hands[0][0].pixel(INDEX_TIP, w, h)
            if self._last_point is not None:
                cv2.line(self.canvas, self._last_point, point, self.COLORS[self.color_idx],
                         self.brush, cv2.LINE_AA)
            self._last_point = point
            cv2.circle(frame, point, self.brush + 4, self.COLORS[self.color_idx], 2, cv2.LINE_AA)
        else:
            self._last_point = None  # pen up

        if hands:
            # Pinch continuously adjusts brush size whenever visible.
            pinch = hands[0][1].pinch
            self.brush = int(np.clip(2 + pinch * 14, 2, 40))

        fired = self.stabilizer.trigger(now)
        if fired == Gesture.PEACE:
            self.color_idx = (self.color_idx + 1) % len(self.COLORS)

        # Clearing is destructive, so require the palm held for a full second.
        if stable == Gesture.OPEN_PALM and gesture == Gesture.OPEN_PALM:
            if self._clear_started is None:
                self._clear_started = now
            held = now - self._clear_started
            hud.banner(frame, f"clearing in {max(0.0, 1.0 - held):.1f}", hud.RED)
            if held >= 1.0:
                self.canvas[:] = 0
                self._clear_started = None
        else:
            self._clear_started = None

        # Composite the ink over the camera image.
        mask = self.canvas.any(axis=2)
        frame[mask] = cv2.addWeighted(self.canvas, 0.9, frame, 0.1, 0)[mask]

        sw = self.COLORS[self.color_idx]
        cv2.circle(frame, (w - 40, 40), self.brush, sw, -1, cv2.LINE_AA)
        cv2.circle(frame, (w - 40, 40), self.brush + 3, hud.WHITE, 1, cv2.LINE_AA)


class RPSMode:
    """Rock-paper-scissors against the computer.

    thumbs up -> start a round; after a 3s countdown your pose is read:
    fist=rock, open palm=paper, peace=scissors. First to 3 wins gloats.
    """

    name = "RPS"
    help = "thumbs up=start round | fist=rock, palm=paper, peace=scissors"

    POSES = {Gesture.FIST: "rock", Gesture.OPEN_PALM: "paper",
             Gesture.FOUR: "paper", Gesture.PEACE: "scissors"}
    BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}

    def __init__(self):
        self.stabilizer = GestureStabilizer(hold_frames=6, cooldown_s=1.0)
        self.you = 0
        self.cpu = 0
        self._state = "idle"  # idle -> countdown -> result
        self._deadline = 0.0
        self._result_msg = ""
        self._result_color = hud.WHITE

    def update(self, frame, hands, now: float) -> None:
        gesture = hands[0][1].gesture if hands else Gesture.UNKNOWN
        self.stabilizer.update(gesture)

        hud.text(frame, f"YOU {self.you}  vs  CPU {self.cpu}", (12, 92), 0.7, hud.CYAN)

        if self._state == "idle":
            hud.text(frame, "thumbs up to play", (12, 118), 0.55, hud.WHITE)
            if self.stabilizer.trigger(now) == Gesture.THUMBS_UP:
                self._state = "countdown"
                self._deadline = now + 3.0
        elif self._state == "countdown":
            remaining = self._deadline - now
            if remaining > 0:
                hud.banner(frame, f"SHOOT IN {math.ceil(remaining)}", hud.YELLOW)
            else:
                self._judge(gesture, now)
        elif self._state == "result":
            hud.banner(frame, self._result_msg, self._result_color)
            if now > self._deadline:
                if self.you >= 3 or self.cpu >= 3:
                    winner = "YOU WIN THE MATCH!" if self.you >= 3 else "CPU TAKES THE MATCH"
                    self._result_msg, self._result_color = winner, hud.MAGENTA
                    self._deadline = now + 2.5
                    self.you = self.cpu = 0
                    self._state = "matchover"
                else:
                    self._state = "idle"
        elif self._state == "matchover":
            hud.banner(frame, self._result_msg, self._result_color)
            if now > self._deadline:
                self._state = "idle"

    def _judge(self, gesture: Gesture, now: float) -> None:
        yours = self.POSES.get(gesture)
        cpu = random.choice(list(self.BEATS))
        if yours is None:
            self._result_msg, self._result_color = "no valid pose — round void", hud.YELLOW
        elif yours == cpu:
            self._result_msg, self._result_color = f"both {yours} — draw", hud.WHITE
        elif self.BEATS[yours] == cpu:
            self.you += 1
            self._result_msg, self._result_color = f"{yours} beats {cpu} — you score!", hud.GREEN
        else:
            self.cpu += 1
            self._result_msg, self._result_color = f"{cpu} beats {yours} — cpu scores", hud.RED
        self._state = "result"
        self._deadline = now + 2.0


class DialMode:
    """A virtual dial driven by your pinch distance — a stand-in for anything
    continuous you might wire it to (volume, brightness, zoom...)."""

    name = "DIAL"
    help = "pinch thumb+index to turn the dial | fist=reset"

    def __init__(self):
        self.value = 50.0
        self._smoothed: float | None = None

    def update(self, frame, hands, now: float) -> None:
        if hands:
            result = hands[0][1]
            if result.gesture == Gesture.FIST:
                self.value, self._smoothed = 50.0, None
            else:
                # Map pinch ~[0.2, 1.8] -> [0, 100], with EMA smoothing.
                target = float(np.clip((result.pinch - 0.2) / 1.6 * 100, 0, 100))
                self._smoothed = target if self._smoothed is None else \
                    0.8 * self._smoothed + 0.2 * target
                self.value = self._smoothed

        h, w = frame.shape[:2]
        cx, cy, radius = w // 2, h // 2, min(w, h) // 5
        cv2.circle(frame, (cx, cy), radius, hud.WHITE, 2, cv2.LINE_AA)
        # Sweep arc from 135 deg to 405 deg (a classic 270-degree dial).
        sweep = 135 + 270 * self.value / 100
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 135, sweep, hud.GREEN, 6, cv2.LINE_AA)
        angle = math.radians(sweep)
        tip = (int(cx + radius * 0.8 * math.cos(angle)), int(cy + radius * 0.8 * math.sin(angle)))
        cv2.line(frame, (cx, cy), tip, hud.YELLOW, 3, cv2.LINE_AA)
        hud.text(frame, f"{self.value:5.1f}", (cx - 30, cy + radius + 36), 0.9, hud.GREEN, 2)


ALL_MODES = [CommandMode, DrawMode, RPSMode, DialMode]
