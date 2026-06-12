"""3D humanoid avatar rendered with OpenGL, driven by pose landmarks.

The avatar is a geometric figure built from spheres (joints) and cylinders
(limbs), posed in real time from MediaPipe's world-coordinate landmarks.
Rendering happens in a pygame window with OpenGL context.
"""

from __future__ import annotations

import math

import numpy as np

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE

from OpenGL.GL import *
from OpenGL.GLU import *

from .pose_detector import (
    NOSE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    LEFT_INDEX, RIGHT_INDEX,
)

# Body part colors (R, G, B, A) in [0,1].
SKIN = (0.93, 0.76, 0.65, 1.0)
SHIRT = (0.25, 0.55, 0.85, 1.0)
PANTS = (0.2, 0.2, 0.35, 1.0)
SHOE = (0.3, 0.3, 0.3, 1.0)
HEAD_COLOR = (0.93, 0.76, 0.65, 1.0)
JOINT_COLOR = (0.8, 0.85, 0.9, 1.0)

# Bone definitions: (start_idx, end_idx, radius, color).
BONES = [
    # Torso
    (LEFT_SHOULDER, RIGHT_SHOULDER, 0.035, SHIRT),
    (LEFT_SHOULDER, LEFT_HIP, 0.032, SHIRT),
    (RIGHT_SHOULDER, RIGHT_HIP, 0.032, SHIRT),
    (LEFT_HIP, RIGHT_HIP, 0.032, PANTS),
    # Left arm
    (LEFT_SHOULDER, LEFT_ELBOW, 0.022, SHIRT),
    (LEFT_ELBOW, LEFT_WRIST, 0.018, SKIN),
    (LEFT_WRIST, LEFT_INDEX, 0.012, SKIN),
    # Right arm
    (RIGHT_SHOULDER, RIGHT_ELBOW, 0.022, SHIRT),
    (RIGHT_ELBOW, RIGHT_WRIST, 0.018, SKIN),
    (RIGHT_WRIST, RIGHT_INDEX, 0.012, SKIN),
    # Left leg
    (LEFT_HIP, LEFT_KNEE, 0.028, PANTS),
    (LEFT_KNEE, LEFT_ANKLE, 0.024, PANTS),
    (LEFT_ANKLE, LEFT_HEEL, 0.016, SHOE),
    (LEFT_ANKLE, LEFT_FOOT_INDEX, 0.016, SHOE),
    # Right leg
    (RIGHT_HIP, RIGHT_KNEE, 0.028, PANTS),
    (RIGHT_KNEE, RIGHT_ANKLE, 0.024, PANTS),
    (RIGHT_ANKLE, RIGHT_HEEL, 0.016, SHOE),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX, 0.016, SHOE),
]

# Joints to draw as spheres: (landmark_idx, radius).
JOINTS = [
    (LEFT_SHOULDER, 0.030), (RIGHT_SHOULDER, 0.030),
    (LEFT_ELBOW, 0.022), (RIGHT_ELBOW, 0.022),
    (LEFT_WRIST, 0.020), (RIGHT_WRIST, 0.020),
    (LEFT_HIP, 0.030), (RIGHT_HIP, 0.030),
    (LEFT_KNEE, 0.026), (RIGHT_KNEE, 0.026),
    (LEFT_ANKLE, 0.022), (RIGHT_ANKLE, 0.022),
]

# GLU quadric detail.
SLICES = 16
STACKS = 8


def _mediapipe_to_gl(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert MediaPipe world coords to OpenGL coords.
    MediaPipe: x right, y down, z toward camera.
    OpenGL:    x right, y up,   z toward viewer."""
    return (x, -y, -z)


def _draw_cylinder_between(p1, p2, radius: float, quadric) -> None:
    """Draw a cylinder from p1 to p2."""
    dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-6:
        return

    glPushMatrix()
    glTranslatef(*p1)

    # Align the Z-axis (cylinder default) with the direction p1->p2.
    ax = -dy * 1  # cross product of (0,0,1) x (dx,dy,dz) = (-dy, dx, 0)
    ay = dx * 1
    az = 0.0
    cross_len = math.sqrt(ax * ax + ay * ay)
    if cross_len > 1e-6:
        angle = math.degrees(math.acos(max(-1, min(1, dz / length))))
        glRotatef(angle, ax, ay, az)
    elif dz < 0:
        glRotatef(180, 1, 0, 0)

    gluCylinder(quadric, radius, radius * 0.85, length, SLICES, 1)
    glPopMatrix()


class AvatarRenderer:
    """Manages a pygame/OpenGL window and draws a 3D humanoid from pose data."""

    def __init__(self, width: int = 800, height: int = 800, title: str = "HandSense — 3D Avatar"):
        pygame.init()
        pygame.display.set_caption(title)
        self.width = width
        self.height = height
        self._surface = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)

        self._cam_yaw = 0.0      # degrees, orbiting the avatar
        self._cam_pitch = 15.0
        self._cam_dist = 1.8
        self._target_y = -0.2    # look slightly above hip center

        self._quadric = gluNewQuadric()
        gluQuadricNormals(self._quadric, GLU_SMOOTH)

        self._pip_texture = glGenTextures(1)

        self._setup_gl()

    def _setup_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_POSITION, (2.0, 3.0, 4.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.35, 0.35, 0.4, 1.0))

        glClearColor(0.12, 0.12, 0.16, 1.0)

    def _set_projection(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width / max(self.height, 1)
        gluPerspective(50, aspect, 0.05, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def _set_camera(self) -> None:
        glLoadIdentity()
        yaw_r = math.radians(self._cam_yaw)
        pitch_r = math.radians(self._cam_pitch)
        cx = self._cam_dist * math.cos(pitch_r) * math.sin(yaw_r)
        cy = self._cam_dist * math.sin(pitch_r)
        cz = self._cam_dist * math.cos(pitch_r) * math.cos(yaw_r)
        gluLookAt(cx, cy + self._target_y, cz,
                  0, self._target_y, 0,
                  0, 1, 0)

    def _draw_ground(self) -> None:
        glDisable(GL_LIGHTING)
        glColor4f(0.25, 0.25, 0.3, 1.0)
        glBegin(GL_LINES)
        extent = 2.0
        step = 0.2
        y = -1.0  # approximate ground level
        x = -extent
        while x <= extent + 0.001:
            glVertex3f(x, y, -extent)
            glVertex3f(x, y, extent)
            glVertex3f(-extent, y, x)
            glVertex3f(extent, y, x)
            x += step
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_head(self, landmarks_gl) -> None:
        """Compute head center from ears/nose and draw an ellipsoid."""
        nose = np.array(landmarks_gl[NOSE])
        left_ear = np.array(landmarks_gl[LEFT_EAR])
        right_ear = np.array(landmarks_gl[RIGHT_EAR])
        head_center = (left_ear + right_ear) / 2
        # Shift head center slightly above nose level.
        head_center[1] += 0.06

        glPushMatrix()
        glTranslatef(*head_center)
        glColor4f(*HEAD_COLOR)
        # Slightly taller than wide.
        glScalef(1.0, 1.25, 1.0)
        gluSphere(self._quadric, 0.09, SLICES * 2, STACKS * 2)
        glPopMatrix()

        # Neck: shoulder midpoint to head center.
        neck_base = (np.array(landmarks_gl[LEFT_SHOULDER]) +
                     np.array(landmarks_gl[RIGHT_SHOULDER])) / 2
        glColor4f(*SKIN)
        _draw_cylinder_between(tuple(neck_base), tuple(head_center - np.array([0, 0.04, 0])),
                               0.02, self._quadric)

    def _draw_torso_fill(self, gl_lm) -> None:
        """Fill the torso as two textured quads for a more solid look."""
        ls = gl_lm[LEFT_SHOULDER]
        rs = gl_lm[RIGHT_SHOULDER]
        lh = gl_lm[LEFT_HIP]
        rh = gl_lm[RIGHT_HIP]

        glDisable(GL_LIGHTING)
        glColor4f(0.22, 0.50, 0.78, 0.85)
        glBegin(GL_QUADS)
        glVertex3f(*ls)
        glVertex3f(*rs)
        glVertex3f(*rh)
        glVertex3f(*lh)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_body(self, world_landmarks: list[tuple[float, float, float]]) -> None:
        gl_lm = [_mediapipe_to_gl(*pt) for pt in world_landmarks]

        # Torso fill.
        self._draw_torso_fill(gl_lm)

        # Bones.
        for start_idx, end_idx, radius, color in BONES:
            glColor4f(*color)
            _draw_cylinder_between(gl_lm[start_idx], gl_lm[end_idx], radius, self._quadric)

        # Joints.
        for idx, radius in JOINTS:
            glPushMatrix()
            glTranslatef(*gl_lm[idx])
            glColor4f(*JOINT_COLOR)
            gluSphere(self._quadric, radius, SLICES, STACKS)
            glPopMatrix()

        # Head.
        self._draw_head(gl_lm)

    def _draw_pip(self, frame_bgr) -> None:
        """Draw camera feed as a picture-in-picture in the bottom-left corner."""
        h, w = frame_bgr.shape[:2]
        pip_w = self.width // 4
        pip_h = int(pip_w * h / w)

        # Flip vertically for OpenGL texture orientation, convert BGR->RGB.
        rgb = frame_bgr[::-1, :, ::-1].copy()

        glBindTexture(GL_TEXTURE_2D, self._pip_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb)

        # Switch to 2D overlay mode.
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)

        margin = 12
        x0, y0 = margin, margin
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x0, y0)
        glTexCoord2f(1, 0); glVertex2f(x0 + pip_w, y0)
        glTexCoord2f(1, 1); glVertex2f(x0 + pip_w, y0 + pip_h)
        glTexCoord2f(0, 1); glVertex2f(x0, y0 + pip_h)
        glEnd()

        # Border.
        glDisable(GL_TEXTURE_2D)
        glColor4f(0.6, 0.8, 0.6, 1)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x0, y0)
        glVertex2f(x0 + pip_w, y0)
        glVertex2f(x0 + pip_w, y0 + pip_h)
        glVertex2f(x0, y0 + pip_h)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def handle_events(self) -> bool:
        """Process pygame events. Returns False if the user wants to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self._surface = pygame.display.set_mode(
                    (self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
                self._setup_gl()

        # Continuous key-hold for camera orbit.
        keys = pygame.key.get_pressed()
        rot_speed = 2.5
        if keys[pygame.K_LEFT]:
            self._cam_yaw -= rot_speed
        if keys[pygame.K_RIGHT]:
            self._cam_yaw += rot_speed
        if keys[pygame.K_UP]:
            self._cam_pitch = min(self._cam_pitch + rot_speed, 80)
        if keys[pygame.K_DOWN]:
            self._cam_pitch = max(self._cam_pitch - rot_speed, -30)
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            self._cam_dist = max(0.5, self._cam_dist - 0.03)
        if keys[pygame.K_MINUS]:
            self._cam_dist = min(5.0, self._cam_dist + 0.03)

        return True

    def render(self, world_landmarks: list[tuple[float, float, float]] | None,
               camera_frame=None) -> None:
        """Render one frame: clear, draw scene, swap buffers."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._set_projection()
        self._set_camera()

        self._draw_ground()

        if world_landmarks is not None:
            self._draw_body(world_landmarks)

        if camera_frame is not None:
            self._draw_pip(camera_frame)

        pygame.display.flip()

    def close(self) -> None:
        gluDeleteQuadric(self._quadric)
        pygame.quit()
