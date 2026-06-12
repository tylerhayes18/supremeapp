"""3D humanoid avatar rendered with OpenGL, driven by holistic landmarks.

Two render modes:
1. **Mesh mode** (default): loads a real 3D humanoid GLB model, segments it
   by body part, and poses each segment to follow MediaPipe landmarks.
2. **Geometric fallback**: capsule limbs + spheres if no model is available.

Both modes include:
- Articulated fingers (from hand landmarks)
- Head with face orientation (yaw/pitch/roll), eyes, and mouth
- Multi-light cinematic setup (warm key + cool fill + rim)
- Ground grid, camera-feed picture-in-picture
"""

from __future__ import annotations

import math

import numpy as np

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE

from OpenGL.GL import *
from OpenGL.GLU import *

from .holistic_detector import (
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
    LEFT_PINKY, RIGHT_PINKY,
    LEFT_THUMB, RIGHT_THUMB,
    HAND_FINGERS, HAND_WRIST,
    HolisticObservation,
)

# ── Colors (RGBA, 0–1) ────────────────────────────────────────────────

SKIN = (0.92, 0.75, 0.62, 1.0)
SKIN_DARK = (0.82, 0.65, 0.52, 1.0)
SHIRT = (0.18, 0.20, 0.25, 1.0)
SHIRT_ACCENT = (0.22, 0.25, 0.32, 1.0)
PANTS = (0.15, 0.15, 0.20, 1.0)
SHOE = (0.12, 0.12, 0.14, 1.0)
JOINT_COLOR = (0.30, 0.32, 0.38, 1.0)
HEAD_COLOR = (0.92, 0.75, 0.62, 1.0)
EYE_WHITE = (0.95, 0.95, 0.97, 1.0)
EYE_IRIS = (0.20, 0.35, 0.55, 1.0)
EYE_PUPIL = (0.08, 0.08, 0.10, 1.0)
MOUTH_COLOR = (0.65, 0.30, 0.30, 1.0)
FINGER_COLOR = (0.88, 0.72, 0.60, 1.0)
FINGER_JOINT = (0.80, 0.65, 0.55, 1.0)

# Mesh segment colors (indexed by segment id from mesh_loader.py).
SEGMENT_COLORS = [
    HEAD_COLOR,     # 0  head
    SHIRT,          # 1  torso_upper
    SHIRT_ACCENT,   # 2  torso_lower
    SHIRT,          # 3  left_upper_arm
    SKIN,           # 4  left_forearm
    SKIN_DARK,      # 5  left_hand
    SHIRT,          # 6  right_upper_arm
    SKIN,           # 7  right_forearm
    SKIN_DARK,      # 8  right_hand
    PANTS,          # 9  left_thigh
    PANTS,          # 10 left_shin
    SHOE,           # 11 left_foot
    PANTS,          # 12 right_thigh
    PANTS,          # 13 right_shin
    SHOE,           # 14 right_foot
]

# ── Geometric fallback bones ──────────────────────────────────────────

BODY_BONES = [
    (LEFT_SHOULDER, LEFT_HIP, 0.038, 0.032, SHIRT),
    (RIGHT_SHOULDER, RIGHT_HIP, 0.038, 0.032, SHIRT),
    (LEFT_HIP, RIGHT_HIP, 0.034, 0.034, PANTS),
    (LEFT_SHOULDER, LEFT_ELBOW, 0.028, 0.024, SHIRT),
    (RIGHT_SHOULDER, RIGHT_ELBOW, 0.028, 0.024, SHIRT),
    (LEFT_ELBOW, LEFT_WRIST, 0.022, 0.018, SKIN),
    (RIGHT_ELBOW, RIGHT_WRIST, 0.022, 0.018, SKIN),
    (LEFT_HIP, LEFT_KNEE, 0.034, 0.028, PANTS),
    (RIGHT_HIP, RIGHT_KNEE, 0.034, 0.028, PANTS),
    (LEFT_KNEE, LEFT_ANKLE, 0.026, 0.022, PANTS),
    (RIGHT_KNEE, RIGHT_ANKLE, 0.026, 0.022, PANTS),
    (LEFT_ANKLE, LEFT_FOOT_INDEX, 0.020, 0.014, SHOE),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX, 0.020, 0.014, SHOE),
    (LEFT_ANKLE, LEFT_HEEL, 0.018, 0.014, SHOE),
    (RIGHT_ANKLE, RIGHT_HEEL, 0.018, 0.014, SHOE),
]

BODY_JOINTS = [
    (LEFT_SHOULDER, 0.032, JOINT_COLOR),
    (RIGHT_SHOULDER, 0.032, JOINT_COLOR),
    (LEFT_ELBOW, 0.024, JOINT_COLOR),
    (RIGHT_ELBOW, 0.024, JOINT_COLOR),
    (LEFT_WRIST, 0.020, JOINT_COLOR),
    (RIGHT_WRIST, 0.020, JOINT_COLOR),
    (LEFT_HIP, 0.034, JOINT_COLOR),
    (RIGHT_HIP, 0.034, JOINT_COLOR),
    (LEFT_KNEE, 0.028, JOINT_COLOR),
    (RIGHT_KNEE, 0.028, JOINT_COLOR),
    (LEFT_ANKLE, 0.024, JOINT_COLOR),
    (RIGHT_ANKLE, 0.024, JOINT_COLOR),
]

SLICES = 24
STACKS = 12

# Mapping from segment index -> pair of MediaPipe landmark indices that
# define the segment's position and orientation in world space.
# (primary_landmark, secondary_landmark) — the segment is translated to
# primary and oriented toward secondary.
SEGMENT_LANDMARK_MAP = {
    0:  (NOSE, LEFT_EAR),              # head
    1:  (LEFT_SHOULDER, LEFT_HIP),     # torso upper
    2:  (LEFT_HIP, LEFT_KNEE),         # torso lower
    3:  (LEFT_SHOULDER, LEFT_ELBOW),   # left upper arm
    4:  (LEFT_ELBOW, LEFT_WRIST),      # left forearm
    5:  (LEFT_WRIST, LEFT_INDEX),      # left hand
    6:  (RIGHT_SHOULDER, RIGHT_ELBOW), # right upper arm
    7:  (RIGHT_ELBOW, RIGHT_WRIST),    # right forearm
    8:  (RIGHT_WRIST, RIGHT_INDEX),    # right hand
    9:  (LEFT_HIP, LEFT_KNEE),         # left thigh
    10: (LEFT_KNEE, LEFT_ANKLE),       # left shin
    11: (LEFT_ANKLE, LEFT_FOOT_INDEX), # left foot
    12: (RIGHT_HIP, RIGHT_KNEE),       # right thigh
    13: (RIGHT_KNEE, RIGHT_ANKLE),     # right shin
    14: (RIGHT_ANKLE, RIGHT_FOOT_INDEX), # right foot
}


def _mp_to_gl(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x, -y, -z)


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_len(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec_normalize(v):
    l = _vec_len(v)
    if l < 1e-8:
        return (0, 0, 1)
    return (v[0] / l, v[1] / l, v[2] / l)


def _vec_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _draw_capsule(p1, p2, r1: float, r2: float, quadric) -> None:
    d = _vec_sub(p2, p1)
    length = _vec_len(d)
    if length < 1e-6:
        return
    glPushMatrix()
    glTranslatef(*p1)
    dn = _vec_normalize(d)
    up = (0, 0, 1)
    cross = _vec_cross(up, dn)
    cross_len = _vec_len(cross)
    if cross_len > 1e-6:
        dot = up[0] * dn[0] + up[1] * dn[1] + up[2] * dn[2]
        angle = math.degrees(math.acos(max(-1, min(1, dot))))
        glRotatef(angle, cross[0], cross[1], cross[2])
    elif dn[2] < 0:
        glRotatef(180, 1, 0, 0)
    gluCylinder(quadric, r1, r2, length, SLICES, 1)
    glPushMatrix()
    glRotatef(180, 1, 0, 0)
    gluSphere(quadric, r1, SLICES, STACKS // 2)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0, 0, length)
    gluSphere(quadric, r2, SLICES, STACKS // 2)
    glPopMatrix()
    glPopMatrix()


def _draw_finger_chain(joints_gl, quadric) -> None:
    for i in range(len(joints_gl) - 1):
        r = 0.006 if i < 2 else 0.005
        _draw_capsule(joints_gl[i], joints_gl[i + 1], r, r * 0.85, quadric)
    for j in joints_gl:
        glPushMatrix()
        glTranslatef(*j)
        glColor4f(*FINGER_JOINT)
        gluSphere(quadric, 0.006, 10, 6)
        glPopMatrix()


class AvatarRenderer:
    def __init__(self, width: int = 900, height: int = 900,
                 title: str = "HandSense — 3D Body",
                 model_path: str | None = None, use_mesh: bool = True):
        pygame.init()
        pygame.display.set_caption(title)
        self.width = width
        self.height = height
        self._surface = pygame.display.set_mode(
            (width, height), DOUBLEBUF | OPENGL | RESIZABLE)

        self._cam_yaw = 0.0
        self._cam_pitch = 12.0
        self._cam_dist = 2.0
        self._target_y = -0.15

        self._quadric = gluNewQuadric()
        gluQuadricNormals(self._quadric, GLU_SMOOTH)

        self._pip_texture = glGenTextures(1)

        # Try to load the 3D mesh model.
        self._mesh = None
        if use_mesh:
            try:
                from .mesh_loader import AvatarMesh
                self._mesh = AvatarMesh(model_path)
                print(f"loaded avatar mesh ({self._mesh._total_verts} verts, "
                      f"{self._mesh._total_faces} faces)")
            except Exception as exc:
                print(f"mesh load failed ({exc}), using geometric fallback")

        self._setup_gl()

    def _setup_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Key light: warm, from upper-right.
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (3.0, 4.0, 5.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.9, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.15, 0.15, 0.18, 1.0))

        # Fill light: cool, from the left.
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, (-4.0, 2.0, 3.0, 0.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.3, 0.35, 0.5, 1.0))
        glLightfv(GL_LIGHT1, GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))

        # Rim light: from behind.
        glEnable(GL_LIGHT2)
        glLightfv(GL_LIGHT2, GL_POSITION, (0.0, 2.0, -5.0, 0.0))
        glLightfv(GL_LIGHT2, GL_DIFFUSE, (0.25, 0.25, 0.3, 1.0))

        glClearColor(0.08, 0.08, 0.11, 1.0)

    # ── Camera ─────────────────────────────────────────────────────────

    def _set_projection(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / max(self.height, 1), 0.05, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def _set_camera(self) -> None:
        glLoadIdentity()
        yr = math.radians(self._cam_yaw)
        pr = math.radians(self._cam_pitch)
        cx = self._cam_dist * math.cos(pr) * math.sin(yr)
        cy = self._cam_dist * math.sin(pr)
        cz = self._cam_dist * math.cos(pr) * math.cos(yr)
        gluLookAt(cx, cy + self._target_y, cz,
                  0, self._target_y, 0,
                  0, 1, 0)

    # ── Ground ─────────────────────────────────────────────────────────

    def _draw_ground(self) -> None:
        glDisable(GL_LIGHTING)
        y = -1.05
        extent = 3.0
        step = 0.25
        x = -extent
        while x <= extent + 0.001:
            fade = max(0.0, 1.0 - abs(x) / extent)
            glColor4f(0.25 * fade, 0.25 * fade, 0.30 * fade, fade * 0.6)
            glBegin(GL_LINES)
            glVertex3f(x, y, -extent)
            glVertex3f(x, y, extent)
            glVertex3f(-extent, y, x)
            glVertex3f(extent, y, x)
            glEnd()
            x += step
        glEnable(GL_LIGHTING)

    # ── Mesh-based body rendering ──────────────────────────────────────

    def _draw_mesh_body(self, obs: HolisticObservation) -> None:
        """Render the loaded 3D mesh, deformed per-vertex from landmarks."""
        gl_lm = [_mp_to_gl(*pt) for pt in obs.pose_world]

        # Compute body scale and position.
        lh = np.array(gl_lm[LEFT_HIP])
        rh = np.array(gl_lm[RIGHT_HIP])
        ls = np.array(gl_lm[LEFT_SHOULDER])
        rs = np.array(gl_lm[RIGHT_SHOULDER])
        hip_center = (lh + rh) / 2
        shoulder_center = (ls + rs) / 2
        torso_len = np.linalg.norm(shoulder_center - hip_center)
        model_torso = self._mesh.model_height * 0.35
        scale = max(torso_len / max(model_torso, 0.01), 0.1)

        # Convert all 33 tracked landmarks from GL space to model space.
        gl_landmarks = np.array(gl_lm, dtype=np.float32)  # (33, 3)
        model_hip_y = self._mesh.bounds_min[1] + self._mesh.model_height * 0.47
        model_hip = np.array([self._mesh.model_center[0], model_hip_y,
                              self._mesh.model_center[2]])

        # tracked_model = (gl_pos - hip_center) / scale + model_hip
        tracked_model = (gl_landmarks - hip_center[None, :]) / scale + model_hip[None, :]

        # Deform the mesh.
        self._mesh.deform(tracked_model)

        # Render.
        glPushMatrix()
        glTranslatef(*hip_center)
        glScalef(scale, scale, scale)
        glTranslatef(-model_hip[0], -model_hip[1], -model_hip[2])

        glColor4f(0.65, 0.65, 0.70, 1.0)
        self._mesh.draw()

        glPopMatrix()

        # Face features and fingers on top of the mesh.
        self._draw_head_features(gl_lm, obs)
        if obs.has_left_hand or obs.has_right_hand:
            self._draw_hands(obs, gl_lm)

    # ── Geometric body rendering (fallback) ────────────────────────────

    def _draw_geometric_body(self, obs: HolisticObservation) -> None:
        gl_lm = [_mp_to_gl(*pt) for pt in obs.pose_world]

        self._draw_torso(gl_lm)

        for start, end, r1, r2, color in BODY_BONES:
            glColor4f(*color)
            _draw_capsule(gl_lm[start], gl_lm[end], r1, r2, self._quadric)

        for idx, radius, color in BODY_JOINTS:
            glPushMatrix()
            glTranslatef(*gl_lm[idx])
            glColor4f(*color)
            gluSphere(self._quadric, radius, SLICES, STACKS)
            glPopMatrix()

        self._draw_head(gl_lm, obs)
        self._draw_hands(obs, gl_lm)
        self._draw_hand_stubs(gl_lm)

    def _draw_torso(self, gl_lm) -> None:
        ls = gl_lm[LEFT_SHOULDER]
        rs = gl_lm[RIGHT_SHOULDER]
        lh = gl_lm[LEFT_HIP]
        rh = gl_lm[RIGHT_HIP]
        v1 = _vec_sub(rs, ls)
        v2 = _vec_sub(lh, ls)
        n = _vec_normalize(_vec_cross(v1, v2))
        glColor4f(*SHIRT)
        glBegin(GL_TRIANGLES)
        glNormal3f(*n)
        glVertex3f(*ls); glVertex3f(*rs); glVertex3f(*rh)
        glVertex3f(*ls); glVertex3f(*rh); glVertex3f(*lh)
        glEnd()
        bn = (-n[0], -n[1], -n[2])
        glBegin(GL_TRIANGLES)
        glNormal3f(*bn)
        glVertex3f(*rs); glVertex3f(*ls); glVertex3f(*lh)
        glVertex3f(*rs); glVertex3f(*lh); glVertex3f(*rh)
        glEnd()
        glColor4f(*SHIRT_ACCENT)
        _draw_capsule(ls, rs, 0.034, 0.034, self._quadric)

    def _draw_head(self, gl_lm, obs: HolisticObservation) -> None:
        left_ear = np.array(gl_lm[LEFT_EAR])
        right_ear = np.array(gl_lm[RIGHT_EAR])
        head_center = (left_ear + right_ear) / 2
        head_center[1] += 0.06
        neck_base = (np.array(gl_lm[LEFT_SHOULDER]) + np.array(gl_lm[RIGHT_SHOULDER])) / 2
        glColor4f(*SKIN)
        _draw_capsule(tuple(neck_base), tuple(head_center - np.array([0, 0.04, 0])),
                      0.022, 0.024, self._quadric)
        glPushMatrix()
        glTranslatef(*head_center)
        fo = obs.face_orientation
        glRotatef(-fo.yaw, 0, 1, 0)
        glRotatef(fo.pitch, 1, 0, 0)
        glRotatef(-fo.roll, 0, 0, 1)
        glColor4f(*HEAD_COLOR)
        glPushMatrix()
        glScalef(0.85, 1.1, 0.9)
        gluSphere(self._quadric, 0.095, SLICES * 2, STACKS * 2)
        glPopMatrix()
        self._draw_face_features_local(fo)
        glPopMatrix()

    def _draw_head_features(self, gl_lm, obs: HolisticObservation) -> None:
        """Draw face features for mesh mode (positioned over the mesh head)."""
        left_ear = np.array(gl_lm[LEFT_EAR])
        right_ear = np.array(gl_lm[RIGHT_EAR])
        head_center = (left_ear + right_ear) / 2
        head_center[1] += 0.06

        glPushMatrix()
        glTranslatef(*head_center)
        fo = obs.face_orientation
        glRotatef(-fo.yaw, 0, 1, 0)
        glRotatef(fo.pitch, 1, 0, 0)
        glRotatef(-fo.roll, 0, 0, 1)
        self._draw_face_features_local(fo)
        glPopMatrix()

    def _draw_face_features_local(self, fo) -> None:
        """Draw eyes and mouth (called in head-local coordinate frame)."""
        eye_y = 0.015
        eye_sep = 0.032
        eye_z = 0.085

        for side in (-1, 1):
            ex = side * eye_sep
            glPushMatrix()
            glTranslatef(ex, eye_y, eye_z)
            glColor4f(*EYE_WHITE)
            glPushMatrix()
            openness = fo.left_eye_open if side < 0 else fo.right_eye_open
            glScalef(1.0, max(0.15, openness * 0.7), 0.5)
            gluSphere(self._quadric, 0.014, 12, 8)
            glPopMatrix()
            glColor4f(*EYE_IRIS)
            glPushMatrix()
            glTranslatef(0, 0, 0.006)
            gluSphere(self._quadric, 0.008, 10, 6)
            glPopMatrix()
            glColor4f(*EYE_PUPIL)
            glPushMatrix()
            glTranslatef(0, 0, 0.010)
            gluSphere(self._quadric, 0.004, 8, 4)
            glPopMatrix()
            glPopMatrix()

        glPushMatrix()
        glTranslatef(0, -0.030, eye_z - 0.005)
        glColor4f(*MOUTH_COLOR)
        glScalef(1.0, max(0.3, fo.mouth_open * 1.5), 0.4)
        gluSphere(self._quadric, 0.016, 12, 6)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, -0.005, eye_z + 0.004)
        glColor4f(*SKIN_DARK)
        glScalef(0.5, 0.8, 0.5)
        gluSphere(self._quadric, 0.010, 8, 6)
        glPopMatrix()

    # ── Hands ──────────────────────────────────────────────────────────

    def _draw_hands(self, obs: HolisticObservation, gl_lm) -> None:
        for hand_world in [obs.left_hand_world, obs.right_hand_world]:
            if hand_world is None:
                continue
            hand_gl = [_mp_to_gl(*pt) for pt in hand_world]
            glColor4f(*FINGER_COLOR)
            palm_indices = [0, 5, 9, 13, 17]
            palm_pts = [hand_gl[i] for i in palm_indices]
            center = tuple(np.mean(palm_pts, axis=0))
            n = _vec_normalize(_vec_cross(
                _vec_sub(palm_pts[1], palm_pts[0]),
                _vec_sub(palm_pts[2], palm_pts[0])))
            glBegin(GL_TRIANGLE_FAN)
            glNormal3f(*n)
            glVertex3f(*center)
            for pt in palm_pts:
                glVertex3f(*pt)
            glVertex3f(*palm_pts[0])
            glEnd()
            glColor4f(*FINGER_COLOR)
            for finger_indices in HAND_FINGERS:
                if finger_indices[0] == 1:
                    chain = [hand_gl[0]] + [hand_gl[i] for i in finger_indices]
                else:
                    chain = [hand_gl[finger_indices[0]]] + [hand_gl[i] for i in finger_indices[1:]]
                _draw_finger_chain(chain, self._quadric)

    def _draw_hand_stubs(self, gl_lm) -> None:
        for wrist, index, pinky, thumb in [
            (LEFT_WRIST, LEFT_INDEX, LEFT_PINKY, LEFT_THUMB),
            (RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY, RIGHT_THUMB),
        ]:
            glColor4f(*SKIN)
            _draw_capsule(gl_lm[wrist], gl_lm[index], 0.014, 0.008, self._quadric)
            _draw_capsule(gl_lm[wrist], gl_lm[pinky], 0.012, 0.006, self._quadric)
            _draw_capsule(gl_lm[wrist], gl_lm[thumb], 0.012, 0.008, self._quadric)

    # ── Picture-in-picture ─────────────────────────────────────────────

    def _draw_pip(self, frame_bgr) -> None:
        h, w = frame_bgr.shape[:2]
        pip_w = self.width // 4
        pip_h = int(pip_w * h / w)
        rgb = frame_bgr[::-1, :, ::-1].copy()
        glBindTexture(GL_TEXTURE_2D, self._pip_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb)
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
        margin = 14
        x0, y0 = margin, margin
        # Shadow.
        glDisable(GL_TEXTURE_2D)
        glColor4f(0, 0, 0, 0.4)
        glBegin(GL_QUADS)
        glVertex2f(x0 + 3, y0 - 3); glVertex2f(x0 + pip_w + 3, y0 - 3)
        glVertex2f(x0 + pip_w + 3, y0 + pip_h - 3); glVertex2f(x0 + 3, y0 + pip_h - 3)
        glEnd()
        # Feed.
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x0, y0)
        glTexCoord2f(1, 0); glVertex2f(x0 + pip_w, y0)
        glTexCoord2f(1, 1); glVertex2f(x0 + pip_w, y0 + pip_h)
        glTexCoord2f(0, 1); glVertex2f(x0, y0 + pip_h)
        glEnd()
        # Border.
        glDisable(GL_TEXTURE_2D)
        glColor4f(0.4, 0.7, 0.4, 0.9)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x0, y0); glVertex2f(x0 + pip_w, y0)
        glVertex2f(x0 + pip_w, y0 + pip_h); glVertex2f(x0, y0 + pip_h)
        glEnd()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # ── HUD ────────────────────────────────────────────────────────────

    def _draw_hud(self, obs: HolisticObservation | None) -> None:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        indicators = []
        if obs is not None:
            indicators.append(("BODY", (0.3, 0.9, 0.3, 1.0)))
            if obs.has_face:
                indicators.append(("FACE", (0.3, 0.7, 0.9, 1.0)))
            if obs.has_left_hand:
                indicators.append(("L-HAND", (0.9, 0.7, 0.3, 1.0)))
            if obs.has_right_hand:
                indicators.append(("R-HAND", (0.9, 0.7, 0.3, 1.0)))
        x = self.width - 16
        for label, color in reversed(indicators):
            glColor4f(*color)
            glPointSize(10)
            glBegin(GL_POINTS)
            glVertex2f(x, self.height - 20)
            glEnd()
            x -= 60
        if self._mesh is not None:
            glColor4f(0.5, 0.5, 0.6, 0.8)
            glPointSize(6)
            glBegin(GL_POINTS)
            glVertex2f(16, self.height - 20)
            glEnd()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # ── Event handling ─────────────────────────────────────────────────

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                elif event.key == pygame.K_r:
                    self._cam_yaw = 0.0
                    self._cam_pitch = 12.0
                    self._cam_dist = 2.0
                elif event.key == pygame.K_TAB:
                    # Toggle mesh / geometric mode.
                    if self._mesh is not None:
                        self._mesh, self._mesh_stash = None, self._mesh
                    elif hasattr(self, "_mesh_stash"):
                        self._mesh = self._mesh_stash
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self._surface = pygame.display.set_mode(
                    (self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
                self._setup_gl()
        keys = pygame.key.get_pressed()
        speed = 2.0
        if keys[pygame.K_LEFT]:
            self._cam_yaw -= speed
        if keys[pygame.K_RIGHT]:
            self._cam_yaw += speed
        if keys[pygame.K_UP]:
            self._cam_pitch = min(self._cam_pitch + speed, 80)
        if keys[pygame.K_DOWN]:
            self._cam_pitch = max(self._cam_pitch - speed, -30)
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            self._cam_dist = max(0.4, self._cam_dist - 0.04)
        if keys[pygame.K_MINUS]:
            self._cam_dist = min(6.0, self._cam_dist + 0.04)
        return True

    # ── Main render ────────────────────────────────────────────────────

    def render(self, obs: HolisticObservation | None, camera_frame=None) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._set_projection()
        self._set_camera()
        self._draw_ground()
        if obs is not None:
            if self._mesh is not None:
                self._draw_mesh_body(obs)
            else:
                self._draw_geometric_body(obs)
        if camera_frame is not None:
            self._draw_pip(camera_frame)
        self._draw_hud(obs)
        pygame.display.flip()

    def close(self) -> None:
        gluDeleteQuadric(self._quadric)
        pygame.quit()
