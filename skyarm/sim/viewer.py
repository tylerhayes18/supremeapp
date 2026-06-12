"""Interactive 3D test environment (pygame + OpenGL).

    python -m skyarm.sim                # interactive viewer
    python -m skyarm.sim --demo         # run the demo mission on launch

Controls
--------
    W/S, A/D     move the target marker in x / y
    R/F          target up / down
    ENTER        send the arm to the target (vertical approach)
    SHIFT+ENTER  send with a 45 degree approach
    G            toggle gripper
    SPACE        run / pause the demo mission
    arrow keys   orbit camera        +/- or wheel  zoom
    H            home pose           ESC           quit
"""

from __future__ import annotations

import math
import sys

import numpy as np

from .. import kinematics as kin
from .. import spec
from . import scene
from .world import World, demo_mission


def run_viewer(demo: bool = False):
    import pygame
    from pygame.locals import (DOUBLEBUF, OPENGL, K_ESCAPE, K_SPACE, K_RETURN,
                               KEYDOWN, QUIT)
    from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                           GL_DEPTH_TEST, GL_LIGHT0, GL_LIGHTING, GL_MODELVIEW,
                           GL_NORMALIZE, GL_POSITION, GL_PROJECTION, GL_TRIANGLES,
                           GL_COLOR_MATERIAL, glBegin, glClear, glClearColor,
                           glColor3f, glEnable, glEnd, glLightfv, glLoadIdentity,
                           glMatrixMode, glNormal3f, glVertex3f, glTranslatef,
                           glRotatef)
    from OpenGL.GLU import gluPerspective, gluLookAt

    pygame.init()
    size = (1280, 860)
    pygame.display.set_mode(size, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("SkyArm test environment — see module docstring for keys")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_NORMALIZE)
    glClearColor(0.09, 0.10, 0.13, 1.0)

    world = World()
    if demo:
        demo_mission(world)
    target = np.array([kin.X_TRAVEL_MM / 2, kin.Y_TRAVEL_MM / 2,
                       kin.CEILING_MM - 1200.0])
    cam_yaw, cam_pitch, cam_dist = 35.0, 28.0, 5200.0
    center = (spec.ROOM["x_mm"] / 2, spec.ROOM["y_mm"] / 2, 1300.0)
    running, paused = True, False
    clock = pygame.time.Clock()

    def draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, size[0] / size[1], 10.0, 40000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        ex = center[0] + cam_dist * math.cos(math.radians(cam_pitch)) * math.cos(math.radians(cam_yaw))
        ey = center[1] + cam_dist * math.cos(math.radians(cam_pitch)) * math.sin(math.radians(cam_yaw))
        ez = center[2] + cam_dist * math.sin(math.radians(cam_pitch))
        gluLookAt(ex, ey, ez, *center, 0, 0, 1)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.3, -0.5, 0.9, 0.0))

        for verts, faces, color in scene.build(world, tuple(target)):
            glColor3f(color[0] / 255, color[1] / 255, color[2] / 255)
            tri = verts[faces]
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            glBegin(GL_TRIANGLES)
            for t, n in zip(tri, normals):
                ln = np.linalg.norm(n)
                if ln > 1e-12:
                    glNormal3f(*(n / ln))
                for v in t:
                    glVertex3f(*v)
            glEnd()
        pygame.display.flip()

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                mods = pygame.key.get_mods()
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    if not world.mission:
                        demo_mission(world)
                    paused = not paused
                elif event.key == K_RETURN:
                    psi = 45.0 if mods & pygame.KMOD_SHIFT else 0.0
                    try:
                        world.goto(tuple(target), psi)
                    except Exception as exc:   # show why in the title bar
                        pygame.display.set_caption(f"SkyArm — {exc}")
                elif event.key == pygame.K_g:
                    world.grip(world.gripper_target > 0.5)
                elif event.key == pygame.K_h:
                    world.goto_joints(yaw=0, shoulder=0, elbow=0,
                                      wrist_pitch=0, wrist_roll=0)
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = max(1500.0, cam_dist - event.y * 300.0)

        keys = pygame.key.get_pressed()
        spd = 900.0 * dt
        if keys[pygame.K_w]:
            target[0] += spd
        if keys[pygame.K_s]:
            target[0] -= spd
        if keys[pygame.K_a]:
            target[1] += spd
        if keys[pygame.K_d]:
            target[1] -= spd
        if keys[pygame.K_r]:
            target[2] += spd
        if keys[pygame.K_f]:
            target[2] -= spd
        if keys[pygame.K_LEFT]:
            cam_yaw -= 60 * dt
        if keys[pygame.K_RIGHT]:
            cam_yaw += 60 * dt
        if keys[pygame.K_UP]:
            cam_pitch = min(85.0, cam_pitch + 40 * dt)
        if keys[pygame.K_DOWN]:
            cam_pitch = max(-10.0, cam_pitch - 40 * dt)
        target[2] = max(spec.FLOOR_CLEARANCE_MM, min(target[2], kin.CEILING_MM - 250))

        if not paused:
            for _ in range(2):           # 120 Hz physics, 60 Hz draw
                try:
                    world.step(1 / 120)
                except Exception as exc:
                    pygame.display.set_caption(f"SkyArm FAULT — {exc}")
        draw()

    pygame.quit()
    return 0
