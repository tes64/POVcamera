"""
pov_3d_viewer.py  –  3D optical-flow POV visualiser
=====================================================
Left-click + drag   : orbit camera
Right-click + drag  : pan
Scroll wheel        : zoom
R                   : reset camera
Space               : pause / resume video
Q / Esc             : quit

Dependencies:
    pip install opencv-python numpy pygame PyOpenGL PyOpenGL_accelerate
"""

import argparse
import math
import sys
import time
from collections import deque

import cv2
import numpy as np
import pygame
from pygame.locals import *

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    sys.exit("PyOpenGL not found.  Run:  pip install PyOpenGL PyOpenGL_accelerate")


# ---------------------------------------------------------------------------
# Optical-flow helpers (same as main script)
# ---------------------------------------------------------------------------

def get_optical_flow(prev_gray, gray, pyr_scale=0.5, levels=3, winsize=15,
                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
    return cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)


def compute_pseudo_3d_motion(flow, motion_threshold=0.2, deadzone=0.15,
                              tz_scale=0.10, sample_step=24):
    dx_f = flow[..., 0]
    dy_f = flow[..., 1]
    mag_all = np.sqrt(dx_f * dx_f + dy_f * dy_f)
    mask = mag_all > motion_threshold
    if mask.mean() < 0.01:
        dx, dy = float(dx_f.mean()), float(dy_f.mean())
    else:
        dx, dy = float(dx_f[mask].mean()), float(dy_f[mask].mean())

    if abs(dx) < deadzone: dx = 0.0
    if abs(dy) < deadzone: dy = 0.0

    mag_mean = float(np.mean(mag_all))

    h, w = flow.shape[:2]
    ys = np.arange(sample_step // 2, h, sample_step, dtype=np.int32)
    xs = np.arange(sample_step // 2, w, sample_step, dtype=np.int32)
    if len(xs) == 0 or len(ys) == 0:
        radial_sign = 1.0
    else:
        xv, yv = np.meshgrid(xs, ys)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        rx = (xv.astype(np.float32) - cx)
        ry = (yv.astype(np.float32) - cy)
        r_norm = np.sqrt(rx * rx + ry * ry) + 1e-6
        rx /= r_norm; ry /= r_norm
        fx = flow[yv, xv, 0].astype(np.float32)
        fy = flow[yv, xv, 1].astype(np.float32)
        radial_score = float(np.mean(fx * rx + fy * ry))
        radial_sign = 1.0 if radial_score >= 0.0 else -1.0

    tz_raw = radial_sign * mag_mean * tz_scale
    if abs(mag_mean) < deadzone:
        tz_raw = 0.0
    return dx, dy, tz_raw, mag_mean


# ---------------------------------------------------------------------------
# OpenGL texture helpers
# ---------------------------------------------------------------------------

def create_texture():
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return tex


def upload_frame_to_texture(tex, bgr_frame):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(np.flipud(rgb))          # OpenGL origin is bottom-left
    h, w = rgb.shape[:2]
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, rgb.tobytes())


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_video_quad(tex, aspect):
    """Draw a textured quad for the video plane at Z=0."""
    hw = aspect * 0.5          # half-width  (height fixed at 0.5)
    hh = 0.5
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex3f(-hw, -hh, 0)
    glTexCoord2f(1, 0); glVertex3f( hw, -hh, 0)
    glTexCoord2f(1, 1); glVertex3f( hw,  hh, 0)
    glTexCoord2f(0, 1); glVertex3f(-hw,  hh, 0)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_grid(size=4, step=0.5):
    glColor3f(0.25, 0.25, 0.25)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    x = -size
    while x <= size + 1e-6:
        glVertex3f(x, -size, -0.01)
        glVertex3f(x,  size, -0.01)
        x += step
    y = -size
    while y <= size + 1e-6:
        glVertex3f(-size, y, -0.01)
        glVertex3f( size, y, -0.01)
        y += step
    glEnd()


def draw_axes(length=0.4):
    glLineWidth(2.5)
    glBegin(GL_LINES)
    glColor3f(1, 0.2, 0.2); glVertex3f(0,0,0); glVertex3f(length,0,0)   # X red
    glColor3f(0.2, 1, 0.2); glVertex3f(0,0,0); glVertex3f(0,length,0)   # Y green
    glColor3f(0.2, 0.5, 1); glVertex3f(0,0,0); glVertex3f(0,0,length)   # Z blue
    glEnd()
    glLineWidth(1.0)


def draw_camera_trail(trail, max_len):
    """Draw the camera motion path as a fading line + small dot per sample."""
    n = len(trail)
    if n < 2:
        return
    glLineWidth(2.0)
    glBegin(GL_LINE_STRIP)
    for i, (x, y, z) in enumerate(trail):
        alpha = (i + 1) / n
        glColor3f(0.2 + 0.8 * alpha, 0.8 * (1 - alpha), 1.0 * alpha)
        glVertex3f(x, y, z)
    glEnd()
    glLineWidth(1.0)

    # Current head dot
    if trail:
        x, y, z = trail[-1]
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.9, 0.1)
        glVertex3f(x, y, z)
        glEnd()
        glPointSize(1.0)


def draw_motion_vectors(trail, scale=2.0):
    """Draw small arrows between consecutive trail points."""
    if len(trail) < 2:
        return
    glLineWidth(1.5)
    glBegin(GL_LINES)
    for i in range(len(trail) - 1):
        x0, y0, z0 = trail[i]
        x1, y1, z1 = trail[i + 1]
        alpha = (i + 1) / len(trail)
        glColor4f(0.4, 1.0, 0.4, alpha)
        glVertex3f(x0, y0, z0)
        glVertex3f(x1, y1, z1)
    glEnd()
    glLineWidth(1.0)


def draw_hud_text(surface, font, lines, x=14, y=14, color=(220, 220, 220)):
    """Render HUD text onto a pygame surface (blitted over OpenGL)."""
    for i, line in enumerate(lines):
        txt = font.render(line, True, color)
        surface.blit(txt, (x, y + i * 18))


# ---------------------------------------------------------------------------
# Orbit camera
# ---------------------------------------------------------------------------

class OrbitCamera:
    def __init__(self, distance=3.5, yaw=30.0, pitch=25.0):
        self.dist   = distance
        self.yaw    = yaw       # degrees, around world-Y
        self.pitch  = pitch     # degrees, elevation
        self.target = [0.0, 0.0, 0.0]
        self.min_dist = 0.5
        self.max_dist = 20.0

    def apply(self):
        glLoadIdentity()
        # Position camera in spherical coords around target.
        rad_y = math.radians(self.yaw)
        rad_p = math.radians(self.pitch)
        ex = self.dist * math.cos(rad_p) * math.sin(rad_y)
        ey = self.dist * math.sin(rad_p)
        ez = self.dist * math.cos(rad_p) * math.cos(rad_y)
        tx, ty, tz = self.target
        gluLookAt(tx + ex, ty + ey, tz + ez,
                  tx,      ty,      tz,
                  0, 1, 0)

    def orbit(self, dyaw, dpitch):
        self.yaw   += dyaw
        self.pitch  = max(-89.0, min(89.0, self.pitch + dpitch))

    def pan(self, dx, dy):
        rad_y = math.radians(self.yaw)
        # Right vector (horizontal pan)
        rx =  math.cos(rad_y)
        rz = -math.sin(rad_y)
        scale = self.dist * 0.001
        self.target[0] -= rx * dx * scale
        self.target[2] -= rz * dx * scale
        self.target[1] += dy * scale

    def zoom(self, delta):
        self.dist = max(self.min_dist, min(self.max_dist, self.dist - delta * 0.15))

    def reset(self):
        self.dist   = 3.5
        self.yaw    = 30.0
        self.pitch  = 25.0
        self.target = [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="3D optical-flow POV viewer with OpenGL.")
    p.add_argument("--input",  required=True, help="Path to input video.")
    p.add_argument("--width",  type=int, default=640,  help="Processing width (0=native).")
    p.add_argument("--win-w",  type=int, default=1280, help="Window width.")
    p.add_argument("--win-h",  type=int, default=720,  help="Window height.")
    p.add_argument("--trail-len", type=int, default=300,
                   help="Number of past camera positions to show in trail.")

    # Flow / motion params (same defaults as main script).
    p.add_argument("--motion-threshold", type=float, default=0.2)
    p.add_argument("--deadzone",         type=float, default=0.15)
    p.add_argument("--tz-scale",         type=float, default=0.10)
    p.add_argument("--sensitivity",      type=float, default=0.9)
    p.add_argument("--rot-scale",        type=float, default=0.01)
    p.add_argument("--inertia-damping",  type=float, default=0.82)
    p.add_argument("--inertia-influence",type=float, default=0.15)
    p.add_argument("--alpha",            type=float, default=0.25)
    p.add_argument("--translation-gain", type=float, default=0.05)
    p.add_argument("--max-translation-px", type=float, default=12.0)
    p.add_argument("--max-rot-deg",      type=float, default=2.5)
    p.add_argument("--zoom-gain",        type=float, default=0.018)
    p.add_argument("--max-zoom-delta",   type=float, default=0.06)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.input}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    if not ok:
        sys.exit("Cannot read first frame.")

    def maybe_resize(img):
        if args.width and args.width > 0:
            h0, w0 = img.shape[:2]
            s = args.width / float(w0)
            return cv2.resize(img, (int(round(w0*s)), int(round(h0*s))),
                              interpolation=cv2.INTER_AREA)
        return img

    first_frame = maybe_resize(first_frame)
    fh, fw = first_frame.shape[:2]
    aspect = fw / fh

    # ---- Pygame + OpenGL window --------------------------------------------
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 14)

    WIN_W, WIN_H = args.win_w, args.win_h
    screen = pygame.display.set_mode(
        (WIN_W, WIN_H),
        DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("POV 3D Optical-Flow Viewer")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.08, 0.08, 0.10, 1.0)

    def set_projection(w, h):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(55.0, w / max(h, 1), 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)

    set_projection(WIN_W, WIN_H)

    video_tex = create_texture()
    upload_frame_to_texture(video_tex, first_frame)

    # ---- Motion state ------------------------------------------------------
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), (5,5), 0)

    rot_scale  = args.sensitivity * args.rot_scale
    max_rot_r  = math.radians(args.max_rot_deg)
    damping    = args.inertia_damping
    influence  = args.inertia_influence
    smooth     = args.alpha
    deadzone   = args.deadzone

    yaw = pitch = 0.0
    yaw_v = pitch_v = 0.0
    yaw_s = pitch_s = 0.0
    tx_s = ty_s = tz_s = 0.0
    tz = tz_v = 0.0

    # Accumulated "virtual camera" position in world space (for trail).
    cam_x = cam_y = cam_z = 0.0
    trail = deque(maxlen=args.trail_len)
    trail.append((cam_x, cam_y, cam_z))

    # ---- Orbit camera ------------------------------------------------------
    orbit = OrbitCamera()

    # ---- Playback state ----------------------------------------------------
    paused     = False
    frame_idx  = 0
    current_frame = first_frame.copy()

    # ---- Mouse state -------------------------------------------------------
    mouse_left_down  = False
    mouse_right_down = False
    last_mouse       = (0, 0)

    clock = pygame.time.Clock()

    # HUD overlay surface (pygame 2D drawn then blitted via glDrawPixels).
    hud_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    # ---- Frame timing (decouple render from video fps) ---------------------
    spf         = 1.0 / fps_video          # seconds per video frame
    accum       = 0.0
    last_time   = time.perf_counter()

    while True:
        now  = time.perf_counter()
        dt   = now - last_time
        last_time = now

        # ---- Events --------------------------------------------------------
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return

            elif event.type == VIDEORESIZE:
                WIN_W, WIN_H = event.w, event.h
                set_projection(WIN_W, WIN_H)
                glViewport(0, 0, WIN_W, WIN_H)
                hud_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

            elif event.type == KEYDOWN:
                if event.key in (K_q, K_ESCAPE):
                    pygame.quit(); return
                elif event.key == K_r:
                    orbit.reset()
                    cam_x = cam_y = cam_z = 0.0
                    trail.clear()
                    trail.append((0.0, 0.0, 0.0))
                    yaw = pitch = yaw_v = pitch_v = yaw_s = pitch_s = 0.0
                    tx_s = ty_s = tz_s = tz = tz_v = 0.0
                elif event.key == K_SPACE:
                    paused = not paused

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_left_down = True
                    last_mouse = event.pos
                elif event.button == 3:
                    mouse_right_down = True
                    last_mouse = event.pos
                elif event.button == 4:
                    orbit.zoom(1.0)
                elif event.button == 5:
                    orbit.zoom(-1.0)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1: mouse_left_down  = False
                if event.button == 3: mouse_right_down = False

            elif event.type == MOUSEMOTION:
                mx, my = event.pos
                lx, ly = last_mouse
                ddx, ddy = mx - lx, my - ly
                last_mouse = event.pos
                if mouse_left_down:
                    orbit.orbit(ddx * 0.4, -ddy * 0.4)
                elif mouse_right_down:
                    orbit.pan(ddx, ddy)

        # ---- Advance video frame -------------------------------------------
        new_frame = False
        if not paused:
            accum += dt
            while accum >= spf:
                accum -= spf
                ok, raw = cap.read()
                if not ok or raw is None:
                    # Loop: rewind.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, raw = cap.read()
                    if not ok or raw is None:
                        break
                    frame_idx = 0
                    # Reset motion state on loop so trail doesn't jump.
                    yaw = pitch = yaw_v = pitch_v = yaw_s = pitch_s = 0.0
                    tx_s = ty_s = tz_s = tz = tz_v = 0.0

                current_frame = maybe_resize(raw)
                new_frame = True
                frame_idx += 1

        # ---- Optical flow + 3D motion (only when a new frame arrived) ------
        if new_frame:
            gray = cv2.GaussianBlur(
                cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), (5,5), 0)
            flow = get_optical_flow(prev_gray, gray)
            prev_gray = gray

            dx, dy, tz_raw, flow_mag = compute_pseudo_3d_motion(
                flow,
                motion_threshold=args.motion_threshold,
                deadzone=deadzone,
                tz_scale=args.tz_scale,
            )

            # Rotation inertia.
            yaw_target   = float(np.clip(-dx * rot_scale, -max_rot_r, max_rot_r))
            pitch_target = float(np.clip(-dy * rot_scale, -max_rot_r, max_rot_r))
            yaw_v   = float(np.clip(yaw_v   * damping + (yaw_target   - yaw)   * influence, -0.02, 0.02))
            pitch_v = float(np.clip(pitch_v * damping + (pitch_target - pitch) * influence, -0.02, 0.02))
            yaw   = float(np.clip(yaw   + yaw_v,   -max_rot_r, max_rot_r))
            pitch = float(np.clip(pitch + pitch_v, -max_rot_r, max_rot_r))
            yaw_s   = smooth * yaw   + (1 - smooth) * yaw_s
            pitch_s = smooth * pitch + (1 - smooth) * pitch_s

            # tz inertia.
            tz_err = tz_raw - tz
            tz_dv  = float(np.clip(tz_v * damping + tz_err * influence - tz_v, -0.004, 0.004))
            tz_v   = float(np.clip(tz_v + tz_dv, -0.02, 0.02))
            tz     = tz + tz_v
            tz_s   = smooth * tz + (1 - smooth) * tz_s

            # Translation.
            tx_cur = float(np.clip(-args.translation_gain * dx,
                                   -args.max_translation_px, args.max_translation_px))
            ty_cur = float(np.clip(-args.translation_gain * dy,
                                   -args.max_translation_px, args.max_translation_px))
            tx_s = smooth * tx_cur + (1 - smooth) * tx_s
            ty_s = smooth * ty_cur + (1 - smooth) * ty_s

            # Accumulate world-space camera position from flow.
            # X = -tx (left/right), Y = -ty (up/down), Z = tz (fwd/back).
            WORLD_SCALE = 0.01
            cam_x += -tx_s   * WORLD_SCALE
            cam_y += -ty_s   * WORLD_SCALE
            cam_z +=  tz_s   * WORLD_SCALE
            trail.append((cam_x, cam_y, cam_z))

            upload_frame_to_texture(video_tex, current_frame)

        # ---- OpenGL render -------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        orbit.apply()

        draw_grid()
        draw_axes()

        # Video quad: placed at origin, tilted slightly so it's visible from default cam.
        glPushMatrix()
        glTranslatef(0, 0, 0)
        draw_video_quad(video_tex, aspect)
        glPopMatrix()

        # Camera trail & motion vectors.
        draw_camera_trail(trail, args.trail_len)
        draw_motion_vectors(trail)

        # Current virtual camera marker (a small cross at cam position).
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.85, 0.1)
        glVertex3f(cam_x, cam_y, cam_z)
        glEnd()
        glPointSize(1.0)

        # Small line from origin to current cam pos.
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)
        glVertex3f(0, 0, 0)
        glVertex3f(cam_x, cam_y, cam_z)
        glEnd()

        # ---- HUD (pygame 2D over OpenGL) -----------------------------------
        # Switch to 2D ortho for HUD.
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIN_W, WIN_H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        hud_surf.fill((0, 0, 0, 0))
        frame_pct = (frame_idx % max(total_frames, 1)) / max(total_frames, 1) * 100
        lines = [
            f"Frame {frame_idx % max(total_frames,1)}/{total_frames}  ({frame_pct:.0f}%)  {'[PAUSED]' if paused else ''}",
            f"tx:{tx_s:+.3f}  ty:{ty_s:+.3f}  tz:{tz_s:+.3f}",
            f"yaw:{math.degrees(yaw_s):+.2f}°  pitch:{math.degrees(pitch_s):+.2f}°",
            f"Cam pos  x:{cam_x:+.3f}  y:{cam_y:+.3f}  z:{cam_z:+.3f}",
            "",
            "LMB drag: orbit   RMB drag: pan   Scroll: zoom",
            "Space: pause   R: reset   Q/Esc: quit",
        ]
        draw_hud_text(hud_surf, font, lines)

        # Blit HUD via glDrawPixels (works without extra FBOs).
        hud_str = pygame.image.tostring(hud_surf, "RGBA", True)
        glRasterPos2i(0, 0)
        glDrawPixels(WIN_W, WIN_H, GL_RGBA, GL_UNSIGNED_BYTE, hud_str)

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()