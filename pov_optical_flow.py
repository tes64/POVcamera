import argparse
import math
import time

import cv2
import numpy as np


def get_optical_flow(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> np.ndarray:
    """
    Dense optical flow using Farneback.

    Returns:
        flow: HxWx2 float32 array, where flow[...,0]=dx and flow[...,1]=dy (pixels per frame).
    """
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )


def compute_motion(flow: np.ndarray, motion_threshold: float = 0.2, min_fraction: float = 0.01):
    """
    Extract average motion (dx, dy) from the optical flow field.

    Args:
        flow: HxWx2 optical flow.
        motion_threshold: only consider vectors with magnitude > this (pixels).
        min_fraction: if fewer than this fraction exceed the threshold, fall back to global mean.
    """
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)

    mask = mag > motion_threshold
    if mask.mean() < min_fraction:
        dx_mean = float(dx.mean())
        dy_mean = float(dy.mean())
    else:
        dx_mean = float(dx[mask].mean())
        dy_mean = float(dy[mask].mean())
    return dx_mean, dy_mean


def compute_pseudo_3d_motion(
    flow: np.ndarray,
    motion_threshold: float,
    deadzone: float,
    tz_scale: float,
    sample_step: int = 24,
):
    """
    Convert 2D flow into a pseudo-3D motion vector (tx, ty, tz) per frame.

    - tx, ty: translation from average flow (pixels/frame) with deadzone applied.
    - tz: forward/backward proxy from average flow magnitude, signed by
      expansion (outward radial flow = moving forward = positive tz) vs
      contraction (inward radial flow = moving backward = negative tz).

    The radial sign uses a coarse sampled dot-product between each flow vector
    and the unit vector pointing away from the image centre.  This avoids any
    look-ahead and is stable for small angles.

    Returns:
        tx, ty   – lateral translation (pixels, deadzone-filtered)
        tz       – forward/back proxy (arbitrary units, signed)
        flow_mag – raw average flow magnitude (useful for bob scaling)
    """
    dx, dy = compute_motion(flow, motion_threshold=motion_threshold)

    # Deadzone filter on lateral translation.
    if abs(dx) < deadzone:
        dx = 0.0
    if abs(dy) < deadzone:
        dy = 0.0

    # Average magnitude across all pixels (speed proxy for tz).
    mag_mean = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))

    # --- Radial flow sign: expand = forward (+tz), contract = backward (-tz) ---
    h, w = flow.shape[:2]
    ys = np.arange(sample_step // 2, h, sample_step, dtype=np.int32)
    xs = np.arange(sample_step // 2, w, sample_step, dtype=np.int32)

    if len(xs) == 0 or len(ys) == 0:
        radial_sign = 1.0
    else:
        xv, yv = np.meshgrid(xs, ys)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5

        # Unit radial vectors pointing away from the image centre.
        rx = (xv.astype(np.float32) - cx)
        ry = (yv.astype(np.float32) - cy)
        r_norm = np.sqrt(rx * rx + ry * ry) + 1e-6
        rx /= r_norm
        ry /= r_norm

        fx = flow[yv, xv, 0].astype(np.float32)
        fy = flow[yv, xv, 1].astype(np.float32)

        # Positive mean → flow points outward → expanding → forward.
        radial_score = float(np.mean(fx * rx + fy * ry))
        radial_sign = 1.0 if radial_score >= 0.0 else -1.0

    # Apply a deadzone to tz as well so pure lateral movement
    # doesn't produce jittery zoom.
    tz_raw = radial_sign * mag_mean * float(tz_scale)
    if abs(mag_mean) < deadzone:
        tz_raw = 0.0

    return dx, dy, tz_raw, mag_mean


def apply_camera_transform(
    frame: np.ndarray,
    yaw: float,
    pitch: float,
    tx: float,
    ty: float,
    focal_length: float,
    roll: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """
    Approximate camera yaw/pitch/roll as a homography (K R K^-1), then add
    image-plane translation and a subtle zoom for the tz forward/back effect.
    """
    h, w = frame.shape[:2]
    cx, cy = w * 0.5, h * 0.5

    K = np.array(
        [[focal_length, 0.0, cx],
         [0.0, focal_length, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    cyaw,  syaw  = math.cos(yaw),   math.sin(yaw)
    cpitch, spitch = math.cos(pitch), math.sin(pitch)
    croll,  sroll  = math.cos(roll),  math.sin(roll)

    R_yaw = np.array(
        [[cyaw,  0.0, syaw],
         [0.0,   1.0, 0.0],
         [-syaw, 0.0, cyaw]],
        dtype=np.float64,
    )
    R_pitch = np.array(
        [[1.0, 0.0,    0.0],
         [0.0, cpitch, -spitch],
         [0.0, spitch,  cpitch]],
        dtype=np.float64,
    )
    R_roll = np.array(
        [[croll, -sroll, 0.0],
         [sroll,  croll, 0.0],
         [0.0,   0.0,   1.0]],
        dtype=np.float64,
    )

    R = R_roll @ R_pitch @ R_yaw
    H = K @ R @ np.linalg.inv(K)

    # Zoom about the image centre (tz forward/back effect).
    if abs(zoom - 1.0) > 1e-6:
        S = np.array(
            [[zoom, 0.0,  cx * (1.0 - zoom)],
             [0.0,  zoom, cy * (1.0 - zoom)],
             [0.0,  0.0,  1.0]],
            dtype=np.float64,
        )
        H = S @ H

    # Image-plane translation.
    T = np.array(
        [[1.0, 0.0, tx],
         [0.0, 1.0, ty],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H = T @ H

    return cv2.warpPerspective(
        frame, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_debug_3d(
    frame: np.ndarray,
    tx: float,
    ty: float,
    tz: float,
    arrow_scale: float = 8.0,
    arrow_min: int = 10,
    arrow_max: int = 80,
) -> None:
    """
    Draw pseudo-3D debug overlays on *frame* in place:
      • Green arrow  – (tx, ty) lateral direction/magnitude.
      • Cyan/Magenta arrow – tz forward (+, cyan) / backward (-, magenta),
        drawn as a vertical arrow whose length encodes magnitude.
      • HUD text with numeric values.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # --- (tx, ty) lateral arrow ---
    mag_xy = math.sqrt(tx * tx + ty * ty)
    if mag_xy > 1e-6:
        ux, uy = tx / mag_xy, ty / mag_xy
        l_xy = int(np.clip(mag_xy * arrow_scale, arrow_min, arrow_max))
        end_xy = (int(cx + ux * l_xy), int(cy + uy * l_xy))
        cv2.arrowedLine(frame, (cx, cy), end_xy, (0, 255, 0), 2, tipLength=0.25)
        cv2.putText(frame, f"tx:{tx:+.2f} ty:{ty:+.2f}",
                    (cx + 6, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    # --- tz vertical arrow ---
    tz_len = int(np.clip(abs(tz) * 600.0, 6, 70))
    if tz_len > 4:
        color_tz = (255, 255, 0) if tz >= 0.0 else (255, 0, 255)   # cyan / magenta
        end_tz = (cx, int(cy - math.copysign(tz_len, tz)))
        cv2.arrowedLine(frame, (cx, cy), end_tz, color_tz, 2, tipLength=0.3)
        label_tz = "FWD" if tz >= 0.0 else "BWD"
        cv2.putText(frame, f"tz:{tz:+.3f} ({label_tz})",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_tz, 1, cv2.LINE_AA)


def parse_args():
    p = argparse.ArgumentParser(description="POV camera simulation from optical flow (Farneback) with pseudo-3D motion.")
    p.add_argument("--input", required=True, help="Path to input video.")
    p.add_argument("--output", default="", help="Optional path to output video (e.g. out.mp4).")
    p.add_argument("--width", type=int, default=640, help="Resize width for speed (set 0 to disable).")

    # Optical flow params.
    p.add_argument("--flow-pyr-scale",   type=float, default=0.5)
    p.add_argument("--flow-levels",      type=int,   default=3)
    p.add_argument("--flow-winsize",     type=int,   default=15)
    p.add_argument("--flow-iterations",  type=int,   default=3)
    p.add_argument("--flow-poly-n",      type=int,   default=5)
    p.add_argument("--flow-poly-sigma",  type=float, default=1.2)
    p.add_argument("--flow-flags",       type=int,   default=0)
    p.add_argument("--motion-threshold", type=float, default=0.2,
                   help="Filter small flow vectors (pixels).")

    # Camera effect tuning.
    p.add_argument("--sensitivity",       type=float, default=0.9,
                   help="Multiplier for motion->rotation scale.")
    p.add_argument("--rot-scale",         type=float, default=0.01,
                   help="Base rotation scale (radians per pixel flow).")
    p.add_argument("--translation-gain",  type=float, default=0.05,
                   help="Image-plane translation gain (pixels per pixel flow).")
    p.add_argument("--max-rot-deg",       type=float, default=2.5,
                   help="Clamp yaw/pitch in degrees for stability.")
    p.add_argument("--focal-length-ratio",type=float, default=0.9,
                   help="Focal length as fraction of frame width.")
    p.add_argument("--roll-factor", "--roll-sensitivity", dest="roll_sensitivity",
                   type=float, default=0.25,
                   help="Camera roll factor based on yaw (subtle handheld).")
    p.add_argument("--max-translation-px", type=float, default=12.0,
                   help="Clamp image-plane translation for stability (pixels).")

    # Deadzone.
    p.add_argument("--deadzone", type=float, default=0.15,
                   help="Ignore flow vectors smaller than this magnitude (pixels).")

    # Inertia (shared for rotation and tz).
    p.add_argument("--inertia-damping",   type=float, default=0.82,
                   help="Damping for velocity (0..1). Higher = longer glide.")
    p.add_argument("--inertia-influence", type=float, default=0.15,
                   help="How strongly target drives velocity.")
    p.add_argument("--max-rot-velocity",  type=float, default=0.02,
                   help="Clamp yaw/pitch velocity (radians/frame).")

    # tz-specific inertia limits (separate from rotation so they're tunable).
    p.add_argument("--max-tz-velocity",   type=float, default=0.02,
                   help="Clamp tz velocity (same units as tz).")
    p.add_argument("--max-tz-accel",      type=float, default=0.004,
                   help="Clamp tz velocity change per frame.")

    # Exponential smoothing.
    p.add_argument("--alpha", type=float, default=0.25,
                   help="Smoothing factor after inertia (0..1). Higher = less smoothing.")

    # Pseudo-3D forward/back (tz → zoom).
    p.add_argument("--tz-scale",       type=float, default=0.10,
                   help="Scale for tz from average flow magnitude.")
    p.add_argument("--zoom-gain",      type=float, default=0.018,
                   help="How strongly tz affects zoom.")
    p.add_argument("--max-zoom-delta", type=float, default=0.06,
                   help="Clamp zoom to 1±delta for stability.")

    # Head-bob (amplitude scaled by flow magnitude for realism).
    p.add_argument("--bob-amplitude",   type=float, default=1.2,
                   help="Head-bob layer1 max amplitude in pixels.")
    p.add_argument("--bob-frequency",   type=float, default=1.7,
                   help="Head-bob layer1 frequency in Hz.")
    p.add_argument("--bob-amplitude2",  type=float, default=0.35,
                   help="Head-bob layer2 max amplitude in pixels.")
    p.add_argument("--bob-frequency2",  type=float, default=3.4,
                   help="Head-bob layer2 frequency in Hz.")
    p.add_argument("--bob-motion-scale",type=float, default=1.0,
                   help="Scale bob amplitude by smoothed flow mag. 0 = always full bob, "
                        "1 = no bob on static scenes.")

    p.add_argument("--max-roll-deg", type=float, default=3.0,
                   help="Clamp roll in degrees.")

    # Debug.
    p.add_argument("--debug-3d",          action="store_true",
                   help="Draw centre arrows for (tx,ty) and tz (forward/back).")
    p.add_argument("--debug-motion-arrow",action="store_true",
                   help="Draw a single arrow showing 2D motion direction (legacy).")

    # Display.
    p.add_argument("--no-preview", action="store_true",
                   help="Disable real-time window.")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-6 else 30.0

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read first frame.")

    def maybe_resize(img):
        if args.width and args.width > 0:
            h0, w0 = img.shape[:2]
            scale = args.width / float(w0)
            w1 = int(round(w0 * scale))
            h1 = int(round(h0 * scale))
            if w1 <= 0 or h1 <= 0:
                return img
            return cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
        return img

    frame = maybe_resize(frame)
    h, w = frame.shape[:2]

    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h), True)
        if not out_writer.isOpened():
            raise RuntimeError(f"Could not open output writer: {args.output}")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    # ---- Camera state ----
    yaw = pitch = 0.0
    yaw_velocity = pitch_velocity = 0.0
    yaw_s = pitch_s = 0.0
    tx_s = ty_s = 0.0

    # ---- tz (pseudo forward/back) state – independent inertia ----
    tz = 0.0
    tz_velocity = 0.0
    tz_s = 0.0

    # ---- Smoothed flow magnitude for bob scaling ----
    flow_mag_s = 0.0

    smooth         = float(args.alpha)
    max_rot_rad    = math.radians(args.max_rot_deg)
    rot_scale      = float(args.sensitivity) * float(args.rot_scale)
    deadzone       = float(args.deadzone)
    damping        = float(args.inertia_damping)
    influence      = float(args.inertia_influence)
    max_rot_vel    = float(args.max_rot_velocity)
    max_tz_vel     = float(args.max_tz_velocity)
    max_tz_accel   = float(args.max_tz_accel)
    max_roll_rad   = math.radians(args.max_roll_deg)
    focal_length   = float(args.focal_length_ratio) * float(w)
    bob_motion_sc  = float(args.bob_motion_scale)

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = maybe_resize(frame)
        if frame.shape[0] != h or frame.shape[1] != w:
            h, w = frame.shape[:2]
            focal_length = float(args.focal_length_ratio) * float(w)
            if out_writer is not None:
                out_writer.release()
                out_writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        flow = get_optical_flow(
            prev_gray, gray,
            pyr_scale=args.flow_pyr_scale,
            levels=args.flow_levels,
            winsize=args.flow_winsize,
            iterations=args.flow_iterations,
            poly_n=args.flow_poly_n,
            poly_sigma=args.flow_poly_sigma,
            flags=args.flow_flags,
        )

        # -------------------------------------------------------
        # 1. Pseudo-3D motion extraction
        # -------------------------------------------------------
        dx, dy, tz_raw, flow_mag = compute_pseudo_3d_motion(
            flow,
            motion_threshold=args.motion_threshold,
            deadzone=deadzone,
            tz_scale=args.tz_scale,
        )

        # Smooth flow magnitude for bob scaling (don't snap).
        flow_mag_s = smooth * flow_mag + (1.0 - smooth) * flow_mag_s

        # -------------------------------------------------------
        # 2. Rotation (yaw / pitch) inertia
        # -------------------------------------------------------
        yaw_target   = float(np.clip(-dx * rot_scale, -max_rot_rad, max_rot_rad))
        pitch_target = float(np.clip(-dy * rot_scale, -max_rot_rad, max_rot_rad))

        yaw_velocity   = float(np.clip(
            yaw_velocity   * damping + (yaw_target   - yaw)   * influence,
            -max_rot_vel, max_rot_vel))
        pitch_velocity = float(np.clip(
            pitch_velocity * damping + (pitch_target - pitch) * influence,
            -max_rot_vel, max_rot_vel))

        yaw   = float(np.clip(yaw   + yaw_velocity,   -max_rot_rad, max_rot_rad))
        pitch = float(np.clip(pitch + pitch_velocity, -max_rot_rad, max_rot_rad))

        yaw_s   = smooth * yaw   + (1.0 - smooth) * yaw_s
        pitch_s = smooth * pitch + (1.0 - smooth) * pitch_s

        # -------------------------------------------------------
        # 3. tz inertia (independent clamps, same pattern)
        # -------------------------------------------------------
        tz_error   = tz_raw - tz
        tz_vel_des = tz_velocity * damping + tz_error * influence
        tz_dv      = float(np.clip(tz_vel_des - tz_velocity, -max_tz_accel, max_tz_accel))
        tz_velocity = float(np.clip(tz_velocity + tz_dv, -max_tz_vel, max_tz_vel))
        tz          = tz + tz_velocity
        tz_s        = smooth * tz + (1.0 - smooth) * tz_s

        # -------------------------------------------------------
        # 4. Image-plane translation (tx, ty) with smoothing
        # -------------------------------------------------------
        tx_cur = float(np.clip(-args.translation_gain * dx,
                               -args.max_translation_px, args.max_translation_px))
        ty_cur = float(np.clip(-args.translation_gain * dy,
                               -args.max_translation_px, args.max_translation_px))
        tx_s = smooth * tx_cur + (1.0 - smooth) * tx_s
        ty_s = smooth * ty_cur + (1.0 - smooth) * ty_s

        # -------------------------------------------------------
        # 5. Rendering effects (bob, roll, zoom)
        # -------------------------------------------------------
        t = frame_idx / fps

        # Bob amplitude scales with motion so it's silent on static scenes.
        bob_scale = min(1.0, flow_mag_s * bob_motion_sc) if bob_motion_sc > 0.0 else 1.0
        bob = bob_scale * (
            float(args.bob_amplitude)  * math.sin(2.0 * math.pi * float(args.bob_frequency)  * t)
            + float(args.bob_amplitude2) * math.sin(2.0 * math.pi * float(args.bob_frequency2) * t)
        )

        ty_bob = ty_s + bob
        roll   = float(np.clip(float(args.roll_sensitivity) * yaw_s, -max_roll_rad, max_roll_rad))

        # tz → subtle zoom about image centre.
        zoom = float(np.clip(
            1.0 + float(args.zoom_gain) * tz_s,
            1.0 - float(args.max_zoom_delta),
            1.0 + float(args.max_zoom_delta),
        ))

        warped = apply_camera_transform(
            frame,
            yaw=yaw_s, pitch=pitch_s,
            tx=tx_s,   ty=ty_bob,
            focal_length=focal_length,
            roll=roll,
            zoom=zoom,
        )

        # -------------------------------------------------------
        # 6. Debug overlays
        # -------------------------------------------------------
        if args.debug_3d:
            draw_debug_3d(warped, tx_s, ty_s, tz_s)

        # Legacy single-arrow debug.
        if args.debug_motion_arrow:
            hw, ww = warped.shape[:2]
            centre = (ww // 2, hw // 2)
            mag = math.sqrt(dx * dx + dy * dy)
            if mag > 1e-6:
                arrow_len = int(np.clip(mag * 8.0, 10, 80))
                end = (int(centre[0] + dx / mag * arrow_len),
                       int(centre[1] + dy / mag * arrow_len))
            else:
                end = centre
            cv2.arrowedLine(warped, centre, end, (0, 255, 0), 3)

        if out_writer is not None:
            out_writer.write(warped)

        if not args.no_preview:
            cv2.imshow("pov optical flow (q to quit)", warped)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        prev_gray = gray
        frame_idx += 1

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()


def draw_flow_debug(frame, flow, step=20, scale=3, color=(0, 255, 0)):
    """Draw a dense grid of motion vectors on the frame (utility, not called in main)."""
    h, w = frame.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            fdx, fdy = flow[y, x]
            end_x = int(x + fdx * scale)
            end_y = int(y + fdy * scale)
            cv2.circle(frame, (x, y), 1, color, -1)
            cv2.line(frame, (x, y), (end_x, end_y), color, 1)
    return frame


if __name__ == "__main__":
    main()
