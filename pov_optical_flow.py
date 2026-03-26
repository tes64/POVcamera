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


def apply_camera_transform(
    frame: np.ndarray,
    yaw: float,
    pitch: float,
    tx: float,
    ty: float,
    focal_length: float,
    roll: float = 0.0,
) -> np.ndarray:
    """
    Approximate camera yaw/pitch by applying a small 3D rotation warp as a homography.
    Then add a small image-plane translation.
    """
    h, w = frame.shape[:2]
    cx, cy = w * 0.5, h * 0.5

    # Camera intrinsics (K) in pixel units.
    K = np.array(
        [
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Rotation matrices for small rotations.
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    cpitch, spitch = math.cos(pitch), math.sin(pitch)
    croll, sroll = math.cos(roll), math.sin(roll)

    # Yaw: rotate about Y axis (camera turns left/right)
    R_yaw = np.array(
        [
            [cyaw, 0.0, syaw],
            [0.0, 1.0, 0.0],
            [-syaw, 0.0, cyaw],
        ],
        dtype=np.float64,
    )

    # Pitch: rotate about X axis (camera tilts up/down)
    R_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cpitch, -spitch],
            [0.0, spitch, cpitch],
        ],
        dtype=np.float64,
    )

    # Optional roll (in case you want a bit more handheld feel).
    R_roll = np.array(
        [
            [croll, -sroll, 0.0],
            [sroll, croll, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Compose rotations. Order here is not "physically perfect" but works well for small angles.
    R = R_roll @ R_pitch @ R_yaw

    # Homography for pure rotation: H = K * R * K^-1
    K_inv = np.linalg.inv(K)
    H = K @ R @ K_inv

    # Image-plane translation: incorporate via an additional 2D translation transform.
    T = np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    H = T @ H

    warped = cv2.warpPerspective(
        frame,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def parse_args():
    p = argparse.ArgumentParser(description="POV camera simulation from optical flow (Farneback).")
    p.add_argument("--input", required=True, help="Path to input video.")
    p.add_argument("--output", default="", help="Optional path to output video (e.g. out.mp4).")
    p.add_argument("--width", type=int, default=640, help="Resize width for speed (set 0 to disable).")

    # Optical flow params (keep modest for near real-time).
    p.add_argument("--flow-pyr-scale", type=float, default=0.5)
    p.add_argument("--flow-levels", type=int, default=3)
    p.add_argument("--flow-winsize", type=int, default=15)
    p.add_argument("--flow-iterations", type=int, default=3)
    p.add_argument("--flow-poly-n", type=int, default=5)
    p.add_argument("--flow-poly-sigma", type=float, default=1.2)
    p.add_argument("--flow-flags", type=int, default=0)
    p.add_argument("--motion-threshold", type=float, default=0.2, help="Filter small flow vectors (pixels).")

    # Controls -> camera effect tuning.
    p.add_argument("--sensitivity", type=float, default=0.9, help="How strongly flow drives yaw/pitch.")
    p.add_argument("--translation-gain", type=float, default=0.05, help="Image-plane translation gain (pixels per pixel flow).")
    p.add_argument("--max-rot-deg", type=float, default=2.5, help="Clamp yaw/pitch in degrees for stability.")
    p.add_argument("--focal-length-ratio", type=float, default=0.9, help="Focal length as fraction of frame width.")
    p.add_argument("--roll-sensitivity", type=float, default=0.0, help="Optional roll induced from yaw (subtle handheld).")
    p.add_argument(
        "--max-translation-px",
        type=float,
        default=12.0,
        help="Clamp image-plane translation for stability (pixels).",
    )

    # Smoothing (exponential).
    p.add_argument("--alpha", type=float, default=0.2, help="Exponential smoothing factor (0..1). Higher = less smoothing.")

    # Head-bob.
    p.add_argument("--bob-amplitude", type=float, default=1.2, help="Vertical bob amplitude in pixels.")
    p.add_argument("--bob-frequency", type=float, default=1.7, help="Bob frequency in Hz.")

    # Display/behavior.
    p.add_argument("--no-preview", action="store_true", help="Disable real-time window.")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-6 else 30.0

    # Read first frame.
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

    # Smoothed camera parameters.
    yaw_s = 0.0
    pitch_s = 0.0
    tx_s = 0.0
    ty_s = 0.0

    alpha = float(args.alpha)
    max_rot_rad = math.radians(args.max_rot_deg)
    focal_length = float(args.focal_length_ratio) * float(w)

    frame_idx = 0
    last_ui_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = maybe_resize(frame)
        if frame.shape[0] != h or frame.shape[1] != w:
            # Resize got different dimensions (should be rare); re-sync output.
            h, w = frame.shape[:2]
            focal_length = float(args.focal_length_ratio) * float(w)
            if out_writer is not None:
                out_writer.release()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h), True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        flow = get_optical_flow(
            prev_gray,
            gray,
            pyr_scale=args.flow_pyr_scale,
            levels=args.flow_levels,
            winsize=args.flow_winsize,
            iterations=args.flow_iterations,
            poly_n=args.flow_poly_n,
            poly_sigma=args.flow_poly_sigma,
            flags=args.flow_flags,
        )

        dx, dy = compute_motion(flow, motion_threshold=args.motion_threshold)
        print(f"dx: {dx:.3f}, dy: {dy:.3f}")
        # draw one big arrow in center showing motion
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        end = (int(center[0] + dx * 10), int(center[1] + dy * 10))
        cv2.arrowedLine(frame, center, end, (0, 255, 0), 3)

        # Convert motion into yaw/pitch.
        # Signs: negative makes the camera move "with" the scene motion.
        yaw_cur = -args.sensitivity * dx * 0.01
        pitch_cur = -args.sensitivity * dy * 0.01

        # Clamp rotations for stability and to avoid warping blow-ups.
        yaw_cur = float(np.clip(yaw_cur, -max_rot_rad, max_rot_rad))
        pitch_cur = float(np.clip(pitch_cur, -max_rot_rad, max_rot_rad))

        # Small translation: proportional to pixel flow (subtle on purpose).
        tx_cur = -args.translation_gain * dx
        ty_cur = -args.translation_gain * dy
        tx_cur = float(np.clip(tx_cur, -args.max_translation_px, args.max_translation_px))
        ty_cur = float(np.clip(ty_cur, -args.max_translation_px, args.max_translation_px))

        # Exponential smoothing to reduce jitter.
        yaw_s = alpha * yaw_cur + (1.0 - alpha) * yaw_s
        pitch_s = alpha * pitch_cur + (1.0 - alpha) * pitch_s
        tx_s = alpha * tx_cur + (1.0 - alpha) * tx_s
        ty_s = alpha * ty_cur + (1.0 - alpha) * ty_s

        # Head bob: subtle vertical sine wave.
        t = frame_idx / fps
        bob = float(args.bob_amplitude) * math.sin(2.0 * math.pi * float(args.bob_frequency) * t)

        ty_bob = ty_s + bob
        roll = float(args.roll_sensitivity) * yaw_s

        warped = apply_camera_transform(
            frame,
            yaw=yaw_s,
            pitch=pitch_s,
            tx=tx_s,
            ty=ty_bob,
            focal_length=focal_length,
            roll=roll,
        )

        if out_writer is not None:
            out_writer.write(warped)

        if not args.no_preview:
            cv2.imshow("pov optical flow preview (press q to quit)", warped)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        prev_gray = gray
        frame_idx += 1

        # Keep UI responsive on slower machines.
        if not args.no_preview and (time.time() - last_ui_time) > 0.5:
            last_ui_time = time.time()

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()


def draw_flow_debug(frame, flow, step=20, scale=3, color=(0, 255, 0)):
    """
    Draw motion vectors (arrows) on the frame.
    step = spacing between points
    scale = how long the arrows are
    """
    h, w = frame.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]

            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)

            cv2.circle(frame, (x, y), 1, color, -1)
            cv2.line(frame, (x, y), (end_x, end_y), color, 1)

    return frame

if __name__ == "__main__":
    main()

