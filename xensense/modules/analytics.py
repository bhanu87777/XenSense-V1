"""Object analytics module: speed, direction, distance, optical flow.

Report 4.2: "Object Analytics Module: uses bounding box centroid shifts to
calculate speed by tracking pixel displacement across frames ...
camera-specific calibration matrices or object size priors are used in
monocular methods for distance estimation."
Report 2.2.5: optical flow (Farneback) for motion estimation.
"""

import math
from collections import deque

import cv2
import numpy as np

from ..config import CLASS_WIDTHS_M
from .segmentation import Detection

COMPASS = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]  # image coords: +y is down


class ObjectAnalytics:
    def __init__(self, fps: float, focal_px_factor: float = 0.9,
                 history_len: int = 30, speed_smooth: int = 5):
        self.fps = fps
        self.focal_px_factor = focal_px_factor
        self.speed_smooth = speed_smooth
        self.history: dict[int, deque] = {}     # track_id -> deque[(t, cx, cy)]
        self.history_len = history_len

    # ---------------- per-object metrics ----------------
    def update(self, detections: list[Detection], frame_w: int, timestamp: float):
        focal_px = frame_w * self.focal_px_factor
        for det in detections:
            if det.track_id is None:
                continue
            cx, cy = det.centroid
            hist = self.history.setdefault(det.track_id,
                                           deque(maxlen=self.history_len))
            hist.append((timestamp, cx, cy))

            # metres-per-pixel scale from the class width prior (monocular)
            real_w = CLASS_WIDTHS_M.get(det.cls_name)
            m_per_px = (real_w / det.width) if (real_w and det.width > 0) else None

            speed_kmh = direction = None
            if len(hist) >= 2:
                # velocity over a short smoothing window
                t0, x0, y0 = hist[max(0, len(hist) - self.speed_smooth - 1)]
                t1, x1, y1 = hist[-1]
                dt = max(t1 - t0, 1e-3)
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt   # px/s
                px_speed = math.hypot(vx, vy)
                if m_per_px is not None:
                    speed_kmh = px_speed * m_per_px * 3.6
                if px_speed > 3:  # ignore jitter
                    angle = math.degrees(math.atan2(vy, vx)) % 360
                    direction = COMPASS[int(((angle + 22.5) % 360) // 45)]

            distance_m = None
            if real_w and det.width > 0:
                distance_m = real_w * focal_px / det.width

            det.analytics = {
                "speed_kmh": round(speed_kmh, 1) if speed_kmh is not None else None,
                "direction": direction,
                "distance_m": round(distance_m, 1) if distance_m is not None else None,
                "trail": [(x, y) for _, x, y in hist],
            }

    def prune(self, live_ids: set):
        for tid in [t for t in self.history if t not in live_ids]:
            del self.history[tid]


class OpticalFlowEstimator:
    """Dense Farneback flow on a downscaled frame -> sparse motion arrows."""

    def __init__(self, scale: float = 0.25, stride: int = 24):
        self.scale = scale
        self.stride = stride
        self.prev_small: np.ndarray | None = None

    def update(self, gray: np.ndarray) -> list[tuple]:
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale)
        arrows = []
        if self.prev_small is not None and self.prev_small.shape == small.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_small, small, None,
                pyr_scale=0.5, levels=2, winsize=13,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            step = max(int(self.stride * self.scale), 4)
            h, w = small.shape
            inv = 1.0 / self.scale
            for y in range(step // 2, h, step):
                for x in range(step // 2, w, step):
                    fx, fy = flow[y, x]
                    if fx * fx + fy * fy > 1.0:   # only meaningful motion
                        arrows.append((int(x * inv), int(y * inv),
                                       int((x + fx * 3) * inv), int((y + fy * 3) * inv)))
        self.prev_small = small
        return arrows
