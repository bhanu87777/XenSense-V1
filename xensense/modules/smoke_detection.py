"""Smoke / fog environmental hazard detection module.

Report 4.2: "Smoke/Fog Detection: to identify smoke or fog, a lightweight CNN
classifier examines temporal patterns and frame-level texture variations. It
employs temporal filtering to lower false positives."
Report 1.7.1: "Smoke detection: using color and temporal cues, detect motion
patterns and textures that resemble smoke."

Two backends, chosen automatically:
  1. Trained lightweight CNN (weights/smoke_cnn.pt, see scripts/train_smoke.py)
     applied to the whole frame — the deep-learning path described in the report.
  2. Classical colour + temporal + texture heuristic — always available, also
     produces candidate smoke *regions* for localisation. Candidate blobs are
     tracked over time and only count as smoke once they prove *stationary
     drift*: internal motion with little net displacement (cars and other
     moving objects displace; plumes billow in place).
A rolling temporal filter (majority vote over `window` frames) suppresses
single-frame false positives for both backends.
"""

import os
from collections import deque

import cv2
import numpy as np


class SmokeFogDetector:
    def __init__(self, cnn_weights: str = "weights/smoke_cnn.pt",
                 area_threshold: float = 0.02, window: int = 15, min_hits: int = 9,
                 fog_brightness: int = 150, fog_contrast_max: float = 32.0):
        self.area_threshold = area_threshold
        self.min_hits = min_hits
        self.fog_brightness = fog_brightness
        self.fog_contrast_max = fog_contrast_max
        self.smoke_votes: deque = deque(maxlen=window)
        self.fog_votes: deque = deque(maxlen=window)
        self.prev_gray: np.ndarray | None = None
        self.blob_tracks: list[dict] = []   # candidate blob histories (drift test)
        self.cnn = self._load_cnn(cnn_weights)

    # ---------------- CNN backend ----------------
    @staticmethod
    def _load_cnn(path: str):
        if not os.path.exists(path):
            return None
        try:
            import torch
            from ..models.smoke_cnn import SmokeCNN
            model = SmokeCNN()
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            print(f"[XenSense] Loaded smoke CNN weights from {path}")
            return model
        except Exception as exc:
            print(f"[XenSense] Could not load smoke CNN ({exc}); using heuristic backend.")
            return None

    def _cnn_smoke_prob(self, image: np.ndarray) -> float:
        import torch
        inp = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.cnn(tensor)).item())

    # ---------------- heuristic backend ----------------
    def _heuristic_regions(self, image: np.ndarray, gray: np.ndarray,
                           exclude_boxes: list[tuple] | None = None):
        """Colour cue: smoke is grey/desaturated. Temporal cue: it drifts.
        Texture cue: smoke interiors have low edge density."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, sat, val = cv2.split(hsv)
        colour_mask = ((sat < 60) & (val > 90) & (val < 235)).astype(np.uint8)

        camera_moving = False
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            motion_mask = (diff > 8).astype(np.uint8)
            # when most of the frame moves it is ego/camera motion, not smoke drift
            camera_moving = motion_mask.mean() > 0.55
            motion_mask = cv2.dilate(motion_mask, np.ones((9, 9), np.uint8))
        else:
            motion_mask = np.zeros_like(colour_mask)

        # cross-module fusion: motion caused by detected objects (vehicles,
        # pedestrians) is not smoke evidence — blank their (grown) regions
        if exclude_boxes:
            h, w = motion_mask.shape
            for x1, y1, x2, y2 in exclude_boxes:
                gx = int((x2 - x1) * 0.25)
                gy = int((y2 - y1) * 0.25)
                motion_mask[max(0, y1 - gy):min(h, y2 + gy),
                            max(0, x1 - gx):min(w, x2 + gx)] = 0

        candidate = cv2.bitwise_and(colour_mask, motion_mask)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        edges = cv2.Canny(gray, 60, 140)
        regions = []
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_h, frame_w = image.shape[:2]
        frame_area = frame_h * frame_w
        smoke_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < frame_area * 0.005:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # smoke plumes rise: ignore blobs whose centre is on the road surface
            if y + h / 2 > frame_h * 0.65:
                continue
            # smoke interiors are smooth: reject busy-textured blobs (foliage, crowds)
            edge_density = edges[y:y + h, x:x + w].mean() / 255.0
            if edge_density > 0.10:
                continue
            regions.append((x, y, x + w, y + h))

        # motion-coherence gate: only blobs with sustained *in-place* churn
        # qualify as smoke (moving vehicles displace across the frame instead)
        qualified = self._drift_filter(regions)
        smoke_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in qualified)
        area_frac = smoke_area / frame_area
        if camera_moving:
            area_frac *= 0.5   # discount evidence gathered under ego-motion
        return qualified, area_frac

    def _drift_filter(self, regions: list[tuple], min_age: int = 8) -> list[tuple]:
        """Track candidate blobs; keep those alive >= min_age frames whose net
        displacement stays small relative to their size."""
        for tr in self.blob_tracks:
            tr["matched"] = False
        qualified = []
        for x1, y1, x2, y2 in regions:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            size = max(x2 - x1, y2 - y1)
            best = None
            best_d = size  # search radius scales with blob size
            for tr in self.blob_tracks:
                if tr["matched"]:
                    continue
                d = ((cx - tr["cx"]) ** 2 + (cy - tr["cy"]) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                self.blob_tracks.append({"cx": cx, "cy": cy, "ox": cx, "oy": cy,
                                         "age": 1, "matched": True})
                continue
            best.update(cx=cx, cy=cy, matched=True)
            best["age"] += 1
            net = ((cx - best["ox"]) ** 2 + (cy - best["oy"]) ** 2) ** 0.5
            if best["age"] >= min_age and net < 0.6 * size:
                qualified.append((x1, y1, x2, y2))
        self.blob_tracks = [t for t in self.blob_tracks if t["matched"]][-32:]
        return qualified

    def _fog_score(self, gray: np.ndarray) -> bool:
        """Global fog cue: bright, low-contrast frame (washed-out histogram)."""
        return (gray.mean() > self.fog_brightness and
                gray.std() < self.fog_contrast_max)

    # ---------------- public API ----------------
    def update(self, image: np.ndarray, gray: np.ndarray,
               exclude_boxes: list[tuple] | None = None) -> dict:
        regions, area_frac = self._heuristic_regions(image, gray, exclude_boxes)

        if self.cnn is not None:
            smoke_now = self._cnn_smoke_prob(image) > 0.5
        else:
            smoke_now = area_frac >= self.area_threshold

        self.smoke_votes.append(1 if smoke_now else 0)
        self.fog_votes.append(1 if self._fog_score(gray) else 0)
        self.prev_gray = gray.copy()

        return {
            "smoke": sum(self.smoke_votes) >= self.min_hits,
            "fog": sum(self.fog_votes) >= self.min_hits,
            "smoke_regions": regions,
            "smoke_area_fraction": round(float(area_frac), 4),
        }
