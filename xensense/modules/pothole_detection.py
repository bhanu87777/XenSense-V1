"""Pothole and speed-bump / surface anomaly detection module.

Report 4.2: "Pothole and Speed Bump Detection: an attention-enhanced YOLO
variant (CBAM-YOLO) is used to identify surface anomalies ... Proactive path
planning is aided by this feature."
Report 2.4: "Detection with Temporal Awareness: ... integrates temporal
filters and flow cues" (most existing pothole systems are single-image).

Backends:
  1. Custom YOLO weights (weights/pothole_yolo.pt) trained on a pothole
     dataset — the CBAM-YOLO path of the report (train with Ultralytics on
     e.g. the RDD2022 / Roboflow pothole datasets).
  2. Classical fallback: dark, elliptical blobs inside the road ROI (lower
     part of the frame), validated by temporal persistence across frames.
"""

import os
from collections import deque

import cv2
import numpy as np


class PotholeDetector:
    def __init__(self, yolo_weights: str = "weights/pothole_yolo.pt",
                 roi_top: float = 0.55, min_area: int = 400,
                 max_area_frac: float = 0.05, persistence: int = 3):
        self.roi_top = roi_top
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.persistence = persistence
        self.history: deque = deque(maxlen=5)   # recent candidate lists (temporal filter)
        self.model = self._load_yolo(yolo_weights)

    @staticmethod
    def _load_yolo(path: str):
        if not os.path.exists(path):
            return None
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            print(f"[XenSense] Loaded pothole YOLO weights from {path}")
            return model
        except Exception as exc:
            print(f"[XenSense] Could not load pothole model ({exc}); using heuristic backend.")
            return None

    # ---------------- YOLO backend ----------------
    def _detect_yolo(self, image: np.ndarray) -> list[tuple]:
        results = self.model.predict(image, conf=0.35, verbose=False)[0]
        out = []
        if results.boxes is not None:
            for box in results.boxes.xyxy.cpu().numpy():
                out.append(tuple(int(v) for v in box))
        return out

    # ---------------- classical backend ----------------
    def _detect_heuristic(self, image: np.ndarray, gray: np.ndarray) -> list[tuple]:
        h, w = gray.shape
        top = int(h * self.roi_top)
        roi = gray[top:, :]
        roi = cv2.medianBlur(roi, 5)
        # saturation channel of the ROI: asphalt is grey, vegetation/dirt is not
        sat = cv2.cvtColor(image[top:, :], cv2.COLOR_BGR2HSV)[:, :, 1]

        # potholes appear darker than surrounding asphalt
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 41, 12)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        candidates = []
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = h * w * self.max_area_frac
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= max_area):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if not (0.5 <= aspect <= 5.0):        # potholes: wide-ish ellipses
                continue
            solidity = area / max(bw * bh, 1)
            if solidity < 0.4:                     # reject stringy shadows / lane paint
                continue
            # contrast check: pothole interior must be clearly darker than the
            # surrounding asphalt ring (rejects texture noise / faded patches)
            pad = 12
            ry0, ry1 = max(0, y - pad), min(roi.shape[0], y + bh + pad)
            rx0, rx1 = max(0, x - pad), min(roi.shape[1], x + bw + pad)
            inner = roi[y:y + bh, x:x + bw].mean()
            ring = roi[ry0:ry1, rx0:rx1].mean()
            if inner > ring * 0.82:
                continue
            # potholes are surrounded by asphalt: reject blobs sitting in
            # saturated surroundings (vegetation, dirt shoulders)
            if sat[ry0:ry1, rx0:rx1].mean() > 55:
                continue
            candidates.append((x, y + top, x + bw, y + bh + top))
        return candidates

    @staticmethod
    def _suppress_under_objects(candidates: list[tuple],
                                exclude_boxes: list[tuple]) -> list[tuple]:
        """Vehicles cast dark shadows beneath them that mimic potholes; drop
        candidates whose centre falls inside (a slightly grown) detection box."""
        kept = []
        for x1, y1, x2, y2 in candidates:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            inside = False
            for bx1, by1, bx2, by2 in exclude_boxes:
                gw = (bx2 - bx1) * 0.15
                gh = (by2 - by1) * 0.3
                if bx1 - gw <= cx <= bx2 + gw and by1 <= cy <= by2 + gh:
                    inside = True
                    break
            if not inside:
                kept.append((x1, y1, x2, y2))
        return kept

    # ---------------- temporal persistence filter ----------------
    def _persistent(self, candidates: list[tuple]) -> list[tuple]:
        """Keep only candidates seen in >= `persistence` of the last 5 frames."""
        self.history.append(candidates)
        confirmed = []
        for box in candidates:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            hits = 0
            for past in self.history:
                for pb in past:
                    pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                    if abs(cx - pcx) < 40 and abs(cy - pcy) < 40:
                        hits += 1
                        break
            if hits >= self.persistence:
                confirmed.append(box)
        return confirmed

    def update(self, image: np.ndarray, gray: np.ndarray,
               exclude_boxes: list[tuple] | None = None) -> list[tuple]:
        if self.model is not None:
            candidates = self._detect_yolo(image)
        else:
            candidates = self._detect_heuristic(image, gray)
            if exclude_boxes:
                candidates = self._suppress_under_objects(candidates, exclude_boxes)
        return self._persistent(candidates)
