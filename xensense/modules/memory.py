"""Lightweight memory recall module for transient occlusion handling.

Report 2.4: "Modular Integration of Memory: in order to manage transient
occlusion and preserve context, we intend to deploy a lightweight memory
recall module that draws inspiration from STM and SRNet."

Instead of a full space-time memory network (too heavy for edge devices,
report 2.3), the module keeps a compact memory entry per track:
  * last bounding box + smoothed velocity  -> motion prediction while occluded
  * HSV colour histogram (appearance key)  -> re-identification on reappearance
When a confirmed track vanishes, its position is extrapolated for up to
`max_predict` frames and rendered as a "ghost" box. When a *new* track appears
that matches a remembered appearance, the old ID is restored so trajectories
and analytics stay continuous through the occlusion.
"""

import cv2
import numpy as np

from .segmentation import Detection


class MemoryEntry:
    __slots__ = ("bbox", "velocity", "hist", "cls_name", "missing", "mask")

    def __init__(self, bbox, velocity, hist, cls_name):
        self.bbox = bbox
        self.velocity = velocity   # (vx, vy) px/frame
        self.hist = hist
        self.cls_name = cls_name
        self.missing = 0


class MemoryBank:
    def __init__(self, max_predict: int = 25, hist_threshold: float = 0.5):
        self.entries: dict[int, MemoryEntry] = {}
        self.max_predict = max_predict
        self.hist_threshold = hist_threshold
        self.id_alias: dict[int, int] = {}   # new tracker id -> remembered original id

    # ---------------- appearance key ----------------
    @staticmethod
    def _histogram(image: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    # ---------------- main update ----------------
    def update(self, detections: list[Detection], image: np.ndarray) -> list[Detection]:
        """Refresh memory with live tracks; emit ghost predictions for lost ones."""
        h, w = image.shape[:2]
        live_ids = set()

        for det in detections:
            if det.track_id is None:
                continue
            # restore identity across occlusion gaps
            if det.track_id in self.id_alias:
                det.track_id = self.id_alias[det.track_id]
            elif det.track_id not in self.entries:
                recovered = self._reidentify(det, image)
                if recovered is not None:
                    self.id_alias[det.track_id] = recovered
                    det.track_id = recovered

            live_ids.add(det.track_id)
            prev = self.entries.get(det.track_id)
            cx, cy = det.centroid
            if prev is not None:
                px = (prev.bbox[0] + prev.bbox[2]) // 2
                py = (prev.bbox[1] + prev.bbox[3]) // 2
                # exponential smoothing of the motion vector
                vx = 0.6 * prev.velocity[0] + 0.4 * (cx - px)
                vy = 0.6 * prev.velocity[1] + 0.4 * (cy - py)
            else:
                vx = vy = 0.0
            hist = self._histogram(image, det.bbox)
            entry = MemoryEntry(det.bbox, (vx, vy), hist, det.cls_name)
            self.entries[det.track_id] = entry

        # ---------------- predict occluded objects ----------------
        ghosts: list[Detection] = []
        for tid, entry in list(self.entries.items()):
            if tid in live_ids:
                entry.missing = 0
                continue
            entry.missing += 1
            if entry.missing > self.max_predict:
                del self.entries[tid]
                continue
            x1, y1, x2, y2 = entry.bbox
            vx, vy = entry.velocity
            bbox = (int(x1 + vx), int(y1 + vy), int(x2 + vx), int(y2 + vy))
            entry.bbox = bbox
            # drop predictions that left the frame
            if bbox[2] < 0 or bbox[0] > w or bbox[3] < 0 or bbox[1] > h:
                del self.entries[tid]
                continue
            ghosts.append(Detection(
                bbox=bbox, conf=0.0, cls_id=-1, cls_name=entry.cls_name,
                track_id=tid, ghost=True,
            ))
        return ghosts

    def _reidentify(self, det: Detection, image: np.ndarray) -> int | None:
        """Match a brand-new track against remembered (occluded) appearances."""
        hist = self._histogram(image, det.bbox)
        if hist is None:
            return None
        best_id, best_score = None, self.hist_threshold
        for tid, entry in self.entries.items():
            if entry.missing == 0 or entry.cls_name != det.cls_name or entry.hist is None:
                continue
            # gate spatially: reappearance must be near the predicted position
            pcx = (entry.bbox[0] + entry.bbox[2]) / 2
            pcy = (entry.bbox[1] + entry.bbox[3]) / 2
            cx, cy = det.centroid
            if abs(cx - pcx) > det.width * 3 or abs(cy - pcy) > det.height * 3:
                continue
            score = cv2.compareHist(entry.hist, hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_id, best_score = tid, score
        return best_id
