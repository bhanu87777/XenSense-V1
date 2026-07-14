"""DeepSORT multi-object tracking module.

Report 4.2: "DeepSORT Tracking Module: creates motion trajectories by using
the bounding box outputs. To assign unique IDs and guarantee continuity across
frames, DeepSORT combines cosine distance-based appearance embeddings with
Kalman filtering."

Falls back to a lightweight IoU tracker when the deep-sort-realtime package is
unavailable (keeps the pipeline edge-deployable, report 2.4).
"""

import numpy as np

from .segmentation import Detection


class DeepSortTracker:
    """Assigns persistent IDs to detections using deep-sort-realtime."""

    def __init__(self, max_age: int = 30, n_init: int = 2):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.tracker = DeepSort(max_age=max_age, n_init=n_init,
                                max_cosine_distance=0.3, embedder="mobilenet",
                                half=False, bgr=True)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Detection]:
        raw = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            raw.append(([x1, y1, x2 - x1, y2 - y1], det.conf, det.cls_name))

        tracks = self.tracker.update_tracks(raw, frame=frame)

        # associate returned tracks back to our detections via the original index
        tracked: list[Detection] = []
        used = set()
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            idx = self._match_index(tr, detections, used)
            if idx is None:
                continue
            det = detections[idx]
            det.track_id = int(tr.track_id)
            used.add(idx)
            tracked.append(det)
        return tracked

    @staticmethod
    def _match_index(track, detections, used):
        """Match a DeepSORT track to the nearest unused detection by IoU."""
        tl, tt, tr_, tb = track.to_ltrb()
        best, best_iou = None, 0.1
        for i, det in enumerate(detections):
            if i in used:
                continue
            iou = _iou((tl, tt, tr_, tb), det.bbox)
            if iou > best_iou:
                best, best_iou = i, iou
        return best


class IoUTracker:
    """Dependency-free fallback tracker (greedy IoU + centroid matching)."""

    def __init__(self, max_age: int = 30, n_init: int = 2):
        self.next_id = 1
        self.tracks: dict[int, dict] = {}   # id -> {bbox, cls, age, hits}
        self.max_age = max_age
        self.n_init = n_init

    def update(self, detections: list[Detection], frame=None) -> list[Detection]:
        # age all tracks
        for t in self.tracks.values():
            t["age"] += 1

        assigned = set()
        for det in detections:
            best_id, best_iou = None, 0.25
            for tid, t in self.tracks.items():
                if tid in assigned or t["cls"] != det.cls_name:
                    continue
                iou = _iou(t["bbox"], det.bbox)
                if iou > best_iou:
                    best_id, best_iou = tid, iou
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {"bbox": det.bbox, "cls": det.cls_name,
                                        "age": 0, "hits": 0}
            t = self.tracks[best_id]
            t.update(bbox=det.bbox, age=0)
            t["hits"] += 1
            det.track_id = best_id
            assigned.add(best_id)

        # prune stale tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if t["age"] <= self.max_age}
        return [d for d in detections
                if d.track_id is not None and self.tracks[d.track_id]["hits"] >= self.n_init]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def build_tracker(max_age: int = 30, n_init: int = 2):
    """Prefer DeepSORT; degrade gracefully to the IoU tracker."""
    try:
        return DeepSortTracker(max_age=max_age, n_init=n_init)
    except Exception as exc:  # package missing or embedder init failure
        print(f"[XenSense] DeepSORT unavailable ({exc}); using IoU fallback tracker.")
        return IoUTracker(max_age=max_age, n_init=n_init)
