"""Data storage and export module.

Report 4.3: "For analytical purposes, processed results may be optionally
saved in CSV or JSON logs" plus "summary logs for post-processing analysis".
"""

import csv
import json
import os
import time
from collections import Counter

from .segmentation import Detection


class ResultExporter:
    CSV_FIELDS = ["frame", "timestamp", "track_id", "class", "confidence",
                  "x1", "y1", "x2", "y2", "speed_kmh", "direction",
                  "distance_m", "occluded", "smoke", "fog", "n_potholes"]

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"xensense_{stamp}.csv")
        self.json_path = os.path.join(log_dir, f"xensense_{stamp}_summary.json")
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.CSV_FIELDS)
        self._writer.writeheader()

        self.class_counts: Counter = Counter()
        self.unique_ids: dict[str, set] = {}
        self.hazard_events: list[dict] = []
        self._active_hazards: dict[str, int] = {}
        self.frames = 0
        self.pothole_frames = 0

    def log_frame(self, index: int, timestamp: float, detections: list[Detection],
                  hazards: dict | None, potholes: list[tuple]):
        self.frames += 1
        smoke = bool(hazards and hazards.get("smoke"))
        fog = bool(hazards and hazards.get("fog"))
        if potholes:
            self.pothole_frames += 1
        self._track_hazard("smoke", smoke, index)
        self._track_hazard("fog", fog, index)

        rows = detections or [None]
        for det in rows:
            row = {"frame": index, "timestamp": round(timestamp, 3),
                   "smoke": smoke, "fog": fog, "n_potholes": len(potholes)}
            if det is not None:
                a = det.analytics or {}
                row.update(track_id=det.track_id, **{"class": det.cls_name},
                           confidence=round(det.conf, 3),
                           x1=det.bbox[0], y1=det.bbox[1],
                           x2=det.bbox[2], y2=det.bbox[3],
                           speed_kmh=a.get("speed_kmh"),
                           direction=a.get("direction"),
                           distance_m=a.get("distance_m"),
                           occluded=det.ghost)
                if not det.ghost:
                    self.class_counts[det.cls_name] += 1
                    if det.track_id is not None:
                        self.unique_ids.setdefault(det.cls_name, set()).add(det.track_id)
            self._writer.writerow(row)

    def _track_hazard(self, name: str, active: bool, frame: int):
        """Collapse per-frame flags into (start, end) event ranges."""
        if active and name not in self._active_hazards:
            self._active_hazards[name] = frame
        elif not active and name in self._active_hazards:
            self.hazard_events.append(
                {"type": name, "start_frame": self._active_hazards.pop(name),
                 "end_frame": frame - 1})

    def close(self, avg_fps: float | None = None):
        for name, start in self._active_hazards.items():
            self.hazard_events.append(
                {"type": name, "start_frame": start, "end_frame": self.frames - 1})
        summary = {
            "frames_processed": self.frames,
            "average_fps": round(avg_fps, 2) if avg_fps else None,
            "detections_per_class": dict(self.class_counts),
            "unique_objects_per_class": {k: len(v) for k, v in self.unique_ids.items()},
            "hazard_events": self.hazard_events,
            "frames_with_potholes": self.pothole_frames,
            "csv_log": self.csv_path,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self._file.close()
        return summary
