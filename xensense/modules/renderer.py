"""Output renderer: composes every module's results onto the frame.

Report 4.2: "Output Renderer: creates a final visual output by combining the
outcomes from every module. Bounding boxes, class labels, speed vectors,
condition warnings (such as smoke detected), and trajectory paths for tracked
objects are superimposed on top of each frame. Both offline video writing and
live preview are supported."
"""

import cv2
import numpy as np

from ..config import ModuleToggles
from .segmentation import Detection

# stable per-class colours (BGR)
CLASS_COLOURS = {
    "person": (66, 135, 245), "car": (80, 200, 120), "bus": (0, 165, 255),
    "truck": (0, 120, 255), "motorcycle": (255, 130, 60), "bicycle": (255, 200, 0),
    "traffic light": (60, 60, 230), "stop sign": (40, 40, 220), "train": (200, 100, 255),
}
DEFAULT_COLOUR = (200, 200, 60)
GHOST_COLOUR = (160, 160, 160)


def _colour_for(det: Detection):
    if det.ghost:
        return GHOST_COLOUR
    return CLASS_COLOURS.get(det.cls_name, DEFAULT_COLOUR)


def _id_colour(track_id: int):
    rng = np.random.default_rng(track_id * 7919)
    c = rng.integers(60, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


class Renderer:
    def __init__(self, toggles: ModuleToggles):
        self.toggles = toggles

    def render(self, image: np.ndarray, detections: list[Detection],
               hazards: dict | None, potholes: list[tuple],
               flow_arrows: list[tuple], fps: float) -> np.ndarray:
        out = image.copy()

        # ---------- instance masks ----------
        if self.toggles.masks:
            overlay = out.copy()
            for det in detections:
                if det.mask is not None:
                    overlay[det.mask.astype(bool)] = _colour_for(det)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

        # ---------- optical flow field ----------
        if self.toggles.optical_flow:
            for x0, y0, x1, y1 in flow_arrows:
                cv2.arrowedLine(out, (x0, y0), (x1, y1), (180, 255, 100), 1,
                                tipLength=0.35)

        # ---------- detections / tracks ----------
        for det in detections:
            colour = _colour_for(det)
            x1, y1, x2, y2 = det.bbox
            if det.ghost:
                self._dashed_rect(out, (x1, y1), (x2, y2), colour)
                label = f"#{det.track_id} {det.cls_name} (occluded)"
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
                label = det.cls_name
                if det.track_id is not None and self.toggles.tracking:
                    label = f"#{det.track_id} {label}"
                label += f" {det.conf:.2f}"

            a = det.analytics
            if a and self.toggles.analytics:
                extras = []
                if a.get("speed_kmh") is not None:
                    extras.append(f"{a['speed_kmh']:.0f}km/h")
                if a.get("direction"):
                    extras.append(a["direction"])
                if a.get("distance_m") is not None:
                    extras.append(f"{a['distance_m']:.0f}m")
                if extras:
                    label += " | " + " ".join(extras)

            self._label(out, label, (x1, max(18, y1 - 6)), colour)

            # trajectory paths
            if self.toggles.trails and a and det.track_id is not None:
                trail = a.get("trail", [])
                if len(trail) >= 2:
                    pts = np.array(trail, dtype=np.int32)
                    cv2.polylines(out, [pts], False, _id_colour(det.track_id), 2)

        # ---------- pothole boxes ----------
        if self.toggles.pothole:
            for x1, y1, x2, y2 in potholes:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
                self._label(out, "POTHOLE", (x1, max(18, y1 - 6)), (0, 200, 255))

        # ---------- hazard banners ----------
        banner_y = 34
        if hazards and self.toggles.smoke:
            for x1, y1, x2, y2 in hazards.get("smoke_regions", []):
                cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 1)
            if hazards.get("smoke"):
                self._banner(out, "!! SMOKE DETECTED !!", banner_y, (0, 0, 220))
                banner_y += 38
            if hazards.get("fog"):
                self._banner(out, "!! FOG / LOW VISIBILITY !!", banner_y, (0, 140, 255))
                banner_y += 38

        # ---------- HUD ----------
        if self.toggles.hud:
            self._hud(out, fps, len([d for d in detections if not d.ghost]))
        return out

    # ---------------- drawing helpers ----------------
    @staticmethod
    def _label(image, text, org, colour):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        x, y = org
        cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 3), (20, 20, 20), -1)
        cv2.putText(image, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, colour, 1, cv2.LINE_AA)

    @staticmethod
    def _banner(image, text, y, colour):
        w = image.shape[1]
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        x = (w - tw) // 2
        cv2.rectangle(image, (x - 12, y - 26), (x + tw + 12, y + 10), (20, 20, 20), -1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.85,
                    colour, 2, cv2.LINE_AA)

    @staticmethod
    def _dashed_rect(image, pt1, pt2, colour, dash=8):
        x1, y1 = pt1
        x2, y2 = pt2
        for x in range(x1, x2, dash * 2):
            cv2.line(image, (x, y1), (min(x + dash, x2), y1), colour, 2)
            cv2.line(image, (x, y2), (min(x + dash, x2), y2), colour, 2)
        for y in range(y1, y2, dash * 2):
            cv2.line(image, (x1, y), (x1, min(y + dash, y2)), colour, 2)
            cv2.line(image, (x2, y), (x2, min(y + dash, y2)), colour, 2)

    def _hud(self, image, fps, n_objects):
        t = self.toggles
        rows = [
            f"XenSense-V1  {fps:5.1f} FPS  objects: {n_objects}",
            f"[d]etect:{'ON' if t.detection else 'off'}  "
            f"[o]masks:{'ON' if t.masks else 'off'}  "
            f"[t]rack:{'ON' if t.tracking else 'off'}  "
            f"[m]emory:{'ON' if t.memory else 'off'}",
            f"[s]moke:{'ON' if t.smoke else 'off'}  "
            f"[p]othole:{'ON' if t.pothole else 'off'}  "
            f"[a]nalytics:{'ON' if t.analytics else 'off'}  "
            f"[f]low:{'ON' if t.optical_flow else 'off'}  "
            f"[r]trails:{'ON' if t.trails else 'off'}  [q]uit",
        ]
        h = image.shape[0]
        y0 = h - 20 * len(rows) - 8
        cv2.rectangle(image, (0, y0 - 14), (image.shape[1], h), (25, 25, 25), -1)
        for i, row in enumerate(rows):
            cv2.putText(image, row, (10, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (230, 230, 230), 1, cv2.LINE_AA)
