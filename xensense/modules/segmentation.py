"""YOLOv8 segmentation and semantic labelling module.

Report 4.2: "YOLOv8 Segmentation Module: in charge of identifying and
classifying objects in every frame ... Confidence scores, class labels, and
bounding boxes are all included in the segmentation output."

Outputs per-object bounding boxes, class labels, confidence scores and
instance masks (pixel-level segmentation, report abstract).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple                # (x1, y1, x2, y2) ints at processing resolution
    conf: float
    cls_id: int
    cls_name: str
    mask: np.ndarray | None = None      # uint8 {0,1} mask at frame resolution
    track_id: int | None = None         # filled in by the tracking module
    ghost: bool = False                 # True when hallucinated by the memory module
    analytics: dict = field(default_factory=dict)  # speed / direction / distance

    @property
    def centroid(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class Segmenter:
    """Thin wrapper around Ultralytics YOLOv8-seg."""

    def __init__(self, weights: str = "yolov8n-seg.pt", conf: float = 0.35,
                 iou: float = 0.5, class_filter: tuple = (), device: str = ""):
        from ultralytics import YOLO  # deferred: heavy import
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.class_filter = set(class_filter)
        self.device = device or None
        self.names = self.model.names  # {id: name}

    def __call__(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            image, conf=self.conf, iou=self.iou, device=self.device, verbose=False
        )[0]

        detections: list[Detection] = []
        if results.boxes is None:
            return detections

        h, w = image.shape[:2]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        # mask polygons in original-image coordinates (robust across model input sizes)
        polygons = results.masks.xy if results.masks is not None else [None] * len(boxes)

        for box, conf, cls_id, poly in zip(boxes, confs, clses, polygons):
            name = self.names[cls_id]
            if self.class_filter and name not in self.class_filter:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            mask = None
            if poly is not None and len(poly) >= 3:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

            detections.append(Detection(
                bbox=(x1, y1, x2, y2), conf=float(conf),
                cls_id=int(cls_id), cls_name=name, mask=mask,
            ))
        return detections
