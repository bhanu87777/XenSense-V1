"""Frame extraction and preprocessing module.

Report 1.7.1: "Frame extraction and preprocessing: resizing, denoising, and
converting video input to frames."

Frames are timestamped and indexed (report 4.3: frames are "stored in a buffer
with a timestamp and frame index").
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    timestamp: float          # seconds since start of stream
    image: np.ndarray         # BGR, resized to processing resolution
    gray: np.ndarray          # grayscale copy (shared by flow / smoke / pothole modules)
    original_size: tuple      # (w, h) of the raw input


class VideoReader:
    """Wraps cv2.VideoCapture: yields preprocessed Frame objects."""

    def __init__(self, source: str, process_width: int = 960, denoise: bool = True):
        # webcam index or file path
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        self.process_width = process_width
        self.denoise = denoise
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps <= 1 or self.fps > 240:  # webcams often report 0
            self.fps = 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._index = 0

    @property
    def size(self) -> tuple:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w > self.process_width:
            scale = self.process_width / w
            image = cv2.resize(image, (self.process_width, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        if self.denoise:
            # light Gaussian denoise; heavy NLM would break the real-time budget
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return image

    def read(self) -> Frame | None:
        ok, raw = self.cap.read()
        if not ok:
            return None
        image = self._preprocess(raw)
        frame = Frame(
            index=self._index,
            timestamp=self._index / self.fps,
            image=image,
            gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            original_size=(raw.shape[1], raw.shape[0]),
        )
        self._index += 1
        return frame

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def release(self):
        self.cap.release()
