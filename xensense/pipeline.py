"""XenSense-V1 pipeline orchestrator.

Implements the multi-stage architecture of report chapter 4:
    video input -> preprocessing -> YOLOv8 segmentation
        -> parallel analysis units (tracking, smoke/fog, pothole, analytics)
        -> memory recall (occlusion) -> output renderer -> export logs.

Every module is a stand-alone component (report 1.7.1) and can be toggled at
runtime from the preview window keyboard.
"""

import time

import cv2

from .config import Config
from .modules.analytics import ObjectAnalytics, OpticalFlowEstimator
from .modules.exporter import ResultExporter
from .modules.memory import MemoryBank
from .modules.pothole_detection import PotholeDetector
from .modules.preprocessing import VideoReader
from .modules.renderer import Renderer
from .modules.segmentation import Segmenter
from .modules.smoke_detection import SmokeFogDetector
from .modules.tracking import build_tracker

KEY_TOGGLES = {
    ord("d"): "detection", ord("o"): "masks", ord("t"): "tracking",
    ord("s"): "smoke", ord("p"): "pothole", ord("a"): "analytics",
    ord("f"): "optical_flow", ord("m"): "memory", ord("r"): "trails",
    ord("h"): "hud",
}


class XenSensePipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(f"[XenSense] Opening source: {cfg.source}")
        self.reader = VideoReader(cfg.source, cfg.process_width)

        print(f"[XenSense] Loading YOLOv8 segmentation model ({cfg.yolo_weights}) ...")
        self.segmenter = Segmenter(cfg.yolo_weights, cfg.conf_threshold,
                                   cfg.iou_threshold, cfg.class_filter, cfg.device)
        self.tracker = build_tracker(cfg.max_track_age, cfg.n_init)
        self.memory = MemoryBank(cfg.memory_max_predict, cfg.memory_hist_threshold)
        self.smoke = SmokeFogDetector(cfg.smoke_cnn_weights, cfg.smoke_area_threshold,
                                      cfg.smoke_window, cfg.smoke_min_hits,
                                      cfg.fog_brightness, cfg.fog_contrast_max)
        self.pothole = PotholeDetector(cfg.pothole_yolo_weights, cfg.pothole_roi_top,
                                       cfg.pothole_min_area, cfg.pothole_max_area_frac,
                                       cfg.pothole_persistence)
        self.analytics = ObjectAnalytics(self.reader.fps, cfg.focal_px_factor,
                                         cfg.history_len, cfg.speed_smooth)
        self.flow = OpticalFlowEstimator()
        self.renderer = Renderer(cfg.toggles)
        self.exporter = ResultExporter(cfg.log_dir)
        self.writer = None
        self._fps_ema = 0.0

    # ---------------- single-frame processing ----------------
    def process_frame(self, frame) -> tuple:
        t = self.cfg.toggles
        detections, ghosts, hazards, potholes, arrows = [], [], None, [], []

        if t.detection:
            detections = self.segmenter(frame.image)
            if t.tracking:
                detections = self.tracker.update(detections, frame.image)
            if t.memory and t.tracking:
                ghosts = self.memory.update(detections, frame.image)
            if t.analytics:
                self.analytics.update(detections + ghosts,
                                      frame.image.shape[1], frame.timestamp)
                self.analytics.prune({d.track_id for d in detections + ghosts
                                      if d.track_id is not None})

        if t.smoke:
            hazards = self.smoke.update(frame.image, frame.gray,
                                        [d.bbox for d in detections])
        if t.pothole:
            potholes = self.pothole.update(frame.image, frame.gray,
                                           [d.bbox for d in detections])
        if t.optical_flow:
            arrows = self.flow.update(frame.gray)

        return detections + ghosts, hazards, potholes, arrows

    # ---------------- main loop ----------------
    def run(self):
        cfg = self.cfg
        window = "XenSense-V1"
        if cfg.display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        start = time.time()
        frames_done = 0
        try:
            for frame in self.reader:
                t0 = time.perf_counter()
                detections, hazards, potholes, arrows = self.process_frame(frame)

                dt = time.perf_counter() - t0
                inst_fps = 1.0 / max(dt, 1e-6)
                self._fps_ema = inst_fps if frames_done == 0 else \
                    0.9 * self._fps_ema + 0.1 * inst_fps

                annotated = self.renderer.render(frame.image, detections, hazards,
                                                 potholes, arrows, self._fps_ema)
                self.exporter.log_frame(frame.index, frame.timestamp, detections,
                                        hazards, potholes)

                if cfg.output_video:
                    if self.writer is None:
                        h, w = annotated.shape[:2]
                        self.writer = cv2.VideoWriter(
                            cfg.output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                            self.reader.fps, (w, h))
                    self.writer.write(annotated)

                if cfg.display:
                    cv2.imshow(window, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    if key in KEY_TOGGLES:
                        name = KEY_TOGGLES[key]
                        setattr(cfg.toggles, name, not getattr(cfg.toggles, name))
                        print(f"[XenSense] toggled {name} -> "
                              f"{'ON' if getattr(cfg.toggles, name) else 'OFF'}")

                frames_done += 1
                if cfg.max_frames and frames_done >= cfg.max_frames:
                    print(f"[XenSense] Reached max_frames={cfg.max_frames}, stopping.")
                    break
                if frames_done % 50 == 0:
                    total = self.reader.frame_count or "?"
                    print(f"[XenSense] frame {frames_done}/{total}  "
                          f"pipeline {self._fps_ema:.1f} FPS")
        finally:
            elapsed = max(time.time() - start, 1e-6)
            summary = self.exporter.close(avg_fps=frames_done / elapsed)
            self.reader.release()
            if self.writer is not None:
                self.writer.release()
            if cfg.display:
                cv2.destroyAllWindows()
            print(f"[XenSense] Done. {frames_done} frames @ "
                  f"{frames_done / elapsed:.1f} FPS average.")
            print(f"[XenSense] CSV log    : {self.exporter.csv_path}")
            print(f"[XenSense] JSON summary: {self.exporter.json_path}")
            if cfg.output_video:
                print(f"[XenSense] Annotated video: {cfg.output_video}")
        return summary
