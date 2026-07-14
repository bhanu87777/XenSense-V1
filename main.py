"""XenSense-V1 command-line entry point.

Examples
--------
Run on a video file with live preview and annotated output video:
    python main.py --source data/drive.mp4 --output out/annotated.mp4

Run on the default webcam:
    python main.py --source 0

Headless benchmark (no preview window):
    python main.py --source data/drive.mp4 --no-display

Runtime keyboard toggles (preview window focused):
    d detection | o masks | t tracking | m memory | s smoke
    p pothole   | a analytics | f optical flow | r trails | h HUD | q quit
"""

import argparse

from xensense.config import Config
from xensense.pipeline import XenSensePipeline


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        prog="XenSense-V1",
        description="Deep learning-based video segmentation and semantic "
                    "labelling for autonomous driving systems.")
    p.add_argument("--source", default="0",
                   help="video file path or webcam index (default: 0)")
    p.add_argument("--output", default=None, help="annotated output video path (.mp4)")
    p.add_argument("--logs", default="logs", help="directory for CSV/JSON logs")
    p.add_argument("--model", default="yolov8n-seg.pt",
                   help="YOLOv8 segmentation weights (n/s/m variants)")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    p.add_argument("--width", type=int, default=960, help="processing width in pixels")
    p.add_argument("--device", default="", help="'cpu', 'cuda', or '' for auto")
    p.add_argument("--no-display", action="store_true", help="run headless")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after N frames (benchmark / smoke test)")
    p.add_argument("--all-classes", action="store_true",
                   help="detect all 80 COCO classes, not just driving-relevant ones")
    p.add_argument("--no-smoke", action="store_true", help="start with smoke module off")
    p.add_argument("--no-pothole", action="store_true", help="start with pothole module off")
    p.add_argument("--no-tracking", action="store_true", help="start with tracking off")
    p.add_argument("--flow", action="store_true", help="start with optical-flow overlay on")
    args = p.parse_args()

    cfg = Config(source=args.source, output_video=args.output, log_dir=args.logs,
                 yolo_weights=args.model, conf_threshold=args.conf,
                 process_width=args.width, device=args.device,
                 display=not args.no_display, max_frames=args.max_frames)
    if args.all_classes:
        cfg.class_filter = ()
    cfg.toggles.smoke = not args.no_smoke
    cfg.toggles.pothole = not args.no_pothole
    cfg.toggles.tracking = not args.no_tracking
    cfg.toggles.optical_flow = args.flow
    return cfg


if __name__ == "__main__":
    XenSensePipeline(parse_args()).run()
