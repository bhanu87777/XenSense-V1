"""Central configuration for the XenSense-V1 pipeline.

Every module can be toggled on/off (Functional Requirement: "Enable modular
toggling of functionalities", report section 3.2). Thresholds gathered here so
tuning does not require touching module code.
"""

from dataclasses import dataclass, field


@dataclass
class ModuleToggles:
    """Runtime on/off switches for each pipeline module (keyboard-controlled)."""
    detection: bool = True       # YOLOv8 segmentation module        [key: d]
    masks: bool = True           # draw instance masks overlay       [key: o]
    tracking: bool = True        # DeepSORT tracking module          [key: t]
    smoke: bool = True           # smoke / fog hazard detection      [key: s]
    pothole: bool = True         # pothole / surface anomaly module  [key: p]
    analytics: bool = True       # speed / direction / distance      [key: a]
    optical_flow: bool = False   # global motion field overlay       [key: f]
    memory: bool = True          # occlusion memory recall           [key: m]
    trails: bool = True          # trajectory paths                  [key: r]
    hud: bool = True             # status heads-up display           [key: h]


@dataclass
class Config:
    # ---------------- input / output ----------------
    source: str = "0"                     # video path or webcam index
    output_video: str | None = None       # annotated video file (optional)
    log_dir: str = "logs"                 # CSV / JSON export directory
    display: bool = True                  # live preview window
    process_width: int = 960              # frames resized to this width (HD-ish, keeps real-time)
    max_frames: int | None = None         # stop after N frames (benchmarks / smoke tests)

    # ---------------- detection ----------------
    yolo_weights: str = "yolov8n-seg.pt"  # nano seg model: edge-deployable (Jetson-class)
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    # COCO classes relevant to driving scenes; empty set = all classes
    class_filter: tuple = (
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "traffic light", "stop sign", "train", "dog", "cow",
    )
    device: str = ""                      # "" = auto (cuda if available), "cpu" to force

    # ---------------- tracking ----------------
    max_track_age: int = 30               # frames a track survives without detection
    n_init: int = 2                       # detections before a track is confirmed

    # ---------------- memory module ----------------
    memory_max_predict: int = 25          # frames to keep predicting an occluded object
    memory_hist_threshold: float = 0.5    # min histogram correlation for re-identification

    # ---------------- smoke / fog detection ----------------
    smoke_cnn_weights: str = "weights/smoke_cnn.pt"   # used when present, else heuristic
    smoke_area_threshold: float = 0.02    # min fraction of frame covered by qualified smoke blobs
    smoke_window: int = 15                # temporal filter window (frames)
    smoke_min_hits: int = 9               # frames within window required to raise alert
    fog_brightness: int = 150             # global fog heuristics
    fog_contrast_max: float = 32.0

    # ---------------- pothole detection ----------------
    pothole_yolo_weights: str = "weights/pothole_yolo.pt"  # custom CBAM-YOLO weights if trained
    pothole_roi_top: float = 0.55         # road region: lower 45% of the frame
    pothole_min_area: int = 400           # px^2 (at process resolution)
    pothole_max_area_frac: float = 0.05
    pothole_persistence: int = 3          # must appear in >=3 of last 5 frames

    # ---------------- analytics ----------------
    focal_px_factor: float = 0.9          # focal length prior ~ 0.9 * frame width (monocular)
    history_len: int = 30                 # centroid history per track
    speed_smooth: int = 5                 # frames used for velocity smoothing

    toggles: ModuleToggles = field(default_factory=ModuleToggles)


# Real-world object size priors (metres) for monocular distance / speed estimation
# (report 4.2: "object size priors are used in monocular methods for distance estimation")
CLASS_WIDTHS_M = {
    "car": 1.8, "bus": 2.55, "truck": 2.5, "motorcycle": 0.8,
    "bicycle": 0.6, "person": 0.5, "train": 3.0, "dog": 0.3, "cow": 0.7,
}
CLASS_HEIGHTS_M = {
    "car": 1.5, "bus": 3.2, "truck": 3.4, "motorcycle": 1.3,
    "bicycle": 1.1, "person": 1.7, "train": 4.0, "dog": 0.5, "cow": 1.4,
}
