"""XenSense-V1: A Deep Learning-Based Framework for Video Segmentation
and Semantic Labelling in Autonomous Driving Systems.

Modular real-time pipeline combining:
  - YOLOv8 instance segmentation + semantic labelling
  - DeepSORT multi-object tracking
  - Smoke / fog hazard detection (CNN + temporal heuristics)
  - Pothole / surface anomaly detection
  - Object analytics (speed, direction, distance estimation)
  - Lightweight memory recall for transient occlusion handling
"""

__version__ = "1.0.0"
__title__ = "XenSense-V1"
