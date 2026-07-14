"""Generate a small synthetic test clip (data/synthetic_test.mp4).

Useful for a quick end-to-end pipeline check without a real driving video:
moving rectangles ("cars"), a drifting grey plume (smoke heuristic) and dark
road blobs (pothole heuristic). YOLO won't classify the shapes, but every
classical module and the renderer/exporter path can be exercised.
"""

import os

import cv2
import numpy as np

W, H, FPS, SECONDS = 960, 540, 25, 8


def main(out_path="data/synthetic_test.mp4"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 12, (H, W), dtype=np.uint8)

    for i in range(FPS * SECONDS):
        frame = np.full((H, W, 3), (105, 105, 105), np.uint8)      # asphalt
        frame[: H // 2] = (200, 160, 120)                          # sky
        # lane markings scrolling toward the viewer
        for k in range(6):
            y = (H // 2 + 40 + k * 80 + (i * 6) % 80)
            if y < H:
                cv2.rectangle(frame, (W // 2 - 8, y), (W // 2 + 8, y + 34),
                              (240, 240, 240), -1)
        # two "vehicles"
        x1 = 120 + int(i * 4.5) % (W - 300)
        cv2.rectangle(frame, (x1, 330), (x1 + 150, 420), (30, 60, 200), -1)
        x2 = W - 260 - int(i * 3) % (W - 320)
        cv2.rectangle(frame, (x2, 300), (x2 + 120, 370), (200, 120, 30), -1)
        # potholes: static dark ellipses on the road
        cv2.ellipse(frame, (300, 480), (46, 20), 0, 0, 360, (35, 35, 35), -1)
        cv2.ellipse(frame, (680, 500), (56, 24), 0, 0, 360, (28, 28, 28), -1)
        # drifting smoke plume during the middle of the clip
        if FPS * 2 < i < FPS * 6:
            plume = np.zeros((H, W), np.uint8)
            cx = 700 + int(30 * np.sin(i / 3))
            r = 80 + (i * 2) % 50
            cv2.circle(plume, (cx, 190), r, 255, -1)
            cv2.circle(plume, (cx - 70 + int(20 * np.cos(i / 4)), 140), r - 20, 255, -1)
            plume = cv2.GaussianBlur(plume, (31, 31), 0)
            grey = cv2.merge([plume, plume, plume]) // 2 + 100
            mask = (plume > 40)
            frame[mask] = cv2.addWeighted(frame, 0.35, grey.astype(np.uint8), 0.65, 0)[mask]
        # sensor noise so denoise/motion paths see texture
        frame = cv2.add(frame, cv2.merge([noise, noise, noise]))
        writer.write(frame)

    writer.release()
    print(f"Wrote {out_path} ({SECONDS}s @ {FPS} FPS, {W}x{H})")


if __name__ == "__main__":
    main()
