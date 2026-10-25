"""Step 26: run the selected detector (YOLO26n, 1280 px) on one real recorded sequence and save
counts plus a few annotated frames, so the dashboard can show the camera-to-count step."""
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

MODEL, IMGSZ, CONF = "yolo26n.pt", 1280, 0.25    # selected detector and its validation-chosen threshold
SECONDS_PER_FRAME = 10                          # synthetic spacing between frames for the replay
SAVE_EVERY = 10                                 # keep every 10th annotated frame for the dashboard
OUT = "results/demo"

# CEPDOF validation sequence (lunch3): the frames and their true counts are already listed here
val = pd.read_csv("results/benchmark_val.csv")
frames = (val.drop_duplicates("image_path")[["image_path", "gt"]]
          .sort_values("image_path").reset_index(drop=True))
print(f"{len(frames)} frames from the CEPDOF validation sequence")

os.makedirs(f"{OUT}/frames", exist_ok=True)
model = YOLO(MODEL)
start = pd.Timestamp("2000-01-01 09:00")
rows = []
for i, r in frames.iterrows():
    res = model.predict(r["image_path"], imgsz=IMGSZ, conf=CONF, classes=[0], verbose=False)[0]
    pred = len(res.boxes)
    saved = ""
    if i % SAVE_EVERY == 0:
        saved = f"{OUT}/frames/frame_{i:04d}.jpg"
        img = res.plot(labels=False, conf=False)                       # boxes drawn on the frame
        h, w = img.shape[:2]
        img = cv2.resize(img, (800, int(800 * h / w)))                 # smaller file for the dashboard
        cv2.imwrite(saved, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    rows.append({"frame": i, "ts": start + pd.Timedelta(seconds=i * SECONDS_PER_FRAME),
                 "gt": r["gt"], "pred": pred, "saved_frame": saved})
    if i % 100 == 0:
        print(f"  frame {i}: true {r['gt']}, detected {pred}")

d = pd.DataFrame(rows)
d.to_csv(f"{OUT}/demo_counts.csv", index=False)
per_min = d.set_index("ts")[["gt", "pred"]].resample("1min").median()
per_min.to_csv(f"{OUT}/demo_per_minute.csv")
err = d["pred"] - d["gt"]
print(f"\nPer frame:  MAE {err.abs().mean():.2f}, bias {err.mean():+.2f}")
e2 = per_min["pred"] - per_min["gt"]
print(f"Per minute (median): MAE {e2.abs().mean():.2f}, bias {e2.mean():+.2f}")
print(f"Saved {OUT}/demo_counts.csv, demo_per_minute.csv and {d['saved_frame'].astype(bool).sum()} frames")