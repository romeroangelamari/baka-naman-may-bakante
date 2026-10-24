# src/detection/sweep.py
# Finds the best confidence threshold per model on the validation frames.
import numpy as np, pandas as pd
from ultralytics import YOLO

MODELS = [ "yolov8s.pt",  "yolo26n.pt", "yolo26s.pt",]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Reuse the image paths found by the benchmark run
frames = pd.read_csv("results/benchmark_val.csv")[["image_path", "gt"]].drop_duplicates()

rows = []
for weights in MODELS:
    model = YOLO(weights)
    for path, gt in zip(frames["image_path"], frames["gt"]):
        r = model.predict(path, classes=[0], conf=0.10, imgsz=1280, verbose=False)[0]
        confs = r.boxes.conf.cpu().numpy()          # confidence of every person found
        for t in THRESHOLDS:
            rows.append({"model": weights, "threshold": t, "gt": gt,
                         "pred": int((confs >= t).sum())})
    print("done", weights)

df = pd.DataFrame(rows)
df["abs_err"] = (df["pred"] - df["gt"]).abs()
table = df.groupby(["model", "threshold"]).abs_err.mean().unstack()   # MAE per model x threshold
table.to_csv("results/threshold_sweep_1280.csv")
print(table.round(2))

best = table.idxmin(axis=1)
print("\nBest threshold per model:")
print(pd.DataFrame({"threshold": best, "MAE": table.min(axis=1)}).sort_values("MAE"))
best.to_csv("results/best_thresholds_1280.csv", header=["threshold"])