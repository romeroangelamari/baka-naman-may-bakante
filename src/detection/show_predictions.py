# Saves 2 images per test dataset with the model's detections drawn on them
import os, pandas as pd
from ultralytics import YOLO

MODEL, CONF = "yolo26s.pt", 0.35
r = pd.read_csv("results/benchmark_test.csv")
r = r[r["model"] == MODEL]
os.makedirs("results/pred_checks", exist_ok=True)
model = YOLO(MODEL)
for ds, g in r.groupby("dataset"):
    for _, row in g.sample(2, random_state=0).iterrows():
        res = model.predict(row["image_path"], classes=[0], conf=CONF, imgsz=1280, verbose=False)[0]
        out = f"results/pred_checks/{ds}_{os.path.basename(row['image_path'])}"
        res.save(out)
        print(ds, "true:", row["gt"], "found:", len(res.boxes), "->", out)