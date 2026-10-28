import time
import numpy as np
import pandas as pd
from ultralytics import YOLO

frames = pd.read_csv("results/benchmark_val.csv").drop_duplicates("image_path")["image_path"].head(50).tolist()
model = YOLO("yolo26n.pt")
for imgsz in [640, 1280]:
    model.predict(frames[0], imgsz=imgsz, device="cpu", verbose=False)        # warm-up
    t = []
    for f in frames:
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, conf=0.25, classes=[0], device="cpu", verbose=False)
        t.append((time.perf_counter() - t0) * 1000)
    ms = np.median(t)
    print(f"YOLO26n on CPU at {imgsz} px: {ms:.0f} ms per frame "
          f"-> about {60_000 / ms:.0f} rooms per CPU process at 1 frame per room per minute")