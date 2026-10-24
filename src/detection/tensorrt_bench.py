# src/detection/tensorrt_bench.py
# Exports models to TensorRT FP16 and compares inference speed with PyTorch.
import pandas as pd
from ultralytics import YOLO

if __name__ == "__main__":                     # required on Windows
    IMGSZ = 1280
    imgs = (pd.read_csv("results/benchmark_test_1280.csv")["image_path"]
              .drop_duplicates().sample(300, random_state=0))

    rows = []
    for w in ["yolo26n.pt", "yolo26s.pt", "yolov8s.pt"]:
        engine = YOLO(w).export(format="engine", half=True, imgsz=IMGSZ, device=0)
        for runtime, model in [("pytorch", YOLO(w)), ("tensorrt", YOLO(engine))]:
            model.predict(imgs.iloc[0], imgsz=IMGSZ, verbose=False)        # warm-up
            ms = [model.predict(p, imgsz=IMGSZ, classes=[0], verbose=False)[0].speed["inference"]
                  for p in imgs]
            rows.append({"model": w, "runtime": runtime, "imgsz": IMGSZ,
                         "ms_per_frame": sum(ms) / len(ms)})
            print(rows[-1])

    df = pd.DataFrame(rows)
    df["fps"] = 1000 / df["ms_per_frame"]
    df.to_csv("results/tensorrt_speed.csv", index=False)
    print(df.round(2))