# src/detection/train.py
import sys
from ultralytics import YOLO

if __name__ == "__main__":           # required on Windows
    weights = sys.argv[1] if len(sys.argv) > 1 else "yolo26s.pt"
    name = weights.replace(".pt", "") + "_cepdof"

    model = YOLO(weights)
    model.train(
        data="data/yolo_cepdof/cepdof.yaml",
        epochs=50,
        imgsz=640,
        batch=8,          # fits a 4 GB RTX 3050; lower to 4 if "CUDA out of memory"
        patience=0,      # stop early if validation stops improving for 20 epochs
        workers=2,        # fewer loader processes: gentler on 8 GB laptop RAM
        project="models",
        fraction=0.5,
        name=name,
    )