# src/detection/benchmark.py
import time, numpy as np, pandas as pd
from ultralytics import YOLO

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt"]

def run(weights, frames, conf=0.25):
    """frames: DataFrame with columns image_path, gt_count."""
    model = YOLO(weights)
    model.predict(frames.image_path.iloc[0], verbose=False)   # warm-up, not timed
    rows = []
    for path, gt in zip(frames.image_path, frames.gt_count):
        t0 = time.perf_counter()
        r = model.predict(path, classes=[0], conf=conf, imgsz=640, verbose=False)[0]
        rows.append({"model": weights, "image_path": path, "gt": gt,
                     "pred": len(r.boxes), "ms": r.speed["inference"],
                     "ms_total": (time.perf_counter() - t0) * 1000})
    return pd.DataFrame(rows)

def summarize(df):
    err = df["pred"] - df["gt"]
    return pd.Series({
        "MAE": err.abs().mean(),                  # average miscount, in people
        "RMSE": np.sqrt((err ** 2).mean()),       # punishes big misses more
        "within_2": (err.abs() <= 2).mean(),      # share of frames off by 2 or fewer
        "ms_per_frame": df.ms.mean()})

if __name__ == "__main__":
    import glob, os, sys
    split = sys.argv[1] if len(sys.argv) > 1 else "val"     # "val" now, "test" in Step 17

    gt = pd.read_csv("data/processed/gt_counts.csv")
    splits = pd.read_csv("data/processed/splits.csv")
    frames = gt.merge(splits, on=["dataset", "sequence"])
    frames = frames[frames.split == split]

    # Find each image file on disk by its name, inside its dataset folder
    paths = {}
    for ds in frames.dataset.unique():
        folder = "airc_labeled" if ds == "airc" else ds
        for p in glob.glob(f"data/raw/{folder}/**/*.*", recursive=True):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                paths[(ds, os.path.basename(p))] = p
    frames["image_path"] = [paths.get((d, f)) for d, f in zip(frames.dataset, frames.frame)]
    missing = frames.image_path.isna().sum()
    print(f"{len(frames)} {split} frames, {missing} images not found")
    frames = frames.dropna(subset=["image_path"])

    os.makedirs("results", exist_ok=True)
    all_results = pd.concat([run(m, frames) for m in MODELS])
    all_results.to_csv(f"results/zero_shot_{split}.csv", index=False)

    summary = all_results.groupby("model").apply(summarize).sort_values("MAE")
    summary.to_csv(f"results/zero_shot_{split}_summary.csv")
    print(summary)