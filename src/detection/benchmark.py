# src/detection/benchmark.py
# Runs YOLO models on the val or test frames and measures counting accuracy and speed.
# Usage:  python src/detection/benchmark.py val
#         python src/detection/benchmark.py test            (640 px, all 9 models)
#         python src/detection/benchmark.py test 1280       (1280 px, 4 key models)
import time, numpy as np, pandas as pd
from ultralytics import YOLO


# All models, used at the normal image size (640 px)
MODELS = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt",
          "models/yolo26s_cepdof/weights/best.pt",
          "models/yolo11s_cepdof/weights/best.pt",
          "models/yolov8s_cepdof/weights/best.pt"]

# Smaller set used at larger image sizes (slower), to save time
MODELS_HIGHRES = ["yolo26n.pt", "yolo26s.pt", "yolov8s.pt",
                  "models/yolo26s_cepdof/weights/best.pt"]

def run(weights, frames, conf=0.25, imgsz=640):
    """frames: DataFrame with columns image_path, gt_count, dataset."""
    model = YOLO(weights)
    model.predict(frames["image_path"].iloc[0], imgsz=imgsz, verbose=False)   # warm-up, not timed
    rows = []
    for i, (path, gt, ds) in enumerate(zip(frames["image_path"], frames["gt_count"], frames["dataset"])):
        t0 = time.perf_counter()
        r = model.predict(path, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        rows.append({"model": weights, "imgsz": imgsz, "dataset": ds, "image_path": path, "gt": gt,
                     "pred": len(r.boxes), "ms": r.speed["inference"],
                     "ms_total": (time.perf_counter() - t0) * 1000})
        if i % 2000 == 0:
            print(f"  {weights} @ {imgsz}px: {i}/{len(frames)}")
    return pd.DataFrame(rows)

def summarize(df):
    err = df["pred"] - df["gt"]
    return pd.Series({
        "frames": len(df),
        "MAE": err.abs().mean(),                  # average miscount, in people
        "RMSE": np.sqrt((err ** 2).mean()),       # punishes big misses more
        "bias": err.mean(),                       # negative = under-counting (missing people)
        "within_2": (err.abs() <= 2).mean(),      # share of frames off by 2 or fewer
        "ms_per_frame": df["ms"].mean()})         # model inference time only

def safe_name(weights):
    """models/yolo26s_cepdof/weights/best.pt -> yolo26s_cepdof ; yolo26s.pt -> yolo26s"""
    parts = weights.replace("\\", "/").split("/")
    return parts[1] if len(parts) > 2 else parts[-1].replace(".pt", "")

if __name__ == "__main__":
    import glob, os, sys
    split = sys.argv[1] if len(sys.argv) > 1 else "val"          # "val" or "test"
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640       # 640 or 1280
    tag = split if imgsz == 640 else f"{split}_{imgsz}"          # keeps 640 and 1280 results apart
    models = MODELS if imgsz == 640 else MODELS_HIGHRES

    gt = pd.read_csv("data/processed/gt_counts.csv")
    splits = pd.read_csv("data/processed/splits.csv")
    frames = gt.merge(splits, on=["dataset", "sequence"])
    frames = frames[frames["split"] == split].copy()

    # Find each image file on disk, inside its dataset folder.
    # FRIDA has the same file names in every camera folder, so it is matched by
    # "camera_folder/file.jpg"; every other dataset is matched by file name alone.
    paths = {}
    for ds in frames["dataset"].unique():
        folder = "airc_labeled" if ds == "airc" else ds
        root = f"data/raw/{folder}"
        for p in glob.glob(f"{root}/**/*.*", recursive=True):
            if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if ds == "frida":
                key = os.path.relpath(p, root).replace("\\", "/").lower()
            else:
                key = os.path.basename(p)
            paths[(ds, key)] = p

    def lookup(ds, frame):
        key = frame.replace("\\", "/").lower() if ds == "frida" else frame
        return paths.get((ds, key))

    frames["image_path"] = [lookup(d, f) for d, f in zip(frames["dataset"], frames["frame"])]
    missing = frames["image_path"].isna()
    print(f"{len(frames)} {split} frames, {missing.sum()} images not found, image size {imgsz}px")
    print(missing.groupby(frames["dataset"]).sum())
    frames = frames[~missing]

    # Each model at its own best threshold, chosen on validation (Step 14)
    best_t = pd.read_csv("results/best_thresholds.csv", index_col=0)["threshold"]
    if imgsz != 640:   # use thresholds chosen at this image size where available
        hi = pd.read_csv(f"results/best_thresholds_{imgsz}.csv", index_col=0)["threshold"]
        best_t = hi.combine_first(best_t)

    # Save each model's results as soon as it finishes; skip models already done
    parts_dir = f"results/benchmark_{tag}_parts"
    os.makedirs(parts_dir, exist_ok=True)
    parts = []
    for m in models:
        part_file = f"{parts_dir}/{safe_name(m)}.csv"
        if os.path.exists(part_file):
            print("already done, skipping:", m)
        else:
            conf = float(best_t[m]) if m in best_t.index else 0.25
            print(f"running {m} at threshold {conf}, {imgsz}px, on {len(frames)} frames")
            run(m, frames, conf=conf, imgsz=imgsz).to_csv(part_file, index=False)
        parts.append(pd.read_csv(part_file))

    all_results = pd.concat(parts)
    all_results.to_csv(f"results/benchmark_{tag}.csv", index=False)

    summary = all_results.groupby("model").apply(summarize).sort_values("MAE")
    summary.to_csv(f"results/benchmark_{tag}_summary.csv")
    print("\nOverall (dominated by FRIDA; compare models per dataset):")
    print(summary.round(3))

    by_dataset = all_results.groupby(["dataset", "model"]).apply(summarize)
    by_dataset.to_csv(f"results/benchmark_{tag}_by_dataset.csv")
    print("\nBy dataset:")
    print(by_dataset[["frames", "MAE", "bias", "within_2"]].round(3))

    # Average across datasets, so each dataset counts equally (not dominated by FRIDA)
    macro = by_dataset.groupby("model")[["MAE", "within_2"]].mean().sort_values("MAE")
    macro.to_csv(f"results/benchmark_{tag}_macro.csv")
    print("\nAverage across datasets (each dataset weighted equally):")
    print(macro.round(3))