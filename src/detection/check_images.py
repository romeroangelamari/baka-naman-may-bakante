# Compares the image names the labels expect with the image files actually on disk.
import glob, os
import pandas as pd

gt = pd.read_csv("data/processed/gt_counts.csv")
for ds in ["airc", "frida", "library_seats", "rhc", "scut_head"]:
    folder = "airc_labeled" if ds == "airc" else ds
    files = [p for p in glob.glob(f"data/raw/{folder}/**/*.*", recursive=True)
             if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    print(f"\n=== {ds}  (looking in data/raw/{folder}) ===")
    print("image files found:", len(files))
    print("on disk:  ", [os.path.relpath(p, f"data/raw/{folder}") for p in files[:3]])
    print("expected: ", gt[gt["dataset"] == ds]["frame"].head(3).tolist())