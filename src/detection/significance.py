# Paired Wilcoxon test: per-frame absolute errors of two models on the same test frames
import pandas as pd
from scipy.stats import wilcoxon

r = pd.read_csv("results/benchmark_test_1280.csv")
r["abs_err"] = (r["pred"] - r["gt"]).abs()

def compare(a, b):
    pa = r[r["model"] == a].set_index("image_path")["abs_err"]
    pb = r[r["model"] == b].set_index("image_path")["abs_err"]
    both = pa.index.intersection(pb.index)                  # same frames, same order
    stat, p = wilcoxon(pa[both], pb[both])
    print(f"{a} vs {b}: MAE {pa[both].mean():.3f} vs {pb[both].mean():.3f}, p = {p:.4g}")

compare("yolo26n.pt", "yolo26s.pt")
compare("yolo26n.pt", "yolov8s.pt")
compare("yolo26n.pt", "models/yolo26s_cepdof/weights/best.pt")