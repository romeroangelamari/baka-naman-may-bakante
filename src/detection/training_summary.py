# Finds the best epoch of every training run and saves one summary table.
import glob, os
import pandas as pd

RUNS = "models"        # or "/content/drive/MyDrive/bakante_models" in Colab

rows = []
for csv in glob.glob(f"{RUNS}/*/results.csv"):
    r = pd.read_csv(csv)
    r.columns = r.columns.str.strip()                      # some versions add spaces
    score = 0.1 * r["metrics/mAP50(B)"] + 0.9 * r["metrics/mAP50-95(B)"]
    best = r.loc[score.idxmax()]
    rows.append({
        "run": os.path.basename(os.path.dirname(csv)),
        "best_epoch": int(best["epoch"]),
        "epochs_run": int(r["epoch"].max()),
        "precision": round(best["metrics/precision(B)"], 3),
        "recall": round(best["metrics/recall(B)"], 3),
        "mAP50": round(best["metrics/mAP50(B)"], 3),
        "mAP50_95": round(best["metrics/mAP50-95(B)"], 3),
        "train_hours": round(r["time"].max() / 3600, 2) if "time" in r else None,
    })

summary = pd.DataFrame(rows).sort_values("mAP50", ascending=False)
os.makedirs("results", exist_ok=True)
summary.to_csv("results/training_summary.csv", index=False)
print(summary.to_string(index=False))