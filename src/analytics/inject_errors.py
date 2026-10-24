import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
engine = sa.create_engine(os.getenv("DB_URL"))

MODEL = "yolo26n.pt"                    # selected detector (Phase 2)
EXCLUDE = ["scut_head", "library_seats"]   # head-only labels / person class not yet verified
MAX_GT = 90                             # our largest room holds 84
SEED = 42

# 1. Real YOLO26n counting errors on the test frames (1280 px, validation-chosen threshold)
bench = pd.read_csv("results/benchmark_test_1280.csv")
print("Models in file:", sorted(bench["model"].unique()))
err = bench[bench["model"] == MODEL].copy()
if err.empty:
    raise SystemExit(f"No rows for model '{MODEL}'. Set MODEL to the exact name printed above.")
err = err[(err["gt"] <= MAX_GT) & (~err["dataset"].isin(EXCLUDE))]
err["e"] = err["pred"] - err["gt"]
err["ratio"] = np.where(err["gt"] > 0, err["pred"] / err["gt"].clip(lower=1), np.nan)
err["level"] = pd.cut(err["gt"], bins=[-1, 5, 15, 1000], labels=["0-5", "6-15", "16+"])
err[["dataset", "image_path", "gt", "pred", "e", "ratio", "level"]].to_csv("results/detector_errors.csv", index=False)

# 0-5 people: add an absolute error; more people: multiply by a "share of people detected" ratio
abs_pool = err.loc[err["level"] == "0-5", "e"].values
ratio_pools = {lvl: err.loc[err["level"] == lvl, "ratio"].dropna().values for lvl in ["6-15", "16+"]}

pool_summary = err.groupby("level", observed=True).agg(frames=("e", "size"), mean_gt=("gt", "mean"),
                                                       mean_error=("e", "mean"), mae=("e", lambda x: x.abs().mean()),
                                                       median_ratio=("ratio", "median")).round(2)
print(f"\nError pools by crowd level (excluded: {EXCLUDE}):")
print(pool_summary)
print("\nFrames per dataset in the pools:")
print(err["dataset"].value_counts())
pool_summary.to_csv("results/error_pools.csv")

# 2. Inject sampled errors into every true count
rng = np.random.default_rng(SEED)
def inject(s, capacity):
    out = np.full(len(s), np.nan)
    for i, x in enumerate(s.values):
        if pd.isna(x):
            continue
        if x <= 5:
            out[i] = x + rng.choice(abs_pool)
        elif x <= 15:
            out[i] = x * rng.choice(ratio_pools["6-15"])
        else:
            out[i] = x * rng.choice(ratio_pools["16+"])
    return pd.Series(np.clip(np.round(out), 0, capacity), index=s.index)

occ = pd.read_sql("""SELECT o.space_id, o.ts, o.count_true, s.capacity
                     FROM occupancy o JOIN spaces s ON s.space_id = o.space_id""", engine)
occ["ts"] = pd.to_datetime(occ["ts"])
parts, report = [], []
for sp, g in occ.groupby("space_id"):
    g = g.sort_values("ts").copy()
    g["count_detector"] = inject(g["count_true"], g["capacity"].iloc[0]).values
    m = g["count_true"].notna()
    d = g.loc[m, "count_detector"] - g.loc[m, "count_true"]
    report.append({"space_id": sp, "intervals": int(m.sum()), "MAE": round(d.abs().mean(), 2),
                   "bias": round(d.mean(), 2), "within_2": round((d.abs() <= 2).mean(), 3)})
    parts.append(g[["space_id", "ts", "count_detector"]])
det = pd.concat(parts)

# 3. Write count_detector back to MySQL
det.to_sql("tmp_detector", engine, if_exists="replace", index=False, chunksize=2000)
with engine.begin() as con:
    con.execute(sa.text("""UPDATE occupancy o JOIN tmp_detector t
                           ON o.space_id = t.space_id AND o.ts = t.ts
                           SET o.count_detector = t.count_detector"""))
    con.execute(sa.text("DROP TABLE tmp_detector"))

rep = pd.DataFrame(report)
rep.to_csv("results/injection_report.csv", index=False)
print("\nInjected error per space (count_detector - count_true):")
print(rep.to_string(index=False))

# 4. Figure: one day of true vs detector-like counts for each space
os.makedirs("results/figures", exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
plot_df = occ.merge(det, on=["space_id", "ts"])
for ax, (sp, g) in zip(axes.flat, plot_df.groupby("space_id")):
    g = g.set_index("ts")
    busiest = g["count_true"].groupby(g.index.date).sum().idxmax()
    day = g[g.index.date == busiest]
    ax.plot(day.index, day["count_true"], label="true")
    ax.plot(day.index, day["count_detector"], label="detector-like", alpha=0.8)
    ax.set_title(f"{sp} on {busiest}"); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
axes.flat[0].legend()
fig.tight_layout(); fig.savefig("results/figures/injected_example.png", dpi=200)
print("\nSaved results/detector_errors.csv, error_pools.csv, injection_report.csv, figures/injected_example.png")