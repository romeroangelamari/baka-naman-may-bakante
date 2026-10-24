import glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

N_RUNS = 20      # repeat the injection with 20 seeds so the result does not depend on one draw

# 1. Rebuild the same error pools used in inject_errors.py
err = pd.read_csv("results/detector_errors.csv")
abs_pool = err.loc[err["level"] == "0-5", "e"].values
ratio_pools = {lvl: err.loc[err["level"] == lvl, "ratio"].dropna().values for lvl in ["6-15", "16+"]}

def inject(gt, rng):
    gt = np.asarray(gt, dtype=float)
    out = np.where(gt <= 5, gt + rng.choice(abs_pool, len(gt)),
          np.where(gt <= 15, gt * rng.choice(ratio_pools["6-15"], len(gt)),
                             gt * rng.choice(ratio_pools["16+"], len(gt))))
    return np.clip(np.round(out), 0, None)

# 2. Load CO2FISHEYE: ground truth and the real fisheye camera count
days = []
for f in sorted(glob.glob("data/raw/co2fisheye/GT+Fisheye day *.csv")):
    d = pd.read_csv(f, encoding="utf-8-sig")
    d.columns = ["time", "gt", "camera"]
    day = int(re.search(r"day (\d+)", f).group(1))
    d["ts"] = pd.to_datetime("2000-01-01 " + d["time"].astype(str).str.strip(), format="%Y-%m-%d %H:%M:%S")
    d["day"] = day
    days.append(d.dropna(subset=["gt", "camera"]))
raw = pd.concat(days)
print(f"CO2FISHEYE: {len(raw)} readings over {raw['day'].nunique()} days, "
      f"true count 0-{raw['gt'].max():.0f}, mean {raw['gt'].mean():.2f}")

# 15-minute medians, the same resolution as our occupancy table
q = (raw.set_index("ts").groupby("day")[["gt", "camera"]].resample("15min").median()
        .dropna().reset_index())

def stats(e):
    e = np.asarray(e)
    return {"MAE": np.abs(e).mean(), "bias": e.mean(), "within_2": (np.abs(e) <= 2).mean(),
            "p10": np.percentile(e, 10), "p90": np.percentile(e, 90)}

rows, sim_example = [], None
for name, df in [("raw readings", raw), ("15-min medians", q)]:
    real_e = (df["camera"] - df["gt"]).values
    r = {"resolution": name, "source": "real fisheye camera", "n": len(real_e), **stats(real_e)}
    rows.append(r)
    sims, ks_d, ks_p = [], [], []
    for seed in range(N_RUNS):
        sim_e = inject(df["gt"], np.random.default_rng(seed)) - df["gt"].values
        sims.append(stats(sim_e))
        res = ks_2samp(real_e, sim_e)
        ks_d.append(res.statistic); ks_p.append(res.pvalue)
        if name == "15-min medians" and seed == 0:
            sim_example = sim_e
    avg = pd.DataFrame(sims).mean().to_dict()
    rows.append({"resolution": name, "source": f"injected (mean of {N_RUNS} runs)", "n": len(real_e), **avg,
                 "KS_D": np.mean(ks_d), "KS_p_median": np.median(ks_p)})

out = pd.DataFrame(rows).round(3)
os.makedirs("results/figures", exist_ok=True)
out.to_csv("results/injection_validation.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("\nReal camera error vs injected error (count - true count):")
print(out.to_string(index=False))

# Error by crowd level at 15-min resolution
q["level"] = pd.cut(q["gt"], bins=[-1, 5, 15, 1000], labels=["0-5", "6-15", "16+"])
q["real_e"] = q["camera"] - q["gt"]
q["sim_e"] = sim_example
by_level = q.groupby("level", observed=True).agg(intervals=("gt", "size"),
            real_MAE=("real_e", lambda x: x.abs().mean()), injected_MAE=("sim_e", lambda x: x.abs().mean()),
            real_bias=("real_e", "mean"), injected_bias=("sim_e", "mean")).round(2)
print("\n15-min errors by crowd level (one injection run):")
print(by_level)
by_level.to_csv("results/injection_validation_by_level.csv")

# Histogram of both error distributions
lo, hi = int(min(q.real_e.min(), q.sim_e.min())), int(max(q.real_e.max(), q.sim_e.max()))
bins = np.arange(lo - 0.5, hi + 1.5, 1)
plt.figure(figsize=(7, 4))
plt.hist(q["real_e"], bins=bins, alpha=0.6, density=True, label="real fisheye camera")
plt.hist(q["sim_e"], bins=bins, alpha=0.6, density=True, label="injected YOLO26n errors")
plt.xlabel("Count error (count - true count), 15-min medians"); plt.ylabel("Share of intervals")
plt.title("CO2FISHEYE: real vs injected counting errors"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/injection_validation_hist.png", dpi=200)
print("\nSaved results/injection_validation.csv, injection_validation_by_level.csv, figures/injection_validation_hist.png")