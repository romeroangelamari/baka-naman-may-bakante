import sys, logging
import numpy as np
import pandas as pd
from prophet import Prophet
from common import SDU, load_series, metrics

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.WARNING)

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "true"      # "true" or "detector"
col = "count_true" if SOURCE == "true" else "count_detector"
DAY_START, DAY_END = 8, 20                                   # "daytime" = 8 AM to 8 PM

occ, cap = load_series()
rows, preds = [], []
for sp, g in occ.groupby("space_id"):
    g = g.sort_values("ts")
    train = g[(g["split"] == "train") & g[col].notna()]

    m = Prophet(growth="flat", daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
                interval_width=0.8)
    m.add_seasonality(name="daily", period=1, fourier_order=10)     # sharp daily pattern
    if sp not in SDU:                                                 # SDU weekdays are synthetic
        m.add_seasonality(name="weekly", period=7, fourier_order=3)
    m.fit(train[["ts", col]].rename(columns={"ts": "ds", col: "y"}))

    future = g[g["split"].isin(["val", "test"])][["ts", "split", "count_true"]]
    fc = m.predict(future[["ts"]].rename(columns={"ts": "ds"}))
    out = future.reset_index(drop=True).copy()
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        out[c] = fc[c].clip(0, cap[sp]).values
    out["space_id"] = sp
    preds.append(out)

    for split in ["val", "test"]:
        for period in ["all_hours", "8am_8pm"]:
            s = out[out["split"] == split]
            if period == "8am_8pm":
                s = s[s["ts"].dt.hour.between(DAY_START, DAY_END - 1)]
            s = s[s["count_true"].notna()]
            inside = ((s["count_true"] >= s["yhat_lower"]) & (s["count_true"] <= s["yhat_upper"])).mean()
            rows.append({"space_id": sp, "source": SOURCE, "model": "prophet", "split": split,
                         "period": period, **metrics(s["count_true"], s["yhat"]),
                         "coverage_80": inside})
    print(f"{sp}: fitted on {len(train)} intervals")

res = pd.DataFrame(rows).round(3)
res.to_csv(f"results/prophet_metrics_{SOURCE}.csv", index=False)
pd.concat(preds).to_csv(f"results/prophet_preds_{SOURCE}.csv", index=False)

# Compare with the baselines on the same test intervals, daytime only
b = pd.read_csv(f"results/baseline_preds_{SOURCE}.csv", parse_dates=["ts"])
b = b[(b["split"] == "test") & b["ts"].dt.hour.between(DAY_START, DAY_END - 1) & b["actual"].notna()]
b["name"] = b["model"] + " " + b["horizon"]
base = b.groupby(["space_id", "name"]).apply(lambda x: (x["actual"] - x["pred"]).abs().mean(),
                                             include_groups=False).unstack()
keep = ["persistence 15min", "persistence 60min", "previous_day 15min", "train_profile 15min"]
table = base[[c for c in keep if c in base.columns]].rename(columns=lambda c: c.replace(" 15min", ""))
p = res[(res["split"] == "test") & (res["period"] == "8am_8pm")].set_index("space_id")
table["prophet"] = p["MAE"]
table["prophet_coverage_80"] = p["coverage_80"]

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(f"\nTest MAE, 8 AM to 8 PM (people), inputs = {col}, scored against count_true:")
print(table.round(2))
print("\n(S2's test week is empty, so its scores are not meaningful.)")