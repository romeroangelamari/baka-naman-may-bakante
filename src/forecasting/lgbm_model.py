import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import HORIZONS, SDU, load_series, metrics

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "true"      # "true" or "detector"
col = "count_true" if SOURCE == "true" else "count_detector"
DAY_START, DAY_END = 8, 20


def build_features(g, cap, h, sp):
    """One row per target interval t. Every input comes from time t-h or earlier,
    so the model only uses what would be known when the forecast is made."""
    g = g.set_index("ts").sort_index()
    y = g[col] / cap                                    # occupancy as a share of capacity
    lag = lambda k: y.shift(h + k, freq="15min").reindex(g.index)
    f = pd.DataFrame(index=g.index)
    f["now"] = lag(0)                                   # latest count when forecasting
    f["lag_15"], f["lag_30"], f["lag_45"] = lag(1), lag(2), lag(3)
    f["mean_last_hour"] = pd.concat([lag(k) for k in range(4)], axis=1).mean(axis=1)
    f["change_last_hour"] = lag(0) - lag(3)
    f["same_time_yesterday"] = y.shift(96, freq="15min").reindex(g.index)
    slot = g.index.hour * 4 + g.index.minute // 15
    f["slot"] = slot                                    # time of day of the target
    f["weekday"] = -1 if sp in SDU else g.index.dayofweek
    train = (g["split"] == "train").values
    profile = y[train].groupby(slot[train]).mean()      # typical training day at this time
    f["typical"] = profile.reindex(slot).values
    f["capacity"] = cap
    f["space"] = sp
    f["target"] = (g["count_true"] / cap).values        # always learn the real occupancy
    f["split"] = g["split"].values
    return f


occ, cap = load_series()
rows, preds = [], []
for hname, h in HORIZONS.items():
    data = pd.concat([build_features(g, cap[sp], h, sp) for sp, g in occ.groupby("space_id")])
    data["space"] = data["space"].astype("category")
    X_cols = [c for c in data.columns if c not in ["target", "split"]]
    tr = data[(data["split"] == "train") & data["target"].notna()]
    va = data[(data["split"] == "val") & data["target"].notna()]

    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42)
    point = lgb.LGBMRegressor(objective="l1", **params)            # median forecast (yhat)
    point.fit(tr[X_cols], tr["target"], eval_set=[(va[X_cols], va["target"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    upper = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **params)   # cautious forecast (yhat_upper)
    upper.fit(tr[X_cols], tr["target"], eval_set=[(va[X_cols], va["target"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"{hname}: trained on {len(tr)} rows, best rounds {point.best_iteration_} / {upper.best_iteration_}")

    out = data[data["split"].isin(["val", "test"])].copy()
    c = out["capacity"].values
    out["yhat"] = np.clip(point.predict(out[X_cols]) * c, 0, c)
    out["yhat_upper"] = np.maximum(np.clip(upper.predict(out[X_cols]) * c, 0, c), out["yhat"])
    out["actual"] = out["target"] * c
    out["horizon"] = hname
    out = out.reset_index()
    preds.append(out[["space", "ts", "horizon", "split", "actual", "yhat", "yhat_upper"]]
                 .rename(columns={"space": "space_id"}))

    for (sp, split), s in out.groupby(["space", "split"], observed=True):
        for period in ["all_hours", "8am_8pm"]:
            q = s if period == "all_hours" else s[s["ts"].dt.hour.between(DAY_START, DAY_END - 1)]
            q = q[q["actual"].notna()]
            rows.append({"space_id": sp, "source": SOURCE, "model": "lightgbm", "horizon": hname,
                         "split": split, "period": period, **metrics(q["actual"], q["yhat"]),
                         "below_upper": (q["actual"] <= q["yhat_upper"]).mean()})

res = pd.DataFrame(rows).round(3)
res.to_csv(f"results/lgbm_metrics_{SOURCE}.csv", index=False)
pd.concat(preds).to_csv(f"results/lgbm_preds_{SOURCE}.csv", index=False)

# Comparison table: test period, 8 AM to 8 PM
b = pd.read_csv(f"results/baseline_preds_{SOURCE}.csv", parse_dates=["ts"])
b = b[(b["split"] == "test") & b["ts"].dt.hour.between(DAY_START, DAY_END - 1) & b["actual"].notna()]
pers = b[b["model"] == "persistence"].groupby(["space_id", "horizon"]).apply(
    lambda x: (x["actual"] - x["pred"]).abs().mean(), include_groups=False).unstack()
prop = pd.read_csv(f"results/prophet_metrics_{SOURCE}.csv")
prop = prop[(prop["split"] == "test") & (prop["period"] == "8am_8pm")].set_index("space_id")["MAE"]
lg = res[(res["split"] == "test") & (res["period"] == "8am_8pm")]
table = pd.DataFrame({"prophet": prop})
for hname in HORIZONS:
    table[f"persistence {hname}"] = pers[hname]
    table[f"lightgbm {hname}"] = lg[lg["horizon"] == hname].set_index("space_id")["MAE"]
cov = lg.groupby("space_id")["below_upper"].mean()
table["share_below_upper (target 0.90)"] = cov

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(f"\nTest MAE, 8 AM to 8 PM (people), inputs = {col}, scored against count_true:")
print(table.round(2))
print("\n(S2's test week is empty, so its scores are not meaningful.)")