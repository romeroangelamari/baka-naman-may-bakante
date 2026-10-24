"""LightGBM forecasting as reusable functions (same features and settings as lgbm_model.py)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import HORIZONS, SDU


def build_features(g, cap, h, sp, col):
    g = g.set_index("ts").sort_index()
    y = g[col] / cap
    lag = lambda k: y.shift(h + k, freq="15min").reindex(g.index)
    f = pd.DataFrame(index=g.index)
    f["now"] = lag(0)
    f["lag_15"], f["lag_30"], f["lag_45"] = lag(1), lag(2), lag(3)
    f["mean_last_hour"] = pd.concat([lag(k) for k in range(4)], axis=1).mean(axis=1)
    f["change_last_hour"] = lag(0) - lag(3)
    f["same_time_yesterday"] = y.shift(96, freq="15min").reindex(g.index)
    slot = g.index.hour * 4 + g.index.minute // 15
    f["slot"] = slot
    f["weekday"] = -1 if sp in SDU else g.index.dayofweek
    train = (g["split"] == "train").values
    f["typical"] = y[train].groupby(slot[train]).mean().reindex(slot).values
    f["capacity"] = cap
    f["space"] = sp
    f["target"] = (g["count_true"] / cap).values
    f["split"] = g["split"].values
    return f


def fit_predict(occ, cap, col, verbose=False):
    """Train on train days, early-stop on val days, return val+test forecasts for every horizon."""
    preds = []
    for hname, h in HORIZONS.items():
        data = pd.concat([build_features(g, cap[sp], h, sp, col) for sp, g in occ.groupby("space_id")])
        data["space"] = data["space"].astype("category")
        X = [c for c in data.columns if c not in ["target", "split"]]
        tr = data[(data["split"] == "train") & data["target"].notna()]
        va = data[(data["split"] == "val") & data["target"].notna()]
        params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42)
        models = {}
        for name, obj in [("point", dict(objective="l1")), ("upper", dict(objective="quantile", alpha=0.9))]:
            m = lgb.LGBMRegressor(**obj, **params)
            m.fit(tr[X], tr["target"], eval_set=[(va[X], va["target"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            models[name] = m
        out = data[data["split"].isin(["val", "test"])].copy()
        c = out["capacity"].values
        out["yhat"] = np.clip(models["point"].predict(out[X]) * c, 0, c)
        out["yhat_upper"] = np.maximum(np.clip(models["upper"].predict(out[X]) * c, 0, c), out["yhat"])
        out["actual"] = out["target"] * c
        out["horizon"] = hname
        out = out.reset_index().rename(columns={"space": "space_id"})
        out["space_id"] = out["space_id"].astype(str)
        preds.append(out[["space_id", "ts", "horizon", "split", "actual", "yhat", "yhat_upper"]])
        if verbose:
            print(f"  {hname}: trained on {len(tr)} rows")
    return pd.concat(preds, ignore_index=True)