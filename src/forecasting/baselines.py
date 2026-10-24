import sys
import numpy as np
import pandas as pd
from common import HORIZONS, SDU, load_series, metrics

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "true"      # "true" or "detector"
col = "count_true" if SOURCE == "true" else "count_detector"

occ, cap = load_series()
preds, rows = [], []
for sp, g in occ.groupby("space_id"):
    g = g.set_index("ts").sort_index()
    y_obs = g[col]                     # what the system sees (true or detector-like counts)
    y_true = g["count_true"]           # what we score against: real occupancy
    slot = g.index.hour * 4 + g.index.minute // 15
    train = g["split"] == "train"

    # same time on the previous day that has data (skips weekends and gaps)
    by_day = y_obs.groupby([g.index.date, slot]).first().unstack()
    prev_day = by_day.shift(1).stack(future_stack=True)
    prev_day_val = pd.Series(prev_day.reindex(list(zip(g.index.date, slot))).values, index=g.index)

    # average of the training days at each time of day (and weekday for ROBOD)
    keys = [slot] if sp in SDU else [slot, g.index.dayofweek]
    profile = y_obs[train].groupby([k[train] for k in keys]).mean()
    idx = list(zip(*[k for k in keys])) if len(keys) > 1 else list(slot)
    profile_val = pd.Series(profile.reindex(idx).values, index=g.index)

    for hname, h in HORIZONS.items():
        cands = {
            "persistence": y_obs.shift(h, freq="15min").reindex(g.index),
            "previous_day": prev_day_val,
            "train_profile": profile_val,
        }
        if sp not in SDU:
            cands["previous_week"] = y_obs.shift(7 * 96, freq="15min").reindex(g.index)
        for model, p in cands.items():
            p = p.clip(0, cap[sp])
            for split in ["val", "test"]:
                m = (g["split"] == split).values
                rows.append({"space_id": sp, "source": SOURCE, "horizon": hname, "model": model,
                             "split": split, **metrics(y_true[m], p[m])})
            preds.append(pd.DataFrame({"space_id": sp, "ts": g.index, "horizon": hname, "model": model,
                                       "split": g["split"].values, "actual": y_true.values, "pred": p.values}))

res = pd.DataFrame(rows).round(3)
res.to_csv(f"results/baseline_metrics_{SOURCE}.csv", index=False)
pd.concat(preds).to_csv(f"results/baseline_preds_{SOURCE}.csv", index=False)

pd.set_option("display.width", 200)
test = res[res.split == "test"]
print(f"Test MAE (people), inputs = {col}, scored against count_true:")
print(test.pivot_table(index=["space_id", "model"], columns="horizon", values="MAE")[list(HORIZONS)].round(2))