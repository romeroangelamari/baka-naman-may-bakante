import sys
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from common import HORIZONS

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "true"
DAY_START, DAY_END = 8, 20


def dm_test(e_a, e_b, h):
    """Diebold-Mariano test on absolute errors, with the Harvey-Leybourne-Newbold correction.
    Negative statistic = model A has smaller errors than model B."""
    d = np.abs(e_a) - np.abs(e_b)
    n = len(d)
    d_c = d - d.mean()
    gamma = [np.sum(d_c[k:] * d_c[:n - k]) / n for k in range(h)]     # autocovariances up to lag h-1
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)              # small-sample correction
    p = 2 * t_dist.sf(abs(stat), df=n - 1)
    return stat, p


lg = pd.read_csv(f"results/lgbm_preds_{SOURCE}.csv", parse_dates=["ts"])
bl = pd.read_csv(f"results/baseline_preds_{SOURCE}.csv", parse_dates=["ts"])
pr = pd.read_csv(f"results/prophet_preds_{SOURCE}.csv", parse_dates=["ts"])

rows = []
for hname, h in HORIZONS.items():
    a = lg[(lg["horizon"] == hname) & (lg["split"] == "test")][["space_id", "ts", "actual", "yhat"]]
    a = a.rename(columns={"yhat": "lightgbm"})
    for model in ["persistence", "train_profile"]:
        b = bl[(bl["horizon"] == hname) & (bl["model"] == model)][["space_id", "ts", "pred"]]
        a = a.merge(b.rename(columns={"pred": model}), on=["space_id", "ts"], how="left")
    a = a.merge(pr[["space_id", "ts", "yhat"]].rename(columns={"yhat": "prophet"}), on=["space_id", "ts"], how="left")
    a = a[a["ts"].dt.hour.between(DAY_START, DAY_END - 1)]

    for sp, g in a.groupby("space_id"):
        for other in ["persistence", "prophet", "train_profile"]:
            q = g.dropna(subset=["actual", "lightgbm", other]).sort_values("ts")
            e_a, e_b = (q["lightgbm"] - q["actual"]).values, (q[other] - q["actual"]).values
            stat, p = dm_test(e_a, e_b, h)
            mae_a, mae_b = np.abs(e_a).mean(), np.abs(e_b).mean()
            verdict = ("LightGBM better" if stat < 0 else "LightGBM worse") if p < 0.05 else "no significant difference"
            rows.append({"source": SOURCE, "space_id": sp, "horizon": hname, "vs": other, "n": len(q),
                         "MAE_lightgbm": round(mae_a, 2), "MAE_other": round(mae_b, 2),
                         "DM_stat": round(stat, 2), "p_value": round(p, 4), "result": verdict})

res = pd.DataFrame(rows)
res.to_csv(f"results/dm_tests_{SOURCE}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(f"Diebold-Mariano tests, test days 8 AM to 8 PM, inputs = {SOURCE} (S2 excluded: empty test week)")
print(res[res["space_id"] != "S2"].drop(columns=["source"]).to_string(index=False))