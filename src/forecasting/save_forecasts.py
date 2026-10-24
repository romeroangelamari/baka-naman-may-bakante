import pandas as pd
import sqlalchemy as sa
from common import get_engine

engine = get_engine()
SHORT = {"true": "true", "detector": "det"}
parts = []
for source, tag in SHORT.items():
    # LightGBM: one model name per horizon, e.g. lgbm_det_60min
    lg = pd.read_csv(f"results/lgbm_preds_{source}.csv", parse_dates=["ts"])
    lg["model"] = "lgbm_" + tag + "_" + lg["horizon"]
    lg["yhat_lower"] = None
    parts.append(lg[["space_id", "ts", "model", "yhat", "yhat_lower", "yhat_upper"]])

    # Prophet: same forecast at every horizon, e.g. prophet_det
    pr = pd.read_csv(f"results/prophet_preds_{source}.csv", parse_dates=["ts"])
    pr["model"] = "prophet_" + tag
    parts.append(pr[["space_id", "ts", "model", "yhat", "yhat_lower", "yhat_upper"]])

fc = pd.concat(parts, ignore_index=True)
fc = fc.drop_duplicates(["space_id", "ts", "model"])
fc[["yhat", "yhat_lower", "yhat_upper"]] = fc[["yhat", "yhat_lower", "yhat_upper"]].astype(float).round(3)

with engine.begin() as con:
    con.execute(sa.text("DELETE FROM forecasts"))
fc.to_sql("forecasts", engine, if_exists="append", index=False, chunksize=2000)

print(fc.groupby("model").agg(rows=("ts", "size"), spaces=("space_id", "nunique"),
                              first=("ts", "min"), last=("ts", "max")).to_string())
print(f"\nSaved {len(fc)} forecast rows to the forecasts table")