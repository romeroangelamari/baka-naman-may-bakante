import os
import numpy as np
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
HORIZONS = {"15min": 1, "30min": 2, "60min": 4}   # steps of 15 minutes
SDU = ["S4", "S5", "S6"]                          # synthetic dates: no weekly pattern


def get_engine():
    return sa.create_engine(os.getenv("DB_URL"))


def load_series(engine=None):
    """Returns occ (one row per space and 15-min interval, with split column) and capacities."""
    engine = engine or get_engine()
    occ = pd.read_sql("SELECT space_id, ts, count_true, count_detector FROM occupancy", engine)
    occ["ts"] = pd.to_datetime(occ["ts"])
    occ["date"] = occ["ts"].dt.date
    splits = pd.read_csv("data/processed/time_splits.csv", parse_dates=["date"])
    splits["date"] = splits["date"].dt.date
    occ = occ.merge(splits, on=["space_id", "date"], how="left")
    cap = pd.read_sql("SELECT space_id, capacity FROM spaces", engine).set_index("space_id")["capacity"]
    return occ.sort_values(["space_id", "ts"]).reset_index(drop=True), cap


def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    m = ~np.isnan(a) & ~np.isnan(p)
    a, p = a[m], p[m]
    if len(a) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "WAPE": np.nan}
    return {"n": len(a), "MAE": np.abs(a - p).mean(), "RMSE": np.sqrt(((a - p) ** 2).mean()),
            "WAPE": np.abs(a - p).sum() / max(np.abs(a).sum(), 1e-9) * 100}