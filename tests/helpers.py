"""Small made-up data so the tests run without the database."""
import numpy as np
import pandas as pd

DAY = pd.Timestamp("2021-12-20")
MODELS = ["lgbm_det_15min", "lgbm_det_30min", "lgbm_det_60min", "prophet_det"]


def spaces():
    return pd.DataFrame({
        "space_id": ["A", "B", "C", "D"],
        "name": ["Quiet room", "Group room", "Mixed room", "Big hall"],
        "capacity": [20, 30, 24, 80],
        "use_type": ["Individual", "Group", "Individual or group", "Group"],
        "quiet": [True, False, False, False],
        "outlets": [True, False, True, True],
        "x": [1, 2, 3, 4], "y": [1, 1, 1, 1]})


def forecasts(levels=None, upper_extra=1.0, rooms=None):
    """Flat forecasts for one day: levels = expected people per room."""
    levels = levels or {"A": 4, "B": 15, "C": 6, "D": 20}
    rows = []
    ts = pd.date_range(DAY, periods=96, freq="15min")
    for sp, y in levels.items():
        if rooms and sp not in rooms:
            continue
        for m in MODELS:
            rows.append(pd.DataFrame({"space_id": sp, "ts": ts, "model": m, "yhat": float(y),
                                      "yhat_lower": np.nan, "yhat_upper": float(y) + upper_extra}))
    return pd.concat(rows, ignore_index=True)


def at(h, m=0):
    return DAY + pd.Timedelta(hours=h, minutes=m)