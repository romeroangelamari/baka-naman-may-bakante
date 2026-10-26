"""Tests that the forecasting code only uses information available at forecast time."""
import numpy as np
import pandas as pd
from common import metrics


def test_metrics_values():
    m = metrics([0, 2, 4], [1, 2, 2])
    assert m["n"] == 3
    assert np.isclose(m["MAE"], 1.0)
    assert np.isclose(m["RMSE"], np.sqrt(5 / 3))
    assert np.isclose(m["WAPE"], 3 / 6 * 100)


def test_metrics_ignore_missing_values():
    m = metrics([1, np.nan, 3], [1, 5, np.nan])
    assert m["n"] == 1 and m["MAE"] == 0


def test_lightgbm_features_never_use_the_future():
    from lgbm_core import build_features
    ts = pd.date_range("2021-12-01", periods=4 * 96, freq="15min")
    g = pd.DataFrame({"ts": ts, "count_true": np.arange(len(ts), dtype=float), "split": "train"})
    g["obs"] = g["count_true"]
    for h in [1, 2, 4]:
        f = build_features(g, cap=1.0, h=h, sp="S1", col="obs")
        t = f.index[200]
        assert f.loc[t, "now"] == f.loc[t, "target"] - h          # latest value is h steps old
        assert f.loc[t, "lag_45"] == f.loc[t, "target"] - h - 3