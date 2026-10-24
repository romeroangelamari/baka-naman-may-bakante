import numpy as np
import pandas as pd

# Which forecast to use, by how far ahead the interval is (minutes after the as-of time)
LEAD_MODELS = [(15, "lgbm_{src}_15min"), (30, "lgbm_{src}_30min"), (60, "lgbm_{src}_60min")]
FAR_MODEL = "prophet_{src}"


def pick_model(lead_minutes, src="det"):
    for limit, name in LEAD_MODELS:
        if lead_minutes <= limit:
            return name.format(src=src)
    return FAR_MODEL.format(src=src)


def empty_reservations():
    return pd.DataFrame({"space_id": pd.Series(dtype=str), "start_ts": pd.Series(dtype="datetime64[ns]"),
                         "end_ts": pd.Series(dtype="datetime64[ns]")})


def reserve(reservations, space_id, start, minutes=60):
    new = pd.DataFrame([{"space_id": space_id, "start_ts": start, "end_ts": start + pd.Timedelta(minutes=minutes)}])
    return pd.concat([reservations, new], ignore_index=True)


def recommend(spaces, forecasts, reservations, as_of, start, end, activity="individual",
              want_quiet=False, want_outlets=False, tie=0.05, k=3, cautious=True, src="det",
              min_coverage=0.5):
    """
    spaces:       space_id, name, capacity, use_type, quiet, outlets
    forecasts:    space_id, ts, model, yhat, yhat_upper   (campus time)
    reservations: space_id, start_ts, end_ts
    as_of:        the 'current' moment; start must not be before it
    """
    as_of, start, end = pd.Timestamp(as_of), pd.Timestamp(start), pd.Timestamp(end)
    if start < as_of:
        start = as_of
    slots = pd.date_range(start.ceil("15min"), end, freq="15min", inclusive="left")
    if len(slots) == 0:
        return pd.DataFrame()

    # 1. Keep only spaces that fit the activity
    ok = spaces.copy()
    if activity == "group":
        ok = ok[ok["use_type"].isin(["Group", "Individual or group"])]
    if want_quiet or activity == "online class":
        ok = ok[ok["quiet"].astype(bool)]

    # 2. For every slot, take the forecast made for that lead time
    want = pd.DataFrame({"ts": slots})
    want["model"] = [pick_model((t - as_of).total_seconds() / 60, src) for t in slots]
    f = forecasts.merge(want, on=["ts", "model"])

    results = []
    for _, sp in ok.iterrows():
        fs = f[f["space_id"] == sp["space_id"]].sort_values("ts")
        if len(fs) < min_coverage * len(slots):
            continue                                        # no forecast: closed or no data
        r = reservations[reservations["space_id"] == sp["space_id"]]
        reserved = np.array([((r["start_ts"] <= t) & (r["end_ts"] > t)).sum() for t in fs["ts"]])
        expected = fs["yhat_upper"].values if cautious else fs["yhat"].values
        por = (expected + reserved) / sp["capacity"]         # predicted occupancy ratio per slot
        results.append({"space_id": sp["space_id"], "name": sp["name"], "capacity": sp["capacity"],
                        "APOR": por.mean(), "PPOR": por.max(),
                        "pref_score": int(bool(want_outlets and sp["outlets"])) + int(bool(want_quiet and sp["quiet"])),
                        "expected_avg": fs["yhat"].mean(), "expected_peak": fs["yhat"].max(),
                        "reserved_now": int(reserved.max()) if len(reserved) else 0})
    df = pd.DataFrame(results)
    if df.empty:
        return df
    not_full = df[df["PPOR"] < 1.0]
    df = not_full if len(not_full) else df                  # never suggest a full room if another exists

    # 3. Rank: lowest worst moment (within the tie tolerance), then lowest average, then preferences
    df["PPOR_band"] = np.floor(df["PPOR"] / tie)
    df = df.sort_values(["PPOR_band", "APOR", "pref_score"], ascending=[True, True, False]).head(k)
    df["reason"] = df.apply(lambda x: f"expected at most {x.PPOR:.0%} full "
                                      f"(about {x.expected_peak:.0f} of {x.capacity} people) "
                                      f"between {start:%I:%M %p} and {end:%I:%M %p}", axis=1)
    return df.drop(columns=["PPOR_band"]).reset_index(drop=True)