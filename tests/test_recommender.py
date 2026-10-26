"""Functional suitability tests for the recommender (ISO/IEC 25010)."""
import numpy as np
import pandas as pd
from recommend import recommend, reserve, empty_reservations, pick_model
from helpers import spaces, forecasts, at


def rec(**kw):
    args = dict(spaces=spaces(), forecasts=forecasts(), reservations=empty_reservations(),
                as_of=at(9, 45), start=at(10), end=at(12), activity="individual")
    args.update(kw)
    return recommend(**args)


# ---------- choosing the forecast by lead time ----------
def test_pick_model_uses_lightgbm_up_to_60_minutes():
    assert pick_model(0) == "lgbm_det_15min"
    assert pick_model(15) == "lgbm_det_15min"
    assert pick_model(30) == "lgbm_det_30min"
    assert pick_model(60) == "lgbm_det_60min"


def test_pick_model_uses_prophet_beyond_60_minutes():
    assert pick_model(75) == "prophet_det"
    assert pick_model(600) == "prophet_det"


# ---------- filtering ----------
def test_returns_at_most_three_rooms():
    assert len(rec()) <= 3


def test_group_request_excludes_individual_rooms():
    top = rec(activity="group", k=10)
    assert "A" not in set(top["space_id"])
    assert set(top["space_id"]) <= {"B", "C", "D"}


def test_quiet_request_returns_only_quiet_rooms():
    top = rec(want_quiet=True, k=10)
    assert set(top["space_id"]) == {"A"}


def test_online_class_needs_a_quiet_room():
    top = rec(activity="online class", k=10)
    assert set(top["space_id"]) == {"A"}


def test_room_without_forecast_is_skipped():
    top = rec(forecasts=forecasts(rooms=["A", "B"]), k=10)
    assert set(top["space_id"]) == {"A", "B"}


def test_no_matching_room_gives_empty_result():
    fc = forecasts(rooms=["B", "C", "D"])                 # the only quiet room has no data
    assert rec(forecasts=fc, want_quiet=True).empty


# ---------- ranking ----------
def test_least_full_room_ranks_first():
    top = rec(k=10)
    assert top.iloc[0]["space_id"] == "A"                 # 5 of 20 = lowest occupancy ratio


def test_never_recommends_a_full_room_when_another_has_space():
    fc = forecasts({"A": 25, "B": 10, "C": 30, "D": 90})  # A, C and D are expected over capacity
    top = rec(forecasts=fc, k=10)
    assert (top["PPOR"] < 1.0).all()
    assert list(top["space_id"]) == ["B"]


def test_busiest_moment_decides_not_the_average():
    fc = forecasts({"A": 2, "C": 6, "B": 20, "D": 60})
    peak = (fc["space_id"] == "A") & (fc["ts"] == at(11))  # A is nearly empty except one full slot
    fc.loc[peak, ["yhat", "yhat_upper"]] = [19.0, 19.5]
    top = rec(forecasts=fc, k=10)
    assert top.iloc[0]["space_id"] == "C"


def test_rooms_within_tie_tolerance_ranked_by_average():
    fc = forecasts({"A": 5, "C": 6})                      # PPOR 0.30 vs 0.29: same 5% band
    top = rec(forecasts=fc, k=10, tie=0.05)
    assert top.iloc[0]["APOR"] <= top.iloc[1]["APOR"]


def test_cautious_forecast_uses_upper_bound():
    cautious = rec(k=10, cautious=True).set_index("space_id")["PPOR"]
    plain = rec(k=10, cautious=False).set_index("space_id")["PPOR"]
    both = cautious.index.intersection(plain.index)
    assert len(both) > 0
    assert (cautious[both] >= plain[both]).all()


# ---------- soft reservations ----------
def test_reserve_adds_one_hour_reservation():
    res = reserve(empty_reservations(), "A", at(10))
    assert len(res) == 1
    assert res.iloc[0]["end_ts"] - res.iloc[0]["start_ts"] == pd.Timedelta(minutes=60)


def test_reservations_make_a_room_look_fuller():
    before = rec(k=10).set_index("space_id").loc["A", "PPOR"]
    res = empty_reservations()
    for _ in range(5):
        res = reserve(res, "A", at(10))
    after = rec(reservations=res, k=10).set_index("space_id").loc["A", "PPOR"]
    assert np.isclose(after - before, 5 / 20)


def test_many_identical_requests_spread_across_rooms():
    res, picks = empty_reservations(), []
    for _ in range(30):
        top = rec(reservations=res, as_of=at(9, 45), start=at(10), end=at(11))
        picks.append(top.iloc[0]["space_id"])
        res = reserve(res, picks[-1], at(10))
    assert len(set(picks)) >= 2


# ---------- time handling and explanations ----------
def test_start_before_as_of_is_moved_to_as_of():
    top = rec(as_of=at(10, 30), start=at(10), end=at(12))
    assert not top.empty


def test_empty_time_window_returns_empty():
    assert rec(start=at(12), end=at(12)).empty


def test_reason_percentage_matches_people_count():
    for _, r in rec(k=10).iterrows():
        people = int(r["reason"].split("up to ")[1].split(" of")[0])
        pct = int(r["reason"].split("at most ")[1].split("%")[0])
        assert abs(pct - round(people / r["capacity"] * 100)) <= 1


def test_reason_has_no_leading_zero_in_times():
    reason = rec(start=at(10), end=at(12)).iloc[0]["reason"]
    assert "10:00 AM" in reason and "12:00 PM" in reason and " 0" not in reason.split("between")[1]