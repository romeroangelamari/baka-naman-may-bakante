"""Shared data loading and helpers for every dashboard page."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "recommender"))
os.chdir(ROOT)                                  # so relative paths like data/processed/... work
import numpy as np
import pandas as pd
import streamlit as st
import sqlalchemy as sa
from data import load_campus, get_engine
from recommend import recommend, reserve, empty_reservations, pick_model

LEVELS = [(0.5, "Available", "#2e7d32"), (0.8, "Moderately occupied", "#f9a825"),
          (1.0, "Nearly full", "#ef6c00"), (np.inf, "Full", "#c62828")]


@st.cache_data(show_spinner="Loading occupancy data...")
def load_all():
    spaces, fc, occ = load_campus()
    spaces["quiet"] = spaces["quiet"].astype(bool)
    spaces["outlets"] = spaces["outlets"].astype(bool)
    return spaces, fc, occ


def level(ratio):
    """Text label and colour for an occupancy ratio (always shown as text, not colour alone)."""
    if ratio is None or pd.isna(ratio):
        return "No data", "#9e9e9e"
    for limit, label, colour in LEVELS:
        if ratio < limit:
            return label, colour
    return LEVELS[-1][1], LEVELS[-1][2]


def as_of_picker():
    """Sidebar control for the simulated 'current' moment. Returns a Timestamp."""
    spaces, fc, occ = load_all()
    days = sorted(fc["ts"].dt.normalize().unique())
    labels = [pd.Timestamp(d).strftime("%a %b %d, %Y") for d in days]
    st.sidebar.header("Simulated clock")
    i = st.sidebar.selectbox("Day", range(len(days)), format_func=lambda k: labels[k],
                             index=max(len(days) - 3, 0), key="day_index")
    times = pd.date_range("07:00", "21:45", freq="15min").strftime("%I:%M %p").tolist()
    t = st.sidebar.select_slider("Time", options=times, value="10:00 AM",
                                 key="time_label")
    as_of = pd.Timestamp(days[i]) + (pd.Timestamp(t) - pd.Timestamp(t).normalize())
    st.sidebar.caption("This is a replay of historical data, not a live feed.")
    return as_of


def as_of_banner(as_of):
    st.caption(f"Data shown as of **{as_of:%A, %B %d, %I:%M %p}** (historical replay, not live)")


def current_counts(occ, spaces, as_of):
    """Latest camera (detector) count at or before as_of on the same day, per room."""
    today = occ[(occ["ts"] <= as_of) & (occ["ts"] >= as_of.normalize())].sort_values("ts")
    last = today.groupby("space_id").tail(1).set_index("space_id")
    out = spaces.set_index("space_id")[["name", "capacity", "x", "y", "quiet", "outlets", "use_type"]].copy()
    out["count_now"] = last["count_detector"].reindex(out.index)
    out["ratio_now"] = out["count_now"] / out["capacity"]
    return out


def forecast_path(fc, space_id, as_of, hours_ahead=None):
    """Forecast for the rest of the day, each slot taken from the model that fits its lead time."""
    first = as_of.floor("15min") + pd.Timedelta(minutes=15)          # the next 15-minute slot
    if hours_ahead is None:
        last = as_of.normalize() + pd.Timedelta(hours=23, minutes=45)  # rest of the day
    else:
        last = as_of + pd.Timedelta(hours=hours_ahead)
    slots = pd.date_range(first, last, freq="15min")
    want = pd.DataFrame({"ts": slots, "model": [pick_model((t - as_of).total_seconds() / 60) for t in slots]})
    f = fc[fc["space_id"] == space_id].merge(want, on=["ts", "model"])
    return f.sort_values("ts")


@st.cache_resource
def engine():
    """One database connection pool shared by all users of the app."""
    return get_engine()


def get_reservations():
    """All soft reservations, shared by every user (stored in MySQL)."""
    df = pd.read_sql("SELECT space_id, start_ts, end_ts FROM reservations", engine(),
                     parse_dates=["start_ts", "end_ts"])
    return df if len(df) else empty_reservations()


def add_reservation(space_id, start, minutes=60):
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(minutes=minutes)
    with engine().begin() as con:
        con.execute(sa.text("INSERT INTO reservations (space_id, start_ts, end_ts) "
                            "VALUES (:s, :a, :b)"),
                    {"s": space_id, "a": start.to_pydatetime(), "b": end.to_pydatetime()})


def clear_reservations():
    with engine().begin() as con:
        con.execute(sa.text("DELETE FROM reservations"))