import os
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()


def get_engine():
    return sa.create_engine(os.getenv("DB_URL"))


def to_campus_time(df, cal):
    """Move each row's timestamp onto the shared campus calendar; drop rows with no campus day."""
    df = df.copy()
    df["source_date"] = df["ts"].dt.normalize()
    df = df.merge(cal[["space_id", "source_date", "campus_date", "split"]], on=["space_id", "source_date"])
    df["ts"] = df["campus_date"] + (df["ts"] - df["source_date"])
    return df.drop(columns=["source_date", "campus_date"])


def load_campus(engine=None):
    engine = engine or get_engine()
    cal = pd.read_csv("data/processed/campus_calendar.csv", parse_dates=["source_date", "campus_date"])
    spaces = pd.read_sql("SELECT * FROM spaces", engine)
    fc = pd.read_sql("SELECT * FROM forecasts", engine, parse_dates=["ts"])
    occ = pd.read_sql("SELECT space_id, ts, count_true, count_detector FROM occupancy", engine, parse_dates=["ts"])
    return spaces, to_campus_time(fc, cal), to_campus_time(occ, cal)