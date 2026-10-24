# src/analytics/load_robod.py
# Loads ROBOD occupancy, converts it to 15-minute series, cleans it and saves it to MySQL.
import os, glob
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
engine = sa.create_engine(os.getenv("DB_URL"))

# Which ROBOD room becomes which study space (matched by occupancy pattern)
ROOM_TO_SPACE = {"Room4": "S1", "Room1": "S2", "Room2": "S3"}
capacity = pd.read_sql("SELECT space_id, capacity FROM spaces", engine).set_index("space_id")["capacity"]

all_rows, report = [], []
for room, space in ROOM_TO_SPACE.items():
    f = glob.glob(f"data/raw/robod/**/combined_{room}.csv", recursive=True)[0]
    d = pd.read_csv(f, usecols=["timestamp", "occupant_count [number]"])
    d["ts"] = pd.to_datetime(d["timestamp"].str[:16])            # drop "+08:00", keep local time
    s = d.drop_duplicates("ts").set_index("ts")["occupant_count [number]"]
    dupes = len(d) - len(s)

    # 15-minute intervals using the median; intervals with no readings stay missing (NaN), never 0
    s15 = s.resample("15min").median()

    negatives = int((s15 < 0).sum())
    s15 = s15.clip(lower=0)
    over_cap = int((s15 > capacity[space]).sum())                 # kept, but counted for the report

    all_rows.append(pd.DataFrame({"space_id": space, "ts": s15.index,
                                  "count_true": s15.values, "count_detector": None}))
    report.append({"space": space, "room": room, "intervals": len(s15),
                   "missing": int(s15.isna().sum()), "duplicates_removed": dupes,
                   "negatives_fixed": negatives, "above_capacity": over_cap,
                   "max": s15.max(), "mean": round(s15.mean(), 2)})

out = pd.concat(all_rows)
with engine.begin() as con:                                       # replace earlier ROBOD rows if re-run
    con.execute(sa.text("DELETE FROM occupancy WHERE space_id IN ('S1','S2','S3')"))
out.to_sql("occupancy", engine, if_exists="append", index=False, chunksize=2000)

print(pd.DataFrame(report).to_string(index=False))
print(f"Saved {len(out)} rows to the occupancy table")