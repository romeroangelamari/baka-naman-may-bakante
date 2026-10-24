import os, glob
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
engine = sa.create_engine(os.getenv("DB_URL"))

ROOM_TO_SPACE = {"room_1": "S4", "room_2": "S5", "room_3": "S6"}
START = "2018-01-01"   # synthetic start date (a Monday)

capacity = pd.read_sql("SELECT space_id, capacity FROM spaces", engine).set_index("space_id")["capacity"]

all_rows, report, day_maps = [], [], []
for room, space in ROOM_TO_SPACE.items():
    f = glob.glob(f"data/raw/sdu/**/occupant_count_{room}.csv", recursive=True)[0]
    d = pd.read_csv(f)

    # SDU gives DayId instead of dates -> place days on consecutive weekdays
    day_ids = (d.groupby("DayId")[["Year", "Month"]].first()
             .sort_values(["Year", "Month"], kind="stable").index.tolist())
    dates = pd.bdate_range(START, periods=len(day_ids))
    day_to_date = dict(zip(day_ids, dates))
    d["ts"] = d["DayId"].map(day_to_date) + pd.to_timedelta(d["Time"])

    s = d.drop_duplicates("ts").set_index("ts")[room].sort_index()
    dupes = len(d) - len(s)

    s15 = s.resample("15min").median()
    s15 = s15[s15.index.normalize().isin(dates)]      # drop the weekend gaps resample creates
    negatives = int((s15 < 0).sum())
    over_cap = int((s15 > capacity[space]).sum())
    max_raw = s15.max()
    s15 = s15.clip(lower=0, upper=capacity[space])

    all_rows.append(pd.DataFrame({"space_id": space, "ts": s15.index,
                                  "count_true": s15.values, "count_detector": None}))
    day_maps.append(pd.DataFrame({"space_id": space, "DayId": day_ids, "synthetic_date": dates.date}))
    report.append({"space": space, "room": room, "days": len(day_ids), "intervals": len(s15),
                   "missing": int(s15.isna().sum()), "duplicates": dupes, "negatives": negatives,
                   "over_capacity_clipped": over_cap, "max_before_clip": max_raw,
                   "mean": round(s15.mean(), 2)})

out = pd.concat(all_rows)
with engine.begin() as con:
    con.execute(sa.text("DELETE FROM occupancy WHERE space_id IN ('S4','S5','S6')"))
out.to_sql("occupancy", engine, if_exists="append", index=False, chunksize=2000)

os.makedirs("results", exist_ok=True)
pd.concat(day_maps).to_csv("data/processed/sdu_day_map.csv", index=False)
pd.DataFrame(report).to_csv("results/sdu_load_report.csv", index=False)
print(pd.DataFrame(report).to_string(index=False))