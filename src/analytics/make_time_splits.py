import os
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
engine = sa.create_engine(os.getenv("DB_URL"))

occ = pd.read_sql("SELECT space_id, ts, count_true FROM occupancy", engine)
occ["date"] = pd.to_datetime(occ["ts"]).dt.date

rows = []
for sp, g in occ.dropna(subset=["count_true"]).groupby("space_id"):
    days = sorted(g["date"].unique())                     # only days that have data
    n_tr, n_va = round(len(days) * 0.70), round(len(days) * 0.15)
    for i, d in enumerate(days):
        split = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
        rows.append({"space_id": sp, "date": d, "split": split})

s = pd.DataFrame(rows)
s.to_csv("data/processed/time_splits.csv", index=False)
print(s.groupby(["space_id", "split"])["date"].agg(["count", "min", "max"]))