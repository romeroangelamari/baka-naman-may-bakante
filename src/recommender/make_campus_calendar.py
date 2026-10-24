"""ROBOD (2021) and SDU (synthetic 2018 dates) do not share a calendar, so rooms cannot be
compared at the same moment. This puts SDU's validation and test days onto ROBOD's library
(S1) dates, split by split, so all six rooms share one 'campus calendar'."""
import pandas as pd

s = pd.read_csv("data/processed/time_splits.csv", parse_dates=["date"])
rows = []
for sp, g in s.groupby("space_id"):
    for _, r in g.iterrows():                      # ROBOD rooms keep their real dates
        rows.append({"space_id": sp, "source_date": r["date"], "campus_date": r["date"], "split": r["split"]})
cal = pd.DataFrame(rows)

ref = s[s["space_id"] == "S1"]
for sp in ["S4", "S5", "S6"]:
    cal = cal[cal["space_id"] != sp]
    for split in ["train", "val", "test"]:
        src = s[(s["space_id"] == sp) & (s["split"] == split)]["date"].sort_values().tolist()
        dst = ref[ref["split"] == split]["date"].sort_values().tolist()
        dst = dst[-len(src):] if len(src) <= len(dst) else dst      # align to the latest S1 dates
        src = src[-len(dst):]
        cal = pd.concat([cal, pd.DataFrame({"space_id": sp, "source_date": src,
                                            "campus_date": dst, "split": split})])

cal = cal.sort_values(["space_id", "source_date"])
cal.to_csv("data/processed/campus_calendar.csv", index=False)
summary = cal[cal["split"].isin(["val", "test"])].groupby(["campus_date"])["space_id"].apply(
    lambda x: " ".join(sorted(x)))
print("Rooms with data on each validation/test campus day:")
print(summary.to_string())