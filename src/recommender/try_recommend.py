import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from data import load_campus
from recommend import recommend, reserve, empty_reservations

spaces, fc, occ = load_campus()
print("spaces columns:", list(spaces.columns))
test_days = sorted(fc[fc["split"] == "test"]["ts"].dt.normalize().unique())
day = pd.Timestamp(test_days[2])
as_of = day + pd.Timedelta(hours=9, minutes=45)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
for activity, quiet in [("individual", False), ("group", False), ("individual", True)]:
    top = recommend(spaces, fc, empty_reservations(), as_of=as_of, start=day + pd.Timedelta(hours=10),
                    end=day + pd.Timedelta(hours=12), activity=activity, want_quiet=quiet)
    print(f"\nRequest on {day:%b %d}, 10:00-12:00, asked at 9:45, activity={activity}, quiet={quiet}")
    print(top[["space_id", "name", "PPOR", "APOR", "reason"]].round(2).to_string(index=False))

# Soft reservations: 30 students ask for the same thing, one after another
res = empty_reservations()
picks = []
for i in range(30):
    top = recommend(spaces, fc, res, as_of=as_of, start=day + pd.Timedelta(hours=10), end=day + pd.Timedelta(hours=11))
    picks.append(top["space_id"].iloc[0])
    res = reserve(res, picks[-1], day + pd.Timedelta(hours=10))
print("\nTop pick for 30 identical requests in a row:", " ".join(picks))
print("Students sent to each room:", pd.Series(picks).value_counts().to_dict())