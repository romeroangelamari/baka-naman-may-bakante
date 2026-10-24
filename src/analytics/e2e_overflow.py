"""Step 41 extension: does counting accuracy matter when rooms get crowded?
Reruns the Step 38 crowding simulation with each scenario's forecasts (saved by end_to_end.py)."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "recommender"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from data import load_campus
from recommend import pick_model

STUDENTS = [50, 100, 200]
SLOTS = [10, 13, 15]
SEEDS = [1, 2, 3, 4, 5]
ACTIVITIES = ["individual", "group", "online class"]
TIE = 0.05

spaces, _, occ = load_campus()
ids = spaces["space_id"].tolist()
capv = spaces["capacity"].astype(float).values
quietv = spaces["quiet"].astype(bool).values
groupv = spaces["use_type"].isin(["Group", "Individual or group"]).values
occ = occ[occ["split"] == "test"]
days = sorted(occ["ts"].dt.normalize().unique())


def slot_table(fc, start, end, as_of):
    slots = pd.date_range(start, end, freq="15min", inclusive="left")
    want = pd.DataFrame({"ts": slots, "model": [pick_model((t - as_of).total_seconds() / 60, "x") for t in slots]})
    f = fc[(fc["ts"] >= start) & (fc["ts"] < end)].merge(want, on=["ts", "model"])
    t = f.pivot_table(index="space_id", columns="ts", values="yhat_upper").reindex(index=ids, columns=slots)
    return t.values, t.notna().sum(axis=1).values >= 0.5 * len(slots)


def eligible_mask(activity, quiet):
    m = np.ones(len(ids), bool)
    if activity == "group":
        m &= groupv
    if quiet or activity == "online class":
        m &= quietv
    return m


def fast_pick(expected, has_data, reserved, elig, quiet):
    cand = np.where(elig & has_data)[0]
    if len(cand) == 0:
        return None
    por = (expected[cand] + reserved[cand, None]) / capv[cand, None]
    ppor, apor = np.nanmax(por, axis=1), np.nanmean(por, axis=1)
    if (ppor < 1.0).any():
        keep = ppor < 1.0
        cand, ppor, apor = cand[keep], ppor[keep], apor[keep]
    pref = quietv[cand].astype(int) * int(quiet)
    return cand[np.lexsort((-pref, apor, np.floor(ppor / TIE)))[0]]


end_to_end = pd.read_csv("results/end_to_end.csv")
end_to_end["slug"] = (end_to_end["scenario"].str.replace(" (selected)", "", regex=False)
                      .str.replace(" errors", "", regex=False).str.replace(" ", "_"))
rows = []
for slug in end_to_end["slug"]:
    fc = pd.read_csv(f"results/e2e_forecasts/{slug}.csv", parse_dates=["ts"])
    for day in days:
        day = pd.Timestamp(day)
        for hour in SLOTS:
            start, end = day + pd.Timedelta(hours=hour), day + pd.Timedelta(hours=hour + 1)
            as_of = start - pd.Timedelta(minutes=15)
            w = occ[(occ["ts"] >= start) & (occ["ts"] < end)]
            background = w.groupby("space_id")["count_true"].max().reindex(ids).values   # real occupants
            if (~np.isnan(background)).sum() < 3:
                continue
            bg = np.nan_to_num(background)
            expected, has_data = slot_table(fc, start, end, as_of)
            for n in STUDENTS:
                for seed in SEEDS:
                    rng = np.random.default_rng(seed * 1000 + n)          # same students in every scenario
                    arrivals = np.zeros(len(ids))
                    overflow = 0
                    for _ in range(n):
                        activity, quiet = rng.choice(ACTIVITIES), bool(rng.random() < 0.3)
                        i = fast_pick(expected, has_data, arrivals, eligible_mask(activity, quiet), quiet)
                        if i is None:
                            continue
                        arrivals[i] += 1
                        overflow += bg[i] + arrivals[i] > capv[i]
                    rows.append({"scenario": slug, "day": day.date(), "hour": hour, "students": n, "seed": seed,
                                 "overflow_rate": overflow / n})
    print(f"{slug} done")

runs = pd.DataFrame(rows)
runs.to_csv("results/e2e_overflow_runs.csv", index=False)
table = runs.pivot_table(index="scenario", columns="students", values="overflow_rate", aggfunc="mean") * 100
table = table.reindex(end_to_end["slug"])

# Paired test of each scenario against perfect counts (same day, hour, students and seed)
key = ["day", "hour", "students", "seed"]
ref = runs[runs["scenario"] == "perfect"].set_index(key)["overflow_rate"]
for n in STUDENTS:
    pvals = []
    for slug in table.index:
        a = runs[(runs["scenario"] == slug) & (runs["students"] == n)].set_index(key)["overflow_rate"]
        b = ref.loc[a.index]
        pvals.append(wilcoxon(a, b).pvalue if (a != b).any() else 1.0)
    table[f"p vs perfect ({n})"] = pvals
table.insert(0, "count_MAE", end_to_end.set_index("slug")["count_MAE"])
table.to_csv("results/e2e_overflow.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\nStudents sent to an already-full room (%), soft reservations + cautious forecast:")
print(table.round(3))

fig, ax = plt.subplots(figsize=(7, 4.5))
t = table.sort_values("count_MAE")
for n in STUDENTS:
    ax.plot(t["count_MAE"], t[n], "o-", label=f"{n} students per hour")
ax.set_xlabel("Counting error (MAE, people per 15-min interval)")
ax.set_ylabel("Students sent to a full room (%)")
ax.set_title("Counting accuracy vs overcrowding")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/e2e_overflow.png", dpi=200)
print("\nSaved results/e2e_overflow.csv and results/figures/e2e_overflow.png")