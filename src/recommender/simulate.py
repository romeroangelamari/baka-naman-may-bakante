import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from data import load_campus
from recommend import recommend, empty_reservations, pick_model

STUDENTS = [25, 50, 100, 200]
SLOTS = [10, 13, 15]            # each simulated hour starts at 10 AM, 1 PM or 3 PM
SEEDS = [1, 2, 3, 4, 5]
POLICIES = {"soft_cautious": (True, True), "soft_plain": (True, False),       # (soft reservations, cautious)
            "greedy_cautious": (False, True), "greedy_plain": (False, False)}
ACTIVITIES = ["individual", "group", "online class"]
TIE = 0.05


def gini(x):
    """0 = load spread evenly across rooms, close to 1 = everyone in one room."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()) if x.sum() else 0.0


spaces, fc, occ = load_campus()
ids = spaces["space_id"].tolist()
capv = spaces["capacity"].astype(float).values
quietv = spaces["quiet"].astype(bool).values
groupv = spaces["use_type"].isin(["Group", "Individual or group"]).values
occ = occ[occ["split"] == "test"]
days = sorted(occ["ts"].dt.normalize().unique())


def slot_tables(start, end, as_of, cautious):
    """Expected count per room and 15-min slot, using the same lead-time rule as recommend()."""
    slots = pd.date_range(start, end, freq="15min", inclusive="left")
    want = pd.DataFrame({"ts": slots, "model": [pick_model((t - as_of).total_seconds() / 60) for t in slots]})
    f = fc[(fc["ts"] >= start) & (fc["ts"] < end)].merge(want, on=["ts", "model"])
    col = "yhat_upper" if cautious else "yhat"
    table = f.pivot_table(index="space_id", columns="ts", values=col).reindex(index=ids, columns=slots)
    return table.values, table.notna().sum(axis=1).values >= 0.5 * len(slots)


def eligible_mask(activity, quiet):
    m = np.ones(len(ids), bool)
    if activity == "group":
        m &= groupv
    if quiet or activity == "online class":
        m &= quietv
    return m


def fast_pick(expected, has_data, reserved, elig, quiet):
    """Same ranking as recommend() for the top room: lowest PPOR band, then APOR, then preferences."""
    cand = np.where(elig & has_data)[0]
    if len(cand) == 0:
        return None
    por = (expected[cand] + reserved[cand, None]) / capv[cand, None]
    ppor, apor = np.nanmax(por, axis=1), np.nanmean(por, axis=1)
    if (ppor < 1.0).any():                                  # never suggest a full room if another exists
        keep = ppor < 1.0
        cand, ppor, apor = cand[keep], ppor[keep], apor[keep]
    pref = quietv[cand].astype(int) * int(quiet)
    order = np.lexsort((-pref, apor, np.floor(ppor / TIE)))
    return cand[order[0]]


# Check once that fast_pick agrees with recommend()
for d in days[-2:]:
    s0 = pd.Timestamp(d) + pd.Timedelta(hours=10)
    a0 = s0 - pd.Timedelta(minutes=15)
    e_, h_ = slot_tables(s0, s0 + pd.Timedelta(hours=1), a0, True)
    for act in ACTIVITIES:
        i = fast_pick(e_, h_, np.zeros(len(ids)), eligible_mask(act, False), False)
        r = recommend(spaces, fc, empty_reservations(), a0, s0, s0 + pd.Timedelta(hours=1), act, k=1)
        assert (i is None and r.empty) or ids[i] == r["space_id"].iloc[0], "fast_pick disagrees with recommend()"
print("Check passed: fast_pick gives the same top room as recommend()")

rows, t0 = [], time.time()
for day in days:
    day = pd.Timestamp(day)
    for hour in SLOTS:
        start, end = day + pd.Timedelta(hours=hour), day + pd.Timedelta(hours=hour + 1)
        as_of = start - pd.Timedelta(minutes=15)
        w = occ[(occ["ts"] >= start) & (occ["ts"] < end)]
        background = w.groupby("space_id")["count_true"].max().reindex(ids).values   # people already there
        open_rooms = ~np.isnan(background)
        if open_rooms.sum() < 3:
            continue
        bg = np.nan_to_num(background)
        tables = {c: slot_tables(start, end, as_of, c) for c in [True, False]}
        for n in STUDENTS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + n)
                students = [(rng.choice(ACTIVITIES), bool(rng.random() < 0.3)) for _ in range(n)]
                for pname, (soft, cautious) in POLICIES.items():
                    expected, has_data = tables[cautious]
                    arrivals = np.zeros(len(ids))
                    overflow = unserved = 0
                    for activity, quiet in students:
                        i = fast_pick(expected, has_data, arrivals if soft else np.zeros(len(ids)),
                                      eligible_mask(activity, quiet), quiet)
                        if i is None:
                            unserved += 1
                            continue
                        arrivals[i] += 1
                        if bg[i] + arrivals[i] > capv[i]:
                            overflow += 1                                   # arrived to a full room
                    load = (bg + arrivals) / capv
                    rows.append({"day": day.date(), "hour": hour, "students": n, "seed": seed, "policy": pname,
                                 "overflow_rate": overflow / n, "unserved_rate": unserved / n,
                                 "peak_load": load[open_rooms].max(), "gini": gini(load[open_rooms]),
                                 "rooms_used": int((arrivals > 0).sum())})
        print(f"{day:%b %d} {hour}:00 done ({time.time() - t0:.0f} s)")

res = pd.DataFrame(rows)
res.to_csv("results/simulation_runs.csv", index=False)
summary = res.groupby(["students", "policy"])[["overflow_rate", "peak_load", "gini", "rooms_used"]].mean()
summary["overflow_rate"] *= 100
summary = summary.round(2)
summary.to_csv("results/simulation_summary.csv")
pd.set_option("display.width", 200)
print("\nAverage over test days, time slots and seeds (overflow_rate in %):")
print(summary)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
piv = summary["overflow_rate"].unstack("policy")
ax = piv.plot(marker="o", figsize=(7, 4))
ax.set_xlabel("Students asking for the same hour"); ax.set_ylabel("Students sent to a full room (%)")
ax.set_title("Overflow by recommendation policy"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/simulation_overflow.png", dpi=200)
print("\nSaved results/simulation_runs.csv, simulation_summary.csv, figures/simulation_overflow.png")