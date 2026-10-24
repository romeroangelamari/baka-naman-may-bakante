import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from data import load_campus
from recommend import recommend, empty_reservations

N_REQUESTS, SEED, TIE = 500, 42, 0.05
ACTIVITIES = ["individual", "group", "online class"]

spaces, fc, occ = load_campus()
occ = occ[occ["split"] == "test"]
test_days = sorted(occ["ts"].dt.normalize().unique())
true_by_room = {sp: g.set_index("ts")["count_true"] for sp, g in occ.groupby("space_id")}
cap = spaces.set_index("space_id")["capacity"]
rng = np.random.default_rng(SEED)


def eligible(activity, want_quiet):
    ok = spaces
    if activity == "group":
        ok = ok[ok["use_type"].isin(["Group", "Individual or group"])]
    if want_quiet or activity == "online class":
        ok = ok[ok["quiet"].astype(bool)]
    return ok["space_id"].tolist()


def actual_peak(sp, start, end):
    s = true_by_room.get(sp)
    if s is None:
        return np.nan
    w = s[(s.index >= start) & (s.index < end)].dropna()
    n_slots = (end - start) / pd.Timedelta(minutes=15)
    return w.max() / cap[sp] if len(w) >= 0.5 * n_slots else np.nan


# Static baseline: always rank rooms by their average daytime occupancy in the training period
train_avg = {sp: np.nan for sp in spaces["space_id"]}
_, _, occ_all = load_campus()
tr = occ_all[(occ_all["split"] == "train") & occ_all["ts"].dt.hour.between(8, 19)]
for sp, g in tr.groupby("space_id"):
    train_avg[sp] = (g["count_detector"] / cap[sp]).mean()

rows = []
for i in range(N_REQUESTS):
    day = pd.Timestamp(rng.choice(test_days))
    start = day + pd.Timedelta(minutes=int(rng.integers(8 * 4, 18 * 4)) * 15)        # 8:00 AM to 5:45 PM
    end = start + pd.Timedelta(minutes=int(rng.choice([60, 90, 120, 150, 180])))
    lead = int(rng.choice([0, 15, 30, 60, 120]))                                      # asked this many minutes before
    as_of = start - pd.Timedelta(minutes=lead)
    activity = rng.choice(ACTIVITIES)
    quiet, outlets = bool(rng.random() < 0.3), bool(rng.random() < 0.5)

    truth = {sp: actual_peak(sp, start, end) for sp in eligible(activity, quiet)}
    truth = {sp: v for sp, v in truth.items() if not np.isnan(v)}
    if len(truth) < 2:
        continue                                            # nothing to choose between
    best = min(truth.values())
    correct = {sp for sp, v in truth.items() if v <= best + TIE}    # ties count as correct

    picks = {}
    for name, kw in [("lgbm_det_cautious", dict(src="det", cautious=True)),
                     ("lgbm_det_plain", dict(src="det", cautious=False)),
                     ("lgbm_true_cautious", dict(src="true", cautious=True))]:
        top = recommend(spaces, fc, empty_reservations(), as_of, start, end, activity, quiet, outlets,
                        tie=TIE, k=3, **kw)
        top = [] if top.empty else top[top["space_id"].isin(truth)]["space_id"].tolist()
        picks[name] = top
    picks["static_least_used"] = sorted(truth, key=lambda sp: train_avg[sp])[:3]
    picks["random"] = list(rng.permutation(list(truth)))[:3]

    for name, top in picks.items():
        rows.append({"request": i, "policy": name, "lead_min": lead, "activity": activity,
                     "n_rooms": len(truth),
                     "top1_correct": bool(top) and top[0] in correct,
                     "top3_hit": bool(set(top) & correct),
                     "top1_had_space": bool(top) and truth[top[0]] < 1.0,
                     "top1_actual_peak": truth[top[0]] if top else np.nan})

res = pd.DataFrame(rows)
res.to_csv("results/rsr_requests.csv", index=False)
summary = res.groupby("policy").agg(requests=("request", "nunique"),
                                    RSR_top1=("top1_correct", "mean"), top3_hit=("top3_hit", "mean"),
                                    had_space=("top1_had_space", "mean"),
                                    avg_actual_peak=("top1_actual_peak", "mean"))
summary[["RSR_top1", "top3_hit", "had_space"]] *= 100
summary = summary.sort_values("RSR_top1", ascending=False).round(1)
summary.to_csv("results/rsr_summary.csv")
by_lead = res.pivot_table(index="lead_min", columns="policy", values="top1_correct", aggfunc="mean") * 100

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("Recommendation success rate on test days (%):")
print(summary)
print("\nRSR (%) by how far ahead the request was made (minutes):")
print(by_lead.round(1))