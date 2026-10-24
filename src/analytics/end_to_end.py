"""Phase 6: run the whole chain (counting errors -> forecast -> recommendation) for several
counting-error scenarios and compare each with the perfect-count pipeline."""
import sys, os, glob, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "forecasting"))
sys.path.insert(0, os.path.join(ROOT, "recommender"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_series, metrics
from lgbm_core import fit_predict
from data import to_campus_time
from recommend import recommend, empty_reservations
from common import SDU
from prophet import Prophet
import logging
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.WARNING)

SEED, N_REQUESTS, TIE = 42, 500, 0.05
EXCLUDE = ["scut_head", "library_seats"]
DAY = (8, 20)

# ---------- 1. Error profiles: how each counting source makes mistakes ----------
def level(x):
    return np.where(x <= 5, 0, np.where(x <= 15, 1, 2))

bench = pd.read_csv("results/benchmark_test_1280.csv")
bench = bench[(bench["gt"] <= 90) & ~bench["dataset"].isin(EXCLUDE)].copy()

def yolo_profile(model, stress=1.0):
    b = bench[bench["model"] == model]
    lv = level(b["gt"].values)
    e = (b["pred"] - b["gt"]).values
    r = (b["pred"] / b["gt"].clip(lower=1)).values
    return {"abs": [e[lv == 0] * stress],
            "ratio": [None, 1 - (1 - r[(lv == 1)]) * stress, 1 - (1 - r[(lv == 2)]) * stress]}

def camera_profile():
    """Real fisheye-camera errors from CO2FISHEYE, at 15-minute resolution."""
    parts = []
    for f in glob.glob("data/raw/co2fisheye/GT+Fisheye day *.csv"):
        d = pd.read_csv(f, encoding="utf-8-sig"); d.columns = ["time", "gt", "camera"]
        d["ts"] = pd.to_datetime("2000-01-01 " + d["time"].astype(str).str.strip(), format="%Y-%m-%d %H:%M:%S")
        d["day"] = f
        parts.append(d.dropna())
    q = (pd.concat(parts).set_index("ts").groupby("day")[["gt", "camera"]].resample("15min").median().dropna())
    lv = level(q["gt"].values); e = (q["camera"] - q["gt"]).values
    return {"abs": [e[lv == k] for k in range(3)], "ratio": None}

def inject(true, prof, rng):
    x = true.values.astype(float)
    out = np.full(len(x), np.nan)
    lv = level(np.nan_to_num(x))
    for i, (v, l) in enumerate(zip(x, lv)):
        if np.isnan(v):
            continue
        if prof["ratio"] is None or l == 0:
            pool = prof["abs"][l] if prof["ratio"] is None else prof["abs"][0]
            out[i] = v + rng.choice(pool)
        else:
            out[i] = v * max(rng.choice(prof["ratio"][l]), 0)
    return out

SCENARIOS = {
    "perfect": None,
    "calibrated_camera": camera_profile(),
    "yolo26s_finetuned": yolo_profile("models/yolo26s_cepdof/weights/best.pt"),
    "yolo26s": yolo_profile("yolo26s.pt"),
    "yolov8s": yolo_profile("yolov8s.pt"),
    "yolo26n (selected)": yolo_profile("yolo26n.pt"),
    "yolo26n x2 errors": yolo_profile("yolo26n.pt", stress=2.0),
    "yolo26n x3 errors": yolo_profile("yolo26n.pt", stress=3.0),
    "yolo26n x4 errors": yolo_profile("yolo26n.pt", stress=4.0),
}


def prophet_far(occ_s, cap):
    """Prophet on the same noisy counts, used for slots more than 60 minutes ahead (as in the real system)."""
    out = []
    for sp, g in occ_s.groupby("space_id"):
        tr = g[(g["split"] == "train") & g["obs"].notna()]
        m = Prophet(growth="flat", daily_seasonality=False, weekly_seasonality=False,
                    yearly_seasonality=False, interval_width=0.8)
        m.add_seasonality(name="daily", period=1, fourier_order=10)
        if sp not in SDU:
            m.add_seasonality(name="weekly", period=7, fourier_order=3)
        m.fit(tr[["ts", "obs"]].rename(columns={"ts": "ds", "obs": "y"}))
        fut = g[g["split"].isin(["val", "test"])][["ts"]]
        p = m.predict(fut.rename(columns={"ts": "ds"}))
        out.append(pd.DataFrame({"space_id": sp, "ts": fut["ts"].values, "model": "prophet_x",
                                 "yhat": p["yhat"].clip(0, cap[sp]).values,
                                 "yhat_upper": p["yhat_upper"].clip(0, cap[sp]).values}))
    return pd.concat(out, ignore_index=True)

# ---------- 2. Shared inputs ----------
occ, cap = load_series()
cal = pd.read_csv("data/processed/campus_calendar.csv", parse_dates=["source_date", "campus_date"])
from data import get_engine
spaces = pd.read_sql("SELECT * FROM spaces", get_engine())
test_true = to_campus_time(occ[["space_id", "ts", "count_true"]], cal)
test_true = test_true[test_true["split"] == "test"]
true_by_room = {sp: g.set_index("ts")["count_true"] for sp, g in test_true.groupby("space_id")}
test_days = sorted(test_true["ts"].dt.normalize().unique())

rng = np.random.default_rng(SEED)
requests = []
for i in range(N_REQUESTS):                      # same requests for every scenario
    day = pd.Timestamp(rng.choice(test_days))
    start = day + pd.Timedelta(minutes=int(rng.integers(32, 72)) * 15)
    end = start + pd.Timedelta(minutes=int(rng.choice([60, 90, 120, 150, 180])))
    lead = int(rng.choice([0, 15, 30, 60, 120]))
    requests.append(dict(as_of=start - pd.Timedelta(minutes=lead), start=start, end=end,
                         activity=str(rng.choice(["individual", "group", "online class"])),
                         want_quiet=bool(rng.random() < 0.3), want_outlets=bool(rng.random() < 0.5)))

def eligible(activity, quiet):
    ok = spaces
    if activity == "group":
        ok = ok[ok["use_type"].isin(["Group", "Individual or group"])]
    if quiet or activity == "online class":
        ok = ok[ok["quiet"].astype(bool)]
    return ok["space_id"].tolist()

def actual_peak(sp, start, end):
    s = true_by_room.get(sp)
    if s is None:
        return np.nan
    w = s[(s.index >= start) & (s.index < end)].dropna()
    return w.max() / cap[sp] if len(w) >= 0.5 * (end - start) / pd.Timedelta(minutes=15) else np.nan

# ---------- 3. Run the chain for each scenario ----------
results, tops, t0 = [], {}, time.time()
for name, prof in SCENARIOS.items():
    print(f"\n{name}  ({time.time() - t0:.0f} s)")
    srng = np.random.default_rng(SEED)
    occ_s = occ.copy()
    occ_s["obs"] = occ_s["count_true"] if prof is None else 0.0
    if prof is not None:
        for sp, idx in occ_s.groupby("space_id").groups.items():
            occ_s.loc[idx, "obs"] = np.clip(np.round(inject(occ_s.loc[idx, "count_true"], prof, srng)), 0, cap[sp])

    day_test = occ_s[(occ_s["split"] == "test") & occ_s["ts"].dt.hour.between(DAY[0], DAY[1] - 1)]
    count_mae = metrics(day_test["count_true"], day_test["obs"])["MAE"]

    preds = fit_predict(occ_s, cap, "obs", verbose=True)
    pt = preds[(preds["split"] == "test") & preds["ts"].dt.hour.between(DAY[0], DAY[1] - 1)]
    fmae = {h: metrics(g["actual"], g["yhat"])["MAE"] for h, g in pt.groupby("horizon")}

    fc = preds.assign(model="lgbm_x_" + preds["horizon"])[["space_id", "ts", "model", "yhat", "yhat_upper"]]
    fc = to_campus_time(pd.concat([fc, prophet_far(occ_s, cap)], ignore_index=True), cal)
    slug = name.replace(" (selected)", "").replace(" errors", "").replace(" ", "_")
    os.makedirs("results/e2e_forecasts", exist_ok=True)
    fc[fc["split"] == "test"].to_csv(f"results/e2e_forecasts/{slug}.csv", index=False)
    tops[name] = []
    for rq in requests:
        top = recommend(spaces, fc, empty_reservations(), rq["as_of"], rq["start"], rq["end"], rq["activity"],
                        rq["want_quiet"], rq["want_outlets"], tie=TIE, k=3, cautious=True, src="x")
        tops[name].append([] if top.empty else top["space_id"].tolist())

    hits = []
    for rq, top in zip(requests, tops[name]):
        truth = {sp: actual_peak(sp, rq["start"], rq["end"]) for sp in eligible(rq["activity"], rq["want_quiet"])}
        truth = {sp: v for sp, v in truth.items() if not np.isnan(v)}
        if len(truth) < 2 or not top:
            continue
        best = min(truth.values())
        hits.append(top[0] in truth and truth[top[0]] <= best + TIE)
    results.append({"scenario": name, "count_MAE": count_mae, "forecast_MAE_15min": fmae.get("15min"),
                    "forecast_MAE_30min": fmae.get("30min"), "forecast_MAE_60min": fmae.get("60min"),
                    "RSR_top1": np.mean(hits) * 100, "requests_scored": len(hits)})

# ---------- 4. Agreement with the perfect-count pipeline ----------
ref = tops["perfect"]
for r in results:
    pairs = [(a, b) for a, b in zip(tops[r["scenario"]], ref) if a and b]
    r["top1_agreement"] = np.mean([a[0] == b[0] for a, b in pairs]) * 100
    r["top3_agreement"] = np.mean([len(set(a) & set(b)) / max(len(set(b)), 1) for a, b in pairs]) * 100

res = pd.DataFrame(results).round(2)
res["forecast_MAE_increase_30min"] = (res["forecast_MAE_30min"] - res.loc[0, "forecast_MAE_30min"]).round(2)
res.to_csv("results/end_to_end.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\nEnd-to-end results (test days, 8 AM to 8 PM):")
print(res.to_string(index=False))

# ---------- 5. Trade-off chart (Step 41) ----------
res = res.sort_values("count_MAE")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(res["count_MAE"], res["top3_agreement"], "o-",
        label="top-3 agreement with perfect counts")
ax.plot(res["count_MAE"], res["RSR_top1"], "s--", label="top-1 RSR")
for _, r in res.iterrows():
    xy = (r["count_MAE"], r["top3_agreement"])
    ax.annotate(r["scenario"], xy, fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Counting error (MAE, people per 15-min interval)")
ax.set_ylabel("%")
ax.set_title("How counting accuracy affects recommendations")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/end_to_end_tradeoff.png", dpi=200)
print("\nSaved results/end_to_end.csv and results/figures/end_to_end_tradeoff.png")