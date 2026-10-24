import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
engine = sa.create_engine(os.getenv("DB_URL"))

OPEN, CLOSE = 8, 20            # "daytime" window used for the main indicators (8 AM to 8 PM)
SDU = ["S4", "S5", "S6"]       # SDU dates are synthetic -> no weekday profile
os.makedirs("results/figures", exist_ok=True)

occ = pd.read_sql("""SELECT o.space_id, s.name, s.capacity, o.ts, o.count_true
                     FROM occupancy o JOIN spaces s ON s.space_id = o.space_id""", engine)
occ["ts"] = pd.to_datetime(occ["ts"])

summary, hourly, weekday = [], [], []
for sp, g in occ.groupby("space_id"):
    s = g.set_index("ts")["count_true"].dropna().sort_index()
    cap, name = g["capacity"].iloc[0], g["name"].iloc[0]
    pct = s / cap * 100
    day = pct[(pct.index.hour >= OPEN) & (pct.index.hour < CLOSE)]
    daily_peak = s.groupby(s.index.date).max()

    summary.append({
        "space_id": sp, "name": name, "capacity": cap,
        "days": s.index.normalize().nunique(),
        "avg_pct_all_hours": round(pct.mean(), 1),
        "avg_pct_8am_8pm": round(day.mean(), 1),
        "peak_count": s.max(),
        "peak_pct": round(s.max() / cap * 100, 1),
        "peak_at": s.idxmax(),
        "median_daily_peak": daily_peak.median(),
        "share_ge_50pct": round((day >= 50).mean() * 100, 2),   # % of daytime intervals
        "share_ge_80pct": round((day >= 80).mean() * 100, 2),
        "share_empty": round((day == 0).mean() * 100, 1),
    })

    h = pct.groupby(pct.index.hour).agg(mean_pct="mean", p90_pct=lambda x: x.quantile(0.9))
    h["space_id"], h["name"] = sp, name
    hourly.append(h.reset_index(names="hour"))

    if sp not in SDU:
        w = pct[(pct.index.hour >= OPEN) & (pct.index.hour < CLOSE)]
        w = w.groupby(w.index.dayofweek).mean().rename("mean_pct_8am_8pm").to_frame()
        w = w.reset_index(names="weekday")
        w["space_id"], w["name"] = sp, name
        weekday.append(w)

ind = pd.DataFrame(summary)
hp = pd.concat(hourly)
wp = pd.concat(weekday)
ind.to_csv("results/indicators.csv", index=False)
hp.to_csv("results/hourly_profile.csv", index=False)
wp.to_csv("results/weekday_profile.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(ind.to_string(index=False))
print("\nMean occupancy % by hour (8 AM to 8 PM):")
print(hp[hp.hour.between(OPEN, CLOSE - 1)].pivot(index="hour", columns="space_id", values="mean_pct").round(1))

# Figure: hourly profiles, one panel per dataset
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (title, ids) in zip(axes, [("ROBOD", ["S1", "S2", "S3"]), ("SDU", SDU)]):
    for sp in ids:
        d = hp[hp.space_id == sp]
        ax.plot(d.hour, d.mean_pct, marker="o", ms=3, label=f"{sp} {d.name.iloc[0]}")
    ax.set_title(f"{title}: mean occupancy by hour")
    ax.set_xlabel("Hour of day"); ax.set_xticks(range(0, 24, 2)); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Occupancy (% of capacity)")
fig.tight_layout(); fig.savefig("results/figures/hourly_profile.png", dpi=200)

# Figure: weekday profile (ROBOD only)
names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
piv = wp.pivot(index="weekday", columns="space_id", values="mean_pct_8am_8pm").reindex(range(5))
ax = piv.plot(kind="bar", figsize=(7, 4), rot=0)
ax.set_xticklabels(names); ax.set_ylabel("Mean occupancy 8 AM to 8 PM (%)")
ax.set_title("ROBOD: mean daytime occupancy by weekday"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/weekday_profile.png", dpi=200)

print("\nSaved results/indicators.csv, hourly_profile.csv, weekday_profile.csv and two figures in results/figures/")