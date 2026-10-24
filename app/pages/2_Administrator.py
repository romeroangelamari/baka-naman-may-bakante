import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Administrator · Baka Naman May Bakante", page_icon="📊", layout="wide")
import db  # noqa: E402

st.title("Administrator view")
password = st.text_input("Admin password", type="password")
if password != st.secrets.get("admin_password", None):
    if password:
        st.error("Wrong password.")
    st.stop()

spaces, fc, occ = db.load_all()
as_of = db.as_of_picker()
db.as_of_banner(as_of)
names = spaces.set_index("space_id")["name"]
cap = spaces.set_index("space_id")["capacity"]

hist = occ[occ["ts"] <= as_of].copy()                  # only what was known at the as-of time
hist["pct"] = hist["count_detector"] / hist["space_id"].map(cap) * 100
hist["hour"] = hist["ts"].dt.hour
hist["weekday"] = hist["ts"].dt.day_name().str[:3]
day = hist[hist["hour"].between(8, 19)]

# ---------- 1. Average use per room ----------
st.subheader("Average daytime occupancy (8 AM to 8 PM)")
avg = day.groupby("space_id")["pct"].mean().rename(index=names).sort_values()
fig = px.bar(avg, orientation="h", labels={"value": "Average occupancy (% of capacity)", "space_id": ""})
fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig)

# ---------- 2. Heatmap ----------
st.subheader("Usage by hour and weekday")
room = st.selectbox("Room", spaces["space_id"], format_func=lambda s: names[s])
h = day[day["space_id"] == room].pivot_table(index="weekday", columns="hour", values="pct", aggfunc="mean")
h = h.reindex(["Mon", "Tue", "Wed", "Thu", "Fri"])
fig = px.imshow(h, aspect="auto", color_continuous_scale="Blues", text_auto=".0f",
                labels=dict(x="Hour of day", y="", color="% full"))
fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig)
if room in ["S4", "S5", "S6"]:
    st.caption("SDU rooms have synthetic weekdays (their real dates were anonymised), so weekday "
               "differences for this room are not meaningful.")

# ---------- 3. Peak periods ----------
st.subheader("Peak periods")
peaks = (day.groupby(["space_id", "hour"])["pct"].mean().reset_index()
         .sort_values("pct", ascending=False).groupby("space_id").head(2))
peaks["Room"] = peaks["space_id"].map(names)
peaks["Hour"] = peaks["hour"].map(lambda x: f"{x:02d}:00–{x + 1:02d}:00")
peaks["Average occupancy"] = peaks["pct"].round(0).astype(int).astype(str) + "%"
st.dataframe(peaks[["Room", "Hour", "Average occupancy"]].sort_values("Room"), hide_index=True)

# ---------- 4. Data quality ----------
st.subheader("Data quality")
q = occ.groupby("space_id").agg(intervals=("ts", "size"),
                                missing=("count_true", lambda s: int(s.isna().sum())),
                                at_or_above_capacity=("count_detector", "count"))
q["at_or_above_capacity"] = [int((occ.loc[occ["space_id"] == s, "count_detector"] >= cap[s]).sum()) for s in q.index]
q["missing %"] = (q["missing"] / q["intervals"] * 100).round(1)
q.insert(0, "Room", q.index.map(names))
st.dataframe(q, hide_index=True)
try:
    m = pd.read_csv("results/lgbm_metrics_detector.csv")
    m = m[(m["split"] == "test") & (m["period"] == "8am_8pm")]
    m = m.pivot_table(index="space_id", columns="horizon", values="MAE").rename(index=names).round(2)
    st.markdown("**Forecast error on test days** (mean absolute error, people, 8 AM to 8 PM)")
    st.dataframe(m)
except FileNotFoundError:
    st.caption("Forecast accuracy file not found (results/lgbm_metrics_detector.csv).")

    # ---------- 5. Soft reservations ----------
st.subheader("Soft reservations")
res = db.get_reservations()
if res.empty:
    st.caption("No soft reservations yet.")
else:
    summary = (res.assign(Room=res["space_id"].map(names))
               .groupby("Room").agg(Reservations=("start_ts", "size"),
                                    First=("start_ts", "min"), Last=("end_ts", "max")))
    st.dataframe(summary)
if st.button("Clear all soft reservations"):
    db.clear_reservations()
    st.success("All soft reservations were removed.")
    st.rerun()