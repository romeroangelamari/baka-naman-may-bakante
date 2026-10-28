import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Student · Baka Naman May Bakante", page_icon="🎓", layout="wide")
import db  # noqa: E402

spaces, fc, occ = db.load_all()
as_of = db.as_of_picker()
st.title("Find a study space")
db.as_of_banner(as_of)

now = db.current_counts(occ, spaces, as_of)
next_peak = {}
for sp in now.index:
    f = db.forecast_path(fc, sp, as_of, hours_ahead=1)
    next_peak[sp] = f["yhat"].max() if len(f) else None
now["next_hour_peak"] = pd.Series(next_peak)

# ---------- 1. Occupancy map ----------
st.subheader("Rooms right now")
labels = [db.level(r) for r in now["ratio_now"]]
fig = go.Figure(go.Scatter(
    x=now["x"], y=now["y"], mode="markers+text",
    marker=dict(size=54, color=[c for _, c in labels], line=dict(width=2, color="white")),
    text=[f"<b>{n}</b><br>{lab}" for n, (lab, _) in zip(now["name"], labels)],
    textposition="bottom center",
    hovertext=[f"{n}: {'no data' if pd.isna(c) else int(c)} of {cap} people"
               for n, c, cap in zip(now["name"], now["count_now"], now["capacity"])],
    hoverinfo="text"))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                  xaxis=dict(visible=False, range=[0, 11]), yaxis=dict(visible=False, range=[0, 7]))
st.plotly_chart(fig)
st.caption("Levels: Available (under 50%), Moderately occupied (50–79%), Nearly full (80–99%), Full (100% or more).")

# ---------- 2. Room cards ----------
cols = st.columns(3)
for k, (sp, r) in enumerate(now.iterrows()):
    lab, _ = db.level(r["ratio_now"])
    with cols[k % 3].container(border=True):
        st.markdown(f"**{r['name']}**  \n{lab}")
        count = "no data" if pd.isna(r["count_now"]) else f"{int(r['count_now'])} of {r['capacity']}"
        st.metric("People now (camera count)", count)
        nxt = r["next_hour_peak"]
        st.caption("Next hour: " + ("no forecast" if pd.isna(nxt) else f"up to about {nxt:.0f} people"))
        tags = [r["use_type"]] + (["quiet"] if r["quiet"] else []) + (["outlets"] if r["outlets"] else [])
        st.caption(" · ".join(tags))

# ---------- 3. Forecast chart ----------
st.subheader("Forecast for the rest of the day")
choice = st.selectbox("Room", now.index, format_func=lambda s: now.loc[s, "name"])
past = occ[(occ["space_id"] == choice) & (occ["ts"] >= as_of.normalize()) & (occ["ts"] <= as_of)]
path = db.forecast_path(fc, choice, as_of)
cap = int(now.loc[choice, "capacity"])
g = go.Figure()
if len(path):
    g.add_trace(go.Scatter(x=path["ts"], y=path["yhat_upper"], line=dict(width=0), showlegend=False,
                           hoverinfo="skip"))
    g.add_trace(go.Scatter(x=path["ts"], y=path["yhat"], fill="tonexty", fillcolor="rgba(25,118,210,0.15)",
                           line=dict(color="#1976d2"), name="Forecast (shaded: cautious upper range)"))
g.add_trace(go.Scatter(x=past["ts"], y=past["count_detector"], line=dict(color="#424242"),
                       name="Camera count so far"))
g.add_hline(y=cap, line_dash="dash", line_color="#c62828", annotation_text=f"Capacity ({cap})")
g.add_vline(x=as_of, line_dash="dot", line_color="#757575")
g.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="People",
                legend=dict(orientation="h", y=-0.2))
st.plotly_chart(g)
st.caption("Up to 60 minutes ahead the forecast uses recent counts (LightGBM); further ahead it uses "
           "the room's usual daily pattern (Prophet).")

# ---------- 4. Recommendation form ----------
st.subheader("Get a recommendation")
times = pd.date_range(as_of.ceil("15min"), as_of.normalize() + pd.Timedelta(hours=22), freq="15min")
if len(times) < 2:
    st.warning("It's too late in the day to plan a session. Pick an earlier time in the sidebar.")
    st.stop()
with st.form("request"):
    c1, c2, c3 = st.columns(3)
    start = c1.selectbox("Start", times[:-1], format_func=lambda t: f"{t:%I:%M %p}")
    hours = c2.selectbox("Length", [1, 1.5, 2, 2.5, 3], index=1, format_func=lambda h: f"{h:g} hours")
    activity = c3.selectbox("Activity", ["individual", "group", "online class"],
                            format_func=lambda a: {"individual": "Studying alone", "group": "Group study",
                                                   "online class": "Online class (needs quiet)"}[a])
    q1, q2 = st.columns(2)
    want_quiet = q1.checkbox("I need a quiet room")
    want_outlets = q2.checkbox("I'd like power outlets")
    submitted = st.form_submit_button("Recommend rooms", type="primary")

if submitted:
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=hours)
    t0, t1 = f"{start:%I:%M %p}".lstrip("0"), f"{end:%I:%M %p}".lstrip("0")
    res = db.get_reservations()                     # soft reservations of all users (MySQL)
    top = db.recommend(spaces, fc, res, as_of, start, end, activity, want_quiet, want_outlets, k=3)

    if top.empty:
        st.error("No room matches this request with forecast data. Try another time or fewer filters.")
    else:
        # True when even the best room is expected to be full at its busiest moment
        all_full = bool((top["PPOR"] >= 1.0).all())
        if all_full:
            st.warning(f"Every suitable room is expected to be full between {t0} and {t1}. "
                       "These are the least crowded options, but a different time is recommended.")
        else:
            st.success(f"Best rooms for {t0} to {t1}:")

        has_outlets = spaces.set_index("space_id")["outlets"]
        for rank, r in top.iterrows():
            lab, _ = db.level(r["PPOR"])
            extra = ["has outlets"] if want_outlets and has_outlets.loc[r["space_id"]] else []
            with st.container(border=True):
                st.markdown(f"**{rank + 1}. {r['name']}** — {lab} at its busiest")
                st.write(r["reason"] + (f"; {', '.join(extra)}" if extra else "") + ".")

        if all_full:
            st.caption("No soft reservation was added, because every suitable room is expected to be full.")
        else:
            db.add_reservation(top["space_id"].iloc[0], start)
            st.caption(f"A soft reservation was added for {top['name'].iloc[0]}, so the next student "
                       f"sees it as one person fuller. Soft reservations for all users: "
                       f"{len(db.get_reservations())}.")