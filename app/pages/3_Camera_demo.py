import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Camera demo · Baka Naman May Bakante", page_icon="🎥", layout="wide")
import db  # noqa: E402,F401  (sets the working folder)

st.title("From camera to count")
st.write("This replays a recorded overhead-camera sequence (CEPDOF, a public dataset) through the selected "
         "detector, YOLO26n at 1280 px. It shows the first step of the system: turning video frames into "
         "people counts. Frame times are spaced 10 seconds apart for the replay.")

path = "results/demo/demo_counts.csv"
if not os.path.exists(path):
    st.warning("Run `python src/detection/replay_demo.py` first to create the demo files.")
    st.stop()
d = pd.read_csv(path, parse_dates=["ts"])
m = pd.read_csv("results/demo/demo_per_minute.csv", parse_dates=["ts"])

c1, c2, c3 = st.columns(3)
c1.metric("Frames processed", len(d))
c2.metric("Counting error per frame (MAE)", f"{(d['pred'] - d['gt']).abs().mean():.2f} people")
c3.metric("Counting error per minute (MAE)", f"{(m['pred'] - m['gt']).abs().mean():.2f} people")

g = go.Figure()
g.add_trace(go.Scatter(x=m["ts"], y=m["gt"], name="True count", line=dict(color="#424242")))
g.add_trace(go.Scatter(x=m["ts"], y=m["pred"], name="Detected (YOLO26n)", line=dict(color="#1976d2")))
g.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="People",
                xaxis=dict(tickformat="%H:%M"), legend=dict(orientation="h", y=-0.25))
st.plotly_chart(g)
st.caption("Per-minute medians, the same smoothing the system uses before forecasting.")

saved = d[d["saved_frame"].fillna("").astype(str).str.len() > 0]
i = st.select_slider("Frame", options=saved["frame"].tolist(),
                     format_func=lambda k: f"{d.loc[k, 'ts']:%H:%M:%S}")
row = d.loc[i]
left, right = st.columns([3, 1])
left.image(row["saved_frame"], caption="Detected people are boxed")
right.metric("True count", int(row["gt"]))
right.metric("Detected", int(row["pred"]), delta=int(row["pred"] - row["gt"]), delta_color="off")