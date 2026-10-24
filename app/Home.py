import streamlit as st

st.set_page_config(page_title="Baka Naman May Bakante", page_icon="🪑", layout="wide")
import db  # noqa: E402  (after set_page_config)

st.title("Baka Naman May Bakante")
st.subheader("Find a study space that will still have room when you get there.")
as_of = db.as_of_picker()
db.as_of_banner(as_of)

spaces, fc, occ = db.load_all()
st.write(
    f"This dashboard covers **{len(spaces)} study spaces**. It replays recorded occupancy data, "
    "forecasts how full each room will be, and recommends rooms for your study session."
)
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### For students")
    st.write("See how full each room is now, how full it is expected to be, and get three room suggestions "
             "for your time, activity and preferences.")
    st.page_link("pages/1_Student.py", label="Open the student page", icon="🎓")
with c2:
    st.markdown("#### For administrators")
    st.write("See usage patterns by hour and weekday, peak periods, and data quality. Password required.")
    st.page_link("pages/2_Administrator.py", label="Open the administrator page", icon="📊")

st.info("Use **Simulated clock** in the sidebar to choose the moment you want to see. "
        "Counts come from simulated camera counts on public datasets (ROBOD and SDU).")