"""KisanSense home page. Run from the project root:
    streamlit run app/Home.py
"""
import streamlit as st

import api_client
import components as ui

st.set_page_config(page_title="KisanSense", page_icon="🌱", layout="centered")

ui.page_header("KisanSense", "Smart agricultural decision support for farmers")
st.write(
    "Three tools in one place: check a tomato leaf for disease, get crop suggestions for your soil and weather, "
    "and see where tomato prices in Karnataka are heading. You do not need to know anything about machine learning."
)

cards = [
    ("pages/1_Disease_Check.py", "🍅", "Disease check", "Upload a tomato leaf photo and see what the model thinks it is, and why."),
    ("pages/2_Crop_Advisor.py", "🌾", "Crop advisor", "Enter soil nutrients and weather to get the three best-matching crops."),
    ("pages/3_Price_Explorer.py", "📈", "Price explorer", "Karnataka tomato prices, a 7-day outlook and how accurate it has been."),
]
cols = st.columns(3)
for col, (path, icon, title, text) in zip(cols, cards):
    with col, st.container(border=True):
        st.subheader(f"{icon} {title}")
        st.write(text)
        st.page_link(path, label="Open")

st.subheader("How to use it")
st.markdown(
    "1. Pick a tool above.\n"
    "2. Give it your photo or numbers.\n"
    "3. Read the result **together with its confidence and limits**. The tools can be wrong."
)

with st.expander("System status"):
    try:
        status = api_client.health()
        names = {"disease": "Disease detection", "crop": "Crop recommendation", "price": "Price forecasting"}
        for key, label in names.items():
            ready = status["models"].get(key, False)
            st.write(("✅ " if ready else "⚠️ ") + f"{label}: " + ("ready" if ready else "model not trained yet"))
    except api_client.ApiError as e:
        st.error(str(e))

ui.footer()
