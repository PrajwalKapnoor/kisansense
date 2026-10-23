import streamlit as st

import api_client
import components as ui

st.set_page_config(page_title="Price explorer · KisanSense", page_icon="📈", layout="centered")
ui.page_header("📈 Tomato price explorer", "Karnataka daily modal price, with a cautious 7-day outlook")


@st.cache_data(ttl=600, show_spinner=False)
def load_forecast(days: int) -> dict:
    return api_client.price_forecast(days)


days = st.select_slider("Past days to show", options=[30, 60, 90, 180, 365], value=90)

try:
    data = load_forecast(days)
except api_client.ApiError as e:
    ui.show_error(e)
else:
    ui.render_price(data)

with st.expander("How accurate has this been?"):
    try:
        ui.render_price_metrics(api_client.metrics("price"))
    except api_client.ApiError as e:
        st.caption(str(e))
    st.caption(
        "Prices depend on transport cost, quality, storage and local demand, which this model does not see. "
        "Use the outlook as one input, not as a reason to sell or hold."
    )

ui.footer()
