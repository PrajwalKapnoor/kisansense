import streamlit as st

import api_client
import components as ui

st.set_page_config(page_title="Crop advisor · KisanSense", page_icon="🌾", layout="centered")
ui.page_header("🌾 Crop advisor", "Enter your soil and weather details to see which crops match best")

st.caption("The starting values are only an example. Replace them with your own soil test and local weather.")

with st.form("crop_form"):
    values = {}
    cols = st.columns(2)
    for i, (key, label, lo, hi, default, step, help_text) in enumerate(ui.CROP_FIELDS):
        with cols[i % 2]:
            values[key] = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step, help=help_text)
    submitted = st.form_submit_button("Suggest crops", type="primary")

if submitted:
    with st.spinner("Working it out..."):
        try:
            result = api_client.recommend_crop(values)
        except api_client.ApiError as e:
            ui.show_error(e)
        else:
            ui.render_crop_result(result)

with st.expander("How accurate is this model?"):
    try:
        ui.render_crop_metrics(api_client.metrics("crop"))
    except api_client.ApiError as e:
        st.caption(str(e))

ui.footer()
