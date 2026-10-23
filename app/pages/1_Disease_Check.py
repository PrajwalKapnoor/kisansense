import io

import streamlit as st
from PIL import Image

import api_client
import components as ui

st.set_page_config(page_title="Disease check · KisanSense", page_icon="🍅", layout="centered")
ui.page_header("🍅 Tomato leaf disease check", "Upload one clear photo of a single tomato leaf")

st.write(
    "Works for six conditions: bacterial spot, early blight, late blight, leaf mold, Septoria leaf spot, or healthy. "
    "Best results: one leaf, good light, leaf filling most of the frame."
)

upload = st.file_uploader("Leaf photo (JPG or PNG, up to 5 MB)", type=["jpg", "jpeg", "png"])

if upload is not None:
    data = upload.getvalue()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        st.error("That file could not be read as an image. Please upload a JPG or PNG photo.")
        st.stop()

    if st.button("Check this leaf", type="primary"):
        with st.spinner("Analysing the leaf. The first request can take up to a minute..."):
            try:
                result = api_client.predict_disease(data, upload.name, upload.type or "image/jpeg")
            except api_client.ApiError as e:
                ui.show_error(e)
            else:
                ui.render_disease_result(result, image.convert("RGB"))
    else:
        st.image(image.convert("RGB"), width=320, caption="Preview")

with st.expander("How accurate is this model?"):
    try:
        ui.render_disease_metrics(api_client.metrics("disease"))
    except api_client.ApiError as e:
        st.caption(str(e))
    st.caption(
        "The model was trained on PlantVillage, a dataset of leaf photos taken on plain backgrounds in a lab. "
        "Photos taken in a field, with soil, hands or other leaves in view, may give worse results."
    )

ui.footer()
