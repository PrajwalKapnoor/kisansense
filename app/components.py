"""Shared UI pieces and result renderers."""
from __future__ import annotations

import base64
import io

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

import api_client

# Mirrors the API's validation ranges so the form can guide the user.
# The API remains the authority and rejects anything outside them.
CROP_FIELDS = [
    # key, label, min, max, default, step, help
    ("N", "Nitrogen (N)", 0.0, 140.0, 90.0, 1.0, "Nitrogen content in the soil"),
    ("P", "Phosphorus (P)", 5.0, 145.0, 42.0, 1.0, "Phosphorus content in the soil"),
    ("K", "Potassium (K)", 5.0, 205.0, 43.0, 1.0, "Potassium content in the soil"),
    ("temperature", "Temperature (°C)", 0.0, 50.0, 25.0, 0.5, "Average temperature"),
    ("humidity", "Humidity (%)", 10.0, 100.0, 80.0, 1.0, "Relative humidity"),
    ("ph", "Soil pH", 3.0, 10.0, 6.5, 0.1, "Soil acidity: 7 is neutral"),
    ("rainfall", "Rainfall (mm)", 20.0, 300.0, 200.0, 1.0, "Rainfall"),
]

DISCLAIMER = (
    "KisanSense is an experimental decision-support tool built as a data science project. "
    "It is not agricultural or financial advice. Confirm important decisions with local experts."
)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)


def show_error(e: api_client.ApiError) -> None:
    if e.status == 503:
        st.warning(f"This feature is not ready yet. {e}")
    else:
        st.error(str(e))


def pretty_class(name: str) -> str:
    return name.replace("Tomato___", "").replace("_", " ").strip().capitalize()


# ---------------------------------------------------------------- disease
def render_disease_result(result: dict, original: Image.Image | None = None) -> None:
    conf = result["confidence"]
    if result["uncertain"]:
        st.warning(
            f"The model is not confident enough to name a disease (top score {conf:.0%}). "
            "Try a clearer, closer photo of a single leaf in good light."
        )
    else:
        st.success(f"**{result['name']}**  ·  model confidence {conf:.0%}")
        if result.get("description"):
            st.write(result["description"])
    st.progress(min(max(conf, 0.0), 1.0), text=f"Top score: {conf:.0%}")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Your photo**")
        if original is not None:
            st.image(original, width=320)
    with cols[1]:
        st.markdown("**Where the model looked**")
        if result.get("gradcam_png_base64"):
            st.image(Image.open(io.BytesIO(base64.b64decode(result["gradcam_png_base64"]))), width=320)
            st.caption("Warmer colours mark areas that influenced the result most. This explains the model, not the plant.")
        else:
            st.caption("No heatmap is available for this model.")

    table = pd.DataFrame(
        [{"Possible condition": t["name"], "Score": f"{t['probability']:.0%}"} for t in result["top"]]
    )
    st.dataframe(table, hide_index=True)
    st.caption(result["reminder"])


def render_disease_metrics(m: dict | None) -> None:
    if not m:
        st.caption("No saved evaluation results yet.")
        return
    rows = [
        {
            "Model": name + (" (selected)" if name == m["selected_model"] else ""),
            "Test accuracy": f"{r['test']['accuracy']:.1%}",
            "Test macro-F1": f"{r['test']['macro_f1']:.3f}",
        }
        for name, r in m["results"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    n = m["data"]["images"]
    st.caption(
        f"Evaluated on {n['test']:,} held-out images. {m['note']}"
    )


# ------------------------------------------------------------------- crop
def render_crop_result(result: dict) -> None:
    recs = result["recommendations"]
    st.subheader("Suggested crops")
    for i, r in enumerate(recs, start=1):
        st.markdown(f"**{i}. {r['crop'].capitalize()}**")
        st.progress(min(max(r["probability"], 0.0), 1.0), text=f"{r['probability']:.0%}")
    st.caption(result["note"])


def render_crop_metrics(m: dict | None) -> None:
    if not m:
        st.caption("No saved evaluation results yet.")
        return
    rows = [
        {
            "Model": name + (" (selected)" if name == m["selected_model"] else ""),
            "CV macro-F1": f"{m['cv'][name]['cv_f1_macro_mean']:.3f}",
            "Test macro-F1": f"{m['test'][name]['test_f1_macro']:.3f}",
            "Test top-3 accuracy": f"{m['test'][name]['test_top3_accuracy']:.3f}",
        }
        for name in m["cv"]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "These scores are very high because the public dataset is clean and the crops are well separated. "
        "It has no region information, so this does not show the model works on real farms."
    )


# ------------------------------------------------------------------ price
def price_chart(data: dict) -> alt.Chart:
    hist = pd.DataFrame({"date": pd.to_datetime(data["history"]["dates"]), "price": data["history"]["prices"]})
    last_date = pd.to_datetime(data["last_date"])
    fc = pd.DataFrame(
        {
            "date": [last_date] + list(pd.to_datetime(data["dates"])),
            "price": [data["last_price"]] + data["forecast"],
            "low": [data["last_price"]] + data["low"],
            "high": [data["last_price"]] + data["high"],
        }
    )
    hist["series"] = "Observed"
    fc["series"] = "Forecast"
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=["Observed", "Forecast"], range=["#475569", "#d97706"]),
        legend=alt.Legend(title=None, orient="top"),
    )
    x = alt.X("date:T", title=None)
    y = alt.Y("price:Q", title="Rs per quintal", scale=alt.Scale(zero=False))
    band = alt.Chart(fc).mark_area(opacity=0.18, color="#d97706").encode(x=x, y="low:Q", y2="high:Q")
    obs = alt.Chart(hist).mark_line(strokeWidth=2).encode(
        x=x, y=y, color=color, tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", format=",.0f")]
    )
    pred = alt.Chart(fc).mark_line(strokeWidth=2, strokeDash=[6, 4]).encode(
        x=x, y=y, color=color, tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", title="forecast", format=",.0f")]
    )
    return (band + obs + pred).properties(height=320, width="container")


def render_price(data: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Latest price ({data['last_date']})", f"Rs {data['last_price']:,.0f}")
    c2.metric("Expected 7-day change", f"{data['expected_change_pct']:+.1f}%")
    c3.metric("Trend signal", data["trend"].capitalize())

    if data["trend"] == "no clear trend":
        st.info(
            f"Any expected move is smaller than this model's typical 7-day error (about "
            f"{data['trend_threshold_pct']:.0f}%), so it cannot be told apart from normal price noise."
        )
    st.altair_chart(price_chart(data))
    st.caption("Shaded band: the range that held for about 80% of past forecasts at each horizon.")

    table = pd.DataFrame(
        {
            "Date": data["dates"],
            "Forecast (Rs/quintal)": [f"{v:,.0f}" for v in data["forecast"]],
            "Likely range": [f"{lo:,.0f} to {hi:,.0f}" for lo, hi in zip(data["low"], data["high"])],
        }
    )
    st.dataframe(table, hide_index=True)
    st.caption(data["disclaimer"])


def render_price_metrics(m: dict | None) -> None:
    if not m:
        st.caption("No saved backtest results yet.")
        return
    rows = []
    for name, t in m["test"].items():
        rows.append(
            {
                "Model": name + (" (selected)" if name == m["selected_model"] else ""),
                "MAE (Rs)": f"{t['mae']:.0f}",
                "RMSE (Rs)": f"{t['rmse']:.0f}",
                "MAPE": f"{t['mape_pct']:.1f}%",
                "Error vs naive": f"{m['test_mae_vs_naive'][name]:.2f}x",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    b, iv = m["backtest"], m["interval"]
    st.caption(
        f"Rolling backtest on {b['test_origins']} held-out forecast dates, 1 to {b['horizon_days']} days ahead. "
        f"The shaded range aimed for {iv['target_coverage']:.0%} coverage and achieved {iv['coverage_on_test']:.0%} on test data. "
        "Prices are Karnataka-wide daily modal prices, not a single market."
    )
