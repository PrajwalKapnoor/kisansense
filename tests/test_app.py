"""Frontend tests: run the Streamlit pages with the API client replaced by fakes."""
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import api_client  # noqa: E402

TIMEOUT = 30

PRICE = {
    "model": "arima", "last_date": "2026-10-01", "last_price": 1245.0,
    "dates": [f"2026-10-0{d}" for d in range(2, 9)],
    "forecast": [1250.0] * 7, "low": [1150.0] * 7, "high": [1350.0] * 7,
    "trend": "no clear trend", "expected_change_pct": 0.4, "trend_threshold_pct": 10.1,
    "history": {"dates": [f"2026-09-{d:02d}" for d in range(1, 31)], "prices": [1200.0 + d for d in range(30)]},
    "unit": "Rs/quintal", "disclaimer": "Experimental trend information. Not selling advice.",
}
PRICE_METRICS = {
    "selected_model": "arima",
    "test": {"arima": {"mae": 188.5, "rmse": 263.5, "mape_pct": 16.1}, "naive_last": {"mae": 204.4, "rmse": 297.8, "mape_pct": 17.7}},
    "test_mae_vs_naive": {"arima": 0.92, "naive_last": 1.0},
    "backtest": {"test_origins": 37, "horizon_days": 7},
    "interval": {"target_coverage": 0.8, "coverage_on_test": 0.75},
}
CROP = {"recommendations": [{"crop": "rice", "probability": 0.8}, {"crop": "jute", "probability": 0.1}, {"crop": "maize", "probability": 0.05}],
        "note": "Scores are model probabilities."}


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_data.clear()
    yield


def _page(name):
    return AppTest.from_file(str(APP / name), default_timeout=TIMEOUT)


def test_home_shows_status(monkeypatch):
    monkeypatch.setattr(api_client, "health", lambda: {"status": "ok", "models": {"crop": True, "price": True, "disease": False}})
    at = _page("Home.py").run()
    assert not at.exception
    assert at.title[0].value == "KisanSense"
    text = " ".join(m.value for m in at.markdown)
    assert "Crop recommendation: ready" in text
    assert "Disease detection: model not trained yet" in text


def test_home_survives_api_down(monkeypatch):
    def down():
        raise api_client.ApiError("Cannot reach the KisanSense API")

    monkeypatch.setattr(api_client, "health", down)
    at = _page("Home.py").run()
    assert not at.exception
    assert any("Cannot reach" in e.value for e in at.error)


def test_crop_page_submits_and_shows_top3(monkeypatch):
    captured = {}

    def fake(payload):
        captured.update(payload)
        return CROP

    monkeypatch.setattr(api_client, "recommend_crop", fake)
    monkeypatch.setattr(api_client, "metrics", lambda m: None)
    at = _page("pages/2_Crop_Advisor.py").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert set(captured) == {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"}
    assert any("Rice" in m.value for m in at.markdown)
    assert any(s.value == "Suggested crops" for s in at.subheader)


def test_crop_page_shows_not_ready_warning(monkeypatch):
    def not_ready(payload):
        raise api_client.ApiError("The crop recommendation model is not available.", 503)

    monkeypatch.setattr(api_client, "recommend_crop", not_ready)
    monkeypatch.setattr(api_client, "metrics", lambda m: None)
    at = _page("pages/2_Crop_Advisor.py").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("not ready" in w.value for w in at.warning)


def test_crop_page_shows_validation_error(monkeypatch):
    def bad(payload):
        raise api_client.ApiError("ph: Input should be less than or equal to 10", 422)

    monkeypatch.setattr(api_client, "recommend_crop", bad)
    monkeypatch.setattr(api_client, "metrics", lambda m: None)
    at = _page("pages/2_Crop_Advisor.py").run()
    at.button[0].click().run()
    assert any("ph:" in e.value for e in at.error)


def test_price_page_renders_metrics_and_chart(monkeypatch):
    monkeypatch.setattr(api_client, "price_forecast", lambda days: PRICE)
    monkeypatch.setattr(api_client, "metrics", lambda m: PRICE_METRICS)
    at = _page("pages/3_Price_Explorer.py").run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Latest price (2026-10-01)"] == "Rs 1,245"
    assert labels["Trend signal"] == "No clear trend"
    assert any("typical 7-day error" in i.value for i in at.info)
    assert len(at.get("vega_lite_chart")) == 1


def test_price_page_warns_when_data_is_old(monkeypatch):
    monkeypatch.setattr(api_client, "price_forecast", lambda days: {**PRICE, "last_date": "2020-01-01"})
    monkeypatch.setattr(api_client, "metrics", lambda m: None)
    at = _page("pages/3_Price_Explorer.py").run()
    assert not at.exception
    assert any("ends on 2020-01-01" in w.value for w in at.warning)


def test_price_page_no_warning_when_data_is_fresh(monkeypatch):
    from datetime import date

    monkeypatch.setattr(api_client, "price_forecast", lambda days: {**PRICE, "last_date": date.today().isoformat()})
    monkeypatch.setattr(api_client, "metrics", lambda m: None)
    at = _page("pages/3_Price_Explorer.py").run()
    assert not at.exception
    assert not any("ends on" in w.value for w in at.warning)


def test_price_page_handles_missing_model(monkeypatch):
    def nope(days):
        raise api_client.ApiError("The price forecasting model is not available.", 503)

    monkeypatch.setattr(api_client, "price_forecast", nope)
    monkeypatch.setattr(api_client, "metrics", lambda m: None)
    at = _page("pages/3_Price_Explorer.py").run()
    assert not at.exception
    assert any("not ready" in w.value for w in at.warning)


def test_disease_page_with_no_upload_is_clean(monkeypatch):
    monkeypatch.setattr(api_client, "metrics", lambda m: None)
    at = _page("pages/1_Disease_Check.py").run()
    assert not at.exception
    assert "Tomato leaf disease check" in at.title[0].value


def _render_uncertain():
    import components as ui

    ui.render_disease_result(
        {"prediction": None, "name": "Uncertain", "description": "", "confidence": 0.31, "uncertain": True,
         "top": [{"class": "Tomato___healthy", "name": "Healthy", "probability": 0.31}],
         "reminder": "Confirm with a local agricultural extension officer.", "gradcam_png_base64": None}
    )


def _render_confident():
    import components as ui

    ui.render_disease_result(
        {"prediction": "Tomato___Late_blight", "name": "Late blight", "description": "Large dark patches.",
         "confidence": 0.93, "uncertain": False,
         "top": [{"class": "Tomato___Late_blight", "name": "Late blight", "probability": 0.93}],
         "reminder": "Confirm with a local agricultural extension officer.", "gradcam_png_base64": None}
    )


def test_disease_result_uncertain_does_not_name_a_disease():
    at = AppTest.from_function(_render_uncertain, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("not confident enough" in w.value for w in at.warning)
    assert not at.success


def test_disease_result_confident_names_the_disease():
    at = AppTest.from_function(_render_confident, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("Late blight" in s.value and "93%" in s.value for s in at.success)
