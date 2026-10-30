"""Results pages: data helpers and the two Streamlit pages, run on a generated results folder."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

import results_data as rd  # noqa: E402
import results_fixture  # noqa: E402

TIMEOUT = 60


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return results_fixture.build(tmp_path_factory.mktemp("results"))


@pytest.fixture
def results_env(results, monkeypatch):
    monkeypatch.setenv("KISANSENSE_RESULTS_DIR", str(results))
    return results


# ---- data helpers ----

def test_training_frame_joins_the_two_transfer_phases(results):
    h = rd.read_json(results / "disease_histories.json")
    f = rd.training_frame(h)
    mob = f[(f.model == "MobileNetV2, fine-tuned") & (f.metric == "loss") & (f.split == "Training")]
    assert list(mob["epoch"]) == list(range(1, 15))  # 8 head epochs then 6 fine-tune epochs, numbered continuously
    assert mob["phase"].nunique() == 2
    base = f[f.model == "Small CNN (baseline)"]
    assert base["epoch"].max() == 13


def test_backtest_test_rows_match_the_training_split(results):
    bt = rd.load_backtest(rd.read_csv(results / "price_backtest_rows.csv"))
    test = rd.test_rows(bt)
    assert bt["origin"].nunique() == 91
    assert test["origin"].nunique() == 37  # same 54 / 37 split as the real run


def test_mae_by_horizon_ignores_filled_in_days_and_grows_with_distance(results):
    bt = rd.load_backtest(rd.read_csv(results / "price_backtest_rows.csv"))
    h = rd.mae_by_horizon(rd.test_rows(bt))
    arima = h[h.model == "arima"].sort_values("h")
    assert list(arima["h"]) == list(range(1, 8))
    assert arima["mae"].iloc[-1] > arima["mae"].iloc[0]


def test_tables_use_real_numbers(results):
    d = rd.disease_table(rd.read_json(results / "disease_metrics.json"))
    assert round(d["Test accuracy"].max(), 3) == 0.935
    p = rd.price_table(rd.read_json(results / "price_metrics.json"))
    assert p.loc[p.key == "naive_last", "Test MAE vs naive"].iloc[0] == 1.0
    c = rd.crop_table(rd.read_json(results / "crop_metrics.json"))
    assert c.loc[c.key == "random_forest", "Model"].iloc[0].endswith("(selected)")


def test_strongest_correlation(results):
    eda = rd.read_json(results / "eda" / "crop_eda.json")
    assert rd.strongest_correlation(eda) == ("P", "K", 0.74)


def test_missing_files_return_none(tmp_path):
    assert rd.read_json(tmp_path / "nope.json") is None
    assert rd.read_csv(tmp_path / "nope.csv") is None


# ---- pages ----

def _run(page: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "app" / "pages" / page), default_timeout=TIMEOUT).run()


def test_eda_page_renders_all_three_tabs(results_env):
    at = _run("4_Data_and_EDA.py")
    assert not at.exception
    assert len(at.tabs) == 3
    labels = {m.label for m in at.metric}
    assert {"Images", "Rows", "Days of data"} <= labels
    assert len(at.get("arrow_vega_lite_chart")) + len(at.get("vega_lite_chart")) >= 6


def test_results_page_renders_all_three_tabs(results_env):
    at = _run("5_Models_and_Results.py")
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Test accuracy"] == "93.5%"
    assert values["Mistakes"] == "122 of 1,873"
    assert values["Test macro-F1"] in ("0.930", "0.993")  # disease and crop tabs both have this card


def test_pages_explain_when_results_are_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KISANSENSE_RESULTS_DIR", str(tmp_path))
    for page in ("4_Data_and_EDA.py", "5_Models_and_Results.py"):
        at = _run(page)
        assert not at.exception
        assert len(at.info) >= 3  # one friendly "not found" note per tab
