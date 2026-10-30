"""Models and results: how each model was trained and tested, and what it got right and wrong."""
import numpy as np
import pandas as pd
import streamlit as st

import charts
import components as ui
import results_data as rd

st.set_page_config(page_title="Models and results · KisanSense", page_icon="📊", layout="wide")
ui.page_header("📊 Models and results", "How each model was trained and tested, and where it goes wrong")

PRICE_COLORS = {  # one fixed colour per model, so a model keeps its colour whichever are shown
    "arima": charts.BLUE,
    "naive_last": charts.ORANGE,
    "lag_gbm": "#1baf7a",
    "moving_average_7": "#eda100",
    "seasonal_naive_7": "#e87ba4",
}


def chart(c):
    st.altair_chart(charts.finish(c), theme=None)


def not_found(what: str, how: str) -> None:
    st.info(f"{what} not found. {how}")


def cards(items: list[tuple[str, str, str | None]]) -> None:
    for col, (label, value, help_) in zip(st.columns(len(items)), items):
        with col, st.container(border=True):
            st.metric(label, value, help=help_)


def numbers(df: pd.DataFrame, **formats) -> None:
    cfg = {c: st.column_config.NumberColumn(format=f) for c, f in formats.items()}
    st.dataframe(df, hide_index=True, column_config=cfg)


tab_d, tab_c, tab_p = st.tabs(["🍅 Leaf disease", "🌾 Crop advisor", "📈 Price forecast"])

# ---------------------------------------------------------------- disease
with tab_d:
    m = rd.read_json(rd.results_dir() / "disease_metrics.json")
    if not m:
        not_found("Disease results", "Train the model in the Colab notebook and copy docs/results into the project.")
    else:
        sel = m["selected_model"]
        t = m["results"][sel]["test"]
        wrong = round((1 - t["accuracy"]) * t["n"])
        st.markdown(f"**Selected model: {rd.DISEASE_NAMES.get(sel, sel)}**, chosen on validation macro-F1.")
        cards(
            [
                ("Test accuracy", f"{t['accuracy']:.1%}", f"On {t['n']:,} images the model never saw during training."),
                ("Test macro-F1", f"{t['macro_f1']:.3f}", "The average of the F1 score of each class, so small classes count as much as large ones."),
                ("Mistakes", f"{wrong} of {t['n']:,}", None),
                ("Conditions", str(len(m["data"]["classes"])), None),
            ]
        )

        st.subheader("Two models, compared fairly")
        numbers(rd.disease_table(m), **{c: "%.3f" for c in ("Validation accuracy", "Validation macro-F1", "Test accuracy", "Test macro-F1")})
        st.caption(m["note"])

        st.subheader("The training process")
        hist = rd.read_json(rd.results_dir() / "disease_histories.json")
        if not hist:
            not_found("Training history", "It is saved as disease_histories.json when the notebook trains the models.")
        else:
            frame = rd.training_frame(hist)
            choice = st.radio("Model", [name for name, _ in rd.TRAINING_PLAN][::-1], horizontal=True)
            f = frame[frame["model"] == choice]
            epochs = int(f["epoch"].max())
            if choice.startswith("MobileNet"):
                st.caption(
                    f"{epochs} epochs in two phases. Phase 1 trains only a new classification head on top of the frozen ImageNet "
                    "base (learning rate 0.001). Phase 2 unfreezes the top 30 layers and trains them gently (learning rate 0.00001). "
                    "The dotted line marks the switch. Early stopping keeps the best epoch."
                )
            else:
                st.caption(f"{epochs} epochs, trained from scratch with learning rate 0.001. Early stopping keeps the best epoch.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Accuracy**")
                chart(charts.learning_curves(f, "accuracy", y_title="Accuracy"))
            with c2:
                st.markdown("**Loss**")
                chart(charts.learning_curves(f, "loss", y_title="Loss"))
            with st.expander("Show the numbers"):
                wide = f.pivot_table(index=["epoch", "phase"], columns=["split", "metric"], values="value").reset_index()
                wide.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in wide.columns]
                st.dataframe(wide, hide_index=True)

        st.subheader("Where it is strong and weak")
        c1, c2 = st.columns([2, 3])
        with c1:
            f1 = rd.disease_class_f1(m)
            st.markdown("**F1 score per condition (test set)**")
            chart(charts.dot_plot(f1, "condition", "f1", x_title="F1 score", domain=(max(0.0, f1["f1"].min() - 0.05), 1.0),
                                  highlight_below=float(f1["f1"].iloc[1]) if len(f1) > 1 else None))
            weakest = f1.iloc[0]
            st.caption(f"Weakest: {weakest['condition']} ({weakest['f1']:.2f}).")
        with c2:
            cm = rd.find_image(f"disease_confusion_{sel}.png")
            if cm:
                st.image(str(cm), caption="Confusion matrix on the test set. Rows are the true condition, columns are the model's answer.")

        errs = rd.read_json(rd.eda_dir() / "disease_errors.json")
        if errs:
            st.markdown("**What it gets wrong most often**")
            tc = pd.DataFrame(errs["top_confusions"]).rename(columns={"true": "True condition", "predicted": "Model said", "count": "Times"})
            st.dataframe(tc, hide_index=True)
            st.caption(
                f"Average confidence was {errs['mean_confidence_when_right']:.0%} when the model was right and "
                f"{errs['mean_confidence_when_wrong']:.0%} when it was wrong."
            )
            gal = rd.find_image("disease_errors.png")
            if gal:
                st.image(str(gal), caption="The twelve mistakes the model was most confident about.")

        gc = rd.find_image("disease_gradcam_examples.png")
        if gc:
            st.markdown("**Where the model looks (Grad-CAM)**")
            st.image(str(gc), caption="Warmer colours mark the areas that pushed the model towards its answer. Mistakes come first.")

# ---------------------------------------------------------------- crop
with tab_c:
    m = rd.read_json(rd.results_dir() / "crop_metrics.json")
    if not m:
        not_found("Crop results", "Run `python scripts/train_crop.py`.")
    else:
        sel = m["selected_model"]
        tbl = rd.crop_table(m)
        row = tbl[tbl["key"] == sel].iloc[0]
        st.markdown(f"**Selected model: {rd.CROP_NAMES.get(sel, sel)}**, chosen on cross-validated macro-F1 using training data only.")
        cards(
            [
                ("Cross-validated macro-F1", f"{row['CV macro-F1']:.3f}", f"Average over 5 folds, plus or minus {row['CV std']:.3f}."),
                ("Test macro-F1", f"{row['Test macro-F1']:.3f}", f"On {m['n_test']} held-out rows, used once."),
                ("Right crop in top 3", f"{row['Test top-3 accuracy']:.1%}", "How often the true crop is among the three suggestions."),
            ]
        )

        st.subheader("Three models, compared with cross-validation")
        numbers(tbl.drop(columns=["key"]), **{c: "%.3f" for c in ("CV macro-F1", "CV std", "Test macro-F1", "Test accuracy", "Test top-3 accuracy")})
        lo = float((tbl["CV macro-F1"] - tbl["CV std"]).min()) - 0.01
        chart(charts.dot_range(tbl, "Model", "CV macro-F1", "CV std", x_title="Cross-validated macro-F1 (dot is the mean, line is ± one standard deviation across folds)",
                               domain=(round(lo, 2), 1.0), highlight=tbl[tbl["key"] == sel]["Model"].iloc[0]))
        st.caption(m["note"] + f" {m['n_train']} rows for training, {m['n_test']} for testing.")

        st.subheader("Where it is strong and weak")
        c1, c2 = st.columns([2, 3])
        with c1:
            f1 = rd.crop_class_f1(m)
            st.markdown("**F1 score per crop (test set)**")
            chart(charts.dot_plot(f1, "crop", "f1", x_title="F1 score", domain=(max(0.0, f1["f1"].min() - 0.03), 1.0),
                                  highlight_below=1.0, height=560))
            below = f1[f1["f1"] < 1.0]
            st.caption(f"{len(f1) - len(below)} of {len(f1)} crops were classified perfectly. Orange dots fell short: " + ", ".join(below["crop"]) + "." if len(below) else "Every crop was classified perfectly.")
        with c2:
            cm = rd.find_image("crop_confusion_matrix.png")
            if cm:
                st.image(str(cm), caption="Confusion matrix on the test set.")
            eda = rd.read_json(rd.eda_dir() / "crop_eda.json")
            if eda and eda.get("importance"):
                st.markdown("**Which inputs mattered most**")
                imp = pd.DataFrame({"input": list(eda["importance"]), "importance": list(eda["importance"].values())})
                chart(charts.hbar(imp, "input", "importance", x_title="Share of the model's decisions", fmt=".2f", height=230))

        st.warning(
            "A score near 1.0 mostly shows this dataset is easy to separate. It has no location or season, so it does not show "
            "the model can choose the right crop for a real farm."
        )

# ---------------------------------------------------------------- price
with tab_p:
    m = rd.read_json(rd.results_dir() / "price_metrics.json")
    raw = rd.read_csv(rd.results_dir() / "price_backtest_rows.csv")
    if not m:
        not_found("Price results", "Run `python scripts/train_price.py`.")
    else:
        sel = m["selected_model"]
        tbl = rd.price_table(m)
        row = tbl[tbl["key"] == sel].iloc[0]
        cov = m["interval"]
        st.markdown(f"**Selected model: {rd.PRICE_NAMES.get(sel, sel)}**, chosen on the earlier (validation) forecast dates.")
        cards(
            [
                ("Typical error", f"Rs {row['Test MAE (Rs)']:.0f}", "Mean absolute error per quintal on the later (test) forecast dates."),
                ("Error vs naive", f"{row['Test MAE vs naive'] - 1:+.0%}", "Change in typical error compared with repeating the last known price. Negative is better."),
                ("Error in percent", f"{row['Test MAPE (%)']:.1f}%", None),
                ("Range that held", f"{cov['coverage_on_test']:.0%}", f"The shaded band aimed for {cov['target_coverage']:.0%}."),
            ]
        )

        st.subheader("Five models, tested on dates they never saw")
        b = m["backtest"]
        st.caption(
            f"Rolling backtest: train on the past only, forecast {b['horizon_days']} days, move forward {b['step_days']} days, repeat. "
            f"{b['origins']} forecast dates in all: the first {b['validation_origins']} choose the model, the last {b['test_origins']} report the result."
        )
        numbers(
            tbl.drop(columns=["key"]),
            **{"Validation MAE (Rs)": "%.0f", "Test MAE (Rs)": "%.0f", "Test RMSE (Rs)": "%.0f", "Test MAPE (%)": "%.1f", "Test MAE vs naive": "%.2f"},
        )
        st.caption("MAE vs naive below 1.00 means fewer errors than repeating the last price. Only ARIMA and the boosting model beat it.")

        if raw is None:
            not_found("Backtest rows", "They are saved as price_backtest_rows.csv when you train the price model.")
        else:
            test = rd.test_rows(rd.load_backtest(raw))

            st.subheader("Error grows with the forecast distance")
            names = {k: rd.PRICE_NAMES.get(k, k) for k in test["model"].unique()}
            default = [k for k in (sel, "naive_last") if k in names]
            shown = st.multiselect("Models", list(names), default=default, format_func=lambda k: names[k])
            h = rd.mae_by_horizon(test[test["model"].isin(shown)])
            if len(h):
                chart(
                    charts.price_lines(h, x="h", y="mae", color="model_name", x_type="Q", x_title="Days ahead", y_title="Mean absolute error (Rs per quintal)",
                                       color_domain=[names[k] for k in shown], color_range=[PRICE_COLORS.get(k, charts.GRAY) for k in shown])
                )

            st.subheader("Forecast against what happened")
            c1, c2 = st.columns([3, 1])
            with c2:
                model_pick = st.selectbox("Model", list(names), index=list(names).index(sel) if sel in names else 0, format_func=lambda k: names[k])
                horizon = st.slider("Days ahead", 1, int(b["horizon_days"]), 1)
            fv = rd.forecast_vs_actual(test, model_pick, horizon)
            with c1:
                long = fv.rename(columns={"y_true": "Actual price", "y_pred": "Forecast"}).melt(
                    id_vars="target_date", value_vars=["Actual price", "Forecast"], var_name="series", value_name="price"
                )
                chart(
                    charts.price_lines(long, x="target_date", y="price", color="series", color_domain=["Actual price", "Forecast"],
                                       color_range=[charts.INK2, charts.BLUE], y_title="Rs per quintal")
                )
            st.caption(f"Each point is a forecast made {horizon} day{'s' if horizon > 1 else ''} earlier, over the {b['test_origins']} test forecast dates.")

            st.subheader("How wrong the forecasts are")
            err = rd.pct_errors(test, model_pick)
            counts, edges = np.histogram(err, bins=24)
            hist = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})
            c1, c2 = st.columns([3, 2])
            with c1:
                chart(charts.histogram(hist, x_title="Forecast error (% of the actual price; below 0 means too low)", fmt=".0f"))
            with c2:
                st.write(
                    f"With {names[model_pick]}, half of all forecasts in the test period were off by less than "
                    f"**{err.abs().median():.0f}%**, and 9 in 10 by less than **{err.abs().quantile(0.9):.0f}%**."
                )
                st.write(
                    "Prices swing a lot from week to week, so the app shows a band and says \"no clear trend\" when the "
                    f"expected move is below the typical error (about {m['trend_threshold_pct']:.0f}%)."
                )

        st.warning("Karnataka-wide prices, one commodity. Transport, quality and local demand are not in the model.")

ui.footer()
