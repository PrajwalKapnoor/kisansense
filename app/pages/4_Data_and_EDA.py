"""Dataset and EDA: what the data looks like and what that means for the models."""
import altair as alt
import pandas as pd
import streamlit as st

import charts
import components as ui
import results_data as rd

st.set_page_config(page_title="Dataset and EDA · KisanSense", page_icon="🔎", layout="wide")
ui.page_header("🔎 Dataset and exploratory analysis", "What each dataset contains, and what that means for the models")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def chart(c):
    st.altair_chart(charts.finish(c), theme=None)


def not_found(what: str, how: str) -> None:
    st.info(f"{what} not found. {how}")


def cards(items: list[tuple[str, str, str | None]]) -> None:
    for col, (label, value, help_) in zip(st.columns(len(items)), items):
        with col, st.container(border=True):
            st.metric(label, value, help=help_)


tab_d, tab_c, tab_p = st.tabs(["🍅 Leaf images", "🌾 Crop data", "📈 Price data"])

# ---------------------------------------------------------------- disease
with tab_d:
    m = rd.read_json(rd.results_dir() / "disease_metrics.json")
    eda = rd.read_json(rd.eda_dir() / "disease_eda.json")
    if not m:
        not_found("Disease results", "Train the model in the Colab notebook and copy docs/results into the project.")
    else:
        data = m["data"]
        n = data["images"]
        total = sum(n.values())
        pct = {k: round(100 * v / total) for k, v in n.items()}
        cards(
            [
                ("Images", f"{total:,}", "Six tomato classes from the PlantVillage colour images."),
                ("Classes", str(len(data["classes"])), "Five diseases and healthy leaves."),
                ("Train / val / test", f"{pct['train']}/{pct['val']}/{pct['test']}",
                 f"{n['train']:,} training, {n['val']:,} validation and {n['test']:,} test images."),
                ("Near-duplicate images", f"{data['images_in_duplicate_groups']}",
                 f"Found in {data['duplicate_groups']} groups of images that look the same."),
            ]
        )

        st.subheader("Class balance")
        bal = rd.class_balance(data["class_counts"])
        big, small = bal.iloc[0], bal.iloc[-1]
        left, right = st.columns([3, 2])
        with left:
            chart(charts.hbar(bal, "condition", "images", x_title="Images", highlight=small["condition"], fmt=",d"))
        with right:
            st.markdown(
                f"- **{big['condition']}** has the most images ({int(big['images']):,}); "
                f"**{small['condition']}** has the fewest ({int(small['images']):,}), "
                f"{big['images'] / small['images']:.1f}x fewer.\n"
                f"- A model that always answered \"{big['condition']}\" would be right "
                f"**{big['images'] / total:.0%}** of the time, so accuracy alone says little. "
                "The results page uses macro-F1, where every class counts equally.\n"
                "- Training used class weights so the smaller classes are not ignored."
            )

        st.subheader("What the images look like")
        grid = rd.find_image("disease_samples.png")
        if grid:
            st.image(str(grid), caption="Four random images per class.")
        else:
            not_found("Sample grid", "Run cell 4b of the Colab notebook and copy docs/results/eda.")

        if eda:
            c1, c2 = st.columns(2)
            with c1:
                sizes = pd.DataFrame(eda["size_counts"])
                st.markdown("**Image sizes** (random sample)")
                st.dataframe(sizes.rename(columns={"size": "Size (pixels)", "count": "Images"}), hide_index=True)
            with c2:
                bright = pd.DataFrame(
                    [{"condition": k, "brightness": v["mean"]} for k, v in eda["brightness_by_class"].items()]
                ).sort_values("brightness", ascending=False)
                st.markdown("**Average brightness per class** (0 is black, 1 is white)")
                chart(charts.hbar(bright, "condition", "brightness", x_title="Mean brightness", fmt=".2f", domain=(0, 1)))

        st.subheader("Guarding against leaked test images")
        st.write(
            f"{data['images_in_duplicate_groups']} images belong to {data['duplicate_groups']} groups of near-identical "
            "pictures (found with an image fingerprint). Each group was kept in a single split, so a picture the model "
            "trained on cannot reappear in the test set and inflate the score."
        )
        st.warning(
            "PlantVillage photos are taken in a lab, one leaf on a plain background. Field photos with soil, hands "
            "and shadows will be harder, so the scores here do not describe field performance."
        )

# ---------------------------------------------------------------- crop
with tab_c:
    eda = rd.read_json(rd.eda_dir() / "crop_eda.json")
    if not eda:
        not_found("Crop EDA", "Run `python scripts/make_eda.py` and commit docs/results/eda.")
    else:
        o = eda["overview"]
        cards(
            [
                ("Rows", f"{o['rows']:,}", None),
                ("Input features", str(len(o["features"])), "N, P, K, temperature, humidity, pH and rainfall."),
                ("Crops", str(o["classes"]), None),
                ("Missing values", str(o["missing_cells"]), None),
                ("Duplicate rows", str(o["duplicate_rows"]), None),
            ]
        )

        st.subheader("Class balance")
        if o["rows_per_class_min"] == o["rows_per_class_max"]:
            st.write(f"Every crop has exactly **{o['rows_per_class_min']}** rows, so there is no class imbalance to correct.")
        else:
            bal = pd.DataFrame({"crop": [rd.pretty_crop(k) for k in eda["class_counts"]], "rows": list(eda["class_counts"].values())})
            chart(charts.hbar(bal, "crop", "rows", x_title="Rows", fmt=",d"))

        st.subheader("How each input is distributed")
        feat = st.selectbox("Input", o["features"], format_func=lambda f: {"ph": "Soil pH", "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)"}.get(f, f.capitalize()))
        stats = eda["feature_stats"][feat]
        chart(charts.histogram(rd.histogram_frame(eda["histograms"][feat]), x_title=feat))
        st.caption(f"min {stats['min']:g} · mean {stats['mean']:g} · max {stats['max']:g} · standard deviation {stats['std']:g}")

        st.subheader("What tells the crops apart")
        med = rd.crop_medians_scaled(eda)
        order = sorted(eda["crop_medians"], key=str)
        chart(
            charts.heatmap(
                med, "feature", "crop", "scaled",
                tooltip=[alt.Tooltip("crop:N"), alt.Tooltip("feature:N"), alt.Tooltip("median:Q", format=".1f")],
                x_sort=o["features"], y_sort=[rd.pretty_crop(c) for c in order], legend_title="Relative level",
            )
        )
        st.caption("Each column is scaled on its own: dark means this crop's typical value is among the highest for that input. Hover for the real median.")

        st.subheader("How the inputs relate to each other")
        a, b, r = rd.strongest_correlation(eda)
        chart(
            charts.heatmap(
                rd.correlation_long(eda), "a", "b", "r",
                tooltip=[alt.Tooltip("a:N"), alt.Tooltip("b:N"), alt.Tooltip("r:Q", format=".2f")],
                diverging=True, height=320, x_sort=o["features"], y_sort=o["features"], legend_title="Correlation", text="r",
            )
        )
        st.caption(f"The strongest link between two inputs is {a} and {b} (correlation {r:+.2f}). Orange means they rise together, blue means one falls as the other rises.")

        if eda.get("importance"):
            st.subheader("Which inputs the model relied on")
            imp = pd.DataFrame({"input": list(eda["importance"]), "importance": list(eda["importance"].values())})
            chart(charts.hbar(imp, "input", "importance", x_title="Share of the model's decisions", fmt=".2f", height=230))

        st.warning(
            "This dataset has no location or season column, and its rows are very cleanly separated by crop. "
            "High scores here do not show it will pick the right crop for a particular farm."
        )

# ---------------------------------------------------------------- price
with tab_p:
    eda = rd.read_json(rd.eda_dir() / "price_eda.json")
    daily = rd.read_csv(rd.eda_dir() / "price_daily.csv")
    pm = rd.read_json(rd.results_dir() / "price_metrics.json")
    if not eda or daily is None:
        not_found("Price EDA", "Run `python scripts/make_eda.py` and commit docs/results/eda.")
    else:
        o = eda["overview"]
        cards(
            [
                ("Days of data", f"{o['days']:,}", f"{o['start']} to {o['end']}"),
                ("Latest price", f"Rs {o['latest_price']:,.0f}", f"Rs per quintal on {o['end']}."),
                ("Highest price", f"Rs {o['max_price']:,.0f}", f"Reached on {o['max_price_date']}."),
                ("Typical price", f"Rs {o['median_price']:,.0f}", "The median over the whole period."),
                ("Days filled in", str(o["missing_days"]), f"Missing days were interpolated; the longest gap was {o['longest_gap_days']} days."),
            ]
        )
        st.caption(f"Karnataka-wide daily modal price for tomato, {o['start']} to {o['end']}. Prices are in {o['price_unit']}.")

        st.subheader("Price history")
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"])
        d["30-day average"] = d["modal_price"].rolling(30, min_periods=10).mean()
        long = d.rename(columns={"modal_price": "Daily price"}).melt(
            id_vars="date", value_vars=["Daily price", "30-day average"], var_name="series", value_name="price"
        ).dropna()
        chart(
            charts.price_lines(long, x="date", y="price", color="series", color_domain=["Daily price", "30-day average"],
                               color_range=[charts.GRAY, charts.BLUE], y_title="Rs per quintal", height=300)
        )

        st.subheader("Seasonality and weekly pattern")
        c1, c2 = st.columns(2)
        with c1:
            mm = pd.DataFrame({"month": [MONTHS[int(k) - 1] for k in eda["monthly_mean"]], "price": list(eda["monthly_mean"].values())})
            top = mm.loc[mm["price"].idxmax(), "month"]
            st.markdown("**Average price by calendar month**")
            chart(charts.vbar(mm, "month", "price", y_title="Rs per quintal", highlight=top, sort=MONTHS))
            st.caption(f"Highest in {top}. With about {o['days'] / 365:.1f} years of data each month is seen only 2 to 3 times, so read this as a hint.")
        with c2:
            wd = pd.DataFrame({"day": [DAYS[int(k)] for k in eda["weekday_ratio"]], "pct": [(v - 1) * 100 for v in eda["weekday_ratio"].values()]})
            st.markdown("**Price by weekday, against that week's average**")
            chart(charts.vbar(wd, "day", "pct", y_title="% above or below", fmt="+.2f", sort=DAYS))
            st.caption(f"Differences are small (from {wd['pct'].min():+.1f}% to {wd['pct'].max():+.1f}%), so the weekday is not a strong signal.")

        st.subheader("How much prices move")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Daily change (log return)**")
            chart(charts.histogram(rd.histogram_frame(eda["daily_log_return_hist"]), x_title="Daily change in log price", fmt=".2f"))
        with c4:
            q = eda["change_7d_pct_quantiles"]
            st.markdown("**Change over 7 days**")
            st.write(
                f"In 8 of 10 weeks the price moved between **{q['0.1']:+.0f}%** and **{q['0.9']:+.0f}%** over the next 7 days "
                f"(the middle half: {q['0.25']:+.0f}% to {q['0.75']:+.0f}%)."
            )
            if pm and pm.get("trend_threshold_pct"):
                st.write(
                    f"The forecast model's typical 7-day error is about **{pm['trend_threshold_pct']:.0f}%**, so the app only "
                    "calls a rise or fall when the expected move is larger than that."
                )

        st.subheader("Arrivals and price")
        ar = pd.DataFrame(eda["median_price_by_arrival_quintile"])
        ar["arrivals"] = ar["arrival_range"]
        chart(charts.vbar(ar, "arrivals", "median_price", y_title="Median price (Rs per quintal)", sort=list(ar["arrivals"])))
        st.caption(f"Markets are grouped into five equal sets by how much tomato arrived that day (tonnes). Correlation of log arrivals with log price: {eda['arrival_price_corr']:+.2f}.")

        st.warning("Prices here are Karnataka-wide daily modal prices, not one market, and the data ends on " + o["end"] + ".")

ui.footer()
