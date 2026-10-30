"""Chart builders for the results pages (Altair). One look for all charts.

Colour roles: BLUE is the main series, ORANGE marks the one thing to look at (the selected model,
the weakest class, the highest month) or a second series, GRAY is context. Charts are drawn with
theme=None so these colours are used as written.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#a9a8a0"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e7e6e1"
SURFACE = "#ffffff"
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = [BLUE, "#ececec", ORANGE]


def finish(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Apply the shared axis, legend and view styling to a finished chart."""
    return (
        chart.configure_view(stroke=None, fill=SURFACE)
        .configure_axis(
            gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=INK2, titleColor=INK2,
            labelFontSize=12, titleFontSize=12, titleFontWeight="normal",
        )
        .configure_legend(labelColor=INK2, titleColor=INK2, labelFontSize=12, orient="top", title=None, symbolType="stroke")
        .configure_title(color=INK, fontSize=14, anchor="start", fontWeight="bold")
    )


def hbar(df: pd.DataFrame, category: str, value: str, *, x_title: str, highlight: str | None = None,
         fmt: str = ",.0f", height: int | None = None, labels: bool = True, domain: tuple[float, float] | None = None,
         sort: list[str] | None = None) -> alt.TopLevelMixin:
    """Horizontal bars from zero, rounded at the data end, optional single highlighted category."""
    order = sort if sort is not None else alt.EncodingSortField(field=value, order="descending")
    color = (
        alt.condition(alt.datum[category] == highlight, alt.value(ORANGE), alt.value(BLUE))
        if highlight is not None
        else alt.value(BLUE)
    )
    x_scale = alt.Scale(domain=list(domain)) if domain else alt.Scale(zero=True)
    base = alt.Chart(df).encode(
        y=alt.Y(f"{category}:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X(f"{value}:Q", title=x_title, scale=x_scale),
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    bars = base.mark_bar(size=16, cornerRadiusEnd=4, color=BLUE).encode(color=color)
    layers = [bars]
    if labels:
        layers.append(base.mark_text(align="left", dx=5, color=INK2, fontSize=12).encode(text=alt.Text(f"{value}:Q", format=fmt)))
    chart = alt.layer(*layers)
    return chart.properties(height=height or max(120, 30 * len(df)), width="container")


def vbar(df: pd.DataFrame, category: str, value: str, *, y_title: str, highlight: str | None = None,
         fmt: str = ",.0f", height: int = 260, sort: list[str] | None = None,
         domain: tuple[float, float] | None = None) -> alt.TopLevelMixin:
    color = (
        alt.condition(alt.datum[category] == highlight, alt.value(ORANGE), alt.value(BLUE))
        if highlight is not None
        else alt.value(BLUE)
    )
    y_scale = alt.Scale(domain=list(domain)) if domain else alt.Scale(zero=True)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color=BLUE)
        .encode(
            x=alt.X(f"{category}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=0), scale=alt.Scale(paddingInner=0.35, paddingOuter=0.1)),
            y=alt.Y(f"{value}:Q", title=y_title, scale=y_scale),
            color=color,
            tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=fmt)],
        )
    )
    return chart.properties(height=height, width="container")


def histogram(df: pd.DataFrame, *, x_title: str, y_title: str = "Count", height: int = 240,
              fmt: str = ",.2f") -> alt.TopLevelMixin:
    """Bars from a precomputed histogram frame with left, right and count columns; a 2px gap separates bins."""
    d = df.copy()
    d["_zero"] = 0
    chart = (
        alt.Chart(d)
        .mark_rect(color=BLUE, stroke=SURFACE, strokeWidth=2, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("left:Q", title=x_title, scale=alt.Scale(zero=False)),
            x2="right:Q",
            y=alt.Y("count:Q", title=y_title),
            y2="_zero:Q",
            tooltip=[alt.Tooltip("left:Q", title="from", format=fmt), alt.Tooltip("right:Q", title="to", format=fmt),
                     alt.Tooltip("count:Q", title="count")],
        )
    )
    return chart.properties(height=height, width="container")


def heatmap(df: pd.DataFrame, x: str, y: str, color: str, *, tooltip: list[alt.Tooltip], diverging: bool = False,
            height: int | None = None, x_sort: list[str] | None = None, y_sort: list[str] | None = None,
            legend_title: str | None = None, text: str | None = None) -> alt.TopLevelMixin:
    scale = (
        alt.Scale(domain=[-1, 0, 1], range=DIVERGING, interpolate="rgb")
        if diverging
        else alt.Scale(domain=[0, 0.25, 0.5, 0.75, 1], range=SEQUENTIAL, interpolate="rgb")
    )
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:N", sort=x_sort, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:N", sort=y_sort, title=None),
    )
    rect = base.mark_rect(stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color(f"{color}:Q", scale=scale, legend=alt.Legend(title=legend_title, orient="right", symbolType="square")),
        tooltip=tooltip,
    )
    layers = [rect]
    if text:
        layers.append(
            base.mark_text(fontSize=11).encode(
                text=alt.Text(f"{text}:Q", format=".2f"),
                color=alt.condition(alt.datum[color] > 0.55 if not diverging else (alt.datum[color] > 0.6) | (alt.datum[color] < -0.6),
                                    alt.value("#ffffff"), alt.value(INK)),
            )
        )
    return alt.layer(*layers).properties(height=height or max(160, 26 * df[y].nunique()), width="container")


def learning_curves(df: pd.DataFrame, metric: str, *, y_title: str, height: int = 260) -> alt.TopLevelMixin:
    """Training vs validation over epochs for one metric; a dotted rule marks where fine-tuning starts."""
    d = df[df["metric"] == metric]
    color = alt.Color(
        "split:N",
        scale=alt.Scale(domain=["Training", "Validation"], range=[BLUE, ORANGE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    x = alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1, format="d"))
    y = alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False))
    tip = [alt.Tooltip("epoch:Q"), alt.Tooltip("split:N"), alt.Tooltip("phase:N"), alt.Tooltip("value:Q", format=".3f")]
    line = alt.Chart(d).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45, filled=True)).encode(x=x, y=y, color=color, tooltip=tip)
    layers = [line]
    phases = d.drop_duplicates("phase").sort_values("epoch")
    if len(phases) > 1:
        starts = pd.DataFrame({"epoch": phases["epoch"].iloc[1:].astype(float) - 0.5, "label": phases["phase"].iloc[1:]})
        layers.append(alt.Chart(starts).mark_rule(strokeDash=[4, 4], color=GRAY).encode(x="epoch:Q", tooltip=["label:N"]))
    return alt.layer(*layers).properties(height=height, width="container")


def price_lines(df: pd.DataFrame, *, x: str, y: str, color: str | None, color_domain: list[str] | None = None,
                color_range: list[str] | None = None, y_title: str, height: int = 280, dash: dict | None = None,
                x_type: str = "T", x_title: str | None = None) -> alt.TopLevelMixin:
    enc = dict(
        x=alt.X(f"{x}:{x_type}", title=x_title, axis=alt.Axis(tickMinStep=1) if x_type == "Q" else alt.Axis()),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip(f"{x}:{x_type}"), alt.Tooltip(f"{y}:Q", format=",.1f")]
        + ([alt.Tooltip(f"{color}:N")] if color else []),
    )
    if color:
        enc["color"] = alt.Color(
            f"{color}:N",
            scale=alt.Scale(domain=color_domain, range=color_range) if color_domain else alt.Undefined,
            legend=alt.Legend(title=None, orient="top"),
        )
    return alt.Chart(df).mark_line(strokeWidth=2).encode(**enc).properties(height=height, width="container")


def dot_range(df: pd.DataFrame, category: str, value: str, err: str, *, x_title: str, domain: tuple[float, float],
              highlight: str | None = None) -> alt.TopLevelMixin:
    """A dot for each mean with a +/- rule. The axis is not from zero, so it is a dot plot, not bars."""
    d = df.copy()
    d["lo"] = d[value] - d[err]
    d["hi"] = d[value] + d[err]
    color = (
        alt.condition(alt.datum[category] == highlight, alt.value(ORANGE), alt.value(BLUE))
        if highlight is not None
        else alt.value(BLUE)
    )
    y = alt.Y(f"{category}:N", sort=None, title=None, axis=alt.Axis(labelLimit=240))
    scale = alt.Scale(domain=list(domain))
    rule = alt.Chart(d).mark_rule(strokeWidth=2, color=GRAY).encode(
        y=y, x=alt.X("lo:Q", scale=scale, title=x_title), x2="hi:Q"
    )
    dot = alt.Chart(d).mark_point(filled=True, size=130, stroke=SURFACE, strokeWidth=2).encode(
        y=y, x=alt.X(f"{value}:Q", scale=scale), color=color,
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=".4f"), alt.Tooltip(f"{err}:Q", title="± std", format=".4f")],
    )
    return alt.layer(rule, dot).properties(height=max(110, 46 * len(d)), width="container")


def dot_plot(df: pd.DataFrame, category: str, value: str, *, x_title: str, domain: tuple[float, float],
             highlight_below: float | None = None, fmt: str = ".3f", height: int | None = None) -> alt.TopLevelMixin:
    """One dot per category on a non-zero axis (used for per-class scores that are all close to 1)."""
    color = (
        alt.condition(alt.datum[value] < highlight_below, alt.value(ORANGE), alt.value(BLUE))
        if highlight_below is not None
        else alt.value(BLUE)
    )
    d = df.copy()
    d["_start"] = domain[0]  # the stem runs from the left edge of the axis to the dot
    y = alt.Y(f"{category}:N", sort=None, title=None)
    x = alt.X(f"{value}:Q", scale=alt.Scale(domain=list(domain)), title=x_title)
    stem = alt.Chart(d).mark_rule(color=GRID, strokeWidth=2).encode(y=y, x=alt.X("_start:Q", scale=alt.Scale(domain=list(domain))), x2=f"{value}:Q")
    dot = alt.Chart(d).mark_point(filled=True, size=110, stroke=SURFACE, strokeWidth=2).encode(
        y=y, x=x, color=color, tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=fmt)]
    )
    return alt.layer(stem, dot).properties(height=height or max(140, 24 * len(df)), width="container")
