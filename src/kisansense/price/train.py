"""Price forecasting: backtest all models, select on validation origins, report
on later test origins, save the selected model.

Protocol:
- Rolling-origin backtest (refit at every origin using past data only).
- Earlier 60% of origins choose the model (lowest MAE); later 40% report it.
- Prediction ranges come from validation-period errors of the selected model
  (10th-90th percentile of actual/predicted); their coverage is checked on the
  test origins, so the range is not tuned on the data used to judge it.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kisansense.common.config import ARTIFACTS_DIR, DOCS_DIR, PRICE_CSV  # noqa: E402
from kisansense.price.backtest import (  # noqa: E402
    rolling_backtest,
    score,
    score_by_horizon,
    split_origins,
)
from kisansense.price.data import gap_report, load_price_data  # noqa: E402
from kisansense.price.features import HORIZON, prepare_daily  # noqa: E402
from kisansense.price.models import MODEL_FACTORIES  # noqa: E402

MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
RESULTS_DIR = DOCS_DIR / "results"
LOW_Q, HIGH_Q = 0.10, 0.90


def _ratio_quantiles(bt_model):
    d = bt_model[~bt_model["target_was_missing"]]
    r = d.assign(ratio=d["y_true"] / d["y_pred"]).groupby("h")["ratio"]
    return {int(h): (float(g.quantile(LOW_Q)), float(g.quantile(HIGH_Q))) for h, g in r}


def _coverage(bt_model, ratios):
    d = bt_model[~bt_model["target_was_missing"]]
    lo = d["h"].map(lambda h: ratios[int(h)][0]) * d["y_pred"]
    hi = d["h"].map(lambda h: ratios[int(h)][1]) * d["y_pred"]
    return float(((d["y_true"] >= lo) & (d["y_true"] <= hi)).mean())


def run(df=None, save: bool = True, initial_train: int = 365, step: int = 7) -> dict:
    df = load_price_data(PRICE_CSV) if df is None else df
    s = prepare_daily(df)
    bt = rolling_backtest(s, initial_train=initial_train, step=step)
    val, test = split_origins(bt)

    val_scores = score(val)
    test_scores = score(test)
    best = val_scores.index[0]  # chosen on validation origins only

    bt_best = bt[bt["model"] == best]
    val_best = val[val["model"] == best]
    test_best = test[test["model"] == best]

    ratios_val = _ratio_quantiles(val_best)
    coverage_test = _coverage(test_best, ratios_val)

    d7 = val_best[(val_best["h"] == HORIZON) & ~val_best["target_was_missing"]]
    trend_threshold = float(((d7["y_pred"] - d7["y_true"]).abs() / d7["y_true"]).median())

    naive_mae = float(test_scores.loc["naive_last", "mae"])
    metrics = {
        "series": {
            "start": str(s.index.min().date()),
            "end": str(s.index.max().date()),
            "days": int(len(s)),
            "interpolated_days": int(s["was_missing"].sum()),
            "gaps": gap_report(df),
        },
        "backtest": {
            "initial_train_days": initial_train,
            "step_days": step,
            "horizon_days": HORIZON,
            "origins": int(bt["origin"].nunique()),
            "validation_origins": int(val["origin"].nunique()),
            "test_origins": int(test["origin"].nunique()),
        },
        "selected_model": best,
        "validation": val_scores.round(4).to_dict(orient="index"),
        "test": test_scores.round(4).to_dict(orient="index"),
        "test_mae_vs_naive": {m: float(test_scores.loc[m, "mae"] / naive_mae) for m in test_scores.index},
        "interval": {
            "quantiles": [LOW_Q, HIGH_Q],
            "target_coverage": HIGH_Q - LOW_Q,
            "coverage_on_test": coverage_test,
        },
        "trend_threshold_pct": trend_threshold * 100,
        "note": "Model chosen on validation origins; test origins used for reporting only.",
    }

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "price_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        bt.to_csv(RESULTS_DIR / "price_backtest_rows.csv", index=False)

        by_h = score_by_horizon(test)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m in by_h.columns:
            ax.plot(by_h.index, by_h[m], marker="o", label=m, linewidth=2 if m == best else 1)
        ax.set_xlabel("Forecast horizon (days ahead)")
        ax.set_ylabel("MAE (Rs/quintal)")
        ax.set_title("Karnataka tomato: backtest error by horizon (test origins)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "price_mae_by_horizon.png", dpi=130)
        plt.close(fig)

        ratios_all = _ratio_quantiles(bt_best)
        model = MODEL_FACTORIES[best]().fit(s)
        joblib.dump(
            {
                "model": model,
                "model_name": best,
                "series": s,
                "interval_ratios": ratios_all,
                "trend_threshold": trend_threshold,
            },
            MODEL_PATH,
        )
    return metrics


def format_table(metrics: dict) -> str:
    lines = [
        "| Model | Validation MAE | Test MAE | Test RMSE | Test MAPE % | Test MAE / naive |",
        "|---|---|---|---|---|---|",
    ]
    for m, t in metrics["test"].items():
        v = metrics["validation"][m]
        mark = " (selected)" if m == metrics["selected_model"] else ""
        lines.append(
            f"| {m}{mark} | {v['mae']:.1f} | {t['mae']:.1f} | {t['rmse']:.1f} "
            f"| {t['mape_pct']:.1f} | {metrics['test_mae_vs_naive'][m]:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    metrics = run()
    print(format_table(metrics))
    b = metrics["backtest"]
    print(f"\nOrigins: {b['origins']} total ({b['validation_origins']} validation, {b['test_origins']} test)")
    iv = metrics["interval"]
    print(f"Prediction range target coverage {iv['target_coverage']:.0%}, measured on test {iv['coverage_on_test']:.0%}")
    print(f"Trend threshold (median 7-day error): {metrics['trend_threshold_pct']:.1f}%")
    print(f"Selected: {metrics['selected_model']}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
