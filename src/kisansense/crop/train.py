"""Crop recommendation: compare models, evaluate honestly, save the best.

Protocol (no leakage):
1. Hold out a stratified test set that is never used for model selection.
2. Compare models with stratified 5-fold CV on the training part only.
3. Pick the model with the best CV macro-F1.
4. Report that model's score on the untouched test set.
5. Refit the chosen model on ALL data for the saved artifact.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from kisansense.common.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CROP_CSV,
    CROP_FEATURES,
    CROP_TARGET,
    DOCS_DIR,
    RANDOM_SEED,
)

MODEL_PATH = ARTIFACTS_DIR / "crop_model.joblib"
RESULTS_DIR = DOCS_DIR / "results"


def load_crop_data(path=CROP_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in CROP_FEATURES + [CROP_TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Crop CSV is missing columns: {missing}")
    return df


def build_models() -> dict[str, Pipeline]:
    """Candidate models. Scaling only matters for logistic regression, but a
    shared pipeline keeps preprocessing fitted on training data only."""
    return {
        "logistic_regression": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))]
        ),
        "random_forest": Pipeline(
            [("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1))]
        ),
        "hist_gradient_boosting": Pipeline(
            [("clf", HistGradientBoostingClassifier(random_state=RANDOM_SEED))]
        ),
    }


def fit_and_evaluate(df: pd.DataFrame, save: bool = True) -> dict:
    X = df[CROP_FEATURES]
    y = df[CROP_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = build_models()
    cv_results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1
        )
        cv_results[name] = {
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "cv_f1_macro_std": float(scores["test_f1_macro"].std()),
        }

    # Select on CV only, never on the test set.
    best_name = max(cv_results, key=lambda n: cv_results[n]["cv_f1_macro_mean"])

    # Evaluate every model on the held-out test set for the comparison table,
    # but the choice above was already made.
    test_results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        test_results[name] = {
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
            "test_top3_accuracy": float(
                top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)
            ),
        }

    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)

    metrics = {
        "selected_model": best_name,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv": cv_results,
        "test": test_results,
        "per_class_f1_selected": {
            k: float(v["f1-score"]) for k, v in report.items() if k in set(y.unique())
        },
        "note": "Selection by CV macro-F1 on training data only. Test set used once for reporting.",
    }

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        (RESULTS_DIR / "crop_metrics.json").write_text(json.dumps(metrics, indent=2))

        fig, ax = plt.subplots(figsize=(11, 10))
        ConfusionMatrixDisplay.from_predictions(
            y_test, best_pred, ax=ax, xticks_rotation=90, colorbar=False
        )
        ax.set_title(f"Crop recommendation: confusion matrix ({best_name}, held-out test)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "crop_confusion_matrix.png", dpi=130)
        plt.close(fig)

        # Refit the selected model on all data for the deployed artifact.
        final = build_models()[best_name].fit(X, y)
        joblib.dump(
            {
                "model": final,
                "features": CROP_FEATURES,
                "classes": list(final.classes_),
                "model_name": best_name,
            },
            MODEL_PATH,
        )

    return metrics


def format_table(metrics: dict) -> str:
    lines = [
        "| Model | CV acc (mean±std) | CV macro-F1 (mean±std) | Test acc | Test macro-F1 | Test top-3 |",
        "|---|---|---|---|---|---|",
    ]
    for name in metrics["cv"]:
        c, t = metrics["cv"][name], metrics["test"][name]
        mark = " (selected)" if name == metrics["selected_model"] else ""
        lines.append(
            f"| {name}{mark} | {c['cv_accuracy_mean']:.3f}±{c['cv_accuracy_std']:.3f} "
            f"| {c['cv_f1_macro_mean']:.3f}±{c['cv_f1_macro_std']:.3f} "
            f"| {t['test_accuracy']:.3f} | {t['test_f1_macro']:.3f} | {t['test_top3_accuracy']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    df = load_crop_data()
    metrics = fit_and_evaluate(df)
    print(format_table(metrics))
    worst = sorted(metrics["per_class_f1_selected"].items(), key=lambda kv: kv[1])[:3]
    print("\nLowest per-class F1 (selected model):", [(k, round(v, 3)) for k, v in worst])
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics and confusion matrix to {RESULTS_DIR}")
    print(f"Duplicates in data: {int(df.duplicated().sum())}")


if __name__ == "__main__":
    main()
