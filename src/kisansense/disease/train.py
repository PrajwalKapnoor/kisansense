"""Train and compare the baseline CNN and MobileNetV2 on the tomato classes.

Selection uses validation macro-F1. The test set is scored once, for reporting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score  # noqa: E402

from kisansense.common.seed import set_seed  # noqa: E402
from kisansense.disease import data as D  # noqa: E402
from kisansense.disease.model import build_baseline, build_transfer, unfreeze_top  # noqa: E402


def _fit(model, train_ds, val_ds, epochs, lr, class_weight, patience=3):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience - 1)),
    ]
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=cbs, verbose=2)
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}


def evaluate(model, ds, class_names):
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_pred = model.predict(ds, verbose=0).argmax(axis=1)
    rep = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": {c: float(rep[c]["f1-score"]) for c in class_names},
        "n": int(len(y_true)),
    }, y_true, y_pred


def _save_confusion(y_true, y_pred, class_names, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    short = [c.replace("Tomato___", "") for c in class_names]
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(len(class_names)), display_labels=short,
        ax=ax, xticks_rotation=45, colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(
    data_root: Path,
    out_dir: Path,
    results_dir: Path,
    classes: list[str] = D.TOMATO_CLASSES,
    image_size=(224, 224),
    batch_size: int = 32,
    baseline_epochs: int = 15,
    head_epochs: int = 8,
    finetune_epochs: int = 8,
    weights: str | None = "imagenet",
    seed: int = 42,
) -> dict:
    set_seed(seed)
    out_dir, results_dir = Path(out_dir), Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = D.add_groups(D.list_images(data_root, classes))
    train_df, val_df, test_df = D.group_split(df, seed)
    class_names = list(classes)
    cw = D.class_weights(train_df, class_names)

    mk = lambda d, training: D.make_dataset(d, class_names, image_size, batch_size, training, seed)  # noqa: E731
    train_ds, val_ds, test_ds = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    info = {
        "classes": class_names,
        "images": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "duplicate_groups": int((df.groupby("group").size() > 1).sum()),
        "images_in_duplicate_groups": int(df.groupby("group")["path"].transform("size").gt(1).sum()),
        "class_counts": df["label"].value_counts().to_dict(),
    }

    histories, results, preds, models = {}, {}, {}, {}

    base = build_baseline(len(class_names), image_size)
    histories["baseline_cnn"] = _fit(base, train_ds, val_ds, baseline_epochs, 1e-3, cw, patience=4)
    models["baseline_cnn"] = base

    tl = build_transfer(len(class_names), image_size, weights)
    histories["transfer_head"] = _fit(tl, train_ds, val_ds, head_epochs, 1e-3, cw, patience=3)
    unfreeze_top(tl, 30)
    histories["transfer_finetune"] = _fit(tl, train_ds, val_ds, finetune_epochs, 1e-5, cw, patience=3)
    models["mobilenetv2_finetuned"] = tl

    for name, m in models.items():
        v, _, _ = evaluate(m, val_ds, class_names)
        t, yt, yp = evaluate(m, test_ds, class_names)
        results[name] = {"validation": v, "test": t}
        preds[name] = (yt, yp)
        _save_confusion(yt, yp, class_names, f"{name}: confusion matrix (test)", results_dir / f"disease_confusion_{name}.png")

    best = max(results, key=lambda n: results[n]["validation"]["macro_f1"])
    models[best].save(out_dir / "disease_model.keras")
    (out_dir / "disease_classes.json").write_text(json.dumps(class_names, indent=2))

    metrics = {
        "data": info,
        "selected_model": best,
        "results": results,
        "note": "Selected by validation macro-F1; test set used once for reporting. "
                "PlantVillage photos are lab images; scores do not describe field performance.",
    }
    (results_dir / "disease_metrics.json").write_text(json.dumps(metrics, indent=2))
    (results_dir / "disease_histories.json").write_text(json.dumps(histories, indent=2))
    return {"metrics": metrics, "model": models[best], "class_names": class_names, "test_df": test_df,
            "image_size": image_size}


def format_table(metrics: dict) -> str:
    lines = ["| Model | Val acc | Val macro-F1 | Test acc | Test macro-F1 |", "|---|---|---|---|---|"]
    for name, r in metrics["results"].items():
        mark = " (selected)" if name == metrics["selected_model"] else ""
        lines.append(
            f"| {name}{mark} | {r['validation']['accuracy']:.3f} | {r['validation']['macro_f1']:.3f} "
            f"| {r['test']['accuracy']:.3f} | {r['test']['macro_f1']:.3f} |"
        )
    return "\n".join(lines)
