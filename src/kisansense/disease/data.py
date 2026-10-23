"""Disease data: extract tomato classes, build duplicate-aware splits, tf.data pipelines."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
]

_ZIP_PATTERN = re.compile(r"(?:^|/)color/(?P<cls>[^/]+)/(?P<name>[^/]+)$", re.IGNORECASE)


def extract_classes(zip_path: Path, dest: Path, classes: list[str] = TOMATO_CLASSES) -> dict[str, int]:
    """Extract only the wanted classes from the `color` version of PlantVillage.

    Works whatever the top-level folder inside the zip is called. Returns
    {class: image_count}. Grayscale and segmented copies are ignored on purpose:
    they are the same pictures processed differently and would leak across splits.
    """
    wanted = set(classes)
    counts: dict[str, int] = {c: 0 for c in classes}
    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            m = _ZIP_PATTERN.search(info.filename)
            if not m or m["cls"] not in wanted:
                continue
            if Path(m["name"]).suffix.lower() not in IMG_EXTS:
                continue
            out = dest / m["cls"] / m["name"]
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as dst:
                dst.write(src.read())
            counts[m["cls"]] += 1
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise ValueError(f"No images found for classes {missing}. Check the zip layout (expects .../color/<class>/...).")
    return counts


def list_images(root: Path, classes: list[str] | None = None) -> pd.DataFrame:
    root = Path(root)
    classes = classes or sorted(d.name for d in root.iterdir() if d.is_dir())
    rows = [
        (str(p), c)
        for c in classes
        for p in sorted((root / c).iterdir())
        if p.suffix.lower() in IMG_EXTS
    ]
    if not rows:
        raise ValueError(f"No images found under {root}")
    return pd.DataFrame(rows, columns=["path", "label"])


def dhash(path: str, size: int = 16) -> int:
    """256-bit difference hash (size=16). Identical hashes mean near-identical
    pictures (re-saves, tiny edits). 16x16 keeps accidental collisions rare."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Give images with the same hash the same group id so they stay in one split."""
    out = df.copy()
    out["hash"] = [dhash(p) for p in out["path"]]
    out["group"] = out.groupby("hash").ngroup()
    return out


def group_split(df: pd.DataFrame, seed: int = 42):
    """Stratified 60/20/20 train/val/test split that never separates a group."""
    from sklearn.model_selection import StratifiedGroupKFold

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tv_idx, test_idx = next(outer.split(df, df["label"], df["group"]))
    tv = df.iloc[tv_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(inner.split(tv, tv["label"], tv["group"]))
    train, val, test = tv.iloc[tr_idx], tv.iloc[va_idx], df.iloc[test_idx]
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(a["group"]) & set(b["group"]), "group leaked across splits"
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def class_weights(train_df: pd.DataFrame, class_names: list[str]) -> dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight

    y = train_df["label"].map({c: i for i, c in enumerate(class_names)}).to_numpy()
    w = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y)
    return {i: float(x) for i, x in enumerate(w)}


def make_dataset(df: pd.DataFrame, class_names: list[str], image_size=(224, 224), batch_size: int = 32,
                 training: bool = False, seed: int = 42):
    import tensorflow as tf

    index = {c: i for i, c in enumerate(class_names)}
    paths = df["path"].to_numpy()
    labels = df["label"].map(index).to_numpy().astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img.set_shape([image_size[0], image_size[1], 3])
        return img, y

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
