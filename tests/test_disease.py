import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kisansense.disease import data as D  # noqa: E402

CLASSES = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"]


def _save(path: Path, seed: int, size=(64, 64)):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(path)


def _make_tree(root: Path, per_class=30, duplicates=4):
    for ci, c in enumerate(CLASSES):
        for i in range(per_class):
            _save(root / c / f"img{i}.jpg", seed=ci * 1000 + i)
        for j in range(duplicates):  # exact copies of earlier images
            src = root / c / f"img{j}.jpg"
            (root / c / f"copy{j}.jpg").write_bytes(src.read_bytes())
    return root


def test_dhash_same_image_same_hash_different_image_different_hash(tmp_path):
    _save(tmp_path / "a.png", 1)
    _save(tmp_path / "b.png", 2)
    (tmp_path / "c.png").write_bytes((tmp_path / "a.png").read_bytes())
    assert D.dhash(str(tmp_path / "a.png")) == D.dhash(str(tmp_path / "c.png"))
    assert D.dhash(str(tmp_path / "a.png")) != D.dhash(str(tmp_path / "b.png"))


def test_group_split_keeps_duplicates_together_and_is_stratified(tmp_path):
    root = _make_tree(tmp_path / "t")
    df = D.add_groups(D.list_images(root, CLASSES))
    assert (df.groupby("group").size() > 1).sum() == 4 * len(CLASSES)
    tr, va, te = D.group_split(df, seed=0)
    assert len(tr) + len(va) + len(te) == len(df)
    assert not set(tr["path"]) & set(te["path"])
    for part in (tr, va, te):
        assert set(part["label"]) == set(CLASSES)
    assert 0.5 < len(tr) / len(df) < 0.7


def test_extract_classes_picks_only_color_folder_and_wanted_classes(tmp_path):
    z = tmp_path / "pv.zip"
    img = tmp_path / "x.jpg"
    _save(img, 0)
    data = img.read_bytes()
    with zipfile.ZipFile(z, "w") as zf:
        for c in CLASSES + ["Apple___healthy"]:
            zf.writestr(f"plantvillage dataset/color/{c}/a.jpg", data)
            zf.writestr(f"plantvillage dataset/grayscale/{c}/a.jpg", data)
            zf.writestr(f"plantvillage dataset/segmented/{c}/a.jpg", data)
    counts = D.extract_classes(z, tmp_path / "out", CLASSES)
    assert counts == {c: 1 for c in CLASSES}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(CLASSES)


def test_extract_classes_raises_if_class_missing(tmp_path):
    z = tmp_path / "pv.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("color/Tomato___healthy/a.jpg", b"x")
    with pytest.raises(ValueError):
        D.extract_classes(z, tmp_path / "out", CLASSES)


def test_class_weights_favour_rare_class(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"label": ["a"] * 90 + ["b"] * 10})
    w = D.class_weights(df, ["a", "b"])
    assert w[1] > w[0]


# ---- TensorFlow-dependent tests (skipped if TensorFlow is not installed) ----
tf = pytest.importorskip("tensorflow")

from kisansense.disease.gradcam import compute_gradcam, overlay  # noqa: E402
from kisansense.disease.model import build_baseline, build_transfer, unfreeze_top  # noqa: E402
from kisansense.disease import predict as P  # noqa: E402

SIZE = (96, 96)


def test_models_output_probabilities():
    x = np.random.default_rng(0).uniform(0, 255, (2, *SIZE, 3)).astype("float32")
    for m in (build_baseline(3, SIZE), build_transfer(3, SIZE, weights=None)):
        p = m.predict(x, verbose=0)
        assert p.shape == (2, 3)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4)


def test_unfreeze_top_keeps_batchnorm_frozen():
    m = build_transfer(3, SIZE, weights=None)
    unfreeze_top(m, 30)
    bb = m.get_layer("backbone")
    assert any(l.trainable for l in bb.layers)
    assert not any(l.trainable for l in bb.layers if isinstance(l, tf.keras.layers.BatchNormalization))


def test_gradcam_shapes_and_range():
    m = build_transfer(3, SIZE, weights=None)
    img = np.random.default_rng(1).uniform(0, 255, (*SIZE, 3)).astype("float32")
    cam, probs, idx = compute_gradcam(m, img)
    assert cam.ndim == 2 and cam.min() >= 0 and cam.max() <= 1.0 + 1e-6
    assert probs.shape == (3,) and 0 <= idx < 3
    out = overlay(img.astype("uint8"), cam)
    assert out.shape == (*SIZE, 3) and out.dtype == np.uint8


def test_predict_image_flags_uncertain_and_returns_gradcam():
    m = build_transfer(3, SIZE, weights=None)  # untrained: near-uniform probabilities
    classes = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"]
    img = Image.fromarray(np.random.default_rng(2).integers(0, 255, (120, 120, 3), dtype=np.uint8))
    out = P.predict_image(img, artifact=(m, classes))
    assert out["uncertain"] is True and out["prediction"] is None
    assert len(out["top"]) == 3
    assert out["gradcam"].shape == (*SIZE, 3)
    assert "extension officer" in out["reminder"]
