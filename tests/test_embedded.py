"""Embedded-mode helpers: model download logic and the on/off switch (no network, no TensorFlow)."""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import embedded  # noqa: E402


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("KISANSENSE_EMBED_API", raising=False)
    assert embedded.enabled() is False


def test_enabled_by_environment(monkeypatch):
    monkeypatch.setenv("KISANSENSE_EMBED_API", "1")
    assert embedded.enabled() is True


def test_downloads_only_missing_files(tmp_path):
    (tmp_path / "crop_model.joblib").write_text("x")
    calls = []

    def fake_download(repo_id, filename, local_dir):
        calls.append((repo_id, filename))
        (Path(local_dir) / filename).write_text("downloaded")
        return str(Path(local_dir) / filename)

    got = embedded.ensure_models(repo="me/models", download=fake_download, target=tmp_path)
    assert "crop_model.joblib" not in got
    assert sorted(got) == sorted(f for f in embedded.MODEL_FILES if f != "crop_model.joblib")
    assert all(repo == "me/models" for repo, _ in calls)
    assert all((tmp_path / f).exists() for f in embedded.MODEL_FILES)


def test_nothing_to_download_when_all_present(tmp_path):
    for f in embedded.MODEL_FILES:
        (tmp_path / f).write_text("x")

    def boom(**kwargs):
        raise AssertionError("should not download")

    assert embedded.ensure_models(repo="me/models", download=boom, target=tmp_path) == []
