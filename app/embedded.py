"""Run the API inside the Streamlit process.

Some free hosts (Streamlit Community Cloud) run only one Streamlit app and nothing else. Setting
KISANSENSE_EMBED_API=1 (as an environment variable or a Streamlit secret) makes the app:
  1. download the trained models from a public Hugging Face *model* repo if they are not on disk, and
  2. start the FastAPI backend in a background thread on 127.0.0.1.
The rest of the frontend is unchanged: it still talks to the API over HTTP.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_FILES = ["crop_model.joblib", "price_model.joblib", "disease_model.keras", "disease_classes.json"]
DEFAULT_MODEL_REPO = "prajwalkapnoor/kisansense-models"
API_PORT = 8000


def _setting(name: str, default: str = "") -> str:
    """Environment variable first, then a Streamlit secret (secrets are optional)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets file configured
        return default


def enabled() -> bool:
    return _setting("KISANSENSE_EMBED_API") == "1"


def ensure_models(
    repo: str | None = None,
    download: Callable[..., str] | None = None,
    target: Path = ARTIFACTS,
) -> list[str]:
    """Fetch any missing model file. Returns the names that were downloaded."""
    repo = repo or _setting("KISANSENSE_MODEL_REPO", DEFAULT_MODEL_REPO)
    missing = [f for f in MODEL_FILES if not (target / f).exists()]
    if not missing:
        return []
    if download is None:
        from huggingface_hub import hf_hub_download as download  # lazy: only needed when hosting

    target.mkdir(parents=True, exist_ok=True)
    for name in missing:
        download(repo_id=repo, filename=name, local_dir=str(target))
    return missing


def _api_up() -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}/health", timeout=2)
        return True
    except Exception:
        return False


def start_api(wait_seconds: int = 90) -> None:
    if _api_up():
        return
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import uvicorn

    from backend.app.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=API_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="kisansense-api").start()
    end = time.time() + wait_seconds
    while time.time() < end:
        if _api_up():
            return
        time.sleep(0.5)
    raise RuntimeError("The API did not start in time.")


@st.cache_resource(show_spinner="Starting KisanSense (first visit takes about a minute)...")
def _start_once() -> bool:
    ensure_models()
    start_api()
    return True


def ensure_ready() -> None:
    """Called by the API client before each request. Does nothing unless embedded mode is on."""
    if enabled():
        _start_once()
