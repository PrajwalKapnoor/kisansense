"""Start the API and the Streamlit app in one container (Hugging Face Spaces runs a single process tree)."""
import os
import signal
import subprocess
import sys
import time
import urllib.request

API_PORT = os.environ.get("API_PORT", "8000")
APP_PORT = os.environ.get("APP_PORT", "7860")


def wait_for_api(timeout: int = 180) -> bool:
    url = f"http://127.0.0.1:{API_PORT}/health"
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def main() -> int:
    # As PID 1 in a container, SIGTERM is ignored unless handled; turn it into a clean exit.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", API_PORT]
    )
    if not wait_for_api():
        print("API did not become healthy in time", file=sys.stderr)
        api.terminate()
        return 1
    # XSRF/CORS are relaxed because the Space shows the app inside an iframe; without this, file uploads fail with 403.
    app = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app/Home.py",
            "--server.address=0.0.0.0", f"--server.port={APP_PORT}", "--server.headless=true",
            "--server.enableXsrfProtection=false", "--server.enableCORS=false",
        ]
    )
    try:
        while api.poll() is None and app.poll() is None:
            time.sleep(2)
    finally:
        for p in (api, app):
            if p.poll() is None:
                p.terminate()
    return 1  # one of them stopped: exit non-zero so the Space restarts


if __name__ == "__main__":
    sys.exit(main())
