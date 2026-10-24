"""Assemble a folder you can push to a Hugging Face Space.

    python scripts/build_space.py                 # creates ./space_build
    python scripts/build_space.py my_folder

Needs the trained models in artifacts/. The output folder is not committed to the main repo.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["crop_model.joblib", "price_model.joblib", "disease_model.keras", "disease_classes.json"]
LFS_PATTERNS = ["*.keras", "*.joblib"]  # Hugging Face needs Git LFS for files over 10 MB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="space_build")
    ap.add_argument("--allow-missing", action="store_true", help="skip missing models (for testing only)")
    args = ap.parse_args()

    missing = [f for f in REQUIRED if not (ROOT / "artifacts" / f).exists()]
    if missing and not args.allow_missing:
        print("Missing in artifacts/:", ", ".join(missing))
        return 1

    out = (ROOT / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for folder in ("src", "backend", "app", ".streamlit"):
        shutil.copytree(ROOT / folder, out / folder, ignore=ignore)
    shutil.copy2(ROOT / "requirements.txt", out / "requirements.txt")

    (out / "artifacts").mkdir()
    for f in REQUIRED:
        if (ROOT / "artifacts" / f).exists():
            shutil.copy2(ROOT / "artifacts" / f, out / "artifacts" / f)

    (out / "docs" / "results").mkdir(parents=True)
    for f in (ROOT / "docs" / "results").glob("*_metrics.json"):
        shutil.copy2(f, out / "docs" / "results" / f.name)

    for name in ("Dockerfile", "start.py", "README.md"):
        shutil.copy2(ROOT / "deploy" / "huggingface" / name, out / name)

    lines = [f"{p} filter=lfs diff=lfs merge=lfs -text" for p in LFS_PATTERNS]
    (out / ".gitattributes").write_text("\n".join(lines) + "\n", newline="\n")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Built {out}  ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
