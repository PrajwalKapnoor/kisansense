"""Train and evaluate the crop recommendation models.

Run from the project root:
    python scripts/train_crop.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kisansense.crop.train import main  # noqa: E402

if __name__ == "__main__":
    main()
