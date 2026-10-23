"""Backtest and train the tomato price forecasting models.

Run from the project root:
    python scripts/train_price.py
Takes a few minutes: every model is refitted at each backtest origin.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kisansense.price.train import main  # noqa: E402

if __name__ == "__main__":
    main()
