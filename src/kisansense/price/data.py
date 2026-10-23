"""Loading and cleaning the Agmarknet daily price export.

The exported CSV has a report title in the first row, and the real column
header on the second row, so a plain pd.read_csv() picks the wrong header.
"""
import csv
from pathlib import Path

import pandas as pd

from kisansense.common.config import PRICE_CSV

# Header text (lowercase prefix) -> clean column name. Order matters:
# "commodity group" must be tested before "commodity".
_COLUMN_RULES = [
    ("state", "state"),
    ("commodity group", "commodity_group"),
    ("commodity", "commodity"),
    ("date", "date"),
    ("arrival quantity", "arrival_qty"),
    ("arrival unit", "arrival_unit"),
    ("modal price", "modal_price"),
    ("price unit", "price_unit"),
]


def _clean_name(header: str) -> str:
    h = header.strip().lower()
    for prefix, name in _COLUMN_RULES:
        if h.startswith(prefix):
            return name
    return h.replace(" ", "_")


def load_price_data(path: Path = PRICE_CSV) -> pd.DataFrame:
    """Return a clean DataFrame sorted by date.

    Columns: state, commodity_group, commodity, date, arrival_qty,
    arrival_unit, modal_price, price_unit (those present in the file).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows[:20]):
        if any(cell.strip().lower() == "date" for cell in row):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header row containing 'Date' found in first 20 rows of {path}")

    header = [_clean_name(c) for c in rows[header_idx]]
    body = [r for r in rows[header_idx + 1 :] if any(c.strip() for c in r)]
    body = [(r + [""] * len(header))[: len(header)] for r in body]
    df = pd.DataFrame(body, columns=header)

    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%d-%m-%Y", errors="coerce")
    for col in ("arrival_qty", "modal_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")

    df = df.dropna(subset=["date"]).drop_duplicates(subset="date", keep="last")
    return df.sort_values("date").reset_index(drop=True)


def gap_report(df: pd.DataFrame) -> dict:
    """Summarise missing calendar days in a daily series."""
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing = full.difference(df["date"])
    longest = 0
    if len(missing):
        run = 1
        longest = 1
        for a, b in zip(missing[:-1], missing[1:]):
            run = run + 1 if (b - a).days == 1 else 1
            longest = max(longest, run)
    return {"expected_days": len(full), "present_days": len(df), "missing_days": len(missing), "longest_gap_days": longest}
