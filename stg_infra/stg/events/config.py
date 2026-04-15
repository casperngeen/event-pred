from __future__ import annotations

import os
from pathlib import Path

import polars as pl


def _parse_date(env_var: str, default: pl.Expr) -> pl.Expr:
    val = os.environ.get(env_var)
    if val is None:
        return default
    y, m, d = (int(x) for x in val.split("-"))
    return pl.date(y, m, d)


# Override via environment variables:
#   KALSHI_DATA_DIR    — path to the kalshi data root (contains markets/ and trades/)
#   KALSHI_DATE_START  — inclusive start date, format YYYY-MM-DD
#   KALSHI_DATE_END    — inclusive end date,   format YYYY-MM-DD
DATA_DIR   = Path(os.environ.get("KALSHI_DATA_DIR", "user_data/data/kalshi/kalshi"))
DATE_START = _parse_date("KALSHI_DATE_START", pl.date(2022, 5, 1))
DATE_END   = _parse_date("KALSHI_DATE_END",   pl.date(2025, 8, 1))
