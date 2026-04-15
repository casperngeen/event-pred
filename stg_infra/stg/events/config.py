from __future__ import annotations

from pathlib import Path

import polars as pl

DATA_DIR   = Path("user_data/data/kalshi/kalshi")
DATE_START = pl.date(2022, 5, 1)
DATE_END   = pl.date(2025, 8, 1)
