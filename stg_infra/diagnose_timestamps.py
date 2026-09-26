"""
diagnose_timestamps.py

Fast, standalone check -- does NOT call build_combined_graph or anything
that can hang. Just loads the 5 months of trades/markets and reports the
full timestamp distribution, so we can confirm or rule out a corrupted/
out-of-range created_time value directly, in seconds, instead of waiting
through another multi-minute hang to find out.

Run from stg_infra/:  python diagnose_timestamps.py
"""
import sys
from pathlib import Path

import polars as pl

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]


def _parity_path(kind: str, month: str) -> str:
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return str(_REPO_ROOT / f"data/{kind}/{kind}_kalshi_{parity}/{kind}_{month}.parquet")


def check(months):
    for kind in ("markets", "trades"):
        paths = [_parity_path(kind, m) for m in months if Path(_parity_path(kind, m)).exists()]
        if not paths:
            print(f"{kind}: no files found")
            continue
        print(f"\n=== {kind} ===")
        df = pl.concat([pl.scan_parquet(p) for p in paths]).collect()
        print(f"  {df.height} rows, columns: {df.columns}")

        if "created_time" not in df.columns:
            print(f"  NO 'created_time' column present -- this is NOT the time column FixedWindowTemporal uses!")
            continue

        col = df["created_time"]
        print(f"  dtype: {col.dtype}")
        print(f"  min: {col.min()}")
        print(f"  max: {col.max()}")
        n_null = col.null_count()
        print(f"  null count: {n_null}")

        # Expected range for the requested months, generously padded by a
        # week on each side so a legitimate boundary trade near month-end
        # doesn't get flagged as an "outlier".
        from datetime import datetime, timedelta
        first, last = min(months), max(months)
        fy, fm = (int(x) for x in first.split("-"))
        ly, lm = (int(x) for x in last.split("-"))
        lm += 1
        if lm == 13:
            lm, ly = 1, ly + 1
        lo = datetime(fy, fm, 1) - timedelta(days=7)
        hi = datetime(ly, lm, 1) + timedelta(days=7)

        outliers = df.filter((pl.col("created_time") < lo) | (pl.col("created_time") > hi))
        print(f"  rows outside padded expected range [{lo}, {hi}]: {outliers.height}")
        if outliers.height > 0:
            print(f"  --> OUTLIER ROWS FOUND. Sample:")
            print(outliers.select([c for c in ("created_time", "ticker") if c in outliers.columns]).head(20))
            print(f"  outlier created_time min/max: {outliers['created_time'].min()} / {outliers['created_time'].max()}")
        else:
            print(f"  --> no outliers found in this column for this table.")


if __name__ == "__main__":
    check(PILOT_MONTHS)