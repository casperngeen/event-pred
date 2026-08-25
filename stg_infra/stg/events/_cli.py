"""Shared CLI runner for the ``stg.events`` per-series scripts.

``cpi.py``, ``gdp.py``, ``payrolls.py``, ``unemployment.py``, and
``core_cpi.py`` each define one or more ``compute_*_series(markets, trades)``
functions, then repeat the same ``__main__`` boilerplate: load markets/trades
from ``DATA_DIR``, run the series function(s), print a short summary, and
save the combined result to disk. That boilerplate lives here once instead
of five times.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Tuple

import polars as pl

from stg.events.config import DATA_DIR

log = logging.getLogger(__name__)

SeriesFn = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]


def load_markets_and_trades() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load the full markets/trades tables from ``DATA_DIR``."""
    log.info("Loading data from %s...", DATA_DIR)
    markets = pl.read_parquet(str(DATA_DIR / "markets/*.parquet"))
    trades = pl.read_parquet(str(DATA_DIR / "trades/*.parquet"))
    return markets, trades


def run_and_save(
    series_fns: Dict[str, SeriesFn],
    output_path: str,
    preview_col: str = "implied_mean",
) -> pl.DataFrame:
    """Run each ``label -> compute_fn`` pair, print a summary, save the union.

    Parameters
    ----------
    series_fns
        Maps a short label (used only in the printed summary, e.g. "CPI MoM")
        to a ``compute_*_series(markets, trades) -> pl.DataFrame`` function.
    output_path
        Where to write the concatenated result as Parquet. Parent
        directories are created if they don't exist.
    preview_col
        Column to filter non-null rows on for the printed preview.

    Returns
    -------
    The concatenated result of every non-empty series.
    """
    markets, trades = load_markets_and_trades()

    results: Dict[str, pl.DataFrame] = {name: fn(markets, trades) for name, fn in series_fns.items()}
    non_empty = [df for df in results.values() if not df.is_empty()]
    combined = pl.concat(non_empty) if non_empty else pl.DataFrame()

    print()
    for name, df in results.items():
        print(f"{name} rows: {len(df)}")
    print(f"Total rows:     {len(combined)}")
    if "event_ticker" in combined.columns:
        print(f"Events covered: {combined['event_ticker'].n_unique()}")
    if preview_col in combined.columns:
        print(combined.filter(pl.col(preview_col).is_not_null()).head(10))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(str(out))
    log.info("Saved to %s", out)

    return combined