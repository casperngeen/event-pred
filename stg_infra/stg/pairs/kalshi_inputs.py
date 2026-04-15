from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import polars as pl

from stg_infra.stg.io.loaders import DatasetLoader


def load_kalshi_trades(
    paths: Union[str, Sequence[str]],
    *,
    columns: Optional[Sequence[str]] = None,
    sort_deterministic: bool = True,
) -> pl.DataFrame:
    """
    Load Kalshi trades parquet(s) for the pairs-arb pipeline.
    - Uses DatasetLoader (lazy scan) 
    - Enforces deterministic ordering for OHLC reconstruction:
      sort by created_time then trade_id (if present).
    """
    if columns is None:
        # Minimum fields required by KalshiOHLCV.build_daily
        columns = ["ticker", "created_time", "yes_price", "count", "trade_id"]

    df = DatasetLoader(paths, file_format="parquet", columns=list(columns)).load()

    if sort_deterministic:
        if "trade_id" in df.columns:
            df = df.sort(["created_time", "trade_id"])
        else:
            df = df.sort(["created_time"])

    return df


def load_kalshi_markets(
    paths: Union[str, Sequence[str]],
    *,
    columns: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    Load Kalshi markets parquet(s) for the pairs-arb pipeline.
    """
    if columns is None:
        columns = ["ticker", "event_ticker", "title", "open_time", "close_time", "result", "_fetched_at"]

    return DatasetLoader(paths, file_format="parquet", columns=list(columns)).load()


def load_trades_and_markets(
    trades_paths: Union[str, Sequence[str]],
    markets_paths: Union[str, Sequence[str]],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Convenience wrapper used by the pipeline runner.
    """
    trades = load_kalshi_trades(trades_paths)
    markets = load_kalshi_markets(markets_paths)
    return trades, markets