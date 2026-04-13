"""Kalshi-specific data preprocessing utilities."""

from __future__ import annotations

import polars as pl


class KalshiOHLCV:
    """Reconstruct daily OHLCV and event-level features from Kalshi trade data.

    Usage
    -----
    daily = KalshiOHLCV.build_daily(trades, markets)
    event = KalshiOHLCV.aggregate_to_event_level(daily)
    """

    @staticmethod
    def build_daily(trades: pl.DataFrame, markets: pl.DataFrame) -> pl.DataFrame:
        """Reconstruct daily OHLCV (yes_price) per market ticker from raw trades.

        One metadata row per ticker is kept (most recently fetched snapshot).
        Missing days within each market's active window are forward-filled and
        rows beyond the market close date are dropped.
        """
        market_meta = (
            markets
            .sort("_fetched_at", descending=True)
            .unique(subset=["ticker"], keep="first")
            .select(["ticker", "event_ticker", "title", "open_time", "close_time", "result"])
        )

        daily = (
            trades
            .sort("created_time")
            .with_columns(pl.col("created_time").dt.date().alias("date"))
            .group_by(["ticker", "date"])
            .agg([
                pl.col("yes_price").first().alias("open"),
                pl.col("yes_price").max().alias("high"),
                pl.col("yes_price").min().alias("low"),
                pl.col("yes_price").last().alias("close"),
                pl.col("count").sum().alias("volume"),
                pl.len().alias("trade_count"),
            ])
            .sort(["ticker", "date"])
            .join(market_meta, on="ticker", how="inner")
            .with_columns([
                pl.col("ticker").str.replace(r"^KX", "").alias("clean_ticker"),
                pl.col("event_ticker").str.replace(r"^KX", "").alias("clean_event_ticker"),
            ])
        )

        filled = (
            daily
            .sort(["ticker", "date"])
            .group_by("ticker")
            .map_groups(lambda df:
                df
                .upsample(time_column="date", every="1d")
                .with_columns(pl.col("close").forward_fill())
                .with_columns([
                    pl.col("open").fill_null(pl.col("close")),
                    pl.col("high").fill_null(pl.col("close")),
                    pl.col("low").fill_null(pl.col("close")),
                    pl.col("ticker").forward_fill(),
                    pl.col("event_ticker").forward_fill(),
                    pl.col("clean_ticker").forward_fill(),
                    pl.col("clean_event_ticker").forward_fill(),
                    pl.col("title").forward_fill(),
                    pl.col("open_time").forward_fill(),
                    pl.col("close_time").forward_fill(),
                    pl.col("result").forward_fill(),
                    pl.col("volume").fill_null(0),
                    pl.col("trade_count").fill_null(0),
                ])
            )
            .filter(pl.col("date") <= pl.col("close_time").dt.date())
            .sort(["ticker", "date"])
        )

        return filled

    @staticmethod
    def aggregate_to_event_level(daily_ohlcv: pl.DataFrame) -> pl.DataFrame:
        """Aggregate per-ticker daily OHLCV up to event-level features.

        Returns a DataFrame with one row per (event_ticker, date) containing
        normalised features suitable for direct use as model inputs.

        Columns
        -------
        event_ticker, date,
        implied_mean_raw    — VWAP across submarkets, normalised to [0, 1]
        implied_mean_norm   — z-scored per event  (model input feature 0)
        price_spread_norm   — z-scored per event  (model input feature 1)
        log_volume_norm     — z-scored per event  (model input feature 2)
        n_submarkets        — number of active submarkets  (model input feature 3)
        daily_return        — day-over-day change in implied_mean_raw  (feature 4)
        days_to_close       — fraction of a year until market close  (feature 5)
        close_time, result
        """
        df = (
            daily_ohlcv
            .group_by(["event_ticker", "date"])
            .agg([
                (
                    (pl.col("close") * pl.col("volume")).sum() /
                    pl.col("volume").sum().clip(lower_bound=1)
                ).alias("implied_mean_raw"),
                (pl.col("close").max() - pl.col("close").min()).alias("price_spread"),
                pl.col("volume").sum().alias("total_volume"),
                pl.col("ticker").n_unique().alias("n_submarkets"),
                pl.col("close_time").min().alias("close_time"),
                pl.col("result").first().alias("result"),
            ])
            .with_columns(
                (pl.col("implied_mean_raw") / 100).alias("implied_mean_raw"),
                (pl.col("price_spread") / 100).alias("price_spread"),
            )
            .sort(["event_ticker", "date"])
        )

        df = df.with_columns(
            (
                pl.col("implied_mean_raw") -
                pl.col("implied_mean_raw").shift(1).over("event_ticker")
            ).fill_null(0.0).alias("daily_return")
        )

        df = df.with_columns(
            (
                (pl.col("close_time").dt.date() - pl.col("date"))
                .dt.total_days()
                .clip(lower_bound=0)
                .cast(pl.Float64) / 365.0
            ).alias("days_to_close")
        )

        df = df.with_columns(
            (pl.col("total_volume") + 1).log().alias("log_volume")
        )

        df = df.with_columns([
            (
                (pl.col("implied_mean_raw") -
                 pl.col("implied_mean_raw").mean().over("event_ticker")) /
                (pl.col("implied_mean_raw").std().over("event_ticker") + 1e-8)
            ).alias("implied_mean_norm"),
            (
                (pl.col("log_volume") -
                 pl.col("log_volume").mean().over("event_ticker")) /
                (pl.col("log_volume").std().over("event_ticker") + 1e-8)
            ).alias("log_volume_norm"),
            (
                (pl.col("price_spread") -
                 pl.col("price_spread").mean().over("event_ticker")) /
                (pl.col("price_spread").std().over("event_ticker") + 1e-8)
            ).alias("price_spread_norm"),
        ])

        return df.select([
            "event_ticker",
            "date",
            "implied_mean_raw",
            "implied_mean_norm",
            "price_spread_norm",
            "log_volume_norm",
            "n_submarkets",
            "daily_return",
            "days_to_close",
            "close_time",
            "result",
        ])
