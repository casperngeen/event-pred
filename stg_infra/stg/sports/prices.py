"""Per-market quote series: bid / ask observations, indexed by ``known_at``.

Two sources, merged per side (the newest observation of each side wins):

    candles   yes_bid_close / yes_ask_close at the end of each period: real
              two-sided quotes (a missing side counts as 0 / 100, i.e. no quote)
    trades    a YES-taker trade at p shows the ask was p; a NO-taker trade at p
              (yes price) shows the bid was p. Until the candles are pulled
              this is the only quote information, and it goes stale between
              trades, so every quote carries the time of each side.

A mid is valid only when both sides are known, not crossed, at most
``MAX_SPREAD`` apart, and neither side older than ``max_age`` (rule 2). Changes
are measured in log-odds (:func:`logodds`).

Series are built once by ``scripts/build_sports_layer.py`` into
``data/sports/derived/quotes/<series>.parquet``, sorted by (ticker, known_at).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from stg.sports.loaders import derived_dir, scan_candles, scan_trades

MAX_SPREAD = 5.0                      # cents (rule 2)
MAX_AGE = dt.timedelta(minutes=10)    # a side older than this is stale
LO_CLIP = 0.5                         # cents: log-odds of 0.5c..99.5c

QUOTE_SCHEMA = {"ticker": pl.String, "known_at": pl.Datetime("us", "UTC"), "src": pl.String,
                "price": pl.Float64, "count": pl.Float64, "taker_side": pl.String,
                "bid": pl.Float64, "bid_at": pl.Datetime("us", "UTC"),
                "ask": pl.Float64, "ask_at": pl.Datetime("us", "UTC")}


def _trade_obs(tr: pl.LazyFrame) -> pl.LazyFrame:
    return tr.select(
        "ticker", "known_at", pl.lit("trade").alias("src"), pl.col("yes_price").alias("price"), "count",
        "taker_side",
        pl.when(pl.col("taker_side") == "no").then(pl.col("yes_price")).alias("_bid"),
        pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price")).alias("_ask"))


def _candle_obs(cd: pl.LazyFrame) -> pl.LazyFrame:
    return cd.select(
        "ticker", "known_at", pl.lit("candle").alias("src"), pl.lit(None, pl.Float64).alias("price"),
        pl.lit(None, pl.Float64).alias("count"), pl.lit(None, pl.String).alias("taker_side"),
        pl.col("yes_bid_close").fill_null(0.0).alias("_bid"),
        pl.col("yes_ask_close").fill_null(100.0).alias("_ask"))


def build_quotes(series: str) -> pl.DataFrame:
    """The merged quote series of every market in one series."""
    parts = [_trade_obs(scan_trades(series))]
    cd = scan_candles(series)
    if cd is not None:
        parts.append(_candle_obs(cd))
    obs = pl.concat(parts, how="vertical_relaxed").sort("ticker", "known_at", "src")
    side_at = lambda c: pl.when(pl.col(c).is_not_null()).then(pl.col("known_at"))  # noqa: E731
    q = obs.with_columns(
        pl.col("_bid").forward_fill().over("ticker").alias("bid"),
        side_at("_bid").forward_fill().over("ticker").alias("bid_at"),
        pl.col("_ask").forward_fill().over("ticker").alias("ask"),
        side_at("_ask").forward_fill().over("ticker").alias("ask_at"),
    ).drop("_bid", "_ask")
    return q.collect().cast(QUOTE_SCHEMA).sort("ticker", "known_at")


def quotes_path(series: str):
    return derived_dir() / "quotes" / f"{series}.parquet"


def write_quotes(series: str) -> int:
    q = build_quotes(series)
    p = quotes_path(series)
    p.parent.mkdir(parents=True, exist_ok=True)
    q.write_parquet(p, statistics=True, row_group_size=256_000)
    return q.height


def scan_quotes(series: str | list[str]) -> pl.LazyFrame:
    names = [series] if isinstance(series, str) else series
    files = [str(quotes_path(s)) for s in names if quotes_path(s).exists()]
    if not files:
        return pl.LazyFrame(schema=QUOTE_SCHEMA)
    return pl.scan_parquet(files)


def logodds(p_cents: pl.Expr) -> pl.Expr:
    p = p_cents.clip(LO_CLIP, 100 - LO_CLIP) / 100
    return (p / (1 - p)).log()


def mid_exprs(t: pl.Expr, max_spread: float = MAX_SPREAD, max_age: dt.timedelta = MAX_AGE) -> list[pl.Expr]:
    """bid / ask / mid / spread / age columns of a quote row read at time ``t``."""
    age = (t - pl.min_horizontal("bid_at", "ask_at"))
    ok = (pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("bid") > 0)
          & (pl.col("ask") < 100) & (pl.col("bid") <= pl.col("ask"))
          & (pl.col("ask") - pl.col("bid") <= max_spread) & (age <= max_age))
    mid = pl.when(ok).then((pl.col("bid") + pl.col("ask")) / 2)
    return [(pl.col("ask") - pl.col("bid")).alias("spread"), age.alias("quote_age"), ok.alias("two_sided"),
            mid.alias("mid"), logodds(mid).alias("mid_lo")]


def quotes_at(q: pl.LazyFrame | pl.DataFrame, at: pl.DataFrame, t_col: str = "t",
              max_spread: float = MAX_SPREAD, max_age: dt.timedelta = MAX_AGE) -> pl.DataFrame:
    """For each (ticker, t) row of ``at``: the latest quote with known_at <= t, and its mid.

    Also the last trade price and the executable prices: buy at the ask, sell at the bid.
    """
    q = q.lazy() if isinstance(q, pl.DataFrame) else q
    tickers = at["ticker"].unique().implode()
    qq = (q.filter(pl.col("ticker").is_in(tickers))
          .with_columns(pl.col("price").forward_fill().over("ticker").alias("last_trade"),
                        pl.col("known_at").alias("quote_known_at"))
          .select("ticker", "known_at", "quote_known_at", "last_trade", "bid", "bid_at", "ask", "ask_at")
          .collect().sort("known_at", maintain_order=True))   # stable: quotes sharing a time keep their file order
    left = at.with_row_index("_row").sort(t_col, maintain_order=True)
    j = left.join_asof(qq, left_on=t_col, right_on="known_at", by="ticker", strategy="backward")
    return (j.with_columns(mid_exprs(pl.col(t_col), max_spread, max_age))
            .sort("_row").drop("_row", "known_at"))


class QuoteBook:
    """One series' quotes as numpy arrays, for fast point lookups by position."""

    def __init__(self, q: pl.DataFrame) -> None:
        q = q.sort("ticker", "known_at")
        self.t = q["known_at"].dt.epoch("us").to_numpy()
        self.bid = q["bid"].to_numpy()
        self.ask = q["ask"].to_numpy()
        self.bid_at = q["bid_at"].dt.epoch("us").fill_null(-1).to_numpy()
        self.ask_at = q["ask_at"].dt.epoch("us").fill_null(-1).to_numpy()
        self.price = q["price"].forward_fill().to_numpy()
        tick = q["ticker"].to_numpy()
        starts = np.r_[0, np.nonzero(tick[1:] != tick[:-1])[0] + 1] if len(tick) else np.array([], int)
        ends = np.r_[starts[1:], len(tick)] if len(tick) else np.array([], int)
        self.span = {tick[s]: (s, e) for s, e in zip(starts, ends)}

    def at(self, ticker: str, t: dt.datetime, max_spread: float = MAX_SPREAD,
           max_age: dt.timedelta = MAX_AGE) -> dict | None:
        """The quote of ``ticker`` as of ``t`` (rows with known_at <= t), or None."""
        s = self.span.get(ticker)
        if s is None:
            return None
        us = int(t.timestamp() * 1_000_000)
        i = s[0] + int(np.searchsorted(self.t[s[0]:s[1]], us, side="right")) - 1
        if i < s[0]:
            return None
        bid, ask = self.bid[i], self.ask[i]
        sides = [x for x in (self.bid_at[i], self.ask_at[i])]
        age = (us - min(sides)) / 1e6 if min(sides) >= 0 else np.inf
        ok = (np.isfinite(bid) and np.isfinite(ask) and 0 < bid <= ask < 100
              and ask - bid <= max_spread and age <= max_age.total_seconds())
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2 if ok else None, "age_s": age,
                "last_trade": self.price[i], "known_at_us": int(self.t[i])}
