"""``SportsData``: the sports archive as it was known at any time t.

    data = SportsData()                  # everything on disk (working block)
    data = SportsData(cutoff=C)          # as if built at C: every row known after C is dropped
    v = data.as_of(t)
    v.markets()                          # listed (open_time <= t) and not yet closed at t
    v.quote("KXNBAGAME-25NOV17CHIDEN-CHI")
    v.results()                          # games whose result was known by t
    v.pbp("NBA", game_id)                # plays known by t

Bulk (for building sample tables): ``data.quotes_at(frame)`` joins each
(ticker, t) row to its latest quote known at or before t. Lookups run on
per-series quote series precomputed by ``scripts/build_sports_layer.py``
(``stg.sports.prices``), loaded lazily per series, so nothing re-filters the
full trade tape.

``close_time``, ``result`` and the other final listing fields are revealed only
once their time has passed; feature code never sees them otherwise.
"""
from __future__ import annotations

import datetime as dt
from functools import cached_property

import polars as pl

from stg.sports import loaders
from stg.sports.contracts import contracts
from stg.sports.prices import MAX_AGE, MAX_SPREAD, QuoteBook, quotes_at, scan_quotes


def _lit(t: dt.datetime) -> pl.Expr:
    return pl.lit(t, dtype=pl.Datetime("us", "UTC"))


class SportsData:
    def __init__(self, cutoff: dt.datetime | None = None) -> None:
        if cutoff is not None and cutoff.tzinfo is None:
            raise ValueError("cutoff must be tz-aware (UTC)")
        self.cutoff = cutoff
        self._books: dict[str, QuoteBook] = {}

    # ------------------------------------------------------------ truncation
    def _cut(self, df, col: str = "known_at"):
        return df if self.cutoff is None else df.filter(pl.col(col) <= _lit(self.cutoff))

    def _check_t(self, t: dt.datetime) -> None:
        if t.tzinfo is None:
            raise ValueError("t must be tz-aware (UTC)")
        if self.cutoff is not None and t > self.cutoff:
            raise ValueError(f"as_of({t}) is after this build's cutoff {self.cutoff}")

    # ------------------------------------------------------------ tables
    @cached_property
    def contracts(self) -> pl.DataFrame:
        """Every market listed by the cutoff, with its fields known at listing."""
        return self._cut(contracts())

    @cached_property
    def finals(self) -> pl.DataFrame:
        """Final listing fields; rows (per field group) known after the cutoff are null."""
        f = loaders.load_listing_final()
        if self.cutoff is None:
            return f
        c = _lit(self.cutoff)
        closed = pl.col("close_time") <= c
        settled = pl.col("known_at") <= c
        return f.with_columns(
            pl.when(closed).then(pl.col("close_time")).alias("close_time"),
            *(pl.when(settled).then(pl.col(x)).alias(x) for x in loaders.FINAL_COLS if x != "close_time"),
            pl.when(settled).then(pl.col("known_at")).alias("known_at"))

    @cached_property
    def results(self) -> pl.DataFrame:
        """Game results (final score, actual start) known by the cutoff."""
        r = loaders.game_results()
        return self._cut(r)

    @cached_property
    def external_results(self) -> pl.DataFrame:
        """Finished games from the leagues' own schedules (for games Kalshi did not list)."""
        return self._cut(loaders.external_results())

    def pbp_all(self, league: str) -> pl.DataFrame:
        return self._cut(loaders.load_pbp(league))

    @cached_property
    def games(self) -> pl.DataFrame:
        """The schedule (known in advance), for games whose markets listed by the cutoff."""
        g = loaders.load_games()
        return g if self.cutoff is None else g.filter(pl.col("key").is_in(
            self.contracts["key"].unique().implode()))

    # ------------------------------------------------------------ quotes
    def book(self, series: str) -> QuoteBook:
        if series not in self._books:
            self._books[series] = QuoteBook(self._cut(scan_quotes(series)).collect())
        return self._books[series]

    def scan_quotes(self, series: str | list[str]) -> pl.LazyFrame:
        return self._cut(scan_quotes(series))

    def quotes_at(self, at: pl.DataFrame, t_col: str = "t", max_spread: float = MAX_SPREAD,
                  max_age: dt.timedelta = MAX_AGE) -> pl.DataFrame:
        """Latest quote at or before each row's t, for a frame with ``ticker`` and ``t_col``."""
        if self.cutoff is not None and at.height and at[t_col].max() > self.cutoff:
            raise ValueError("quotes_at: a requested time is after the cutoff")
        series = (at.select("ticker").unique()
                  .join(self.contracts.select("ticker", "series_ticker"), on="ticker")["series_ticker"]
                  .unique().to_list())
        return quotes_at(self.scan_quotes(series), at, t_col, max_spread, max_age)

    def as_of(self, t: dt.datetime) -> "AsOfView":
        self._check_t(t)
        return AsOfView(self, t)


class AsOfView:
    """What was known at one time t."""

    def __init__(self, data: SportsData, t: dt.datetime) -> None:
        self.data, self.t = data, t

    def markets(self, open_only: bool = True) -> pl.DataFrame:
        """Markets listed by t, with ``closed_at`` / ``result`` filled only once known."""
        t = _lit(self.t)
        c = self.data.contracts.filter(pl.col("open_time") <= t)
        f = self.data.finals.select("ticker", "close_time", "result", pl.col("known_at").alias("settled_at"))
        c = c.join(f, on="ticker", how="left").with_columns(
            pl.when(pl.col("close_time") <= t).then(pl.col("close_time")).alias("closed_at"),
            pl.when(pl.col("settled_at") <= t).then(pl.col("result")).alias("result"),
            pl.when(pl.col("settled_at") <= t).then(pl.col("settled_at")).alias("settled_at"),
        ).drop("close_time")
        return c.filter(pl.col("closed_at").is_null()) if open_only else c

    def quote(self, ticker: str, max_spread: float = MAX_SPREAD, max_age: dt.timedelta = MAX_AGE) -> dict | None:
        series = self.data.contracts.filter(pl.col("ticker") == ticker)["series_ticker"]
        if series.is_empty():
            return None
        return self.data.book(series[0]).at(ticker, self.t, max_spread, max_age)

    def quotes(self, tickers: list[str], **kw) -> pl.DataFrame:
        at = pl.DataFrame({"ticker": tickers}).with_columns(_lit(self.t).alias("t"))
        return self.data.quotes_at(at, **kw)

    def results(self) -> pl.DataFrame:
        return self.data.results.filter(pl.col("known_at") <= _lit(self.t))

    def pbp(self, league: str, game_id: str | None = None) -> pl.DataFrame:
        ev = self.data.pbp_all(league).filter(pl.col("known_at") <= _lit(self.t))
        return ev if game_id is None else ev.filter(pl.col("game_id") == game_id)
