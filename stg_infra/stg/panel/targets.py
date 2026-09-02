"""Target-side responses: how a target series' price moves after a trigger
resolves.

Promotes ``sign_diagnostics.reps_for`` (pick one representative liquid ticker
per target event) and the response-window logic shared by
``liquid_window.py`` / ``sign_diagnostics.py`` / ``structure_discovery.py``.

Response windows (research_summary.md §5 / update_2026_08.md §5):

``dormant``  trigger resolution -> the 3rd subsequent trade in the target
             (median length ~0.5h; the primary window)
``liquid``   trigger resolution -> mean price over the target's final 7 days
             (reported as a robustness check; does not survive the corrected null)
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import polars as pl

from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import series_filter_expr

LIQUID_DAYS = 7
MIN_LIQUID_TRADES = 3
MAX_GAP_DAYS = 60


def _side_expr() -> pl.Expr:
    """FEDDECISION side from the ticker suffix: -H* hike, -C* cut, -H0 hold."""
    return (pl.when(pl.col("ticker").str.contains(r"-H0$")).then(pl.lit("hold"))
            .when(pl.col("ticker").str.contains(r"-H\d")).then(pl.lit("hike"))
            .when(pl.col("ticker").str.contains(r"-C\d")).then(pl.lit("cut"))
            .otherwise(pl.lit("any")).alias("side"))


def representative_tickers(
    target: str,
    side: str = "any",
    *,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """One row per target event: the most-traded ticker and its close time.

    Columns: ``event_ticker, ticker, close_time, n_trades``.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    m = (mk.filter(series_filter_expr(target) & pl.col("close_time").is_not_null())
         .select("event_ticker", "ticker", "close_time")
         .with_columns(_side_expr()))
    if side != "any":
        m = m.filter(pl.col("side") == side)
    tk = m["ticker"].unique().to_list()
    counts = (tr.filter(pl.col("ticker").is_in(tk)).group_by("ticker")
              .agg(pl.len().alias("n_trades")).collect())
    m = m.join(counts, on="ticker", how="left").with_columns(
        pl.col("n_trades").fill_null(0))
    return (m.filter(pl.col("n_trades") > 0)
            .sort("n_trades", descending=True)
            .group_by("event_ticker")
            .agg(pl.col("ticker").first(), pl.col("close_time").first(),
                 pl.col("n_trades").first())
            .sort("close_time"))


def response_panel(
    surprise_panel: pl.DataFrame,
    target: str,
    side: str = "any",
    *,
    horizon: str = "dormant",
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """Match every trigger event in ``surprise_panel`` to the next target event
    and measure the target's price response.

    Returns one row per matched pair:
    ``trigger, target, side, event(trigger), target_event, surprise,
    gap_days, p0, response, days_to_close(target at match)``.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    reps = representative_tickers(target, side, markets=mk, trades=tr)
    if reps.height == 0:
        return pl.DataFrame()
    rep_rows = list(zip(reps["close_time"].to_list(), reps["ticker"].to_list(),
                        reps["event_ticker"].to_list()))
    rep_tickers = reps["ticker"].unique().to_list()
    tt = (tr.filter(pl.col("ticker").is_in(rep_tickers))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())

    out: list[dict] = []
    for r in surprise_panel.iter_rows(named=True):
        t_res = r["close_time"]
        nxt = [row for row in rep_rows if row[0] > t_res]
        if not nxt:
            continue
        c_close, c_ticker, c_event = min(nxt, key=lambda x: x[0])
        gap = (c_close - t_res).total_seconds() / 86400
        if not (0 <= gap <= MAX_GAP_DAYS):
            continue
        sub = tt.filter(pl.col("ticker") == c_ticker)
        pre = sub.filter(pl.col("created_time") <= t_res)
        if pre.height == 0:
            continue
        p0 = float(pre["yes_price"][-1])
        if horizon == "dormant":
            post = sub.filter(pl.col("created_time") > t_res)
            if post.height < 3:
                continue
            p1 = float(post["yes_price"][2])
        elif horizon == "liquid":
            lw = sub.filter(
                (pl.col("created_time") >= c_close - dt.timedelta(days=LIQUID_DAYS))
                & (pl.col("created_time") <= c_close))
            if lw.height < MIN_LIQUID_TRADES:
                continue
            p1 = float(lw["yes_price"].mean())
        else:
            raise ValueError(f"unknown horizon {horizon!r}")
        out.append(dict(
            trigger=r["series"], target=target, side=side,
            trigger_event=r.get("event_ticker"), target_event=c_event,
            surprise=float(r["surprise"]), gap_days=gap,
            p0=p0, response=p1 - p0,
            days_to_close=(c_close - t_res).days,
        ))
    return pl.DataFrame(out)
