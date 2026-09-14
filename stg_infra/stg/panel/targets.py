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

import numpy as np
import polars as pl

from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import series_filter_expr

LIQUID_DAYS = 7
MIN_LIQUID_TRADES = 3
MAX_GAP_DAYS = 60


def _group_times(tt: pl.DataFrame):
    """``(tickers, sorted epoch-ns arrays)`` for binary-search lookups.

    Epoch integers rather than ``datetime64``: numpy has no tz-aware dtype, so
    converting a UTC-aware polars column to ``datetime64[ns]`` warns and relies
    on both sides being silently naive-ised the same way. Comparing int64
    nanoseconds is unambiguous.
    """
    g = (tt.sort("ticker", "created_time")
         .with_columns(pl.col("created_time").dt.epoch("ns").alias("_ns"))
         .group_by("ticker", maintain_order=True).agg(pl.col("_ns")))
    arrays = [np.asarray(v, dtype=np.int64) for v in g["_ns"].to_list()]
    return g["ticker"].to_list(), arrays


def _epoch_ns(ts) -> int:
    """UTC epoch nanoseconds for a tz-aware or naive datetime."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(round(ts.timestamp() * 1_000_000_000))


def _side_expr() -> pl.Expr:
    """FEDDECISION side from the ticker suffix: -H* hike, -C* cut, -H0 hold."""
    return (pl.when(pl.col("ticker").str.contains(r"-H0$")).then(pl.lit("hold"))
            .when(pl.col("ticker").str.contains(r"-H\d")).then(pl.lit("hike"))
            .when(pl.col("ticker").str.contains(r"-C\d")).then(pl.lit("cut"))
            .otherwise(pl.lit("any")).alias("side"))


def target_legs(
    target: str,
    side: str = "any",
    *,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """Every traded leg of every target event: ``event_ticker, ticker,
    close_time, n_trades``, one row per leg rather than per event.

    The candidate set. Which leg *represents* an event is decided at match time
    against the trigger's resolution instant, not here -- see
    :func:`response_panel`.
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
    return (m.join(counts, on="ticker", how="left")
            .with_columns(pl.col("n_trades").fill_null(0))
            .filter(pl.col("n_trades") > 0)
            .unique(subset=["event_ticker", "ticker"])
            .sort(["close_time", "event_ticker", "ticker"]))


def representative_tickers(
    target: str,
    side: str = "any",
    *,
    as_of: Optional[dt.datetime] = None,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """One row per target event: the most-traded ticker and its close time.

    Columns: ``event_ticker, ticker, close_time, n_trades``.

    ``as_of`` restricts the trade count to prints at or before that instant.
    **Leave it None only for descriptive use.** Without it the count runs over
    the event's whole life, including everything after the trigger fired, so the
    instrument the response is measured on is chosen with information from after
    the decision. Measured: the leg picked on full-life volume differs from the
    one picked on first-half volume for 56% of CPI events and 63% of WTI events.
    It does not bias the *direction* (the chosen leg settles YES 43-52% of the
    time, near the at-the-money 50% you would expect), which is why the
    published sign results survived it -- but it is still a look-ahead, and
    ``response_panel`` now avoids it by selecting per trigger instant.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    if as_of is not None:
        tr = tr.filter(pl.col("created_time") <= as_of)
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
    # ``ticker`` breaks ties in both sorts. Without it the most-traded ticker
    # for an event, and the order of equal-close_time events, depend on the
    # order polars happens to emit groups in — which made the panel differ by a
    # row between identical runs.
    return (m.filter(pl.col("n_trades") > 0)
            .sort(["n_trades", "ticker"], descending=[True, False])
            .group_by("event_ticker", maintain_order=True)
            .agg(pl.col("ticker").first(), pl.col("close_time").first(),
                 pl.col("n_trades").first())
            .sort(["close_time", "ticker"]))


def target_frames(
    target: str,
    side: str = "any",
    *,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The per-*target* work: ``(candidate legs, their trades)``.

    Depends only on ``(target, side)``, never on the trigger, so a sweep over
    many triggers should build it once. Hoisted out of :func:`response_panel`
    because leaving it inside made the grid quadratic in the wrong variable:
    the structure estimator calls ``response_panel`` once per (trigger,
    target) pair, so every one of the 21 triggers rebuilt the same frames.
    Harmless while targets were macro ladders of a few hundred tickers;
    crippling once INXU (15,289 tickers, 646k trades) joined the target
    universe, where a single rebuild costs ~2 minutes.

    Pass the result back via ``response_panel(..., frames=...)``.

    Returns **all** traded legs, not one per event. Picking the representative
    leg needs the trigger's resolution instant (see
    :func:`representative_tickers` on ``as_of``), which is not known here -- so
    the trigger-independent half of the work stays cached and the per-trigger
    choice happens in :func:`response_panel`. The cost is carrying every leg's
    prints rather than one leg's: 646k rows for INXU, which fits.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    legs = target_legs(target, side, markets=mk, trades=tr)
    if legs.height == 0:
        return legs, pl.DataFrame(
            schema={"ticker": pl.Utf8, "yes_price": pl.Float64,
                    "created_time": legs.schema.get("close_time", pl.Datetime)})
    leg_tickers = legs["ticker"].unique().to_list()
    tt = (tr.filter(pl.col("ticker").is_in(leg_tickers))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    return legs, tt


def response_panel(
    surprise_panel: pl.DataFrame,
    target: str,
    side: str = "any",
    *,
    horizon: str = "dormant",
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
    frames: Optional[tuple[pl.DataFrame, pl.DataFrame]] = None,
    match: str = "live",
) -> pl.DataFrame:
    """Match every trigger event in ``surprise_panel`` to a target event
    and measure the target's price response.

    Returns one row per matched pair:
    ``trigger, target, side, event(trigger), target_event, surprise,
    gap_days, p0, response, days_to_close(target at match)``, plus the
    execution columns ``target_ticker, t_p0, n_pre, t_entry, p_entry, t_exit``.

    ``p0`` is the last trade *before* the trigger resolved — the reference the
    structure estimator uses, but a price no one can still transact at once the
    trigger has resolved. ``p_entry``/``t_entry`` are the first print after
    resolution: the earliest a trader could actually have acted. The gap
    between the two is where update_2026_08.md §5's "48% of the move is in the
    first print" shows up as an execution cost rather than a return.

    ``t_p0`` dates the reference price, so how *stale* it was at the trigger's
    resolution is measurable rather than assumed — research_summary.md §4.1's
    central threat, since a p0 that is days old makes ``p1 - p0`` partly a
    staleness correction rather than a response.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    if frames is None:
        frames = target_frames(target, side, markets=mk, trades=tr)
    legs, tt = frames
    if legs.height == 0:
        return pl.DataFrame()
    if match not in ("live", "next_close"):
        raise ValueError(f"unknown match {match!r}")

    # Events in close order, each carrying its candidate legs. One entry per
    # event, so the walk below is over events and the leg choice happens inside.
    ev_rows: dict = {}
    for c_close, c_event, c_ticker in zip(legs["close_time"].to_list(),
                                          legs["event_ticker"].to_list(),
                                          legs["ticker"].to_list()):
        ev_rows.setdefault((c_close, c_event), []).append(c_ticker)
    event_order = sorted(ev_rows)

    # Per-leg print times, sorted, so "how many prints had this leg seen by
    # t_res" is a binary search rather than a filter.
    times: dict = {}
    if tt.height:
        for tk, g in zip(*_group_times(tt)):
            times[tk] = g

    def _best_leg(tickers, t_res):
        """The leg that represents this event at ``t_res``.

        The most-traded leg **among prints at or before the trigger resolved**.
        Counting over the event's whole life instead -- what this did before --
        chooses the instrument using trades that happen after the decision; for
        CPI that picks a different leg 56% of the time. Ties break on ticker
        name so identical rebuilds agree.

        Returns ``(ticker, n_pre)``; ``n_pre == 0`` means no leg had traded yet,
        so there is no ``p0`` and the observation is not usable.
        """
        cut = _epoch_ns(t_res)
        best, best_n = None, 0
        for tk in sorted(tickers):
            ts = times.get(tk)
            if ts is None or len(ts) == 0:
                continue
            n = int(np.searchsorted(ts, cut, side="right"))
            if n > best_n:
                best, best_n = tk, n
        return best, best_n

    def _pick(t_res):
        """The target event this trigger is matched to, and the leg to use.

        ``next_close`` is the original rule: the earliest-closing event after
        the trigger, take it or leave it. ``live`` walks candidates in close
        order and takes the first that was *already trading* at ``t_res``.

        They agree whenever the nearest event is already open, which is the
        normal case for macro ladders that live for weeks. They diverge for
        daily-cadence targets, where the next event to close has often only
        just opened -- no pre-resolution trade, so no ``p0``, and
        ``next_close`` discards the observation rather than looking past it.
        That is why 3,423 INXU/INXD/NASDAQ100U events yielded 9 usable pairs
        and zero rows in the ``bh`` gate.

        "Already trading" is now the same condition as "has a usable leg":
        ``_best_leg`` returns a leg only if it has a pre-``t_res`` print, which
        is exactly what ``p0`` needs.
        """
        for c_close, c_event in event_order:
            if c_close <= t_res:
                continue
            gap = (c_close - t_res).total_seconds() / 86400
            if gap > MAX_GAP_DAYS:
                return None
            c_ticker, n_pre = _best_leg(ev_rows[(c_close, c_event)], t_res)
            if match == "next_close":
                if c_ticker is None:
                    return None
                return c_close, c_ticker, c_event, gap
            if c_ticker is not None:
                return c_close, c_ticker, c_event, gap
        return None

    out: list[dict] = []
    for r in surprise_panel.iter_rows(named=True):
        t_res = r["close_time"]
        picked = _pick(t_res)
        if picked is None:
            continue
        c_close, c_ticker, c_event, gap = picked
        sub = tt.filter(pl.col("ticker") == c_ticker)
        pre = sub.filter(pl.col("created_time") <= t_res)
        if pre.height == 0:
            continue
        p0 = float(pre["yes_price"][-1])
        t_p0 = pre["created_time"][-1]
        n_pre = pre.height
        post = sub.filter(pl.col("created_time") > t_res)
        t_entry = post["created_time"][0] if post.height else None
        p_entry = float(post["yes_price"][0]) if post.height else None
        if horizon == "dormant":
            if post.height < 3:
                continue
            p1 = float(post["yes_price"][2])
            t_exit = post["created_time"][2]
        elif horizon == "liquid":
            lw = sub.filter(
                (pl.col("created_time") >= c_close - dt.timedelta(days=LIQUID_DAYS))
                & (pl.col("created_time") <= c_close))
            if lw.height < MIN_LIQUID_TRADES:
                continue
            p1 = float(lw["yes_price"].mean())
            t_exit = c_close
        else:
            raise ValueError(f"unknown horizon {horizon!r}")
        out.append(dict(
            trigger=r["series"], target=target, side=side,
            trigger_event=r.get("event_ticker"), target_event=c_event,
            surprise=float(r["surprise"]), gap_days=gap,
            p0=p0, response=p1 - p0,
            days_to_close=(c_close - t_res).days,
            target_ticker=c_ticker, t_p0=t_p0, n_pre=n_pre,
            t_entry=t_entry, p_entry=p_entry, t_exit=t_exit,
        ))
    return pl.DataFrame(out)
