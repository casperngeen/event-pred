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
    """The per-*target* work: ``(representative tickers, their trades)``.

    Depends only on ``(target, side)``, never on the trigger, so a sweep over
    many triggers should build it once. Hoisted out of :func:`response_panel`
    because leaving it inside made the grid quadratic in the wrong variable:
    the structure estimator calls ``response_panel`` once per (trigger,
    target) pair, so every one of the 21 triggers rebuilt the same frames.
    Harmless while targets were macro ladders of a few hundred tickers;
    crippling once INXU (15,289 tickers, 646k trades) joined the target
    universe, where a single rebuild costs ~2 minutes.

    Pass the result back via ``response_panel(..., frames=...)``.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    reps = representative_tickers(target, side, markets=mk, trades=tr)
    if reps.height == 0:
        return reps, pl.DataFrame(
            schema={"ticker": pl.Utf8, "yes_price": pl.Float64,
                    "created_time": reps.schema.get("close_time", pl.Datetime)})
    rep_tickers = reps["ticker"].unique().to_list()
    tt = (tr.filter(pl.col("ticker").is_in(rep_tickers))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    return reps, tt


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
    reps, tt = frames
    if reps.height == 0:
        return pl.DataFrame()
    if match not in ("live", "next_close"):
        raise ValueError(f"unknown match {match!r}")
    rep_rows = sorted(zip(reps["close_time"].to_list(), reps["ticker"].to_list(),
                          reps["event_ticker"].to_list()))

    # First trade per ticker, so "was this contract already trading when the
    # trigger resolved?" is a dict lookup rather than a filter per candidate.
    if tt.height:
        _ft = tt.group_by("ticker").agg(pl.col("created_time").min().alias("t0"))
        first_trade = dict(zip(_ft["ticker"].to_list(), _ft["t0"].to_list()))
    else:
        first_trade = {}

    def _pick(t_res):
        """The target event this trigger is matched to.

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
        """
        for c_close, c_ticker, c_event in rep_rows:
            if c_close <= t_res:
                continue
            gap = (c_close - t_res).total_seconds() / 86400
            if gap > MAX_GAP_DAYS:
                return None
            if match == "next_close":
                return c_close, c_ticker, c_event, gap
            ft = first_trade.get(c_ticker)
            if ft is not None and ft <= t_res:
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
