from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import polars as pl

from stg.pairs.config import EventPairsConfig
from stg.pairs.stats import rolling_beta, rolling_z


def representative_tickers(
    daily: pl.DataFrame,
    events: Sequence[str],
    *,
    start_date,
    end_date,
) -> pl.DataFrame:
    daily_sel = (
        daily.filter(pl.col("event_ticker").is_in(list(events)))
        .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
    )

    rep = (
        daily_sel.group_by(["event_ticker", "ticker"])
        .agg(pl.col("volume").mean().alias("avg_daily_volume"))
        .sort(["event_ticker", "avg_daily_volume"], descending=[False, True])
        .group_by("event_ticker")
        .agg([
            pl.col("ticker").first().alias("rep_ticker"),
            pl.col("avg_daily_volume").first().alias("rep_avg_daily_volume"),
        ])
    )
    return rep


def map_event_pairs_to_ticker_pairs(
    event_pairs: Sequence[Tuple[str, str]],
    rep_df: pl.DataFrame,
) -> Tuple[List[Tuple[str, str]], pl.DataFrame]:
    rep_map: Dict[str, str] = dict(zip(rep_df["event_ticker"].to_list(), rep_df["rep_ticker"].to_list()))

    rows = []
    ticker_pairs: List[Tuple[str, str]] = []

    for e1, e2 in event_pairs:
        t1 = rep_map.get(e1)
        t2 = rep_map.get(e2)
        if t1 is None or t2 is None or t1 == t2:
            continue
        ticker_pairs.append((t1, t2))
        rows.append({"event_A": e1, "event_B": e2, "ticker_A": t1, "ticker_B": t2})

    mapping = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame({"event_A": [], "event_B": [], "ticker_A": [], "ticker_B": []})
    )
    return ticker_pairs, mapping


def build_wide_ticker_close_panel(
    daily: pl.DataFrame,
    tickers: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    wide_t = (
        daily.filter(pl.col("ticker").is_in(list(tickers)))
        .select(["date", "ticker", "close"])
        .pivot(index="date", on="ticker", values="close", aggregate_function="first")
        .sort("date")
    )

    tdates = wide_t["date"].to_numpy()
    tcols = [c for c in wide_t.columns if c != "date"]
    tseries = {c: wide_t[c].to_numpy() for c in tcols}
    return tdates, tseries


def backtest_zscore_pairs(
    tdates: np.ndarray,
    tseries: Dict[str, np.ndarray],
    ticker_pairs: Sequence[Tuple[str, str]],
    cfg: EventPairsConfig,
) -> pl.DataFrame:
    pos = {tp: {"dir": 0, "entry_i": None} for tp in ticker_pairs}
    pnl = np.zeros(len(tdates), dtype=float)

    for t in range(len(tdates)):
        for (A, B) in ticker_pairs:
            a = tseries[A][:t + 1]
            b = tseries[B][:t + 1]

            if len(a) < max(cfg.beta_lookback, cfg.z_window) + 2:
                continue
            if not (np.isfinite(a[-cfg.beta_lookback:]).all() and np.isfinite(b[-cfg.beta_lookback:]).all()):
                continue

            beta = rolling_beta(a, b, cfg.beta_lookback)
            if not np.isfinite(beta):
                continue

            spread = a - beta * b
            z = rolling_z(spread, cfg.z_window)
            if not np.isfinite(z):
                continue

            st = pos[(A, B)]

            if t >= 1:
                pnl[t] += st["dir"] * (float(spread[-1]) - float(spread[-2]))

            if st["dir"] != 0 and st["entry_i"] is not None and (t - st["entry_i"]) >= cfg.max_hold:
                st["dir"] = 0
                st["entry_i"] = None
                pnl[t] -= cfg.cost_per_pair_turn

            if st["dir"] == 0:
                if z >= cfg.entry_z:
                    st["dir"] = -1
                    st["entry_i"] = t
                    pnl[t] -= cfg.cost_per_pair_turn
                elif z <= -cfg.entry_z:
                    st["dir"] = +1
                    st["entry_i"] = t
                    pnl[t] -= cfg.cost_per_pair_turn
            else:
                if abs(z) <= cfg.exit_z:
                    st["dir"] = 0
                    st["entry_i"] = None
                    pnl[t] -= cfg.cost_per_pair_turn

    return pl.DataFrame({"date": tdates, "pnl": pnl}).with_columns(
        pl.col("pnl").cum_sum().alias("equity")
    )
    