from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

from stg.io.kalshi import KalshiOHLCV
from stg.pairs.config import EventPairsConfig
from stg.pairs.kalshi_inputs import load_trades_and_markets
from stg.pairs.selection import build_event_panel, select_event_pairs, select_event_universe
from stg.pairs.tradable_proxy import (
    backtest_zscore_pairs,
    build_wide_ticker_close_panel,
    map_event_pairs_to_ticker_pairs,
    representative_tickers,
)


def run_event_pairs_arb(
    *,
    trades_glob: str,
    markets_glob: str,
    cfg: EventPairsConfig,
    backtest_trades_glob: Optional[str] = None,
    backtest_markets_glob: Optional[str] = None,
) -> Dict[str, object]:
    trades, markets = load_trades_and_markets(trades_glob, markets_glob)

    daily = KalshiOHLCV.build_daily(trades, markets)
    event = KalshiOHLCV.aggregate_to_event_level(daily)
    eprice = event.select(["event_ticker", "date", "implied_mean_raw", "log_volume_norm"])

    events = select_event_universe(
        eprice,
        min_event_days=cfg.min_event_days,
        top_events=cfg.top_events,
    )

    dates, ecols, EX = build_event_panel(eprice, events)
    event_pairs, diagnostics = select_event_pairs(dates, ecols, EX, cfg)
    if not event_pairs:
        raise RuntimeError("No event pairs selected; relax filters or increase lookback.")

    # Representative tickers chosen over the selection window
    start_date = dates[-cfg.lookback_sel]
    end_date = dates[-1]
    rep_tickers = representative_tickers(daily, events, start_date=start_date, end_date=end_date)

    ticker_pairs, pair_mapping = map_event_pairs_to_ticker_pairs(event_pairs, rep_tickers)
    if not ticker_pairs:
        raise RuntimeError("No ticker pairs mapped from event pairs (rep tickers missing or same ticker).")

    tickers_needed = sorted({t for a, b in ticker_pairs for t in (a, b)})

    if backtest_trades_glob is not None or backtest_markets_glob is not None:
        if backtest_trades_glob is None or backtest_markets_glob is None:
            raise ValueError(
                "backtest_trades_glob and backtest_markets_glob must be given together."
            )

        bt_trades, bt_markets = load_trades_and_markets(
            backtest_trades_glob,
            backtest_markets_glob
        )

        daily_bt = KalshiOHLCV.build_daily(bt_trades, bt_markets)

        # Representative tickers are re-resolved over the backtest window itself,
        # rather than reusing rep_tickers/ticker_pairs from the training window
        # above -- a ticker picked as "representative" for an event in training
        # typically won't exist at all in a later, disjoint backtest window.
        bt_start_date = daily_bt["date"].min()
        bt_end_date = daily_bt["date"].max()
        rep_tickers_bt = representative_tickers(
            daily_bt, events, start_date=bt_start_date, end_date=bt_end_date
        )
        ticker_pairs_bt, pair_mapping_bt = map_event_pairs_to_ticker_pairs(
            event_pairs, rep_tickers_bt
        )

        if not ticker_pairs_bt:
            raise RuntimeError(
                f"0 of {len(event_pairs)} event pairs have a representative ticker "
                f"tradable during the backtest window ({bt_start_date}..{bt_end_date}). "
                "The events selected in the training window most likely resolved/"
                "expired before the backtest period started -- try a backtest period "
                "closer to the training window, or restrict event selection to events "
                "with longer typical lifetimes."
            )

        tickers_needed_bt = sorted({t for a, b in ticker_pairs_bt for t in (a, b)})
        tdates, tseries = build_wide_ticker_close_panel(daily_bt, tickers_needed_bt)

        if len(tdates) == 0:
            raise RuntimeError(
                "Backtest ticker panel has 0 rows even though "
                f"{len(ticker_pairs_bt)} ticker pair(s) were resolved for the backtest "
                f"window ({bt_start_date}..{bt_end_date}) -- check that daily_bt actually "
                "has price data (not just ticker matches) in that range."
            )

        in_sample = np.zeros(len(tdates), dtype=bool)
        bt_ticker_pairs = ticker_pairs_bt
        bt_pair_mapping = pair_mapping_bt

    else:
        logger.warning(
            "run_event_pairs_arb: no backtest_trades_glob/backtest_markets_glob given — "
            "backtesting on the same data used for pair selection. This is in-sample and "
            "will overstate performance."
        )

        daily_bt = daily
        tdates, tseries = build_wide_ticker_close_panel(
            daily,
            tickers_needed
        )

        in_sample = tdates <= end_date
        bt_ticker_pairs = ticker_pairs
        bt_pair_mapping = pair_mapping


    equity = backtest_zscore_pairs(tdates, tseries, bt_ticker_pairs, cfg)

    equity = equity.with_columns(
        pl.Series("in_sample", in_sample)
    )

    return {
        "daily": daily,
        "daily_bt": daily_bt,
        "event": event,
        "events": events,
        "event_pairs": event_pairs,
        "ticker_pairs": bt_ticker_pairs,
        "diagnostics": diagnostics,
        "rep_tickers": rep_tickers,
        "pair_mapping": bt_pair_mapping,
        "equity": equity,
        # (start_date, end_date) of the training/selection window rep_tickers
        # and event_pairs were chosen over.
        "selection_window": (start_date, end_date),
    }