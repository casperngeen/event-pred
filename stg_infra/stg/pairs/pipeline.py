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

        tdates, tseries = build_wide_ticker_close_panel(
            daily_bt,
            tickers_needed
        )

        in_sample = np.zeros(len(tdates), dtype=bool)

    else:
        logger.warning(
            "run_event_pairs_arb: no backtest_trades_glob/backtest_markets_glob given — "
            "backtesting on the same data used for pair selection. This is in-sample and "
            "will overstate performance."
        )

        tdates, tseries = build_wide_ticker_close_panel(
            daily,
            tickers_needed
        )

        in_sample = tdates <= end_date


    equity = backtest_zscore_pairs(tdates, tseries, ticker_pairs, cfg)

    equity = equity.with_columns(
        pl.Series("in_sample", in_sample)
    )

    return {
        "daily": daily,
        "event": event,
        "events": events,
        "event_pairs": event_pairs,
        "ticker_pairs": ticker_pairs,
        "diagnostics": diagnostics,
        "rep_tickers": rep_tickers,
        "pair_mapping": pair_mapping,
        "equity": equity,
    }