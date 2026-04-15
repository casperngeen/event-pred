from __future__ import annotations

from typing import Dict

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
    tdates, tseries = build_wide_ticker_close_panel(daily, tickers_needed)

    equity = backtest_zscore_pairs(tdates, tseries, ticker_pairs, cfg)

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