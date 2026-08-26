from stg.io.kalshi import KalshiOHLCV
from stg.pairs.config import EventPairsConfig
from stg.pairs.kalshi_inputs import load_trades_and_markets
from stg.pairs.selection import select_event_universe

cfg = EventPairsConfig(
    min_event_days=10, top_events=300, lookback_sel=20, beta_lookback=10,
    z_window=10, corr_min=0.20, corr_max=0.95, half_life_min=1.0,
    half_life_max=10.0, top_pairs=50, entry_z=2.0, exit_z=0.5,
    max_hold=20, cost_per_pair_turn=0.01,
)

# --- October (training) side, same as run_event_pairs_arb does internally ---
trades, markets = load_trades_and_markets(
    "data/trades/trades_kalshi_even/trades_2025-10.parquet",
    "data/markets/markets_kalshi_even/markets_2025-10.parquet",
)
daily = KalshiOHLCV.build_daily(trades, markets)
event = KalshiOHLCV.aggregate_to_event_level(daily)
eprice = event.select(["event_ticker", "date", "implied_mean_raw", "log_volume_norm"])
oct_events = set(select_event_universe(
    eprice, min_event_days=cfg.min_event_days, top_events=cfg.top_events,
))
print(f"October selected {len(oct_events)} events")

# --- November (backtest) side ---
bt_trades, bt_markets = load_trades_and_markets(
    "data/trades/trades_kalshi_odd/trades_2025-11.parquet",
    "data/markets/markets_kalshi_odd/markets_2025-11.parquet",
)
daily_bt = KalshiOHLCV.build_daily(bt_trades, bt_markets)
nov_events = set(daily_bt["event_ticker"].unique().to_list())
print(f"November has {len(nov_events)} distinct event_tickers")

overlap = oct_events & nov_events
print(f"{len(overlap)} of October's {len(oct_events)} events also appear in November")
print("A few example October event_tickers:", list(oct_events)[:5])
print("A few example November event_tickers:", list(nov_events)[:5])