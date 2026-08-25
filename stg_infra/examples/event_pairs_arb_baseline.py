from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl
import pandas as pd
import numpy as np

from stg_infra.stg.pairs.config import EventPairsConfig
from stg_infra.stg.pairs.pipeline import run_event_pairs_arb
from stg_infra.stg.pairs.metrics.report import build_full_metrics_report

# --------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------
TRADES_GLOB = "data/trades/trades_kalshi_even/trades_2025-10.parquet"
MARKETS_GLOB = "data/markets/markets_kalshi_even/markets_2025-10.parquet"
METRICS_OUT_DIR = "artifacts/event_pairs_arb_baseline"

# cost_per_pair_turn was 0.0 in the original script. A Sharpe of 5.5+ with
# ~3% max drawdown on a first baseline run is a strong signal that zero-cost
# backtesting is flattering the result, on top of the 0-pairs metrics bug.
# Replace this placeholder with your real estimated round-trip cost, then
# compare against a 0.0 run to see how much of the "edge" is unmodeled cost.
REALISTIC_COST_PER_PAIR_TURN = 0.01  # <-- placeholder, replace with your real estimate

cfg = EventPairsConfig(
    min_event_days=10, top_events=300, lookback_sel=20, beta_lookback=10,
    z_window=10, corr_min=0.20, corr_max=0.95, half_life_min=1.0,
    half_life_max=10.0, top_pairs=50, entry_z=2.0, exit_z=0.5,
    max_hold=20, cost_per_pair_turn=REALISTIC_COST_PER_PAIR_TURN,
)


def _pick_first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _inspect_schema(name: str, df: pd.DataFrame) -> None:
    print(f"[DIAG] {name} columns: {list(df.columns)}")
    print(f"[DIAG] {name} shape: {df.shape}")
    if len(df) > 0:
        print(f"[DIAG] {name} sample row:\n{df.iloc[0].to_dict()}")


def _compute_pair_spreads_from_daily(
    daily_pl: pl.DataFrame,
    ticker_pairs: List[Tuple[str, str]],
    pair_mapping_pl: pl.DataFrame,
    verbose: bool = True,
) -> Dict[str, pd.Series]:
    """
    Builds spreads directly from the pipeline's own `daily` panel and
    `ticker_pairs` -- the exact data the backtest traded on -- instead of
    reloading and re-deriving prices from the raw markets file. This
    guarantees ticker alignment by construction, since these are the same
    objects run_event_pairs_arb used internally to build tseries/backtest.

    NOTE: column names below (`ticker`, `date`, and the price column) are
    best guesses based on the class name `KalshiOHLCV` / `build_daily`.
    Run this once, check the [DIAG] schema printout, and adjust
    PRICE_COL / TICKER_COL / DATE_COL below if they don't match.
    """
    ddf = daily_pl.to_pandas()

    if verbose:
        _inspect_schema("daily", ddf)

    TICKER_COL = _pick_first(ddf.columns, ["ticker", "market_ticker"])
    DATE_COL = _pick_first(ddf.columns, ["date", "created_time", "datetime", "timestamp"])
    PRICE_COL = _pick_first(
        ddf.columns,
        ["close", "px", "last_price", "implied_mean_raw", "mid", "yes_close"],
    )

    if TICKER_COL is None or DATE_COL is None or PRICE_COL is None:
        raise ValueError(
            f"Could not confidently identify ticker/date/price columns in `daily`. "
            f"Columns present: {list(ddf.columns)}. "
            f"Set TICKER_COL / DATE_COL / PRICE_COL explicitly based on this list."
        )

    if verbose:
        print(f"[DIAG] Using TICKER_COL='{TICKER_COL}', DATE_COL='{DATE_COL}', PRICE_COL='{PRICE_COL}'")

    ddf[DATE_COL] = pd.to_datetime(ddf[DATE_COL], errors="coerce")
    pivot = (
        ddf.sort_values(DATE_COL)
        .groupby([TICKER_COL, DATE_COL])[PRICE_COL]
        .last()
        .unstack(level=0)
    )

    if verbose:
        print(f"[DIAG] Pivot built from daily: {pivot.shape[1]} unique tickers, {pivot.shape[0]} dates")

    # ticker_pairs is the pipeline's own list of (ticker_A, ticker_B) actually
    # traded -- use this directly rather than re-deriving from pair_mapping.
    pm_df = pair_mapping_pl.to_pandas()
    event_a_col = _pick_first(pm_df.columns, ["event_A", "event_a"])
    event_b_col = _pick_first(pm_df.columns, ["event_B", "event_b"])

    if verbose:
        _inspect_schema("pair_mapping", pm_df)
        print(f"[DIAG] ticker_pairs sample (first 5): {ticker_pairs[:5]}")

    res: Dict[str, pd.Series] = {}
    n_missing = 0
    missing_examples = []

    for i, (tk_a, tk_b) in enumerate(ticker_pairs):
        a_present = tk_a in pivot.columns
        b_present = tk_b in pivot.columns

        if not (a_present and b_present):
            n_missing += 1
            if len(missing_examples) < 5:
                missing_examples.append((tk_a, a_present, tk_b, b_present))
            continue

        s = (pivot[tk_a] - pivot[tk_b]).dropna()
        if len(s) >= 2:
            # label using event ids if pair_mapping has them and rows align,
            # otherwise fall back to raw ticker pair as the key
            if event_a_col and event_b_col and i < len(pm_df):
                label = f"{pm_df.iloc[i][event_a_col]}__{pm_df.iloc[i][event_b_col]}"
            else:
                label = f"{tk_a}__{tk_b}"
            res[label] = s

    if verbose:
        print(f"[DIAG] Missing ticker matches: {n_missing}/{len(ticker_pairs)}")
        if missing_examples:
            print("[DIAG] First few unmatched examples (ticker_A, found_A, ticker_B, found_B):")
            for ex in missing_examples:
                print(f"        {ex}")
        print(f"[DIAG] Successfully built spreads for {len(res)}/{len(ticker_pairs)} ticker pairs")

    if len(res) == 0:
        raise RuntimeError(
            "0 pairs recovered even from the pipeline's own `daily` panel. "
            "This means ticker_pairs itself doesn't align with `daily`'s ticker "
            "column, which would be surprising since build_wide_ticker_close_panel "
            "uses the same `daily` + `tickers_needed` internally. Check the [DIAG] "
            "schema output above -- likely PRICE_COL/TICKER_COL/DATE_COL guessed "
            "wrong, or `daily` contains multiple ticker naming conventions."
        )

    return res


def _extract_equity_series(equity_df: pd.DataFrame) -> pd.Series:
    """Explicit, named-column equity extraction -- replaces positional
    `eq_df.iloc[:, -1]`, which silently breaks if column order changes."""
    _inspect_schema("equity", equity_df)

    time_col = _pick_first(equity_df.columns, ["date", "created_time", "datetime", "timestamp"])
    if time_col is None:
        raise ValueError(f"No time column found in equity output. Columns: {list(equity_df.columns)}")

    equity_col = _pick_first(
        equity_df.columns,
        ["equity", "cum_pnl", "cumulative_pnl", "portfolio_value", "nav", "pnl"],
    )
    if equity_col is None:
        raise ValueError(
            f"Could not find a recognizable equity column. Columns present: "
            f"{list(equity_df.columns)}. Update the candidate list to match."
        )

    print(f"[DIAG] Using time_col='{time_col}', equity_col='{equity_col}'")

    series = (100.0 + equity_df[equity_col]) / 100.0
    series.index = pd.to_datetime(equity_df[time_col], errors="coerce")
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    return series.sort_index()


# --------------------------------------------------------------------------
# 2. Run Pipeline
# --------------------------------------------------------------------------
print("Running Strategy...")
out = run_event_pairs_arb(
    trades_glob=TRADES_GLOB,
    markets_glob=MARKETS_GLOB,
    cfg=cfg,
    backtest_trades_glob="data/trades/trades_kalshi_even/trades_2025-11.parquet",
    backtest_markets_glob="data/markets/markets_kalshi_even/markets_2025-11.parquet",
)

in_sample_frac = out["equity"]["in_sample"].mean()

print(
    f"\nWARNING: {in_sample_frac:.0%} of the backtest window overlaps pair selection "
    f"(selection_window={out['selection_window']}) — treat this equity curve as in-sample."
)

# --------------------------------------------------------------------------
# 3. Process Equity (fixed: explicit named column, not positional index)
# --------------------------------------------------------------------------
equity_df = out["equity"].to_pandas() if hasattr(out["equity"], "to_pandas") else out["equity"]
equity_series = _extract_equity_series(equity_df)
returns = equity_series.pct_change().fillna(0.0)

# --------------------------------------------------------------------------
# 4. Recover Pair History (fixed: use the pipeline's own `daily` panel and
#    `ticker_pairs`, not a re-derived pivot from the raw markets file)
# --------------------------------------------------------------------------
print("Recovering Price History for Metrics...")
pair_to_spread = _compute_pair_spreads_from_daily(
    out["daily"], out["ticker_pairs"], out["pair_mapping"]
)
print(f"[DEBUG] Metrics successfully recovered price history for {len(pair_to_spread)} pairs.")

n_traded_pairs = len(out["ticker_pairs"])
n_recovered_pairs = len(pair_to_spread)
if n_recovered_pairs < n_traded_pairs:
    print(
        f"[WARN] Only recovered {n_recovered_pairs}/{n_traded_pairs} traded pairs' "
        f"spread history. stat_arb diagnostics below will undercount coverage."
    )

# --------------------------------------------------------------------------
# 5. Build Metadata
# --------------------------------------------------------------------------
event_pairs = [(str(a), str(b)) for (a, b) in out["event_pairs"]]
dated_edge_sets = [(ts, set(event_pairs)) for ts in equity_series.index]

pm = out["pair_mapping"].to_pandas()
cat_col = _pick_first(pm.columns, ["category", "event_category", "sector"])
event_a_col = _pick_first(pm.columns, ["event_A", "event_a"])
if cat_col and event_a_col:
    node_to_cat = dict(zip(pm[event_a_col].astype(str), pm[cat_col].astype(str)))
else:
    print("[WARN] No category column found in pair_mapping -- all nodes labeled 'General'.")
    node_to_cat = {str(a): "General" for a, b in event_pairs}

print(f"\n[DIAG] cost_per_pair_turn used this run: {REALISTIC_COST_PER_PAIR_TURN}")
print("[DIAG] Re-run with cost_per_pair_turn=0.0 and compare Sharpe/PnL to see "
      "how much of the edge is unmodeled transaction cost.\n")

# --------------------------------------------------------------------------
# 6. Report Generation
# --------------------------------------------------------------------------
Path(METRICS_OUT_DIR).mkdir(parents=True, exist_ok=True)
build_full_metrics_report(
    out_dir=METRICS_OUT_DIR,
    baseline_daily_returns=returns,
    baseline_trade_pnls=returns,
    baseline_pair_to_spread=pair_to_spread,
    baseline_dated_edge_sets=dated_edge_sets,
    node_to_category=node_to_cat,
    ann_factor=252,
    save_plots=True,
)

print(f"\nFinal Analysis Complete. Final Equity: {equity_series.iloc[-1]:.4f}")