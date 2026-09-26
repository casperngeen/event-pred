"""
mece_liquidity_check.py

mece_sum_to_one_check.py's "top 10 largest-magnitude snapshots" table
ranks events by |deviation|, computed from the LAST trade price per leg
per day -- but daily_last_trade's aggregation only keeps the last price
and last trade time, discarding how many trades actually happened. A
basket summing to $2.01 could be a real, liquid mispricing, or it could
be one or two thin trades on an illiquid leg -- exactly the thin-liquidity
lesson this project already learned once on the ladder side
(MIN_N_PER_SIDE=20), which has no equivalent anywhere in the MECE
pipeline yet (confirmed: zero matches for MIN_N/liquidity/volume in
mece_sum_to_one_pnl_backtest.py).

This re-scans just the trades files needed for the top-N largest-deviation
snapshots specifically -- not the whole window -- and counts actual
trades per leg, so you know whether these are real opportunities or noise
before trusting them or feeding them into the PnL backtest.

Run from the same folder as the other example scripts, after
mece_sum_to_one_check.py has already produced mece_sum_to_one_results.parquet
and mece_sum_to_one_leg_prices.parquet.

NOTE: does NOT import from mece_sum_to_one_check.py -- that file is
top-level executable script code (not function-wrapped), so importing
anything from it would re-run its entire expensive computation as a side
effect. The one helper needed (_month_globs) is duplicated here instead.
"""

import glob

import polars as pl

try:
    from data_windows import MECE_MONTHS
except ImportError:
    from .data_windows import MECE_MONTHS

pl.Config.set_tbl_rows(-1)

RESULTS_PATH = "mece_sum_to_one_results.parquet"
LEG_PRICES_PATH = "mece_sum_to_one_leg_prices.parquet"
TOP_N = 20  # wider than the printed top-10, to also catch borderline cases


def _trades_month_globs(month: str) -> str:
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/trades/trades_kalshi_{parity}/trades_{month}.parquet"


def check() -> None:
    results = pl.read_parquet(RESULTS_PATH)
    leg_prices = pl.read_parquet(LEG_PRICES_PATH)

    top = results.sort("abs_deviation", descending=True).head(TOP_N)
    print(f"Checking liquidity for the top {top.height} largest-magnitude full-basket snapshots...\n")

    keys = top.select(["event_ticker", "date"])
    relevant_legs = leg_prices.join(keys, on=["event_ticker", "date"], how="inner")
    unique_tickers = relevant_legs["ticker"].unique().to_list()

    months_needed = sorted({d.strftime("%Y-%m") for d in relevant_legs["date"].to_list()})
    print(f"Months needing a targeted rescan (trade counts, not full tables): {months_needed}\n")

    trade_counts_parts = []
    for m in months_needed:
        tp = _trades_month_globs(m)
        if not glob.glob(tp):
            print(f"WARNING: no trades file for {m}: {tp}")
            continue
        hit = (
            pl.scan_parquet(tp)
            .select(["ticker", "created_time"])
            .filter(pl.col("ticker").is_in(unique_tickers))
            .with_columns(pl.col("created_time").dt.date().alias("date"))
            .collect()
        )
        trade_counts_parts.append(hit.group_by(["ticker", "date"]).agg(pl.len().alias("n_trades")))

    trade_counts = (
        pl.concat(trade_counts_parts)
        if trade_counts_parts
        else pl.DataFrame(schema={"ticker": pl.Utf8, "date": pl.Date, "n_trades": pl.Int64})
    )

    detail = relevant_legs.join(trade_counts, on=["ticker", "date"], how="left").with_columns(
        pl.col("n_trades").fill_null(0)
    )

    print("=== Per-leg trade counts for each top snapshot ===")
    for row in top.iter_rows(named=True):
        evt, day, dev = row["event_ticker"], row["date"], row["deviation"]
        legs = detail.filter((pl.col("event_ticker") == evt) & (pl.col("date") == day)).sort("n_trades")
        min_n = legs["n_trades"].min() if legs.height else None
        print(f"\n{evt}  {day}  dev=${dev:+.2f}  MIN_LEG_TRADES={min_n}")
        for leg in legs.iter_rows(named=True):
            print(f"    {leg['ticker']:30s}  close=${leg['close']/100.0:.2f}  n_trades={leg['n_trades']}")

    print("\n=== How many of the top snapshots would survive a MIN_LEG_TRADES filter? ===")
    min_per_snapshot = detail.group_by(["event_ticker", "date"]).agg(pl.col("n_trades").min().alias("min_leg_trades"))
    for threshold in [1, 5, 10, 20]:
        survive = min_per_snapshot.filter(pl.col("min_leg_trades") >= threshold).height
        print(f"  MIN_LEG_TRADES >= {threshold:2d}: {survive}/{top.height} snapshots survive")


if __name__ == "__main__":
    check()