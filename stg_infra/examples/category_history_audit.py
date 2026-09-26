"""
category_history_audit.py

Run this BEFORE deciding how many months of history to pull for STGAT
training. Every existing script in this project (mece_*,
pairwise_monotonicity_*, the PnL backtests) is scoped to
TARGET_MONTHS = ["2025-10", "2025-11"] -- two months out of the ~54
months of Kalshi data actually available (June 2021 - November 2025).

This does NOT rerun any arbitrage analysis. It answers a cheaper, prior
question: across the full available history, how many DISTINCT markets
show up per month, broken down by category (via the same
classify_ticker() already used everywhere else in this project)? That
tells you:

  1. When each category (weather/climate, crypto, financials, ...)
     actually starts appearing in the data -- Kalshi's catalog grew a
     lot over 2021-2025, so "54 months available" does not mean "54
     months of weather markets available".
  2. Which specific months are worth pulling to fix the MECE
     sample-size problem (only 267 opportunities found in 2 months),
     versus which months would just add I/O cost with no new signal.

Drop this into stg_infra/examples/ (same folder as
pairwise_monotonicity_pnl_backtest.py, so the classify_ticker import
resolves) and run with `data/` populated from Google Drive as usual.

Output: a per-(month, category) market count table, printed and saved
to category_month_counts.parquet, plus a printed first-seen-month
summary per category.
"""

import argparse
import glob
import os

import polars as pl

pl.Config.set_tbl_rows(-1)  # never truncate printed tables -- this is what cut the
                            # first-seen table off mid-list on the first run

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

# Full available range. Confirmed elsewhere in this project that data
# stops at November 2025 (no December 2025 file) -- see
# mece_sum_to_one_check.py's TARGET_MONTHS comment. Adjust START_MONTH
# down further if you find markets data older than June 2021.
START_MONTH = (2021, 6)
END_MONTH = (2025, 11)
CACHE_PATH = "category_month_counts.parquet"

# Categories most relevant to the two mechanisms -- ladder monotonicity
# leans on crypto/financials, MECE full-basket leans overwhelmingly on
# weather/climate per the existing findings summary. exotics added after
# exotics_category_check.py confirmed it's a real, distinct category
# (NFL/NBA parlays) -- correctly out of scope for both mechanisms, but
# worth tracking here so its footprint doesn't need rediscovering later.
# sports added 2026-09: diagnose_basket_composition.py measured the MECE
# basket population on the 2025-05..09 training window and found it is
# 51.9% SPORTS against 32.0% weather -- and at a 5c opportunity threshold,
# 24.4% sports / 58.0% weather. The "MECE is overwhelmingly weather (93%)"
# premise that set MECE_START = WEATHER_START in data_windows.py therefore
# does not hold on the window the model is actually trained on, so the
# MECE window is currently bounded by a MINORITY category.
#
# mece_sports_breakdown.py independently found sports carried 67% of
# realistic MECE PnL from 22% of opportunities at a 91.2% win rate.
#
# The point of watching it here is to find sports' step-change month the
# same way financials (~14x, 2024-04) and crypto (~17x, 2024-12) were
# found, and then to move MECE_START to the earliest month where the
# categories that actually dominate the mechanism are dense.
WATCH_CATEGORIES = ["weather/climate", "crypto", "financials", "exotics",
                    "sports", "politics"]


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[str]:
    y, m = start
    out = []
    while (y, m) <= end:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def month_path(month: str) -> str:
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet"


def audit() -> pl.DataFrame:
    rows = []
    missing = []
    for month in month_range(START_MONTH, END_MONTH):
        path = month_path(month)
        if not glob.glob(path):
            missing.append(month)
            continue

        # Minimal column selection -- this needs to scan 54 months cheaply.
        tickers = (
            pl.scan_parquet(path)
            .select(["ticker"])
            .unique()
            .collect()
            .get_column("ticker")
        )

        counts: dict[str, int] = {}
        for t in tickers:
            cat = classify_ticker(t)
            counts[cat] = counts.get(cat, 0) + 1

        for cat, n in counts.items():
            rows.append({"month": month, "category": cat, "n_markets": n})

    if missing:
        print(f"WARNING: {len(missing)} months had no markets file: {missing}")

    if not rows:
        print("No data found at all -- check that data/markets/ is populated.")
        return pl.DataFrame(schema={"month": pl.Utf8, "category": pl.Utf8, "n_markets": pl.Int64})

    return pl.DataFrame(rows)


def summarize(df: pl.DataFrame) -> None:
    if df.is_empty():
        return

    print("\n=== Category first-seen month (and total markets across history) ===")
    first_seen = (
        df.group_by("category")
        .agg(
            pl.col("month").min().alias("first_month"),
            pl.col("n_markets").sum().alias("total_markets_all_months"),
        )
        .sort("first_month")
    )
    print(first_seen)

    print(f"\n=== First-seen month + total markets: {', '.join(WATCH_CATEGORIES)} ===")
    watch_summary = first_seen.filter(pl.col("category").is_in(WATCH_CATEGORIES))
    print(watch_summary)

    print(f"\n=== Monthly market counts: {', '.join(WATCH_CATEGORIES)} ===")
    watch = df.filter(pl.col("category").is_in(WATCH_CATEGORIES))
    if watch.is_empty():
        print("None of the watch categories appear anywhere in the scanned history.")
        return
    pivot = watch.pivot(values="n_markets", index="month", on="category").sort("month")
    print(pivot)

    # The financials/crypto windows were set by a step change plus a
    # "under 2% of total volume before this point" rule. Compute both
    # here so the same standard is applied to every category rather than
    # eyeballed off the table above.
    print("\n=== Step-change candidates (same standard used for "
          "financials 2024-04 and crypto 2024-12) ===")
    for cat in WATCH_CATEGORIES:
        sub = watch.filter(pl.col("category") == cat).sort("month")
        if sub.height < 3:
            print(f"  {cat:<18} too few months to judge ({sub.height})")
            continue
        months = sub["month"].to_list()
        n = sub["n_markets"].to_list()
        total = sum(n)
        # largest month-on-month multiple, ignoring tiny bases
        best_i, best_mult = None, 0.0
        for i in range(1, len(n)):
            if n[i - 1] >= 5:
                mult = n[i] / n[i - 1]
                if mult > best_mult:
                    best_i, best_mult = i, mult
        # first month at which the cumulative share left BEHIND exceeds 2%
        cum, two_pct_month = 0, None
        for mth, cnt in zip(months, n):
            if cum > 0.02 * total and two_pct_month is None:
                two_pct_month = mth
                break
            cum += cnt
        step = f"{months[best_i]} ({best_mult:.1f}x)" if best_i else "none"
        print(f"  {cat:<18} first={months[0]}  total={total:>7,}  "
              f"biggest step={step:<20} 2%-of-volume month={two_pct_month}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rescan all 54 months from data/markets/ instead of reusing "
             f"the existing {CACHE_PATH}, if present",
    )
    args = parser.parse_args()

    if not args.refresh and os.path.exists(CACHE_PATH):
        print(f"Reusing existing {CACHE_PATH} (pass --refresh to rescan from scratch)")
        result = pl.read_parquet(CACHE_PATH)
    else:
        result = audit()
        if not result.is_empty():
            result.write_parquet(CACHE_PATH)
            print(f"\nSaved full breakdown to {CACHE_PATH}")

    summarize(result)