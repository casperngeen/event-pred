"""
mece_sports_breakdown.py

The "sports" category in mece_sum_to_one_pnl_backtest.py's output (91.2%
win rate, n=68, $716.15 realistic PnL -- 67% of total PnL from 22% of
opportunities) is Kalshi's own category label, but it bundles several
structurally different competitions: EPL/MLS/Club World Cup soccer
(3-way win/tie/loss), F1 and NASCAR races (20-41 way winner-take-all).
Before writing this up as a headline finding, this checks whether the
91.2% figure is a genuine property of "sports" broadly, or concentrated
in one specific league -- exactly the kind of aggregate-hides-structure
surprise this project has hit repeatedly (weather's bracket/hurricane
split, crypto's duplicate-phrase artifact).

Run from the same folder as the other example scripts, after
mece_sum_to_one_pnl_backtest.py has already produced
mece_sum_to_one_pnl_results.parquet.
"""

import re

import polars as pl

pl.Config.set_tbl_rows(-1)

PNL_RESULTS_PATH = "mece_sum_to_one_pnl_results.parquet"

SERIES_RE = re.compile(r"^([^-]+)")


def series_prefix(event_ticker: str) -> str:
    m = SERIES_RE.match(event_ticker)
    return m.group(1) if m else event_ticker


def check() -> None:
    pnl = pl.read_parquet(PNL_RESULTS_PATH)
    sports = pnl.filter(pl.col("category") == "sports").with_columns(
        pl.col("event_ticker").map_elements(series_prefix, return_dtype=pl.Utf8).alias("series")
    )

    print(f"Total sports opportunities: {sports.height}\n")

    print("=== Sports PnL broken down by series/league ===")
    breakdown = (
        sports.group_by("series")
        .agg(
            pl.len().alias("n_opportunities"),
            pl.col("idealized_pnl_usd").sum().alias("idealized_total_usd"),
            pl.col("realistic_pnl_usd").sum().alias("realistic_total_usd"),
            (pl.col("realistic_pnl_usd") > 0).mean().alias("realistic_win_rate"),
            pl.col("realistic_pnl_usd").median().alias("realistic_median_usd"),
        )
        .sort("realistic_total_usd", descending=True)
    )
    print(breakdown)

    print("\n=== Per-opportunity detail, sorted by realistic PnL ===")
    print(
        sports.select(["series", "event_ticker", "date", "deviation", "side", "realistic_pnl_usd", "min_leg_trades"])
        .sort("realistic_pnl_usd", descending=True)
    )


if __name__ == "__main__":
    check()