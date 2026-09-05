"""
pairwise_monotonicity_category_magnitude_wellsampled.py

Same as pairwise_monotonicity_category_magnitude.py's Check 3, but
restricted to the well-sampled subset (same definition as Check 1: both
same_side_yes_low_n and same_side_no_low_n are False, i.e. every leg has
>= MIN_N_PER_SIDE trades on both taker_side values). This tests whether
weather/climate's ~4x gap/spread ratio and ~50% violation rate hold up once
you exclude the thin, noisy pairs, or whether that signal was itself
partly a small-sample artifact.

ASSUMPTIONS: same as the previous two follow-up scripts — results parquet
schema from pairwise_monotonicity_taker_side_check.py, category keyword
lists from pairwise_monotonicity_family_breakdown.py.
"""

import polars as pl

try:
    # classify_ticker() now shared from pairwise_monotonicity_pnl_backtest.py
    # (prefers Kalshi's real per-series category via kalshi_series_categories.
    # parquet, falls back to the keyword heuristic) instead of the local
    # keyword-only copy this file used to carry.
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

RESULTS_PATH = "pairwise_monotonicity_taker_side_results_corrected.parquet"


def composition(results: pl.DataFrame, label: str):
    comp = results.group_by("category").agg(pl.len().alias("n_pairs")).sort("category")
    total = results.height
    comp = comp.with_columns((pl.col("n_pairs") / total).alias("share"))
    print(f"--- category composition: {label} (n={total}) ---")
    print(comp)
    print()


def check_3_by_category(results: pl.DataFrame, label: str):
    print(f"=== Check 3 by category ({label}) ===")
    for side in ("yes", "no"):
        viol = results.filter(pl.col(f"same_side_{side}_violation") == True)  # noqa: E712
        if viol.height == 0:
            print(f"same-side '{side}': no violations at all")
            continue
        summary = (
            viol.with_columns(((pl.col("spread_a") + pl.col("spread_b")) / 2).alias("avg_leg_spread"))
            .group_by("category")
            .agg(
                pl.col(f"same_side_{side}_gap").median().alias("median_gap"),
                pl.col("avg_leg_spread").mean().alias("avg_spread"),
                pl.len().alias("n_violations"),
            )
            .with_columns((pl.col("median_gap") / pl.col("avg_spread")).alias("gap_spread_ratio"))
            .sort("category")
        )
        print(f"--- same_side_{side}_violation ---")
        print(summary)
    print()


def rate_by_category(results: pl.DataFrame, label: str):
    summary = results.group_by("category").agg(
        pl.col("same_side_yes_violation").mean().alias("yes_violation_rate"),
        pl.col("same_side_no_violation").mean().alias("no_violation_rate"),
        pl.len().alias("n_pairs"),
    ).sort("category")
    print(f"--- violation rate by category: {label} ---")
    print(summary)
    print()


def main():
    results = pl.read_parquet(RESULTS_PATH)
    results = results.with_columns(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )

    well_sampled = results.filter(
        ~pl.col("same_side_yes_low_n") & ~pl.col("same_side_no_low_n")
    )

    composition(results, "all pairs")
    composition(well_sampled, "well-sampled only")
    rate_by_category(well_sampled, "well-sampled only")
    check_3_by_category(well_sampled, "well-sampled only")


if __name__ == "__main__":
    main()