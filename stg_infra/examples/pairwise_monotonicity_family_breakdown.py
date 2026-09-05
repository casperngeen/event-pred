"""
pairwise_monotonicity_family_breakdown.py

Investigates the Check 4 surprise (violation rate rises as the MIN_N_PER_SIDE
cutoff rises) by breaking results down by underlying market category —
crypto / sports / weather-climate / other. The hypothesis: high-volume pairs
skew toward volatile, event-driven categories (crypto, sports), where the
same-side average can still get contaminated by within-window price drift
even after controlling for bid/ask side — the same effect that inflated the
early KXMLB whole-history and single-day checks earlier in this
investigation.

ASSUMPTIONS:
  - results parquet columns match pairwise_monotonicity_taker_side_check.py's
    output (leg_a, leg_b, same_side_yes_violation, same_side_no_violation,
    same_side_yes_n_a/n_b, same_side_no_n_a/n_b).
  - The category keyword lists below are a best-effort guess based on
    tickers seen so far in this conversation (KXBTC*, KXETH*, KXSHIBA*,
    KXXRP*, KXMLB*, KXNBAWINS*, KXHIGHMIA*, KXHIGHLAX*, KXARCTICICEMIN*).
    You almost certainly have more families than this — extend the keyword
    lists, or better, join in whatever category/family metadata your
    27-family MECE reconstruction already has, if it captures ladder
    families too.
"""

import polars as pl

try:
    # classify_ticker() now shared from pairwise_monotonicity_pnl_backtest.py
    # (prefers Kalshi's real per-series category via kalshi_series_categories.
    # parquet, falls back to the keyword heuristic) instead of the local
    # best-effort keyword copy this file used to carry.
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

RESULTS_PATH = "pairwise_monotonicity_taker_side_results_corrected.parquet"


def add_category(results: pl.DataFrame) -> pl.DataFrame:
    # leg_a and leg_b are adjacent strikes in the same family, so classifying
    # leg_a is enough — sanity-check this assumption isn't silently wrong:
    mismatched = results.filter(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8)
        != pl.col("leg_b").map_elements(classify_ticker, return_dtype=pl.Utf8)
    )
    if mismatched.height:
        print(f"WARNING: {mismatched.height} pairs have leg_a/leg_b classified differently — check these tickers")
    return results.with_columns(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )


def composition(results: pl.DataFrame, label: str):
    comp = results.group_by("category").agg(pl.len().alias("n_pairs")).sort("category")
    total = results.height
    comp = comp.with_columns((pl.col("n_pairs") / total).alias("share"))
    print(f"--- category composition: {label} (n={total}) ---")
    print(comp)
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
    results = add_category(results)

    # 1. Overall composition — what does the full pair list look like by category?
    composition(results, "all pairs")

    # 2. Composition shift as MIN_N rises — does the pair mix skew toward
    #    crypto/sports as you require more trades per side? That would help
    #    explain Check 4's rising rate as a category effect, not a pure
    #    "more data = more real violations" effect.
    for thresh in (10, 20, 50, 100):
        sub = results.filter(
            (pl.col("same_side_yes_n_a") >= thresh) & (pl.col("same_side_yes_n_b") >= thresh)
        )
        composition(sub, f"min_n={thresh} (yes-side)")

    # 3. Violation rate by category, holding n roughly fixed at a few cutoffs —
    #    isolates whether category itself (not just liquidity) drives the rate.
    for thresh in (20, 100):
        sub = results.filter(
            (pl.col("same_side_yes_n_a") >= thresh) & (pl.col("same_side_yes_n_b") >= thresh)
        )
        rate_by_category(sub, f"min_n={thresh} (yes-side)")


if __name__ == "__main__":
    main()