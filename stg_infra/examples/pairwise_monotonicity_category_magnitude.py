"""
pairwise_monotonicity_category_magnitude.py

Two things, both aimed at explaining why weather/climate showed a ~50%
same-side violation rate versus crypto's ~9-11%:

  1. A sanity check: print a sample of tickers tagged into each category,
     so you can eyeball whether "weather/climate" is really catching
     temperature-threshold contracts and not something else that happens
     to contain HIGH/LOW as a substring.
  2. Check 3 (median violation gap vs. typical leg spread), broken down by
     category instead of pooled — tests whether weather's high violation
     RATE corresponds to small, noise-scale gaps (weak evidence) or gaps
     that are still clearly larger than the spread (real evidence).
  3. A descriptive look at adjacent-threshold spacing by category (leg_b's
     strike minus leg_a's strike, extracted from the ticker name) — the
     working hypothesis is that weather ladders have finer-grained,
     closer-together thresholds, which would make them structurally more
     prone to flipping into an apparent violation from ordinary spread
     noise, independent of whether real mispricing exists. NOTE: these
     numbers are in each ticker's native units (degrees for weather,
     dollars for crypto, win-counts for sports) so they are NOT directly
     comparable across categories in absolute terms — only useful as a
     within-category descriptive stat and a starting point, not a final
     answer.

ASSUMPTIONS: same as pairwise_monotonicity_family_breakdown.py (results
parquet schema, category keyword lists — extend them if needed) plus the
ticker naming pattern 'FAMILY-DATE-B123.45' / 'FAMILY-DATE-T123.45' used to
extract threshold values. If your ladder tickers don't follow a trailing
letter+number pattern, the threshold extraction will just come back empty
for those and print a warning — not fatal, but check the warning count.
"""

import re
import polars as pl

try:
    # classify_ticker() used to be duplicated locally in this file with its own
    # CRYPTO_KEYS/SPORTS_KEYS/WEATHER_KEYS keyword lists. Now imported from
    # pairwise_monotonicity_pnl_backtest.py instead, which prefers Kalshi's real
    # per-series category (via kalshi_series_categories.parquet, see
    # fetch_kalshi_series_categories.py) and only falls back to the keyword
    # heuristic for tickers not in that lookup -- one source of truth for
    # categorization across every script in this investigation instead of N
    # copies drifting apart.
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

RESULTS_PATH = "pairwise_monotonicity_taker_side_results_corrected.parquet"

THRESHOLD_PATTERN = re.compile(r"^.+-[A-Z]([0-9]+(?:\.[0-9]+)?)$")


def extract_threshold(ticker: str):
    m = THRESHOLD_PATTERN.match(ticker)
    return float(m.group(1)) if m else None


def sample_tickers_by_category(results: pl.DataFrame):
    print("=== Sanity check: sample tickers per category ===")
    # Dynamic now instead of a hardcoded 4-category list -- with real Kalshi
    # categories in play there are more than crypto/sports/weather/other
    # (financials, economics, elections, politics, etc. all show up now).
    categories = (
        results.select("category").unique().sort("category").to_series().to_list()
    )
    for cat in categories:
        sample = (
            results.filter(pl.col("category") == cat)
            .select("leg_a", "leg_b")
            .unique()
            .head(10)
        )
        print(f"--- {cat} ({results.filter(pl.col('category') == cat).height} pairs total) ---")
        print(sample)
    print()


def check_3_by_category(results: pl.DataFrame):
    print("=== Check 3 by category: violation magnitude vs. spread ===")
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
    print("(ratio near/below 1 = violations are roughly spread-sized, weak evidence; well above 1 = stronger evidence)")
    print()


def threshold_spacing_by_category(results: pl.DataFrame):
    print("=== Adjacent-threshold spacing by category (native units — not cross-comparable) ===")
    with_thresh = results.with_columns(
        pl.col("leg_a").map_elements(extract_threshold, return_dtype=pl.Float64).alias("threshold_a"),
        pl.col("leg_b").map_elements(extract_threshold, return_dtype=pl.Float64).alias("threshold_b"),
    ).with_columns((pl.col("threshold_b") - pl.col("threshold_a")).alias("threshold_gap"))

    missing = with_thresh.filter(pl.col("threshold_gap").is_null()).height
    if missing:
        print(f"WARNING: {missing} of {with_thresh.height} pairs had no extractable threshold — excluded below")

    summary = (
        with_thresh.filter(pl.col("threshold_gap").is_not_null())
        .group_by("category")
        .agg(
            pl.col("threshold_gap").median().alias("median_spacing"),
            pl.col("threshold_gap").mean().alias("mean_spacing"),
            pl.len().alias("n_pairs"),
        )
        .sort("category")
    )
    print(summary)
    print()
    return with_thresh


def main():
    results = pl.read_parquet(RESULTS_PATH)
    results = results.with_columns(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )

    sample_tickers_by_category(results)
    check_3_by_category(results)
    threshold_spacing_by_category(results)


if __name__ == "__main__":
    main()