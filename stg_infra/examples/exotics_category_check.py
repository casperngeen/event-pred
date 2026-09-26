"""
exotics_category_check.py

Quick diagnostic on the "exotics" category surfaced by
category_history_audit.py -- first seen 2025-09, and already the single
largest category in the whole audit (549,877 markets across just 3
months, which overlaps entirely with the original 2-month sample used
for every result in the findings summary).

Before treating this as ladder-eligible, MECE-eligible, or neither, this
checks:

  1. Whether Kalshi itself tags these series_tickers as "exotics" (via
     kalshi_series_categories.parquet -- the same source of truth
     classify_ticker() uses), or whether classify_ticker() is falling
     back to the keyword heuristic and mislabeling something else.
  2. How concentrated the volume is -- one series_ticker flooding the
     count, or many distinct series.
  3. A raw sample of tickers + titles/sub_titles per top series, so you
     can eyeball whether these look ladder-shaped (upper/lower tail
     thresholds), MECE-shaped (bracket partitions), or neither (e.g.
     single-shot novelty contracts with no structural pair/basket at
     all).

Run from the same folder as category_history_audit.py (needs
category_month_counts.parquet to already exist from that run).
"""

import glob

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import (
        classify_ticker,
        _series_prefix,
        SERIES_CATEGORIES_PATH,
    )
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import (
        classify_ticker,
        _series_prefix,
        SERIES_CATEGORIES_PATH,
    )

try:
    from category_history_audit import month_path, month_range, START_MONTH, END_MONTH
except ImportError:
    from .category_history_audit import month_path, month_range, START_MONTH, END_MONTH

N_SAMPLE_PER_SERIES = 5
N_TOP_SERIES = 10


def series_kalshi_tags_as_exotics() -> list[str]:
    """Ground truth: which series_tickers Kalshi's own taxonomy calls 'exotics'."""
    lookup = pl.read_parquet(SERIES_CATEGORIES_PATH)
    return (
        lookup.filter(pl.col("category").str.to_lowercase() == "exotics")
        .get_column("series_ticker")
        .to_list()
    )


def find_exotics_months() -> list[str]:
    """Reuse the already-saved audit output instead of rescanning all 54 months."""
    try:
        counts = pl.read_parquet("category_month_counts.parquet")
        months = (
            counts.filter((pl.col("category") == "exotics") & (pl.col("n_markets") > 0))
            .get_column("month")
            .to_list()
        )
        if months:
            return sorted(months)
    except FileNotFoundError:
        pass
    print("WARNING: category_month_counts.parquet not found -- scanning full history instead.")
    return month_range(START_MONTH, END_MONTH)


def load_exotics_rows(months: list[str]) -> pl.DataFrame:
    frames = []
    for month in months:
        path = month_path(month)
        matches = glob.glob(path)
        if not matches:
            continue

        available = pl.scan_parquet(path).columns
        select_cols = [c for c in ["ticker", "title", "yes_sub_title"] if c in available]
        df = pl.scan_parquet(path).select(select_cols).unique(subset=["ticker"]).collect()

        df = df.with_columns(
            pl.col("ticker").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category"),
            pl.col("ticker").map_elements(_series_prefix, return_dtype=pl.Utf8).alias("series_ticker"),
        )
        frames.append(df.filter(pl.col("category") == "exotics"))

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def check() -> None:
    tagged_series = series_kalshi_tags_as_exotics()
    print(f"Series Kalshi's own taxonomy tags 'exotics': {len(tagged_series)}")
    if tagged_series:
        print(tagged_series[:30], "..." if len(tagged_series) > 30 else "")
    else:
        print("NONE -- if classify_ticker() is still labeling markets 'exotics', it's "
              "falling through the keyword fallback, not Kalshi's real taxonomy. Check "
              "classify_ticker()'s fallback branch for what's catching these.")

    months = find_exotics_months()
    print(f"\nScanning months where exotics appeared: {months}\n")

    exotics = load_exotics_rows(months)
    if exotics.is_empty():
        print("No exotics rows found in scanned months.")
        return

    print(f"Total exotics rows across scanned months: {exotics.height}")
    print(f"Distinct series_tickers within exotics: {exotics['series_ticker'].n_unique()}")

    print("\n=== Top series_tickers by market count ===")
    top_series = (
        exotics.group_by("series_ticker")
        .agg(pl.len().alias("n_markets"))
        .sort("n_markets", descending=True)
    )
    print(top_series.head(N_TOP_SERIES))

    print(f"\n=== Sample tickers per top series (up to {N_SAMPLE_PER_SERIES} each) ===")
    for series in top_series.head(N_TOP_SERIES).get_column("series_ticker"):
        sample = exotics.filter(pl.col("series_ticker") == series).head(N_SAMPLE_PER_SERIES)
        print(f"\n-- {series} --")
        print(sample)

    check_threshold_phrasing_leak(months)


try:
    from pairwise_monotonicity_taker_side_check_v2 import UPPER_PHRASES, LOWER_PHRASES
except ImportError:
    from .pairwise_monotonicity_taker_side_check_v2 import UPPER_PHRASES, LOWER_PHRASES

THRESHOLD_PHRASES = UPPER_PHRASES + LOWER_PHRASES


def _matches_any_phrase(col: pl.Expr, phrases: tuple[str, ...]) -> pl.Expr:
    """OR'd literal substring match, built by hand instead of str.contains_any --
    that method isn't guaranteed to exist across the polars>=0.20 range this
    project's requirements.txt allows, and this check is cheap enough that
    portability matters more than a one-line convenience call."""
    expr = pl.lit(False)
    for phrase in phrases:
        expr = expr | col.str.contains(phrase, literal=True)
    return expr


def check_threshold_phrasing_leak(months: list[str]) -> None:
    """Confirms exotics/parlay tickers aren't accidentally matching the
    ladder pipeline's threshold-phrase classifier despite not having a
    single ordered variable -- checked across every month exotics actually
    appears in, not just one."""
    print("\n=== Threshold-phrasing leak check (all exotics months) ===")
    total_leaked = 0
    for month in months:
        path = month_path(month)
        if not glob.glob(path):
            continue
        available = pl.scan_parquet(path).collect_schema().names()
        if "yes_sub_title" not in available:
            print(f"{month}: no yes_sub_title column present, skipping")
            continue

        df = pl.scan_parquet(path).select(["ticker", "yes_sub_title"]).collect()
        leaked = df.filter(
            pl.col("ticker").str.starts_with("KXMVE")
            & _matches_any_phrase(pl.col("yes_sub_title"), THRESHOLD_PHRASES)
        )
        total_leaked += leaked.height
        print(f"{month}: {leaked.height} exotics rows matched threshold phrasing")
        if leaked.height:
            print(leaked.head(5))

    print(f"\nTotal across all exotics months: {total_leaked}")


if __name__ == "__main__":
    check()