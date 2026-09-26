"""
weather_spike_check.py

category_history_audit.py surfaced a spike in weather/climate market
counts: Dec 2024 (4,713) and Jan 2025 (8,488) vs. ~900-1,500/month in
the surrounding months. Before pooling all of 2025 into the MECE
training window, this checks whether the spike is:

  (a) a real seasonal event -- e.g. a winter storm driving many new
      local/regional temperature markets, or
  (b) a platform-side artifact -- e.g. Kalshi briefly issuing far more
      granular threshold buckets, a new short-lived series launching
      and inflating the count, or a ticker-dedup quirk in that month's
      file specifically.

Does this by comparing distinct series_tickers active in the spike
months against the surrounding baseline months, and printing sample
titles from whichever series grew the most.
"""

import glob

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker, _series_prefix
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker, _series_prefix

try:
    from category_history_audit import month_path
except ImportError:
    from .category_history_audit import month_path

pl.Config.set_tbl_rows(-1)

SPIKE_MONTHS = ["2024-12", "2025-01"]
BASELINE_MONTHS = ["2024-10", "2024-11", "2025-02", "2025-03"]


def load_weather_series_counts(month: str) -> pl.DataFrame:
    path = month_path(month)
    if not glob.glob(path):
        print(f"WARNING: no markets file for {month}: {path}")
        return pl.DataFrame(schema={"series_ticker": pl.Utf8, "n_markets": pl.Int64})

    available = pl.scan_parquet(path).collect_schema().names()
    select_cols = [c for c in ["ticker", "title"] if c in available]
    df = pl.scan_parquet(path).select(select_cols).unique(subset=["ticker"]).collect()

    df = df.with_columns(
        pl.col("ticker").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category"),
        pl.col("ticker").map_elements(_series_prefix, return_dtype=pl.Utf8).alias("series_ticker"),
    )
    weather = df.filter(pl.col("category") == "weather/climate")
    return weather.group_by("series_ticker").agg(pl.len().alias("n_markets")), weather


def check() -> None:
    print("=== Distinct weather series_tickers: spike months vs baseline ===\n")

    for month in SPIKE_MONTHS + BASELINE_MONTHS:
        counts, _ = load_weather_series_counts(month)
        label = "SPIKE" if month in SPIKE_MONTHS else "baseline"
        print(f"{month} ({label}): {counts['n_markets'].sum()} total markets, "
              f"{counts.height} distinct series_tickers")

    print("\n=== Top weather series_tickers in each spike month (by market count) ===")
    for month in SPIKE_MONTHS:
        counts, weather = load_weather_series_counts(month)
        top = counts.sort("n_markets", descending=True).head(10)
        print(f"\n-- {month} --")
        print(top)

        # Sample titles from the single largest series that month, so you
        # can eyeball whether it's a real distinct weather event or an
        # unusually fine-grained/duplicated bucket structure.
        if top.height:
            top_series = top.get_column("series_ticker")[0]
            print(f"\nSample titles from {top_series} ({month}):")
            print(
                weather.filter(pl.col("series_ticker") == top_series)
                .select(["ticker", "title"])
                .head(8)
            )


if __name__ == "__main__":
    check()