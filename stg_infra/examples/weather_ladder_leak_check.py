"""
weather_ladder_leak_check.py

pairwise_monotonicity_pair_type_composition_by_category.py found 228
weather/climate pairs in the full 20-month ladder run, all classified as
valid upper_tail-upper_tail pairs. This directly touches a foundational
finding of the project -- weather was established as 100% bracket-shaped
with ZERO valid ladder pairs, which is why it was excluded from the
ladder mechanism entirely and routed to MECE instead.

228 out of 2.47M pairs (0.009%) can't move any PnL number, but the
REASON matters more than the size, since so much scoping (including
data_windows.py's category split) assumes weather doesn't do this. This
checks two things:

  1. Which series_tickers these pairs actually come from -- the same
     city-temperature series the original bracket finding was about
     (KXHIGHNY, KXCITIESWEATHER, etc.), or a different weather variable
     entirely (snow, wind, precipitation) that's genuinely
     threshold-structured and just wasn't present in the original
     2-month sample.
  2. Which months they trade in -- LADDER_MONTHS starts 2024-04, well
     before weather's own designated window (MECE_MONTHS, 2025-02
     onward). If these pairs cluster in 2024, that's a different
     question (does weather's sub_title convention predate
     KXCITIESWEATHER's rollout?) than if they're recent.

Run from the same folder as the other example scripts, with data/
populated as usual.
"""

import glob

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
    from pairwise_monotonicity_taker_side_check_v2 import (
        classify_subtitle,
        _month_globs,
    )
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker
    from .pairwise_monotonicity_taker_side_check_v2 import (
        classify_subtitle,
        _month_globs,
    )

try:
    from data_windows import LADDER_MONTHS
except ImportError:
    from .data_windows import LADDER_MONTHS

pl.Config.set_tbl_rows(-1)

RESULTS_PATH = "pairwise_monotonicity_taker_side_results_corrected.parquet"


def _series_prefix(ticker: str) -> str:
    return ticker.split("-", 1)[0]


def find_weather_pairs() -> pl.DataFrame:
    results = pl.read_parquet(RESULTS_PATH)
    weather = results.filter(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8) == "weather/climate"
    )
    return weather


def load_ticker_metadata(tickers: set[str]) -> pl.DataFrame:
    """Looks up sub_title and which month(s) each ticker appears in, across
    every LADDER_MONTHS file -- not just one -- so we know both the
    structural shape (sub_title) and the time distribution."""
    frames = []
    for month in LADDER_MONTHS:
        path = _month_globs(month)
        if not glob.glob(path):
            continue
        available = pl.scan_parquet(path).collect_schema().names()
        select_cols = [c for c in ["ticker", "yes_sub_title"] if c in available]
        df = (
            pl.scan_parquet(path)
            .select(select_cols)
            .filter(pl.col("ticker").is_in(list(tickers)))
            .unique(subset=["ticker"])
            .collect()
            .with_columns(pl.lit(month).alias("first_seen_in_scan"))
        )
        if df.height:
            frames.append(df)
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames)
    # keep the earliest month each ticker was seen in, for the time-distribution question
    return combined.sort("first_seen_in_scan").unique(subset=["ticker"], keep="first")


def check() -> None:
    weather = find_weather_pairs()
    print(f"Weather pairs found in results: {weather.height}")
    if weather.is_empty():
        return

    tickers = set(weather["leg_a"].to_list()) | set(weather["leg_b"].to_list())
    print(f"Distinct tickers involved: {len(tickers)}")

    meta = load_ticker_metadata(tickers)
    if meta.is_empty():
        print("Could not find these tickers in any LADDER_MONTHS markets file -- "
              "check that data/markets/ covers the full window.")
        return

    meta = meta.with_columns(
        pl.col("ticker").map_elements(_series_prefix, return_dtype=pl.Utf8).alias("series_ticker"),
        pl.col("yes_sub_title").map_elements(classify_subtitle, return_dtype=pl.Utf8).alias("leg_type"),
    )

    print("\n=== Distinct weather series_tickers behind these pairs ===")
    by_series = meta.group_by("series_ticker").agg(
        pl.len().alias("n_tickers"),
        pl.col("first_seen_in_scan").min().alias("earliest_month_seen"),
        pl.col("first_seen_in_scan").max().alias("latest_month_seen"),
    ).sort("n_tickers", descending=True)
    print(by_series)

    print("\n=== Sample tickers + sub_titles per series ===")
    for series in by_series["series_ticker"]:
        sample = meta.filter(pl.col("series_ticker") == series).select(
            ["ticker", "yes_sub_title", "leg_type", "first_seen_in_scan"]
        ).head(6)
        print(f"\n-- {series} --")
        print(sample)

    print("\n=== Month distribution of first appearance (all involved tickers) ===")
    print(meta.group_by("first_seen_in_scan").agg(pl.len().alias("n_tickers")).sort("first_seen_in_scan"))


if __name__ == "__main__":
    check()