"""
classify_subtitle_diff_check.py

The compound-subtitle fix (occurrence-counting in classify_subtitle) was
built and verified against KXCITIESWEATHER's compound conditions. The
full rerun showed weather pairs drop 228 -> 204 (-24) as expected, but
ALSO showed crypto drop 914,510 -> 914,462 (-48) -- a category the fix
was never specifically checked against.

This isolates exactly which tickers flipped from a tail classification
to "unrecognized" under the new logic, broken down by category, so the
crypto drop can be confirmed as a genuine compound-condition exclusion
(correct) rather than an over-eager false positive introduced by the
fix (a new bug).

Run from the same folder as the other example scripts, with data/
populated as usual.
"""

import glob

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
    from pairwise_monotonicity_taker_side_check_v2 import _month_globs, UPPER_PHRASES, LOWER_PHRASES
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker
    from .pairwise_monotonicity_taker_side_check_v2 import _month_globs, UPPER_PHRASES, LOWER_PHRASES

try:
    from data_windows import LADDER_MONTHS
except ImportError:
    from .data_windows import LADDER_MONTHS

pl.Config.set_tbl_rows(-1)


def old_classify(sub_title):
    """The pre-fix logic: any() presence check, no occurrence counting."""
    if sub_title is None:
        return "unrecognized"
    s = sub_title.lower()
    if " to " in s:
        return "bracket"
    if any(p in s for p in UPPER_PHRASES):
        return "upper_tail"
    if any(p in s for p in LOWER_PHRASES):
        return "lower_tail"
    return "unrecognized"


def new_classify(sub_title):
    """The current classify_subtitle logic (colon-based compound
    detection), reimplemented here so this script has both versions side
    by side for direct comparison. Keep this in sync with the real
    function in pairwise_monotonicity_taker_side_check_v2.py."""
    if sub_title is None:
        return "unrecognized"
    s = sub_title.lower()
    if " to " in s:
        return "bracket"
    if ":" in s:
        return "unrecognized"
    if any(p in s for p in UPPER_PHRASES):
        return "upper_tail"
    if any(p in s for p in LOWER_PHRASES):
        return "lower_tail"
    return "unrecognized"


def load_markets() -> pl.DataFrame:
    frames = []
    for m in LADDER_MONTHS:
        p = _month_globs(m)
        if not glob.glob(p):
            print(f"WARNING: no markets file for {m}: {p}")
            continue
        available = pl.scan_parquet(p).collect_schema().names()
        select_cols = [c for c in ["ticker", "yes_sub_title"] if c in available]
        frames.append(pl.scan_parquet(p).select(select_cols).unique(subset=["ticker"]).collect())
    if not frames:
        raise FileNotFoundError("No markets files found across LADDER_MONTHS -- check data/markets/.")
    return pl.concat(frames).unique(subset=["ticker"])


def check() -> None:
    markets = load_markets()
    markets = markets.with_columns(
        pl.col("ticker").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category"),
        pl.col("yes_sub_title").map_elements(old_classify, return_dtype=pl.Utf8).alias("old_type"),
        pl.col("yes_sub_title").map_elements(new_classify, return_dtype=pl.Utf8).alias("new_type"),
    )

    changed = markets.filter(
        pl.col("old_type").is_in(["upper_tail", "lower_tail"]) & (pl.col("new_type") == "unrecognized")
    )

    print("=== Tickers that flipped from tail -> unrecognized under the new logic, by category ===")
    print(changed.group_by("category").agg(pl.len().alias("n_tickers")).sort("n_tickers", descending=True))

    for cat in changed["category"].unique().to_list():
        subset = changed.filter(pl.col("category") == cat)
        print(f"\n=== {cat}: {subset.height} flipped tickers, sample sub_titles ===")
        print(subset.select(["ticker", "yes_sub_title"]).head(10))


if __name__ == "__main__":
    check()