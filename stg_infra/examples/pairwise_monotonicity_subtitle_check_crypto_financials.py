"""
pairwise_monotonicity_subtitle_check_crypto_financials.py

Direct eyeball check for crypto/financials, mirroring
pairwise_monotonicity_subtitle_check.py's weather verification -- which
printed real yes_sub_title/no_sub_title sample rows and null coverage
before trusting the field, rather than assuming it behaved the same way
everywhere it hadn't been looked at directly.

WHY THIS SCRIPT: build_valid_pairs() in
pairwise_monotonicity_taker_side_check_v2.py reported crypto and
financials coming back 100% upper_tail-classified with zero brackets --
inferred from the classifier's own output composition, not from directly
reading the underlying text the way weather's sub_title check did. This
closes that gap: prints real yes_sub_title samples for both categories,
runs the exact same classify_subtitle() used by the pair-builder over
every crypto/financials row, and reports the full classification
breakdown (not just the pairs that survived pairing) plus null coverage,
so "100% upper_tail" can be confirmed as a property of the raw data
instead of an artifact of how pairs happened to get built.

If a category comes back with meaningfully more than a handful of
'bracket' or 'unrecognized' rows, that's inconsistent with what the
corrected pair-builder assumed and is worth chasing down before trusting
the crypto/financials ladder result as fully verified.
"""

import glob

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
    from pairwise_monotonicity_taker_side_check_v2 import classify_subtitle
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker
    from .pairwise_monotonicity_taker_side_check_v2 import classify_subtitle

pl.Config.set_fmt_str_lengths(200)
pl.Config.set_tbl_width_chars(220)

TARGET_MONTHS = ["2025-10", "2025-11"]
CATEGORIES_TO_CHECK = ["crypto", "financials"]


def _month_globs(month: str):
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet"


def load_markets() -> pl.DataFrame:
    paths = []
    for m in TARGET_MONTHS:
        mp = _month_globs(m)
        if glob.glob(mp):
            paths.append(mp)
        else:
            print(f"WARNING: no markets file for {m}: {mp}")
    markets = pl.concat([
        pl.scan_parquet(p).select([
            "ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title", "_fetched_at"
        ]).collect()
        for p in paths
    ])
    return markets.sort("_fetched_at", descending=True).unique(subset=["ticker"], keep="first")


def main():
    markets = load_markets()
    markets = markets.with_columns(
        pl.col("ticker").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )

    for cat in CATEGORIES_TO_CHECK:
        cat_df = markets.filter(pl.col("category") == cat)
        print(f"=== {cat}: {cat_df.height} legs in markets table ===\n")

        sample = cat_df.select("ticker", "yes_sub_title", "no_sub_title").unique().sample(
            n=min(20, cat_df.height), seed=0
        )
        print(f"--- random sample of {sample.height} yes_sub_title/no_sub_title rows ---")
        print(sample)
        print()

        null_check = cat_df.select(
            pl.col("yes_sub_title").null_count().alias("yes_sub_title_nulls"),
            pl.col("no_sub_title").null_count().alias("no_sub_title_nulls"),
            pl.len().alias("total"),
        )
        print("--- null coverage ---")
        print(null_check)
        print()

        classified = cat_df.with_columns(
            pl.col("yes_sub_title").map_elements(classify_subtitle, return_dtype=pl.Utf8).alias("leg_type")
        )
        breakdown = classified.group_by("leg_type").agg(pl.len().alias("n")).sort("n", descending=True)
        print(f"--- full classification breakdown, ALL {cat_df.height} {cat} legs (not just paired ones) ---")
        print(breakdown)

        non_upper = classified.filter(pl.col("leg_type") != "upper_tail")
        if non_upper.height:
            print(f"\n{non_upper.height} legs classified as something other than upper_tail -- sample below:")
            print(non_upper.select("ticker", "yes_sub_title", "leg_type").unique().head(15))
        else:
            print(f"\nEvery single {cat} leg classified upper_tail -- confirms this isn't an artifact of "
                  f"which legs happened to survive pairing.")
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()