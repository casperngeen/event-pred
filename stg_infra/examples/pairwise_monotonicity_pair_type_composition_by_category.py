"""
pairwise_monotonicity_pair_type_composition_by_category.py

Generalizes pairwise_monotonicity_weather_pair_type_composition.py (which
found that 100% of the 1,142 weather pairs tested have no valid
no-arbitrage interpretation -- bracket/bracket or bracket/tail comparisons,
zero genuine same-direction nested thresholds) to EVERY category in the
report, not just weather.

Why this can't just be assumed to be fine elsewhere: every number
currently in the report -- crypto, financials, elections, politics,
sports, weather -- was built with the same regex ticker-suffix pairing
that just failed catastrophically for weather. We already saw brackets
show up in non-weather tickers too (KXNASDAQ100Y-...-B24750/B24250,
KXINXY-25DEC31-B5700/B5500), so the same corruption may be present
elsewhere and just hasn't been checked category by category yet.

What this does NOT assume: that the original feasibility report's §4.3
ladder result (669 events, crypto/FX/equity-index, validated via a
title-keyword method that structurally excludes bracket-style titles) is
automatically safe just because it used a different method. This script
only tells you about the CURRENT report's pairs (built with the regex
method) -- if a category comes back with zero valid pairs here, that
means the CURRENT numbers for that category need the same treatment
weather just got, regardless of what the original report separately
validated with a different pipeline.

WHAT "VALID" MEANS HERE: a pair is only a legitimate no-arbitrage test if
both legs are nested outcomes of the SAME direction -- both upper-tail
thresholds ("X or above"/"or higher"/"or greater"), or both lower-tail
thresholds ("X or below"/"or lower"/"or less"). Bracket legs ("X to Y") are
never valid pair members, in either direction, against anything -- disjoint
partition buckets have no monotonicity relationship to any other leg.

WHAT THIS SCRIPT DOES: joins every already-tested pair (across every
category, from pairwise_monotonicity_taker_side_results.parquet) against
yes_sub_title, classifies each leg as bracket / upper_tail / lower_tail /
unrecognized, and reports PER CATEGORY: total pairs tested, how many are
actually valid (upper-upper or lower-lower), and -- for categories with
any valid pairs at all -- the headline violation rate / gap-spread ratio
recomputed on just the valid subset, next to the full (currently-reported)
number, so you can see directly what survives.
"""

import glob

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

RESULTS_PATH = "pairwise_monotonicity_taker_side_results_corrected.parquet"
TARGET_MONTHS = ["2025-10", "2025-11"]  # keep in sync with the results file's window

# Broadened to match pairwise_monotonicity_taker_side_check_corrected.py's phrase list (elections
# revealed "less than X" / "X and above" wording that this file's original list didn't cover) --
# kept in sync so this composition check classifies legs exactly the same way the pair-BUILDER did,
# and should now report ~100% valid for every category, since invalid pairs are no longer generated
# in the first place. Any category NOT at 100% here means this classifier and build_valid_pairs()
# have drifted out of sync somewhere -- worth investigating rather than assuming it's fine.
UPPER_PHRASES = ("or above", "or higher", "or greater", "or more", "and above")
LOWER_PHRASES = ("or below", "or lower", "or less", "less than")


def _month_globs(month: str):
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet"


def load_subtitle_lookup() -> pl.DataFrame:
    paths = []
    for m in TARGET_MONTHS:
        mp = _month_globs(m)
        if glob.glob(mp):
            paths.append(mp)
        else:
            print(f"WARNING: no markets file for {m}: {mp}")
    markets = pl.concat([
        pl.scan_parquet(p).select(["ticker", "yes_sub_title", "_fetched_at"]).collect()
        for p in paths
    ])
    return (
        markets.sort("_fetched_at", descending=True)
        .unique(subset=["ticker"], keep="first")
        .select(["ticker", "yes_sub_title"])
    )


def classify_subtitle(sub_title: str) -> str:
    """Broadened beyond the weather-only wording ('or above'/'or below') to
    also catch 'or higher'/'or greater'/'or more' and 'or lower'/'or less',
    in case other categories phrase thresholds differently. Still generic
    on the bracket check (' to ' substring) -- no assumption about degree
    symbols or currency signs, so it should transfer to crypto/index/FX
    titles without modification. If a category's sub_title format doesn't
    match any of these, it falls into 'unrecognized' and gets printed as a
    sample below rather than silently miscounted."""
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


def pair_type(type_a: str, type_b: str) -> str:
    if type_a == "unrecognized" or type_b == "unrecognized":
        return "unrecognized"
    if type_a == type_b:
        return f"{type_a}-{type_a}"
    lo, hi = sorted([type_a, type_b])
    return f"{lo} / {hi} (mixed)"


def is_valid(pt: str) -> bool:
    return pt in ("upper_tail-upper_tail", "lower_tail-lower_tail")


def headline_stats(df: pl.DataFrame) -> dict:
    if df.height == 0:
        return {}
    rate = df["same_side_yes_violation"].mean()
    viol = df.filter(pl.col("same_side_yes_violation") == True)  # noqa: E712
    out = {"n": df.height, "violation_rate": rate}
    if viol.height:
        avg_spread = ((viol["spread_a"] + viol["spread_b"]) / 2).mean()
        median_gap = viol["same_side_yes_gap"].median()
        out["median_gap"] = median_gap
        out["gap_spread_ratio"] = (median_gap / avg_spread) if avg_spread else None
    return out


def main():
    results = pl.read_parquet(RESULTS_PATH)
    results = results.with_columns(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )

    sub_lookup = load_subtitle_lookup()
    results = (
        results.join(sub_lookup.rename({"ticker": "leg_a", "yes_sub_title": "sub_title_a"}), on="leg_a", how="left")
        .join(sub_lookup.rename({"ticker": "leg_b", "yes_sub_title": "sub_title_b"}), on="leg_b", how="left")
    )
    results = results.with_columns([
        pl.col("sub_title_a").map_elements(classify_subtitle, return_dtype=pl.Utf8).alias("type_a"),
        pl.col("sub_title_b").map_elements(classify_subtitle, return_dtype=pl.Utf8).alias("type_b"),
    ])
    results = results.with_columns(
        pl.struct(["type_a", "type_b"]).map_elements(
            lambda row: pair_type(row["type_a"], row["type_b"]), return_dtype=pl.Utf8
        ).alias("pair_type")
    )
    results = results.with_columns(
        pl.col("pair_type").map_elements(is_valid, return_dtype=pl.Boolean).alias("is_valid_pair")
    )

    print("=== Per-category: how many currently-tested pairs are actually valid ladder pairs? ===\n")
    categories = results["category"].unique().sort().to_list()
    for cat in categories:
        cat_df = results.filter(pl.col("category") == cat)
        n_total = cat_df.height
        n_valid = cat_df.filter(pl.col("is_valid_pair")).height
        print(f"--- {cat}: {n_total} pairs tested, {n_valid} valid ({n_valid / n_total:.1%}) ---")

        comp = cat_df.group_by("pair_type").agg(pl.len().alias("n")).sort("n", descending=True)
        print(comp)

        full_stats = headline_stats(cat_df)
        print(f"  FULL (as currently in the report): {full_stats}")
        if n_valid:
            valid_stats = headline_stats(cat_df.filter(pl.col("is_valid_pair")))
            print(f"  VALID-ONLY (what actually survives): {valid_stats}")
        else:
            print("  VALID-ONLY: n/a -- zero valid pairs, same situation as weather.")
        print()

    unrecognized = results.filter((pl.col("type_a") == "unrecognized") | (pl.col("type_b") == "unrecognized"))
    if unrecognized.height:
        print(f"=== {unrecognized.height} pairs had at least one leg 'unrecognized' -- sample sub_titles below "
              f"(missing from markets table, or a phrasing this classifier doesn't cover yet) ===")
        sample = (
            unrecognized.select("category", "leg_a", "sub_title_a", "leg_b", "sub_title_b")
            .unique()
            .head(20)
        )
        print(sample)


if __name__ == "__main__":
    main()