"""
pairwise_monotonicity_weather_pair_type_composition.py

v2 -- classifies by yes_sub_title text instead of the ticker suffix
letter. The letter-only version of this script (v1) treated every T
ticker as one direction ("more than X"), but the sub_title check
(pairwise_monotonicity_subtitle_check.py) showed T tickers appear at BOTH
ends of a weather event's range: an upper-tail catch-all ("88 or above")
AND a separate lower-tail catch-all ("70 or below"), both using the same
letter. A letter-only classifier would wrongly call those two a "T-T"
(nested, valid) pair when they're actually opposite-direction tails --
just as invalid as a bracket/threshold mismatch. sub_title is the only
field in this markets table that actually distinguishes direction, and
it's cleanly formatted and 100% populated across all 2,411 weather rows
checked, so it's the right thing to classify on.

Confirmed contract anatomy for a weather event (e.g. one city, one day):
  "N° to M°"     -> a BRACKET: one discrete bucket in the temperature
                    distribution. Disjoint from every other bucket -- NOT
                    a nested outcome, so no pairwise monotonicity relation
                    to any other bracket.
  "N° or above"  -> the event's UPPER TAIL catch-all.
  "N° or below"  -> the event's LOWER TAIL catch-all.
Brackets + one upper tail + one lower tail partition the full outcome
space and should sum to $1 for that event -- this is a single-winner MECE
structure (§4.5 of the feasibility report), not a nested threshold ladder
(§4.3). The only pairwise-valid comparison left is two UPPER tails (or two
LOWER tails) from different events/thresholds, if enough of those exist
to compare at all -- almost certainly rare, since each event seems to
carry just one of each tail.

WHAT THIS SCRIPT DOES: joins the already-tested weather pairs (from
pairwise_monotonicity_taker_side_results.parquet) against the markets
table's yes_sub_title, classifies each leg as bracket / upper_tail /
lower_tail / unrecognized, labels each pair by the combination, and
recomputes the headline violation rate / gap-spread ratio per bucket --
quantifying exactly how much of the current weather finding rests on
comparisons with no valid no-arbitrage meaning.
"""

import glob

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

RESULTS_PATH = "pairwise_monotonicity_taker_side_results.parquet"
TARGET_MONTHS = ["2025-10", "2025-11"]  # keep in sync with the results file's window


def _month_globs(month: str):
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet"


def load_subtitle_lookup() -> pl.DataFrame:
    """ticker -> yes_sub_title, deduped to the latest snapshot per ticker."""
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
    if sub_title is None:
        return "unrecognized"
    s = sub_title.lower()
    if " to " in s:
        return "bracket"
    if "or above" in s:
        return "upper_tail"
    if "or below" in s:
        return "lower_tail"
    return "unrecognized"


def pair_type(type_a: str, type_b: str) -> str:
    if type_a == "unrecognized" or type_b == "unrecognized":
        return "unrecognized"
    if type_a == type_b:
        return f"{type_a}-{type_a}"
    lo, hi = sorted([type_a, type_b])
    return f"{lo} / {hi} (mixed)"


def main():
    results = pl.read_parquet(RESULTS_PATH)
    results = results.with_columns(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )
    weather = results.filter(pl.col("category") == "weather/climate")
    print(f"weather/climate pairs in results file: {weather.height}\n")

    if weather.height == 0:
        print("No weather/climate pairs found -- check classify_ticker() / kalshi_series_categories.parquet.")
        return

    sub_lookup = load_subtitle_lookup()
    weather = (
        weather.join(sub_lookup.rename({"ticker": "leg_a", "yes_sub_title": "sub_title_a"}), on="leg_a", how="left")
        .join(sub_lookup.rename({"ticker": "leg_b", "yes_sub_title": "sub_title_b"}), on="leg_b", how="left")
    )

    unmatched_to_markets = weather.filter(pl.col("sub_title_a").is_null() | pl.col("sub_title_b").is_null()).height
    if unmatched_to_markets:
        print(f"NOTE: {unmatched_to_markets} of {weather.height} weather pairs have a leg missing from the "
              f"markets table (not found in {TARGET_MONTHS} snapshots) -- these fall into 'unrecognized' below "
              f"rather than being silently dropped.\n")

    weather = weather.with_columns([
        pl.col("sub_title_a").map_elements(classify_subtitle, return_dtype=pl.Utf8).alias("type_a"),
        pl.col("sub_title_b").map_elements(classify_subtitle, return_dtype=pl.Utf8).alias("type_b"),
    ])
    weather = weather.with_columns(
        pl.struct(["type_a", "type_b"]).map_elements(
            lambda row: pair_type(row["type_a"], row["type_b"]), return_dtype=pl.Utf8
        ).alias("pair_type")
    )

    print("=== Composition by pair type ===")
    comp = weather.group_by("pair_type").agg(pl.len().alias("n_pairs")).sort("n_pairs", descending=True)
    comp = comp.with_columns((pl.col("n_pairs") / weather.height).alias("share"))
    print(comp)
    print()

    print("=== Headline stats by pair type (this is the actual answer) ===")
    for pt in comp["pair_type"].to_list():
        sub = weather.filter(pl.col("pair_type") == pt)
        rate = sub["same_side_yes_violation"].mean()
        viol = sub.filter(pl.col("same_side_yes_violation") == True)  # noqa: E712
        print(f"--- {pt} (n={sub.height}) ---")
        print(f"  same-side yes violation rate: {rate}")
        if viol.height:
            avg_spread = ((viol["spread_a"] + viol["spread_b"]) / 2).mean()
            median_gap = viol["same_side_yes_gap"].median()
            print(f"  median gap: {median_gap}")
            print(f"  gap/spread ratio (pooled): {median_gap / avg_spread if avg_spread else None}")
        print()

    print(
        "bracket-bracket and any mixed rows above (bracket/upper_tail, bracket/lower_tail, and critically "
        "upper_tail/lower_tail -- two OPPOSITE-direction tails, which a letter-only classifier would have "
        "wrongly called a valid 'T-T' pair) have NO valid no-arbitrage interpretation -- none of these are "
        "nested outcomes, so a 'violation' there isn't mispricing, it's a category error in what's being "
        "compared. upper_tail-upper_tail or lower_tail-lower_tail rows (if present in meaningful numbers) "
        "are the only pair types that are legitimate ladder-monotonicity tests for weather -- expect these "
        "to be rare or empty, since each event seems to carry only one tail of each direction. If so, the "
        "fix isn't tuning MIN_OUTCOMES or pairing direction -- it's replacing the ladder test for weather "
        "with the full-basket MECE sum-to-$1 test (mece_sum_to_one_check.py's methodology, §4.5) as the "
        "primary evidentiary mechanism for this category."
    )


if __name__ == "__main__":
    main()