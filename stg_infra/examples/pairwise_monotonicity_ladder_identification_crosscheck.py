"""
pairwise_monotonicity_ladder_identification_crosscheck.py

Answers the open Limitations item ("Ladder-pair identification... has not
yet been formally cross-checked against the original, validated
family-identification logic") and the first half of the STGAT-scoping
question: is the regex-based pair identification used throughout this
investigation (guess_ladder_pairs_from_ticker_names(), operating on the
TICKER string, e.g. 'KXHIGHMIA-25NOV07-B86.5') actually the same set of
pairs that the ORIGINAL, validated pairwise_monotonicity_check.py (§4.3 of
the feasibility report) identifies via its title-template-homogeneity
method (operating on the EVENT TITLE, masking only the numeric threshold,
and explicitly excluding combo-prop markets that a naive keyword/ticker
match would wrongly include)?

These are two different methods, on two different fields, that are
SUPPOSED to converge on the same answer. They have never been directly
diffed against each other -- only the aggregate family COUNT (27, from the
separate §4.5 MECE family-consistency check) has been cross-checked so
far, which is a much weaker guarantee than a pair-level diff.

Why this matters now specifically: every downstream number in the current
report (violation rates, gap/spread ratios, PnL backtest, sensitivity
sweep, out-of-sample replication) was computed on the regex method's pair
list. If the regex method systematically includes spurious pairs (e.g. two
tickers that share a ticker-prefix pattern but are NOT actually the same
underlying, the same failure mode that the original title-template method
was specifically built to exclude for combo-prop markets) or misses real
ones, every one of those downstream numbers inherits that error silently.
This matters even more once the pairs become GRAPH EDGES for an STGAT --
a statistical average tolerates some pair noise, but a GAT will happily
learn confident-looking attention weights over wrong edges.

WHAT THIS SCRIPT DOES:
  1. Loads the regex-identified pairs (already computed, from
     pairwise_monotonicity_taker_side_results.parquet -- no need to
     recompute, that file's leg_a/leg_b columns ARE the regex method's
     pair list).
  2. Loads the ORIGINAL validated pairs -- see REQUIRED SETUP below, this
     is the one thing this script cannot do for you, since that logic
     lives only in your pairwise_monotonicity_check.py / event-pred repo.
  3. Diffs the two pair sets: pairs only the regex method found (candidate
     false positives), pairs only the validated method found (candidate
     false negatives / coverage gaps), and pairs both methods agree on.
  4. Reconstructs "families" from each pair list via union-find (a family
     is just the connected component of a chain of adjacent-threshold
     pairs) and compares family counts/boundaries -- this is the
     pair-level analogue of the family-COUNT check already done in §2.
  5. THE ACTUAL ANSWER TO "can arbitrage be reliably identified using the
     current method": recomputes the headline same-side violation rate,
     median gap, and gap/spread ratio restricted to the INTERSECTION
     (pairs both methods agree are real) and compares those numbers
     against the full regex-based numbers already in the report. If
     they're close, the regex method's disagreements (if any) don't
     materially change what you've already reported -- trustworthy enough
     to build STGAT graph edges from directly. If they diverge, the
     regex-only pairs are contributing signal (or noise) that isn't
     independently confirmed, and the report / STGAT training data should
     switch to the validated method (or the intersection) as the source
     of truth.

REQUIRED SETUP: load_validated_ladder_legs() below reproduces the
markets-loading + strike-extraction + title-template-homogeneity +
MIN_OUTCOMES logic from your pasted v3 pairwise_monotonicity_check.py
directly, so there's nothing to import -- just make sure TARGET_MONTHS
and MIN_OUTCOMES here match what you actually ran that script with (they
also need to match whatever months pairwise_monotonicity_taker_side_
results.parquet was built from, or you're comparing pairs from different
windows). If your markets files don't follow the
data/markets/markets_kalshi_{even,odd}/markets_{month}.parquet naming
convention _month_globs() assumes, fix that one function and everything
downstream of it keeps working unchanged.

Deliberately NOT reproduced: the per-day dynamic adjacency your script
computes via .shift(-1).over(["event_ticker", "date"]) (pairs whichever
legs actually traded that specific day, which can differ day to day).
This script instead derives STATIC pairs from the verified leg list --
sort by strike within a family, pair consecutive tickers -- since that's
the fair, apples-to-apples comparison against the regex method (which is
also static). If this first pass comes back clean, redoing it with the
dynamic per-day version is a natural follow-up, not a blocker.
"""

import glob
import re

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

RESULTS_PATH = "pairwise_monotonicity_taker_side_results.parquet"

# Must match whatever months pairwise_monotonicity_taker_side_results.parquet
# was built from, otherwise you're comparing pairs from different windows.
TARGET_MONTHS = ["2025-10", "2025-11"]

# Same threshold your v3 script uses to gate which events count as ladder
# candidates at all. Real -- not a placeholder -- but see the note in
# diff_pair_sets() about why this makes some "regex-only" pairs expected
# rather than a genuine disagreement.
MIN_OUTCOMES = 20

LADDER_KEYWORDS_PATTERN = r"\b(above|below|or higher|or lower|over|under|at least|at most|exceed)\b"
STRIKE_RE = re.compile(
    r"(?:above|below|or higher than|or lower than|over|under|at least|at most|exceed[s]?)\s*\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _extract_strike(title):
    if title is None:
        return None
    m = STRIKE_RE.search(title)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _title_template(title):
    if title is None:
        return None
    m = STRIKE_RE.search(title)
    if not m:
        return None
    return title[:m.start(1)] + "X" + title[m.end(1):]


def _month_globs(month: str):
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return (
        f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet",
        f"data/trades/trades_kalshi_{parity}/trades_{month}.parquet",
    )


def load_validated_ladder_legs() -> pl.DataFrame:
    """
    Reproduces the FAMILY/LEG identification half of your original
    pairwise_monotonicity_check.py (v3, multi-month): markets loading,
    strike extraction, title-template-homogeneity filter, MIN_OUTCOMES
    event-size gate. Returns (ticker, event_ticker, strike) -- one row per
    verified ladder leg.

    Deliberately does NOT reproduce the per-day dynamic adjacency (the
    .shift(-1).over(["event_ticker", "date"]) step in your script, which
    pairs whichever legs actually traded on a given day -- that can differ
    day to day, e.g. if a leg has no trade on a given date it gets skipped
    and its neighbours connect directly). This returns the STATIC
    per-family leg list instead, so pairs get derived the same static way
    the regex method derives them: sort by strike within a family, pair
    consecutive tickers. That's the fair, apples-to-apples comparison for
    a first pass -- if this cross-check comes back clean, the dynamic
    per-day version is a natural follow-up refinement, not a prerequisite.
    """
    markets_paths = []
    for m in TARGET_MONTHS:
        mp, _ = _month_globs(m)
        if glob.glob(mp):
            markets_paths.append(mp)
        else:
            print(f"WARNING: no markets file for {m}: {mp}")

    markets = pl.concat([
        pl.scan_parquet(p).select(["ticker", "event_ticker", "title", "_fetched_at"]).collect()
        for p in markets_paths
    ])
    meta = (
        markets.sort("_fetched_at", descending=True)
        .unique(subset=["ticker"], keep="first")
        .select(["ticker", "event_ticker", "title"])
    )
    event_size = meta.group_by("event_ticker").agg(pl.col("ticker").n_unique().alias("n_legs_total"))
    big_events = event_size.filter(pl.col("n_legs_total") >= MIN_OUTCOMES)

    ladder_legs = (
        meta.join(big_events, on="event_ticker", how="inner")
        .filter(pl.col("title").str.to_lowercase().str.contains(LADDER_KEYWORDS_PATTERN))
        .with_columns(pl.col("title").map_elements(_extract_strike, return_dtype=pl.Float64).alias("strike"))
        .filter(pl.col("strike").is_not_null())
        .with_columns(pl.col("title").map_elements(_title_template, return_dtype=pl.Utf8).alias("template"))
    )
    homogeneous_events = (
        ladder_legs.group_by("event_ticker").agg(pl.col("template").n_unique().alias("n_templates"))
        .filter(pl.col("n_templates") == 1).select("event_ticker")
    )
    return ladder_legs.join(homogeneous_events, on="event_ticker", how="inner").select(
        ["ticker", "event_ticker", "strike"]
    )


def load_validated_pairs() -> list[tuple[str, str]]:
    """Static consecutive-strike pairs within each verified ladder family --
    see load_validated_ladder_legs() docstring for what "static" means here
    and why."""
    legs = load_validated_ladder_legs()
    pairs = []
    for event_ticker in legs["event_ticker"].unique().to_list():
        g = legs.filter(pl.col("event_ticker") == event_ticker).sort("strike")
        tickers = g["ticker"].to_list()
        for a, b in zip(tickers, tickers[1:]):
            pairs.append((a, b))
    return pairs


def union_find_families(pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Assigns each ticker a family id = connected-component id over the
    pair graph. Two tickers are in the same family iff there's a chain of
    pairs connecting them, regardless of how many hops apart."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    roots = {t: find(t) for t in parent}
    root_to_id = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {t: root_to_id[r] for t, r in roots.items()}


def diff_pair_sets(regex_pairs: list[tuple[str, str]], validated_pairs: list[tuple[str, str]]):
    # Compare as unordered frozensets first -- direction (which leg is
    # "lower threshold") is a separate, secondary check below, since a
    # pair found by both methods but in opposite order is a more specific
    # and more interesting disagreement than a pair only one method found.
    regex_unordered = {frozenset(p) for p in regex_pairs}
    validated_unordered = {frozenset(p) for p in validated_pairs}

    only_regex = regex_unordered - validated_unordered
    only_validated = validated_unordered - regex_unordered
    both = regex_unordered & validated_unordered

    # MIN_OUTCOMES caveat: the validated script only considers events with
    # >= MIN_OUTCOMES legs. Split only_regex into "family is big enough that
    # the validated method should have seen it too" (real disagreement) vs.
    # "family is smaller than MIN_OUTCOMES" (expected -- validated script's
    # own size gate excludes it, not a method disagreement).
    regex_family_id = union_find_families(regex_pairs)
    family_sizes: dict[int, int] = {}
    for fid in regex_family_id.values():
        family_sizes[fid] = family_sizes.get(fid, 0) + 1
    ticker_of_pair = {frozenset(p): p for p in regex_pairs}

    only_regex_comparable, only_regex_small_family = set(), set()
    for p in only_regex:
        a, b = ticker_of_pair.get(p, tuple(p))
        fid = regex_family_id.get(a)
        size = family_sizes.get(fid, 0)
        (only_regex_comparable if size >= MIN_OUTCOMES else only_regex_small_family).add(p)

    print("=== Pair-set diff: regex method vs. validated method ===")
    print(f"regex method:     {len(regex_unordered)} unique pairs")
    print(f"validated method: {len(validated_unordered)} unique pairs")
    print(f"agree (intersection): {len(both)}")
    print(f"regex-only, family < {MIN_OUTCOMES} legs (expected -- outside validated script's own scope): "
          f"{len(only_regex_small_family)}")
    print(f"regex-only, family >= {MIN_OUTCOMES} legs (REAL candidate disagreements -- validated method "
          f"should have caught these too): {len(only_regex_comparable)}")
    print(f"validated-only (candidate false negatives / coverage gaps): {len(only_validated)}")
    print()

    if only_regex_comparable:
        print("--- sample regex-only pairs from big-enough families (inspect these by hand first) ---")
        for p in list(only_regex_comparable)[:15]:
            print(f"  {tuple(p)}")
        print()

    if only_validated:
        print("--- sample validated-only pairs (regex method is missing these) ---")
        for p in list(only_validated)[:15]:
            print(f"  {tuple(p)}")
        print()

    # Direction check on the agreed-upon pairs: does "leg_a = lower
    # threshold" agree between the two methods for pairs both found?
    validated_ordered = {frozenset((a, b)): (a, b) for a, b in validated_pairs}
    direction_mismatches = []
    for a, b in regex_pairs:
        key = frozenset((a, b))
        if key in both and key in validated_ordered:
            v_a, v_b = validated_ordered[key]
            if (a, b) != (v_a, v_b):
                direction_mismatches.append((a, b, v_a, v_b))
    if direction_mismatches:
        print(f"WARNING: {len(direction_mismatches)} pairs agree on membership but DISAGREE "
              f"on which leg is the lower threshold -- this flips the sign of the violation "
              f"test for that pair and should be checked first, it's the most concerning "
              f"category of disagreement:")
        for a, b, va, vb in direction_mismatches[:10]:
            print(f"  regex says ({a} < {b}), validated says ({va} < {vb})")
        print()

    return only_regex, only_validated, both


def compare_family_reconstruction(regex_pairs, validated_pairs):
    regex_families = union_find_families(regex_pairs)
    validated_families = union_find_families(validated_pairs)

    n_regex_families = len(set(regex_families.values()))
    n_validated_families = len(set(validated_families.values()))
    print("=== Family reconstruction comparison ===")
    print(f"regex method:     {n_regex_families} families over {len(regex_families)} tickers")
    print(f"validated method: {n_validated_families} families over {len(validated_families)} tickers")
    print("(the report's §2 cross-check only compared this COUNT -- 27 -- against the "
          "independent MECE family reconstruction; it did NOT check whether the actual "
          "ticker-to-family assignments agree pair-by-pair, which is what this script adds.)")
    print()

    # tickers in both methods -- do they land in "the same shaped" family?
    # (can't compare family IDs directly since numbering is arbitrary; instead
    # check that every pair of tickers grouped together by one method is also
    # grouped together by the other, for tickers present in both.)
    common_tickers = set(regex_families) & set(validated_families)
    disagreements = 0
    for t1 in common_tickers:
        for t2 in common_tickers:
            if t1 >= t2:
                continue
            same_in_regex = regex_families[t1] == regex_families[t2]
            same_in_validated = validated_families[t1] == validated_families[t2]
            if same_in_regex != same_in_validated:
                disagreements += 1
    print(f"ticker pairs where family co-membership disagrees between methods: {disagreements}")
    print("(0 is the ideal result -- means every ticker the two methods both touch is grouped "
          "identically, even if one method covers more tickers overall than the other.)")
    print()


def recompute_headline_stats_on_intersection(both_pairs: set):
    """The actual answer to 'can arbitrage be reliably identified using the
    current method': restrict the already-computed results to ONLY the
    pairs both methods agree are real, and compare the headline numbers
    against the full regex-based numbers already in the report."""
    results = pl.read_parquet(RESULTS_PATH)
    results = results.with_columns(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )

    agreed_mask = results.select(
        pl.struct(["leg_a", "leg_b"]).map_elements(
            lambda row: frozenset((row["leg_a"], row["leg_b"])) in both_pairs,
            return_dtype=pl.Boolean,
        ).alias("agreed")
    )["agreed"]
    intersection_results = results.filter(agreed_mask)

    for label, df in (("FULL regex-based (as currently reported)", results),
                       ("INTERSECTION only (both methods agree)", intersection_results)):
        weather = df.filter(pl.col("category") == "weather/climate")
        if weather.height == 0:
            print(f"--- {label}: no weather/climate pairs in this subset ---\n")
            continue
        viol = weather.filter(pl.col("same_side_yes_violation") == True)  # noqa: E712
        rate = weather["same_side_yes_violation"].mean()
        print(f"--- {label} (weather/climate, n={weather.height}) ---")
        print(f"  same-side yes violation rate: {rate}")
        if viol.height:
            gap_spread = (
                viol.with_columns(((pl.col("spread_a") + pl.col("spread_b")) / 2).alias("avg_spread"))
                .select(
                    (pl.col("same_side_yes_gap").median() /
                     ((pl.col("spread_a") + pl.col("spread_b")) / 2).mean()).alias("gap_spread_ratio")
                )
            )
            print(f"  median gap: {viol['same_side_yes_gap'].median()}")
            print(f"  gap/spread ratio (pooled): {gap_spread.item()}")
        print()
    print(
        "If the INTERSECTION numbers above are close to the FULL regex-based numbers, the "
        "regex method's disagreements with the validated method (if any) are not driving the "
        "headline finding -- safe to keep using the regex method's broader pair coverage for "
        "STGAT graph construction. If they diverge meaningfully, switch the report AND the "
        "STGAT graph-edge source to the validated method (or the intersection) instead, since "
        "that would mean some of the current headline signal is coming from pairs that only "
        "the less-rigorous method identified."
    )


def main():
    print("Loading regex-identified pairs from already-computed results...")
    results = pl.read_parquet(RESULTS_PATH)
    regex_pairs = list(zip(results["leg_a"].to_list(), results["leg_b"].to_list()))
    print(f"  {len(regex_pairs)} pairs\n")

    print("Loading validated pairs (see REQUIRED SETUP if this fails)...")
    validated_pairs = load_validated_pairs()
    print(f"  {len(validated_pairs)} pairs\n")

    only_regex, only_validated, both = diff_pair_sets(regex_pairs, validated_pairs)
    compare_family_reconstruction(regex_pairs, validated_pairs)
    recompute_headline_stats_on_intersection(both)


if __name__ == "__main__":
    main()