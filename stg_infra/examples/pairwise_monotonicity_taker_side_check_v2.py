"""
pairwise_monotonicity_taker_side_check_corrected.py

Replaces guess_ladder_pairs_from_ticker_names() -- which builds a pair
from ANY two tickers sharing a ticker-prefix, sorted by the numeric
suffix, with zero regard for what kind of contract each leg actually is
-- with a pair-BUILDER that only ever constructs a pair when both legs
are confirmed, same-direction nested outcomes, checked against
yes_sub_title (the field this investigation confirmed is 100% populated
and cleanly structured, unlike the ticker suffix letter which turned out
to mean different things in different categories).

This is the fix the whole pair_type_composition_by_category.py run was
building toward: rather than generating every pair and then filtering out
the invalid ones after the fact (bracket-bracket, mixed-direction,
combo-prop), don't generate them in the first place. Concretely, for each
event_ticker, legs are classified as bracket / upper_tail / lower_tail /
unrecognized from yes_sub_title, grouped by (event_ticker, direction), and
only consecutive same-direction legs (sorted by their extracted numeric
strike) get paired. Bracket legs and unrecognized legs (which includes
every combo-prop ticker seen so far, e.g. KXMVENFLMULTIGAMEEXTENDED, whose
sub_title is a list of player names -- never matches a bracket or tail
phrase at all) never enter a pair.

Phrase coverage was broadened after the elections category revealed
different wording for the same concept: "less than X" (lower tail) and
"X and above" (upper tail), alongside weather's "or below"/"or above".
Economics' discrete policy-action titles ("Cut 25bps", "Maintains rate")
correctly stay unrecognized -- they don't contain any of these phrases,
so there's no risk of a false match there.

WHAT THIS SCRIPT DOES: builds the corrected pair list across ALL
categories from the markets table, runs the existing (unmodified)
check_pair() from pairwise_monotonicity_taker_side_check.py against
trades data using these pairs instead of the old regex-guessed ones, and
writes pairwise_monotonicity_taker_side_results_corrected.parquet --
deliberately a NEW file, not overwriting the original, so the two can be
compared directly rather than losing the audit trail of what changed.

REQUIRED SETUP: trades are now loaded the same way markets already are --
one file per TARGET_MONTHS entry, via the data/trades/trades_kalshi_{even,
odd}/trades_{month}.parquet convention (mirrors data/markets/). Adjust
_trades_month_globs() if your layout differs. This replaces an earlier
version of this script that pointed at a single hardcoded TRADES_PATH --
which is what produced an Oct-only run even though pairs are built from
both Oct and Nov markets.
"""

import glob
import re

import polars as pl

try:
    import pairwise_monotonicity_taker_side_check as check_mod
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from . import pairwise_monotonicity_taker_side_check as check_mod
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

TARGET_MONTHS = ["2025-10", "2025-11"]

UPPER_PHRASES = ("or above", "or higher", "or greater", "or more", "and above")
LOWER_PHRASES = ("or below", "or lower", "or less", "less than")

NUMBER_RE = re.compile(r"([\d,]+(?:\.\d+)?)")


def _month_globs(month: str):
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet"


def _trades_month_globs(month: str):
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/trades/trades_kalshi_{parity}/trades_{month}.parquet"


def load_trades() -> pl.DataFrame:
    """Combines every month in TARGET_MONTHS -- pairs are built from BOTH
    Oct and Nov markets, so checking them against only one month's trades
    (as the last run did, Oct-only) silently drops every violation that
    only shows up in Nov's trading, and understates n for pairs where one
    leg only traded in Nov. Same even/odd parity file convention as
    load_markets(), just under data/trades/ instead of data/markets/."""
    paths = []
    for m in TARGET_MONTHS:
        tp = _trades_month_globs(m)
        if glob.glob(tp):
            paths.append(tp)
        else:
            print(f"WARNING: no trades file for {m}: {tp}")
    if not paths:
        raise FileNotFoundError(
            f"No trades files found for {TARGET_MONTHS} -- check the data/trades/trades_kalshi_{{even,odd}}/ "
            f"trades_{{month}}.parquet convention matches your actual layout."
        )
    return pl.concat([pl.scan_parquet(p).collect() for p in paths])


def load_markets() -> pl.DataFrame:
    paths = []
    for m in TARGET_MONTHS:
        mp = _month_globs(m)
        if glob.glob(mp):
            paths.append(mp)
        else:
            print(f"WARNING: no markets file for {m}: {mp}")
    markets = pl.concat([
        pl.scan_parquet(p).select(["ticker", "event_ticker", "yes_sub_title", "_fetched_at"]).collect()
        for p in paths
    ])
    return (
        markets.sort("_fetched_at", descending=True)
        .unique(subset=["ticker"], keep="first")
        .select(["ticker", "event_ticker", "yes_sub_title"])
    )


def classify_subtitle(sub_title):
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


def extract_number(sub_title):
    if sub_title is None:
        return None
    m = NUMBER_RE.search(sub_title)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def build_valid_pairs(markets: pl.DataFrame) -> list[tuple[str, str]]:
    legs = markets.with_columns([
        pl.col("yes_sub_title").map_elements(classify_subtitle, return_dtype=pl.Utf8).alias("leg_type"),
        pl.col("yes_sub_title").map_elements(extract_number, return_dtype=pl.Float64).alias("strike"),
    ])
    tails = legs.filter(pl.col("leg_type").is_in(["upper_tail", "lower_tail"]) & pl.col("strike").is_not_null())

    # v2 -- the previous version filtered the whole `tails` frame once per
    # distinct (event_ticker, leg_type) combo in a Python loop: fine for
    # one category's worth of events, but O(groups x rows) blows up once
    # every category's event_tickers are in play (potentially tens of
    # thousands of groups here, vs. a few hundred/thousand in earlier,
    # single-category runs). Replaced with a single vectorized sort +
    # group_by/agg pass -- polars does the grouping natively in one pass,
    # and the only Python-level loop left is over the already-small
    # per-group ticker lists, not the full table.
    grouped = (
        tails.sort(["event_ticker", "leg_type", "strike"])
        .group_by(["event_ticker", "leg_type"], maintain_order=True)
        .agg(pl.col("ticker").alias("tickers"))
    )

    pairs = []
    for tickers in grouped["tickers"].to_list():
        for a, b in zip(tickers, tickers[1:]):
            pairs.append((a, b))
    return pairs


def precompute_ticker_stats(trades: pl.DataFrame) -> dict:
    """Single-pass equivalent of check_mod.side_stats() / last_trade_price()
    computed for every ticker in trades up front, instead of re-filtering
    the full trades table on demand per pair (check_pair() does 4 filters
    per pair -- side_stats() x2, last_trade_price() x2 -- which is
    O(pairs x trades) and the other likely source of "taking forever" once
    the pair count spans every category instead of just one)."""
    ticker_col, side_col = check_mod.TICKER_COL, check_mod.SIDE_COL
    price_col, ts_col = check_mod.PRICE_COL, check_mod.TIMESTAMP_COL

    side_agg = (
        trades.group_by([ticker_col, side_col])
        .agg(pl.col(price_col).mean().alias("avg_price"), pl.len().alias("n"))
    )
    stats: dict = {}
    for row in side_agg.iter_rows(named=True):
        t = row[ticker_col]
        stats.setdefault(t, {"yes": {"avg_price": None, "n": 0}, "no": {"avg_price": None, "n": 0}})
        side = row[side_col]
        if side in stats[t]:
            stats[t][side] = {"avg_price": row["avg_price"], "n": row["n"]}

    last_trade = (
        trades.sort(ts_col)
        .group_by(ticker_col, maintain_order=True)
        .agg(pl.col(price_col).last().alias("last_price"))
    )
    last_price = dict(zip(last_trade[ticker_col].to_list(), last_trade["last_price"].to_list()))

    return {"side_stats": stats, "last_price": last_price}


def check_pair_fast(precomputed: dict, leg_a: str, leg_b: str) -> dict:
    """Same output schema as check_mod.check_pair() (so every downstream
    script in this investigation reads it unchanged), computed from O(1)
    dict lookups against precompute_ticker_stats() instead of filtering
    the full trades table per pair."""
    side_stats_map, last_price_map = precomputed["side_stats"], precomputed["last_price"]
    empty = {"yes": {"avg_price": None, "n": 0}, "no": {"avg_price": None, "n": 0}}
    stats_a = side_stats_map.get(leg_a, empty)
    stats_b = side_stats_map.get(leg_b, empty)

    result = {"leg_a": leg_a, "leg_b": leg_b}

    last_a, last_b = last_price_map.get(leg_a), last_price_map.get(leg_b)
    result["naive_violation"] = None if last_a is None or last_b is None else last_b > last_a
    result["naive_gap"] = None if last_a is None or last_b is None else last_b - last_a

    for side in ("yes", "no"):
        pa, na = stats_a[side]["avg_price"], stats_a[side]["n"]
        pb, nb = stats_b[side]["avg_price"], stats_b[side]["n"]
        key = f"same_side_{side}"
        if pa is None or pb is None:
            result[f"{key}_violation"] = None
            result[f"{key}_gap"] = None
        else:
            result[f"{key}_violation"] = pb > pa
            result[f"{key}_gap"] = pb - pa
        result[f"{key}_n_a"] = na
        result[f"{key}_n_b"] = nb
        result[f"{key}_low_n"] = (na < check_mod.MIN_N_PER_SIDE) or (nb < check_mod.MIN_N_PER_SIDE)

    def implied_spread(stats):
        a, b = stats["yes"]["avg_price"], stats["no"]["avg_price"]
        return None if a is None or b is None else abs(a - b)

    spread_a, spread_b = implied_spread(stats_a), implied_spread(stats_b)
    result["spread_a"], result["spread_b"] = spread_a, spread_b
    if spread_a and spread_b and spread_a > 0 and spread_b > 0:
        result["spread_ratio"] = max(spread_a, spread_b) / min(spread_a, spread_b)
    else:
        result["spread_ratio"] = None

    return result


def main():
    print("Building corrected pair list from yes_sub_title (bracket/upper_tail/lower_tail)...")
    markets = load_markets()
    pairs = build_valid_pairs(markets)
    print(f"  {len(pairs)} valid same-direction pairs (vs. the old regex method's numeric-adjacency guess)\n")

    if not pairs:
        print("Zero valid pairs found -- check TARGET_MONTHS / markets path, or that this markets table "
              "actually has yes_sub_title populated the way pairwise_monotonicity_subtitle_check.py confirmed.")
        return

    print("Category breakdown of corrected pairs:")
    pair_categories = pl.DataFrame({"leg_a": [p[0] for p in pairs]}).with_columns(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )
    print(pair_categories.group_by("category").agg(pl.len().alias("n_pairs")).sort("n_pairs", descending=True))
    print()

    print(f"Loading trades for {TARGET_MONTHS}...")
    trades = load_trades()

    print("Precomputing per-ticker stats once (instead of check_pair() filtering the full trades "
          "table twice per pair -- O(pairs x trades) was the other likely slow stage here)...")
    precomputed = precompute_ticker_stats(trades)

    print("Checking all pairs via O(1) lookups against the precomputed stats...")
    rows = [check_pair_fast(precomputed, a, b) for a, b in pairs]

    # Explicit schema instead of letting pl.DataFrame(rows) infer one from a
    # sample of the 283k dicts. check_pair_fast() legitimately mixes None
    # with True/False in the *_violation columns (None when one leg never
    # traded on that side) -- if the first chunk polars samples for
    # inference happens to be all-None, it guesses Null/some other dtype for
    # the column and then chokes the moment it reaches a real bool further
    # down the list. An explicit schema sidesteps inference entirely rather
    # than just widening the sample (infer_schema_length) and hoping the
    # real values are close enough to the front.
    RESULTS_SCHEMA = {
        "leg_a": pl.Utf8,
        "leg_b": pl.Utf8,
        "naive_violation": pl.Boolean,
        "naive_gap": pl.Float64,
        "same_side_yes_violation": pl.Boolean,
        "same_side_yes_gap": pl.Float64,
        "same_side_yes_n_a": pl.Int64,
        "same_side_yes_n_b": pl.Int64,
        "same_side_yes_low_n": pl.Boolean,
        "same_side_no_violation": pl.Boolean,
        "same_side_no_gap": pl.Float64,
        "same_side_no_n_a": pl.Int64,
        "same_side_no_n_b": pl.Int64,
        "same_side_no_low_n": pl.Boolean,
        "spread_a": pl.Float64,
        "spread_b": pl.Float64,
        "spread_ratio": pl.Float64,
    }
    results = pl.DataFrame(rows, schema=RESULTS_SCHEMA)

    def rate(col):
        sub = results.filter(pl.col(col).is_not_null())
        return sub[col].mean() if sub.height else None

    print(f"\npairs checked: {results.height}")
    print(f"same-side ('yes') violation rate: {rate('same_side_yes_violation')}")
    print(f"same-side ('no') violation rate:  {rate('same_side_no_violation')}")

    out_path = "pairwise_monotonicity_taker_side_results_corrected.parquet"
    results.write_parquet(out_path)
    print(f"\nfull corrected results written to {out_path} "
          f"(original pairwise_monotonicity_taker_side_results.parquet left untouched for comparison)")


if __name__ == "__main__":
    main()