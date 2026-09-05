"""
Direction-B check: full-basket sum-to-$1 test for the single-winner MECE
family, using TRADES data instead of the markets (quote) file.

Context
-------
Section 4.1's full-basket test (sum-to-1 / sum-to-K) was run against the
markets file and came back inconclusive: that file is one static snapshot
per ticker per month, not a live quote series, so there was almost nothing
to test against. Section 4.4 already redid this swap-to-trades fix for the
K-of-N categorical family and found it doesn't help there either -- not
because trades was the wrong data source, but because K-of-N events simply
don't trade enough legs simultaneously for a full-basket test regardless of
data source.

This script applies the same trades-based fix to the OTHER family §4.1
left untested: single-winner MECE clusters (exactly one leg resolves
"yes" -- e.g. "who wins the election", "which team wins the tournament").
Unlike K-of-N's season-long standings markets (20-36 legs each), MECE
clusters can be much smaller, so it's worth checking empirically whether
they have better per-leg liquidity rather than assuming they hit the same
wall.

Method
------
1. Use each event's RESOLVED outcomes (the `result` field) to identify
   fully-resolved, single-winner (n_yes == 1) clusters -- this needs no
   assumption about quote/trade timing, same as §4.2.
2. Exclude anything that's actually a threshold ladder (same title-template
   homogeneity test used in pairwise_monotonicity_check.py) -- a ladder can
   incidentally resolve with n_yes == 1 if settlement landed below every
   strike, but it isn't a MECE categorical event and shouldn't be counted
   as one here.
3. For the surviving candidates, check trades data for how many days (if
   any) have EVERY leg of the event trading -- the full-basket requirement
   a genuine sum-to-$1 arbitrage test needs -- and what the implied sum is
   on those days.

Crash-safety: same two-pass, filter-before-collecting pattern as
kof_n_stg_sparsity_demo.py / pairwise_monotonicity_check.py. Markets is the
small table (read in full, projected to 5 columns); trades is the big
table, so it's scanned lazily and filtered down to just the candidate
clusters' legs BEFORE collecting, one file at a time.

Run with:  python -m stg_infra.examples.mece_sum_to_one_check

PERFORMANCE NOTE (v2): Section 5's per-event scoring loop originally
filtered the full `trades` table once per candidate event -- the exact
O(events x trades) shape that made the corrected ladder script "take
forever" before it was vectorized (build_valid_pairs() in
pairwise_monotonicity_taker_side_check_corrected.py had the identical
per-group-filter-in-a-loop bug). Replaced below with a single join +
group_by/agg pass; the printed output and the `scored` list's shape
(event_ticker, n_legs, traded_legs, n_trades) are unchanged, so nothing
downstream (best_event selection, the top-20 printout) needed to change.
"""
from __future__ import annotations

import glob
import os
import re

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TARGET_MONTHS = ["2025-10", "2025-11"]  # confirmed actual data range: no Dec 2025
MIN_LEGS = 3    # excludes binary tiers, matches the >=3 definition of "N-way"
# v3 -- was 60 ("generous cap, avoids monster clusters dominating runtime"), but that comment was
# never actually checked against the data. mece_family_audit.py's oversized-event check found 2,989
# events above 60 legs that would have passed every other single-winner/non-ladder/non-combo-prop
# test -- almost entirely KXDOGE (64-67 legs), a crypto bracket-basket family structurally identical
# to the already-validated KXBTC/KXETH/KXSHIBA (39 legs each), just needing more brackets to cover
# Dogecoin's price range at the same granularity. The real gatekeeping here is n_yes==1 plus the
# family-consistency check, not leg count -- K-of-N-shaped multi-winner clusters get excluded on
# that basis regardless of size, so raising this doesn't reopen the door to that noise. Raised to
# 100 to safely clear KXDOGE without removing the bound entirely.
MAX_LEGS = 100

LADDER_KEYWORDS_PATTERN = r"\b(above|below|or higher|or lower|over|under|at least|at most|exceed)\b"
STRIKE_RE = re.compile(
    r"(?:above|below|or higher than|or lower than|over|under|at least|at most|exceed[s]?)\s*\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_strike(title):
    if title is None:
        return None
    m = STRIKE_RE.search(title)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def title_template(title):
    if title is None:
        return None
    m = STRIKE_RE.search(title)
    if not m:
        return None
    return title[:m.start(1)] + "X" + title[m.end(1):]


def _month_globs(month: str):
    year, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return (
        f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet",
        f"data/trades/trades_kalshi_{parity}/trades_{month}.parquet",
    )


markets_paths, trades_paths = [], []
for m in TARGET_MONTHS:
    mp, tp = _month_globs(m)
    if glob.glob(mp):
        markets_paths.append(mp)
    else:
        print(f"WARNING: no markets file for {m}: {mp}")
    if glob.glob(tp):
        trades_paths.append(tp)
    else:
        print(f"WARNING: no trades file for {m}: {tp}")

print(f"Months included: {TARGET_MONTHS}")
print(f"Markets files: {markets_paths}")
print(f"Trades files:  {trades_paths}\n")


# --------------------------------------------------------------------------
# 1. Markets: small table, read in full (projected columns only), combined
#    across months and deduped to the freshest row per ticker.
# --------------------------------------------------------------------------
markets = pl.concat([
    pl.scan_parquet(p).select(["ticker", "event_ticker", "title", "result", "_fetched_at"]).collect()
    for p in markets_paths
])
meta = (
    markets.sort("_fetched_at", descending=True)
    .unique(subset=["ticker"], keep="first")
)

event_size = meta.group_by("event_ticker").agg(pl.col("ticker").n_unique().alias("n_legs_total"))
big_events = event_size.filter(
    (pl.col("n_legs_total") >= MIN_LEGS) & (pl.col("n_legs_total") <= MAX_LEGS)
)

# --------------------------------------------------------------------------
# 2. Identify (and exclude) threshold ladders -- same template-homogeneity
#    test as pairwise_monotonicity_check.py, so a ladder that happens to
#    resolve n_yes==1 doesn't get miscounted as a MECE categorical event.
# --------------------------------------------------------------------------
ladder_candidates = (
    meta.join(big_events, on="event_ticker", how="inner")
    .filter(pl.col("title").str.to_lowercase().str.contains(LADDER_KEYWORDS_PATTERN))
    .with_columns(pl.col("title").map_elements(extract_strike, return_dtype=pl.Float64).alias("strike"))
    .filter(pl.col("strike").is_not_null())
    .with_columns(pl.col("title").map_elements(title_template, return_dtype=pl.Utf8).alias("template"))
)
ladder_event_tickers = set(
    ladder_candidates.group_by("event_ticker")
    .agg(pl.col("template").n_unique().alias("n_templates"))
    .filter(pl.col("n_templates") == 1)
    ["event_ticker"].to_list()
)
print(f"Excluding {len(ladder_event_tickers)} event(s) identified as threshold ladders.\n")

# --------------------------------------------------------------------------
# 3. Resolved-outcome validation: fully-resolved, single-winner (n_yes==1)
#    clusters, excluding ladders. This is the same logic as §4.2's
#    resolved-outcome check, just filtered down to n_yes==1 specifically.
# --------------------------------------------------------------------------
resolved = meta.filter(pl.col("result").is_in(["yes", "no"]))
resolved_counts = resolved.group_by("event_ticker").agg([
    pl.col("ticker").n_unique().alias("n_resolved"),
    (pl.col("result") == "yes").sum().alias("n_yes"),
])

candidates = (
    big_events.join(resolved_counts, on="event_ticker", how="inner")
    .filter(
        (pl.col("n_resolved") == pl.col("n_legs_total"))  # fully resolved
        & (pl.col("n_yes") == 1)                            # single-winner
    )
)
if ladder_event_tickers:
    candidates = candidates.filter(~pl.col("event_ticker").is_in(list(ladder_event_tickers)))

# Exclude the "combo prop" product family (single/multi-game player-prop
# bundles) -- the SAME false-positive category that contaminated the
# pairwise ladder test earlier in this project: these events' "legs" are
# different players' unrelated stat lines bundled under one event_ticker,
# not a genuine mutually-exclusive partition, so there is no structural
# reason their prices should sum to $1. A resolved n_yes==1 among them is
# coincidental, not evidence of a real no-arbitrage constraint -- keeping
# them in would show up as spurious multi-hundred-percent "deviations."
EXCLUDE_TICKER_SUBSTRINGS = ["SINGLEGAME", "MULTIGAME"]
n_before_combo_filter = candidates.height
candidates = candidates.filter(
    ~pl.any_horizontal([
        pl.col("event_ticker").str.contains(s, literal=True) for s in EXCLUDE_TICKER_SUBSTRINGS
    ])
)
print(f"Excluded {n_before_combo_filter - candidates.height} combo-prop event(s) "
      f"(matching {EXCLUDE_TICKER_SUBSTRINGS}).")

# --------------------------------------------------------------------------
# Family-level consistency check.
#
# The title-keyword ladder filter above only catches a ladder if its title
# text literally says "above/over/under/...". It MISSES real ladders that:
#   (a) put the threshold only in the ticker suffix with a generic title
#       (e.g. "KXMLBTOTAL-...-3" / "-4" / "-5", title just "Total Runs?"),
#   (b) use "N+" notation instead of a keyword ("Bam Adebayo records 15+
#       points" -- no "over"/"above" to match), or
#   (c) bundle two interleaved per-team ladders under one event (a "spread"
#       market: Philly's "over 1.5/2.5/3.5" AND LA's "over 1.5/2.5/3.5" in
#       the same event) -- correctly excluded from the LADDER pool by the
#       template-homogeneity check (two team names = two templates), but
#       then wrongly left sitting in the MECE pool instead of being
#       recognised as neither.
#
# A genuine ladder/scaling structure resolving n_yes==1 for one specific
# game is a coincidence of that game's final score (e.g. the margin landed
# in exactly the lowest bucket), not evidence the family is a real
# partition. The reliable test: does this EVENT FAMILY (ticker prefix
# stripped of the per-instance game/date suffix, e.g. "KXMLBTOTAL") resolve
# n_yes==1 on EVERY historical instance, or does it vary (3, 5, 11...
# depending on the game)? Only families that are consistently single-winner
# across multiple independent instances are kept -- this needs no title
# parsing at all, so it catches all three cases above at once.
# --------------------------------------------------------------------------
# v2 -- was r"^([A-Za-z]+)" (leading letters only, stopping at the first
# digit). That works when a series' NAME is pure letters (KXBTC vs KXBTCD,
# KXUSDJPY vs KXUSDJPYH -- both variants hit their hyphen before any digit,
# so they stay correctly separated either way), but breaks for any series
# whose name embeds a digit: "KXNASDAQ100" (bracket-basket, genuine MECE)
# and "KXNASDAQ100U" (threshold ladder, NOT MECE) both used to collapse to
# family "KXNASDAQ", pooling the ladder's variable/multi-winner resolutions
# into the bracket family's stats and disqualifying it entirely.
# mece_family_key_diagnostic.py confirmed this empirically against the real
# data: 13 old-regex keys were merging structurally distinct prefixes, and
# 3 families -- including KXNASDAQ100 -- flipped from excluded to included
# once separated, recovering 58 candidate events with zero regressions.
# Fixed by taking everything before the first hyphen instead, matching the
# actual SERIES-DATE/INSTANCE structure every ticker in this project uses.
FAMILY_RE = re.compile(r"^([^-]+)")


def event_family(event_ticker):
    m = FAMILY_RE.match(event_ticker)
    return m.group(1) if m else event_ticker


MIN_INSTANCES_TO_JUDGE = 3
# v2 -- also bounded by MAX_LEGS now (previously only >= MIN_LEGS, no upper
# bound), matching big_events/candidates. Without this, an oversized event
# that could never itself become a candidate could still contribute its
# n_yes into a family's min/max stats and wrongly swing that family's
# pass/fail verdict.
all_resolved_sized = (
    event_size.filter((pl.col("n_legs_total") >= MIN_LEGS) & (pl.col("n_legs_total") <= MAX_LEGS))
    .join(resolved_counts, on="event_ticker", how="inner")
    .filter(pl.col("n_resolved") == pl.col("n_legs_total"))
    .with_columns(pl.col("event_ticker").map_elements(event_family, return_dtype=pl.Utf8).alias("family"))
)
family_stats = all_resolved_sized.group_by("family").agg([
    pl.col("event_ticker").n_unique().alias("n_instances"),
    pl.col("n_yes").min().alias("min_n_yes"),
    pl.col("n_yes").max().alias("max_n_yes"),
])
mece_families = (
    family_stats.filter(
        (pl.col("n_instances") >= MIN_INSTANCES_TO_JUDGE)
        & (pl.col("min_n_yes") == 1)
        & (pl.col("max_n_yes") == 1)
    )
    ["family"].to_list()
)
print(f"{len(mece_families)} event famil(y/ies) resolve n_yes==1 consistently across "
      f">= {MIN_INSTANCES_TO_JUDGE} historical instances each -- these are the ones trusted "
      f"as genuine single-winner MECE structures.")

n_before_family_filter = candidates.height
candidates = candidates.with_columns(
    pl.col("event_ticker").map_elements(event_family, return_dtype=pl.Utf8).alias("family")
).filter(pl.col("family").is_in(mece_families))
print(f"Family-consistency check: kept {candidates.height}/{n_before_family_filter} candidates "
      f"(dropped events whose family only coincidentally showed n_yes==1 for this instance).\n")

candidate_event_tickers = candidates["event_ticker"].to_list()
print(f"Found {len(candidate_event_tickers)} fully-resolved, single-winner MECE candidate "
      f"event(s) ({MIN_LEGS}-{MAX_LEGS} legs, non-ladder).\n")

if not candidate_event_tickers:
    raise RuntimeError(
        "0 candidate MECE events found -- widen MIN_LEGS/MAX_LEGS, or check that "
        "resolved 'result' values in your markets file are exactly 'yes'/'no'."
    )

candidate_legs_df = meta.filter(pl.col("event_ticker").is_in(candidate_event_tickers))
all_candidate_tickers = candidate_legs_df["ticker"].unique().to_list()
print(f"{len(all_candidate_tickers)} candidate legs total across these events.\n")


# --------------------------------------------------------------------------
# 4. Trades: the big table. Scan+filter+collect ONE FILE AT A TIME, filtered
#    to only the candidate legs, before collecting -- same crash-safety
#    lesson as the other scripts in this series.
# --------------------------------------------------------------------------
trade_parts = []
for i, p in enumerate(trades_paths):
    lf = pl.scan_parquet(p).select(["ticker", "created_time", "yes_price"])
    lf = lf.filter(pl.col("ticker").is_in(all_candidate_tickers))
    hit = lf.collect()
    print(f"  [{i+1}/{len(trades_paths)}] {os.path.basename(p)}  matched_rows={hit.height}")
    if not hit.is_empty():
        trade_parts.append(hit)
trades = pl.concat(trade_parts) if trade_parts else pl.DataFrame(schema=["ticker", "created_time", "yes_price"])
print(f"\n{trades.height} total candidate-leg trade rows across {len(trades_paths)} month(s)\n")

if trades.is_empty():
    raise RuntimeError(
        f"0 trades across all {len(all_candidate_tickers)} candidate legs -- "
        "there is nothing to build a full-basket test from. This alone is a complete "
        "(if maximally blunt) demonstration that the coverage problem extends to the "
        "MECE family too."
    )

daily_last_trade = (
    trades.with_columns(pl.col("created_time").dt.date().alias("date"))
    .sort("created_time")
    .group_by(["ticker", "date"])
    .agg([
        pl.col("yes_price").last().alias("close"),
        pl.col("created_time").last().alias("trade_time"),
    ])
)


# --------------------------------------------------------------------------
# 5. Score each candidate event by trade activity (best-case first), same
#    as k_to_n_check.py / kof_n_stg_sparsity_demo.py.
#
#    v2 -- vectorized. The original version looped over every candidate
#    event and re-filtered the full `trades` table each time (O(events x
#    trades)): the exact shape that made build_valid_pairs() "take forever"
#    in the ladder script before it was fixed the same way there. Replaced
#    with one join (trades -> event_ticker) + one group_by/agg pass; the
#    `scored` list's shape (event_ticker, n_legs, traded_legs, n_trades) is
#    unchanged, so best_event selection and the printout below are untouched.
# --------------------------------------------------------------------------
trades_with_event = trades.join(
    candidate_legs_df.select(["ticker", "event_ticker"]), on="ticker", how="inner"
)
event_trade_stats = (
    trades_with_event.group_by("event_ticker")
    .agg([
        pl.len().alias("n_trades"),
        pl.col("ticker").n_unique().alias("traded_legs"),
    ])
)
scored_df = (
    candidates.select(["event_ticker", "n_legs_total"])
    .join(event_trade_stats, on="event_ticker", how="left")
    .with_columns([
        pl.col("n_trades").fill_null(0),
        pl.col("traded_legs").fill_null(0),
    ])
    .sort("n_trades", descending=True)
)
scored = list(zip(
    scored_df["event_ticker"].to_list(),
    scored_df["n_legs_total"].to_list(),
    scored_df["traded_legs"].to_list(),
    scored_df["n_trades"].to_list(),
))

print("Candidate MECE events (event_ticker, n_legs, legs_ever_traded, n_trades_total):")
for evt, n_legs, traded_legs, n_trades in scored[:20]:
    print(f"    {evt:30s}  legs={n_legs:4d}  traded_legs={traded_legs:4d}  trades={n_trades:5d}")
if len(scored) > 20:
    print(f"    ... and {len(scored) - 20} more candidate event(s)")
print()


# --------------------------------------------------------------------------
# 6a. Full-basket check on the single best-covered candidate -- kept as a
#     concrete, human-readable illustration (this is what turned up the
#     $1.21 example) -- but ONE event's ONE snapshot isn't a distribution,
#     so §6b below aggregates across ALL candidates properly.
# --------------------------------------------------------------------------
best_event = scored[0][0]
best_legs = candidate_legs_df.filter(pl.col("event_ticker") == best_event)["ticker"].to_list()
n_legs_best = len(best_legs)
print(f"Full-basket check on best-covered candidate: {best_event} ({n_legs_best} legs)\n")

panel = daily_last_trade.filter(pl.col("ticker").is_in(best_legs))
per_day = panel.group_by("date").agg([
    pl.col("ticker").n_unique().alias("legs_traded"),
    pl.col("close").sum().alias("sum_cents"),
]).sort("legs_traded", descending=True)

full_snapshots = per_day.filter(pl.col("legs_traded") == n_legs_best)
print(f"{full_snapshots.height} of {per_day.height} trading day(s) have ALL {n_legs_best} legs trading "
      f"(the full-basket requirement for a genuine sum-to-$1 test).")

if full_snapshots.height > 0:
    print("\nFully-covered (event, day) snapshots -- implied sum vs. $1.00:")
    for row in full_snapshots.iter_rows(named=True):
        implied = row["sum_cents"] / 100.0
        print(f"    {row['date']}: sum=${implied:.2f}  (deviation from $1.00: ${implied - 1.0:+.2f})")
else:
    print("\nNo day has every leg trading. Closest approach (top 5 days by coverage):")
    for row in per_day.head(5).iter_rows(named=True):
        cov = row["legs_traded"] / n_legs_best
        print(f"    {row['date']}: {row['legs_traded']}/{n_legs_best} legs traded ({cov:.1%} coverage), "
              f"sum of traded legs' prices=${row['sum_cents']/100.0:.2f}")


# --------------------------------------------------------------------------
# 6b. Full-basket check aggregated across ALL {len(candidate_event_tickers)}
#     candidates -- this is the actual distribution, analogous to the
#     ladder test's thousands of adjacent-strike pairs rather than one
#     anecdote. Also applies the same lesson §4.3 learned the hard way:
#     "same calendar day" isn't the same as "simultaneous" -- each
#     full-basket snapshot's time-gap span (latest last-trade minus
#     earliest last-trade, among that day's legs) is computed and binned,
#     so a wide-spread "full coverage" day (legs trading hours apart)
#     doesn't get counted the same as a tightly-clustered one.
# --------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print(f"FULL-BASKET TEST AGGREGATED ACROSS ALL {len(candidate_event_tickers)} CANDIDATES")
print(f"{'=' * 70}")

panel_all = daily_last_trade.join(
    candidate_legs_df.select(["ticker", "event_ticker"]), on="ticker", how="inner"
)
per_event_day = (
    panel_all.group_by(["event_ticker", "date"])
    .agg([
        pl.col("ticker").n_unique().alias("legs_traded"),
        pl.col("close").sum().alias("sum_cents"),
        pl.col("trade_time").min().alias("min_trade_time"),
        pl.col("trade_time").max().alias("max_trade_time"),
    ])
    .join(candidates.select(["event_ticker", "n_legs_total"]), on="event_ticker", how="inner")
)

full_all = (
    per_event_day.filter(pl.col("legs_traded") == pl.col("n_legs_total"))
    .with_columns([
        (pl.col("sum_cents") / 100.0 - 1.0).alias("deviation"),
        ((pl.col("max_trade_time") - pl.col("min_trade_time")).dt.total_seconds() / 3600.0).alias("time_gap_hours"),
    ])
    .with_columns(pl.col("deviation").abs().alias("abs_deviation"))
)

print(f"{full_all.height} fully-covered (event, day) snapshots found across all candidates "
      f"(out of {per_event_day.height} (event, day) rows with >=1 trade).\n")

if full_all.height == 0:
    print("No fully-covered snapshot exists anywhere in the candidate pool -- the single "
          "example above was the only one found, and even it came from just 2 total "
          "trading days for that event. Treat the earlier $1.21 result as a single "
          "anecdote, not a distribution, until more months of data are available.")
else:
    VIOLATION_THRESHOLD = 0.05  # 5 cents -- an arbitrary but explicit bar for "meaningfully mispriced"
    bins = [(0, 0.25), (0.25, 1), (1, 4), (4, 12), (12, 24), (24, 1e9)]
    print(f"Deviation from $1.00, binned by time-gap span within each full-basket snapshot "
          f"(>${VIOLATION_THRESHOLD:.2f} abs deviation counted as 'meaningfully mispriced'):\n")
    for lo, hi in bins:
        sub = full_all.filter((pl.col("time_gap_hours") >= lo) & (pl.col("time_gap_hours") < hi))
        n = sub.height
        hi_label = "inf" if hi >= 1e9 else f"{hi:>5.2f}h"
        if n > 0:
            n_meaningful = sub.filter(pl.col("abs_deviation") > VIOLATION_THRESHOLD).height
            print(f"  {lo:>5.2f}h - {hi_label}: n={n:4d}  mean|dev|=${sub['abs_deviation'].mean():.3f}  "
                  f"median|dev|=${sub['abs_deviation'].median():.3f}  "
                  f"{n_meaningful}/{n} ({n_meaningful/n:.1%}) exceed ${VIOLATION_THRESHOLD:.2f}")
        else:
            print(f"  {lo:>5.2f}h - {hi_label}: 0 snapshots")

    print("\nTop 10 largest-magnitude full-basket snapshots (any time-gap):")
    top10 = full_all.sort("abs_deviation", descending=True).head(10)
    for row in top10.iter_rows(named=True):
        print(f"    {row['event_ticker']:30s} {str(row['date']):12s} "
              f"sum=${1.0 + row['deviation']:.2f}  dev=${row['deviation']:+.2f}  "
              f"time_gap={row['time_gap_hours']:.2f}h")

    # Leg-level detail for the top 5 -- don't just trust the aggregate number
    # for the biggest examples; look at what's actually inside the basket.
    # A genuine partition reads as one shared subject with mutually exclusive
    # outcomes ("Team A wins" / "Team B wins" / "Team C wins", or "Temp
    # 70-72F" / "73-75F" / ...); a contaminated one reads as a grab-bag of
    # unrelated bets (different players, different stats) that only look
    # like a "basket" because they share an event_ticker.
    print("\nLeg-level detail for the top 5 (sanity-check these by eye):")
    ticker_title = candidate_legs_df.select(["ticker", "title"])
    for row in top10.head(5).iter_rows(named=True):
        evt, day = row["event_ticker"], row["date"]
        print(f"\n  {evt}  {day}  (sum=${1.0 + row['deviation']:.2f}, dev=${row['deviation']:+.2f}):")
        legs_detail = (
            daily_last_trade.filter((pl.col("date") == day))
            .join(candidate_legs_df.filter(pl.col("event_ticker") == evt).select("ticker"), on="ticker", how="inner")
            .join(ticker_title, on="ticker", how="left")
            .sort("close", descending=True)
        )
        for leg in legs_detail.iter_rows(named=True):
            print(f"      ${leg['close']/100.0:.2f}  {leg['ticker']:20s}  {leg['title'] or ''}")

# --------------------------------------------------------------------------
# 7. Persist results for downstream PnL analysis
#    (mece_sum_to_one_pnl_backtest.py), mirroring how
#    pairwise_monotonicity_taker_side_check_corrected.py writes its results
#    parquet for pairwise_monotonicity_pnl_backtest.py to consume.
#
#    Two files, because a basket has N legs (not a fixed 2 like the ladder
#    pairs) -- the PnL script needs both the event-day summary (sum,
#    deviation, time_gap) AND the individual leg prices that made up that
#    sum, to size per-leg taker fees correctly instead of falling back to a
#    flat assumption for every leg.
# --------------------------------------------------------------------------
RESULTS_OUT_PATH = "mece_sum_to_one_results.parquet"
LEG_PRICES_OUT_PATH = "mece_sum_to_one_leg_prices.parquet"

full_all.write_parquet(RESULTS_OUT_PATH)
print(f"\nWrote {full_all.height} full-basket (event, day) snapshot(s) to {RESULTS_OUT_PATH}")

if full_all.height > 0:
    full_keys = full_all.select(["event_ticker", "date"])
    leg_prices = (
        daily_last_trade
        .join(candidate_legs_df.select(["ticker", "event_ticker"]), on="ticker", how="inner")
        .join(full_keys, on=["event_ticker", "date"], how="inner")
        .select(["event_ticker", "date", "ticker", "close", "trade_time"])
    )
    leg_prices.write_parquet(LEG_PRICES_OUT_PATH)
    print(f"Wrote {leg_prices.height} leg-level price row(s) for those snapshots to {LEG_PRICES_OUT_PATH}")
else:
    pl.DataFrame(schema={"event_ticker": pl.Utf8, "date": pl.Date, "ticker": pl.Utf8,
                          "close": pl.Float64, "trade_time": pl.Datetime}).write_parquet(LEG_PRICES_OUT_PATH)
    print(f"Wrote empty leg-price file to {LEG_PRICES_OUT_PATH} (no full-basket snapshots to detail).")

print("\nDone.")