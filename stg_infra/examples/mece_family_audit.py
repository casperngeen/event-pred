"""
mece_family_audit.py

Answers a more basic question than mece_sum_to_one_check.py ever prints an
answer to: out of the WHOLE population of N-way clusters (event_tickers
with 3-60 legs), how much is actually accounted for by a known structure
(threshold ladder / combo-prop / genuine single-winner MECE), and how much
is left over, unclassified? And for the 29 families currently trusted as
genuine single-winner MECE, is that trust actually justified when you look
at the real leg titles -- not just the top-5-by-deviation days the main
script already shows, but every family, including the ones that never
happen to post a large deviation.

Two things this checks that the main script doesn't surface:

1. FUNNEL ACCOUNTING. mece_sum_to_one_check.py never prints the total
   big_events count, so there's no way to tell what fraction of the N-way
   universe ends up in one of the three known buckets (ladder / combo-prop
   / genuine MECE) versus an unexplained leftover (still-open events, or
   fully-resolved multi-winner events that were never assumed to be a
   ladder or a tagged combo-prop -- these ARE the K-of-N-shaped residual,
   or something not yet examined at all).

2. FAMILY-BY-FAMILY EYEBALL. The combo-prop filter is a literal substring
   match on exactly "SINGLEGAME"/"MULTIGAME" -- it removed 4,452 events on
   the last run, ~4x the 1,133 that became base candidates, so it is doing
   enormous, load-bearing work with only two hardcoded keywords. Anything
   shaped like a combo-prop under a DIFFERENT naming convention would slip
   past that filter, leaving the family-consistency check as the only
   remaining defense -- which is a real structural test, but a family of
   genuinely unrelated bundled props COULD still pass it by chance over
   only MIN_INSTANCES_TO_JUDGE=3 instances. This prints a representative
   full leg-list (ticker + title) for every one of the 29 whitelisted
   families, not just the ones that happened to post a large deviation, so
   each one can be manually confirmed as a genuine shared-subject partition
   rather than trusted on the strength of the consistency check alone.

Read-only -- does not modify mece_sum_to_one_check.py.

Run with:  python -m stg_infra.examples.mece_family_audit
"""
from __future__ import annotations

import glob
import os
import re

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl

pl.Config.set_fmt_str_lengths(200)
pl.Config.set_tbl_width_chars(220)

TARGET_MONTHS = ["2025-10", "2025-11"]
MIN_LEGS = 3
MAX_LEGS = 100  # v2 -- was 60; raised after this script found 2,989 events (almost entirely KXDOGE)
                # above 60 legs that would otherwise pass every other single-winner test. Kept in
                # sync with mece_sum_to_one_check.py.
MIN_INSTANCES_TO_JUDGE = 3

LADDER_KEYWORDS_PATTERN = r"\b(above|below|or higher|or lower|over|under|at least|at most|exceed)\b"
STRIKE_RE = re.compile(
    r"(?:above|below|or higher than|or lower than|over|under|at least|at most|exceed[s]?)\s*\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
EXCLUDE_TICKER_SUBSTRINGS = ["SINGLEGAME", "MULTIGAME"]
FAMILY_RE = re.compile(r"^([^-]+)")  # the fixed version, matching mece_sum_to_one_check.py v2


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


def event_family(event_ticker):
    m = FAMILY_RE.match(event_ticker)
    return m.group(1) if m else event_ticker


def _month_globs(month: str):
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet"


markets_paths = []
for m in TARGET_MONTHS:
    mp = _month_globs(m)
    if glob.glob(mp):
        markets_paths.append(mp)
    else:
        print(f"WARNING: no markets file for {m}: {mp}")

markets = pl.concat([
    pl.scan_parquet(p).select(["ticker", "event_ticker", "title", "result", "_fetched_at"]).collect()
    for p in markets_paths
])
meta = markets.sort("_fetched_at", descending=True).unique(subset=["ticker"], keep="first")

event_size = meta.group_by("event_ticker").agg(pl.col("ticker").n_unique().alias("n_legs_total"))
total_events = event_size.height
big_events = event_size.filter((pl.col("n_legs_total") >= MIN_LEGS) & (pl.col("n_legs_total") <= MAX_LEGS))

# --- resolved-outcome validation (moved up so STEP 0b can use it too) ------
resolved = meta.filter(pl.col("result").is_in(["yes", "no"]))
resolved_counts = resolved.group_by("event_ticker").agg([
    pl.col("ticker").n_unique().alias("n_resolved"),
    (pl.col("result") == "yes").sum().alias("n_yes"),
])

print("=" * 70)
print("STEP 1: FUNNEL ACCOUNTING -- where does the whole N-way universe go?")
print("=" * 70)
print(f"Total distinct event_tickers in markets table: {total_events}")
print(f"N-way clusters ({MIN_LEGS}-{MAX_LEGS} legs): {big_events.height} "
      f"({big_events.height/total_events:.1%} of all events)\n")

# --------------------------------------------------------------------------
# STEP 0b: is MAX_LEGS=60 actually cutting off real evidence, or is it a
# non-binding cap? Checks the >60-leg population against the SAME
# fully-resolved/single-winner/non-ladder/non-combo-prop tests used for
# everything else, so this is a real yes/no answer instead of trusting the
# "generous cap, avoids runtime issues" comment in mece_sum_to_one_check.py.
# --------------------------------------------------------------------------
oversized = event_size.filter(pl.col("n_legs_total") > MAX_LEGS)
print(f"Events ABOVE the {MAX_LEGS}-leg cap: {oversized.height}\n")

if oversized.height:
    oversized_resolved = oversized.join(resolved_counts, on="event_ticker", how="left")
    oversized_fully_resolved = oversized_resolved.filter(pl.col("n_resolved") == pl.col("n_legs_total"))
    oversized_single_winner = oversized_fully_resolved.filter(pl.col("n_yes") == 1)

    oversized_ladder_candidates = (
        meta.join(oversized, on="event_ticker", how="inner")
        .filter(pl.col("title").str.to_lowercase().str.contains(LADDER_KEYWORDS_PATTERN))
        .with_columns(pl.col("title").map_elements(extract_strike, return_dtype=pl.Float64).alias("strike"))
        .filter(pl.col("strike").is_not_null())
        .with_columns(pl.col("title").map_elements(title_template, return_dtype=pl.Utf8).alias("template"))
    )
    oversized_ladder_tickers = set(
        oversized_ladder_candidates.group_by("event_ticker")
        .agg(pl.col("template").n_unique().alias("n_templates"))
        .filter(pl.col("n_templates") == 1)
        ["event_ticker"].to_list()
    )
    oversized_is_ladder = (
        pl.col("event_ticker").is_in(list(oversized_ladder_tickers)) if oversized_ladder_tickers else pl.lit(False)
    )
    oversized_is_combo = pl.any_horizontal([
        pl.col("event_ticker").str.contains(s, literal=True) for s in EXCLUDE_TICKER_SUBSTRINGS
    ])
    oversized_would_be_candidate = oversized_single_winner.filter(~oversized_is_ladder & ~oversized_is_combo)

    print(f"  of which fully resolved: {oversized_fully_resolved.height}")
    print(f"  of which fully resolved AND single-winner (n_yes==1): {oversized_single_winner.height}")
    print(f"  of which would ALSO pass non-ladder + non-combo-prop (i.e. would have been a base "
          f"candidate if MAX_LEGS were lifted): {oversized_would_be_candidate.height}\n")

    if oversized_would_be_candidate.height:
        print(f"  MAX_LEGS=60 IS cutting off potential evidence -- {oversized_would_be_candidate.height} "
              f"event(s) below, with their leg counts (still subject to the same family-consistency "
              f"check and liquidity reality as everything else, but worth knowing about):")
        print(
            oversized_would_be_candidate.select("event_ticker", "n_legs_total")
            .sort("n_legs_total")
            .head(20)
        )
    else:
        print(f"  MAX_LEGS=60 is NOT cutting off any potential single-winner MECE evidence in this "
              f"window -- every event above the cap is either still open, multi/zero-winner (the "
              f"K-of-N-shaped residual), a ladder, or a combo-prop. The cap is a safe, non-binding "
              f"guard rather than a real exclusion.")
    print()

# --- ladder exclusion -------------------------------------------------------
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

big_with_resolution = big_events.join(resolved_counts, on="event_ticker", how="left")
not_fully_resolved = big_with_resolution.filter(
    pl.col("n_resolved").is_null() | (pl.col("n_resolved") != pl.col("n_legs_total"))
)
fully_resolved = big_with_resolution.filter(pl.col("n_resolved") == pl.col("n_legs_total"))
fully_resolved_single_winner = fully_resolved.filter(pl.col("n_yes") == 1)
fully_resolved_multi_or_zero_winner = fully_resolved.filter(pl.col("n_yes") != 1)

is_ladder = pl.col("event_ticker").is_in(list(ladder_event_tickers)) if ladder_event_tickers else pl.lit(False)
sw_ladder = fully_resolved_single_winner.filter(is_ladder)
sw_nonladder = fully_resolved_single_winner.filter(~is_ladder)

is_combo = pl.any_horizontal([
    pl.col("event_ticker").str.contains(s, literal=True) for s in EXCLUDE_TICKER_SUBSTRINGS
])
sw_combo = sw_nonladder.filter(is_combo)
sw_base_candidates = sw_nonladder.filter(~is_combo)

print(f"Of the {big_events.height} N-way clusters:")
print(f"  not fully resolved yet (still open, or partially settled): {not_fully_resolved.height}")
print(f"  fully resolved, multi-winner or zero-winner (n_yes != 1)  : {fully_resolved_multi_or_zero_winner.height}"
      f"   <- the K-of-N-shaped / genuinely-not-single-winner residual")
print(f"  fully resolved, single-winner, IS a threshold ladder      : {sw_ladder.height}")
print(f"  fully resolved, single-winner, matches combo-prop keyword : {sw_combo.height}")
print(f"  fully resolved, single-winner, non-ladder, non-combo-prop : {sw_base_candidates.height}  "
      f"<- 'base candidates' before the family-consistency check\n")

# --- family-consistency check (fixed version) -------------------------------
all_resolved_sized = (
    big_events  # already 3-60 legs
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
    .sort("n_instances", descending=True)
)

candidates = sw_base_candidates.with_columns(
    pl.col("event_ticker").map_elements(event_family, return_dtype=pl.Utf8).alias("family")
).filter(pl.col("family").is_in(mece_families["family"].to_list()))

print(f"Family-consistency check: {mece_families.height} families whitelisted "
      f"(>= {MIN_INSTANCES_TO_JUDGE} instances, consistently n_yes==1) "
      f"-> {candidates.height}/{sw_base_candidates.height} base candidates kept "
      f"(dropped {sw_base_candidates.height - candidates.height} whose family only coincidentally "
      f"showed n_yes==1 for that one instance).\n")

print("=" * 70)
print(f"STEP 2: ALL {mece_families.height} WHITELISTED FAMILIES -- eyeball every one, not just the "
      f"highest-deviation days")
print("=" * 70)
print("For each family: n_instances (how many times this consistency check was judged on), and one "
      "representative event's FULL leg list (ticker + title) so you can confirm it's a genuine "
      "shared-subject partition, not a bundle of unrelated props that happened to pass by chance.\n")

candidate_legs_df = meta.filter(pl.col("event_ticker").is_in(candidates["event_ticker"].to_list()))

for row in mece_families.iter_rows(named=True):
    fam = row["family"]
    fam_events = candidates.filter(pl.col("family") == fam)["event_ticker"].to_list()
    if not fam_events:
        print(f"--- family={fam!r}  n_instances={row['n_instances']}  "
              f"(0 of its instances survived to the final candidate pool -- whitelisted but "
              f"contributes no events here) ---\n")
        continue
    example_evt = sorted(fam_events)[0]
    legs = (
        candidate_legs_df.filter(pl.col("event_ticker") == example_evt)
        .select("ticker", "title")
        .sort("ticker")
    )
    print(f"--- family={fam!r}  n_instances={row['n_instances']}  "
          f"n_events_in_final_pool={len(fam_events)}  example={example_evt} ---")
    for leg in legs.iter_rows(named=True):
        print(f"      {leg['ticker']:35s}  {leg['title'] or '(no title)'}")
    print()

print("Done. For each family above: do the leg titles share one clear common subject with "
      "genuinely mutually-exclusive outcomes (a real partition), or do they read as a grab-bag of "
      "unrelated bets that only look bundled because they share an event_ticker? Any family that "
      "reads as the latter should be added to EXCLUDE_TICKER_SUBSTRINGS (or excluded by family name) "
      "in mece_sum_to_one_check.py even though it technically passed the consistency check.")