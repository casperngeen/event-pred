"""
mece_family_key_diagnostic.py

Checks a specific hypothesis about mece_sum_to_one_check.py's family-level
consistency check: FAMILY_RE = r"^([A-Za-z]+)" derives a "family" key by
taking only the leading run of LETTERS from event_ticker, stopping at the
first digit. That's fine for series whose name is pure letters (KXBTC vs
KXBTCD, KXUSDJPY vs KXUSDJPYH -- both variants land on a hyphen either way,
so they stay correctly separated). It breaks for any series whose NAME
itself embeds a digit -- e.g. "KXNASDAQ100" (bracket-basket, confirmed
valid MECE structure via pairwise_monotonicity_subtitle_check_crypto_
financials.py) and "KXNASDAQ100U" (threshold ladder, NOT a valid MECE
structure) both reduce to the same family key "KXNASDAQ", pooling their
n_yes histories together. Since the ladder variant generally does NOT
resolve n_yes==1 (multiple thresholds below settlement resolve yes
simultaneously), that pollutes the merged family's max_n_yes above 1 and
disqualifies the WHOLE family -- including the genuinely valid bracket-
basket instances.

This script reruns the exact same family-consistency logic from
mece_sum_to_one_check.py TWICE -- once with the original letters-only
regex, once with a fixed "everything before the first hyphen" version --
and reports:
  1. How many event_ticker prefixes collide under the old regex that the
     fixed version keeps separate (concrete evidence the merging happens
     at all, not just in the 2 tickers spot-checked by hand).
  2. Which families flip from EXCLUDED to INCLUDED once the fix separates
     them from a polluting sibling.
  3. How many additional MIN_LEGS/MAX_LEGS/ladder/combo-prop-filtered
     candidate events the fix actually recovers -- the real, measured
     impact, not just a plausibility argument.

Does NOT modify mece_sum_to_one_check.py -- this is read-only diagnostics
against the same markets data, safe to run independently.

Run with:  python -m stg_infra.examples.mece_family_key_diagnostic
"""
from __future__ import annotations

import glob
import os
import re

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl

TARGET_MONTHS = ["2025-10", "2025-11"]
MIN_LEGS = 3
MAX_LEGS = 60
MIN_INSTANCES_TO_JUDGE = 3

LADDER_KEYWORDS_PATTERN = r"\b(above|below|or higher|or lower|over|under|at least|at most|exceed)\b"
STRIKE_RE = re.compile(
    r"(?:above|below|or higher than|or lower than|over|under|at least|at most|exceed[s]?)\s*\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
EXCLUDE_TICKER_SUBSTRINGS = ["SINGLEGAME", "MULTIGAME"]

FAMILY_RE_OLD = re.compile(r"^([A-Za-z]+)")  # the version currently in mece_sum_to_one_check.py
FAMILY_RE_NEW = re.compile(r"^([^-]+)")      # proposed fix: everything before the first hyphen


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


def family_old(event_ticker):
    m = FAMILY_RE_OLD.match(event_ticker)
    return m.group(1) if m else event_ticker


def family_new(event_ticker):
    m = FAMILY_RE_NEW.match(event_ticker)
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
big_events = event_size.filter((pl.col("n_legs_total") >= MIN_LEGS) & (pl.col("n_legs_total") <= MAX_LEGS))

# --- ladder exclusion (unchanged from mece_sum_to_one_check.py) ------------
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

# --- resolved-outcome validation (unchanged) -------------------------------
resolved = meta.filter(pl.col("result").is_in(["yes", "no"]))
resolved_counts = resolved.group_by("event_ticker").agg([
    pl.col("ticker").n_unique().alias("n_resolved"),
    (pl.col("result") == "yes").sum().alias("n_yes"),
])

base_candidates = (
    big_events.join(resolved_counts, on="event_ticker", how="inner")
    .filter((pl.col("n_resolved") == pl.col("n_legs_total")) & (pl.col("n_yes") == 1))
)
if ladder_event_tickers:
    base_candidates = base_candidates.filter(~pl.col("event_ticker").is_in(list(ladder_event_tickers)))
base_candidates = base_candidates.filter(
    ~pl.any_horizontal([
        pl.col("event_ticker").str.contains(s, literal=True) for s in EXCLUDE_TICKER_SUBSTRINGS
    ])
)

print(f"Base candidates (fully-resolved, single-winner, non-ladder, non-combo-prop, "
      f"{MIN_LEGS}-{MAX_LEGS} legs): {base_candidates.height}\n")

# --- STEP 1: quantify the raw collision, independent of any pass/fail logic
all_events_meta = event_size.select("event_ticker").with_columns([
    pl.col("event_ticker").map_elements(family_old, return_dtype=pl.Utf8).alias("family_old"),
    pl.col("event_ticker").map_elements(family_new, return_dtype=pl.Utf8).alias("family_new"),
])
collisions = (
    all_events_meta.group_by("family_old")
    .agg(pl.col("family_new").n_unique().alias("n_distinct_family_new"))
    .filter(pl.col("n_distinct_family_new") > 1)
    .sort("n_distinct_family_new", descending=True)
)
print(f"=== STEP 1: family_old keys that actually merge >1 distinct family_new group ===")
print(f"{collisions.height} old-regex family key(s) are merging multiple structurally-different "
      f"ticker prefixes together.\n")
if collisions.height:
    print(collisions.head(20))
    print()
    print("Example event_tickers for the top few collisions:")
    for fam_old in collisions["family_old"].head(5).to_list():
        examples = (
            all_events_meta.filter(pl.col("family_old") == fam_old)
            .select("family_new", "event_ticker").unique(subset=["family_new"], keep="first")
        )
        print(f"\n  family_old={fam_old!r} contains {examples.height} distinct family_new group(s):")
        for row in examples.iter_rows(named=True):
            print(f"      family_new={row['family_new']!r:20s}  e.g. {row['event_ticker']}")
print()


def run_family_check(family_fn, label: str):
    all_resolved_sized = (
        event_size  # NOTE: also fixing the missing MAX_LEGS bound found in mece_sum_to_one_check.py's
                    # all_resolved_sized (it only filtered n_legs_total >= MIN_LEGS, no upper bound) --
                    # both regex variants are run through the SAME (now bounded) base so this comparison
                    # isolates the family-key effect specifically, not conflating it with the other bug.
        .filter((pl.col("n_legs_total") >= MIN_LEGS) & (pl.col("n_legs_total") <= MAX_LEGS))
        .join(resolved_counts, on="event_ticker", how="inner")
        .filter(pl.col("n_resolved") == pl.col("n_legs_total"))
        .with_columns(pl.col("event_ticker").map_elements(family_fn, return_dtype=pl.Utf8).alias("family"))
    )
    family_stats = all_resolved_sized.group_by("family").agg([
        pl.col("event_ticker").n_unique().alias("n_instances"),
        pl.col("n_yes").min().alias("min_n_yes"),
        pl.col("n_yes").max().alias("max_n_yes"),
    ])
    mece_families = set(
        family_stats.filter(
            (pl.col("n_instances") >= MIN_INSTANCES_TO_JUDGE)
            & (pl.col("min_n_yes") == 1)
            & (pl.col("max_n_yes") == 1)
        )["family"].to_list()
    )
    candidates = base_candidates.with_columns(
        pl.col("event_ticker").map_elements(family_fn, return_dtype=pl.Utf8).alias("family")
    ).filter(pl.col("family").is_in(mece_families))
    print(f"[{label}] {len(mece_families)} whitelisted families -> {candidates.height} candidate events")
    return mece_families, candidates


print("=== STEP 2: rerun the family-consistency whitelist both ways ===")
families_old, candidates_old = run_family_check(family_old, "OLD (letters-only)")
families_new, candidates_new = run_family_check(family_new, "NEW (up to first hyphen)")
print()

recovered_families = families_new - families_old
print(f"=== STEP 3: families that flip from EXCLUDED to INCLUDED after the fix ===")
print(f"{len(recovered_families)} famil(y/ies) newly pass the consistency check under the fix.\n")
if recovered_families:
    for fam in sorted(recovered_families):
        example = (
            event_size.filter(pl.col("event_ticker").str.starts_with(fam))
            ["event_ticker"].head(1).to_list()
        )
        print(f"    {fam!r:20s}  e.g. {example[0] if example else '(no example found)'}")
print()

old_evt_set = set(candidates_old["event_ticker"].to_list())
new_evt_set = set(candidates_new["event_ticker"].to_list())
print(f"=== STEP 4: net effect on the actual candidate pool ===")
print(f"OLD regex: {len(old_evt_set)} candidate events")
print(f"NEW regex: {len(new_evt_set)} candidate events")
print(f"Gained (in NEW, not in OLD): {len(new_evt_set - old_evt_set)}")
print(f"Lost   (in OLD, not in NEW): {len(old_evt_set - new_evt_set)}")

gained = new_evt_set - old_evt_set
if gained:
    gained_df = candidates_new.filter(pl.col("event_ticker").is_in(list(gained)))
    print(f"\nCategory-ish breakdown of gained events (by family):")
    print(gained_df.group_by("family").agg(pl.len().alias("n_events")).sort("n_events", descending=True).head(20))

print("\nDone.")