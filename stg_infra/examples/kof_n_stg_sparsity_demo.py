"""
K-of-N sparsity demonstration -- built on the REAL semester-1 STG pipeline.

Purpose
-------
The earlier `k_to_n_check.py` diagnostic found that, across the 5 candidate
K-of-N categorical events (>=20 legs, non-ladder), only 7 (ticker, day) rows
had an actual trade at all -- and 0 (event, day) snapshots had every leg
trading the same day. That was a hand-rolled script, separate from the
actual STG infrastructure.

This script instead runs the *real* `GraphBuilder` + `KalshiTickerNodes` +
`FixedWindowTemporal` pipeline (the same classes used in kalshi_example.py /
kalshi_traintest.py) on just one K-of-N event's legs, and reads the
sparsity directly off the resulting SpatioTemporalGraph.

Memory/CPU note (v2)
---------------------
The first version of this script loaded the FULL markets + trades tables
(both parities, every month) into memory eagerly before filtering down to
one event -- trades files in particular are the largest table in this
dataset (a print per executed trade, not per ticker), so scanning a full
year of both parities at once is exactly the kind of thing that pegs every
core and can make a modest machine unresponsive. This version instead:

  1. Scans markets file-by-file (lazy, 2-column projection) just to find
     which event_ticker(s) match the 5 known K-of-N candidates -- markets is
     small (recall: ~1 row per ticker per month), so this is cheap.
  2. Only THEN scans trades file-by-file, filtering to just those events'
     ticker lists BEFORE collecting each file -- so a file with millions of
     trade rows for thousands of unrelated tickers never gets materialised
     in full, only the handful of rows that match.
  3. Processes one file at a time (not one big multi-file lazy concat), so
     peak memory/CPU is bounded to a single file's cost, and progress is
     printed per file so you can see where it's spending time if it's slow.

If it's still too heavy on your machine, lower MAX_FILES_PER_GLOB below to
process only the first N files per parity directory (this trades completeness
for speed -- it will UNDER-count trade activity, never over-count).

Run with:  python -m stg_infra.examples.kof_n_stg_sparsity_demo
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional

# Cap Polars' internal thread pool BEFORE importing polars -- this is the
# single biggest lever if the crash was "machine became unresponsive" rather
# than an actual out-of-memory kill. Raise this back up if 4 threads turns
# out to be too conservative for your machine.
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl

from stg.builders.builder import GraphBuilder
from stg.edges.kalshi import KalshiEventEdges
from stg.nodes.kalshi import KalshiTickerNodes
from stg.temporal.strategies import FixedWindowTemporal

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------
# Set to a single "YYYY-MM" (e.g. "2025-10") to restrict the whole run to
# one month's files only -- the fastest, lightest option, but likely
# incomplete for a season-long event (see TARGET_MONTHS below). Takes
# precedence over TARGET_MONTHS if both are set.
TARGET_MONTH: Optional[str] = None  # e.g. "2025-10"

# Set to an explicit list of "YYYY-MM" strings to restrict the run to just
# those months' files -- lighter than scanning your ENTIRE dataset's history,
# while still covering a specific event's full lifetime.
#
# KXEPLTOP6-26's *listed* close_time is 2026-05-25, but the actual dataset
# only has data through Dec 2025 (per your last message) -- Jan-May 2026
# simply don't exist as files yet. Listing them anyway would be harmless
# (glob just finds 0 files and logs a warning, contributing nothing), but
# there's no reason to: capped here at Oct-Dec 2025, i.e. every month that
# can possibly contain data for this event given what you actually have.
#
# If you pick a different / additional candidate event, or your dataset
# grows to cover more months later, widen this list accordingly.
#
# Leave TARGET_MONTH=None and TARGET_MONTHS=None (empty list) to fall back to
# scanning every available file across both parities (heaviest, most complete).
TARGET_MONTHS: Optional[List[str]] = ["2025-10", "2025-11", "2025-12"]


def _month_globs(month: str) -> "tuple[List[str], List[str]]":
    """Given 'YYYY-MM', return ([markets_glob], [trades_glob]) for whichever
    single parity directory that month actually lives in (even month number
    -> *_even, odd -> *_odd -- matching the existing data layout)."""
    year, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return (
        [f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet"],
        [f"data/trades/trades_kalshi_{parity}/trades_{month}.parquet"],
    )


if TARGET_MONTH is not None:
    MARKETS_GLOBS, TRADES_GLOBS = _month_globs(TARGET_MONTH)
    print(f"TARGET_MONTH={TARGET_MONTH} set -- restricting to a single month's files:")
    print(f"  markets: {MARKETS_GLOBS[0]}")
    print(f"  trades:  {TRADES_GLOBS[0]}")
    print("  (season-long K-of-N events may have legs/activity outside this "
          "month -- this run will under-count, not over-count, coverage.)\n")
elif TARGET_MONTHS:
    MARKETS_GLOBS, TRADES_GLOBS = [], []
    for m in TARGET_MONTHS:
        mg, tg = _month_globs(m)
        MARKETS_GLOBS.extend(mg)
        TRADES_GLOBS.extend(tg)
    print(f"TARGET_MONTHS set -- restricting to {len(TARGET_MONTHS)} month(s): "
          f"{TARGET_MONTHS[0]}..{TARGET_MONTHS[-1]}\n")
else:
    MARKETS_GLOBS = [
        "data/markets/markets_kalshi_even/*.parquet",
        "data/markets/markets_kalshi_odd/*.parquet",
    ]
    TRADES_GLOBS = [
        "data/trades/trades_kalshi_even/*.parquet",
        "data/trades/trades_kalshi_odd/*.parquet",
    ]

# The 5 K-of-N candidate events identified by k_to_n_check.py. Matched by
# substring rather than exact equality in case the real event_ticker carries
# a suffix (date/season code) beyond what was printed there.
CANDIDATE_EVENT_SUBSTRINGS = [
    "KXUCLTOP8",
    "KXEPLTOP6",
    "KXEPLRELEGATION",
    "KXLALIGATOP4",
    "KXUCLFINALIST",
]

WINDOW_EVERY = "1d"  # matches the daily granularity used throughout the trades-based checks

# Set to a small int (e.g. 2) to only scan the first N files per glob pattern
# if the full scan is still too slow/heavy. None = scan everything.
MAX_FILES_PER_GLOB: Optional[int] = None

MARKETS_SCAN_COLS = ["ticker", "event_ticker"]
MARKETS_FULL_COLS = ["ticker", "event_ticker", "title", "status", "market_type",
                      "open_time", "close_time", "result", "_fetched_at"]
TRADES_COLS = ["ticker", "created_time", "yes_price", "count", "trade_id", "taker_side"]


def _resolve_files(globs: List[str]) -> List[str]:
    files: List[str] = []
    for g in globs:
        matched = sorted(glob.glob(g))
        if not matched:
            logger.warning("Pattern matched no files: %s", g)
        files.extend(matched)
    if MAX_FILES_PER_GLOB is not None:
        # Applied per-glob-pattern, not to the combined list, so both
        # parities still get represented even under a tight cap.
        capped: List[str] = []
        for g in globs:
            matched = sorted(glob.glob(g))[:MAX_FILES_PER_GLOB]
            capped.extend(matched)
        files = capped
    return sorted(set(files))


# --------------------------------------------------------------------------
# 2. Pass 1 -- find candidate event_tickers by scanning markets file-by-file,
#    projected down to just 2 columns. This never materialises a full month
#    of markets at once, let alone a full year of trades.
# --------------------------------------------------------------------------
def find_candidate_events() -> pl.DataFrame:
    files = _resolve_files(MARKETS_GLOBS)
    print(f"Scanning {len(files)} markets file(s) for K-of-N candidates...")
    parts = []
    for i, f in enumerate(files):
        lf = pl.scan_parquet(f).select(MARKETS_SCAN_COLS)
        lf = lf.filter(
            pl.any_horizontal([
                pl.col("event_ticker").str.contains(s, literal=True)
                for s in CANDIDATE_EVENT_SUBSTRINGS
            ])
        )
        hit = lf.collect()
        if not hit.is_empty():
            parts.append(hit)
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}  matched_rows={hit.height}")
    return pl.concat(parts).unique() if parts else pl.DataFrame(schema=MARKETS_SCAN_COLS)


candidate_hits = find_candidate_events()
if candidate_hits.is_empty():
    raise RuntimeError(
        "None of the 5 known K-of-N candidate event_tickers "
        f"({CANDIDATE_EVENT_SUBSTRINGS}) were found in this markets data. "
        "Re-run k_to_n_check.py against the same data to get current candidates, "
        "or edit CANDIDATE_EVENT_SUBSTRINGS here manually."
    )

all_candidate_tickers = candidate_hits["ticker"].unique().to_list()
print(f"\nFound {candidate_hits['event_ticker'].n_unique()} candidate event(s), "
      f"{len(all_candidate_tickers)} candidate legs total.\n")


# --------------------------------------------------------------------------
# 3. Pass 2 -- scan trades file-by-file, filtering to ONLY the candidate
#    legs found above before collecting each file. This is the step that
#    was previously loading full months of trades into memory.
# --------------------------------------------------------------------------
def scan_candidate_trades() -> pl.DataFrame:
    files = _resolve_files(TRADES_GLOBS)
    print(f"Scanning {len(files)} trades file(s), filtered to {len(all_candidate_tickers)} candidate legs...")
    parts = []
    for i, f in enumerate(files):
        lf = pl.scan_parquet(f)
        avail = lf.collect_schema().names()
        cols = [c for c in TRADES_COLS if c in avail]
        lf = lf.select(cols).filter(pl.col("ticker").is_in(all_candidate_tickers))
        hit = lf.collect()
        if not hit.is_empty():
            parts.append(hit)
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}  matched_rows={hit.height}")
    return pl.concat(parts) if parts else pl.DataFrame(schema=TRADES_COLS)


candidate_trades = scan_candidate_trades()
print(f"\nTotal trades across all candidate events' legs: {candidate_trades.height}\n")


# --------------------------------------------------------------------------
# 4. Pick the K-of-N event with the most trade activity across all legs
#    (i.e. give the STG the *best* case, not a cherry-picked worst case).
# --------------------------------------------------------------------------
def pick_best_event() -> str:
    scored = []
    for evt in candidate_hits["event_ticker"].unique().to_list():
        legs = candidate_hits.filter(pl.col("event_ticker") == evt)["ticker"].unique().to_list()
        n_trades = candidate_trades.filter(pl.col("ticker").is_in(legs)).height
        scored.append((evt, len(legs), n_trades))
    scored.sort(key=lambda r: r[2], reverse=True)
    print("Candidate K-of-N events (event_ticker, n_legs, n_trades_total):")
    for evt, n_legs, n_trades in scored:
        print(f"    {evt:30s}  legs={n_legs:4d}  trades={n_trades:4d}")
    print()
    return scored[0][0]


event_ticker = pick_best_event()
print(f"Selected event for demonstration: {event_ticker}\n")


# --------------------------------------------------------------------------
# 5. Now that we have one small event, re-scan markets ONE more time
#    (still cheap -- filtered to a single event_ticker) to get the full
#    metadata columns KalshiTickerNodes needs (close_time, title, etc.),
#    which we deliberately didn't pull in the wide, all-candidates pass 1.
# --------------------------------------------------------------------------
def load_full_markets_for_event(evt: str) -> pl.DataFrame:
    files = _resolve_files(MARKETS_GLOBS)
    parts = []
    for f in files:
        lf = pl.scan_parquet(f)
        avail = lf.collect_schema().names()
        cols = [c for c in MARKETS_FULL_COLS if c in avail]
        lf = lf.select(cols).filter(pl.col("event_ticker") == evt)
        hit = lf.collect()
        if not hit.is_empty():
            parts.append(hit)
    return pl.concat(parts).unique(subset=["ticker"], keep="last") if parts else pl.DataFrame(schema=MARKETS_FULL_COLS)


event_markets = load_full_markets_for_event(event_ticker)
event_legs: List[str] = event_markets["ticker"].unique().sort().to_list()
total_legs = len(event_legs)

event_trades = candidate_trades.filter(pl.col("ticker").is_in(event_legs))
traded_legs = set(event_trades["ticker"].unique().to_list())
never_traded_legs = [t for t in event_legs if t not in traded_legs]

print(f"Event: {event_ticker}")
print(f"  Total legs listed:        {total_legs}")
print(f"  Legs with >=1 trade ever: {len(traded_legs)}  ({len(traded_legs)/max(total_legs,1):.1%})")
print(f"  Legs with 0 trades ever:  {len(never_traded_legs)}  ({len(never_traded_legs)/max(total_legs,1):.1%})")
if never_traded_legs:
    print(f"  Example never-traded legs: {never_traded_legs[:5]}")
print()

if event_trades.is_empty():
    raise RuntimeError(
        f"Event {event_ticker} has 0 trades across all {total_legs} legs -- "
        "there is nothing to build a graph from. This alone is a complete "
        "(if maximally blunt) demonstration of the coverage problem."
    )


# --------------------------------------------------------------------------
# 6. Build the REAL STG for just this event's legs (small data by now --
#    at most a few hundred legs' worth of trades, not a full year of the
#    whole exchange).
# --------------------------------------------------------------------------
print(f"Building STG (window={WINDOW_EVERY}) from {event_trades.height} trades "
      f"across {len(traded_legs)} traded legs...\n")
stg = (
    GraphBuilder()
    .with_temporal(FixedWindowTemporal(time_col="created_time", every=WINDOW_EVERY))
    .with_nodes(KalshiTickerNodes(markets_df=event_markets))
    # KalshiEventEdges connects every node currently present in a snapshot to
    # every other node of the same event -- i.e. it IS the full-basket graph
    # structure the K-of-N sum-to-K constraint would need. Its edge count per
    # snapshot is a second, equivalent way to see the same sparsity: with L
    # legs present, KalshiEventEdges draws L*(L-1) directed edges -- far short
    # of total_legs*(total_legs-1) whenever L < total_legs.
    .with_edges(KalshiEventEdges())
    .build(event_trades, auxiliary={"markets": event_markets})
)

print(f"Built: {stg}")
print(f"Summary: {stg.summary()}\n")


# --------------------------------------------------------------------------
# 7. Read sparsity directly off the graph
# --------------------------------------------------------------------------
ever_seen = set(stg.all_node_ids())
print("=" * 70)
print("CEILING: legs that appear as a node in AT LEAST ONE snapshot, ever")
print("=" * 70)
print(f"  {len(ever_seen)}/{total_legs} legs ({len(ever_seen)/max(total_legs,1):.1%}) "
      f"ever become a graph node, across all {len(stg)} snapshots combined.")
print(f"  The remaining {total_legs - len(ever_seen)} legs never appear as a node "
      f"in this graph at all, under any window choice.\n")

print("=" * 70)
print(f"PER-SNAPSHOT COVERAGE  (total legs in event = {total_legs})")
print("=" * 70)
max_possible_edges = total_legs * (total_legs - 1)
print(f"{'timestamp':22s} {'nodes_present':>13s} {'coverage':>10s} {'edges':>8s} {'of_max':>8s}")
coverages = []
for snap in stg:
    n_present = snap.num_nodes
    cov = n_present / max(total_legs, 1)
    coverages.append(cov)
    edge_frac = snap.num_edges / max(max_possible_edges, 1)
    print(f"{str(snap.timestamp):22s} {n_present:>13d} {cov:>10.1%} {snap.num_edges:>8d} {edge_frac:>8.1%}")

if coverages:
    import numpy as np
    coverages_arr = np.array(coverages)
    print()
    print(f"Snapshots built: {len(coverages_arr)}")
    print(f"Mean per-snapshot coverage: {coverages_arr.mean():.1%}")
    print(f"Max per-snapshot coverage:  {coverages_arr.max():.1%}")
    print(f"Min per-snapshot coverage:  {coverages_arr.min():.1%}")
    print(
        "\nNote: FixedWindowTemporal only emits a snapshot for a window that has "
        ">=1 trade among these legs at all -- calendar days where NONE of the "
        f"event's {total_legs} legs traded produce no snapshot whatsoever, so "
        "the true sparsity (relative to the event's full calendar lifetime) is "
        "worse than the per-snapshot coverage numbers above suggest on their own."
    )

open_time = event_markets["open_time"].min() if "open_time" in event_markets.columns else None
close_time = event_markets["close_time"].max() if "close_time" in event_markets.columns else None
if open_time is not None and close_time is not None:
    lifetime_days = max((close_time - open_time).total_seconds() / 86400.0, 0.0)
    print(f"\nEvent lifetime: {open_time} -> {close_time}  (~{lifetime_days:.0f} calendar days)")
    print(f"Snapshots actually built: {len(stg)}  "
          f"({len(stg)/max(lifetime_days,1):.1%} of calendar days have ANY trade at all)")

print("\nDone.")