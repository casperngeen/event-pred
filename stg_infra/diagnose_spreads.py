"""
diagnose_spreads.py

Before substituting measured spreads into the ladder cost model, establish
whether the measured-spread file can actually speak to the markets we
traded. Two questions, in this order, because the second is pointless if
the first fails:

  1. COVERAGE. The spreads file is keyed by (leg_a, leg_b) ticker pair.
     Kalshi tickers embed the event date (BTCD-24APR02-17-64250), so a
     pair from September 2025 can only match a row in this file if the
     file covers September 2025. If it does not, an exact-pair join
     silently matches ~nothing and any "measured spread" result is really
     the fallback value in disguise. This script reports the exact-pair,
     exact-ticker, and series-level hit rates separately so the fallback
     can never hide inside a headline.

  2. MAGNITUDE. The ladder strategy's breakeven is ~3c per leg. If the
     measured half-spread is routinely above that, the strategy does not
     clear costs and no amount of model selection changes it. Reported as
     a distribution, not a mean, because a mean over a long tail of
     illiquid legs is not the number that decides anything.

Writes nothing and changes nothing. Read-only diagnostic.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
_DATE_RE = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})")


def series_of(ticker: str) -> str:
    """Everything before the first '-' is the series (BTCD, KXHIGHNY, ...)."""
    return ticker.split("-", 1)[0] if ticker else ""


def yyyymm_of(ticker: str):
    """Kalshi tickers embed the event date in field 2 as YYMMMDD."""
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    m = _DATE_RE.match(parts[1])
    if not m:
        return None
    yy, mon, _dd = m.groups()
    mm = _MONTHS.get(mon)
    if mm is None:
        return None
    return 2000 + int(yy), mm


def _pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def _dist(name, s: pd.Series):
    s = s.dropna()
    if s.empty:
        print(f"  {name:<12} (empty)")
        return
    qs = [s.quantile(q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)]
    print(f"  {name:<12} n={len(s):>9,}  p5={qs[0]:7.3f}  p25={qs[1]:7.3f}  "
          f"med={qs[2]:7.3f}  p75={qs[3]:7.3f}  p95={qs[4]:7.3f}  max={s.max():8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="pairwise_monotonicity_taker_side_results_corrected.parquet")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--chunk-len", type=int, default=84)
    ap.add_argument("--breakeven", type=float, default=3.0,
                    help="cents per leg the ladder strategy survives (from the friction sweep)")
    args = ap.parse_args()

    pq = Path(args.parquet)
    if not pq.is_absolute():
        for cand in (_THIS_DIR / pq, _REPO_ROOT / pq, Path.cwd() / pq):
            if cand.exists():
                pq = cand
                break
    if not pq.exists():
        raise SystemExit(f"spreads parquet not found: {args.parquet}\n"
                         f"  try: find ~ -name 'pairwise_monotonicity_taker_side*'")

    df = pd.read_parquet(pq)
    print("=" * 96)
    print(f"MEASURED-SPREAD FILE: {pq.name}   rows={len(df):,}")
    print("=" * 96)

    # ---- 1. what period does this file cover? -------------------------
    ym = pd.Series([yyyymm_of(t) for t in df["leg_a"].astype(str)])
    ym_ok = ym.dropna()
    if ym_ok.empty:
        print("  WARNING: no parseable event dates in leg_a -- cannot date this file.")
        months_present = Counter()
    else:
        months_present = Counter(ym_ok)
        keys = sorted(months_present)
        print(f"\nEvent-date coverage: {keys[0][0]}-{keys[0][1]:02d} .. "
              f"{keys[-1][0]}-{keys[-1][1]:02d}  ({len(keys)} distinct months)")
        print("  rows per month (top 12):")
        for (y, m), c in sorted(months_present.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {y}-{m:02d}  {c:>9,}")

    study = args.months or PILOT_MONTHS
    study_keys = {(int(s[:4]), int(s[5:7])) for s in study}
    overlap_rows = sum(c for k, c in months_present.items() if k in study_keys)
    print(f"\nRows whose event month falls inside the study period {study}: "
          f"{overlap_rows:,} ({_pct(overlap_rows, len(df))})")
    if overlap_rows == 0:
        print("  >>> NO TEMPORAL OVERLAP. An exact-ticker join will match nothing.")
        print("  >>> Any measured-spread result must come from a SERIES-level fallback,")
        print("  >>> i.e. 'spreads typical of this market family', not 'this market'.")

    # ---- 2. how big are the spreads? ----------------------------------
    print("\nSpread distribution (cents per leg, as stored):")
    _dist("spread_a", df["spread_a"])
    _dist("spread_b", df["spread_b"])
    both = pd.concat([df["spread_a"], df["spread_b"]]).dropna()
    _dist("both legs", both)
    half = both / 2.0
    _dist("half-spread", half)
    print(f"\n  Ladder breakeven from the friction sweep: ~{args.breakeven:.1f}c per leg.")
    for label, s in (("full spread", both), ("half spread", half)):
        share = (s > args.breakeven).mean()
        print(f"    legs whose {label:<12} exceeds breakeven: {100 * share:5.1f}%   "
              f"median {s.median():.3f}c")

    # ---- 3. per-series spreads, for the fallback join -----------------
    df = df.assign(_series=df["leg_a"].astype(str).map(series_of))
    ser = (df.groupby("_series")
             .agg(n=("spread_a", "size"),
                  med_a=("spread_a", "median"),
                  med_b=("spread_b", "median"))
             .sort_values("n", ascending=False))
    ser["med_leg"] = (ser["med_a"] + ser["med_b"]) / 2.0
    print(f"\nPer-series median spread (top 15 of {len(ser)} series):")
    print(f"  {'series':<16} {'rows':>9} {'median spread/leg (c)':>22}")
    for name, row in ser.head(15).iterrows():
        print(f"  {name:<16} {int(row['n']):>9,} {row['med_leg']:>22.3f}")

    # ---- 4. do OUR traded pairs appear at all? ------------------------
    cache = _REPO_ROOT / "cache"
    if not cache.exists():
        print("\n(no cache/ directory -- skipping coverage against traded pairs)")
        return
    print("\n" + "=" * 96)
    print(f"COVERAGE AGAINST ACTUAL {args.split.upper()}-SPLIT LADDER PAIRS")
    print("=" * 96)
    store = MonthlyBundleStore(build_month_paths(study, cache), verbose=False)
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        print(f"  no '{args.split}' chunks -- nothing to check.")
        return

    exact_pairs = set(zip(df["leg_a"].astype(str), df["leg_b"].astype(str)))
    known_tickers = set(df["leg_a"].astype(str)) | set(df["leg_b"].astype(str))
    known_series = set(ser.index)

    seen = hit_pair = hit_ticker = hit_series = 0
    our_series = Counter()
    for c in chunks:
        ct = store.materialize_chunk(c)
        ladder_adj = ct["adjacency_by_type"].get("ladder_monotonic")
        node_ids = ct.get("node_ids")
        if ladder_adj is None or not node_ids:
            del ct
            continue
        for t in range(len(ladder_adj)):
            e = ladder_adj[t]
            if e.edge_index.numel() == 0:
                continue
            for k in range(e.edge_index.shape[1]):
                a = str(node_ids[int(e.edge_index[0, k])])
                b = str(node_ids[int(e.edge_index[1, k])])
                seen += 1
                our_series[series_of(a)] += 1
                if (a, b) in exact_pairs or (b, a) in exact_pairs:
                    hit_pair += 1
                if a in known_tickers and b in known_tickers:
                    hit_ticker += 1
                if series_of(a) in known_series and series_of(b) in known_series:
                    hit_series += 1
        del ct

    print(f"  ladder edges examined:        {seen:>10,}")
    print(f"  exact (leg_a, leg_b) match:   {hit_pair:>10,}  {_pct(hit_pair, seen)}")
    print(f"  both tickers known:           {hit_ticker:>10,}  {_pct(hit_ticker, seen)}")
    print(f"  both series known:            {hit_series:>10,}  {_pct(hit_series, seen)}")
    print("\n  Our ladder series (top 12), with the file's median spread if known:")
    print(f"  {'series':<16} {'our edges':>11} {'file median c/leg':>19}")
    for name, cnt in our_series.most_common(12):
        med = f"{ser.loc[name, 'med_leg']:.3f}" if name in ser.index else "-- NOT IN FILE"
        print(f"  {name:<16} {cnt:>11,} {med:>19}")

    print("\n" + "-" * 96)
    if hit_pair == 0 and hit_series == 0:
        print("  VERDICT: the file cannot inform this study's costs at any level.")
        print("  The flat haircut stays an assumption; report the breakeven, not a point estimate.")
    elif hit_pair == 0:
        print("  VERDICT: no per-market spreads, but SERIES-level spreads are usable.")
        print("  Substitute per-series medians and label the result 'spreads typical of")
        print("  the series', never 'measured spreads for these trades'.")
    else:
        print("  VERDICT: exact per-pair spreads exist for some trades. Use them where")
        print("  available, series median elsewhere, and REPORT THE SPLIT.")


if __name__ == "__main__":
    main()