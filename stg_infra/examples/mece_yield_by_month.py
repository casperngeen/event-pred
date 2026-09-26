"""
mece_yield_by_month.py

Answers the question the "ten months is too short" objection actually
turns on: how many USABLE MECE basket-snapshots exist per month across
the whole 54-month Kalshi history, and in which categories?

Nothing in this project has measured that. Every window decision so far
has been argued from *market listing counts*
(category_history_audit.py) or *platform volume*
(sports_categorisation_check.py). Neither is the quantity that bounds
the training set. The binding quantity is the number of
(event, day) snapshots where EVERY leg of a genuine single-winner MECE
basket traded -- because that is the unit the STGAT is trained and
scored on.

Why this can go back further than sports does: the MECE identification
in mece_sum_to_one_check.py is category-agnostic. It keeps an event if
(a) every leg is resolved, (b) exactly one leg resolved yes, (c) it is
not a threshold ladder, (d) it is not a combo prop, and (e) its FAMILY
resolves n_yes==1 across >=3 independent instances. Weather temperature
brackets ("high temp 70-72F / 73-75F / ...") satisfy all five, and
weather markets exist from 2021-07. Sports launching in 2025-01
therefore bounds the *multi-category* population, not the mechanism.

If this scan shows weather baskets with full coverage back through, say,
2023, then a weather-scoped MECE study spans 30+ months and the
supervisor's objection is satisfiable. If full coverage collapses before
2025 because the legs were too thin to all trade on one day, the
objection is not satisfiable with this venue, and that is a measured
result rather than an excuse.

METHOD
------
Identification is done ONCE over the pooled markets files, not per
month. That is both cheaper and more correct: the family-consistency
test needs >=3 instances per family, and a per-month run would fail
families that only reach 3 instances across a year.

Coverage is then counted per trades month, reusing the same
daily-last-trade / all-legs-traded definition as
mece_sum_to_one_check.py.

Each check in the logic below has a synthetic counterpart in
--self-test, including a ladder that must be rejected and a family that
must be rejected for inconsistent n_yes.

CAVEAT -- this reimplements mece_sum_to_one_check.py's identification
rather than importing it, because that file is a top-level script with
no importable functions. The predicates are transcribed from it and
unit-tested here, but to be safe: run this with
`--markets-from 2025-10 --markets-to 2025-11` and confirm the candidate
count matches what mece_sum_to_one_check.py printed for the same
window. If they differ, trust that script and tell me.

    python -m stg_infra.examples.mece_yield_by_month --self-test
    python -m stg_infra.examples.mece_yield_by_month --trades-from 2024-01
    python -m stg_infra.examples.mece_yield_by_month            # all 54 months, slow
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl

pl.Config.set_tbl_rows(-1)

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker  # type: ignore

START_MONTH = (2021, 6)
END_MONTH = (2025, 11)

# Transcribed from mece_sum_to_one_check.py -- keep in sync.
MIN_LEGS = 3
MAX_LEGS = 100
MIN_INSTANCES_TO_JUDGE = 3
EXCLUDE_TICKER_SUBSTRINGS = ["SINGLEGAME", "MULTIGAME"]
LADDER_KEYWORDS_PATTERN = (
    r"\b(above|below|or higher|or lower|over|under|at least|at most|exceed)\b"
)
STRIKE_RE = re.compile(
    r"(?:above|below|or higher than|or lower than|over|under|at least|at most|exceed[s]?)"
    r"\s*\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
FAMILY_RE = re.compile(r"^([^-]+)")


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
    return title[: m.start(1)] + "X" + title[m.end(1):]


def event_family(event_ticker):
    m = FAMILY_RE.match(event_ticker)
    return m.group(1) if m else event_ticker


def month_range(start, end):
    y, m = start
    out = []
    while (y, m) <= end:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


def _parity(month: str) -> str:
    return "even" if int(month.split("-")[1]) % 2 == 0 else "odd"


def markets_path(month: str) -> str:
    return f"data/markets/markets_kalshi_{_parity(month)}/markets_{month}.parquet"


def trades_path(month: str) -> str:
    return f"data/trades/trades_kalshi_{_parity(month)}/trades_{month}.parquet"


# ----------------------------------------------------------------------
# identification (pooled across months)
# ----------------------------------------------------------------------

def load_meta(months) -> pl.DataFrame:
    parts, missing = [], []
    for m in months:
        p = markets_path(m)
        if not glob.glob(p):
            missing.append(m)
            continue
        parts.append(
            pl.scan_parquet(p)
            .select(["ticker", "event_ticker", "title", "result", "_fetched_at"])
            .collect()
        )
    if missing:
        print(f"WARNING: no markets file for {len(missing)} month(s): {missing}")
    if not parts:
        raise SystemExit("no markets files found -- is data/ populated and are you in the repo root?")
    return (
        pl.concat(parts)
        .sort("_fetched_at", descending=True)
        .unique(subset=["ticker"], keep="first")
    )


def identify_candidates(meta: pl.DataFrame, verbose=True):
    """Returns (candidates, candidate_legs_df). Mirrors mece_sum_to_one_check.py."""
    event_size = meta.group_by("event_ticker").agg(
        pl.col("ticker").n_unique().alias("n_legs_total")
    )
    big_events = event_size.filter(
        (pl.col("n_legs_total") >= MIN_LEGS) & (pl.col("n_legs_total") <= MAX_LEGS)
    )

    ladder_candidates = (
        meta.join(big_events, on="event_ticker", how="inner")
        .filter(pl.col("title").str.to_lowercase().str.contains(LADDER_KEYWORDS_PATTERN))
        .with_columns(
            pl.col("title").map_elements(extract_strike, return_dtype=pl.Float64).alias("strike")
        )
        .filter(pl.col("strike").is_not_null())
        .with_columns(
            pl.col("title").map_elements(title_template, return_dtype=pl.Utf8).alias("template")
        )
    )
    ladder_events = set(
        ladder_candidates.group_by("event_ticker")
        .agg(pl.col("template").n_unique().alias("n_templates"))
        .filter(pl.col("n_templates") == 1)["event_ticker"]
        .to_list()
    )

    resolved = meta.filter(pl.col("result").is_in(["yes", "no"]))
    resolved_counts = resolved.group_by("event_ticker").agg(
        pl.col("ticker").n_unique().alias("n_resolved"),
        (pl.col("result") == "yes").sum().alias("n_yes"),
    )

    candidates = (
        big_events.join(resolved_counts, on="event_ticker", how="inner")
        .filter(
            (pl.col("n_resolved") == pl.col("n_legs_total")) & (pl.col("n_yes") == 1)
        )
    )
    if ladder_events:
        candidates = candidates.filter(~pl.col("event_ticker").is_in(list(ladder_events)))
    candidates = candidates.filter(
        ~pl.any_horizontal(
            [pl.col("event_ticker").str.contains(s, literal=True)
             for s in EXCLUDE_TICKER_SUBSTRINGS]
        )
    )

    all_resolved_sized = (
        event_size.filter(
            (pl.col("n_legs_total") >= MIN_LEGS) & (pl.col("n_legs_total") <= MAX_LEGS)
        )
        .join(resolved_counts, on="event_ticker", how="inner")
        .filter(pl.col("n_resolved") == pl.col("n_legs_total"))
        .with_columns(
            pl.col("event_ticker").map_elements(event_family, return_dtype=pl.Utf8).alias("family")
        )
    )
    family_stats = all_resolved_sized.group_by("family").agg(
        pl.col("event_ticker").n_unique().alias("n_instances"),
        pl.col("n_yes").min().alias("min_n_yes"),
        pl.col("n_yes").max().alias("max_n_yes"),
    )
    mece_families = family_stats.filter(
        (pl.col("n_instances") >= MIN_INSTANCES_TO_JUDGE)
        & (pl.col("min_n_yes") == 1)
        & (pl.col("max_n_yes") == 1)
    )["family"].to_list()

    candidates = candidates.with_columns(
        pl.col("event_ticker").map_elements(event_family, return_dtype=pl.Utf8).alias("family")
    ).filter(pl.col("family").is_in(mece_families))

    if verbose:
        print(f"  excluded {len(ladder_events)} threshold-ladder event(s)")
        print(f"  {len(mece_families)} consistently single-winner famil(y/ies)")
        print(f"  {candidates.height} candidate MECE event(s), "
              f"{candidates['family'].n_unique()} famil(y/ies)")

    legs = meta.join(candidates.select(["event_ticker"]), on="event_ticker", how="inner").select(
        ["ticker", "event_ticker", "title"]
    )
    return candidates, legs


# ----------------------------------------------------------------------
# per-month coverage
# ----------------------------------------------------------------------

def coverage_by_month(candidates, legs, months):
    leg_tickers = legs["ticker"].unique().to_list()
    leg_to_event = legs.select(["ticker", "event_ticker"])
    sizes = candidates.select(["event_ticker", "n_legs_total", "family"])
    fam_cat = {
        f: classify_ticker(f) for f in candidates["family"].unique().to_list()
    }

    rows = []
    for month in months:
        p = trades_path(month)
        if not glob.glob(p):
            print(f"  {month}: no trades file")
            continue
        hit = (
            pl.scan_parquet(p)
            .select(["ticker", "created_time", "yes_price"])
            .filter(pl.col("ticker").is_in(leg_tickers))
            .collect()
        )
        if hit.is_empty():
            print(f"  {month}: 0 candidate-leg trades")
            continue

        daily = (
            hit.with_columns(pl.col("created_time").dt.date().alias("date"))
            .sort("created_time")
            .group_by(["ticker", "date"])
            .agg(pl.col("yes_price").last().alias("close"),
                 pl.col("created_time").last().alias("trade_time"))
        )
        per_event_day = (
            daily.join(leg_to_event, on="ticker", how="inner")
            .group_by(["event_ticker", "date"])
            .agg(pl.col("ticker").n_unique().alias("legs_traded"),
                 pl.col("close").sum().alias("sum_cents"),
                 pl.col("trade_time").min().alias("t0"),
                 pl.col("trade_time").max().alias("t1"))
            .join(sizes, on="event_ticker", how="inner")
        )
        full = per_event_day.filter(pl.col("legs_traded") == pl.col("n_legs_total"))
        if full.is_empty():
            print(f"  {month}: {per_event_day.height} partial (event,day) rows, 0 FULL baskets")
            rows.append({"month": month, "full_snapshots": 0, "partial_rows": per_event_day.height,
                         "families": 0, "top_category": None, "median_abs_dev_cents": None,
                         "median_gap_hours": None})
            continue

        full = full.with_columns(
            (pl.col("sum_cents") - 100.0).abs().alias("abs_dev_cents"),
            ((pl.col("t1") - pl.col("t0")).dt.total_seconds() / 3600.0).alias("gap_h"),
            pl.col("family").replace_strict(fam_cat, default="other").alias("category"),
        )
        by_cat = (
            full.group_by("category").agg(pl.len().alias("n")).sort("n", descending=True)
        )
        top = by_cat["category"][0]
        rows.append({
            "month": month,
            "full_snapshots": full.height,
            "partial_rows": per_event_day.height,
            "families": full["family"].n_unique(),
            "top_category": f"{top} {by_cat['n'][0] / full.height:.0%}",
            "median_abs_dev_cents": round(float(full["abs_dev_cents"].median()), 2),
            "median_gap_hours": round(float(full["gap_h"].median()), 2),
        })
        print(f"  {month}: {full.height:>6,} FULL baskets  "
              f"({full['family'].n_unique()} families, top {top})")
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def report(df: pl.DataFrame, cat_detail: pl.DataFrame | None = None):
    print("\n" + "=" * 92)
    print("USABLE MECE BASKET-SNAPSHOTS PER MONTH")
    print("=" * 92)
    if df.is_empty():
        print("  nothing found")
        return
    print(df)
    tot = int(df["full_snapshots"].sum())
    nz = df.filter(pl.col("full_snapshots") > 0)
    print(f"\n  total full-basket snapshots: {tot:,}")
    if nz.is_empty():
        print("  no month has a single fully-covered basket.")
        return
    print(f"  months with >=1:    {nz.height} of {df.height}  "
          f"({nz['month'].min()} .. {nz['month'].max()})")
    for bar in (10, 100, 500, 1000):
        sub = df.filter(pl.col("full_snapshots") >= bar)
        if sub.is_empty():
            print(f"  months with >={bar:>5,}: 0")
        else:
            print(f"  months with >={bar:>5,}: {sub.height:>3}  "
                  f"earliest {sub['month'].min()}")

    print("""
  HOW TO READ THIS
  The last line is the answer to 'how many months of data are available'.
  Pick the threshold that makes a month worth training on, and the
  earliest month at or above it is the true start of the window --
  regardless of when a category was first listed. A month with 12
  snapshots is not a month of data.

  If a long run of months clears the bar, the supervisor's objection is
  satisfiable: scope one experiment to the categories that survive back
  that far (weather brackets, most likely) and report the multi-category
  result on the shorter window as a second experiment.

  If coverage collapses before 2025 because legs were too thin to all
  trade on the same day, then no reanalysis creates more months, and the
  limit is a measured property of the venue. That is a defensible
  finding, and a stronger answer than 'the data does not go back'.
""")


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------

def _self_test() -> int:
    fails = []

    def ck(name, cond):
        if not cond:
            fails.append(name)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    print("Self-test: title/ladder predicates")
    ck("strike extracted from 'above 100'", extract_strike("Will BTC be above 100?") == 100.0)
    ck("comma strike parsed", extract_strike("above $70,500 today") == 70500.0)
    ck("no strike in a MECE title", extract_strike("Will the high temp be 70-72F?") is None)
    ck("templates collapse across strikes of one ladder",
       title_template("BTC above 100?") == title_template("BTC above 200?"))
    ck("templates differ across two teams' ladders",
       title_template("Philly over 1.5 goals?") != title_template("LA over 1.5 goals?"))
    ck("family is the pre-hyphen segment", event_family("KXHIGHNY-25JUL04") == "KXHIGHNY")
    ck("family keeps embedded digits (KXNASDAQ100 bug)",
       event_family("KXNASDAQ100-25JUL") == "KXNASDAQ100"
       and event_family("KXNASDAQ100U-25JUL") == "KXNASDAQ100U")

    print("\nSelf-test: identification on synthetic markets")
    rows = []

    def ev(evt, titles, results, fetched=1):
        for i, (t, r) in enumerate(zip(titles, results)):
            rows.append({"ticker": f"{evt}-L{i}", "event_ticker": evt, "title": t,
                         "result": r, "_fetched_at": fetched})

    # a genuine MECE weather family: 3 instances, always exactly one yes
    for d in ("01", "02", "03"):
        ev(f"KXHIGHNY-25JUL{d}",
           ["High 70-72F?", "High 73-75F?", "High 76-78F?"],
           ["yes", "no", "no"])
    # a threshold ladder: homogeneous template, must be excluded
    for d in ("01", "02", "03"):
        ev(f"KXBTCLADDER-25JUL{d}",
           ["BTC above 100?", "BTC above 200?", "BTC above 300?"],
           ["yes", "no", "no"])
    # an inconsistent family: n_yes varies across instances, must be excluded
    ev("KXWOBBLY-25JUL01", ["a?", "b?", "c?"], ["yes", "no", "no"])
    ev("KXWOBBLY-25JUL02", ["a?", "b?", "c?"], ["yes", "yes", "no"])
    ev("KXWOBBLY-25JUL03", ["a?", "b?", "c?"], ["yes", "no", "no"])
    # a combo prop, must be excluded by substring
    for d in ("01", "02", "03"):
        ev(f"KXNBASINGLEGAME-25JUL{d}", ["p1 pts?", "p2 pts?", "p3 pts?"],
           ["yes", "no", "no"])
    # a 2-leg event, below MIN_LEGS
    for d in ("01", "02", "03"):
        ev(f"KXBINARY-25JUL{d}", ["up?", "down?"], ["yes", "no"])

    meta = pl.DataFrame(rows)
    cands, legs = identify_candidates(meta, verbose=False)
    fams = set(cands["family"].to_list())
    ck("genuine weather MECE family kept", "KXHIGHNY" in fams)
    ck("threshold ladder excluded", "KXBTCLADDER" not in fams)
    ck("inconsistent-n_yes family excluded", "KXWOBBLY" not in fams)
    ck("combo prop excluded", "KXNBASINGLEGAME" not in fams)
    ck("2-leg event excluded", "KXBINARY" not in fams)
    ck("exactly one family survives", len(fams) == 1)
    ck("3 candidate events (one per instance)", cands.height == 3)
    ck("legs carry through", legs.height == 9)

    print("\nSelf-test: full-basket coverage counting")
    import datetime as dt
    daily = pl.DataFrame({
        # event A: all 3 legs trade on the same day -> FULL
        "ticker": ["KXHIGHNY-25JUL01-L0", "KXHIGHNY-25JUL01-L1", "KXHIGHNY-25JUL01-L2",
                   # event B: only 2 of 3 legs -> partial, must not count
                   "KXHIGHNY-25JUL02-L0", "KXHIGHNY-25JUL02-L1"],
        "date": [dt.date(2025, 7, 1)] * 3 + [dt.date(2025, 7, 2)] * 2,
        "close": [40.0, 35.0, 30.0, 50.0, 50.0],
    })
    per = (
        daily.join(legs.select(["ticker", "event_ticker"]), on="ticker", how="inner")
        .group_by(["event_ticker", "date"])
        .agg(pl.col("ticker").n_unique().alias("legs_traded"),
             pl.col("close").sum().alias("sum_cents"))
        .join(cands.select(["event_ticker", "n_legs_total"]), on="event_ticker", how="inner")
    )
    full = per.filter(pl.col("legs_traded") == pl.col("n_legs_total"))
    ck("one full basket found", full.height == 1)
    ck("partial basket excluded", per.height == 2)
    ck("deviation is +5c on the full basket",
       abs(float(full["sum_cents"][0]) - 105.0) < 1e-9)

    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets-from", default="2021-06")
    ap.add_argument("--markets-to", default="2025-11")
    ap.add_argument("--trades-from", default=None,
                    help="first month to count coverage in (default: same as markets-from)")
    ap.add_argument("--trades-to", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    all_months = month_range(START_MONTH, END_MONTH)
    mmonths = [m for m in all_months if args.markets_from <= m <= args.markets_to]
    tfrom = args.trades_from or args.markets_from
    tto = args.trades_to or args.markets_to
    tmonths = [m for m in all_months if tfrom <= m <= tto]

    print(f"Identifying MECE candidates from {len(mmonths)} months of markets files "
          f"({mmonths[0]}..{mmonths[-1]}) ...")
    meta = load_meta(mmonths)
    print(f"  {meta.height:,} unique tickers pooled")
    cands, legs = identify_candidates(meta)
    if cands.is_empty():
        raise SystemExit("0 candidate MECE events -- nothing to count.")

    fam_counts = (
        cands.group_by("family")
        .agg(pl.len().alias("n_events"), pl.col("n_legs_total").median().alias("med_legs"))
        .sort("n_events", descending=True)
    )
    fam_counts = fam_counts.with_columns(
        pl.col("family").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )
    print("\n  candidate families (top 25 by event count):")
    print(fam_counts.head(25))

    print(f"\nCounting full-basket coverage across {len(tmonths)} trades months "
          f"({tmonths[0]}..{tmonths[-1]}) ...")
    df = coverage_by_month(cands, legs, tmonths)
    if not df.is_empty():
        df.write_parquet("mece_yield_by_month.parquet")
        print("\nSaved to mece_yield_by_month.parquet")
    report(df)


if __name__ == "__main__":
    main()