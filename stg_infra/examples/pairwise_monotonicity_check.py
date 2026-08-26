"""
Direction-B check (trades-based, v3 -- multi-month): does the monotonicity
violation rate hold up when we look at actual clock-time proximity between
the two trades, not just "same calendar day"? Wide-gap strikes violating
almost as often as narrow-gap ones (v2's result) suggests same-day is too
loose a window for volatile underlyings (Nasdaq, crypto) -- this checks
that directly.

v3 -> multi-month: originally run on October 2025 only (the report's §4.3
headline finding -- 29.3% violation rate at <15min gap, median $0.08 -- is
an October-only number). This extends the same test to November 2025 too,
to check whether that finding replicates out-of-sample or was an October
quirk. December 2025 is left out of TARGET_MONTHS below since (per the
K-of-N runs) there's no file for it yet -- add it once it exists.

Crash-safety note: markets is the small table here (~1 row per ticker per
month), so it's read in full per month with only the 4 needed columns
projected. Trades is the big table (a print per executed trade across the
whole exchange), so -- same lesson as kof_n_stg_sparsity_demo.py -- each
trades file is scanned lazily, projected to only the 3 columns this test
needs, and filtered down to just the identified ladder legs BEFORE
collecting, one file at a time. Only the (much smaller) ladder-leg subset
of each month's trades ever gets materialised.
"""
import glob
import os
import re

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl

TARGET_MONTHS = ["2025-10", "2025-11"]  # Dec 2025 has no data file yet -- add once it does
MIN_OUTCOMES = 20

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
# 1. Markets: small table, safe to read in full per month, then combine.
# --------------------------------------------------------------------------
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
    .with_columns(pl.col("title").map_elements(extract_strike, return_dtype=pl.Float64).alias("strike"))
    .filter(pl.col("strike").is_not_null())
    .with_columns(pl.col("title").map_elements(title_template, return_dtype=pl.Utf8).alias("template"))
)
homogeneous_events = (
    ladder_legs.group_by("event_ticker").agg(pl.col("template").n_unique().alias("n_templates"))
    .filter(pl.col("n_templates") == 1).select("event_ticker")
)
ladder_legs = ladder_legs.join(homogeneous_events, on="event_ticker", how="inner")
print(f"{ladder_legs.height} legs across {ladder_legs['event_ticker'].n_unique()} verified ladder events\n")

ladder_ticker_list = ladder_legs["ticker"].unique().to_list()


# --------------------------------------------------------------------------
# 2. Trades: the big table. Scan+filter+collect ONE FILE AT A TIME, so a
#    month with millions of trade rows across thousands of unrelated
#    tickers never gets materialised in full -- only rows matching the
#    ladder-leg subset do.
# --------------------------------------------------------------------------
trade_parts = []
for i, p in enumerate(trades_paths):
    lf = pl.scan_parquet(p).select(["ticker", "created_time", "yes_price"])
    lf = lf.filter(pl.col("ticker").is_in(ladder_ticker_list))
    hit = lf.collect()
    print(f"  [{i+1}/{len(trades_paths)}] {os.path.basename(p)}  matched_rows={hit.height}")
    if not hit.is_empty():
        trade_parts.append(hit)
trades = pl.concat(trade_parts) if trade_parts else pl.DataFrame(schema=["ticker", "created_time", "yes_price"])
print(f"\n{trades.height} total ladder-leg trade rows across {len(trades_paths)} month(s)\n")


# --------------------------------------------------------------------------
# 3. Same analysis as before, just running over the combined months.
# --------------------------------------------------------------------------
daily_last_trade = (
    trades.join(ladder_legs.select("ticker"), on="ticker", how="inner")
    .with_columns(pl.col("created_time").dt.date().alias("date"))
    .sort("created_time")
    .group_by(["ticker", "date"])
    .agg([
        pl.col("yes_price").last().alias("close"),
        pl.col("created_time").last().alias("trade_time"),
    ])
)
print(f"{daily_last_trade.height} (ticker, day) rows have an actual trade for a ladder leg")

panel = daily_last_trade.join(ladder_legs.select(["ticker", "event_ticker", "strike"]), on="ticker", how="inner")

edges = (
    panel.sort(["event_ticker", "date", "strike"])
    .with_columns([
        pl.col("strike").shift(-1).over(["event_ticker", "date"]).alias("next_strike"),
        pl.col("close").shift(-1).over(["event_ticker", "date"]).alias("next_close"),
        pl.col("trade_time").shift(-1).over(["event_ticker", "date"]).alias("next_trade_time"),
        pl.col("ticker").shift(-1).over(["event_ticker", "date"]).alias("next_ticker"),
    ])
    .filter(pl.col("next_strike").is_not_null())
    .with_columns([
        ((pl.col("next_close") - pl.col("close")) / 100.0).alias("violation_edge"),
        ((pl.col("next_trade_time") - pl.col("trade_time")).dt.total_seconds().abs() / 3600.0).alias("time_gap_hours"),
        pl.col("date").dt.strftime("%Y-%m").alias("month"),
    ])
)

violations = edges.filter(pl.col("violation_edge") > 0)
print(f"\n{len(violations)}/{len(edges)} adjacent-strike same-day pairs show a monotonicity violation "
      f"(combined across {TARGET_MONTHS})")

bins = [(0, 0.25), (0.25, 1), (1, 4), (4, 12), (12, 24)]


def print_time_gap_table(df, label):
    print(f"\nViolation rate by time gap between the two trades -- {label}:")
    for lo, hi in bins:
        sub_edges = df.filter((pl.col("time_gap_hours") >= lo) & (pl.col("time_gap_hours") < hi))
        sub_violations = sub_edges.filter(pl.col("violation_edge") > 0)
        n = len(sub_edges)
        if n > 0:
            print(f"  {lo:>5.2f}h - {hi:>5.2f}h: {len(sub_violations)}/{n} violate ({len(sub_violations)/n:.1%}), "
                  f"median edge {sub_violations['violation_edge'].median() if len(sub_violations) else float('nan')}")
        else:
            print(f"  {lo:>5.2f}h - {hi:>5.2f}h: 0 pairs")


# Combined (all months together) -- same shape as the original single-month output.
print_time_gap_table(edges, f"ALL MONTHS COMBINED ({', '.join(TARGET_MONTHS)})")

# Per-month breakdown -- this is the part that actually answers "does it
# replicate": if November's numbers look like October's independently,
# that's real out-of-sample confirmation, not just a bigger pooled sample.
for month in sorted(edges["month"].unique().to_list()):
    print_time_gap_table(edges.filter(pl.col("month") == month), f"month = {month} only")

print("\nDone.")