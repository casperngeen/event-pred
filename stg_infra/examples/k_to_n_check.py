"""
Direction-B check: full-basket sum-to-K test for K-of-N categorical events
(e.g. "Top 6", "Top 4", "relegation", "finalist" -- exactly K outcomes
resolve yes out of N), using trades data instead of the markets file (which
we already established is a single static, mostly-unquoted snapshot per
ticker and cannot support this).

Unlike the pairwise ladder test, this needs EVERY leg of an event to trade
within the same window, not just two adjacent ones -- a much harder
liquidity bar that gets combinatorially worse as N grows. Scoped to
moderate-sized events (20-60 legs) rather than the full N-way universe,
where the biggest clusters (400+ legs) have essentially no chance of
clearing this bar.

Trades give a single yes_price per trade, not separate bid/ask, so this is
a single deviation-from-K signal (sum of last-traded prices vs K), not a
two-sided buy/sell test -- and the timing-quality control uses the MAX
trade-time gap across ALL legs of the snapshot, not just one pair.
"""
import re
import polars as pl

MARKETS_PATH = "data/markets/markets_kalshi_even/markets_2025-10.parquet"
TRADES_PATH = "data/trades/trades_kalshi_even/trades_2025-10.parquet"
MIN_OUTCOMES = 20
MAX_OUTCOMES = 60

LADDER_KEYWORDS_PATTERN = r"\b(above|below|or higher|or lower|over|under|at least|at most|exceed)\b"
AMBIGUOUS_KEYWORDS = ["relegat", "final", "advance", "qualify", "promot", "playoff"]
TOP_N_RE = re.compile(r"top\s*(\d+)", re.IGNORECASE)

def infer_k(title):
    if title is None:
        return 1, "default_k1_unverified"
    m = TOP_N_RE.search(title)
    if m:
        return int(m.group(1)), "top_n_regex"
    lowered = title.lower()
    if any(kw in lowered for kw in AMBIGUOUS_KEYWORDS):
        return 1, "default_k1_UNVERIFIED_ambiguous_title"
    return 1, "default_k1"


markets = pl.read_parquet(MARKETS_PATH)
trades = pl.read_parquet(TRADES_PATH)

meta = (
    markets.sort("_fetched_at", descending=True)
    .unique(subset=["ticker"], keep="first")
    .select(["ticker", "event_ticker", "title"])
)

event_size = meta.group_by("event_ticker").agg(pl.col("ticker").n_unique().alias("n_legs_total"))
candidate_events = event_size.filter(
    (pl.col("n_legs_total") >= MIN_OUTCOMES) & (pl.col("n_legs_total") <= MAX_OUTCOMES)
)

event_titles = meta.group_by("event_ticker").agg(pl.col("title").last().alias("event_title"))
k_table = event_titles.with_columns([
    pl.col("event_title").map_elements(lambda t: infer_k(t)[0], return_dtype=pl.Int64).alias("K"),
    pl.col("event_title").map_elements(lambda t: infer_k(t)[1], return_dtype=pl.Utf8).alias("k_method"),
])

# Exclude events that are actually threshold ladders (handled separately by
# the monotonicity check) -- they'd otherwise get a spurious default K=1 here.
is_ladder = event_titles.with_columns(
    pl.col("event_title").str.to_lowercase().str.contains(LADDER_KEYWORDS_PATTERN).alias("is_ladder")
).filter(pl.col("is_ladder")).select("event_ticker")

k_of_n_events = (
    candidate_events.join(k_table.select(["event_ticker", "K", "k_method"]), on="event_ticker", how="left")
    .join(is_ladder, on="event_ticker", how="anti")
    .filter(pl.col("K") > 1)  # genuine multi-winner events only; single-winner (K=1) would need a separate, lower-cost pass
)
print(f"{k_of_n_events.height} candidate K-of-N events (K>1, {MIN_OUTCOMES}-{MAX_OUTCOMES} legs, non-ladder)")
n_unverified = k_of_n_events.filter(pl.col("k_method").str.contains("UNVERIFIED")).height
print(f"  {n_unverified} of these have an UNVERIFIED default K -- check manually before trusting them")

legs = meta.join(k_of_n_events.select(["event_ticker", "n_legs_total", "K", "k_method"]), on="event_ticker", how="inner")

daily_last_trade = (
    trades.join(legs.select("ticker"), on="ticker", how="inner")
    .with_columns(pl.col("created_time").dt.date().alias("date"))
    .sort("created_time")
    .group_by(["ticker", "date"])
    .agg([
        pl.col("yes_price").last().alias("close"),
        pl.col("created_time").last().alias("trade_time"),
    ])
)
print(f"{daily_last_trade.height} (ticker, day) rows have an actual trade for a K-of-N leg")

panel = daily_last_trade.join(legs, on="ticker", how="inner")

snapshots = (
    panel.group_by(["event_ticker", "date", "n_legs_total", "K", "k_method"])
    .agg([
        pl.col("ticker").n_unique().alias("n_legs_quoted"),
        pl.col("close").sum().alias("sum_close_cents"),
        (pl.col("trade_time").max() - pl.col("trade_time").min()).dt.total_seconds().alias("max_time_gap_seconds"),
    ])
    .filter(pl.col("n_legs_quoted") == pl.col("n_legs_total"))  # every leg traded that day
    .with_columns([
        (pl.col("sum_close_cents") / 100.0).alias("sum_close"),
        (pl.col("max_time_gap_seconds") / 3600.0).alias("max_time_gap_hours"),
    ])
    .with_columns((pl.col("sum_close") - pl.col("K")).alias("deviation_from_k"))
)

print(f"\n{len(snapshots)} (event, day) snapshots have EVERY leg of the event trading that day")
if len(snapshots) > 0:
    print(snapshots.sort("max_time_gap_hours")
          .select(["event_ticker", "date", "n_legs_total", "K", "k_method",
                    "sum_close", "deviation_from_k", "max_time_gap_hours"])
          .head(20))
    print("\nDeviation-from-K distribution:")
    print(snapshots["deviation_from_k"].describe())

    print("\nDeviation magnitude by max time gap across all legs:")
    bins = [(0, 1), (1, 4), (4, 12), (12, 24), (24, 1e9)]
    for lo, hi in bins:
        sub = snapshots.filter((pl.col("max_time_gap_hours") >= lo) & (pl.col("max_time_gap_hours") < hi))
        n = len(sub)
        if n > 0:
            print(f"  {lo:>6.1f}h - {hi if hi < 1e9 else 'inf':>6}h: {n} snapshots, "
                  f"median |deviation| {sub['deviation_from_k'].abs().median():.3f}, "
                  f"mean |deviation| {sub['deviation_from_k'].abs().mean():.3f}")
        else:
            print(f"  {lo:>6.1f}h - {hi if hi < 1e9 else 'inf':>6}h: 0 snapshots")
else:
    print("No fully-simultaneous-day snapshots found -- expected, since requiring every leg of a K-of-N "
          "event to trade on the same day is a much harder bar than the pairwise ladder test. A 0 here "
          "would itself be evidence that per-leg (not full-basket) modeling is the right framing for "
          "Direction B across the board, not just for the largest ladders.")