"""
Direction-B feasibility check #1 (v3): does the sum-to-K constraint across an
N-way event's legs actually get violated in the data, in a way that could be
profitable net of transaction costs?

v2's "every leg must post a fresh quote within the same 1-hour bucket"
requirement was too strict -- most legs update a handful of times a day or
less, so the odds of all N legs of a big event updating within the same hour
are close to zero, and v2 came back with 0 usable snapshots everywhere.

v3 replaces "same bucket" with an as-of (forward-filled) lookup: for each
evaluation day, every leg contributes its most recent quote *at or before*
that day, however old it is -- but the staleness of that quote (days since
it last updated) is computed and reported explicitly, so you can judge which
"profitable" snapshots reflect genuinely comparable prices across all legs
versus one stale leg distorting the sum, rather than either extreme of v1
(silently allowed any staleness within the same day) or v2 (required none).
"""
import re
import polars as pl

MARKETS_PATH = "data/markets/markets_kalshi_even/markets_2025-10.parquet"
MIN_OUTCOMES = 20
EST_COST_PER_LEG = 0.01
MAX_STALENESS_DAYS = 2  # placeholder -- drop any snapshot where a leg's quote is older than this

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


df = pl.read_parquet(MARKETS_PATH).filter((pl.col("yes_ask") > 0) & (pl.col("yes_bid") > 0))

event_size = df.group_by("event_ticker").agg(pl.col("ticker").n_unique().alias("n_legs_total"))
big_events = event_size.filter(pl.col("n_legs_total") >= MIN_OUTCOMES)

big_tickers = (
    df.join(big_events.select("event_ticker"), on="event_ticker", how="inner")
      .select(["ticker", "event_ticker"]).unique()
)

quotes = (
    df.join(big_tickers.select("ticker"), on="ticker", how="inner")
      .select(["ticker", "_fetched_at", "yes_bid", "yes_ask"])
)

# One evaluation point per calendar day present in the data, at end-of-day
# (implemented as the *start of the next* day, which is equivalent for an
# as-of backward lookup).
days = df["_fetched_at"].dt.date().unique().sort()
fetched_dtype = df.schema["_fetched_at"]  # match eval_time's dtype to the real _fetched_at dtype exactly
clock = pl.DataFrame({"date": days}).with_columns(
    (pl.col("date").cast(pl.Datetime) + pl.duration(days=1)).cast(fetched_dtype).alias("eval_time")
)

ticker_clock = big_tickers.select("ticker").unique().join(clock, how="cross")

asof = (
    ticker_clock.sort("eval_time")
    .join_asof(
        quotes.sort("_fetched_at"),
        left_on="eval_time", right_on="_fetched_at",
        by="ticker", strategy="backward",
    )
    .drop_nulls(["yes_bid", "yes_ask"])  # no quote yet at all for this ticker as of that day
    .with_columns(
        ((pl.col("eval_time") - pl.col("_fetched_at")).dt.total_seconds() / 86400.0).alias("staleness_days")
    )
)

event_titles = df.sort("_fetched_at").group_by("event_ticker").agg(pl.col("title").last().alias("event_title"))
k_table = event_titles.with_columns([
    pl.col("event_title").map_elements(lambda t: infer_k(t)[0], return_dtype=pl.Int64).alias("K"),
    pl.col("event_title").map_elements(lambda t: infer_k(t)[1], return_dtype=pl.Utf8).alias("k_method"),
])

cluster_days = (
    asof.join(big_tickers, on="ticker", how="inner")
    .join(big_events, on="event_ticker", how="inner")
    .join(k_table.select(["event_ticker", "K", "k_method"]), on="event_ticker", how="left")
    .group_by(["event_ticker", "date", "n_legs_total", "K", "k_method"])
    .agg([
        pl.col("ticker").n_unique().alias("n_legs_quoted"),
        pl.col("yes_ask").sum().alias("sum_yes_ask_cents"),
        pl.col("yes_bid").sum().alias("sum_yes_bid_cents"),
        pl.col("staleness_days").max().alias("max_staleness_days"),
    ])
    .filter(
        (pl.col("n_legs_quoted") == pl.col("n_legs_total")) &
        (pl.col("max_staleness_days") <= MAX_STALENESS_DAYS)
    )
    .with_columns([
        (pl.col("sum_yes_ask_cents") / 100.0).alias("sum_yes_ask"),
        (pl.col("sum_yes_bid_cents") / 100.0).alias("sum_yes_bid"),
    ])
    .with_columns([
        (pl.col("K") - pl.col("sum_yes_ask")).alias("buy_basket_edge"),
        (pl.col("sum_yes_bid") - pl.col("K")).alias("sell_basket_edge"),
    ])
)

print(f"{len(cluster_days)} snapshots (every leg quoted within {MAX_STALENESS_DAYS} day(s)) "
      f"across {cluster_days['event_ticker'].n_unique()} events with >= {MIN_OUTCOMES} legs")

n_ambiguous = cluster_days.filter(pl.col("k_method").str.contains("UNVERIFIED")).select("event_ticker").unique().height
print(f"NOTE: {n_ambiguous} distinct events have an UNVERIFIED default K=1 -- check manually.")

for side, edge_col in [("BUY basket (sum yes_ask < K)", "buy_basket_edge"),
                        ("SELL basket (sum yes_bid > K)", "sell_basket_edge")]:
    hits = cluster_days.with_columns(
        (pl.col(edge_col) - pl.col("n_legs_total") * EST_COST_PER_LEG).alias("edge_after_cost")
    ).filter(pl.col("edge_after_cost") > 0)
    print(f"\n{side}: {len(hits)}/{len(cluster_days)} snapshots profitable after cost")
    if len(hits) > 0:
        print(hits.sort("edge_after_cost", descending=True)
              .select(["event_ticker", "date", "n_legs_total", "K", "k_method",
                        edge_col, "edge_after_cost", "max_staleness_days"])
              .head(15))