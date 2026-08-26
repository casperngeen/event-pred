import polars as pl

MARKETS_PATH = "data/markets/markets_kalshi_even/markets_2025-10.parquet"
MIN_OUTCOMES = 20

df = pl.read_parquet(MARKETS_PATH)
resolved = df.filter(pl.col("result").is_in(["yes", "no"]))  # unresolved legs tell you nothing yet

event_size = df.group_by("event_ticker").agg(pl.col("ticker").n_unique().alias("n_legs_total"))
big_events = event_size.filter(pl.col("n_legs_total") >= MIN_OUTCOMES)

yes_counts = (
    resolved.join(big_events, on="event_ticker", how="inner")
    .group_by(["event_ticker", "n_legs_total"])
    .agg([
        (pl.col("result") == "yes").sum().alias("n_yes_legs"),
        pl.col("ticker").n_unique().alias("n_legs_resolved"),
        pl.col("title").first().alias("sample_title"),
    ])
    .filter(pl.col("n_legs_resolved") == pl.col("n_legs_total"))  # only fully-settled clusters
)

print(f"{len(yes_counts)} fully-resolved N-way clusters (>= {MIN_OUTCOMES} legs)")
print(yes_counts["n_yes_legs"].value_counts().sort("n_yes_legs"))

print("\nClusters where n_yes_legs looks large/inconsistent (may not be a hard constraint at all):")
print(yes_counts.filter(pl.col("n_yes_legs") > 10)
      .select(["event_ticker", "n_legs_total", "n_yes_legs", "sample_title"])
      .head(10))