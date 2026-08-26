import polars as pl

# 1. Configuration for clear output
pl.Config.set_fmt_str_lengths(1000)  # Don't truncate long tickers
pl.Config.set_tbl_rows(50)           # Show more rows

def analyze_n_way_clusters(markets_path: str):
    # Load the market data
    df = pl.read_parquet(markets_path)
    
    print(f"Analyzing data from: {markets_path}")
    print(f"Total Tickers in file: {len(df)}")

    # 2. Group by Event to count outcomes
    event_counts = (
        df.group_by("event_ticker")
        .agg([
            pl.count("ticker").alias("num_outcomes"),
            pl.col("title").first().alias("sample_title") # To see what the event is
        ])
    )

    # 3. Categorize the events based on the number of outcomes.
    # num_outcomes == 1 is an implicit-binary market (a single ticker carries
    # both yes_bid/yes_ask and no_bid/no_ask -- there's no separate NO
    # ticker). The original ladder started at == 2, so a 1-ticker event
    # matched none of the `when` branches and fell through to `otherwise`,
    # silently mislabeling every implicit-binary event as "Monster N-Way".
    event_distribution = (
        event_counts.with_columns(
            pl.when(pl.col("num_outcomes") == 1)
            .then(pl.lit("0. Implicit Binary (1 ticker)"))
            .when(pl.col("num_outcomes") == 2)
            .then(pl.lit("1. Explicit Binary (2 tickers)"))
            .when(pl.col("num_outcomes").is_between(3, 10))
            .then(pl.lit("2. Small N-Way (3-10)"))
            .when(pl.col("num_outcomes").is_between(11, 50))
            .then(pl.lit("3. Medium N-Way (11-50)"))
            .when(pl.col("num_outcomes").is_between(51, 100))
            .then(pl.lit("4. Large N-Way (51-100)"))
            .otherwise(pl.lit("5. Monster N-Way (100+)"))
            .alias("cluster_type")
        )
        .group_by("cluster_type")
        .agg(pl.count("event_ticker").alias("count_of_events"))
        .sort("cluster_type")
    )

    print("\n--- Market Universe Distribution ---")
    print(event_distribution)

    # 4. Identify the Top 20 High-Dimensional Events
    print("\n--- Top 20 High-Dimensional Events (Direction B Candidates) ---")
    top_monster_events = (
        event_counts.filter(pl.col("num_outcomes") > 50)
        .sort("num_outcomes", descending=True)
        .head(20)
    )
    print(top_monster_events.select(["num_outcomes", "event_ticker", "sample_title"]))

    # 5. Implicit-binary vs explicit-binary vs N-way counts (fixed: these three
    # buckets are now exhaustive and mutually exclusive, so the percentages
    # below actually sum to 100% -- previously num_outcomes == 1 events were
    # excluded from every bucket here even though they're the majority tier.
    total_events = len(event_counts)
    implicit_binary_total = len(event_counts.filter(pl.col("num_outcomes") == 1))
    explicit_binary_total = len(event_counts.filter(pl.col("num_outcomes") == 2))
    n_way_total = len(event_counts.filter(pl.col("num_outcomes") > 2))

    print(f"\n--- Summary ---")
    print(f"Total Unique Events:   {total_events}")
    print(f"Implicit Binary (1):   {implicit_binary_total} ({implicit_binary_total/total_events:.1%})")
    print(f"Explicit Binary (2):   {explicit_binary_total} ({explicit_binary_total/total_events:.1%})")
    print(f"N-Way Clusters (3+):   {n_way_total} ({n_way_total/total_events:.1%})")

if __name__ == "__main__":
    # Update this path to your actual markets folder or file
    PATH = "data/markets/markets_kalshi_even/markets_2025-10.parquet"
    analyze_n_way_clusters(PATH)