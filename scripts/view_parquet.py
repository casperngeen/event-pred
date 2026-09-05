from stg.io.loaders import DatasetLoader  
import polars as pl

# Load markets 
markets_loader = DatasetLoader("data/markets/markets_kalshi/markets_120000_130000.parquet", file_format="parquet")
df_markets = markets_loader.load()

print(df_markets.columns)
print(df_markets["created_time"])
print(df_markets.head())

# Load trades 
trades_loader = DatasetLoader("data/trades/trades_kalshi/trades_10000_20000.parquet", file_format="parquet")
df_trades = trades_loader.load()

print(df_trades.columns)
print(df_trades["created_time"])
print(df_trades["taker_side"].unique())
print(df_trades.head())

check = (
    df_trades
    .filter(pl.col("ticker") == "KXHIGHLAX-25NOV23-B70")
    .group_by("taker_side")
    .agg(pl.col("yes_price").mean().alias("avg_yes_price"), pl.count())
)
print(check)

import polars as pl

trades = pl.read_parquet("data/trades/trades_kalshi_odd/trades_2025-11.parquet")
print("trades shape:", trades.shape)

busiest = (
    trades.group_by("ticker")
    .agg(pl.len().alias("n_trades"))
    .sort("n_trades", descending=True)
    .head(1)
)
print(busiest)

some_ticker = busiest["ticker"][0]
print("picked ticker:", some_ticker)

check = (
    trades
    .filter(pl.col("ticker") == some_ticker)
    .group_by("taker_side")
    .agg(pl.col("yes_price").mean().alias("avg_yes_price"), pl.len().alias("n_trades"))
)
print(check)

one_day = (
    trades.filter(pl.col("ticker") == "KXMLB-25-LAD")
    .with_columns(pl.col("created_time").dt.date().alias("date"))
)
busiest_day = (
    one_day.group_by("date").agg(pl.len().alias("n")).sort("n", descending=True).head(1)
)
print(busiest_day)

target_date = busiest_day["date"][0]
check_one_day = (
    one_day.filter(pl.col("date") == target_date)
    .group_by("taker_side")
    .agg(pl.col("yes_price").mean().alias("avg_yes_price"), pl.len().alias("n_trades"))
)
print(check_one_day)

import re
import polars as pl

FAMILY_RE = re.compile(r"^([A-Za-z]+)")
MIN_INSTANCES = 3

def event_family(event_ticker):
    if event_ticker is None:
        return None
    m = FAMILY_RE.match(event_ticker)
    return m.group(1) if m else None

resolved = df_markets.filter(pl.col("result").is_in(["yes", "no"]))

n_yes_per_event = (
    resolved.group_by("event_ticker")
    .agg([
        (pl.col("result") == "yes").sum().alias("n_yes"),
        pl.col("ticker").n_unique().alias("n_legs"),
    ])
    .filter((pl.col("n_legs") >= 3) & (pl.col("n_legs") <= 60))
)

families = (
    n_yes_per_event
    .with_columns(pl.col("event_ticker").map_elements(event_family, return_dtype=pl.Utf8).alias("family"))
    .group_by("family")
    .agg([
        pl.col("n_yes").min().alias("min_n_yes"),
        pl.col("n_yes").max().alias("max_n_yes"),
        pl.len().alias("n_instances"),
    ])
    .filter(
        (pl.col("n_instances") >= MIN_INSTANCES)
        & (pl.col("min_n_yes") == 1)
        & (pl.col("max_n_yes") == 1)
    )
    .sort("n_instances", descending=True)
)
print(families)
weather_ticker = "KXHIGHLAX-25NOV03-B70"  # swap for an actual instance from your MECE list, an ordinary calm day

check_calm = (
    trades.filter(pl.col("ticker") == weather_ticker)
    .group_by("taker_side")
    .agg(pl.col("yes_price").mean().alias("avg_yes_price"), pl.len().alias("n_trades"))
)
print(check_calm)

print(df_markets.head())

print(df_markets.columns)
print(df_markets.select("result").unique())

markets_old = pl.read_parquet("data/markets/markets_kalshi_even/markets_2025-10.parquet")
print(markets_old.select("result").unique())

resolved = markets_old.filter(pl.col("result").is_in(["yes", "no"]))

n_yes_per_event = (
    resolved.group_by("event_ticker")
    .agg([
        (pl.col("result") == "yes").sum().alias("n_yes"),
        pl.col("ticker").n_unique().alias("n_legs"),
    ])
    .filter((pl.col("n_legs") >= 3) & (pl.col("n_legs") <= 60))
)

families = (
    n_yes_per_event
    .with_columns(pl.col("event_ticker").map_elements(event_family, return_dtype=pl.Utf8).alias("family"))
    .group_by("family")
    .agg([
        pl.col("n_yes").min().alias("min_n_yes"),
        pl.col("n_yes").max().alias("max_n_yes"),
        pl.len().alias("n_instances"),
    ])
    .filter(
        (pl.col("n_instances") >= MIN_INSTANCES)
        & (pl.col("min_n_yes") == 1)
        & (pl.col("max_n_yes") == 1)
    )
    .sort("n_instances", descending=True)
)
print(families)

family_tickers = (
    trades
    .filter(pl.col("ticker").str.starts_with("KXHIGHMIA"))
    .group_by("ticker")
    .agg(pl.len().alias("n_trades"))
    .sort("n_trades", descending=True)
)
print(family_tickers.head(10))

some_calm_ticker = family_tickers["ticker"][0]

check_calm = (
    trades.filter(pl.col("ticker") == some_calm_ticker)
    .group_by("taker_side")
    .agg(pl.col("yes_price").mean().alias("avg_yes_price"), pl.len().alias("n_trades"))
)
print(check_calm)

import polars as pl

markets = pl.read_parquet("data/markets/markets_kalshi_odd/markets_2026-01.parquet")  # T30499.99 leg is 26JAN01 -- odd month
markets.filter(pl.col("ticker").is_in([
    "KXNASDAQ100Y-26JAN01H1000-T30499.99",
    "KXNASDAQ100Y-26JAN01H1000-B30250",
])).select("ticker", "title")