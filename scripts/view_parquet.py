from stg.io.loaders import DatasetLoader   
import polars as pl

# Load markets 
markets_loader = DatasetLoader("data/markets/markets_kalshi/markets_120000_130000.parquet", file_format="parquet")
df_markets = markets_loader.load()

print(df_markets.columns)
print(df_markets["created_time"])

# Load trades 
trades_loader = DatasetLoader("data/trades/trades_kalshi/trades_10000_20000.parquet", file_format="parquet")
df_trades = trades_loader.load()

print(df_trades.columns)
print(df_trades["created_time"])
