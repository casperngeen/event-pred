import pandas as pd

df = pd.read_parquet("data/markets/markets_kalshi_even/markets_2025-06.parquet")
print(df.columns)
print(df["created_time"])

df_trades = pd.read_parquet("data/trades/trades_kalshi_even/trades_2025-04.parquet")
print(df_trades.columns)
print(df_trades["created_time"])

