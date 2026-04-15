import pandas as pd
import glob, os
from collections import defaultdict

# Input files
markets_files = glob.glob("data/markets/markets_kalshi/*.parquet")
trades_files  = glob.glob("data/trades/trades_kalshi/*.parquet")

# Output dirs
os.makedirs("data/markets/markets_kalshi_odd", exist_ok=True)
os.makedirs("data/markets/markets_kalshi_even", exist_ok=True)
os.makedirs("data/trades/trades_kalshi_odd", exist_ok=True)
os.makedirs("data/trades/trades_kalshi_even", exist_ok=True)

def split_by_month(files, time_col, odd_dir, even_dir, prefix):
    monthly_groups = defaultdict(list)

    # Process each file one at a time
    for f in files:
        df = pd.read_parquet(f)
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=[time_col])

        # Group rows by month
        for month, group in df.groupby(df[time_col].dt.to_period("M")):
            monthly_groups[str(month)].append(group)

    # Write one parquet per month into odd/even folders
    for month_str, groups in monthly_groups.items():
        combined = pd.concat(groups, ignore_index=True)
        if pd.Period(month_str).month % 2 == 1:
            out_file = os.path.join(odd_dir, f"{prefix}_{month_str}.parquet")
        else:
            out_file = os.path.join(even_dir, f"{prefix}_{month_str}.parquet")
        combined.to_parquet(out_file)

# Run for markets and trades
split_by_month(markets_files, "created_time", "data/markets/markets_kalshi_odd", "data/markets/markets_kalshi_even", "markets")
split_by_month(trades_files, "created_time", "data/trades/trades_kalshi_odd", "data/trades/trades_kalshi_even", "trades")