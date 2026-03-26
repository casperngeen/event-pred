import os
import polars as pl
from collections import defaultdict
from stg_infra.stg.io.loaders import DatasetLoader 

# Input patterns
markets_pattern = "data/markets/markets_kalshi/*.parquet"
trades_pattern  = "data/trades/trades_kalshi/*.parquet"

# Output dirs
os.makedirs("data/markets/markets_kalshi_odd", exist_ok=True)
os.makedirs("data/markets/markets_kalshi_even", exist_ok=True)
os.makedirs("data/trades/trades_kalshi_odd", exist_ok=True)
os.makedirs("data/trades/trades_kalshi_even", exist_ok=True)

def split_by_month(loader: DatasetLoader, time_col: str, odd_dir: str, even_dir: str, prefix: str):
    monthly_groups = defaultdict(list)

    for chunk in loader.load_iter(files_per_batch=10):
        # Normalize timezone and drop nulls
        chunk = chunk.with_columns(
            pl.col(time_col).dt.replace_time_zone(None)
        ).drop_nulls(subset=[time_col])

        # Sort by time column (required for group_by_dynamic)
        chunk = chunk.sort(time_col)

        # Group rows by month
        for month, group in chunk.group_by_dynamic(time_col, every="1mo"):
            # take the first datetime in this group
            month_str = group[time_col][0].strftime("%Y-%m")
            monthly_groups[month_str].append(group)


    # Write one parquet per month into odd/even folders
    for month_str, groups in monthly_groups.items():
        combined = pl.concat(groups)
        month_num = int(month_str.split("-")[1])
        if month_num % 2 == 1:
            out_file = os.path.join(odd_dir, f"{prefix}_{month_str}.parquet")
        else:
            out_file = os.path.join(even_dir, f"{prefix}_{month_str}.parquet")
        combined.write_parquet(out_file)

markets_loader = DatasetLoader(markets_pattern, file_format="parquet")
trades_loader  = DatasetLoader(trades_pattern, file_format="parquet")

split_by_month(markets_loader, "created_time", "data/markets/markets_kalshi_odd", "data/markets/markets_kalshi_even", "markets")
split_by_month(trades_loader, "created_time", "data/trades/trades_kalshi_odd", "data/trades/trades_kalshi_even", "trades")

