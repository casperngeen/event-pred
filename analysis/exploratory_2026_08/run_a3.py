import sys, json
sys.path.insert(0, "scripts")
from fetch_kalshi_data import fetch_all_econ_trades
SC="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"
tickers = json.load(open(f"{SC}/newtickers.json"))
print(f"Fetching trades for {len(tickers)} tickers...", flush=True)
df = fetch_all_econ_trades(tickers, output_path=f"{SC}/new_trades_2026.parquet", rate_limit_delay=0.1)
