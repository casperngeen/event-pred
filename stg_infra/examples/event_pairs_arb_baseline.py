from __future__ import annotations

from stg.pairs.config import EventPairsConfig
from stg.pairs.pipeline import run_event_pairs_arb


TRADES_GLOB = "data/trades/trades_kalshi_even/trades_2025-10.parquet"
MARKETS_GLOB = "data/markets/markets_kalshi_even/markets_2025-10.parquet"


cfg = EventPairsConfig(
    min_event_days=10,
    top_events=300,
    lookback_sel=20,
    beta_lookback=10,
    z_window=10,
    corr_min=0.20,
    corr_max=0.95,
    half_life_min=1.0,
    half_life_max=10.0,
    top_pairs=50,
    entry_z=2.0,
    exit_z=0.5,
    max_hold=20,
    cost_per_pair_turn=0.0,
)

out = run_event_pairs_arb(trades_glob=TRADES_GLOB, markets_glob=MARKETS_GLOB, cfg=cfg)

# Write outputs
out["equity"].write_parquet("data/markets/event_pairs_equity.parquet")
out["pair_mapping"].write_parquet("data/markets/event_pairs_mapping.parquet")
out["diagnostics"].write_parquet("data/markets/event_pairs_diagnostics.parquet")

print("Wrote:")
print(" - data/markets/event_pairs_equity.parquet")
print(" - data/markets/event_pairs_mapping.parquet")
print(" - data/markets/event_pairs_diagnostics.parquet")

# Print event pairs
print("\nSelected event pairs (top):")
for i, (a, b) in enumerate(out["event_pairs"][:25], start=1):
    print(f"{i:>2}. ({a}, {b})")

# Print diagnostics for the top event pairs
print("\nDiagnostics (top pairs):")
diag_top = (
    out["diagnostics"]
    .sort("corr", descending=True)
    .select(["event_A", "event_B", "corr", "half_life", "overlap_n"])
    .head(25)
)
print(diag_top)

# Small previews
print("\nPair mapping (head):")
print(out["pair_mapping"].head(10))
print("\nEquity (tail):")
print(out["equity"].tail(5))
