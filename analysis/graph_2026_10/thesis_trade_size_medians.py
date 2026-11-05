"""Median observed trade quantity in the six development spread/total series.

Uses the normal sports loader's deduplication and pre-holdout date filter.
Run from the repository root without STG_SPORTS_UNSEAL set.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.splits import sports_holdout_unsealed
from stg.sports.loaders import scan_trades


def main():
    assert not sports_holdout_unsealed(), "Sizing rationale uses development data only"
    rows = []
    for league in ("MLB", "NBA", "NFL"):
        for kind in ("spread", "total"):
            result = scan_trades(f"KX{league}{kind.upper()}").select(
                pl.len().alias("trades"), pl.col("count").median().alias("median_quantity")
            ).collect().row(0, named=True)
            rows.append(dict(league=league, kind=kind, **result))
            print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(Path(__file__).parent / "out/thesis_trade_size_medians.csv")


if __name__ == "__main__":
    main()
