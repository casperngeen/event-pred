"""
pairwise_monotonicity_pnl_sensitivity.py

Companion to pairwise_monotonicity_pnl_backtest.py. That script's realistic
scenario subtracts a spread haircut (SPREAD_HAIRCUT_FRACTION * (spread_a +
spread_b)) as a stand-in for real execution cost, since good order-book
fill data wasn't available. SPREAD_HAIRCUT_FRACTION=1.0 (full combined
spread on both legs) is a deliberately pessimistic choice -- this script
sweeps that fraction from 0 (fees only) to 1 (full haircut) and reports
realistic PnL by category at each step, so you can report a defensible
range instead of a single point estimate, and see whether/where each
category's PnL crosses zero.

Run this from the same directory as pairwise_monotonicity_pnl_backtest.py
(or put it on your PYTHONPATH) -- it imports that module directly rather
than duplicating its logic, so any fixes you make there (fee formula,
schema, paths) are picked up automatically here too.

USAGE: python pairwise_monotonicity_pnl_sensitivity.py
"""

import polars as pl

try:
    # Works when run as a plain script: `python pairwise_monotonicity_pnl_sensitivity.py`
    # with both files in the same directory.
    import pairwise_monotonicity_pnl_backtest as base
except ImportError:
    # Works when run as part of a package, e.g.
    # `python -m stg_infra.examples.pairwise_monotonicity_pnl_sensitivity`,
    # where the sibling module must be imported relative to the package.
    from . import pairwise_monotonicity_pnl_backtest as base

HAIRCUT_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_one(results: pl.DataFrame, prices: pl.DataFrame | None, haircut_fraction: float) -> pl.DataFrame:
    base.SPREAD_HAIRCUT_FRACTION = haircut_fraction  # module-level global read inside build_opportunities
    opportunities = base.build_opportunities(results)
    opportunities = base.attach_leg_prices(opportunities, prices)
    return base.compute_pnl(opportunities)


def main():
    results = pl.read_parquet(base.RESULTS_PATH)
    base.print_category_coverage(results)
    prices = base.real_leg_prices(base.TRADES_PATH)
    if prices is None:
        print(f"NOTE: real per-leg trade prices not available -- using the conservative "
              f"50-cent fee-sizing fallback for every haircut level below.\n")

    rows = []
    for frac in HAIRCUT_FRACTIONS:
        pnl = run_one(results, prices, frac)
        if pnl.height == 0:
            continue
        by_cat = pnl.group_by("category").agg(
            pl.len().alias("n_opportunities"),
            pl.col("realistic_pnl_usd").sum().alias("realistic_total_usd"),
            (pl.col("realistic_pnl_usd") > 0).mean().alias("realistic_win_rate"),
        ).with_columns(pl.lit(frac).alias("haircut_fraction"))
        rows.append(by_cat)

        overall_total = pnl["realistic_pnl_usd"].sum()
        overall_win = (pnl["realistic_pnl_usd"] > 0).mean()
        print(f"--- haircut_fraction={frac:.2f} ---")
        print(f"  overall: total=${overall_total:,.2f}  win_rate={overall_win:.1%}")
        for r in by_cat.sort("category").iter_rows(named=True):
            sign = "+" if r["realistic_total_usd"] >= 0 else ""
            print(f"  {r['category']:<16} n={r['n_opportunities']:<5} "
                  f"total={sign}${r['realistic_total_usd']:,.2f}  win_rate={r['realistic_win_rate']:.1%}")
        print()

    if rows:
        sweep = pl.concat(rows).select(
            ["haircut_fraction", "category", "n_opportunities", "realistic_total_usd", "realistic_win_rate"]
        )
        out_path = "pairwise_monotonicity_pnl_sensitivity_results.parquet"
        sweep.write_parquet(out_path)
        print(f"Wrote full sweep detail to {out_path} -- use it to plot realistic_total_usd "
              f"vs. haircut_fraction per category and find each category's zero-crossing point.")


if __name__ == "__main__":
    main()