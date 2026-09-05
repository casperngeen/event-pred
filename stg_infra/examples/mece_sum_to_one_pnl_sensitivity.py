"""
mece_sum_to_one_pnl_sensitivity.py

Companion to mece_sum_to_one_pnl_backtest.py. That script's realistic
scenario subtracts PER_LEG_HAIRCUT_CENTS (a flat, non-data-driven execution
cost per leg) since -- unlike the ladder script -- mece_sum_to_one_check.py's
leg-price output carries no bid/ask spread signal to derive a haircut from,
only each leg's last trade price. That default (1 cent/leg, Kalshi's
minimum tick) is a best case, not a measured one.

This sweeps PER_LEG_HAIRCUT_CENTS across a range and reports realistic PnL
by category at each step -- same purpose as
pairwise_monotonicity_pnl_sensitivity.py's SPREAD_HAIRCUT_FRACTION sweep:
report a defensible range instead of a single point estimate, and see
where (if anywhere) each category's PnL crosses zero. Because the haircut
here is charged PER LEG rather than per pair, this also surfaces how much
faster large baskets erode than small ones as the assumption rises --
sweep results include avg_legs per category so that pattern is visible
directly instead of needing a second script.

Run this from the same directory as mece_sum_to_one_pnl_backtest.py (or put
it on your PYTHONPATH) -- it imports that module directly rather than
duplicating its logic, so any fixes made there (fee formula, schema, paths)
are picked up automatically here too.

USAGE: python mece_sum_to_one_pnl_sensitivity.py
"""

import polars as pl

try:
    import mece_sum_to_one_pnl_backtest as base
except ImportError:
    from . import mece_sum_to_one_pnl_backtest as base

HAIRCUT_CENTS_LEVELS = [0.0, 1.0, 2.0, 5.0, 10.0]


def run_one(results: pl.DataFrame, leg_prices: pl.DataFrame, haircut_cents: float) -> pl.DataFrame:
    opportunities = base.build_opportunities(results)
    opportunities = base.attach_leg_fees(opportunities, leg_prices)
    return base.compute_pnl(opportunities, per_leg_haircut_cents=haircut_cents)


def main():
    results, leg_prices = base.load_inputs()
    print(f"Loaded {results.height} full-basket snapshot(s) and {leg_prices.height} leg-price row(s).\n")

    rows = []
    for cents in HAIRCUT_CENTS_LEVELS:
        pnl = run_one(results, leg_prices, cents)
        if pnl.height == 0:
            continue
        by_cat = pnl.group_by("category").agg(
            pl.len().alias("n_opportunities"),
            pl.col("n_legs_total").mean().alias("avg_legs"),
            pl.col("realistic_pnl_usd").sum().alias("realistic_total_usd"),
            (pl.col("realistic_pnl_usd") > 0).mean().alias("realistic_win_rate"),
        ).with_columns(pl.lit(cents).alias("haircut_cents_per_leg"))
        rows.append(by_cat)

        overall_total = pnl["realistic_pnl_usd"].sum()
        overall_win = (pnl["realistic_pnl_usd"] > 0).mean()
        print(f"--- haircut = {cents:.1f}c per leg ---")
        print(f"  overall: total=${overall_total:,.2f}  win_rate={overall_win:.1%}")
        for r in by_cat.sort("category").iter_rows(named=True):
            sign = "+" if r["realistic_total_usd"] >= 0 else ""
            print(f"  {r['category']:<16} n={r['n_opportunities']:<5} avg_legs={r['avg_legs']:<6.1f} "
                  f"total={sign}${r['realistic_total_usd']:,.2f}  win_rate={r['realistic_win_rate']:.1%}")
        print()

    if rows:
        sweep = pl.concat(rows).select(
            ["haircut_cents_per_leg", "category", "n_opportunities", "avg_legs",
             "realistic_total_usd", "realistic_win_rate"]
        )
        out_path = "mece_sum_to_one_pnl_sensitivity_results.parquet"
        sweep.write_parquet(out_path)
        print(f"Wrote full sweep detail to {out_path} -- use it to plot realistic_total_usd vs. "
              f"haircut_cents_per_leg per category, and to check whether larger-avg_legs categories "
              f"erode faster than smaller ones as the per-leg assumption rises.")


if __name__ == "__main__":
    main()