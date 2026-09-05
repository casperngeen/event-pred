"""
mece_sum_to_one_pnl_backtest.py

PnL backtest for the full-basket sum-to-$1 deviations found by
mece_sum_to_one_check.py, mirroring the treatment
pairwise_monotonicity_pnl_backtest.py gave the ladder result: an IDEALIZED
zero-friction scenario, a FEES-ONLY middle scenario, and a REALISTIC
scenario that also charges an execution-cost haircut. Same three-bracket
convention, adapted for an N-leg basket instead of a fixed 2-leg pair.

ECONOMIC LOGIC
--------------
A full-basket snapshot is a day where every leg of a resolved, single-
winner MECE event traded, with implied sum S = sum of each leg's last
YES trade price that day (in dollars). Exactly one leg resolves YES; every
other leg resolves NO. Two positions, one for each sign of deviation:

  S > $1.00 (overpriced) -- SELL 1 YES contract of every leg.
    Collect S at trade time. At settlement you owe $1 (only the winning
    leg pays out to its buyer). Guaranteed profit = S - 1.00 = deviation.

  S < $1.00 (underpriced) -- BUY 1 YES contract of every leg.
    Pay S at trade time. At settlement you receive $1 (the winning leg
    pays $1, every other leg pays $0). Guaranteed profit = 1.00 - S = -deviation.

In both cases the guaranteed minimum profit per "1 unit of the N-leg
bundle" is abs(deviation) dollars -- structurally the same guarantee the
ladder script's gap represented, just realized across N legs simultaneously
resolving to exactly one winner instead of 2 legs whose relative order is
what's guaranteed.

CONTRACTS_PER_OPPORTUNITY here means contracts *per leg* -- executing one
"unit" of an N-leg bundle at size C means trading C contracts on EACH of
the N legs (that's what "sell every leg" / "buy every leg" means). This is
the key structural cost difference from the 2-leg ladder case: fees and
the execution-cost haircut below are paid once per leg, so an 8-leg basket
pays 4x the friction of a 2-leg pair for the same per-contract size, and a
40-leg basket (if one were ever found fully covered) would pay 20x. That
scaling, not the fee/haircut rate itself, is the main reason to expect
N-way MECE PnL to degrade faster under realistic assumptions than the
ladder result did.

WHERE THIS IS WEAKER THAN THE LADDER BACKTEST -- read before trusting the
realistic scenario:
  The ladder script derived its spread haircut from real trade-implied
  spreads per leg (spread_a/spread_b, computed upstream from actual
  same-side trade dispersion). mece_sum_to_one_check.py's leg-price output
  only carries each leg's single LAST trade price for that day -- there is
  no bid/ask or spread signal in this table at all. PER_LEG_HAIRCUT_CENTS
  below is therefore a flat, non-data-driven assumption (default: 1 cent,
  Kalshi's minimum tick), not a measured cost. Treat REALISTIC_PNL as
  indicative at best; run mece_sum_to_one_pnl_sensitivity.py to see how
  fast it degrades as that assumption is raised, rather than trusting the
  single point estimate at the default.

ASSUMPTIONS -- verify before trusting the output, then delete this checklist:
  [ ] RESULTS_PATH / LEG_PRICES_PATH columns match mece_sum_to_one_check.py's
      §7 output: event_ticker, date, legs_traded, sum_cents, deviation,
      abs_deviation, time_gap_hours, n_legs_total (results); event_ticker,
      date, ticker, close, trade_time (leg prices).
  [ ] Prices/sums are in CENTS (0-100 scale) in RESULTS_PATH, dollars
      already applied via /100.0 below -- matches every other script in
      this investigation.
  [ ] Kalshi standard taker fee formula (checked 2026-09-02): fee =
      ceil_to_centicent(0.07 * C * P * (1-P)), P in dollars, C = contract
      count, standard multiplier. Re-check
      https://kalshi.com/docs/kalshi-fee-schedule.pdf before relying on
      this for anything real.
  [ ] No settlement/exercise fee assumed -- same caveat as the ladder
      script.
  [ ] VIOLATION_THRESHOLD matches mece_sum_to_one_check.py's $0.05 bar for
      "meaningfully mispriced" -- only snapshots exceeding it are treated
      as tradeable opportunities here, same filtering logic as the
      ladder's same_side_*_violation flag gating build_opportunities().

Run with:  python -m stg_infra.examples.mece_sum_to_one_pnl_backtest
"""

import math
import os

import polars as pl

try:
    from pairwise_monotonicity_pnl_backtest import classify_ticker
except ImportError:
    from .pairwise_monotonicity_pnl_backtest import classify_ticker

RESULTS_PATH = "mece_sum_to_one_results.parquet"
LEG_PRICES_PATH = "mece_sum_to_one_leg_prices.parquet"

PRICE_SCALE = 100.0                # cents scale
CONTRACTS_PER_OPPORTUNITY = 100    # contracts PER LEG (see docstring) -- not total across the basket
TAKER_FEE_RATE = 0.07
VIOLATION_THRESHOLD = 0.05         # dollars -- matches mece_sum_to_one_check.py's bar
PER_LEG_HAIRCUT_CENTS = 1.0        # flat, non-data-driven assumption -- see docstring caveat


def taker_fee_dollars(price_cents: float, contracts: float) -> float:
    """Same formula/derivation as pairwise_monotonicity_pnl_backtest.taker_fee_dollars --
    duplicated (not imported) so this script has no import-order dependency on that
    module's own state, since it's only borrowing classify_ticker from it."""
    p = price_cents / PRICE_SCALE
    raw = TAKER_FEE_RATE * contracts * p * (1 - p)
    return math.ceil(raw * 10000) / 10000


def load_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    for p in (RESULTS_PATH, LEG_PRICES_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found -- run mece_sum_to_one_check.py first (it now writes both "
                f"{RESULTS_PATH} and {LEG_PRICES_PATH} as its final step)."
            )
    return pl.read_parquet(RESULTS_PATH), pl.read_parquet(LEG_PRICES_PATH)


def build_opportunities(results: pl.DataFrame) -> pl.DataFrame:
    """One row per full-basket (event_ticker, date) snapshot whose |deviation|
    exceeds VIOLATION_THRESHOLD -- the MECE analogue of build_opportunities()
    filtering to only flagged same_side_*_violation rows in the ladder script."""
    opp = (
        results.filter(pl.col("abs_deviation") > VIOLATION_THRESHOLD)
        .with_columns([
            pl.when(pl.col("deviation") > 0).then(pl.lit("sell_all_legs")).otherwise(pl.lit("buy_all_legs")).alias("side"),
            pl.col("event_ticker").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category"),
        ])
    )
    return opp


def attach_leg_fees(opportunities: pl.DataFrame, leg_prices: pl.DataFrame,
                     contracts: float = CONTRACTS_PER_OPPORTUNITY) -> pl.DataFrame:
    """Per-opportunity total fee = sum of taker_fee_dollars(leg_close, contracts)
    across every leg in that (event_ticker, date) basket -- the N-leg generalization
    of the ladder's fee_a + fee_b. Uses each leg's own exact last-trade price for
    that snapshot (leg_prices), which is sharper than the ladder script's
    monthly-average fallback since we already know precisely which trades made up
    the sum being tested."""
    if opportunities.height == 0:
        return opportunities.with_columns(
            pl.lit(0.0).alias("fees_usd"),
            pl.lit(0).alias("n_legs_priced"),
        )
    fee_detail = leg_prices.with_columns(
        pl.col("close").map_elements(lambda c: taker_fee_dollars(c, contracts), return_dtype=pl.Float64).alias("leg_fee")
    )
    per_snapshot_fees = fee_detail.group_by(["event_ticker", "date"]).agg([
        pl.col("leg_fee").sum().alias("fees_usd"),
        pl.col("ticker").n_unique().alias("n_legs_priced"),
    ])
    return opportunities.join(per_snapshot_fees, on=["event_ticker", "date"], how="left").with_columns([
        pl.col("fees_usd").fill_null(0.0),
        pl.col("n_legs_priced").fill_null(0),
    ])


def compute_pnl(opportunities: pl.DataFrame, contracts: float = CONTRACTS_PER_OPPORTUNITY,
                 per_leg_haircut_cents: float = PER_LEG_HAIRCUT_CENTS) -> pl.DataFrame:
    if opportunities.height == 0:
        return opportunities
    idealized = opportunities["abs_deviation"] * contracts
    haircut_dollars = (per_leg_haircut_cents / PRICE_SCALE) * contracts * opportunities["n_legs_total"]
    fees_only_pnl = idealized - opportunities["fees_usd"]
    realistic_pnl = fees_only_pnl - haircut_dollars
    return opportunities.with_columns([
        idealized.alias("idealized_pnl_usd"),
        haircut_dollars.alias("haircut_usd"),
        fees_only_pnl.alias("fees_only_pnl_usd"),
        realistic_pnl.alias("realistic_pnl_usd"),
    ])


def summarize(pnl: pl.DataFrame, label: str):
    if pnl.height == 0:
        print(f"=== {label}: no opportunities (check VIOLATION_THRESHOLD, or that full_all was non-empty) ===\n")
        return
    print(f"=== {label} (n={pnl.height} full-basket opportunities, "
          f"{CONTRACTS_PER_OPPORTUNITY} contracts per leg, VIOLATION_THRESHOLD=${VIOLATION_THRESHOLD:.2f}) ===")
    print(f"idealized total PnL (zero friction):              ${pnl['idealized_pnl_usd'].sum():,.2f}")
    print(f"fees-only total PnL (taker fees, no haircut):      ${pnl['fees_only_pnl_usd'].sum():,.2f}")
    print(f"realistic total PnL (fees + ${PER_LEG_HAIRCUT_CENTS:.1f}c/leg haircut): "
          f"${pnl['realistic_pnl_usd'].sum():,.2f}")
    print(f"realistic win rate (opportunities with PnL > 0):   {(pnl['realistic_pnl_usd'] > 0).mean():.1%}")
    print(f"realistic median PnL per opportunity:              ${pnl['realistic_pnl_usd'].median():,.2f}")
    print(f"total fees paid:                                   ${pnl['fees_usd'].sum():,.2f}")
    print(f"total haircut charged:                             ${pnl['haircut_usd'].sum():,.2f}")
    print(f"avg legs per opportunity:                          {pnl['n_legs_total'].mean():.1f}")
    print()


def summarize_by_category(pnl: pl.DataFrame):
    if pnl.height == 0:
        return
    summary = (
        pnl.group_by("category")
        .agg(
            pl.len().alias("n_opportunities"),
            pl.col("n_legs_total").mean().alias("avg_legs"),
            pl.col("idealized_pnl_usd").sum().alias("idealized_total_usd"),
            pl.col("realistic_pnl_usd").sum().alias("realistic_total_usd"),
            (pl.col("realistic_pnl_usd") > 0).mean().alias("realistic_win_rate"),
            pl.col("realistic_pnl_usd").median().alias("realistic_median_usd"),
        )
        .sort("realistic_total_usd", descending=True)
    )
    print("=== PnL by category ===")
    print(summary)
    print()


def summarize_by_side(pnl: pl.DataFrame):
    if pnl.height == 0:
        return
    summary = (
        pnl.group_by("side")
        .agg(
            pl.len().alias("n_opportunities"),
            pl.col("idealized_pnl_usd").sum().alias("idealized_total_usd"),
            pl.col("realistic_pnl_usd").sum().alias("realistic_total_usd"),
            (pl.col("realistic_pnl_usd") > 0).mean().alias("realistic_win_rate"),
        )
        .sort("n_opportunities", descending=True)
    )
    print("=== PnL by side (overpriced->sell_all_legs vs. underpriced->buy_all_legs) ===")
    print(summary)
    print()


def main():
    results, leg_prices = load_inputs()
    print(f"Loaded {results.height} full-basket snapshot(s) from {RESULTS_PATH} "
          f"and {leg_prices.height} leg-price row(s) from {LEG_PRICES_PATH}.\n")

    opportunities = build_opportunities(results)
    print(f"{opportunities.height} of {results.height} full-basket snapshots exceed "
          f"VIOLATION_THRESHOLD=${VIOLATION_THRESHOLD:.2f} and are treated as tradeable "
          f"opportunities below.\n")

    opportunities = attach_leg_fees(opportunities, leg_prices)
    pnl = compute_pnl(opportunities)

    summarize(pnl, "All full-basket sum-to-$1 opportunities")
    summarize_by_category(pnl)
    summarize_by_side(pnl)

    out_path = "mece_sum_to_one_pnl_results.parquet"
    pnl.write_parquet(out_path)
    print(f"Wrote per-opportunity PnL detail to {out_path}")


if __name__ == "__main__":
    main()