"""
pairwise_monotonicity_followup_checks.py

Follow-up analysis on top of pairwise_monotonicity_taker_side_check.py's
output (pairwise_monotonicity_taker_side_results.parquet). Covers, in
priority order:

  1. Well-sampled-only violation rates (re-trust check — your run showed
     76% of pairs flagged low-n, so the raw 23% rate is likely noisy)
  2. Spread-ratio bucket breakdown (are violations concentrated where the
     same-side cancellation assumption is weakest?)
  3. Magnitude of surviving violations vs. typical spread (size, not just
     rate — mirrors the report's original "~8 cent median" framing)
  4. Sensitivity of the same-side violation rate to the MIN_N_PER_SIDE cutoff
     (how much is the 23% figure an artifact of where you drew that line?)
  5. Time gap between the two legs' trades for violating pairs (does the
     violation ever exist "at the same moment," i.e. plausibly capturable?)
  6. (best-effort/optional) implied spread vs. real quoted spread, where a
     markets snapshot exists near the relevant trades

ASSUMPTIONS to verify against your real schema, same caveats as the
previous script:
  - results parquet columns match what pairwise_monotonicity_taker_side_check.py
    wrote out (leg_a, leg_b, naive_violation, same_side_yes_violation,
    same_side_yes_gap, same_side_yes_n_a/n_b, same_side_yes_low_n, same for
    "no", spread_a, spread_b, spread_ratio).
  - trades columns: ticker, taker_side, yes_price, and a timestamp column
    (TIMESTAMP_COL below — adjust if not "created_time").
  - markets columns (only needed for check 6): ticker, yes_bid, yes_ask.
"""

import polars as pl

RESULTS_PATH = "pairwise_monotonicity_taker_side_results.parquet"
TRADES_PATH = "data/trades/trades_kalshi_even/trades_2025-10.parquet"        # adjust to your real path
MARKETS_PATH = "data/markets/markets_kalshi_even/trades_2025-10.parquet"      # adjust to your real path (check 6 only)

TICKER_COL = "ticker"
SIDE_COL = "taker_side"
PRICE_COL = "yes_price"
TIMESTAMP_COL = "created_time"        # adjust if needed


def _fmt(x, nd=4):
    return "N/A" if x is None else f"{x:.{nd}f}"


def check_1_well_sampled(results: pl.DataFrame):
    well_sampled = results.filter(
        ~pl.col("same_side_yes_low_n") & ~pl.col("same_side_no_low_n")
    )
    print("=== Check 1: well-sampled-only rates ===")
    print(f"well-sampled pairs: {well_sampled.height} of {results.height}")
    if well_sampled.height:
        print(f"naive violation rate (well-sampled):            {_fmt(well_sampled['naive_violation'].mean())}")
        print(f"same-side 'yes' violation rate (well-sampled):  {_fmt(well_sampled['same_side_yes_violation'].mean())}")
        print(f"same-side 'no' violation rate (well-sampled):   {_fmt(well_sampled['same_side_no_violation'].mean())}")
    else:
        print("no well-sampled pairs at all — consider lowering MIN_N_PER_SIDE in the original script and rerunning it")
    print()
    return well_sampled


def check_2_spread_ratio_buckets(results: pl.DataFrame):
    bucketed = results.with_columns(
        pl.when(pl.col("spread_ratio") > 3).then(pl.lit("high (>3x)"))
        .when(pl.col("spread_ratio") > 1.5).then(pl.lit("mid (1.5-3x)"))
        .when(pl.col("spread_ratio").is_not_null()).then(pl.lit("low (<1.5x)"))
        .otherwise(pl.lit("unknown"))
        .alias("spread_ratio_bucket")
    )
    summary = bucketed.group_by("spread_ratio_bucket").agg(
        pl.col("same_side_yes_violation").mean().alias("yes_violation_rate"),
        pl.col("same_side_no_violation").mean().alias("no_violation_rate"),
        pl.len().alias("n_pairs"),
    )
    print("=== Check 2: violation rate by spread-ratio bucket ===")
    print(summary)
    print("(if violations concentrate in the 'high' bucket, treat that share of the 23% with more skepticism)")
    print()
    return bucketed


def check_3_violation_magnitude(results: pl.DataFrame):
    print("=== Check 3: magnitude of surviving violations vs. typical spread ===")
    for side in ("yes", "no"):
        viol = results.filter(pl.col(f"same_side_{side}_violation") == True)  # noqa: E712
        if viol.height == 0:
            print(f"same-side '{side}': no violations found")
            continue
        median_gap = viol[f"same_side_{side}_gap"].median()
        avg_spread_a = viol["spread_a"].mean()
        avg_spread_b = viol["spread_b"].mean()
        avg_spread = None
        if avg_spread_a is not None and avg_spread_b is not None:
            avg_spread = (avg_spread_a + avg_spread_b) / 2
        ratio = (median_gap / avg_spread) if (median_gap is not None and avg_spread) else None
        print(
            f"same-side '{side}': n={viol.height}, median_gap={_fmt(median_gap)}, "
            f"avg_leg_spread={_fmt(avg_spread)}, gap/spread ratio={_fmt(ratio, 2)}"
        )
    print("(ratio well above 1 suggests the violation is bigger than typical spread noise; near/below 1 is weak evidence)")
    print()


def check_4_min_n_sensitivity(results: pl.DataFrame, thresholds=(10, 20, 50, 100)):
    print("=== Check 4: sensitivity of same-side violation rate to MIN_N_PER_SIDE ===")
    for thresh in thresholds:
        sub_yes = results.filter(
            (pl.col("same_side_yes_n_a") >= thresh) & (pl.col("same_side_yes_n_b") >= thresh)
        )
        rate_yes = sub_yes["same_side_yes_violation"].mean() if sub_yes.height else None
        sub_no = results.filter(
            (pl.col("same_side_no_n_a") >= thresh) & (pl.col("same_side_no_n_b") >= thresh)
        )
        rate_no = sub_no["same_side_no_violation"].mean() if sub_no.height else None
        print(
            f"min_n={thresh}: yes-side pairs={sub_yes.height}, yes_rate={_fmt(rate_yes)}, "
            f"no-side pairs={sub_no.height}, no_rate={_fmt(rate_no)}"
        )
    print("(a rate that stays roughly flat as the cutoff rises is trustworthy; one that swings a lot is fragile)")
    print()


def check_5_time_gap(results: pl.DataFrame, trades: pl.DataFrame, side: str = "yes"):
    """
    For violating pairs (on the given side), compare the time window each
    leg's same-side trades actually occurred in. A large gap between the two
    legs' trading windows means the 'violation' was never simultaneously
    observable, so it says less about live tradability even if real.
    """
    print(f"=== Check 5: time gap between legs' trades, same_side_{side}_violation pairs ===")
    viol = results.filter(pl.col(f"same_side_{side}_violation") == True)  # noqa: E712
    rows = []
    for row in viol.iter_rows(named=True):
        leg_a, leg_b = row["leg_a"], row["leg_b"]
        ta = trades.filter((pl.col(TICKER_COL) == leg_a) & (pl.col(SIDE_COL) == side))
        tb = trades.filter((pl.col(TICKER_COL) == leg_b) & (pl.col(SIDE_COL) == side))
        if ta.height == 0 or tb.height == 0:
            continue
        a_mid = ta[TIMESTAMP_COL].cast(pl.Int64).mean()
        b_mid = tb[TIMESTAMP_COL].cast(pl.Int64).mean()
        rows.append({"leg_a": leg_a, "leg_b": leg_b, "time_gap": abs(a_mid - b_mid)})
    if not rows:
        print("no violating pairs with trades on this side found")
        return
    gaps = pl.DataFrame(rows)
    print(gaps.describe())
    print("(units match your TIMESTAMP_COL's native representation once cast to Int64 — convert as needed)")
    print()


def check_6_spread_validation(trades: pl.DataFrame, markets: pl.DataFrame, tickers: list[str]):
    """
    Best-effort: compare the taker_side-implied spread against the real
    quoted spread, for tickers where a markets snapshot exists. Only run
    this if you have a markets file with yes_bid/yes_ask populated for the
    relevant tickers — the earlier investigation found this snapshot is
    sparse (monthly), so expect a small/partial result, not full coverage.
    """
    print("=== Check 6 (best-effort): implied spread vs. quoted spread ===")
    quoted = markets.filter(pl.col(TICKER_COL).is_in(tickers)).select(
        TICKER_COL, (pl.col("yes_ask") - pl.col("yes_bid")).alias("quoted_spread")
    )
    implied_rows = []
    for t in tickers:
        sub = trades.filter(pl.col(TICKER_COL) == t)
        grouped = sub.group_by(SIDE_COL).agg(pl.col(PRICE_COL).mean().alias("avg_price"))
        prices = {r[SIDE_COL]: r["avg_price"] for r in grouped.iter_rows(named=True)}
        if "yes" in prices and "no" in prices:
            implied_rows.append({TICKER_COL: t, "implied_spread": abs(prices["no"] - prices["yes"])})
    implied = pl.DataFrame(implied_rows)
    comparison = implied.join(quoted, on=TICKER_COL, how="inner")
    print(f"tickers with both implied and quoted spread available: {comparison.height} of {len(tickers)}")
    if comparison.height:
        print(comparison)
        corr = comparison.select(pl.corr("implied_spread", "quoted_spread")).item()
        print(f"correlation(implied, quoted): {corr}")
    print()


def main():
    results = pl.read_parquet(RESULTS_PATH)

    check_1_well_sampled(results)
    check_2_spread_ratio_buckets(results)
    check_3_violation_magnitude(results)
    check_4_min_n_sensitivity(results)

    trades = pl.read_parquet(TRADES_PATH)
    check_5_time_gap(results, trades, side="yes")
    check_5_time_gap(results, trades, side="no")

    # Check 6 is optional / best-effort — uncomment if you have a usable markets snapshot
    # markets = pl.read_parquet(MARKETS_PATH)
    # sample_tickers = list(set(
    #     results.select("leg_a").to_series().to_list()
    #     + results.select("leg_b").to_series().to_list()
    # ))
    # check_6_spread_validation(trades, markets, sample_tickers)


if __name__ == "__main__":
    main()