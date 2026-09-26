"""
mece_spread_artefact_check.py

Is the measured MECE deviation a real dislocation, or an artefact of
pricing a partition from non-simultaneous LAST TRADES?

THE CONCERN
-----------
A last trade is struck at the bid or the ask, never the mid. For a 6-leg
basket with a per-leg spread around 5.6c, if each leg's last print lands
on either side at random then the sum of last prices has a standard
deviation of roughly sqrt(6) * 2.8c ~= 6.9c around $1 with NO mispricing
at all, whose median |deviation| is about 0.674 * 6.9 ~= 4.6c.

The measured median |dev| on this project's population is 4c.

So the bulk of the deviation population is quantitatively consistent with
bid/ask mixing noise. That does not by itself invalidate anything -- the
cost gate fires at 19c to 37c, far into the tail, and the tail is fatter
than the noise model predicts (the gate fires on ~9% of sized events,
where pure mixing predicts ~0.6% above 19c). But "consistent with noise"
is not a thing to leave untested in a thesis.

THE TEST
--------
Kalshi's trades carry `taker_side`. A taker buying YES lifts the ask, so
its `yes_price` is an ASK print. A taker buying NO sells YES to the book,
so that trade's `yes_price` is a BID print. This is the same convention
build_mece_leg_spreads.py already uses to estimate per-leg spreads.

That lets each leg be priced three ways on the same day:

    ask_i  = yes_price from taker_side == "yes"
    bid_i  = yes_price from taker_side == "no"
    mid_i  = (ask_i + bid_i) / 2

and each basket four ways:

    sum_last = sum of the mixed last prints   <- what the project measures
    sum_ask  = sum of ask prints              <- biased UP by sum(half-spread)
    sum_bid  = sum of bid prints              <- biased DOWN by the same
    sum_mid  = sum of mids                    <- the unbiased estimate

DECISION RULE
-------------
  * If |sum_mid - 1| collapses toward zero while |sum_last - 1| does not,
    the deviation is mostly a spread artefact.
  * If |sum_mid - 1| stays large, the dislocation is real and the
    last-trade measure was merely noisy, not wrong.
  * If deviations computed from ask prints skew positive and from bid
    prints skew negative by about the same amount, that is the artefact
    signature: the "deviation" flips with execution side.
  * Sign asymmetry is a second, cheaper tell. A genuine mispricing
    population should be roughly balanced between sum > $1 and sum < $1.
    A strong positive skew is the classic betting-market OVERROUND: every
    leg printing at the ask.

WHAT SURVIVES EITHER WAY
------------------------
The PnL framework never assumed the deviations were true arbitrage. It
requires |dev| to exceed total transaction cost, and at
--spread-multiplier 1.0 it charges the full spread on every leg, which
absorbs the artefact. This script decides how to DESCRIBE the deviation,
and whether the gated population survives being repriced at mid.

Run from the repo root, after mece_sum_to_one_check.py has written its
leg-price file:

    python -m stg_infra.examples.mece_spread_artefact_check --self-test
    python -m stg_infra.examples.mece_spread_artefact_check
    python -m stg_infra.examples.mece_spread_artefact_check --gate-threshold 19
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl

pl.Config.set_tbl_rows(-1)

LEG_PRICES_PATH = "mece_sum_to_one_leg_prices.parquet"
TICKER_COL, SIDE_COL, PRICE_COL, TIME_COL = "ticker", "taker_side", "yes_price", "created_time"
PRICE_SCALE = 100.0

# taker_side value that means the taker BOUGHT YES, i.e. lifted the ask.
ASK_SIDE, BID_SIDE = "yes", "no"


def trades_path(month: str) -> str:
    parity = "even" if int(month.split("-")[1]) % 2 == 0 else "odd"
    return f"data/trades/trades_kalshi_{parity}/trades_{month}.parquet"


def months_of(dates) -> list[str]:
    return sorted({f"{d.year:04d}-{d.month:02d}" for d in dates})


def load_side_prices(tickers, months, min_side_trades: int) -> pl.DataFrame:
    """Per (ticker, date): last and mean yes_price on each taker side.

    Both LAST and MEAN are computed. Last is directly comparable to the
    leg-price file's `close` (also a last). Mean is the estimator
    build_mece_leg_spreads.py uses and is less noisy. If the two disagree
    about the conclusion, that disagreement is itself the finding, so
    neither is silently preferred.
    """
    parts = []
    for m in months:
        p = trades_path(m)
        if not glob.glob(p):
            print(f"  {m}: no trades file")
            continue
        t = (
            pl.scan_parquet(p)
            .select([TICKER_COL, TIME_COL, PRICE_COL, SIDE_COL])
            .filter(pl.col(TICKER_COL).is_in(tickers))
            .collect()
        )
        if t.is_empty():
            continue
        t = t.with_columns(pl.col(TIME_COL).dt.date().alias("date"))
        g = (
            t.sort(TIME_COL)
            .group_by([TICKER_COL, "date", SIDE_COL])
            .agg(
                pl.col(PRICE_COL).last().alias("last"),
                pl.col(PRICE_COL).mean().alias("mean"),
                pl.col(TIME_COL).last().alias("last_time"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= min_side_trades)
        )
        parts.append(g)
        print(f"  {m}: {t.height:,} candidate-leg trades")
    if not parts:
        return pl.DataFrame(
            schema={TICKER_COL: pl.Utf8, "date": pl.Date, SIDE_COL: pl.Utf8,
                    "last": pl.Float64, "mean": pl.Float64, "n": pl.UInt32}
        )
    return pl.concat(parts)


def build_basket_table(legs: pl.DataFrame, sides: pl.DataFrame,
                       max_side_gap_min: float = 0.0,
                       max_basket_span_min: float = 0.0) -> pl.DataFrame:
    """One row per (event_ticker, date) with all four basket sums.

    Only baskets where EVERY leg has BOTH sides present survive, because a
    mid cannot be formed otherwise. The attrition is reported, not hidden:
    requiring both sides selects toward two-sided, better-traded legs, so
    the surviving population is more liquid than the full one.
    """
    has_time = "last_time" in sides.columns
    acols = [TICKER_COL, "date", pl.col("last").alias("ask_last"),
             pl.col("mean").alias("ask_mean")]
    bcols = [TICKER_COL, "date", pl.col("last").alias("bid_last"),
             pl.col("mean").alias("bid_mean")]
    if has_time:
        acols.append(pl.col("last_time").alias("ask_time"))
        bcols.append(pl.col("last_time").alias("bid_time"))

    ask = sides.filter(pl.col(SIDE_COL) == ASK_SIDE).select(acols)
    bid = sides.filter(pl.col(SIDE_COL) == BID_SIDE).select(bcols)

    j = (
        legs.join(ask, on=[TICKER_COL, "date"], how="left")
            .join(bid, on=[TICKER_COL, "date"], how="left")
            .with_columns(
                ((pl.col("ask_last") + pl.col("bid_last")) / 2.0).alias("mid_last"),
                ((pl.col("ask_mean") + pl.col("bid_mean")) / 2.0).alias("mid_mean"),
                (pl.col("ask_mean") - pl.col("bid_mean")).alias("leg_spread"),
            )
    )

    # A mid built from an ask print at 09:00 and a bid print at 16:00 is not a
    # mid, it is two prices from different markets. This is the same drift
    # contamination that made the POOLED per-leg spread read 5.61c against the
    # windowed 4.68c, and the same reason only 75.6% of tickers show a
    # positive yes-minus-no difference rather than nearly all of them. When a
    # gap limit is set, a leg whose two prints are further apart than that
    # contributes no mid, and its basket drops out of the mid population.
    if max_side_gap_min > 0 and has_time:
        gap = (pl.col("ask_time") - pl.col("bid_time")).dt.total_seconds().abs() / 60.0
        j = j.with_columns(gap.alias("side_gap_min")).with_columns(
            pl.when(pl.col("side_gap_min") <= max_side_gap_min)
              .then(pl.col("mid_last")).otherwise(None).alias("mid_last"),
            pl.when(pl.col("side_gap_min") <= max_side_gap_min)
              .then(pl.col("mid_mean")).otherwise(None).alias("mid_mean"),
        )
    elif has_time:
        j = j.with_columns(
            ((pl.col("ask_time") - pl.col("bid_time")).dt.total_seconds().abs() / 60.0)
            .alias("side_gap_min"))
    else:
        j = j.with_columns(pl.lit(None, dtype=pl.Float64).alias("side_gap_min"))

    per_basket = (
        j.group_by(["event_ticker", "date"])
        .agg(
            pl.len().alias("n_legs"),
            pl.col("close").sum().alias("sum_last"),
            pl.col("ask_last").sum().alias("sum_ask_last"),
            pl.col("bid_last").sum().alias("sum_bid_last"),
            pl.col("mid_last").sum().alias("sum_mid_last"),
            pl.col("ask_mean").sum().alias("sum_ask_mean"),
            pl.col("bid_mean").sum().alias("sum_bid_mean"),
            pl.col("mid_mean").sum().alias("sum_mid_mean"),
            pl.col("leg_spread").sum().alias("basket_spread"),
            pl.col("ask_last").is_null().any().alias("miss_ask"),
            pl.col("bid_last").is_null().any().alias("miss_bid"),
            pl.col("mid_mean").is_null().any().alias("miss_mid"),
            pl.col("side_gap_min").max().alias("max_side_gap_min"),
            (pl.col("ask_time").max() - pl.col("ask_time").min())
                .dt.total_seconds().alias("_ask_span_s")
                if has_time else pl.lit(None, dtype=pl.Float64).alias("_ask_span_s"),
            (pl.col("bid_time").max() - pl.col("bid_time").min())
                .dt.total_seconds().alias("_bid_span_s")
                if has_time else pl.lit(None, dtype=pl.Float64).alias("_bid_span_s"),
        )
    )
    per_basket = per_basket.with_columns(
        (pl.max_horizontal(pl.col("_ask_span_s"), pl.col("_bid_span_s")) / 60.0)
        .alias("basket_span_min")
    ).drop(["_ask_span_s", "_bid_span_s"])

    # CROSS-LEG simultaneity. The within-leg limit above only ensures each
    # leg's own ask and bid prints are close together. A basket sum still
    # requires the LEGS to be contemporaneous with each other: six legs whose
    # prints are spread over four hours do not sum to anything meaningful,
    # because the underlying moved between them. The first version of this
    # script constrained only the within-leg gap, which is why tightening it
    # did not clean the mid estimate.
    if max_basket_span_min > 0:
        per_basket = per_basket.with_columns(
            pl.when(pl.col("basket_span_min") <= max_basket_span_min)
              .then(pl.col("miss_mid")).otherwise(True).alias("miss_mid")
        )
    return per_basket


def _dev(col: str) -> pl.Expr:
    return (pl.col(col) - PRICE_SCALE)


def describe(df: pl.DataFrame, label: str, col: str) -> dict:
    d = df.select(_dev(col).alias("d")).drop_nulls()
    if d.is_empty():
        return {"basis": label, "n": 0}
    s = d.get_column("d")
    return {
        "basis": label,
        "n": s.len(),
        "signed_mean": round(float(s.mean()), 2),
        "signed_median": round(float(s.median()), 2),
        "median_abs": round(float(s.abs().median()), 2),
        "pct_positive": round(float((s > 0).mean()) * 100, 1),
        "pct_gt_5c": round(float((s.abs() > 5).mean()) * 100, 1),
    }


def report(per_basket: pl.DataFrame, gate_threshold: float) -> None:
    n_all = per_basket.height
    both = per_basket.filter(~pl.col("miss_ask") & ~pl.col("miss_bid")
                             & ~pl.col("miss_mid"))

    print("\n" + "=" * 92)
    print("POPULATION")
    print("=" * 92)
    print(f"  full-basket (event, day) snapshots      : {n_all:,}")
    print(f"  with BOTH taker sides on EVERY leg      : {both.height:,} "
          f"({both.height / max(n_all, 1):.1%})")
    print(f"  missing an ask print on some leg        : "
          f"{per_basket.filter(pl.col('miss_ask')).height:,}")
    print(f"  missing a bid print on some leg         : "
          f"{per_basket.filter(pl.col('miss_bid')).height:,}")
    n_gap = per_basket.filter(pl.col("miss_mid") & ~pl.col("miss_ask")
                              & ~pl.col("miss_bid")).height
    if n_gap:
        print(f"  dropped: a leg's two prints too far apart: {n_gap:,}")
    g = per_basket.get_column("max_side_gap_min").drop_nulls()
    if g.len():
        print(f"  median worst-leg ask/bid print gap      : "
              f"{float(g.median()):.0f} min  (within a leg)")
    bsp = per_basket.get_column("basket_span_min").drop_nulls()
    if bsp.len():
        print(f"  median CROSS-LEG print span             : "
              f"{float(bsp.median()):.0f} min  (across the basket)")
    print("  Requiring both sides selects toward two-sided, better-traded legs,")
    print("  so the surviving population is MORE liquid than the full one. Read")
    print("  every number below as conditional on that.")

    print("\n" + "=" * 92)
    print("TEST 1 -- SIGN ASYMMETRY (last-trade basis, all snapshots)")
    print("=" * 92)
    r = describe(per_basket, "last trade", "sum_last")
    if r["n"]:
        print(f"  n={r['n']:,}   sum > $1: {r['pct_positive']}%   "
              f"sum < $1: {round(100 - r['pct_positive'], 1)}%")
        print(f"  signed mean {r['signed_mean']:+.2f}c   signed median "
              f"{r['signed_median']:+.2f}c   median |dev| {r['median_abs']:.2f}c")
        print("\n  A genuine mispricing population is roughly BALANCED between")
        print("  sum > $1 and sum < $1. A strong positive skew is the classic")
        print("  betting-market OVERROUND: every leg printing at the ask.")
        if r["pct_positive"] >= 65:
            print(f"  -> {r['pct_positive']}% positive is a MARKED skew. Treat the")
            print("     last-trade deviation as contaminated by the ask side.")
        elif r["pct_positive"] <= 35:
            print(f"  -> {r['pct_positive']}% positive is a marked NEGATIVE skew,")
            print("     i.e. bid-side contamination. Same concern, other direction.")
        else:
            print(f"  -> {r['pct_positive']}% positive is roughly balanced. No")
            print("     overround signature in the sign distribution.")

    if both.is_empty():
        print("\nNo basket has both taker sides on every leg; Test 2 cannot run.")
        return

    print("\n" + "=" * 92)
    print("TEST 2 -- THE SAME BASKETS PRICED FOUR WAYS")
    print("=" * 92)
    rows = [
        describe(both, "last trade (mixed)", "sum_last"),
        describe(both, "ask side  (taker=yes)", "sum_ask_last"),
        describe(both, "bid side  (taker=no)", "sum_bid_last"),
        describe(both, "MID (last-based)", "sum_mid_last"),
        describe(both, "MID (mean-based, diag)", "sum_mid_mean"),
    ]
    print(f"  {'basis':<24} {'n':>7} {'signed mean':>12} {'median |dev|':>13} "
          f"{'% sum>$1':>9} {'% |dev|>5c':>11}")
    for r in rows:
        if not r["n"]:
            continue
        print(f"  {r['basis']:<24} {r['n']:>7,} {r['signed_mean']:>+12.2f} "
              f"{r['median_abs']:>13.2f} {r['pct_positive']:>9.1f} "
              f"{r['pct_gt_5c']:>11.1f}")

    bs = both.get_column("basket_spread").drop_nulls()
    if bs.len():
        print(f"\n  median summed per-leg spread across the basket: "
              f"{float(bs.median()):.2f}c")
        print("  The ask and bid rows should straddle $1 by about half this each.")
        print("  If they do, and the MID row collapses toward zero, the deviation")
        print("  measured from mixed last prints is mostly spread, not mispricing.")

    # LAST-based, not mean-based: --max-basket-span constrains the times of the
    # last prints, so it has no purchase on a statistic averaged over the whole
    # day. The mean-based row stays in the table as a diagnostic only.
    mid_col = "sum_mid_last"
    last_med = float(both.select(_dev("sum_last").abs().median()).item() or 0)
    mid_med = float(both.select(_dev(mid_col).abs().median()).item() or 0)
    if last_med > 0:
        print(f"\n  VERDICT: median |dev| {last_med:.2f}c -> {mid_med:.2f}c at mid.")
        if mid_med > last_med * 1.2:
            print("  -> The mid estimate is LARGER than the last-trade one, which no")
            print("     unbiased mid can be. The mid is contaminated, almost always by")
            print("     cross-leg non-simultaneity: summing per-leg daily statistics")
            print("     taken at different times of day does not produce a basket.")
            print("     Re-run with --max-basket-span to force the legs to be")
            print("     contemporaneous before reading anything into the mid rows.")
            print("     Until then, read the ASK and BID rows, not the MID rows.")
        else:
            shrink = 1 - mid_med / last_med
            print(f"     ({shrink:.0%} of the deviation was spread.)")
            if shrink >= 0.6:
                print("  -> Most of the headline deviation is a spread artefact. The")
                print("     thesis must describe it as 'deviation in traded prices',")
                print("     never as 'arbitrage', and quote the mid-based figure.")
            elif shrink <= 0.25:
                print("  -> The dislocation largely survives repricing at mid. The")
                print("     last-trade measure was noisy, not wrong.")
            else:
                print("  -> Partial. Report both bases side by side.")

    # The ask and bid rows need no mid and no cross-leg assumption beyond the
    # one the project already makes, so they carry the cleanest signal here.
    a = both.select(_dev("sum_ask_last")).drop_nulls().to_series()
    b = both.select(_dev("sum_bid_last")).drop_nulls().to_series()
    if a.len() and b.len():
        am, bm = float(a.mean()), float(b.mean())
        print(f"\n  EXECUTABLE READING (no mid required):")
        print(f"    buying every leg at the ask  costs  ${1 + am / 100:.4f} per basket")
        print(f"    selling every leg at the bid yields ${1 + bm / 100:.4f} per basket")
        print(f"    a basket settles at exactly $1.00.")
        if bm > 0.5:
            print(f"    -> Selling at the bid nets {bm:+.2f}c gross before fees.")
        elif bm > -0.5:
            print(f"    -> Selling at the bid nets {bm:+.2f}c gross: ZERO edge before")
            print(f"       fees. The apparent deviation in last prints is the spread,")
            print(f"       and it is not capturable at executable prices.")
        else:
            print(f"    -> Selling at the bid LOSES {-bm:.2f}c gross before fees.")
        if am < -0.5:
            print(f"    -> Buying at the ask nets {-am:+.2f}c gross before fees.")

    print("\n" + "=" * 92)
    print(f"TEST 3 -- EXECUTABLE EDGE ON THE GATED TAIL (|dev_last| > {gate_threshold:g}c)")
    print("=" * 92)
    print("  The cost gate does not trade the average basket, it trades the tail.")
    print("  So ask the tail the executable question directly, with no mid and no")
    print("  cross-leg averaging: a basket priced ABOVE $1 is SOLD, which means")
    print("  hitting the bid on every leg; one priced BELOW $1 is BOUGHT at the ask.")
    print("  Either way it settles at exactly $1.00.\n")

    gated = both.filter(_dev("sum_last").abs() > gate_threshold).with_columns(
        pl.when(_dev("sum_last") > 0)
          .then(_dev("sum_bid_last"))                 # sell: receive bids, pay $1
          .otherwise(-_dev("sum_ask_last"))           # buy: pay asks, receive $1
          .alias("gross_c")
    ).drop_nulls("gross_c")

    print(f"  baskets the gate would fire on : {gated.height:,} "
          f"({gated.height / max(both.height, 1):.1%} of two-sided baskets)")
    if gated.is_empty():
        print("  none survive with both sides priced; lower --gate-threshold or")
        print("  loosen --max-basket-span")
        return
    g = gated.get_column("gross_c")
    n_sell = gated.filter(_dev("sum_last") > 0).height
    print(f"    of which SOLD (sum > $1)     : {n_sell:,}")
    print(f"    of which BOUGHT (sum < $1)   : {gated.height - n_sell:,}")
    print(f"\n  GROSS edge at executable prices, before any fee:")
    print(f"    median  {float(g.median()):+.2f}c per basket")
    print(f"    mean    {float(g.mean()):+.2f}c")
    print(f"    share positive  {float((g > 0).mean()):.1%}")
    print(f"    share above 5c  {float((g > 5).mean()):.1%}")
    print("\n  This is the whole PnL question in one number. A gross edge at or")
    print("  below zero means the gate's deviations are not capturable at the")
    print("  prices actually available, whatever the last-trade series shows,")
    print("  and no fee model can rescue it. A clearly positive gross edge means")
    print("  the tail is real and fees decide the rest.")

    # Same question for the whole two-sided population, as context.
    allx = both.with_columns(
        pl.when(_dev("sum_last") > 0).then(_dev("sum_bid_last"))
          .otherwise(-_dev("sum_ask_last")).alias("gross_c")
    ).drop_nulls("gross_c").get_column("gross_c")
    if allx.len():
        print(f"\n  For contrast, the SAME calculation on all {allx.len():,} two-sided")
        print(f"  baskets: median {float(allx.median()):+.2f}c, "
              f"share positive {float((allx > 0).mean()):.1%}")


def _self_test() -> int:
    import datetime as dt
    fails = []

    def ck(name, cond):
        if not cond:
            fails.append(name)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    D = dt.date(2025, 7, 4)

    def make(evt, mids, half_spreads, last_sides):
        """Build leg + side frames for one basket from known mids."""
        lrows, srows = [], []
        for i, (m, h, side) in enumerate(zip(mids, half_spreads, last_sides)):
            tk = f"{evt}-L{i}"
            ask, bid = m + h, m - h
            lrows.append({"event_ticker": evt, "date": D, "ticker": tk,
                          "close": ask if side == "ask" else bid})
            srows.append({"ticker": tk, "date": D, "taker_side": "yes",
                          "last": ask, "mean": ask, "n": 5})
            srows.append({"ticker": tk, "date": D, "taker_side": "no",
                          "last": bid, "mean": bid, "n": 5})
        return pl.DataFrame(lrows), pl.DataFrame(srows)

    print("Pure spread artefact: mids sum to exactly $1, all legs print at the ask")
    legs, sides = make("A", [20.0, 30.0, 50.0], [3.0, 3.0, 3.0], ["ask"] * 3)
    t = build_basket_table(legs, sides)
    ck("mid sums to 100 exactly", abs(float(t["sum_mid_mean"][0]) - 100.0) < 1e-9)
    ck("ask sum is 100 + sum(half) = 109", abs(float(t["sum_ask_mean"][0]) - 109.0) < 1e-9)
    ck("bid sum is 100 - sum(half) = 91", abs(float(t["sum_bid_mean"][0]) - 91.0) < 1e-9)
    ck("last-trade sum inherits the ask bias", abs(float(t["sum_last"][0]) - 109.0) < 1e-9)
    ck("basket spread is 18c", abs(float(t["basket_spread"][0]) - 18.0) < 1e-9)
    ck("no leg is missing a side", not bool(t["miss_ask"][0]) and not bool(t["miss_bid"][0]))

    print("\nReal mispricing: mids sum to $1.20, legs print on mixed sides")
    legs, sides = make("B", [40.0, 40.0, 40.0], [3.0, 3.0, 3.0], ["ask", "bid", "ask"])
    t = build_basket_table(legs, sides)
    ck("mid sum is 120, i.e. survives repricing",
       abs(float(t["sum_mid_mean"][0]) - 120.0) < 1e-9)
    ck("last-trade sum is 123 (two asks, one bid)",
       abs(float(t["sum_last"][0]) - 123.0) < 1e-9)

    print("\nMixed sides on a fair basket: last-trade dev is pure noise")
    legs, sides = make("C", [20.0, 30.0, 50.0], [3.0, 3.0, 3.0], ["ask", "bid", "ask"])
    t = build_basket_table(legs, sides)
    ck("mid still exactly 100", abs(float(t["sum_mid_mean"][0]) - 100.0) < 1e-9)
    ck("last-trade sum is 103, a fictitious +3c", abs(float(t["sum_last"][0]) - 103.0) < 1e-9)

    print("\nA leg with only one side is excluded from the mid population")
    legs = pl.DataFrame([{"event_ticker": "D", "date": D, "ticker": "D-L0", "close": 50.0},
                         {"event_ticker": "D", "date": D, "ticker": "D-L1", "close": 55.0}])
    sides = pl.DataFrame([
        {"ticker": "D-L0", "date": D, "taker_side": "yes", "last": 52.0, "mean": 52.0, "n": 3},
        {"ticker": "D-L0", "date": D, "taker_side": "no", "last": 48.0, "mean": 48.0, "n": 3},
        {"ticker": "D-L1", "date": D, "taker_side": "yes", "last": 57.0, "mean": 57.0, "n": 3},
    ])
    t = build_basket_table(legs, sides)
    ck("basket flagged as missing a bid", bool(t["miss_bid"][0]))
    ck("it would be filtered out of Test 2",
       t.filter(~pl.col("miss_ask") & ~pl.col("miss_bid")).height == 0)

    print("\nAsk/bid print gap limit")
    base = [{"ticker": "E-L0", "date": D, "taker_side": "yes", "last": 52.0,
             "mean": 52.0, "last_time": dt.datetime(2025, 7, 4, 10, 0), "n": 3},
            {"ticker": "E-L0", "date": D, "taker_side": "no", "last": 48.0,
             "mean": 48.0, "last_time": dt.datetime(2025, 7, 4, 10, 20), "n": 3},
            {"ticker": "E-L1", "date": D, "taker_side": "yes", "last": 54.0,
             "mean": 54.0, "last_time": dt.datetime(2025, 7, 4, 9, 0), "n": 3},
            {"ticker": "E-L1", "date": D, "taker_side": "no", "last": 46.0,
             "mean": 46.0, "last_time": dt.datetime(2025, 7, 4, 16, 0), "n": 3}]
    elegs = pl.DataFrame([{"event_ticker": "E", "date": D, "ticker": "E-L0", "close": 52.0},
                          {"event_ticker": "E", "date": D, "ticker": "E-L1", "close": 54.0}])
    esides = pl.DataFrame(base)
    t0 = build_basket_table(elegs, esides, max_side_gap_min=0)
    ck("no limit: mid is formed despite a 7-hour gap", not bool(t0["miss_mid"][0]))
    ck("worst-leg gap reported as 420 min",
       abs(float(t0["max_side_gap_min"][0]) - 420.0) < 1e-6)
    t1 = build_basket_table(elegs, esides, max_side_gap_min=30)
    ck("30-min limit drops the basket (one leg 7h apart)", bool(t1["miss_mid"][0]))
    ck("both sides are still present, so it is the GAP that excluded it",
       not bool(t1["miss_ask"][0]) and not bool(t1["miss_bid"][0]))
    t2 = build_basket_table(elegs.head(1), esides.head(2), max_side_gap_min=30)
    ck("the 20-min leg alone survives a 30-min limit", not bool(t2["miss_mid"][0]))
    ck("its mid is 50", abs(float(t2["sum_mid_mean"][0]) - 50.0) < 1e-9)

    print("\nCross-leg span limit (the constraint that actually matters)")
    # two legs, each internally tight, but 3 hours apart from each other
    flegs = pl.DataFrame([{"event_ticker": "F", "date": D, "ticker": "F-L0", "close": 52.0},
                          {"event_ticker": "F", "date": D, "ticker": "F-L1", "close": 54.0}])
    fsides = pl.DataFrame([
        {"ticker": "F-L0", "date": D, "taker_side": "yes", "last": 52.0, "mean": 52.0,
         "last_time": dt.datetime(2025, 7, 4, 9, 0), "n": 3},
        {"ticker": "F-L0", "date": D, "taker_side": "no", "last": 48.0, "mean": 48.0,
         "last_time": dt.datetime(2025, 7, 4, 9, 5), "n": 3},
        {"ticker": "F-L1", "date": D, "taker_side": "yes", "last": 54.0, "mean": 54.0,
         "last_time": dt.datetime(2025, 7, 4, 12, 0), "n": 3},
        {"ticker": "F-L1", "date": D, "taker_side": "no", "last": 46.0, "mean": 46.0,
         "last_time": dt.datetime(2025, 7, 4, 12, 5), "n": 3}])
    t = build_basket_table(flegs, fsides, max_side_gap_min=30, max_basket_span_min=0)
    ck("within-leg limit alone ACCEPTS legs 3h apart", not bool(t["miss_mid"][0]))
    ck("cross-leg span reported as 180 min",
       abs(float(t["basket_span_min"][0]) - 180.0) < 1e-6)
    t = build_basket_table(flegs, fsides, max_side_gap_min=30, max_basket_span_min=60)
    ck("cross-leg limit REJECTS them", bool(t["miss_mid"][0]))
    t = build_basket_table(flegs, fsides, max_side_gap_min=30, max_basket_span_min=240)
    ck("a loose cross-leg limit accepts them again", not bool(t["miss_mid"][0]))

    print("\nSign-asymmetry detector")
    pos = pl.DataFrame({"sum_last": [104.0] * 9 + [96.0]})
    r = describe(pos, "x", "sum_last")
    ck("90% positive is reported as such", r["pct_positive"] == 90.0)
    ck("signed mean is +3.2c", abs(r["signed_mean"] - 3.2) < 1e-9)
    bal = pl.DataFrame({"sum_last": [105.0] * 5 + [95.0] * 5})
    r = describe(bal, "x", "sum_last")
    ck("balanced population reads 50%", r["pct_positive"] == 50.0)
    ck("balanced signed mean is 0", abs(r["signed_mean"]) < 1e-9)
    ck("median |dev| is 5c either way", abs(r["median_abs"] - 5.0) < 1e-9)

    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg-prices", default=LEG_PRICES_PATH,
                    help="output of mece_sum_to_one_check.py")
    ap.add_argument("--min-side-trades", type=int, default=1,
                    help="require this many trades on a side before using it (default 1)")
    ap.add_argument("--max-side-gap", type=float, default=0.0,
                    help="minutes; require a leg's ask print and bid print to be "
                         "within this of each other before their average is used "
                         "as a mid. 0 = no limit (whole-day means, drift-prone). "
                         "Try 30 or 60: the project already found pooled per-leg "
                         "spreads read 5.61c against a windowed 4.68c, so a mid "
                         "built from prints hours apart carries the same bias.")
    ap.add_argument("--max-basket-span", type=float, default=0.0,
                    help="minutes; require ALL legs' prints within this span of "
                         "each other before a mid is used. This is the constraint "
                         "that matters: --max-side-gap only aligns each leg's own "
                         "two sides. 0 = no limit. The project's own full-basket "
                         "snapshots have a median cross-leg gap near 4 hours, so "
                         "expect heavy attrition at 30 or 60.")
    ap.add_argument("--gate-threshold", type=float, default=19.0,
                    help="cents; the cost gate's median |dev| on val was 19c")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    if not os.path.exists(args.leg_prices):
        raise SystemExit(f"{args.leg_prices} not found -- run mece_sum_to_one_check.py first")

    legs = pl.read_parquet(args.leg_prices)
    need = {"event_ticker", "date", "ticker", "close"}
    if not need.issubset(set(legs.columns)):
        raise SystemExit(f"{args.leg_prices} lacks {need - set(legs.columns)}")
    legs = legs.select(["event_ticker", "date", "ticker", "close"])
    print(f"{legs.height:,} leg-price rows across "
          f"{legs.select(['event_ticker', 'date']).unique().height:,} basket-days")

    months = months_of(legs.get_column("date").unique().to_list())
    print(f"scanning trades for {len(months)} month(s): {months[0]}..{months[-1]}")
    sides = load_side_prices(legs.get_column("ticker").unique().to_list(),
                             months, args.min_side_trades)
    if sides.is_empty():
        raise SystemExit("no side-split trades found -- check taker_side values")

    vals = sides.get_column(SIDE_COL).unique().to_list()
    print(f"taker_side values seen: {vals}")
    if ASK_SIDE not in vals or BID_SIDE not in vals:
        print(f"  WARNING: expected '{ASK_SIDE}' and '{BID_SIDE}'. If this venue "
              f"uses different labels the ask/bid assignment below is wrong.")

    per_basket = build_basket_table(legs, sides, args.max_side_gap,
                                    args.max_basket_span)
    per_basket.write_parquet("mece_spread_artefact_check.parquet")
    print("saved per-basket detail to mece_spread_artefact_check.parquet")
    report(per_basket, args.gate_threshold)


if __name__ == "__main__":
    main()