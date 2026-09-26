"""
build_mece_leg_spreads.py

MEASURE THE SPREAD ON MECE BASKET LEGS INSTEAD OF BORROWING THE LADDER'S.

WHY THIS EXISTS. Every MECE PnL result currently prices execution with a
single number: the GLOBAL MEDIAN spread from
pairwise_monotonicity_taker_side_results_corrected.parquet. Every run
prints `spread sources: global=N` with no ticker or series matches at all,
because that file is keyed by LADDER PAIRS -- threshold markets -- and the
MECE legs are weather brackets. Different population, one borrowed
constant, applied to every leg.

It is not a minor assumption. Backed out of a hold-to-settlement run, the
implied figure is ~4.7c per leg and the spread is 77-87% of total cost. And
the headline conclusions move with it:

    spread/leg   gate net/trade   gate total   does the model help?
    1.0c (flat)       +$0.775        +$2,293   +0.066 [+0.037, +0.103] YES
    4.7c (borrowed)   +$1.145          +$663   -0.011 [-0.118, +0.104] null

So whether the forecasting model adds measurable value to the profitable
rule currently depends on a number nobody measured on this population.
This script measures it.

THE METHOD, DELIBERATELY IDENTICAL TO THE LADDER'S. From
pairwise_monotonicity_taker_side_check_corrected.py:

    implied_spread(ticker) = | mean(yes_price | taker_side == "yes")
                            - mean(yes_price | taker_side == "no")  |

A taker buying YES lifts the offer; a taker selling YES (recorded as the
"no" side) hits the bid. The gap between the two sides' average traded
prices is therefore an estimate of the bid-ask spread, inferred from
trades because the order book is not retained historically.

Using the SAME estimator on both mechanisms is the point. A MECE-specific
spread computed a different way would not be comparable to the ladder
figure, and comparability between the two mechanisms is what this whole
project is built on.

WHAT THIS IS NOT. An implied spread from average traded prices is a proxy,
not a quote. It is biased by order flow imbalance: a leg that mostly traded
on one side has a noisy or absent estimate, which is why --min-n exists. It
cannot see the spread at any particular instant, only an average over the
window. Those limitations apply equally to the ladder numbers already in
use, which is the point -- this does not make the cost model right, it
makes it CONSISTENT and MEASURED rather than borrowed.

OUTPUT. A parquet in SpreadLookup's exact four-column shape
(leg_a, leg_b, spread_a, spread_b) with leg_a == leg_b == ticker, so it
drops straight into the existing evaluators with no code change:

    python evaluate_mece_settlement_pnl.py --split val \\
        --spreads mece_leg_spreads.parquet --max-participation 0.25

SpreadLookup.load() builds its per-ticker median from both the a and b
positions, so emitting each ticker as its own degenerate "pair" populates
the ticker table exactly as intended. The runs will then print
`spread sources: ticker=N` instead of `global=N`, which is itself the
check that this worked.

VERIFY IT FIRST:  python build_mece_leg_spreads.py --self-test
That plants a known spread in synthetic trades and asserts the estimator
recovers it, and asserts it returns nothing when there is no spread to
find. A measurement tool that has not been shown to measure a planted
effect is not evidence.
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]

TICKER_COL = "ticker"
PRICE_COL = "yes_price"
SIDE_COL = "taker_side"
COUNT_COL = "count"
TIME_COL = "created_time"


def _parity_path(kind: str, month: str) -> Path:
    """data/trades/trades_kalshi_{even,odd}/trades_{month}.parquet --
    the same convention run_real_training.load_months() uses. Only used
    when --trades is not given."""
    parity = "even" if int(month.split("-")[1]) % 2 == 0 else "odd"
    return _REPO_ROOT / "data" / kind / f"{kind}_kalshi_{parity}" / f"{kind}_{month}.parquet"


def _resolve_trade_paths(patterns, months):
    """Explicit --trades paths (globs allowed) win; otherwise fall back to
    the even/odd month convention.

    One file per month is the common layout, so --trades takes any number
    of paths and expands shell-style globs itself -- passing a quoted
    pattern therefore works whether or not the shell expanded it.
    """
    if patterns:
        out, missing = [], []
        for pat in patterns:
            p = Path(pat)
            if not p.is_absolute():
                # try as given, then relative to the script dir and repo root
                cands = [Path(pat), _THIS_DIR / pat, _REPO_ROOT / pat]
            else:
                cands = [p]
            hits = []
            for c in cands:
                hits = sorted(Path(h) for h in _glob.glob(str(c)))
                if hits:
                    break
            if hits:
                out.extend(hits)
            else:
                missing.append(pat)
        if missing:
            raise SystemExit(
                "these --trades patterns matched no files:\n  "
                + "\n  ".join(missing)
                + "\n\nPaths are tried as given, then relative to "
                  f"{_THIS_DIR} and {_REPO_ROOT}. Globs are expanded here, so "
                  "quote them if your shell does not match them.")
        # de-duplicate while keeping order
        seen, uniq = set(), []
        for p in out:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                uniq.append(p)
        return uniq

    paths = [_parity_path("trades", m) for m in months]
    found = [p for p in paths if p.exists()]
    for m, p in zip(months, paths):
        if not p.exists():
            print(f"  WARNING: no trades file for {m}: {p}")
    if not found:
        raise SystemExit(
            f"no trades files found for {months} under "
            f"{_REPO_ROOT / 'data' / 'trades'}.\n"
            f"  Pass them explicitly instead, e.g.\n"
            f"    --trades data/trades/trades_2025-0*.parquet\n"
            f"    --trades /abs/path/trades_2025-05.parquet /abs/path/trades_2025-06.parquet")
    return found


def load_trades(paths, leg_tickers, need_time: bool = False):
    """Scan each file separately, keeping only the columns needed and only
    the rows on MECE legs.

    Per-file scanning rather than one concat matters for two reasons: the
    column set can drift between months (a missing optional column in one
    file would otherwise break the whole concat), and pruning columns plus
    pushing the ticker filter into the scan keeps this cheap over many
    months instead of materialising every trade first.
    """
    import polars as pl

    want = [TICKER_COL, PRICE_COL, SIDE_COL] + ([TIME_COL] if need_time else [])
    frames, used = [], []
    for path in paths:
        lf = pl.scan_parquet(str(path))
        cols = lf.collect_schema().names()
        miss = [c for c in want if c not in cols]
        if miss:
            raise SystemExit(f"{path} is missing {miss}; it has {cols}")
        sel = [pl.col(TICKER_COL), pl.col(PRICE_COL).cast(pl.Float64),
               pl.col(SIDE_COL)]
        if need_time:
            sel.append(pl.col(TIME_COL))
        # `count` is optional: it is only needed for --weighted, and a file
        # without it should still contribute to the unweighted estimate.
        sel.append(pl.col(COUNT_COL).cast(pl.Float64) if COUNT_COL in cols
                   else pl.lit(1.0).alias(COUNT_COL))
        frames.append(lf.select(sel)
                        .filter(pl.col(TICKER_COL).is_in(list(leg_tickers))))
        used.append((path, COUNT_COL in cols))

    trades = pl.concat(frames, how="vertical_relaxed").collect()
    print(f"trades files used ({len(used)}):")
    for path, had_count in used:
        try:
            rel = path.resolve().relative_to(_REPO_ROOT)
        except ValueError:
            rel = path
        print(f"    {rel}" + ("" if had_count else f"   [no '{COUNT_COL}' column]"))
    return trades


def implied_spreads(trades, weighted: bool, min_n: int):
    """Per-ticker implied spread in cents, plus the per-side counts that
    decide whether to trust it.

    Returns a polars DataFrame: ticker, spread, n_yes, n_no.
    """
    import polars as pl

    have_count = COUNT_COL in trades.columns
    if weighted and not have_count:
        raise SystemExit(f"--weighted needs a '{COUNT_COL}' column; columns are "
                         f"{trades.columns}")

    price = pl.col(PRICE_COL).cast(pl.Float64)
    if weighted:
        cnt = pl.col(COUNT_COL).cast(pl.Float64)
        agg = [(price * cnt).sum().alias("_num"), cnt.sum().alias("_den"),
               pl.len().alias("n")]
    else:
        # Unweighted mean, matching the ladder estimator exactly.
        agg = [price.mean().alias("_num"), pl.lit(1.0).alias("_den"),
               pl.len().alias("n")]

    side = (trades.group_by([TICKER_COL, SIDE_COL]).agg(agg)
            .with_columns((pl.col("_num") / pl.col("_den")).alias("avg_price")))

    yes = (side.filter(pl.col(SIDE_COL) == "yes")
           .select([TICKER_COL, pl.col("avg_price").alias("p_yes"),
                    pl.col("n").alias("n_yes")]))
    no = (side.filter(pl.col(SIDE_COL) == "no")
          .select([TICKER_COL, pl.col("avg_price").alias("p_no"),
                   pl.col("n").alias("n_no")]))

    out = (yes.join(no, on=TICKER_COL, how="inner")
           .with_columns((pl.col("p_yes") - pl.col("p_no")).abs().alias("spread"))
           .filter((pl.col("n_yes") >= min_n) & (pl.col("n_no") >= min_n))
           .select([TICKER_COL, "spread", "n_yes", "n_no"]))
    return out


def windowed_spreads(trades, every: str, min_n: int, min_windows: int):
    """Implied spread estimated WITHIN short windows, then taken as the
    per-ticker median across windows.

    WHY THIS EXISTS. The pooled estimator averages every YES-side trade and
    every NO-side trade over a ticker's entire life. If the two sides did
    not trade at the same times -- and in a resolving market they rarely do
    -- then the difference between those averages contains the PRICE DRIFT
    between them, not just the bid-ask. A leg whose YES trades cluster early
    at 80c and whose NO trades cluster late at 30c records a 50c "spread"
    while the true spread might be 2c.

    The contamination is systematic and one-directional: drift can only
    widen |mean_yes - mean_no|, never narrow it, so the pooled figure is an
    UPPER BOUND. That is consistent with the pooled run's implausible tail
    (p90 15.8c, max 51.9c) against a venue where a real spread is low
    single digits.

    Estimating inside a 2-hour window -- the same grid the snapshots use --
    bounds how far the price can drift between the two sides' trades, so
    what is left is much closer to the spread itself. The per-ticker median
    across windows then discards the windows where flow was still lopsided.
    """
    import polars as pl

    if TIME_COL not in trades.columns:
        raise SystemExit(f"--window needs a '{TIME_COL}' column; columns are "
                         f"{trades.columns}")

    t = trades
    if not t.schema[TIME_COL].is_temporal():
        # epoch seconds are the only other plausible encoding here
        t = t.with_columns(
            pl.from_epoch(pl.col(TIME_COL).cast(pl.Int64), time_unit="s").alias(TIME_COL))

    per_side = (
        t.with_columns(pl.col(TIME_COL).dt.truncate(every).alias("_win"))
         .group_by([TICKER_COL, "_win", SIDE_COL])
         .agg(pl.col(PRICE_COL).mean().alias("avg_price"), pl.len().alias("n"))
    )
    yes = (per_side.filter(pl.col(SIDE_COL) == "yes")
           .select([TICKER_COL, "_win", pl.col("avg_price").alias("p_yes"),
                    pl.col("n").alias("n_yes")]))
    no = (per_side.filter(pl.col(SIDE_COL) == "no")
          .select([TICKER_COL, "_win", pl.col("avg_price").alias("p_no"),
                   pl.col("n").alias("n_no")]))

    per_window = (yes.join(no, on=[TICKER_COL, "_win"], how="inner")
                  .filter((pl.col("n_yes") >= min_n) & (pl.col("n_no") >= min_n))
                  .with_columns((pl.col("p_yes") - pl.col("p_no")).abs().alias("spread")))

    return (per_window.group_by(TICKER_COL)
            .agg(pl.col("spread").median().alias("spread"),
                 pl.len().alias("n_windows"),
                 pl.col("n_yes").sum().alias("n_yes"),
                 pl.col("n_no").sum().alias("n_no"))
            .filter(pl.col("n_windows") >= min_windows)
            .select([TICKER_COL, "spread", "n_yes", "n_no", "n_windows"]))


# ---------------------------------------------------------------------------
# self-test: prove the estimator can both find and fail to find a spread
# ---------------------------------------------------------------------------

def _self_test():
    import random
    import polars as pl

    def synth(ticker, mid, spread, n=400, seed=0):
        """Trades around `mid` where YES takers lift the offer (mid +
        spread/2) and NO takers hit the bid (mid - spread/2), plus noise."""
        import random
        r = random.Random(seed)
        rows = []
        for i in range(n):
            s = "yes" if i % 2 == 0 else "no"
            half = spread / 2.0 if s == "yes" else -spread / 2.0
            rows.append({TICKER_COL: ticker, SIDE_COL: s,
                         PRICE_COL: mid + half + r.gauss(0, 0.5),
                         COUNT_COL: r.randint(1, 20)})
        return pl.DataFrame(rows)

    ok, bad = [], []

    # 1. a planted 4c spread must be recovered
    df = synth("PLANTED", 30.0, 4.0, seed=1)
    got = float(implied_spreads(df, weighted=False, min_n=5)["spread"][0])
    (ok if abs(got - 4.0) < 0.3 else bad).append(
        f"planted 4.0c -> recovered {got:.3f}c")

    # 2. no spread planted: the estimate must be near zero
    df0 = synth("FLAT", 30.0, 0.0, seed=2)
    got0 = float(implied_spreads(df0, weighted=False, min_n=5)["spread"][0])
    (ok if got0 < 0.3 else bad).append(f"planted 0.0c -> recovered {got0:.3f}c")

    # 3. a leg trading on only one side must be EXCLUDED, not guessed at
    one_sided = pl.DataFrame([{TICKER_COL: "ONESIDE", SIDE_COL: "yes",
                               PRICE_COL: 30.0, COUNT_COL: 1}] * 50)
    n_rows = implied_spreads(one_sided, weighted=False, min_n=5).height
    (ok if n_rows == 0 else bad).append(
        f"one-sided leg -> {n_rows} rows (must be 0)")

    # 4. min_n must actually exclude thin legs
    thin = synth("THIN", 30.0, 4.0, n=6, seed=3)
    n_thin = implied_spreads(thin, weighted=False, min_n=20).height
    (ok if n_thin == 0 else bad).append(
        f"3 trades/side under --min-n 20 -> {n_thin} rows (must be 0)")

    # 5. volume weighting must not change a spread that is flat in volume
    dfw = synth("WEIGHTED", 30.0, 4.0, seed=4)
    a = float(implied_spreads(dfw, weighted=False, min_n=5)["spread"][0])
    b = float(implied_spreads(dfw, weighted=True, min_n=5)["spread"][0])
    (ok if abs(a - b) < 0.6 else bad).append(
        f"unweighted {a:.3f}c vs weighted {b:.3f}c")

    # 6. THE DECISIVE ONE: a 2c spread plus 20c of drift, with YES trading
    #    early and NO late. The pooled estimator must be badly inflated and
    #    the windowed estimator must recover ~2c. If the windowed version
    #    cannot separate these, it is not worth running.
    from datetime import datetime, timedelta
    base, rows = datetime(2025, 5, 1), []
    rr = random.Random(9)
    for w in range(12):                       # 12 two-hour windows
        drift = 20.0 * w / 11.0               # price walks 0 -> 20c
        for i in range(12):
            # inside each window both sides trade, so the window-local
            # estimate sees the spread; across windows the level moves
            s = "yes" if i % 2 == 0 else "no"
            half = 1.0 if s == "yes" else -1.0     # 2c spread
            rows.append({TICKER_COL: "DRIFTY", SIDE_COL: s,
                         PRICE_COL: 20.0 + drift + half + rr.gauss(0, .2),
                         COUNT_COL: 1,
                         TIME_COL: base + timedelta(hours=2 * w, minutes=5 * i)})
    # and make the POOLED estimate lopsided in time: extra YES early, NO late
    for i in range(40):
        rows.append({TICKER_COL: "DRIFTY", SIDE_COL: "yes", PRICE_COL: 21.0,
                     COUNT_COL: 1, TIME_COL: base + timedelta(minutes=i)})
        rows.append({TICKER_COL: "DRIFTY", SIDE_COL: "no", PRICE_COL: 39.0,
                     COUNT_COL: 1, TIME_COL: base + timedelta(hours=22, minutes=i)})
    dfd = pl.DataFrame(rows)
    pooled = float(implied_spreads(dfd, weighted=False, min_n=5)["spread"][0])
    win = windowed_spreads(dfd, every="2h", min_n=3, min_windows=3)
    winv = float(win["spread"][0]) if win.height else float("nan")
    (ok if pooled > 5.0 else bad).append(
        f"drift case, POOLED -> {pooled:.2f}c (must be badly inflated, >5c)")
    (ok if abs(winv - 2.0) < 1.0 else bad).append(
        f"drift case, WINDOWED -> {winv:.2f}c (planted 2.0c)")

    print("SELF-TEST")
    for line in ok:
        print(f"  PASS  {line}")
    for line in bad:
        print(f"  FAIL  {line}")
    print(f"\n{len(ok)}/{len(ok) + len(bad)} checks passed")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--self-test", action="store_true",
                    help="plant a known spread in synthetic trades and assert "
                         "the estimator recovers it. Run this first.")
    ap.add_argument("--trades", nargs="+", default=None, metavar="PATH",
                    help="explicit trades parquet paths, one or many; shell-style "
                         "globs are expanded here too, e.g. "
                         "--trades data/trades_2025-0*.parquet. Paths are tried "
                         "as given, then relative to the script dir and repo "
                         "root. Overrides --months.")
    ap.add_argument("--months", nargs="+", default=None,
                    help="only used when --trades is not given: falls back to the "
                         "data/trades/trades_kalshi_{even,odd}/ convention.")
    ap.add_argument("--legs", default="mece_sum_to_one_leg_prices.parquet",
                    help="parquet listing the MECE basket legs; needs a "
                         "'ticker' column.")
    ap.add_argument("--out", default="mece_leg_spreads.parquet")
    ap.add_argument("--min-n", type=int, default=10,
                    help="minimum trades on EACH side before a leg's spread is "
                         "trusted. Below this the estimate is order-flow noise.")
    ap.add_argument("--window", default=None, metavar="EVERY",
                    help="estimate the spread WITHIN windows of this size and "
                         "take the per-ticker median, e.g. --window 2h. This "
                         "strips the price drift that inflates the pooled "
                         "estimate. Needs a '" + TIME_COL + "' column.")
    ap.add_argument("--min-windows", type=int, default=3,
                    help="--window only: minimum qualifying windows per leg.")
    ap.add_argument("--weighted", action="store_true",
                    help="volume-weight the per-side average. OFF by default so "
                         "the estimator matches the ladder's exactly.")
    ap.add_argument("--compare", type=float, default=4.7,
                    help="the borrowed global median, in cents, to compare "
                         "against. Set from the value your runs imply.")
    args = ap.parse_args()

    if args.self_test:
        _self_test()

    import polars as pl

    # ---- MECE leg universe ---------------------------------------------
    legs_path = Path(args.legs)
    if not legs_path.is_absolute():
        for cand in (_THIS_DIR / legs_path, _REPO_ROOT / legs_path, Path.cwd() / legs_path):
            if cand.exists():
                legs_path = cand
                break
    if not legs_path.exists():
        raise SystemExit(f"MECE legs file not found: {args.legs}\n"
                         f"  It is written by mece_sum_to_one_check.py and is the "
                         f"same file run_real_training.py reads as "
                         f"MECE_LEG_PRICES_PATH.")
    legs = pl.read_parquet(legs_path)
    if TICKER_COL not in legs.columns:
        raise SystemExit(f"{legs_path} has no '{TICKER_COL}' column; columns are "
                         f"{legs.columns}")
    leg_tickers = set(legs[TICKER_COL].unique().to_list())
    print(f"MECE legs: {len(leg_tickers):,} distinct tickers from {legs_path.name}")

    # ---- trades ---------------------------------------------------------
    paths = _resolve_trade_paths(args.trades, args.months or PILOT_MONTHS)
    trades = load_trades(paths, leg_tickers, need_time=bool(args.window))
    print(f"  {trades.height:,} rows on MECE legs "
          f"({trades[TICKER_COL].n_unique():,} distinct legs traded)")
    if trades.height == 0:
        raise SystemExit(
            "no trades on any MECE leg. Either the trades files and the legs "
            "file cover different periods, or the ticker spellings differ "
            "between them -- print a few of each before assuming the former.")

    # ---- estimate --------------------------------------------------------
    if args.window:
        sp = windowed_spreads(trades, every=args.window, min_n=args.min_n,
                              min_windows=args.min_windows)
        if sp.height == 0:
            raise SystemExit(
                f"no leg had >= {args.min_windows} windows of {args.window} with "
                f">= {args.min_n} trades on BOTH sides. Loosen --min-n / "
                f"--min-windows, or widen --window.")
        pooled = implied_spreads(trades, weighted=args.weighted, min_n=args.min_n)
        _pooled_med = float(pooled["spread"].median()) if pooled.height else float("nan")
    else:
        sp = implied_spreads(trades, weighted=args.weighted, min_n=args.min_n)
        if sp.height == 0:
            raise SystemExit(f"no leg had >= {args.min_n} trades on BOTH sides. "
                             f"Lower --min-n, but treat the result as noise.")
        _pooled_med = None

    s = sp["spread"]
    q = [float(s.quantile(x)) for x in (0.10, 0.25, 0.50, 0.75, 0.90)]
    print("\n" + "=" * 78)
    print("MEASURED IMPLIED SPREAD ON MECE LEGS (cents per leg)")
    print("=" * 78)
    print(f"  legs with >= {args.min_n} trades per side : {sp.height:,} "
          f"of {len(leg_tickers):,} ({100.0 * sp.height / len(leg_tickers):.1f}%)")
    print(f"  weighting                            : "
          f"{'volume' if args.weighted else 'unweighted (matches ladder)'}")
    if args.window:
        print(f"  estimator                            : WINDOWED ({args.window}), "
              f"median over >= {args.min_windows} windows per leg")
        print(f"  pooled median on the same trades     : {_pooled_med:.2f}c")
        print(f"  drift removed by windowing           : "
              f"{_pooled_med - q[2]:+.2f}c "
              f"({100.0 * (1 - q[2] / _pooled_med):.0f}% of the pooled figure)")
    print(f"  p10 {q[0]:.2f}   p25 {q[1]:.2f}   MEDIAN {q[2]:.2f}   "
          f"p75 {q[3]:.2f}   p90 {q[4]:.2f}")
    print(f"  mean {float(s.mean()):.2f}   min {float(s.min()):.2f}   "
          f"max {float(s.max()):.2f}")

    print("\n  AGAINST THE BORROWED LADDER MEDIAN")
    ratio = q[2] / args.compare if args.compare else float("nan")
    print(f"    borrowed (ladder global median) : {args.compare:.2f}c")
    print(f"    measured (MECE legs, median)    : {q[2]:.2f}c   "
          f"({ratio:.2f}x the borrowed value)")
    if ratio < 0.7:
        print("    -> MECE legs are TIGHTER than the borrowed figure. Every cost")
        print("       number in the PnL work is overstated, the cost gate admits")
        print("       more events, and the model's marginal contribution should")
        print("       be re-tested at this spread.")
    elif ratio > 1.3:
        print("    -> MECE legs are WIDER than the borrowed figure. The existing")
        print("       results are optimistic and the profitable population")
        print("       shrinks. Re-run every PnL table with this file.")
    else:
        print("    -> Close to the borrowed figure. The substitution was")
        print("       defensible after all, which is itself worth reporting.")

    # ---- write in SpreadLookup's shape ----------------------------------
    out = sp.select([
        pl.col(TICKER_COL).alias("leg_a"),
        pl.col(TICKER_COL).alias("leg_b"),
        pl.col("spread").alias("spread_a"),
        pl.col("spread").alias("spread_b"),
    ])
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _THIS_DIR / out_path
    out.write_parquet(out_path)

    print(f"\n  written: {out_path}")
    print("  Each leg is emitted as its own degenerate pair so SpreadLookup's")
    print("  per-ticker median picks it up unchanged. Use it with:")
    print(f"    python evaluate_mece_settlement_pnl.py --split val \\")
    print(f"        --spreads {out_path.name} --max-participation 0.25")
    print("  and confirm the run prints 'spread sources: ticker=N' rather than")
    print("  'global=N'. If it still says global, the tickers did not match and")
    print("  the file is not being used.")

    print("\n" + "=" * 78)
    print("LIMITATIONS THAT TRAVEL WITH THIS NUMBER")
    print("=" * 78)
    print("  An implied spread from average traded prices is a proxy, not a")
    print("  quote. It is biased by one-sided order flow, it is an average over")
    print("  the whole period rather than the spread at any instant, and legs")
    print(f"  below --min-n {args.min_n} are excluded entirely rather than")
    print("  estimated -- so the covered population is the more liquid one and")
    print("  the median here is likely TIGHTER than the spread a trader would")
    print("  face on the thin legs the cost gate tends to select.")
    print("  Every one of these caveats applies equally to the ladder numbers")
    print("  already in use. This makes the cost model consistent and measured,")
    print("  not correct.")


if __name__ == "__main__":
    main()