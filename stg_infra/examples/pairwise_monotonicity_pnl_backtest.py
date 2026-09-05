"""
pairwise_monotonicity_pnl_backtest.py

Two-scenario PnL backtest for the ladder same-side monotonicity violations
found by pairwise_monotonicity_taker_side_check.py. Answers "does this
create PnL" by simulating the riskless cross-strike position implied by
each flagged violation, under two brackets:

  1. IDEALIZED -- captures the full trade-price-implied gap with zero
     execution friction (no fees, no spread cost). Upper bound.
  2. REALISTIC -- subtracts (a) Kalshi's actual taker fee on both legs,
     and (b) a conservative execution-cost haircut derived from the
     trade-implied spread already computed per leg (spread_a/spread_b),
     since a taker order can't be assumed to fill at the exact same-side
     average price used to compute the gap.

This brackets the estimate rather than pretending to a false precision --
the convention used in prediction-market / event-driven backtesting
research (e.g. Wolfers & Zitzewitz's NBER survey on prediction markets,
and applied studies like Quantpedia's Polymarket mean-reversion backtest,
which reports a zero-cost case alongside a flat-friction case rather than
one point estimate). It does NOT require order-book/quote data -- this
investigation's Kalshi orderbook snapshots turned out to be too sparse
and coarse (~30s polling cadence, long stretches of byte-identical
snapshots) to serve as a precise fill simulator, so the realistic
scenario instead uses the trade-implied spread already on hand as a
conservative proxy. If you later get better-quality quote data, replace
`spread_haircut_cents` with a real quoted-spread-based cost for a
tighter estimate.

ECONOMIC LOGIC (why the "gap" is a locked-in minimum profit, not just a
statistical curiosity):
For adjacent ladder legs A (lower threshold) and B (higher threshold) in
the same family, crossing threshold B implies you also crossed threshold
A, so P(A) >= P(B) must hold in a no-arbitrage market, i.e.
yes_price(A) >= yes_price(B). A "same-side violation" is
yes_price(A) < yes_price(B), observed on trades sharing a taker_side.
The position "buy YES-A, buy NO-B" costs price_A + (100 - price_B) cents
per contract and pays out:
  - outcome < A:        NO-B pays $1, YES-A pays $0  -> total $1
  - A <= outcome < B:    NO-B pays $1, YES-A pays $1  -> total $2
  - outcome >= B:        NO-B pays $0, YES-A pays $1  -> total $1
Payout is *always* >= $1, so the guaranteed minimum profit per contract is
  100 - (price_A + (100 - price_B)) = price_B - price_A = the violation gap
This script uses that guaranteed minimum (not the optional upside from
landing between the strikes) as the conservative basis for PnL -- actual
realized payout could be higher, essentially never lower (barring a
contract that fails to resolve/settle at all, which this script does not
model).

ASSUMPTIONS -- verify each of these against your real schema before
trusting the output, then delete this checklist:
  [ ] RESULTS_PATH columns match pairwise_monotonicity_taker_side_check.py's
      output: leg_a, leg_b, same_side_yes_violation, same_side_yes_gap,
      same_side_no_violation, same_side_no_gap, spread_a, spread_b,
      same_side_yes_n_a/n_b, same_side_no_n_a/n_b.
  [ ] Prices and gaps are in CENTS (0-100 scale), matching every other
      script in this investigation. If yours are in dollars (0-1), set
      PRICE_SCALE = 1.0 instead of 100.0.
  [ ] Kalshi's standard taker fee formula, as published in their fee
      schedule (checked 2026-09-02): fee = ceil_to_centicent(
      0.07 * C * P * (1-P)), P = contract price in DOLLARS, C = contract
      count, standard multiplier M=1. Some series carry non-standard
      multipliers -- this script assumes the standard case throughout.
      Re-check https://kalshi.com/docs/kalshi-fee-schedule.pdf for the
      current schedule before relying on this for anything real.
  [ ] No settlement/exercise fee is assumed at resolution -- the
      published schedule only documents entry taker/maker fees. If
      Kalshi charges anything at settlement, realistic PnL below is
      slightly overstated.
  [ ] TRADES_PATH / TICKER_COL / SIDE_COL / PRICE_COL below, if you want
      the sharper real-leg-price fee sizing (optional -- see
      `real_leg_prices()`). Without it, the script falls back to a
      conservative fixed-price assumption for fee sizing (see
      `price_legs_for_fees` docstring for why 50 cents is a safe
      worst-case default).
"""

import glob
import math
import os
import polars as pl

# Points at the CORRECTED pairs file (built from yes_sub_title
# bracket/upper_tail/lower_tail classification -- see
# pairwise_monotonicity_taker_side_check_corrected.py) rather than the
# original regex-ticker-suffix results. The corrected file only contains
# crypto/financials (+ a handful of mentions/entertainment/politics) pairs,
# since weather/economics/exotics correctly produce zero valid pairs now.
RESULTS_PATH = "pairwise_monotonicity_taker_side_results_corrected.parquet"

TARGET_MONTHS = ["2025-10", "2025-11"]  # same window the corrected results were built from


def _trades_month_globs(month: str) -> str:
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/trades/trades_kalshi_{parity}/trades_{month}.parquet"


TRADES_PATH = None  # kept for backwards compat with anything importing this name directly; real_leg_prices()
                     # below now combines every month in TARGET_MONTHS instead of reading a single file

TICKER_COL = "ticker"
SIDE_COL = "taker_side"
PRICE_COL = "yes_price"

PRICE_SCALE = 100.0                     # cents scale; set to 1.0 if your prices are already 0-1 dollars
CONTRACTS_PER_OPPORTUNITY = 100         # flat size assumption -- no order-book depth cap backs this, see caveat above
TAKER_FEE_RATE = 0.07
SPREAD_HAIRCUT_FRACTION = 1.0           # realistic scenario: fraction of (spread_a + spread_b) charged as extra cost
MIN_N_PER_SIDE = 20                     # restrict to the well-sampled subset by default (0 = include everything)

# --- category classification -------------------------------------------
#
# Kalshi's own API docs say NOT to infer category from the ticker string --
# "categories and subcategories ... are not part of the ticker convention,"
# they're "discovery tools only." The documented correct source is the
# `category` field Kalshi returns directly on each series object via
# GET /series (see fetch_kalshi_series_categories.py, which pulls this into
# SERIES_CATEGORIES_PATH below). When that lookup file is present, it takes
# priority; the CRYPTO_KEYS/SPORTS_KEYS/WEATHER_KEYS keyword heuristic below
# is now only a FALLBACK for tickers whose series isn't in the lookup (e.g.
# lookup file not yet fetched, or a series created after the fetch) -- and
# unmapped tickers are labeled "other (unmapped)", not silently "other", so
# it stays visible which ones still aren't backed by a real Kalshi category.
SERIES_CATEGORIES_PATH = "kalshi_series_categories.parquet"  # from fetch_kalshi_series_categories.py

CRYPTO_KEYS = ["BTC", "ETH", "SHIBA", "XRP", "SOL", "DOGE"]
SPORTS_KEYS = ["NBA", "MLB", "NFL", "NHL", "WINS", "NCAAF", "NCAAB", "SOCCER", "EPL"]
WEATHER_KEYS = ["HIGH", "LOW", "ARCTICICE", "TEMP", "RAIN", "SNOW", "HURRICANE"]

# Kalshi's raw category strings, normalized to the coarse labels used
# throughout this investigation's report. Anything not listed here is kept
# as Kalshi's own category string, lowercased (e.g. "financials",
# "economics", "politics", "companies", "world", "science and technology") --
# this is exactly what replaces the old catch-all "other" bucket with real,
# distinct categories.
KALSHI_CATEGORY_NORMALIZATION = {
    "crypto": "crypto",
    "sports": "sports",
    "climate and weather": "weather/climate",
    "weather": "weather/climate",
    "climate": "weather/climate",
}

_series_category_lookup: dict[str, str] | None = None  # lazy-loaded, module-level cache


def _load_series_category_lookup() -> dict[str, str]:
    global _series_category_lookup
    if _series_category_lookup is not None:
        return _series_category_lookup
    if not os.path.exists(SERIES_CATEGORIES_PATH):
        _series_category_lookup = {}
        return _series_category_lookup
    lookup_df = pl.read_parquet(SERIES_CATEGORIES_PATH)
    _series_category_lookup = {
        row["series_ticker"]: row["category"]
        for row in lookup_df.iter_rows(named=True)
        if row["series_ticker"] is not None and row["category"] is not None
    }
    return _series_category_lookup


def _series_prefix(ticker: str) -> str:
    """Kalshi's documented ticker format is SERIES-EVENT-STRIKE (e.g.
    'KXHIGHMIA-25NOV07-B86.5' -> series ticker 'KXHIGHMIA'). The series
    ticker is always the leading segment before the first hyphen."""
    return ticker.split("-", 1)[0]


def print_category_coverage(results: pl.DataFrame) -> None:
    """Diagnostic printed automatically at the start of main() (and reused by
    pairwise_monotonicity_pnl_sensitivity.py): how many of these pairs got a
    real Kalshi-sourced category (via SERIES_CATEGORIES_PATH) vs. fell back
    to the keyword heuristic, plus the resulting category breakdown. This
    used to be a separate script (pairwise_monotonicity_recategorize_check.py)
    -- folded in here so it runs as part of the normal pipeline instead of
    requiring an extra step."""
    lookup = _load_series_category_lookup()
    if not lookup:
        print(f"NOTE: {SERIES_CATEGORIES_PATH} not found or empty -- every ticker will fall "
              f"back to the keyword heuristic ('other (unmapped)' for anything that isn't "
              f"crypto/sports/weather). Run fetch_kalshi_series_categories.py first to use "
              f"Kalshi's real per-series categories instead.\n")
    else:
        print(f"Loaded {len(lookup)} series -> category mappings from {SERIES_CATEGORIES_PATH}.")

    categorized = results.with_columns(
        pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category")
    )
    n_unmapped = categorized.filter(pl.col("category") == "other (unmapped)").height
    if n_unmapped:
        print(f"  {n_unmapped} / {categorized.height} pairs still fall back to the keyword "
              f"heuristic ('other (unmapped)') -- their series ticker wasn't found in "
              f"{SERIES_CATEGORIES_PATH}.")

    by_cat = (
        categorized.group_by("category")
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("same_side_yes_violation").mean().alias("same_side_yes_rate"),
            pl.col("same_side_no_violation").mean().alias("same_side_no_rate"),
        )
        .sort("n_pairs", descending=True)
    )
    print("category breakdown (pair count + same-side violation rate):")
    for r in by_cat.iter_rows(named=True):
        yr = f"{r['same_side_yes_rate']:.1%}" if r["same_side_yes_rate"] is not None else "n/a"
        nr = f"{r['same_side_no_rate']:.1%}" if r["same_side_no_rate"] is not None else "n/a"
        print(f"  {r['category']:<22} n={r['n_pairs']:<6} same_side_yes_rate={yr:<8} same_side_no_rate={nr}")
    print()


def classify_ticker(ticker: str) -> str:
    lookup = _load_series_category_lookup()
    raw_category = lookup.get(_series_prefix(ticker))
    if raw_category is not None:
        return KALSHI_CATEGORY_NORMALIZATION.get(raw_category.lower(), raw_category.lower())

    # Fallback: lookup file missing, or this series isn't in it yet.
    t = ticker.upper()
    if any(k in t for k in CRYPTO_KEYS):
        return "crypto"
    if any(k in t for k in SPORTS_KEYS):
        return "sports"
    if any(k in t for k in WEATHER_KEYS):
        return "weather/climate"
    return "other (unmapped)"


def taker_fee_dollars(price_cents: float, contracts: float) -> float:
    """Kalshi standard taker fee: ceil_to_centicent(0.07 * C * P * (1-P)), P in dollars.

    Note P*(1-P) is symmetric under P -> 1-P, so this is correct whether
    you plug in a leg's YES price or its NO price (100 - YES price) --
    the fee on "buy NO-B at (100-price_B)" equals the fee you'd compute
    from price_B directly. That symmetry is what lets the rest of this
    script use each leg's own price_cents without converting to NO terms.
    """
    p = price_cents / PRICE_SCALE
    raw = TAKER_FEE_RATE * contracts * p * (1 - p)
    return math.ceil(raw * 10000) / 10000  # round UP to the nearest $0.0001 (centicent)


def price_legs_for_fees(gap_cents: float) -> tuple[float, float]:
    """
    Conservative fallback when real per-leg prices aren't available: price
    both legs at 50 cents, which is where Kalshi's fee formula is at its
    MAXIMUM (P*(1-P) peaks at P=0.5) -- so fees computed this way are an
    upper bound on the true fee, keeping the realistic scenario
    conservative (understating profit) rather than overstating it.
    Superseded automatically by real_leg_prices() below when trades data
    is available and successfully joins.
    """
    return 50.0, 50.0


def real_leg_prices(trades_path: str | None = None) -> pl.DataFrame | None:
    """avg yes_price per (ticker, taker_side) -- sharper fee sizing than the 50/50 fallback.

    trades_path is accepted for backwards compatibility (e.g. pnl_sensitivity.py calling
    real_leg_prices(base.TRADES_PATH)) but is no longer the primary path: with TRADES_PATH now
    None by default, this combines every month in TARGET_MONTHS (mirrors
    pairwise_monotonicity_taker_side_check_corrected.py's load_trades()) so fee sizing reflects
    both Oct and Nov trades instead of silently only one month. Pass an explicit trades_path to
    override with a single file instead."""
    paths = []
    if trades_path:
        if os.path.exists(trades_path):
            paths = [trades_path]
        else:
            print(f"WARNING: could not build real leg prices from {trades_path}: file not found")
            return None
    else:
        for m in TARGET_MONTHS:
            tp = _trades_month_globs(m)
            if glob.glob(tp):
                paths.append(tp)
            else:
                print(f"WARNING: no trades file for {m}: {tp}")
        if not paths:
            return None
    try:
        trades = pl.concat([pl.scan_parquet(p).collect() for p in paths])
        return trades.group_by([TICKER_COL, SIDE_COL]).agg(
            pl.col(PRICE_COL).mean().alias("avg_price")
        )
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not build real leg prices from {paths}: {e}")
        return None


def build_opportunities(results: pl.DataFrame) -> pl.DataFrame:
    """Long-format: one row per (pair, side) flagged same-side violation."""
    rows = []
    for side in ("yes", "no"):
        viol_col = f"same_side_{side}_violation"
        gap_col = f"same_side_{side}_gap"
        n_a_col = f"same_side_{side}_n_a"
        n_b_col = f"same_side_{side}_n_b"

        sub = results.filter(pl.col(viol_col) == True)  # noqa: E712
        if MIN_N_PER_SIDE > 0:
            sub = sub.filter(
                (pl.col(n_a_col) >= MIN_N_PER_SIDE) & (pl.col(n_b_col) >= MIN_N_PER_SIDE)
            )
        if sub.height == 0:
            continue
        sub = sub.with_columns(
            pl.lit(side).alias("side"),
            pl.col(gap_col).alias("gap_cents"),
            ((pl.col("spread_a") + pl.col("spread_b")) * SPREAD_HAIRCUT_FRACTION).alias("spread_haircut_cents"),
            pl.col("leg_a").map_elements(classify_ticker, return_dtype=pl.Utf8).alias("category"),
        )
        rows.append(sub.select(["leg_a", "leg_b", "side", "category", "gap_cents", "spread_haircut_cents"]))

    if not rows:
        return pl.DataFrame(
            schema={
                "leg_a": pl.Utf8, "leg_b": pl.Utf8, "side": pl.Utf8, "category": pl.Utf8,
                "gap_cents": pl.Float64, "spread_haircut_cents": pl.Float64,
            }
        )
    return pl.concat(rows)


def attach_leg_prices(opportunities: pl.DataFrame, prices: pl.DataFrame | None) -> pl.DataFrame:
    if prices is None or opportunities.height == 0:
        return opportunities.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("price_a_cents"),
            pl.lit(None, dtype=pl.Float64).alias("price_b_cents"),
        )
    out = opportunities.join(
        prices.rename({TICKER_COL: "leg_a", SIDE_COL: "side", "avg_price": "price_a_cents"}),
        on=["leg_a", "side"], how="left",
    ).join(
        prices.rename({TICKER_COL: "leg_b", SIDE_COL: "side", "avg_price": "price_b_cents"}),
        on=["leg_b", "side"], how="left",
    )
    return out


def compute_pnl(opportunities: pl.DataFrame, contracts: float = CONTRACTS_PER_OPPORTUNITY) -> pl.DataFrame:
    if opportunities.height == 0:
        return opportunities

    idealized_pnl, realistic_pnl, fees_dollars, priced_from_trades = [], [], [], []

    for row in opportunities.iter_rows(named=True):
        gap_cents = row["gap_cents"]
        haircut_cents = row.get("spread_haircut_cents") or 0.0

        idealized = (gap_cents / PRICE_SCALE) * contracts  # no cost at all

        price_a, price_b = row.get("price_a_cents"), row.get("price_b_cents")
        used_real_prices = price_a is not None and price_b is not None
        if not used_real_prices:
            price_a, price_b = price_legs_for_fees(gap_cents)

        fee_a = taker_fee_dollars(price_a, contracts)
        fee_b = taker_fee_dollars(price_b, contracts)
        total_fee = fee_a + fee_b

        realistic_gap_cents = gap_cents - haircut_cents  # can go negative -- that's a real loss after costs
        realistic = (realistic_gap_cents / PRICE_SCALE) * contracts - total_fee

        idealized_pnl.append(idealized)
        realistic_pnl.append(realistic)
        fees_dollars.append(total_fee)
        priced_from_trades.append(used_real_prices)

    return opportunities.with_columns(
        pl.Series("idealized_pnl_usd", idealized_pnl),
        pl.Series("realistic_pnl_usd", realistic_pnl),
        pl.Series("fees_usd", fees_dollars),
        pl.Series("priced_from_real_trades", priced_from_trades),
    )


def summarize(pnl: pl.DataFrame, label: str):
    if pnl.height == 0:
        print(f"=== {label}: no opportunities (check MIN_N_PER_SIDE, or that violations exist) ===\n")
        return
    print(f"=== {label} (n={pnl.height} opportunities, {CONTRACTS_PER_OPPORTUNITY} contracts each, "
          f"MIN_N_PER_SIDE={MIN_N_PER_SIDE}) ===")
    print(f"idealized total PnL (zero friction):        ${pnl['idealized_pnl_usd'].sum():,.2f}")
    print(f"realistic total PnL (fees + spread haircut): ${pnl['realistic_pnl_usd'].sum():,.2f}")
    print(f"realistic win rate (opportunities with PnL > 0): {(pnl['realistic_pnl_usd'] > 0).mean():.1%}")
    print(f"realistic median PnL per opportunity: ${pnl['realistic_pnl_usd'].median():,.2f}")
    print(f"total fees paid: ${pnl['fees_usd'].sum():,.2f}")
    n_real_priced = int(pnl["priced_from_real_trades"].sum())
    print(f"opportunities priced from real trade data (vs. 50c conservative fallback): "
          f"{n_real_priced} of {pnl.height}")
    print()


def summarize_by_category(pnl: pl.DataFrame):
    if pnl.height == 0:
        return
    summary = (
        pnl.group_by("category")
        .agg(
            pl.len().alias("n_opportunities"),
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


def sensitivity_fees_only(pnl: pl.DataFrame):
    """Middle-ground scenario: fees only, no spread haircut -- since the haircut
    is the most subjective assumption in this script, this shows how much of
    the realistic-scenario gap is fees vs. the haircut."""
    if pnl.height == 0:
        return
    fees_only = (pnl["gap_cents"] / PRICE_SCALE * CONTRACTS_PER_OPPORTUNITY) - pnl["fees_usd"]
    print("=== Sensitivity: fees-only scenario (no spread haircut) ===")
    print(f"total PnL: ${fees_only.sum():,.2f}")
    print(f"win rate: {(fees_only > 0).mean():.1%}")
    print()


def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found -- update RESULTS_PATH at the top of this script "
            f"to point at your actual pairwise_monotonicity_taker_side_check.py output."
        )

    results = pl.read_parquet(RESULTS_PATH)
    print_category_coverage(results)
    opportunities = build_opportunities(results)

    prices = real_leg_prices(TRADES_PATH)
    if prices is None:
        print(f"NOTE: real per-leg trade prices not available (TARGET_MONTHS={TARGET_MONTHS} trades missing "
              f"or unreadable) -- falling back to the conservative 50-cent fee-sizing assumption for all "
              f"opportunities.\n")
    opportunities = attach_leg_prices(opportunities, prices)

    pnl = compute_pnl(opportunities)

    summarize(pnl, "All flagged same-side violations")
    summarize_by_category(pnl)
    sensitivity_fees_only(pnl)

    out_path = "pairwise_monotonicity_pnl_results.parquet"
    pnl.write_parquet(out_path)
    print(f"Wrote per-opportunity PnL detail to {out_path}")


if __name__ == "__main__":
    main()