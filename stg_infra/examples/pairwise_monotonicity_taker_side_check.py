"""
pairwise_monotonicity_taker_side_check.py

Extends the original §4.3 ladder monotonicity check with:
  1. A same-taker_side-only version of the violation test (removes bid-ask
     bounce as a possible cause of a "violation").
  2. A per-pair implied-spread-ratio flag, so you can see which violations
     are being tested on two legs with very different spread widths (where
     the same-side cancellation argument is weaker).

ASSUMPTIONS — please check these against your actual schema before trusting
the output, I don't have direct access to your data so these are inferred
from what you've pasted earlier in this conversation:
  - trades columns: "ticker" (str), "taker_side" (str, "yes"/"no"),
    "yes_price" (numeric, cents or dollars — doesn't matter, just be
    consistent), and some timestamp column (default assumed "created_time",
    change TIMESTAMP_COL below if yours is named differently).
  - You already have your own logic (from the original
    pairwise_monotonicity_check.py) that identifies which tickers are
    adjacent-strike pairs within a ladder family. I don't have that file,
    so this script does NOT try to rediscover ladder families from scratch.
    Instead, plug your existing pair list in where `ladder_pairs` is built
    in main() — a best-effort regex-based fallback is included but it is
    NOT a substitute for your validated family logic, only a placeholder
    so this script runs standalone if you want to try it on a subset first.

Important note on direction: earlier empirical checks on this dataset found
taker_side == "no" trades tend to sit HIGHER than taker_side == "yes"
trades for at least the tickers tested (KXMLB-25-LAD, KXHIGHMIA-25NOV07-
B86.5) — the reverse of the naive "yes = ask-side/higher" assumption. This
script deliberately does NOT assume a fixed direction (since that may not
generalize to every family) — same-side checks compare like-for-like taker_
side values only, and the spread estimate uses abs() so it doesn't depend
on knowing which side is bid vs ask.
"""

import re
import polars as pl

TICKER_COL = "ticker"
SIDE_COL = "taker_side"
PRICE_COL = "yes_price"
TIMESTAMP_COL = "created_time"  # change if your trades schema names this differently
MIN_N_PER_SIDE = 20  # below this, flag the pair as low-confidence rather than trusting the same-side average


def side_stats(trades: pl.DataFrame, ticker: str) -> dict:
    """avg price and trade count for each taker_side value, for one ticker."""
    sub = trades.filter(pl.col(TICKER_COL) == ticker)
    grouped = (
        sub.group_by(SIDE_COL)
        .agg(pl.col(PRICE_COL).mean().alias("avg_price"), pl.len().alias("n"))
    )
    out = {"yes": {"avg_price": None, "n": 0}, "no": {"avg_price": None, "n": 0}}
    for row in grouped.iter_rows(named=True):
        side = row[SIDE_COL]
        if side in out:
            out[side] = {"avg_price": row["avg_price"], "n": row["n"]}
    return out


def last_trade_price(trades: pl.DataFrame, ticker: str):
    """Reproduces the original (unfiltered) 'last traded price' used in the naive check."""
    sub = trades.filter(pl.col(TICKER_COL) == ticker)
    if sub.height == 0:
        return None
    sub = sub.sort(TIMESTAMP_COL)
    return sub[PRICE_COL][-1]


def implied_spread(stats: dict):
    """abs() so this doesn't depend on knowing which taker_side is bid vs ask."""
    a, b = stats["yes"]["avg_price"], stats["no"]["avg_price"]
    if a is None or b is None:
        return None
    return abs(a - b)


def check_pair(trades: pl.DataFrame, leg_a: str, leg_b: str) -> dict:
    """
    leg_a = less restrictive leg (lower threshold), leg_b = more restrictive
    leg (higher threshold). A violation means leg_b's price > leg_a's price,
    which shouldn't happen since leg_b entails leg_a.
    """
    stats_a = side_stats(trades, leg_a)
    stats_b = side_stats(trades, leg_b)

    result = {"leg_a": leg_a, "leg_b": leg_b}

    # naive check, same as the original §4.3 approach: last trade price, no side filtering
    last_a = last_trade_price(trades, leg_a)
    last_b = last_trade_price(trades, leg_b)
    result["naive_violation"] = (
        None if last_a is None or last_b is None else last_b > last_a
    )
    result["naive_gap"] = None if last_a is None or last_b is None else last_b - last_a

    # same-side checks: only compare like-for-like taker_side prints
    for side in ("yes", "no"):
        pa, na = stats_a[side]["avg_price"], stats_a[side]["n"]
        pb, nb = stats_b[side]["avg_price"], stats_b[side]["n"]
        key = f"same_side_{side}"
        if pa is None or pb is None:
            result[f"{key}_violation"] = None
            result[f"{key}_gap"] = None
        else:
            result[f"{key}_violation"] = pb > pa
            result[f"{key}_gap"] = pb - pa
        result[f"{key}_n_a"] = na
        result[f"{key}_n_b"] = nb
        result[f"{key}_low_n"] = (na < MIN_N_PER_SIDE) or (nb < MIN_N_PER_SIDE)

    # spread-ratio flag
    spread_a = implied_spread(stats_a)
    spread_b = implied_spread(stats_b)
    result["spread_a"] = spread_a
    result["spread_b"] = spread_b
    if spread_a and spread_b and spread_a > 0 and spread_b > 0:
        result["spread_ratio"] = max(spread_a, spread_b) / min(spread_a, spread_b)
    else:
        result["spread_ratio"] = None

    return result


def guess_ladder_pairs_from_ticker_names(tickers: list[str]) -> list[tuple[str, str]]:
    """
    BEST-EFFORT FALLBACK ONLY. Tries to spot a family/threshold pattern like
    'KXHIGHMIA-25NOV07-B86.5' (family+date prefix, then a trailing numeric
    threshold after a letter such as B/T). Groups by the prefix, sorts by
    threshold, and pairs up adjacent thresholds.

    This is NOT a replacement for your validated family-identification logic
    from the original script — use that instead if you have it. This exists
    only so the script can run end-to-end on a sample if you want to sanity
    check the mechanics first.
    """
    pattern = re.compile(r"^(?P<prefix>.+)-[A-Z](?P<threshold>[0-9]+(\.[0-9]+)?)$")
    families: dict[str, list[tuple[float, str]]] = {}
    for t in tickers:
        m = pattern.match(t)
        if not m:
            continue
        prefix = m.group("prefix")
        threshold = float(m.group("threshold"))
        families.setdefault(prefix, []).append((threshold, t))

    pairs = []
    for prefix, items in families.items():
        items.sort(key=lambda x: x[0])
        for (t1, tick1), (t2, tick2) in zip(items, items[1:]):
            pairs.append((tick1, tick2))  # tick1 = lower threshold = less restrictive
    return pairs


def main():
    trades = pl.read_parquet("data/trades/trades_kalshi_even/trades_2025-10.parquet")  # adjust path

    # --- Plug in your existing ladder-pair identification here ---
    # ladder_pairs = your_existing_pair_list  # list of (leg_a_ticker, leg_b_ticker)
    #
    # Fallback (best-effort, verify before trusting):
    all_tickers = trades.select(TICKER_COL).unique().to_series().to_list()
    ladder_pairs = guess_ladder_pairs_from_ticker_names(all_tickers)

    rows = [check_pair(trades, a, b) for a, b in ladder_pairs]
    results = pl.DataFrame(rows)

    def rate(col):
        sub = results.filter(pl.col(col).is_not_null())
        if sub.height == 0:
            return None
        return sub[col].mean()

    print(f"pairs checked: {results.height}")
    print(f"naive violation rate:            {rate('naive_violation')}")
    print(f"same-side ('yes') violation rate: {rate('same_side_yes_violation')}")
    print(f"same-side ('no') violation rate:  {rate('same_side_no_violation')}")
    print(
        "pairs with low n on at least one side (low confidence): "
        f"{results.filter(pl.col('same_side_yes_low_n') | pl.col('same_side_no_low_n')).height}"
    )
    print(
        "pairs with spread ratio > 3x (weaker cancellation assumption): "
        f"{results.filter(pl.col('spread_ratio') > 3).height}"
    )

    results.write_parquet("pairwise_monotonicity_taker_side_results.parquet")
    print("full results written to pairwise_monotonicity_taker_side_results.parquet")


if __name__ == "__main__":
    main()