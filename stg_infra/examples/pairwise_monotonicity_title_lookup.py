"""
pairwise_monotonicity_title_lookup.py

One-off: pulls the actual title text for a specific pair of tickers from
the markets table, to check whether the T/B-suffix direction-mismatch
hypothesis is real (i.e. whether these two legs are genuinely phrased as
"above X" vs "below Y" -- which would NOT be a valid monotonicity pair --
or whether they're both the same direction after all).

Checks BOTH Oct and Nov 2025 markets files, not a Jan 2026 one -- the
ticker's '26JAN01' is the contract's settlement/target date, not the trade
date. This pair only exists in pairwise_monotonicity_taker_side_results.
parquet because check_pair() found actual trade data for it, and that
trade data can only have come from your Oct/Nov 2025 trades files, so the
listing (and its title) must already be present in Oct or Nov's markets
snapshot.

Edit TICKERS_TO_CHECK below to look up any other disputed pair the same
way.
"""

import polars as pl

pl.Config.set_fmt_str_lengths(200)
pl.Config.set_tbl_width_chars(200)

TICKERS_TO_CHECK = [
    "KXHIGHMIA-25NOV07-B86.5",
    "KXHIGHMIA-25NOV07-T87",
    "KXHIGHMIA-25NOV17-B82.5",
    "KXHIGHMIA-25NOV17-B80.5",
    "KXHIGHMIA-25NOV17-T83",
    "KXHIGHMIA-25NOV11-B67.5",
    "KXHIGHMIA-25NOV24-B82.5",
    "KXHIGHMIA-25NOV23-B83.5",
    "KXHIGHMIA-25NOV13-B77.5",
    "KXHIGHMIA-25NOV18-B81.5",
]

MARKETS_PATHS = [
    "data/markets/markets_kalshi_even/markets_2025-10.parquet",
    "data/markets/markets_kalshi_odd/markets_2025-11.parquet",
]


def main():
    frames = []
    for p in MARKETS_PATHS:
        try:
            frames.append(pl.scan_parquet(p).select(["ticker", "event_ticker", "title"]).collect())
        except Exception as e:
            print(f"could not read {p}: {e}")

    if not frames:
        print("No markets files could be read -- check MARKETS_PATHS matches your actual layout.")
        return

    markets = pl.concat(frames).unique(subset=["ticker"], keep="first")

    result = (
        markets.filter(pl.col("ticker").is_in(TICKERS_TO_CHECK))
        .select("ticker", "event_ticker", "title")
        .sort("ticker")
    )

    print(f"Found {result.height} of {len(TICKERS_TO_CHECK)} requested tickers:\n")
    print(result)

    missing = set(TICKERS_TO_CHECK) - set(result["ticker"].to_list())
    if missing:
        print(f"\nNOT found in either markets file: {missing}")
        print("(try adding more months to MARKETS_PATHS, or double-check the path/parity convention)")

    # The direct test case: same event date, one B ticker and one T ticker,
    # both high-liquidity. If their titles read as opposite directions
    # (e.g. "87 or above" vs "86.5 or below") that confirms the B/T
    # direction-mismatch hypothesis for weather specifically, not just for
    # the Nasdaq example -- which matters more, since weather is the
    # report's flagship category.
    pair_a, pair_b = "KXHIGHMIA-25NOV07-B86.5", "KXHIGHMIA-25NOV07-T87"
    both = result.filter(pl.col("ticker").is_in([pair_a, pair_b]))
    if both.height == 2:
        titles = dict(zip(both["ticker"].to_list(), both["title"].to_list()))
        print(f"\n=== Direct same-date B vs T comparison ===")
        print(f"{pair_a}: {titles[pair_a]!r}")
        print(f"{pair_b}: {titles[pair_b]!r}")
        print(
            "If these read as opposite directions (one 'X or below', the other 'Y or above'), "
            "guess_ladder_pairs_from_ticker_names() would currently group and pair them together "
            "as if they were adjacent same-direction strikes -- confirming the bug for weather, "
            "not just for the Nasdaq example this started from."
        )


if __name__ == "__main__":
    main()