"""
pairwise_monotonicity_pnl_OOT.py

Out-of-sample validation for the weather/climate finding. Everything so
far -- the 48-50% same-side violation rate, the ~4-5.5x gap/spread ratio,
and the PnL backtest + sensitivity sweep -- was computed on one mined
period (October). This script re-runs the same pipeline (violation
detection -> PnL backtest) separately on two ALREADY-SEPARATED trades
files: IN_SAMPLE_TRADES_PATH (October -- re-derived fresh here, so it
should closely match your original numbers as a sanity check) and
HELD_OUT_TRADES_PATH (November -- never looked at before). If the
weather edge is real, both windows should look similar. If it collapses
in the held-out window, the original finding doesn't generalize and the
report claim needs to be walked back.

(Earlier draft of this script assumed a single combined trades file
split by a HOLD_OUT_START date cutoff -- switched to two separate files
since that's how the data is actually organized here, and it sidesteps
having to verify TIMESTAMP_COL's dtype supports a clean `<`/`>=`
datetime comparison.)

Restricted to weather/climate ladder families only, on purpose -- the
sensitivity sweep already showed crypto/other don't survive realistic
execution costs even in-sample, so there's no reason to spend a held-out
check on categories that didn't hold up in the first place.

Reuses your existing modules directly rather than reimplementing their
logic, so any fixes you've made there (schema, fee formula, thresholds)
apply here too:
  - pairwise_monotonicity_taker_side_check.py: check_pair() (the
    same-side violation test) and guess_ladder_pairs_from_ticker_names()
    (the regex-based pair identification actually used to build this
    investigation's results -- see the "Still open" item in the notes
    about cross-checking this against a more rigorous family list; that
    check hasn't been done, so this script inherits the same caveat).
  - pairwise_monotonicity_pnl_backtest.py: classify_ticker(),
    build_opportunities(), attach_leg_prices(), compute_pnl(). This also
    means MIN_N_PER_SIDE and SPREAD_HAIRCUT_FRACTION are inherited from
    whatever you currently have set at the top of that file -- if you
    want a full cost-sensitivity sweep on the held-out window too (not
    just the single haircut level currently configured), wrap this
    script's run_window() calls the same way
    pairwise_monotonicity_pnl_sensitivity.py wraps the main backtest.

REQUIRED SETUP:
  - Set IN_SAMPLE_TRADES_PATH and HELD_OUT_TRADES_PATH below to your
    October and November trades files respectively. Each file is assumed
    to already be scoped to its own window -- this script does no
    date filtering of its own, it just runs the identical pipeline on
    each file and compares.
"""

import polars as pl

try:
    import pairwise_monotonicity_taker_side_check as check_mod
    import pairwise_monotonicity_pnl_backtest as pnl_mod
except ImportError:
    from . import pairwise_monotonicity_taker_side_check as check_mod
    from . import pairwise_monotonicity_pnl_backtest as pnl_mod

IN_SAMPLE_TRADES_PATH = "data/trades/trades_kalshi_even/trades_2025-10.parquet"  # <<< SET THIS -- the window already mined
HELD_OUT_TRADES_PATH = "data/trades/trades_kalshi_odd/trades_2025-11.parquet"  # <<< SET THIS -- the genuinely unseen window

TICKER_COL = check_mod.TICKER_COL
SIDE_COL = check_mod.SIDE_COL
PRICE_COL = check_mod.PRICE_COL


def weather_ladder_pairs(*trades_frames: pl.DataFrame) -> list[tuple[str, str]]:
    """Ladder pairs restricted to weather/climate tickers, built from the UNION of
    tickers across all given trades frames, so a family that only traded in one
    window isn't dropped from that window's comparison."""
    all_tickers: set[str] = set()
    for trades in trades_frames:
        all_tickers.update(trades.select(TICKER_COL).unique().to_series().to_list())
    weather_tickers = [t for t in all_tickers if pnl_mod.classify_ticker(t) == "weather/climate"]
    pairs = check_mod.guess_ladder_pairs_from_ticker_names(weather_tickers)
    return pairs


def avg_leg_prices(trades_window: pl.DataFrame) -> pl.DataFrame:
    """Same shape as pnl_mod.real_leg_prices(), but from an in-memory window
    instead of a file path, since each window needs its own average prices."""
    return trades_window.group_by([TICKER_COL, SIDE_COL]).agg(
        pl.col(PRICE_COL).mean().alias("avg_price")
    )


def run_window(trades_window: pl.DataFrame, pairs: list[tuple[str, str]], label: str) -> dict:
    rows = [check_mod.check_pair(trades_window, a, b) for a, b in pairs]
    results = pl.DataFrame(rows) if rows else pl.DataFrame()

    if results.height == 0:
        print(f"--- {label}: no pairs -- check pairs list / trades window ---\n")
        return {}

    def rate(col):
        sub = results.filter(pl.col(col).is_not_null())
        return sub[col].mean() if sub.height else None

    n_well_sampled = results.filter(
        ~pl.col("same_side_yes_low_n") & ~pl.col("same_side_no_low_n")
    ).height

    prices = avg_leg_prices(trades_window)
    opportunities = pnl_mod.build_opportunities(results)
    opportunities = pnl_mod.attach_leg_prices(opportunities, prices)
    pnl = pnl_mod.compute_pnl(opportunities)

    summary = {
        "label": label,
        "n_pairs": results.height,
        "n_well_sampled": n_well_sampled,
        "same_side_yes_rate": rate("same_side_yes_violation"),
        "same_side_no_rate": rate("same_side_no_violation"),
        "n_opportunities": pnl.height,
        "realistic_total_usd": pnl["realistic_pnl_usd"].sum() if pnl.height else 0.0,
        "realistic_win_rate": (pnl["realistic_pnl_usd"] > 0).mean() if pnl.height else None,
        "realistic_median_usd": pnl["realistic_pnl_usd"].median() if pnl.height else None,
    }

    print(f"--- {label} ---")
    print(f"  pairs checked: {summary['n_pairs']} (well-sampled: {summary['n_well_sampled']})")
    print(f"  same-side violation rate: yes={summary['same_side_yes_rate']}, no={summary['same_side_no_rate']}")
    print(f"  PnL opportunities (well-sampled subset, per current MIN_N_PER_SIDE): {summary['n_opportunities']}")
    print(f"  realistic total PnL: ${summary['realistic_total_usd']:,.2f}")
    print(f"  realistic win rate: {summary['realistic_win_rate']}")
    print(f"  realistic median PnL: ${summary['realistic_median_usd']}")
    print()

    results.write_parquet(f"pairwise_monotonicity_weather_results_{label}.parquet")
    pnl.write_parquet(f"pairwise_monotonicity_weather_pnl_{label}.parquet")
    return summary


def main():
    for path, label in ((IN_SAMPLE_TRADES_PATH, "IN_SAMPLE_TRADES_PATH"), (HELD_OUT_TRADES_PATH, "HELD_OUT_TRADES_PATH")):
        if path in ("trades_october.parquet", "trades_november.parquet"):
            print(f"NOTE: {label} is still the placeholder default ({path!r}) -- "
                  f"set it at the top of this script if that's not actually your real file.")

    in_sample_trades = pl.read_parquet(IN_SAMPLE_TRADES_PATH)
    held_out_trades = pl.read_parquet(HELD_OUT_TRADES_PATH)
    print(f"in-sample (October) trades: {in_sample_trades.height}")
    print(f"held-out (November) trades: {held_out_trades.height}\n")

    lookup = pnl_mod._load_series_category_lookup()
    if lookup:
        print(f"Using {len(lookup)} real Kalshi series->category mappings from "
              f"{pnl_mod.SERIES_CATEGORIES_PATH} to identify weather/climate tickers "
              f"(falls back to the keyword heuristic for any series not in that file).\n")
    else:
        print(f"NOTE: {pnl_mod.SERIES_CATEGORIES_PATH} not found -- identifying weather/climate "
              f"tickers via the keyword heuristic only. Run fetch_kalshi_series_categories.py "
              f"for Kalshi's real per-series categories instead.\n")

    pairs = weather_ladder_pairs(in_sample_trades, held_out_trades)
    print(f"weather/climate ladder pairs identified: {len(pairs)}\n")

    in_sample_summary = run_window(in_sample_trades, pairs, "in_sample")
    held_out_summary = run_window(held_out_trades, pairs, "held_out")

    if in_sample_summary and held_out_summary:
        print("=== Comparison: does the weather edge replicate out of sample? ===")
        for key in ("same_side_yes_rate", "same_side_no_rate", "realistic_total_usd", "realistic_win_rate"):
            print(f"  {key:<22} in_sample={in_sample_summary.get(key)!r:<12} held_out={held_out_summary.get(key)!r}")
        print(
            "\nA held-out result in the same ballpark as in-sample (violation rate still roughly "
            "40-50%, PnL still clearly positive, win rate still well above 50%) supports the weather "
            "edge being real and stable rather than an artifact of the mined period. A held-out result "
            "that collapses toward naive noise levels or near-zero/negative PnL would mean the original "
            "finding doesn't generalize and the report claim should be scaled back accordingly."
        )
    else:
        print("Could not compare -- one or both windows produced no results. Check the trades counts above.")


if __name__ == "__main__":
    main()