"""
baselines.py

Consolidates the two naive rule-based strategies built so far -- the ladder
same-side monotonicity backtest (pairwise_monotonicity_pnl_backtest.py) and
the MECE full-basket sum-to-$1 backtest (mece_sum_to_one_pnl_backtest.py) --
into ONE shared metrics interface and ONE saved results file. Purpose: keep
these numbers in view (KIV) as the fixed comparison point for whatever the
STGAT eventually produces, computed the same way, so "beats baseline" is a
literal column-vs-column comparison rather than something reconstructed
from memory or old console output later.

Each mechanism already has three cost scenarios (idealized / fees-only /
realistic) sitting in its own PnL results parquet from the last runs. This
script does NOT recompute anything from raw trades -- it just standardizes
metrics across both mechanisms' *already-computed* per-opportunity PnL
tables, so it stays cheap to rerun and never drifts from what the actual
backtests produced.

INPUTS (must already exist -- run these first if they don't):
  pairwise_monotonicity_pnl_results.parquet   <- pairwise_monotonicity_pnl_backtest.py
  mece_sum_to_one_pnl_results.parquet         <- mece_sum_to_one_pnl_backtest.py

METRICS COMPUTED PER (mechanism, scenario):
  n_opportunities, universe_size, coverage_pct  -- coverage = how much of the
      total candidate pool this strategy actually trades. Deliberately kept
      visible per-row: naive full-basket/threshold rules have near-100%
      coverage of their OWN candidate pool by construction, but that pool is
      itself a small fraction of all N-way clusters found upstream (e.g. MECE:
      618 full-basket snapshots out of 2,354 candidate events; 14+ leg
      families contribute ZERO of them). A model that can price partially-
      covered baskets could raise coverage well above what any naive full-
      basket rule can reach -- that's tracked here as the second axis of
      "beating baseline," not just PnL.
  total_pnl_usd, mean_pnl_usd, median_pnl_usd, win_rate
  cross_sectional_sharpe  -- mean/std of per-opportunity PnL. NOT an
      annualized financial Sharpe ratio (these are one-off arbitrage
      captures, not a return series on continuously-held capital) -- treat
      it as a comparable risk-adjusted score across strategies, nothing more.
  max_drawdown_usd  -- only computed where a chronological ordering exists
      (MECE's per-event-day snapshots have a real trade date; the ladder's
      per-pair opportunities don't carry one -- they're a cross-sectional
      summary per leg-pair, not a dated sequence of trades -- so this is
      null there. Documented rather than faked.)
  total_fees_usd

UNIVERSE_SIZE CONSTANTS: these are the upstream candidate-pool sizes
reported by the identification scripts in this investigation (see
kalshi-arbitrage-findings-summary.md in the project for full provenance).
They are NOT recomputed here -- if you rerun the upstream scripts with
different settings (MIN_N_PER_SIDE, MAX_LEGS, VIOLATION_THRESHOLD, etc.)
and get different candidate-pool sizes, update these to match before
trusting coverage_pct.
"""

import os

import polars as pl

LADDER_PNL_PATH = "pairwise_monotonicity_pnl_results.parquet"
MECE_PNL_PATH = "mece_sum_to_one_pnl_results.parquet"

OUT_PATH = "baseline_results_summary.parquet"

# --- universe sizes, for the coverage_pct column -----------------------
# Ladder: 464,709 valid pairs identified by
#   pairwise_monotonicity_taker_side_check_corrected.py (100% correctly
#   classified pair types, crypto+financials). The PnL backtest further
#   restricts to MIN_N_PER_SIDE=20 same-side-violation pairs before
#   computing PnL -- that restricted count IS n_opportunities below, so
#   coverage_pct here answers "what fraction of ALL valid pairs does even
#   the well-sampled violation subset represent."
LADDER_UNIVERSE_SIZE = 464_709

# MECE: 2,354 fully-resolved, single-winner, non-ladder, non-combo-prop
# candidate events found by mece_sum_to_one_check.py (MAX_LEGS=100 run).
# Of those, only 618 (event, day) pairs ever achieve full-basket coverage
# at all -- every 14+ leg family contributes zero. n_opportunities below
# (267) is the subset of those 618 that also clears VIOLATION_THRESHOLD.
MECE_UNIVERSE_SIZE = 2_354


def _safe_col(df: pl.DataFrame, name: str):
    return df[name] if name in df.columns else None


def cross_sectional_sharpe(pnl_series: pl.Series) -> float | None:
    if pnl_series is None or pnl_series.len() < 2:
        return None
    std = pnl_series.std()
    if std is None or std == 0:
        return None
    return pnl_series.mean() / std


def max_drawdown_from_dates(df: pl.DataFrame, date_col: str, pnl_col: str) -> float | None:
    """Sorts opportunities chronologically, builds a cumulative PnL curve,
    and returns the largest peak-to-trough drop. Only meaningful when a
    real trade date exists per opportunity (MECE) -- see module docstring."""
    if date_col not in df.columns or df.height == 0:
        return None
    ordered = df.sort(date_col).with_columns(pl.col(pnl_col).cum_sum().alias("_cum_pnl"))
    cum = ordered["_cum_pnl"].to_list()
    running_max = float("-inf")
    max_dd = 0.0
    for v in cum:
        running_max = max(running_max, v)
        max_dd = min(max_dd, v - running_max)
    return max_dd


def metrics_row(mechanism: str, scenario: str, df: pl.DataFrame, pnl_col: str,
                 universe_size: int, fees_col: str | None = None,
                 date_col: str | None = None) -> dict:
    n = df.height
    pnl = _safe_col(df, pnl_col)
    win_rate = (pnl > 0).mean() if pnl is not None and n > 0 else None
    total_pnl = pnl.sum() if pnl is not None else None
    mean_pnl = pnl.mean() if pnl is not None and n > 0 else None
    median_pnl = pnl.median() if pnl is not None and n > 0 else None
    sharpe = cross_sectional_sharpe(pnl) if pnl is not None else None
    total_fees = df[fees_col].sum() if fees_col and fees_col in df.columns else None
    max_dd = max_drawdown_from_dates(df, date_col, pnl_col) if date_col else None

    return {
        "mechanism": mechanism,
        "scenario": scenario,
        "n_opportunities": n,
        "universe_size": universe_size,
        "coverage_pct": (n / universe_size * 100.0) if universe_size else None,
        "total_pnl_usd": total_pnl,
        "mean_pnl_usd": mean_pnl,
        "median_pnl_usd": median_pnl,
        "win_rate": win_rate,
        "cross_sectional_sharpe": sharpe,
        "max_drawdown_usd": max_dd,
        "total_fees_usd": total_fees,
    }


def build_ladder_rows(ladder: pl.DataFrame) -> list[dict]:
    rows = []
    # idealized: zero-friction PnL, no fee/haircut column needed
    rows.append(metrics_row("ladder", "idealized", ladder, "idealized_pnl_usd", LADDER_UNIVERSE_SIZE))
    # fees-only: idealized_pnl_usd - fees_usd, matching
    # pairwise_monotonicity_pnl_backtest.sensitivity_fees_only()'s definition
    if "idealized_pnl_usd" in ladder.columns and "fees_usd" in ladder.columns:
        fees_only_df = ladder.with_columns(
            (pl.col("idealized_pnl_usd") - pl.col("fees_usd")).alias("_fees_only_pnl_usd")
        )
        rows.append(metrics_row("ladder", "fees_only", fees_only_df, "_fees_only_pnl_usd",
                                 LADDER_UNIVERSE_SIZE, fees_col="fees_usd"))
    # realistic: fees + spread haircut, already computed
    rows.append(metrics_row("ladder", "realistic", ladder, "realistic_pnl_usd",
                             LADDER_UNIVERSE_SIZE, fees_col="fees_usd"))
    return rows


def build_mece_rows(mece: pl.DataFrame) -> list[dict]:
    rows = []
    rows.append(metrics_row("mece", "idealized", mece, "idealized_pnl_usd", MECE_UNIVERSE_SIZE,
                             date_col="date"))
    if "fees_only_pnl_usd" in mece.columns:
        rows.append(metrics_row("mece", "fees_only", mece, "fees_only_pnl_usd", MECE_UNIVERSE_SIZE,
                                 fees_col="fees_usd", date_col="date"))
    rows.append(metrics_row("mece", "realistic", mece, "realistic_pnl_usd", MECE_UNIVERSE_SIZE,
                             fees_col="fees_usd", date_col="date"))
    return rows


def print_table(summary: pl.DataFrame):
    print(f"{'mechanism':<10} {'scenario':<10} {'n':>6} {'universe':>9} {'cov%':>7} "
          f"{'total_pnl':>12} {'win_rate':>9} {'sharpe':>8} {'max_dd':>10}")
    for r in summary.iter_rows(named=True):
        cov = f"{r['coverage_pct']:.2f}" if r["coverage_pct"] is not None else "n/a"
        total = f"${r['total_pnl_usd']:,.2f}" if r["total_pnl_usd"] is not None else "n/a"
        win = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "n/a"
        sharpe = f"{r['cross_sectional_sharpe']:.3f}" if r["cross_sectional_sharpe"] is not None else "n/a"
        dd = f"${r['max_drawdown_usd']:,.2f}" if r["max_drawdown_usd"] is not None else "n/a"
        print(f"{r['mechanism']:<10} {r['scenario']:<10} {r['n_opportunities']:>6} "
              f"{r['universe_size']:>9} {cov:>7} {total:>12} {win:>9} {sharpe:>8} {dd:>10}")


def main():
    missing = [p for p in (LADDER_PNL_PATH, MECE_PNL_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing input(s): {missing}. Run pairwise_monotonicity_pnl_backtest.py and "
            f"mece_sum_to_one_pnl_backtest.py first -- this script only standardizes their "
            f"already-computed outputs, it doesn't rebuild them."
        )

    ladder = pl.read_parquet(LADDER_PNL_PATH)
    mece = pl.read_parquet(MECE_PNL_PATH)

    rows = build_ladder_rows(ladder) + build_mece_rows(mece)
    summary = pl.DataFrame(rows)

    print("=== BASELINE STRATEGY RESULTS (KIV -- fixed comparison point for the STGAT) ===\n")
    print_table(summary)
    print()

    summary.write_parquet(OUT_PATH)
    print(f"Wrote {summary.height}-row baseline summary to {OUT_PATH}.")
    print("Keep this file -- whatever the STGAT produces should be scored with the SAME metric "
          "definitions (this module's metrics_row()) and compared row-for-row against the "
          "'realistic' scenario here, which is the actual bar to beat.")


if __name__ == "__main__":
    main()