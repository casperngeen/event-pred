#!/usr/bin/env python
"""Does Stage 1 survive a survey-consensus surprise?

    venv/bin/python analysis/relations_2026_09/consensus_surprise.py \
        > analysis/relations_2026_09/out/consensus_surprise.txt

The published surprise is market-relative: ``resolved - implied_mean``, the
Kalshi ladder's own forecast error. The event-study literature (Gürkaynak,
Sack & Swanson 2005; Kuttner 2001) uses the survey-consensus surprise,
``actual - consensus``. This script:

  1. matches each surprise-panel event to its economic-calendar row (actual,
     consensus estimate), validating the match by actual vs Kalshi's settled
     value;
  2. compares the two surprises — rank correlation, sign agreement, and which
     forecast was closer to the print;
  3. re-runs the full Stage-1 sweep (same grid, horizon, permutation, BH q) on
     the events that have a consensus, once with each surprise, so the two
     differ only in the number fed in as the trigger's surprise.

Calendar: lum.id findata ``/macro/economic-calendar`` (FMP-sourced), US,
2021-10 → 2025-12, cached at ``data/external/econ_calendar_us_2021q4_2025.parquet``
(fetched 2026-09-25; nothing from 2026 was requested). Series without a
consensus (CPI components, WTI) drop out of the sweep in both arms.

In-sample only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, "stg_infra")

from stg.panel._io import load_markets, scan_trades
from stg.splits import assert_no_oos
from stg.structure import estimate_adjacency

PANELS = Path("artifacts/panels")
CAL = Path("data/external/econ_calendar_us_2021q4_2025.parquet")
OUT = Path("analysis/relations_2026_09/out")

# series -> (calendar event, multiplier from calendar units to Kalshi units)
MAP = {
    "CPI": ("Inflation Rate MoM", 1.0),
    "CPICORE": ("Core Inflation Rate MoM", 1.0),
    "CPIYOY": ("Inflation Rate YoY", 1.0),
    "CPICOREYOY": ("Core Inflation Rate YoY", 1.0),
    "PCECORE": ("Core PCE Price Index MoM", 1.0),
    "PAYROLLS": ("Non Farm Payrolls", 1000.0),
    "ADP": ("ADP Employment Change", 1000.0),
    "U3": ("Unemployment Rate", 1.0),
    "JOBLESSCLAIMS": ("Initial Jobless Claims", 1000.0),
    "GDP": ("GDP Growth Rate QoQ", 1.0),
    "ISMPMI": ("ISM Manufacturing PMI", 1.0),
    "FED": ("Fed Interest Rate Decision", 1.0),
}
MAX_LAG_DAYS = 3


def rule(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def load_calendar() -> pl.DataFrame:
    c = pl.read_parquet(CAL)
    c = c.with_columns(
        pl.col("event").str.replace(r"\s*\(.*\)$", "").str.strip_chars().alias("base"),
        pl.col("date").str.to_datetime(time_zone="UTC").alias("cal_time"),
    ).filter(pl.col("estimate").is_not_null() & pl.col("actual").is_not_null())
    # duplicates (same release listed twice) collapse to one row
    return c.unique(["base", "cal_time", "actual", "estimate"]).sort("cal_time")


def match(sp: pl.DataFrame, cal: pl.DataFrame) -> pl.DataFrame:
    """Each event's calendar release within ±MAX_LAG_DAYS of close_time.

    Where several candidate rows fall in the window (the calendar files U-6
    under "Unemployment Rate" on some dates; revisions are listed twice), keep
    the one whose ``actual`` is nearest Kalshi's settled value. That picks
    which *record* is the release; it never touches the consensus estimate.
    """
    out = []
    for series, (event, mult) in MAP.items():
        s = sp.filter(pl.col("series") == series)
        c = (cal.filter(pl.col("base") == event)
             .select("cal_time", "event", pl.col("actual") * mult, pl.col("estimate") * mult))
        if s.height == 0 or c.height == 0:
            continue
        m = (s.join(c, how="cross")
             .filter((pl.col("close_time") - pl.col("cal_time")).abs()
                     <= pl.duration(days=MAX_LAG_DAYS))
             .sort((pl.col("actual") - pl.col("resolved_value")).abs())
             .unique(["series", "event_ticker"], keep="first", maintain_order=True))
        out.append(m)
    return pl.concat(out, how="diagonal_relaxed").with_columns(
        (pl.col("actual") - pl.col("estimate")).alias("surprise_consensus"))


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    cal = load_calendar()
    m = match(sp, cal)

    rule("1  MATCH QUALITY — calendar actual vs Kalshi settled value")
    print("A match is kept only if the release is within ±3 days of close_time.")
    print("'agree' = |actual − resolved| ≤ 1% of |resolved| or ≤ half a tick; mismatches")
    print("are usually a revised print, an advance/second GDP estimate, or FED's")
    print("range-midpoint vs upper-bound convention. Rows that disagree are DROPPED.\n")
    tol = pl.max_horizontal(pl.col("resolved_value").abs() * 0.01, pl.lit(0.051))
    m = m.with_columns(
        ((pl.col("actual") - pl.col("resolved_value")).abs() <= tol).alias("agree"),
        (pl.col("actual") - pl.col("resolved_value")).alias("act_minus_res"),
    )
    q = (m.group_by("series").agg(
            pl.len().alias("matched"),
            pl.col("agree").sum().alias("agree"),
            pl.col("act_minus_res").median().alias("median actual−resolved"))
         .join(sp.group_by("series").agg(pl.len().alias("panel")), on="series")
         .sort("series"))
    with pl.Config(tbl_rows=30, tbl_cols=10):
        print(q)
    good = m.filter(pl.col("agree"))

    # FED: Kalshi settles on the range midpoint (upper − 0.125); the difference is
    # a constant, so actual − estimate is unaffected. Re-admit it on that basis.
    fed = m.filter((pl.col("series") == "FED")
                   & ((pl.col("act_minus_res") - 0.125).abs() < 1e-6))
    if fed.height:
        print(f"\nFED: {fed.height} rows differ by exactly +0.125 (upper bound vs midpoint)"
              " — re-admitted; the surprise is a difference, so the offset cancels.")
        good = pl.concat([good, fed.with_columns(pl.lit(True).alias("agree"))])

    rule("2  MARKET-RELATIVE vs CONSENSUS SURPRISE (matched, agreeing rows)")
    print("ρ = Spearman; sign agree over rows where both are non-zero; 'cons=0' share")
    print("of consensus surprises that are exactly zero (print = consensus). 'Kalshi")
    print("closer' = |resolved − implied_mean| < |actual − consensus|.\n")
    print(f"{'series':14} {'n':>4} {'ρ':>7} {'sign agree':>11} {'cons=0':>7} {'Kalshi closer':>14} "
          f"{'med |mkt|':>10} {'med |cons|':>11}")
    for series in sorted(good["series"].unique()):
        g = good.filter(pl.col("series") == series)
        a, b = g["surprise"].to_numpy(), g["surprise_consensus"].to_numpy()
        nz = (a != 0) & (b != 0)
        rho = spearmanr(a, b).correlation if len(a) > 3 else np.nan
        closer = np.mean(np.abs(a) < np.abs(b))
        print(f"{series:14} {len(a):>4} {rho:>+7.3f} {np.mean(np.sign(a[nz]) == np.sign(b[nz])):>11.2f} "
              f"{np.mean(b == 0):>7.2f} {closer:>14.2f} {np.median(np.abs(a)):>10.4g} "
              f"{np.median(np.abs(b)):>11.4g}")
    a, b = good["surprise"].to_numpy(), good["surprise_consensus"].to_numpy()
    print(f"\npooled rank corr within series (mean of per-series ρ is above); pooled sign "
          f"agreement {np.mean(np.sign(a[(a != 0) & (b != 0)]) == np.sign(b[(a != 0) & (b != 0)])):.2f} "
          f"over {int(((a != 0) & (b != 0)).sum())} rows")

    rule("3  STAGE-1 SWEEP — same events, market vs consensus surprise")
    base = good.select(sp.columns + ["surprise_consensus"])
    mk = load_markets(is_only=True)
    tr = scan_trades(is_only=True)
    res = {}
    for arm, col in (("market", "surprise"), ("consensus", "surprise_consensus")):
        t0 = time.time()
        panel = base.with_columns(pl.col(col).alias("surprise")).drop("surprise_consensus")
        e = estimate_adjacency(panel, markets=mk, trades=tr)
        res[arm] = e.with_columns(pl.lit(arm).alias("arm"))
        print(f"{arm:10} pairs {e.height:>4}  nominal p<.05 {int((e['p_permutation'] < .05).sum()):>3}"
              f"  (expected {0.05 * e.height:.1f})  BH survivors {int(e['survives'].sum())}"
              f"   [{time.time() - t0:.0f}s]", flush=True)
    both = pl.concat(list(res.values()))
    both.write_parquet(OUT / "consensus_surprise.parquet")

    pub = pl.read_parquet("artifacts/adjacency_is.parquet").filter(pl.col("survives"))
    key = ["trigger", "target", "side"]
    wide = (res["market"].select(*key, pl.col("n").alias("n"), pl.col("rho").alias("ρ mkt"),
                                 pl.col("p_permutation").alias("p mkt"), pl.col("survives").alias("BH mkt"))
            .join(res["consensus"].select(*key, pl.col("rho").alias("ρ cons"),
                                          pl.col("p_permutation").alias("p cons"),
                                          pl.col("survives").alias("BH cons")), on=key, how="full",
                  coalesce=True))
    print("\nThe four published survivors, on the consensus-matched events:")
    with pl.Config(tbl_rows=40, tbl_cols=12, float_precision=3):
        print(pub.select(*key, pl.col("n").alias("n pub"), pl.col("rho").alias("ρ pub"))
              .join(wide, on=key, how="left"))
        print("\nEvery pair that survives BH in either arm:")
        print(wide.filter(pl.col("BH mkt").fill_null(False) | pl.col("BH cons").fill_null(False))
              .sort("p cons"))
        a = wide.drop_nulls(["ρ mkt", "ρ cons"])
        print(f"\nacross all {a.height} pairs: corr(ρ mkt, ρ cons) = "
              f"{np.corrcoef(a['ρ mkt'], a['ρ cons'])[0, 1]:+.3f}; same sign "
              f"{np.mean(np.sign(a['ρ mkt'].to_numpy()) == np.sign(a['ρ cons'].to_numpy())):.2f}")


if __name__ == "__main__":
    main()
