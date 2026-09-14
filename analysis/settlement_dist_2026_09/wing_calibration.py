#!/usr/bin/env python
"""Item 1, the gate: do wing legs settle at their price?

    venv/bin/python analysis/settlement_dist_2026_09/wing_calibration.py

``settlement_distribution_plan.md`` proposes moving the edge from ``mu`` to
``sigma``, on the argument that a digital has no vega at the money, so a better
uncertainty estimate pays near |z| = 1 -- prices near 16c and 84c -- where the
fee is roughly half what it is at 50c and where longshot bias is supposed to
leave the wings overpriced.

That whole programme is unfunded if the wings are already fair. This asks the
question directly, and it is deliberately the cheapest possible version:

* **No reconstruction anywhere.** ``recover_pdf`` is not called, no implied
  mean, no implied std, no PIT. Item 1's calibration result is currently read
  as a finding about the reconstruction rather than about the market
  (``relations_findings.md``, "Superseded"), so any wing claim inferred from
  the recovered pdf inherits that doubt. Here the entry is a real traded price
  and the outcome is the market's own ``result`` field.
* **Zero parameters.** The bands are fixed in advance and nothing is fitted.
* **Both contract kinds.** Threshold ladders (CPI, U3, PAYROLLS, ...) and
  bucket ladders (WTI). WTI matters because it is the series whose PIT
  signature looked like textbook longshot bias, and it was absent from
  ``settlement_trade.py`` entirely -- that script kept ``kind == threshold``.

Design
------
* Snapshot = the last pre-resolution day carrying ``MIN_FRESH_LEGS`` legs that
  actually traded, the same rule the surprise panel uses, so the entry price is
  one a trader could have seen. Entry = that day's last trade on the leg.
* Outcome = ``result == "yes"`` from the markets table. Ground truth, not a
  reconstruction.
* Fold every leg to its **longshot side**: a leg at 88c YES is a leg at 12c NO.
  ``p_long = min(p, 100-p)``, and ``long_win`` is whether that cheap side paid.
  Longshot bias predicts ``long_win < p_long/100``.
* The pre-specified trade is therefore **sell the longshot** (equivalently, buy
  the favourite at ``100 - p_long``) and hold. One crossing, no exit fee.
* Bootstrap clusters on ``event_ticker``. Within one event the ladder legs are
  nearly perfectly dependent -- one print settles all of them -- so the leg
  count is not the sample size and an unclustered t on 494 legs would be
  fiction.
* Year split, because ``settlement_trade.py``'s CPI-family hit rate ran
  0.58 / 0.50 / 0.69 / 0.26 over 2022-25 and did not survive year-clustering.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import (
    BUCKET, THRESHOLD, classify_contract, parse_bucket, parse_threshold,
)
from stg.io.kalshi import KalshiOHLCV
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, series_filter_expr, trigger_universe
from stg.panel.surprise import MIN_FRESH_LEGS
from stg.splits import assert_no_oos

OUT = Path("analysis/settlement_dist_2026_09/out")
FEE_RATE, CONTRACTS = 0.07, 100
N_BOOT = 10000


def fee_cents(price_cents) -> np.ndarray:
    """Kalshi's 0.07*p(1-p), charged once -- settlement is not a trade."""
    p = np.asarray(price_cents, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(price_cents) -> np.ndarray:
    """Measured round-trip effective spread by moneyness (research_log §11.2)."""
    out = np.full(np.shape(price_cents), 2.0)
    out[np.abs(np.asarray(price_cents, dtype=float) - 50.0) > 40.0] = 1.0
    return out


# --------------------------------------------------------------------------
# leg construction -- one row per (event, leg) on the snapshot day
# --------------------------------------------------------------------------
def legs_for(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame) -> pl.DataFrame:
    """Price and settlement for every leg that traded on the snapshot day.

    Kind-agnostic: a threshold leg and a bucket leg are both just "a contract
    with a price and a yes/no result", which is all this test needs. The
    ladder geometry only enters through ``kind``, reported so the wings can be
    split by contract type.
    """
    sub = mk.filter(series_filter_expr(canon)).select(
        "ticker", "event_ticker", "yes_sub_title", "result", "close_time")
    if sub.is_empty():
        return pl.DataFrame()

    kinds, strikes = [], []
    for t, s in zip(sub["ticker"], sub["yes_sub_title"]):
        c = classify_contract(t, s)
        kinds.append(c)
        if c == THRESHOLD:
            strikes.append(parse_threshold(t, s))
        elif c == BUCKET:
            b = parse_bucket(t, s)
            strikes.append((b[0] + b[1]) / 2.0 if b else None)
        else:
            strikes.append(None)
    sub = sub.with_columns(pl.Series("kind", kinds),
                           pl.Series("strike", strikes, dtype=pl.Float64))
    sub = sub.filter(pl.col("kind").is_in([THRESHOLD, BUCKET]),
                     pl.col("result").is_in(["yes", "no"]))
    if sub.is_empty():
        return pl.DataFrame()

    tickers = sub["ticker"].unique().to_list()
    trades = tr.filter(pl.col("ticker").is_in(tickers)).collect()
    if trades.is_empty():
        return pl.DataFrame()
    daily = KalshiOHLCV.build_daily(trades, mk.filter(series_filter_expr(canon)))
    daily = daily.join(sub.select("ticker", "kind", "strike", "result"),
                       on="ticker", how="inner").filter(pl.col("trade_count") > 0)
    if daily.is_empty():
        return pl.DataFrame()

    rows = []
    for ev in daily["event_ticker"].unique().to_list():
        ev_fresh = daily.filter(pl.col("event_ticker") == ev)
        cand = (ev_fresh.group_by("date").agg(pl.len().alias("legs"))
                .filter(pl.col("legs") >= MIN_FRESH_LEGS).sort("date"))
        if cand.height == 0:
            continue
        day = cand["date"][-1]
        lad = ev_fresh.filter(pl.col("date") == day)
        close = ev_fresh["close_time"].min()
        for r in lad.iter_rows(named=True):
            rows.append(dict(series=canon, event_ticker=ev, ticker=r["ticker"],
                             kind=r["kind"], close_time=close, snap_date=day,
                             strike=r["strike"], price=float(r["close"]),
                             win=int(r["result"] == "yes")))
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# clustered inference
# --------------------------------------------------------------------------
def cluster_boot(df: pl.DataFrame, col: str, by: str = "event_ticker",
                 seed: int = 0) -> tuple[float, float, float, float]:
    """Resample whole clusters. Returns (observed, lo, hi, P(mean<=0))."""
    groups = [g[col].to_numpy() for _, g in df.group_by(by)]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(groups))
    boot = np.array([
        np.concatenate([groups[i] for i in rng.choice(idx, len(idx), replace=True)]).mean()
        for _ in range(N_BOOT)])
    obs = float(np.concatenate(groups).mean())
    return obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), float((boot <= 0).mean())


def band_table(d: pl.DataFrame, bands: list[tuple[float, float]], label: str) -> pl.DataFrame:
    rows = []
    for lo, hi in bands:
        b = d.filter(pl.col("p_long").is_between(lo, hi, closed="left"))
        if b.height < 10:
            continue
        # edge in percentage points: what you paid vs what it was worth
        e = b.with_columns((100.0 * pl.col("long_win") - pl.col("p_long")).alias("e"))
        obs, clo, chi, pneg = cluster_boot(e, "e")
        rows.append(dict(band=f"{lo:.0f}-{hi:.0f}c", n_legs=b.height,
                         n_events=b["event_ticker"].n_unique(),
                         mean_price=float(b["p_long"].mean()),
                         realised=float(b["long_win"].mean()) * 100.0,
                         edge_pp=obs, ci_lo=clo, ci_hi=chi, p_ge0=1.0 - pneg))
    out = pl.DataFrame(rows)
    print(f"\n=== {label} ===")
    print("longshot side: paid `mean_price`, worth `realised`. "
          "edge_pp < 0 means the longshot was overpriced (longshot bias).")
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=200):
        print(out)
    return out


def main() -> None:
    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
    canons = [c for c in trigger_universe(5, mk) if SPECS[c].kind != "categorical"]
    print(f"series: {len(canons)} -> {sorted(canons)}\n")

    parts = [d for c in canons if (d := legs_for(c, mk, tr)).height]
    legs = pl.concat(parts)
    assert_no_oos(legs, time_col="close_time")

    legs = legs.filter(pl.col("price").is_between(1, 99)).with_columns(
        pl.min_horizontal("price", 100.0 - pl.col("price")).alias("p_long"),
    ).with_columns(
        pl.when(pl.col("price") <= 50.0).then(pl.col("win"))
          .otherwise(1 - pl.col("win")).alias("long_win"),
        pl.col("close_time").dt.year().alias("yr"),
    )
    print(f"legs: {legs.height}   events: {legs['event_ticker'].n_unique()}   "
          f"series: {legs['series'].n_unique()}")
    with pl.Config(tbl_rows=30, float_precision=3):
        print(legs.group_by("series").agg(
            pl.len().alias("legs"), pl.col("event_ticker").n_unique().alias("events"),
            pl.col("kind").first(), pl.col("p_long").mean().alias("mean_p_long"),
        ).sort("legs", descending=True))

    BANDS = [(3, 10), (10, 20), (20, 35), (35, 50)]
    band_table(legs, BANDS, "ALL series")
    band_table(legs.filter(pl.col("kind") == THRESHOLD), BANDS, "threshold ladders (macro releases)")
    if legs.filter(pl.col("kind") == BUCKET).height:
        band_table(legs.filter(pl.col("kind") == BUCKET), BANDS, "bucket ladders (WTI -- the longshot-bias prior)")

    # ---------------------------------------------------------------- trade
    # Pre-specified: sell the longshot / buy the favourite, hold to settlement.
    wing = legs.filter(pl.col("p_long").is_between(3, 20, closed="left"))
    fav_price = 100.0 - wing["p_long"].to_numpy()
    fav_win = 1 - wing["long_win"].to_numpy()
    gross = 100.0 * fav_win - fav_price
    cost = fee_cents(fav_price) + spread_cents(fav_price) / 2.0
    wing = wing.with_columns(pl.Series("gross", gross),
                             pl.Series("net", gross - cost),
                             pl.Series("cost", cost))

    print("\n=== the trade: SELL the longshot at 3-20c, hold to settlement ===")
    print("cents per contract, entry crossed once, no exit fee\n")
    rows = []
    for label, d in [("ALL", wing),
                     ("threshold", wing.filter(pl.col("kind") == THRESHOLD)),
                     ("bucket (WTI)", wing.filter(pl.col("kind") == BUCKET)),
                     ("CPI family", wing.filter(pl.col("series").is_in(
                         ["CPI", "CPICORE", "CPIYOY", "CPICOREYOY"])))]:
        if d.height < 20:
            continue
        obs, lo, hi, pneg = cluster_boot(d, "net")
        rows.append(dict(subset=label, n_legs=d.height,
                         n_events=d["event_ticker"].n_unique(),
                         mean_cost=float(d["cost"].mean()),
                         gross=float(d["gross"].mean()), net=obs,
                         ci_lo=lo, ci_hi=hi, p_le0=pneg))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    print("\n=== by year — is it one regime? ===")
    with pl.Config(tbl_rows=20, float_precision=2):
        print(wing.group_by("yr").agg(
            pl.len().alias("n_legs"), pl.col("event_ticker").n_unique().alias("events"),
            pl.col("p_long").mean().alias("price"),
            (100.0 * pl.col("long_win").mean()).alias("realised"),
            pl.col("gross").mean().alias("gross"), pl.col("net").mean().alias("net"),
        ).sort("yr"))

    print("\n=== year-clustered (resample whole years) ===")
    obs, lo, hi, pneg = cluster_boot(wing, "net", by="yr", seed=1)
    print(f"observed {obs:+.2f}c   95% CI [{lo:+.2f}, {hi:+.2f}]   P(mean<=0) = {pneg:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    legs.write_parquet(OUT / "wing_legs.parquet")
    print(f"\nwrote {OUT / 'wing_legs.parquet'}")


if __name__ == "__main__":
    main()
