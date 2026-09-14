#!/usr/bin/env python
"""Hold to settlement: does the calibration bias pay after costs?

    venv/bin/python analysis/relations_2026_09/settlement_trade.py

Every strategy priced in this project so far has a payoff equal to a *price
change*, and price changes here are ~1c against a 1.2-5.3c round trip. Granting
perfect foresight over direction and strike, the dormant trade still nets
+0.09c and the pre-resolution jump +0.37c (``strike_selection.py``). Any
price-change payoff is dead on arrival.

Holding to settlement changes the payoff object: 0 or 100c, with **no exit fee
and no exit spread**, because settlement is free. Buying at 30c costs ~3.5c all
in, so break-even needs P(win) > 33.5% -- a 3.5 percentage-point edge, not a 5c
move in 17 minutes.

Item 1 found an edge of roughly that shape and far that size: realised values
land above the market's own implied centre ~70% of the time for the CPI family.
This prices that directly, on the **trigger's own ladder**, using real traded
strike prices rather than the approximation in the write-up.

Design
------
* Snapshot = the same last-pre-resolution day the surprise panel uses, so the
  entry price is one a trader could actually have seen.
* Entry = that day's last trade on the chosen strike. Costs = Kalshi's
  ``0.07*p(1-p)`` **once** plus a spread charge, both reported at half and full
  the measured round-trip spread since only one crossing is required.
* Settlement = the true printed value (``expiration_value`` since §14.1).
* The rule is fixed in advance and fits nothing: **buy YES at the strike nearest
  the implied median**, every event, every series.
* WTI is the control. §1's calibration study found it *under*confident and
  centred, so the rule must not pay there.
* Year-clustered bootstrap, because the obvious threat is that this is one
  inflation regime rather than a structural bias.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import (
    normalise_to_exclusive, parse_threshold, parse_threshold_from_subtitle,
)
from stg.io.kalshi import KalshiOHLCV
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, series_filter_expr, trigger_universe
from stg.panel.surprise import MIN_FRESH_LEGS
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/relations_2026_09/out")
FEE_RATE, CONTRACTS = 0.07, 100


def fee_cents(price_cents) -> np.ndarray:
    p = np.asarray(price_cents, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(price_cents) -> np.ndarray:
    """Measured round-trip effective spread by moneyness (research_log §11.2)."""
    out = np.full(np.shape(price_cents), 2.0)
    out[np.abs(np.asarray(price_cents, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def strike_panel(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame) -> pl.DataFrame:
    """One row per (event, strike) on the snapshot day: price and settlement.

    Mirrors ``surprise.py::_threshold_surprise``'s snapshot selection exactly --
    same freshness rule, same day -- so the entry price is the one the surprise
    measure was computed against.
    """
    sub_mk = mk.filter(series_filter_expr(canon))
    tickers = sub_mk["ticker"].unique().to_list()
    trades = tr.filter(pl.col("ticker").is_in(tickers)).collect()
    if trades.is_empty():
        return pl.DataFrame()
    daily = KalshiOHLCV.build_daily(trades, sub_mk).join(
        sub_mk.select("ticker", "yes_sub_title").unique(subset=["ticker"]),
        on="ticker", how="left")
    daily = daily.with_columns(
        pl.struct("ticker", "yes_sub_title").map_elements(
            lambda r: parse_threshold(r["ticker"], r["yes_sub_title"]),
            return_dtype=pl.Float64).alias("threshold"),
        pl.col("yes_sub_title").map_elements(
            lambda s: parse_threshold_from_subtitle(s)[1],
            return_dtype=pl.Utf8).alias("conv"),
    ).filter(pl.col("threshold").is_not_null())
    fresh = daily.filter(pl.col("trade_count") > 0)

    rows = []
    for ev in fresh["event_ticker"].unique().to_list():
        ev_fresh = fresh.filter(pl.col("event_ticker") == ev)
        cand = (ev_fresh.group_by("date").agg(pl.len().alias("legs"))
                .filter(pl.col("legs") >= MIN_FRESH_LEGS).sort("date"))
        if cand.height == 0:
            continue
        day = cand["date"][-1]
        lad = ev_fresh.filter(pl.col("date") == day).sort("threshold")
        thr = lad["threshold"].to_numpy().astype(float)
        conv = lad["conv"].to_list()
        if any(c == "inclusive" for c in conv):
            thr = normalise_to_exclusive(thr, conv)
        px = lad["close"].to_numpy().astype(float)
        close = ev_fresh["close_time"].min()
        for k, p in zip(thr, px):
            rows.append(dict(series=canon, event_ticker=ev, close_time=close,
                             snap_date=day, strike=float(k), price=float(p)))
    return pl.DataFrame(rows)


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)

    trigs = [c for c in trigger_universe(5, mk) if SPECS[c].kind == "threshold"]
    parts = [d for c in trigs if (d := strike_panel(c, mk, tr)).height]
    legs = pl.concat(parts)
    # keep only events the surprise panel kept, and attach the true outcome
    legs = legs.join(sp.select("event_ticker", "resolved_value", "implied_median",
                               "implied_mean", "pit"),
                     on="event_ticker", how="inner")
    legs = legs.filter(pl.col("price").is_between(1, 99)).with_columns(
        (pl.col("resolved_value") > pl.col("strike")).cast(pl.Int8).alias("win"),
        (pl.col("strike") - pl.col("implied_median")).abs().alias("d_median"))
    assert_no_oos(legs, time_col="close_time")
    print(f"(event, strike) rows: {legs.height}   events: {legs['event_ticker'].n_unique()}")

    price = legs["price"].to_numpy()
    gross = 100.0 * legs["win"].to_numpy() - price
    cost_half = fee_cents(price) + spread_cents(price) / 2.0
    cost_full = fee_cents(price) + spread_cents(price)
    legs = legs.with_columns(pl.Series("gross", gross),
                             pl.Series("net", gross - cost_half),
                             pl.Series("net_wide", gross - cost_full),
                             pl.Series("cost", cost_half))

    # the fixed rule: one strike per event, nearest the implied median
    pick = (legs.sort("d_median", "strike")
            .group_by("event_ticker", maintain_order=True).first())

    print("\n=== BUY YES at the strike nearest the implied median, hold to settlement ===")
    print("cents per contract; payoff is 0 or 100, entry crossed once\n")
    rows = []
    for label, d in [("ALL series", pick),
                     ("CPI family", pick.filter(pl.col("series").is_in(
                         ["CPI", "CPICORE", "CPIYOY", "CPICOREYOY"]))),
                     ("CPI only", pick.filter(pl.col("series") == "CPI")),
                     ("PAYROLLS", pick.filter(pl.col("series") == "PAYROLLS")),
                     ("U3", pick.filter(pl.col("series") == "U3")),
                     ("WTI (control)", pick.filter(pl.col("series") == "WTI"))]:
        if d.height < 8:
            continue
        n = d["net"].to_numpy()
        rows.append(dict(subset=label, n=d.height,
                         mean_price=float(d["price"].mean()),
                         hit=float(d["win"].mean()),
                         mean_gross=float(d["gross"].mean()),
                         mean_net=float(n.mean()),
                         median_net=float(np.median(n)),
                         net_wide=float(d["net_wide"].mean()),
                         t=float(n.mean() / (n.std(ddof=1) / np.sqrt(len(n))))))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=220):
        print(pl.DataFrame(rows))

    fam = pick.filter(pl.col("series").is_in(["CPI", "CPICORE", "CPIYOY", "CPICOREYOY"]))
    print("\n=== CPI family by year — is it one regime? ===")
    with pl.Config(tbl_rows=10, float_precision=2):
        print(fam.with_columns(pl.col("close_time").dt.year().alias("yr"))
              .group_by("yr").agg(pl.len().alias("n"),
                                  pl.col("price").mean().alias("price"),
                                  pl.col("win").mean().alias("hit"),
                                  pl.col("net").mean().alias("net")).sort("yr"))

    print("=== year-clustered bootstrap (resample whole years) ===")
    f = fam.with_columns(pl.col("close_time").dt.year().alias("yr"))
    yrs = sorted(f["yr"].unique().to_list())
    by = {y: f.filter(pl.col("yr") == y)["net"].to_numpy() for y in yrs}
    rng = np.random.default_rng(0)
    boot = np.array([np.concatenate([by[y] for y in rng.choice(yrs, len(yrs), replace=True)]).mean()
                     for _ in range(10000)])
    obs = f["net"].mean()
    print(f"observed {obs:+.2f}c   95% CI [{np.percentile(boot,2.5):+.2f}, "
          f"{np.percentile(boot,97.5):+.2f}]   P(mean<=0) = {(boot<=0).mean():.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    legs.write_parquet(OUT / "settlement_legs.parquet")


if __name__ == "__main__":
    main()
