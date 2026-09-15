#!/usr/bin/env python
"""Direction 1, the strong form: strike-matched arbitrage between MoM and YoY.

    venv/bin/python analysis/arbitrage_2026_09/identity_arb.py

``identity_cpi.py`` compared the two ladders through ``recover_pdf`` and found
the YoY ladder implying ~21% more width than the MoM ladder for the same
unknown. That could be a reconstruction artifact -- different strike spacing,
different open-tail mass. This test removes the reconstruction entirely.

The mapping
-----------
``YoY_t = c_t + MoM_t`` with ``c_t = YoY_{t-1} - MoM_{t-12}``, both **already
published** when the two ladders trade. So the contract

    "YoY above K"      and      "MoM above K - c_t"

are the *same event*. Two tickers, two order books, one payoff. Their prices
must be equal, and any difference is money: buy the cheap one, sell the dear
one, hold both to settlement, and the pair pays 100c whichever way the print
lands.

Recovering the published prints
-------------------------------
``resolved_value`` stores the ladder midpoint, so a print of 6.5 is stored as
6.45. Adding 0.05 recovers it exactly -- verified against the real BLS prints
for Dec-2022 (6.5 / -0.1), Jun-2023 (3.0 / 0.2) and Dec-2023 core (3.9). That
matters because ``c_t`` has to land on the 0.1 grid for strikes to match.

Basis risk, which is the whole question
---------------------------------------
BLS rounds MoM and YoY to 0.1pp *independently*, and the identity carries a
dropped second-order term. So a matched pair can settle differently, and the
payoff is brutally asymmetric::

    both legs agree      ->  +(p_dear - p_cheap)
    cheap YES, dear NO   ->  +100 + (p_dear - p_cheap)
    cheap NO,  dear YES  ->  -100 + (p_dear - p_cheap)

A 1-in-20 break at -100c wipes out a lot of 2c edges. §3 measures the break
rate on realised settlements rather than assuming it.

In-sample only.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import THRESHOLD, classify_contract, parse_threshold
from stg.io.kalshi import KalshiOHLCV
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import series_filter_expr
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/arbitrage_2026_09/out")

PAIRS = [("CPI", "CPIYOY"), ("CPICORE", "CPICOREYOY")]
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
FEE_RATE, CONTRACTS = 0.07, 100
GRID = 0.1                      # BLS reporting granularity, pp


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def parse_month(ticker: str):
    tail = ticker.split("-", 1)[1] if "-" in ticker else ""
    if len(tail) < 5:
        return None
    yy, mmm = tail[:2], tail[2:5].upper()
    if not yy.isdigit() or mmm not in MONTHS:
        return None
    return dt.date(2000 + int(yy), MONTHS[mmm], 1)


def shift_months(d: dt.date, k: int) -> dt.date:
    m = d.month - 1 + k
    return dt.date(d.year + m // 12, m % 12 + 1, 1)


def daily_legs(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame) -> pl.DataFrame:
    """event_ticker, date, ticker, strike, close(last trade), result."""
    sub = (mk.filter(series_filter_expr(canon) & pl.col("close_time").is_not_null())
           .select("ticker", "event_ticker", "yes_sub_title", "result", "close_time")
           .unique(subset=["ticker"]))
    if sub.is_empty():
        return pl.DataFrame()
    kinds, strikes = [], []
    for t, s in zip(sub["ticker"], sub["yes_sub_title"]):
        c = classify_contract(t, s)
        kinds.append(c)
        strikes.append(parse_threshold(t, s) if c == THRESHOLD else None)
    sub = (sub.with_columns(pl.Series("kind", kinds),
                            pl.Series("strike", strikes, dtype=pl.Float64))
           .filter((pl.col("kind") == THRESHOLD) & pl.col("strike").is_not_null()
                   & pl.col("result").is_in(["yes", "no"])))
    if sub.is_empty():
        return pl.DataFrame()
    trades = tr.filter(pl.col("ticker").is_in(sub["ticker"].unique().to_list())).collect()
    if trades.is_empty():
        return pl.DataFrame()
    daily = KalshiOHLCV.build_daily(trades, mk.filter(series_filter_expr(canon)))
    return (daily.join(sub.select("ticker", "strike", "result"), on="ticker", how="inner")
            .filter(pl.col("trade_count") > 0)
            .select("event_ticker", "date", "ticker", "strike",
                    pl.col("close").alias("price"),
                    (pl.col("result") == "yes").cast(pl.Int8).alias("win")))


def main() -> None:
    n = pl.read_parquet(PANELS / "node_panel_event.parquet")
    n = n.with_columns(
        pl.col("event_ticker").map_elements(parse_month, return_dtype=pl.Date).alias("ref"))
    # true published print = stored ladder midpoint + 0.05
    truth = {(r["series"], r["ref"]): round(r["resolved_value"] + 0.05, 2)
             for r in (n.group_by("series", "ref")
                       .agg(pl.col("resolved_value").first())
                       .drop_nulls().iter_rows(named=True)) if r["ref"] is not None}

    print("=== 1. does the identity hold EXACTLY on published prints? ===")
    print("YoY_t - (c_t + MoM_t) where c_t = YoY_{t-1} - MoM_{t-12}, all on the")
    print("0.1 grid. Exact means the strike mapping is exact.\n")
    cmap: dict = {}
    for mom_s, yoy_s in PAIRS:
        res = []
        for (s, ref), v in truth.items():
            if s != yoy_s:
                continue
            prev = truth.get((yoy_s, shift_months(ref, -1)))
            back = truth.get((mom_s, shift_months(ref, -12)))
            cur = truth.get((mom_s, ref))
            if prev is None or back is None:
                continue
            c = round(prev - back, 2)
            cmap[(mom_s, ref)] = c
            if cur is not None:
                res.append(round(v - (c + cur), 2))
        if res:
            a = np.array(res)
            print(f"  {mom_s}/{yoy_s}: n = {len(a):2d}   exact (|r| < 0.001): "
                  f"{int((np.abs(a) < 0.001).sum())}   |r| <= 0.1: "
                  f"{int((np.abs(a) <= 0.1001).sum())}   sd {a.std():.3f}pp   "
                  f"max |r| {np.abs(a).max():.2f}")

    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
    legs = {}
    for s in {x for p in PAIRS for x in p}:
        legs[s] = daily_legs(s, mk, tr)
        print(f"  {s:<12} daily leg-days: {legs[s].height}")

    # ---------------------------------------------------------------- §2
    print("\n=== 2. strike-matched pairs: two tickers, one event ===")
    rows = []
    for mom_s, yoy_s in PAIRS:
        a, b = legs.get(mom_s), legs.get(yoy_s)
        if a is None or b is None or a.is_empty() or b.is_empty():
            continue
        a = a.with_columns(pl.col("event_ticker").map_elements(
            parse_month, return_dtype=pl.Date).alias("ref"))
        b = b.with_columns(pl.col("event_ticker").map_elements(
            parse_month, return_dtype=pl.Date).alias("ref"))
        for ref in sorted(set(a["ref"].drop_nulls().to_list())
                          & set(b["ref"].drop_nulls().to_list())):
            c = cmap.get((mom_s, ref))
            if c is None:
                continue
            am = a.filter(pl.col("ref") == ref).with_columns(
                (pl.col("strike") + c).round(2).alias("key"))
            bm = b.filter(pl.col("ref") == ref).with_columns(
                pl.col("strike").round(2).alias("key"))
            j = am.join(bm, on=["date", "key"], how="inner", suffix="_yoy")
            for r in j.iter_rows(named=True):
                rows.append(dict(
                    pair=f"{mom_s}/{yoy_s}", ref=ref, date=r["date"], c_t=c,
                    k_mom=r["strike"], k_yoy=r["strike_yoy"],
                    p_mom=float(r["price"]), p_yoy=float(r["price_yoy"]),
                    win_mom=int(r["win"]), win_yoy=int(r["win_yoy"]),
                    diff=float(r["price_yoy"]) - float(r["price"])))
    if not rows:
        print("no matched pairs")
        return
    m = pl.DataFrame(rows)
    assert_no_oos(m.with_columns(
        pl.col("date").cast(pl.Datetime).dt.replace_time_zone("UTC").alias("t")),
        time_col="t")
    print(f"matched (date, strike) pairs: {m.height}   "
          f"reference months: {m['ref'].n_unique()}\n")
    with pl.Config(tbl_rows=10, float_precision=2, tbl_width_chars=200):
        print(m.group_by("pair").agg(
            pl.len().alias("pairs"), pl.col("ref").n_unique().alias("months"),
            pl.col("diff").abs().mean().alias("mean_abs_diff"),
            pl.col("diff").abs().median().alias("p50_abs_diff"),
            pl.col("diff").abs().quantile(0.9).alias("p90_abs_diff"),
            (pl.col("diff").abs() > 5).mean().alias("frac_gap_gt_5c"),
            pl.col("diff").mean().alias("mean_signed"),
        ))

    # ---------------------------------------------------------------- §3
    print("\n=== 3. basis risk: how often do the two legs settle differently? ===")
    m = m.with_columns((pl.col("win_mom") != pl.col("win_yoy")).alias("break"))
    with pl.Config(tbl_rows=10, float_precision=4, tbl_width_chars=200):
        print(m.group_by("pair").agg(
            pl.len().alias("pairs"),
            pl.col("break").mean().alias("break_rate"),
            pl.col("ref").n_unique().alias("months"),
        ))
    print("\nbreak rate by |distance of the matched strike from the eventual print|:")
    m = m.with_columns(
        pl.struct("pair", "ref", "k_mom").map_elements(
            lambda r: (lambda t: abs(r["k_mom"] - t) if t is not None else None)(
                truth.get((r["pair"].split("/")[0], r["ref"]))),
            return_dtype=pl.Float64).alias("dist"))
    with pl.Config(tbl_rows=12, float_precision=3):
        print(m.drop_nulls("dist").with_columns(
            pl.when(pl.col("dist") <= 0.051).then(pl.lit("<=0.05"))
              .when(pl.col("dist") <= 0.151).then(pl.lit("0.05-0.15"))
              .when(pl.col("dist") <= 0.351).then(pl.lit("0.15-0.35"))
              .otherwise(pl.lit(">0.35")).alias("d")
        ).group_by("d").agg(pl.len().alias("n"),
                            pl.col("break").mean().alias("break_rate"),
                            pl.col("diff").abs().mean().alias("abs_diff")).sort("d"))

    # ---------------------------------------------------------------- §4
    print("\n=== 4. the trade: buy the cheap leg, sell the dear one, hold both ===")
    print("Capital = p_cheap + (100 - p_dear). Pays 100c if the legs agree.")
    print("A break costs -100c, so the break rate is the whole question.\n")
    p_lo = np.minimum(m["p_mom"].to_numpy(), m["p_yoy"].to_numpy())
    p_hi = np.maximum(m["p_mom"].to_numpy(), m["p_yoy"].to_numpy())
    cheap_is_mom = m["p_mom"].to_numpy() <= m["p_yoy"].to_numpy()
    win_cheap = np.where(cheap_is_mom, m["win_mom"].to_numpy(), m["win_yoy"].to_numpy())
    win_dear = np.where(cheap_is_mom, m["win_yoy"].to_numpy(), m["win_mom"].to_numpy())
    payoff = 100.0 * win_cheap + 100.0 * (1 - win_dear)
    cost = (fee_cents(p_lo) + fee_cents(100.0 - p_hi)
            + spread_cents(p_lo) / 2.0 + spread_cents(100.0 - p_hi) / 2.0)
    gross = payoff - (p_lo + (100.0 - p_hi))
    net = gross - cost
    m = m.with_columns(pl.Series("gross", gross), pl.Series("net", net),
                       pl.Series("cost", cost),
                       pl.Series("capital", p_lo + (100.0 - p_hi)),
                       pl.Series("edge", p_hi - p_lo))

    rows = []
    for thr in (0.0, 1.0, 2.0, 3.0, 5.0):
        s = m.filter(pl.col("edge") > thr)
        if s.height < 10:
            continue
        rows.append(dict(min_edge_c=thr, n=s.height, months=s["ref"].n_unique(),
                         mean_edge=float(s["edge"].mean()),
                         break_rate=float(s["break"].mean()),
                         mean_cost=float(s["cost"].mean()),
                         gross=float(s["gross"].mean()),
                         net=float(s["net"].mean()),
                         capital=float(s["capital"].mean()),
                         ret_cap=100.0 * float((s["net"] / s["capital"]).mean())))
    with pl.Config(tbl_rows=12, float_precision=2, tbl_width_chars=220):
        print(pl.DataFrame(rows))

    print("\nclustered on reference month (one print settles every pair in a month):")
    for thr in (0.0, 2.0):
        s = m.filter(pl.col("edge") > thr)
        if s.height < 10:
            continue
        g = [x["net"].to_numpy() for _, x in s.group_by("ref")]
        rng = np.random.default_rng(0)
        idx = np.arange(len(g))
        boot = np.array([np.concatenate([g[i] for i in rng.choice(idx, len(idx), True)]).mean()
                         for _ in range(5000)])
        print(f"  edge > {thr:.0f}c:  net {s['net'].mean():+.2f}c   "
              f"95% CI [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]"
              f"   P(<=0) = {(boot <= 0).mean():.3f}   months = {s['ref'].n_unique()}")

    # ---------------------------------------------------------------- §5
    print("\n=== 5. reconciliation: the width gap is NOT a mispricing ===")
    print("identity_cpi.py found the YoY ladder implying ~21% more width than")
    print("the MoM ladder and read it as an inconsistency. §1 says why it is not:")
    print("YoY = c + MoM + e, where e is the identity residual (BLS rounds MoM and")
    print("YoY independently, and YoY is computed off unrounded index levels). If")
    print("e is independent of MoM then sd(YoY) = sqrt(sd(MoM)^2 + sd(e)^2), so a")
    print("WIDER YoY ladder is correct pricing, not an arbitrage.\n")
    ident = pl.read_parquet(OUT / "identity_cpi.parquet") \
        if (OUT / "identity_cpi.parquet").exists() else None
    rows = []
    for mom_s, yoy_s in PAIRS:
        res = []
        for (s_, ref), v in truth.items():
            if s_ != yoy_s:
                continue
            prev = truth.get((yoy_s, shift_months(ref, -1)))
            back = truth.get((mom_s, shift_months(ref, -12)))
            cur = truth.get((mom_s, ref))
            if None in (prev, back, cur):
                continue
            res.append(v - (round(prev - back, 2) + cur))
        if not res or ident is None:
            continue
        sd_e = float(np.std(res))
        sub = ident.filter(pl.col("pair") == f"{mom_s}/{yoy_s}")
        if sub.is_empty():
            continue
        sd_mom = float(sub["sd_mom"].mean())
        rows.append(dict(pair=f"{mom_s}/{yoy_s}", sd_mom=sd_mom, sd_resid=sd_e,
                         predicted_ratio=float(np.sqrt(sd_mom**2 + sd_e**2) / sd_mom),
                         observed_ratio=float(sub["sd_ratio"].median())))
    with pl.Config(float_precision=3, tbl_width_chars=200):
        print(pl.DataFrame(rows))
    print("\nIf observed <= predicted, the YoY ladder is if anything too NARROW,")
    print("and there is no width arbitrage to collect.")

    OUT.mkdir(parents=True, exist_ok=True)
    m.write_parquet(OUT / "identity_arb.parquet")
    print(f"\nwrote {OUT / 'identity_arb.parquet'}")


if __name__ == "__main__":
    main()
