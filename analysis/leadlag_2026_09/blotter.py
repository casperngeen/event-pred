#!/usr/bin/env python
"""Full trade blotter for the frozen specification.

    venv/bin/python analysis/leadlag_2026_09/blotter.py
    (needs build_panel.py)

Writes every position the spec in reports/strategy_spec.md would have taken,
one row per trade, to out/trade_blotter.csv.

Columns
-------
trigger / trigger_event / trigger_resolved   the release that fired the signal
z_surprise / direction / signal              the signal and its components
target / target_event / target_ticker        what was traded
strike                                        the leg's threshold
prediction                                    the claim in words
entry_date / entry_price                      the fill (second post-news print)
p0 / p_first / move_c                         pre-news price, confirming print, the move
close_date / hold_days                        settlement
settled_yes / we_won                          outcome
gross_c / cost_c / net_c                      P&L in cents per contract

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import scan_trades

OUT = Path("analysis/leadlag_2026_09/out")
FEE_RATE, CONTRACTS = 0.07, 100
BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]
CONFIRM_C, FLOOR_C = 2.0, 20.0


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet").drop_nulls("p0")
    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99))
    zc = np.clip(z, -lim, lim)
    dirn = d["direction"].to_numpy().astype(float)
    sig = dirn * zc
    p0, pe = d["p0"].to_numpy(), d["p_entry"].to_numpy()
    win = d["win"].to_numpy().astype(float)
    yr = d["yr"].to_numpy()
    tk = d["target_ticker"].to_numpy()
    tres_s = d["t_res"].dt.epoch("s").to_numpy()
    years = sorted(np.unique(yr))

    side = np.zeros(len(p0))
    for lo, hi in BUCKETS:
        m = (p0 >= lo) & (p0 < hi)
        if m.sum() < 60:
            continue
        idx = np.where(m)[0]
        for Y in years[1:]:
            tr, te = idx[yr[idx] < Y], idx[yr[idx] == Y]
            if len(tr) < 60 or len(te) == 0:
                continue
            t = np.percentile(sig[tr], [33.3, 66.7])
            side[te] = np.where(sig[te] > t[1], 1.0,
                                np.where(sig[te] <= t[0], -1.0, 0.0))

    tt = (scan_trades(is_only=True)
          .filter(pl.col("ticker").is_in(d["target_ticker"].unique().to_list()))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    g = (tt.with_columns(pl.col("created_time").dt.epoch("s").alias("s"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("s"), pl.col("yes_price")))
    tape = {t: (np.asarray(a, dtype=np.int64), np.asarray(b, dtype=float))
            for t, a, b in zip(g["ticker"], g["s"], g["yes_price"])}

    p2 = np.full(len(p0), np.nan)
    t2 = np.full(len(p0), np.nan)
    for i in range(len(p0)):
        arr = tape.get(tk[i])
        if arr is None:
            continue
        s_, px_ = arr
        j = int(np.searchsorted(s_, tres_s[i], side="right"))
        if j + 1 < len(s_):
            p2[i], t2[i] = px_[j + 1], s_[j + 1]

    entry = np.where(side > 0, p2, 100.0 - p2)
    keep = ((side != 0) & np.isfinite(p2) & (entry >= FLOOR_C)
            & (((pe - p0) * np.sign(side)) >= CONFIRM_C))
    i = np.where(keep)[0]

    payoff = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
    cost = fee_cents(entry) + spread_cents(entry) / 2.0
    gross = payoff - entry
    net = gross - cost

    b = d[i].with_columns([
        pl.Series("side", side[i]),
        pl.Series("yes_price_at_fill", np.round(p2[i], 1)),
        pl.Series("entry_price", np.round(entry[i], 1)),
        pl.Series("entry_s", t2[i]),
        pl.Series("p_first", pe[i]),
        pl.Series("move_c", np.round(((pe - p0) * np.sign(side))[i], 1)),
        pl.Series("z_w", np.round(zc[i], 3)),
        pl.Series("signal_v", np.round(sig[i], 3)),
        pl.Series("gross_c", np.round(gross[i], 2)),
        pl.Series("cost_c", np.round(cost[i], 2)),
        pl.Series("net_c", np.round(net[i], 2)),
        pl.Series("we_won", (payoff[i] > 50).astype(int)),
    ])
    b = b.with_columns([
        pl.from_epoch(pl.col("entry_s").cast(pl.Int64), time_unit="s")
          .dt.replace_time_zone("UTC").alias("entry_date"),
        pl.when(pl.col("side") > 0).then(pl.lit("BUY YES"))
          .otherwise(pl.lit("SELL YES")).alias("action"),
        pl.col("win").alias("settled_yes"),
    ])
    b = b.with_columns([
        (pl.col("close_time") - pl.col("entry_date")).dt.total_days().alias("hold_days"),
        (pl.when(pl.col("side") > 0)
         .then(pl.lit("above "))
         .otherwise(pl.lit("at or below ")) + pl.col("strike").cast(pl.Utf8)).alias("pred_side"),
    ])
    b = b.with_columns(
        (pl.col("target") + " settles " + pl.col("pred_side")).alias("prediction"))

    # yes_price_at_fill is the YES quote we transacted against; entry_price is
    # what we actually paid, which equals it for a BUY YES and (100 - it) for a
    # SELL YES, because selling YES on a fully collateralised venue IS buying NO.
    cols = ["trigger", "trigger_event", "t_res", "z_w", "direction", "signal_v",
            "target", "target_event", "target_ticker", "strike", "action",
            "prediction", "p0", "p_first", "move_c", "entry_date",
            "yes_price_at_fill", "entry_price",
            "close_time", "hold_days", "settled_yes", "we_won",
            "gross_c", "cost_c", "net_c"]
    b = (b.select(cols)
         .rename({"t_res": "trigger_resolved", "close_time": "close_date",
                  "z_w": "z_surprise", "signal_v": "signal"})
         .sort("trigger_resolved", "target_event", "strike"))

    OUT.mkdir(parents=True, exist_ok=True)
    b.write_csv(OUT / "trade_blotter.csv")
    b.write_parquet(OUT / "trade_blotter.parquet")
    print(f"wrote {OUT / 'trade_blotter.csv'}   rows: {b.height}")

    print(f"\ntrades {b.height}   trigger events {b['trigger_event'].n_unique()}   "
          f"target events {b['target_event'].n_unique()}   "
          f"pairs {b.select(['trigger', 'target']).unique().height}")
    print(f"date range: {b['trigger_resolved'].min().date()} -> {b['close_date'].max().date()}")
    print(f"win rate {b['we_won'].mean():.3f}   mean entry {b['entry_price'].mean():.1f}c   "
          f"mean hold {b['hold_days'].mean():.0f}d   total net {b['net_c'].sum():.0f}c")

    print("\n--- first 12 trades ---")
    show = ["trigger_resolved", "trigger", "target_event", "strike", "action",
            "entry_date", "entry_price", "close_date", "settled_yes", "net_c"]
    with pl.Config(tbl_rows=14, tbl_width_chars=250, float_precision=1,
                   fmt_str_lengths=22):
        print(b.select(show).head(12))
        print("\n--- 5 biggest winners ---")
        print(b.select(show).sort("net_c", descending=True).head(5))
        print("\n--- 5 biggest losers ---")
        print(b.select(show).sort("net_c").head(5))

    print("\n--- by year ---")
    with pl.Config(float_precision=2):
        print(b.with_columns(pl.col("trigger_resolved").dt.year().alias("yr"))
              .group_by("yr").agg(pl.len().alias("trades"),
                                  pl.col("we_won").mean().alias("win_rate"),
                                  pl.col("net_c").mean().alias("mean_net"),
                                  pl.col("net_c").sum().alias("total_net")).sort("yr"))


if __name__ == "__main__":
    main()
