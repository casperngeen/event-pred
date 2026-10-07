"""Maker simulation from trade prints: does the model's signal beat a baseline maker?

Every 60 s of game time, for each spread / total / winner contract with a recent print (the reference
price, <= 60 s old), resting orders are posted for FILL_S seconds:

    buy  at ref - k        filled if a NO-taker print (someone selling YES) trades at <= that price
    sell at ref + k        filled if a YES-taker print trades at >= that price

"at" counts a print at our price as a fill (queue position ignored: optimistic); "through" needs a
print strictly past our price (we would surely have been filled: conservative). Fill price = our
price. Markout = volume-weighted price of prints in [fill + 60 s, fill + 90 s] (about the mid);
P&L per fill = side x (markout - fill price) before any exit cost, and less a maker fee of
MAKER_RATE x p (1 - p) (unverified; 0 on many series).

Strategies (k in K_CENTS):
    baseline     post both sides at every decision, ignoring the model
    signal       post only the side the model favours, when |signal| >= m (signal = model level at
                 +60 s minus ref)
    against      post only the side the model disfavours (the trades the signal avoids)

Reported per fill and per decision (P&L summed over fills / decisions: value of running the strategy).

    venv/bin/python -W ignore analysis/graph_2026_10/b6_maker.py results/ingame NFL 10 [panel_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_longshort import LEAGUE, DELAY, PANEL, Panel, attach_tickers, rows   # noqa: E402
from b5_backtest import clustered_t                                           # noqa: E402
from stg.sports.loaders import scan_trades                                    # noqa: E402

FILL_S, MARK_S, MARK_W, REF_MAX_S = 60, 60, 30, 60
K_CENTS = [1, 2]
MARGINS = [2.0, 3.0, 5.0]
MAKER_RATE = 1.75


def simulate(r: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    """One row per (decision, side, k, rule) with a fill: fill price, markout."""
    out = []
    by_t = {t: d for (t,), d in tr.partition_by("ticker", as_dict=True).items()}
    for (tk,), d in r.partition_by("ticker", as_dict=True).items():
        p = by_t.get(tk)
        if p is None:
            continue
        pt, px = p["t"].to_numpy(), p["yes_price"].to_numpy()
        yes = (p["taker_side"] == "yes").to_numpy()
        cv, cn = np.cumsum(p["count"].to_numpy()), np.cumsum(p["count"].to_numpy() * px)
        T, rid = d["t"].to_numpy(), d["row"].to_numpy()
        i_ref = np.searchsorted(pt, T, side="right") - 1
        ok = (i_ref >= 0) & ((T - pt[np.maximum(i_ref, 0)]) <= REF_MAX_S)
        a, b = np.searchsorted(pt, T, side="right"), np.searchsorted(pt, T + FILL_S, side="right")
        for j in np.flatnonzero(ok):
            ref = px[i_ref[j]]
            w = slice(a[j], b[j])
            wp, wy, wt = px[w], yes[w], pt[w]
            for k in K_CENTS:
                for side, pr, hit_at, hit_thr in (
                        (1, ref - k, (~wy) & (wp <= ref - k), (~wy) & (wp < ref - k)),
                        (-1, ref + k, wy & (wp >= ref + k), wy & (wp > ref + k))):
                    if pr <= 0 or pr >= 100:
                        continue
                    for rule, hit in (("at", hit_at), ("through", hit_thr)):
                        if not hit.any():
                            continue
                        tf = wt[np.argmax(hit)]
                        m0, m1 = np.searchsorted(pt, tf + MARK_S, side="right"), np.searchsorted(pt, tf + MARK_S + MARK_W, side="right")
                        v = cv[m1 - 1] - cv[m0 - 1] if m1 > m0 else 0
                        if v <= 0:
                            continue
                        mark = (cn[m1 - 1] - cn[m0 - 1]) / v
                        out.append((int(rid[j]), side, k, rule, float(pr), float(mark)))
    return pl.DataFrame(out, schema=["row", "side", "k", "rule", "price", "mark"], orient="row")


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(220)
    pn = Panel(PANEL)
    games = json.loads((PANEL / "manifest.json").read_text())["games"]
    r = attach_tickers(rows(pn, games)).with_row_index("row")
    tr = (scan_trades(r["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(r["ticker"].unique().implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side", "count")
          .collect().sort("ticker", "t"))
    # reference price (last print) and signal at each decision
    r = (r.sort("t").join_asof(tr.select("ticker", "t", pl.col("yes_price").alias("ref"), pl.col("t").alias("t_ref")).sort("t"),
                               on="t", by="ticker", strategy="backward")
         .filter((pl.col("t") - pl.col("t_ref")) <= REF_MAX_S).with_columns((pl.col("level") - pl.col("ref")).alias("signal")))
    f = simulate(r, tr).join(r.select("row", "kind", "game_id", "signal"), on="row")
    q = (pl.col("price") / 100).clip(0, 1)
    f = f.with_columns((pl.col("side") * (pl.col("mark") - pl.col("price"))).alias("gross"))
    f = f.with_columns((pl.col("gross") - MAKER_RATE * q * (1 - q)).alias("net_fee"))
    print(f"{LEAGUE}, feed delay {DELAY:g} s: {r.height:,} contract-decisions with a print in the last {REF_MAX_S} s")
    res = []
    for kind in ("spread", "total", "winner"):
        n_dec = r.filter(pl.col("kind") == kind).height
        fk = f.filter(pl.col("kind") == kind)
        strategies = [("baseline", None, pl.lit(True))]
        for m in MARGINS:
            strategies.append(("signal", m, pl.col("side") * pl.col("signal") >= m))
            strategies.append(("against", m, pl.col("side") * pl.col("signal") <= -m))
        for name, m, cond in strategies:
            for (k, rule), g in fk.filter(cond).group_by("k", "rule"):
                gm = g["game_id"].to_numpy()
                res.append(dict(kind=kind, strategy=name, margin=m, k=k, rule=rule, fills=g.height,
                                gross=round(g["gross"].mean(), 2), t=round(clustered_t(g["gross"].to_numpy(), gm), 1),
                                net_fee=round(g["net_fee"].mean(), 2),
                                per_100_decisions=round(100 * g["gross"].sum() / max(n_dec, 1), 2)))
    out = pl.DataFrame(res).sort("kind", "rule", "k", "strategy", "margin", nulls_last=False)
    print("markout per fill (cents, before exit costs); per_100_decisions = summed gross per 100 contract-decisions")
    print(out)
    out.write_parquet(Path(__file__).resolve().parent / "out" / f"b6_maker_{LEAGUE}_d{DELAY:g}.parquet")


if __name__ == "__main__":
    main()
