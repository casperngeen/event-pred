"""Maker entry with a real exit, priced only from trade prints. Fills need the market to trade THROUGH
our limit (a print strictly past our price), so a fill is certain whatever our queue position.

Every 60 s of game time, for each spread / total / winner contract with a print in the last 60 s
(the reference price, ref):

    entry      buy limit at ref - 1 (filled by a NO-taker print below it) or sell limit at ref + 1
               (filled by a YES-taker print above it), resting ENTRY_S seconds
    exit       at fill + hold (30 / 60 / 120 s), one of:
       take    cross the spread: the price the next taker on our exit side paid (a NO-taker print
               if we sell, a YES-taker print if we buy back), within EXIT_S; none -> hold to settlement
       limit   rest at ref_exit + 1 (selling) / ref_exit - 1 (buying back), filled only by a print
               through it within EXIT_S; not filled -> take as above (from the end of that wait)
    P&L        side x (exit - entry) cents per contract
    fees       maker MAKER_RATE x p (1 - p) on a limit fill (unverified), taker FEE_RATE x p (1 - p) on
               a taken exit, none at settlement

Strategies: baseline = post both sides at every decision; signal = only the side the model favours,
when |model level at +hold - ref| >= m. t clustered by game.

    venv/bin/python -W ignore analysis/graph_2026_10/b6_maker_exit.py results/ingame NFL 10 [panel_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_longshort import LEAGUE, DELAY, PANEL, RUN_TAG, Panel, attach_tickers, rows   # noqa: E402
from b5_backtest import FEE_RATE, clustered_t                                 # noqa: E402
from stg.sports.loaders import scan_trades                                    # noqa: E402

ENTRY_S, EXIT_S, REF_MAX_S, K = 60, 120, 60, 1
HOLDS = {"30s": 30, "60s": 60, "120s": 120}
MARGINS = [2.0, 3.0, 5.0, 8.0]
MAKER_RATE = 1.75


def first_through(px, yes, t, a, b, side_taker_yes: bool, price, above: bool):
    """Index of the first print in [a, b) by the given taker side strictly through price, else -1."""
    if b <= a:
        return -1
    w = yes[a:b] if side_taker_yes else ~yes[a:b]
    hit = w & ((px[a:b] > price) if above else (px[a:b] < price))
    return a + int(np.argmax(hit)) if hit.any() else -1


def first_taker(yes, a, b, taker_yes: bool):
    if b <= a:
        return -1
    w = yes[a:b] if taker_yes else ~yes[a:b]
    return a + int(np.argmax(w)) if w.any() else -1


def simulate(r: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    out = []
    by_t = {t: d for (t,), d in tr.partition_by("ticker", as_dict=True).items()}
    for (tk,), d in r.partition_by("ticker", as_dict=True).items():
        p = by_t.get(tk)
        if p is None:
            continue
        pt, px = p["t"].to_numpy(), p["yes_price"].to_numpy()
        yes = (p["taker_side"] == "yes").to_numpy()
        T, rid, ys = d["t"].to_numpy(), d["row"].to_numpy(), d["y"].to_numpy()
        i_ref = np.searchsorted(pt, T, side="right") - 1
        ok = (i_ref >= 0) & ((T - pt[np.maximum(i_ref, 0)]) <= REF_MAX_S)
        a0, b0 = np.searchsorted(pt, T, side="right"), np.searchsorted(pt, T + ENTRY_S, side="right")
        for j in np.flatnonzero(ok):
            ref = px[i_ref[j]]
            for side in (1, -1):
                lim = ref - K if side == 1 else ref + K
                if lim <= 0 or lim >= 100:
                    continue
                # long: a NO-taker (seller) prints below our bid; short: a YES-taker prints above our ask
                f = first_through(px, yes, pt, a0[j], b0[j], side_taker_yes=(side == -1), price=lim, above=(side == -1))
                if f < 0:
                    continue
                tf = pt[f]
                for hn, hs in HOLDS.items():
                    te = tf + hs
                    ae = np.searchsorted(pt, te, side="right")
                    # taker exit: long sells (a NO-taker print), short buys back (a YES-taker print)
                    be = np.searchsorted(pt, te + EXIT_S, side="right")
                    x = first_taker(yes, ae, be, taker_yes=(side == -1))
                    take_px, take_how = (px[x], "taken") if x >= 0 else (100 * ys[j], "settled")
                    out.append((int(rid[j]), side, hn, "take", float(lim), float(take_px), take_how))
                    # limit exit at the last print at te, +-1, through-fill within EXIT_S, else take
                    i_e = ae - 1
                    ref_e = px[i_e] if i_e >= 0 else ref
                    lx = ref_e + K if side == 1 else ref_e - K
                    fx = first_through(px, yes, pt, ae, be, side_taker_yes=(side == 1), price=lx, above=(side == 1)) \
                        if 0 < lx < 100 else -1
                    if fx >= 0:
                        out.append((int(rid[j]), side, hn, "limit", float(lim), float(lx), "limit filled"))
                    else:
                        a2, b2 = be, np.searchsorted(pt, te + 2 * EXIT_S, side="right")
                        x2 = first_taker(yes, a2, b2, taker_yes=(side == -1))
                        px2, how2 = (px[x2], "taken") if x2 >= 0 else (100 * ys[j], "settled")
                        out.append((int(rid[j]), side, hn, "limit", float(lim), float(px2), how2))
    return pl.DataFrame(out, schema=["row", "side", "hold", "exit_rule", "entry", "exit", "how"], orient="row")


def main() -> None:
    pl.Config.set_tbl_rows(120)
    pl.Config.set_tbl_width_chars(230)
    pl.Config.set_tbl_cols(20)
    pn = Panel(PANEL)
    games = json.loads((PANEL / "manifest.json").read_text())["games"]
    r = attach_tickers(rows(pn, games)).filter(pl.col("y").is_finite()).with_row_index("row")
    tr = (scan_trades(r["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(r["ticker"].unique().implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side")
          .collect().sort("ticker", "t"))
    r = (r.sort("t").join_asof(tr.select("ticker", "t", pl.col("yes_price").alias("ref"), pl.col("t").alias("t_ref")).sort("t"),
                               on="t", by="ticker", strategy="backward")
         .filter((pl.col("t") - pl.col("t_ref")) <= REF_MAX_S))
    lv = [c for c in r.columns if c.startswith("level_")]
    f = simulate(r, tr).join(r.select("row", "kind", "game_id", "ref", *lv), on="row")
    # signal at the matching horizon (60 s if the model has no such horizon)
    f = f.with_columns(pl.coalesce(*[pl.when(pl.col("hold") == c[6:]).then(pl.col(c)) for c in lv], pl.col("level_60s"))
                       .sub(pl.col("ref")).alias("signal"))
    q = lambda c: (pl.col(c) / 100).clip(0, 1)   # noqa: E731
    fee_in = MAKER_RATE * q("entry") * (1 - q("entry"))
    fee_out = (pl.when(pl.col("how") == "taken").then(FEE_RATE * q("exit") * (1 - q("exit")))
               .when(pl.col("how") == "limit filled").then(MAKER_RATE * q("exit") * (1 - q("exit"))).otherwise(0.0))
    f = f.with_columns((pl.col("side") * (pl.col("exit") - pl.col("entry"))).alias("gross"))
    f = f.with_columns((pl.col("gross") - fee_in - fee_out).alias("net"),
                       (pl.col("gross") - pl.when(pl.col("how") == "taken").then(FEE_RATE * q("exit") * (1 - q("exit"))).otherwise(0.0))
                       .alias("net_no_maker_fee"))
    n_dec = r.group_by("kind").len()
    print(f"{LEAGUE}, feed delay {DELAY:g} s: {r.height:,} contract-decisions; entries filled (through): "
          f"{f.filter(pl.col('hold') == '60s', pl.col('exit_rule') == 'take').height:,}")
    res = []
    for (kind, hold, rule), g0 in f.group_by("kind", "hold", "exit_rule"):
        nd = n_dec.filter(pl.col("kind") == kind)["len"].item()
        for name, m, cond in [("baseline", None, pl.lit(True))] + [("signal", m, pl.col("side") * pl.col("signal") >= m) for m in MARGINS]:
            g = g0.filter(cond)
            if g.height < 30:
                continue
            gm = g["game_id"].to_numpy()
            res.append(dict(kind=kind, hold=hold, exit=rule, strategy=name, margin=m, trades=g.height,
                            gross=round(g["gross"].mean(), 2), net=round(g["net"].mean(), 2),
                            t_net=round(clustered_t(g["net"].to_numpy(), gm), 1),
                            net_no_maker_fee=round(g["net_no_maker_fee"].mean(), 2),
                            exit_limit_filled=round((g["how"] == "limit filled").mean(), 2),
                            settled=round((g["how"] == "settled").mean(), 3),
                            net_per_100_decisions=round(100 * g["net"].sum() / nd, 2)))
    out = pl.DataFrame(res).sort("kind", "exit", "hold", "strategy", "margin", nulls_last=False)
    out.write_parquet(Path(__file__).resolve().parent / "out" / f"b6_maker_exit_{LEAGUE}_d{DELAY:g}{RUN_TAG}.parquet")
    print("cents per trade (one contract)")
    print(out)


if __name__ == "__main__":
    main()
