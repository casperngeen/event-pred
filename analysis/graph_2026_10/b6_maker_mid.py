"""Maker entry, then an exit order that follows the market's estimated middle price, with a deadline.
Priced only from trade prints; a fill needs a print strictly THROUGH our price (certain whatever our
queue position, and conservative: a print exactly at our price is not counted).

Entry as in b6_maker_exit.py: every 60 s of game time, for each spread / total / winner contract with
a print in the last 60 s (ref), buy limit at ref - 1 or sell limit at ref + 1, resting ENTRY_S
seconds; only the side the model favours, when |model level at +hold - ref| >= m.

Exit, from fill + hold (30 / 60 / 120 s):
    fixed      (as b6_maker_exit "limit") rest at last print -+ 1, filled by a print through it within
               EXIT_S; else take
    mid_X      rest at the estimated mid; after every print, move the order to the new mid; filled by
               a print through it within X s (X = 30 / 60 / 120); else take
    mid        = (last YES-taker price + last NO-taker price) / 2, each from the last MID_WIN s; one
               side only -> that price; neither -> the last print. Our bid is floor(mid), our ask
               ceil(mid).
    take       the price the next taker on our exit side paid, within EXIT_S; none -> settlement
Fees: maker MAKER_RATE x p (1 - p) on limit fills (unverified), taker FEE_RATE x p (1 - p) on a taken
exit, none at settlement. t clustered by game.

    RUN_TAG=_h30-60-120-1800_tpf venv/bin/python -W ignore analysis/graph_2026_10/b6_maker_mid.py \
        results/ingame NBA 0 data/sports/derived/ingame/NBA/panel_v3f
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_longshort import LEAGUE, DELAY, PANEL, RUN_TAG, Panel, attach_tickers, rows   # noqa: E402
from b5_backtest import FEE_RATE, clustered_t                                 # noqa: E402
from b6_maker_exit import ENTRY_S, EXIT_S, REF_MAX_S, K, HOLDS, MAKER_RATE, first_through, first_taker   # noqa: E402
from stg.sports.loaders import scan_trades                                    # noqa: E402

DEADLINES = [30, 60, 120]
MID_WIN = 60
MARGINS = [5.0, 8.0]


def mid_at(pt, px, yes, i):
    """Estimated mid from prints before index i (within MID_WIN s of the latest of them)."""
    if i <= 0:
        return None
    t_now, lo = pt[i - 1], np.searchsorted(pt, pt[i - 1] - MID_WIN, side="left")
    w_px, w_yes = px[lo:i], yes[lo:i]
    a = w_px[w_yes][-1] if w_yes.any() else None          # last YES-taker: near the ask
    b = w_px[~w_yes][-1] if (~w_yes).any() else None      # last NO-taker: near the bid
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b if b is not None else px[i - 1]


def chase(pt, px, yes, side, a, t_end):
    """Mid-following exit order from print index a until time t_end. Returns (price, index) or None."""
    b = np.searchsorted(pt, t_end, side="right")
    for i in range(a, b):
        m = mid_at(pt, px, yes, i)
        if m is None:
            continue
        if side == 1:      # long: sell at ceil(mid), filled by a YES-taker print above it
            q = math.ceil(m)
            if 0 < q < 100 and yes[i] and px[i] > q:
                return float(q), i
        else:              # short: buy back at floor(mid), filled by a NO-taker print below it
            q = math.floor(m)
            if 0 < q < 100 and not yes[i] and px[i] < q:
                return float(q), i
    return None


def take_after(pt, px, yes, side, t0, y):
    a, b = np.searchsorted(pt, t0, side="right"), np.searchsorted(pt, t0 + EXIT_S, side="right")
    x = first_taker(yes, a, b, taker_yes=(side == -1))
    return (float(px[x]), "taken") if x >= 0 else (100.0 * y, "settled")


def simulate(r: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    out = []
    by_t = {t: d for (t,), d in tr.partition_by("ticker", as_dict=True).items()}
    for (tk,), d in r.partition_by("ticker", as_dict=True).items():
        p = by_t.get(tk)
        if p is None:
            continue
        pt, px = p["t"].to_numpy(), p["yes_price"].to_numpy()
        yes = (p["taker_side"] == "yes").to_numpy()
        T, rid, ys, refs = d["t"].to_numpy(), d["row"].to_numpy(), d["y"].to_numpy(), d["ref"].to_numpy()
        sig = {hn: d[f"sig_{hn}"].to_numpy() for hn in HOLDS}
        a0, b0 = np.searchsorted(pt, T, side="right"), np.searchsorted(pt, T + ENTRY_S, side="right")
        for j in range(len(T)):
            ref = refs[j]
            for side in (1, -1):
                if max(side * sig[hn][j] for hn in HOLDS) < MARGINS[0]:
                    continue
                lim = ref - K if side == 1 else ref + K
                if lim <= 0 or lim >= 100:
                    continue
                f = first_through(px, yes, pt, a0[j], b0[j], side_taker_yes=(side == -1), price=lim, above=(side == -1))
                if f < 0:
                    continue
                tf = pt[f]
                for hn, hs in HOLDS.items():
                    if side * sig[hn][j] < MARGINS[0]:
                        continue
                    te = tf + hs
                    ae = np.searchsorted(pt, te, side="right")
                    rec = lambda rule, x, how: out.append((int(rid[j]), side, hn, rule, float(lim), x, how))   # noqa: E731
                    # fixed: last print at te -+ 1, through-fill within EXIT_S, else take
                    ref_e = px[ae - 1] if ae > 0 else ref
                    lx = ref_e + K if side == 1 else ref_e - K
                    be = np.searchsorted(pt, te + EXIT_S, side="right")
                    fx = first_through(px, yes, pt, ae, be, side_taker_yes=(side == 1), price=lx, above=(side == 1)) \
                        if 0 < lx < 100 else -1
                    if fx >= 0:
                        rec("fixed", float(lx), "limit filled")
                    else:
                        rec("fixed", *take_after(pt, px, yes, side, te + EXIT_S, ys[j]))
                    # mid-following with a deadline
                    for X in DEADLINES:
                        got = chase(pt, px, yes, side, ae, te + X)
                        if got is not None:
                            rec(f"mid_{X}", got[0], "limit filled")
                        else:
                            rec(f"mid_{X}", *take_after(pt, px, yes, side, te + X, ys[j]))
    return pl.DataFrame(out, schema=["row", "side", "hold", "exit_rule", "entry", "exit", "how"], orient="row")


def main() -> None:
    pl.Config.set_tbl_rows(200)
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
    # signal at each hold: the model's level at that horizon (60 s if it has none) minus ref
    r = r.with_columns([(pl.col(f"level_{hn}") if f"level_{hn}" in r.columns else pl.col("level_60s")).sub(pl.col("ref")).alias(f"sig_{hn}")
                        for hn in HOLDS])
    n_dec = r.group_by("kind").len()
    r = r.filter(pl.max_horizontal(*[pl.col(f"sig_{hn}").abs() for hn in HOLDS]) >= MARGINS[0])
    f = simulate(r, tr).join(r.select("row", "kind", "game_id", *[f"sig_{hn}" for hn in HOLDS]), on="row")
    f = f.with_columns(pl.coalesce(*[pl.when(pl.col("hold") == hn).then(pl.col(f"sig_{hn}")) for hn in HOLDS]).alias("signal"))
    q = lambda c: (pl.col(c) / 100).clip(0, 1)   # noqa: E731
    fee_out = (pl.when(pl.col("how") == "taken").then(FEE_RATE * q("exit") * (1 - q("exit")))
               .when(pl.col("how") == "limit filled").then(MAKER_RATE * q("exit") * (1 - q("exit"))).otherwise(0.0))
    f = f.with_columns((pl.col("side") * (pl.col("exit") - pl.col("entry"))).alias("gross"))
    f = f.with_columns((pl.col("gross") - MAKER_RATE * q("entry") * (1 - q("entry")) - fee_out).alias("net"))
    print(f"{LEAGUE}, feed delay {DELAY:g} s: {r.height:,} contract-decisions with |signal| >= {MARGINS[0]:g}")
    res = []
    for (kind, hold, rule), g0 in f.group_by("kind", "hold", "exit_rule"):
        nd = n_dec.filter(pl.col("kind") == kind)["len"].item()
        for m in MARGINS:
            g = g0.filter(pl.col("side") * pl.col("signal") >= m)
            if g.height < 30:
                continue
            gm = g["game_id"].to_numpy()
            res.append(dict(kind=kind, hold=hold, exit=rule, margin=m, trades=g.height,
                            gross=round(g["gross"].mean(), 2), net=round(g["net"].mean(), 2),
                            t_net=round(clustered_t(g["net"].to_numpy(), gm), 1),
                            exit_limit_filled=round((g["how"] == "limit filled").mean(), 2),
                            taken=round((g["how"] == "taken").mean(), 2), settled=round((g["how"] == "settled").mean(), 3),
                            net_settled=round(g.filter(pl.col("how") == "settled")["net"].mean() or 0.0, 2),
                            net_per_100_decisions=round(100 * g["net"].sum() / nd, 2)))
    out = pl.DataFrame(res).sort("kind", "hold", "margin", "exit")
    out.write_parquet(Path(__file__).resolve().parent / "out" / f"b6_maker_mid_{LEAGUE}_d{DELAY:g}{RUN_TAG}.parquet")
    print("cents per trade (one contract)")
    print(out)


if __name__ == "__main__":
    main()
