"""Long-short test of the in-game model's signal, priced only from trade prints (no rebuilt quotes).

Every DECIDE_S seconds of game time, for each spread / total / winner contract the model scored:

    reference price    the last trade print at or before the decision (within REF_MAX_S)
    signal             model level at +h (value + predicted change) minus the reference price
    entry price        volume-weighted price of all prints in (T, T + W]       (both taker sides)
    exit price         volume-weighted price of all prints in (T + h, T + h + W]
    return             exit - entry (cents per YES contract); only contracts with prints in both windows

Prints of both taker sides average out to roughly the mid, so "gross" is the move at traded prices;
"net" also takes off half the bid-ask gap at entry and at exit (from the rebuilt quote, capped at 5c)
and the taker fee both ways, i.e. what a taker would roughly pay.

Reports (pooled over the five test folds; t clustered by game):
    deciles      mean return by signal decile; top-minus-bottom long-short; rank correlation
    threshold    trade sign(signal) when |signal| >= M
    pairs        within a game at the same moment, long the highest-signal contract and short the
                 lowest (needs a gap of at least M between them): game-outcome risk largely cancels

    venv/bin/python -W ignore analysis/graph_2026_10/b5_longshort.py results/ingame NFL 10 [panel_dir]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stg.models.ingame import KIND_NAMES, Panel           # noqa: E402
from stg.sports.loaders import derived_dir, scan_trades    # noqa: E402
from b5_backtest import FEE_RATE, clustered_t             # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ingame")
LEAGUE = sys.argv[2] if len(sys.argv) > 2 else "NFL"
DELAY = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
TAB = derived_dir() / "ingame" / LEAGUE
PANEL = Path(sys.argv[4]) if len(sys.argv) > 4 else TAB / "panel_v2"
RUN_TAG = os.environ.get("RUN_TAG", "")   # e.g. "_h30-60-120_tp" for runs on the traded-price panel ("..._tpf_r0": no message passing)
MEMORY = os.environ.get("MEMORY", "gru")  # "none": the no-memory ablation
SEED = int(os.environ.get("SEED", 0))
HOLD, H_S, DECIDE_S, STEP = "60s", 60, 60, 5
W_S, REF_MAX_S = 15, 60
MARGINS = [0.0, 2.0, 3.0, 5.0, 8.0]
SIDE = {"home": 1, "away": -1}


def rows(pn: Panel, games: list[str]) -> pl.DataFrame:
    prs, hz = [], None
    for f in sorted(RES.glob(f"{LEAGUE}_fold*_{MEMORY}_d{DELAY:g}{RUN_TAG}_seed{SEED}.npz")):
        hz = json.loads(f.with_suffix(".json").read_text())["horizons"]
        prs.append(dict(np.load(f)))
    pr = {k: np.concatenate([p[k] for p in prs]) for k in prs[0]}
    h = hz.index(HOLD)
    step = pr["step"]
    # prediction rows are the panel's contract rows of each step, in order
    first = np.r_[True, step[1:] != step[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(step)), 0))
    node = pn.step_node_ptr[step] + (np.arange(len(step)) - start)
    keep = ((step - pn.game_step_ptr[pr["game"]]) % (DECIDE_S // STEP) == 0) & np.isfinite(pr["y"])
    i = np.flatnonzero(keep)
    gap = np.where(pr["has_bid"][i] & pr["has_ask"][i], pr["entry_ask"][i] - pr["entry_bid"][i], 5.0)
    return pl.DataFrame({
        "game_id": np.asarray(games)[pr["game"][i]], "t": pn.steps_t[step[i]].astype(float),
        "kind": np.asarray(KIND_NAMES)[pr["kind"][i]], "side": pn.node_side[node[i]].astype(np.int64),
        "strike": pn.node_strike[node[i]].astype(float), "level": (pr["value"] + pr["delta"][:, h])[i].astype(float),
        "gap": np.clip(gap, 0, 5).astype(float), "y": pr["y"][i].astype(float), "anchor": pr["value"][i].astype(float),
        **{f"level_{hn}": (pr["value"] + pr["delta"][:, hi])[i].astype(float) for hi, hn in enumerate(hz)}})


def attach_tickers(r: pl.DataFrame) -> pl.DataFrame:
    con = pl.read_parquet(TAB / "contracts.parquet").select(
        "ticker", "series_ticker", "game_id", "kind", pl.col("side").replace_strict(SIDE, default=0).cast(pl.Int64),
        pl.when(pl.col("kind") == "winner").then(-1.0).otherwise(pl.col("strike").cast(pl.Float64).round(2)).alias("strike"))
    r = r.with_columns(pl.when(pl.col("kind") == "winner").then(-1.0).otherwise(pl.col("strike").round(2)).alias("strike"))
    dup = con.group_by("game_id", "kind", "side", "strike").len().filter(pl.col("len") > 1).height
    out = r.join(con.unique(["game_id", "kind", "side", "strike"], keep="first"), on=["game_id", "kind", "side", "strike"],
                 how="left", nulls_equal=True)
    print(f"contract match {out['ticker'].is_not_null().mean():.3f} (ambiguous contract keys: {dup}); unmatched by kind:",
          out.filter(pl.col("ticker").is_null()).group_by("kind").len().rows())
    return out.filter(pl.col("ticker").is_not_null())


def print_prices(r: pl.DataFrame) -> pl.DataFrame:
    series = r["series_ticker"].unique().to_list()
    tr = (scan_trades(series).filter(pl.col("ticker").is_in(r["ticker"].unique().implode()))
          .select("ticker", pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6, "yes_price", "count")
          .collect().sort("ticker", "known_at"))
    tr = tr.with_columns(pl.col("count").cum_sum().over("ticker").alias("cv"),
                         (pl.col("count") * pl.col("yes_price")).cum_sum().over("ticker").alias("cn"),
                         pl.col("known_at").alias("t_print"))
    cum = tr.select("ticker", "known_at", "cv", "cn", "yes_price", "t_print")

    def at(df: pl.DataFrame, col: str, pre: str) -> pl.DataFrame:
        j = df.sort(col).join_asof(cum.sort("known_at"), left_on=col, right_on="known_at", by="ticker", strategy="backward")
        return j.rename({"cv": f"{pre}cv", "cn": f"{pre}cn", "yes_price": f"{pre}px", "t_print": f"{pre}tp"}).drop("known_at")

    r = r.with_columns((pl.col("t") + W_S).alias("t_e1"), (pl.col("t") + H_S).alias("t_x0"), (pl.col("t") + H_S + W_S).alias("t_x1"))
    for col, pre in (("t", "e0_"), ("t_e1", "e1_"), ("t_x0", "x0_"), ("t_x1", "x1_")):
        r = at(r, col, pre)
    z = lambda c: pl.col(c).fill_null(0)  # noqa: E731
    r = r.with_columns(
        pl.when(pl.col("t") - pl.col("e0_tp") <= REF_MAX_S).then(pl.col("e0_px")).alias("ref"),
        ((z("e1_cn") - z("e0_cn")) / (z("e1_cv") - z("e0_cv"))).alias("entry"),
        ((z("x1_cn") - z("x0_cn")) / (z("x1_cv") - z("x0_cv"))).alias("exit"))
    return r.filter(pl.col("ref").is_not_null() & pl.col("entry").is_finite() & pl.col("exit").is_finite())


def fee_expr(c: str) -> pl.Expr:
    q = (pl.col(c) / 100).clip(0, 1)
    return FEE_RATE * q * (1 - q)


def stats(x: np.ndarray, g: np.ndarray) -> dict:
    return dict(n=len(x), games=len(np.unique(g)), mean=round(float(x.mean()), 2) if len(x) else None,
                t=round(clustered_t(x, g), 1) if len(x) else None)


def main() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(200)
    pn = Panel(PANEL)
    games = json.loads((PANEL / "manifest.json").read_text())["games"]
    r = print_prices(attach_tickers(rows(pn, games)))
    r = r.with_columns((pl.col("level") - pl.col("ref")).alias("signal"), (pl.col("exit") - pl.col("entry")).alias("ret"),
                       (pl.col("gap") + fee_expr("entry") + fee_expr("exit")).alias("cost"))
    print(f"{LEAGUE}, feed delay {DELAY:g} s, hold {HOLD}: {r.height:,} contract-decisions with prints at entry and exit")
    for kind in ("spread", "total", "winner"):
        k = r.filter(pl.col("kind") == kind)
        if k.height < 200:
            continue
        k = k.with_columns((pl.col("signal").rank("ordinal") * 10 / (pl.len() + 1)).floor().cast(pl.Int64).alias("decile"))
        d = k.group_by("decile").agg(pl.len(), pl.col("signal").mean().round(2), pl.col("ret").mean().round(2)).sort("decile")
        top, bot = k.filter(pl.col("decile") == 9), k.filter(pl.col("decile") == 0)
        rc = k.select(pl.corr("signal", "ret", method="spearman")).item()
        print(f"\n=== {kind}: {k.height:,} decisions, rank corr(signal, return) {rc:.3f}")
        print("deciles of the signal (mean signal, mean return in cents):", d.select("decile", "len", "signal", "ret").rows())
        print(f"long top decile, short bottom: {top['ret'].mean() - bot['ret'].mean():.2f}c per pair "
              f"(top {top['ret'].mean():.2f}, bottom {bot['ret'].mean():.2f})")
        out = []
        for m in MARGINS:
            s = k.filter(pl.col("signal").abs() >= m)
            g = s["game_id"].to_numpy()
            pnl = (np.sign(s["signal"].to_numpy()) * s["ret"].to_numpy())
            out.append(dict(margin=m, **{f"gross_{a}": b for a, b in stats(pnl, g).items()},
                            net=round(float((pnl - s["cost"].to_numpy()).mean()), 2) if len(pnl) else None,
                            t_net=round(clustered_t(pnl - s["cost"].to_numpy(), g), 1) if len(pnl) else None))
        print("trade sign(signal) when |signal| >= margin:")
        print(pl.DataFrame(out))
    # within-game pairs: long the highest signal, short the lowest, at the same decision
    pr_rows = []
    for m in MARGINS:
        p = (r.group_by("game_id", "t").agg(pl.col("signal").max().alias("smax"), pl.col("signal").min().alias("smin"),
                                            pl.col("ret").sort_by("signal").last().alias("r_long"),
                                            pl.col("ret").sort_by("signal").first().alias("r_short"),
                                            pl.col("cost").sort_by("signal").last().alias("c_long"),
                                            pl.col("cost").sort_by("signal").first().alias("c_short"), pl.len())
             .filter(pl.col("len") >= 2, pl.col("smax") > 0, pl.col("smin") < 0, (pl.col("smax") - pl.col("smin")) >= m))
        x = (p["r_long"] - p["r_short"]).to_numpy()
        c = (p["c_long"] + p["c_short"]).to_numpy()
        g = p["game_id"].to_numpy()
        pr_rows.append(dict(min_gap=m, **{f"gross_{a}": b for a, b in stats(x, g).items()},
                            net=round(float((x - c).mean()), 2) if len(x) else None, t_net=round(clustered_t(x - c, g), 1) if len(x) else None))
    print("\n=== within-game pairs (long highest signal, short lowest; both legs; cents per pair)")
    print(pl.DataFrame(pr_rows))


if __name__ == "__main__":
    main()
