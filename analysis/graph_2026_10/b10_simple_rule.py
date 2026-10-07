"""No-model baseline: does a simple market rule trade as well as the graph model? Same decisions (every
60 s, contracts with a print in the last 60 s, the model runs' rows and test folds), same maker logic
(b7_maker_trades.simulate: post 1c inside the last print on the chosen side, strictly-through fills,
fixed exit after 120 s), same fees (b7_maker_robust.with_net). Feed delay 10 s rows (the rule uses no
plays, so the delay does not matter for it).

Rule "fade the last trade": signal = trade-based mid - last print, mid = (last YES-taker price + last
NO-taker price) / 2. A YES-taker buys at the ask, so after one the mid is below the last print: sell, by
half the width; after a NO-taker: buy. Two versions: both sides at most 120 s old (the model's width
measure), or the other side at any age. Sizes: hard width cut-offs, and size = width / 10 (capped at 1).

Compared with the model's frozen sizing (b9_sizing.py: ((|signal| - 5) / 10)^2 x width / 10, missing
width = 1) on t of the mean P&L per game over all games with a decision, and net per contract. Also: how
often the model's side is the rule's side, and the model's P&L when it agrees / disagrees.

    venv/bin/python -W ignore analysis/graph_2026_10/b10_simple_rule.py
    -> out/b10_simple_rule.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

os.environ.setdefault("MIN_SIGNAL", "0.5")
sys.argv = ["x", "results/ingame", "NBA", "10"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b5_longshort as B                                    # noqa: E402
import b7_maker_trades as T                                 # noqa: E402
from b7_maker_liquidity import liquidity                   # noqa: E402
from b7_maker_robust import with_net, OUT                  # noqa: E402
from b9_sizing import RULES                                 # noqa: E402
from stg.sports.loaders import derived_dir, scan_trades     # noqa: E402

CANDS = [("NBA", "spread"), ("NBA", "total"), ("MLB", "total"), ("NFL", "total")]
TAG = "_h30-60-120-1800_tpf"
MODEL_SIZE = RULES["quadratic signal x linear width (missing 1)"]
W = pl.col("width")
RULE_SIZES = {"width >= 5": (W >= 5).cast(pl.Float64), "width >= 9": (W >= 9).cast(pl.Float64),
              "width >= 15": (W >= 15).cast(pl.Float64), "width >= 20": (W >= 20).cast(pl.Float64),
              "size = width / 10": (W / 10).clip(0, 1)}
T.HOLDS = {"120s": 120}
T.VARIANTS = [(1, 0)]


def game_t(g: pl.DataFrame, games: pl.Series) -> float:
    per = pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left").fill_null(0)["pnl"].to_numpy()
    return float(per.mean() / per.std(ddof=1) * np.sqrt(len(per)))


def summary(g: pl.DataFrame, games: pl.Series, **kw) -> dict:
    g = g.filter(pl.col("size") > 0).with_columns((pl.col("size") * pl.col("net")).alias("pnl"))
    c = g["size"].sum()
    folds = {f"fold{k}": round(x["pnl"].sum() / x["size"].sum(), 1) for (k,), x in sorted(g.group_by("fold"), key=lambda z: z[0])}
    return kw | dict(trades=g.height, total_size=round(c), per_contract=round(g["pnl"].sum() / c, 2) if c else None,
                     t=round(game_t(g, games), 1) if c else None, stuck=round((g["how"] == "settled").mean(), 2) if c else None) | folds


def last_sides(r: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    yes = tr.filter(pl.col("taker_side") == "yes").select("ticker", "t", pl.col("t").alias("t_yes"), pl.col("yes_price").alias("p_yes")).sort("t", "p_yes")
    no = tr.filter(pl.col("taker_side") == "no").select("ticker", "t", pl.col("t").alias("t_no"), pl.col("yes_price").alias("p_no")).sort("t", "p_no")
    last = tr.select("ticker", "t", pl.col("taker_side").alias("last_taker")).sort("t", "last_taker")
    r = (r.sort("t").join_asof(yes, on="t", by="ticker", strategy="backward").join_asof(no, on="t", by="ticker", strategy="backward")
         .join_asof(last, on="t", by="ticker", strategy="backward"))
    fresh = ((pl.col("t") - pl.col("t_yes")) <= 120) & ((pl.col("t") - pl.col("t_no")) <= 120)
    return r.with_columns(pl.when(fresh).then(pl.col("p_yes") - pl.col("p_no")).alias("width"),
                          (pl.col("p_yes") - pl.col("p_no")).alias("width_any_age"),
                          pl.when(pl.col("last_taker") == "yes").then(-1).otherwise(1).alias("fade_side"))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_fmt_str_lengths(40)
    lines, agree_rows = [], []
    for L in sorted({c[0] for c in CANDS}):
        B.LEAGUE, B.DELAY, B.RUN_TAG = L, 10.0, TAG
        B.TAB = derived_dir() / "ingame" / L
        B.PANEL = B.TAB / "panel_v3f"
        import json
        pn = B.Panel(B.PANEL)
        gl = json.loads((B.PANEL / "manifest.json").read_text())["games"]
        kinds = [k for l_, k in CANDS if l_ == L]
        r = B.attach_tickers(B.rows(pn, gl)).filter(pl.col("y").is_finite(), pl.col("kind").is_in(kinds)).with_row_index("row")
        tr = (scan_trades(r["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(r["ticker"].unique().implode()))
              .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side")
              .collect().sort("ticker", "t", "yes_price"))
        r = (r.sort("t").join_asof(tr.select("ticker", "t", pl.col("yes_price").alias("ref"), pl.col("t").alias("t_ref")).sort("t", "ref"),
                                   on="t", by="ticker", strategy="backward").filter((pl.col("t") - pl.col("t_ref")) <= T.REF_MAX_S))
        r = last_sides(r, tr)
        g = pl.read_parquet(B.PANEL.parent / "games.parquet").select("game_id", "fold", "first_play")
        model = liquidity(with_net(pl.read_parquet(OUT / f"b7_maker_trades_{L}_d10{TAG}.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s"))).with_columns(MODEL_SIZE.alias("size"))
        model = model.join(r.select("ticker", "t", "fade_side"), on=["ticker", "t"], how="left")
        for kind in kinds:
            rk = r.filter(pl.col("kind") == kind)
            games = rk["game_id"].unique()
            rows = [summary(model.filter(pl.col("kind") == kind), games, rule="MODEL, frozen sizing")]
            for ver, wcol in (("sides <= 120 s", "width"), ("other side any age", "width_any_age")):
                x = rk.filter(pl.col(wcol).is_not_null()).with_columns((-pl.col("fade_side") * 0 + pl.col("fade_side") * pl.col(wcol).abs() / 2).alias("sig"))
                x = x.with_columns(pl.col("sig").alias("sig_120s"), pl.col("sig").alias("sig_60s"), pl.col(wcol).alias("width"))
                x = x.filter(pl.col("width") > 0)
                f = (T.simulate(x, tr).join(x.select("row", "kind", "game_id", "ticker", "t", "ref", "strike", "y", "width"), on="row")
                     .join(g, on="game_id", how="left"))
                f = with_net(f)
                for name, sz in RULE_SIZES.items():
                    rows.append(summary(f.with_columns(sz.alias("size")), games, rule=f"RULE ({ver}), {name}"))
            lines += [f"\n================ {L} {kind}, 120 s, 10 s delay rows, all test folds", str(pl.DataFrame(rows))]
            m = model.filter(pl.col("kind") == kind, pl.col("size") > 0, pl.col("fade_side").is_not_null()).with_columns(
                (pl.col("size") * pl.col("net")).alias("pnl"), (pl.col("side") == pl.col("fade_side")).alias("agree"))
            for a, x in ((True, m.filter(pl.col("agree"))), (False, m.filter(~pl.col("agree")))):
                agree_rows.append(dict(contract=f"{L} {kind}", model_side_is_fade_side=a, trades=x.height,
                                       share_of_size=round(x["size"].sum() / m["size"].sum(), 2),
                                       per_contract=round(x["pnl"].sum() / x["size"].sum(), 2) if x.height else None,
                                       share_of_pnl=round(x["pnl"].sum() / m["pnl"].sum(), 2)))
    text = ("No-model rule vs the graph model (net cents per contract; t = mean P&L per game / standard error over all games with a decision)\n"
            + "\n".join(lines) + "\n\n######## the model's sized trades: on the rule's side (fade the last trade) or not\n" + str(pl.DataFrame(agree_rows)))
    (OUT / "b10_simple_rule.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
