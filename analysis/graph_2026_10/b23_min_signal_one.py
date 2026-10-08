"""Minimum signal under the one-at-a-time rules, where a dropped weak order frees the contract for a later, stronger
one (so the trades are re-simulated per minimum, b7_maker_trades.py MIN_SIGNAL=8/10/12, not filtered from the 5c log).

Same setup as b22_league_thresholds.py (120 s hold, offset 1c, strictly-through fills, frozen sizing; MLB: the
every-pitch model) for two decision rules: every 60 s, one at a time; every 5 s, one at a time.
1. Each minimum fixed, all test folds: net per contract, game-level t, win rate.
2. Walk-forward, as b22: the minimum for fold k (k >= 2) picked on folds 1..k-1 (most net dollars), shared 5c / one
   for all / per league / per contract; scored folds 2-5.
Game-level t: per-game net dollars over the games the 5c rule traded (games the stricter rule skips count as 0).

    venv/bin/python -W ignore analysis/graph_2026_10/b23_min_signal_one.py
    -> out/b23_min_signal_one.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b19_decision_timing as B                          # noqa: E402
from b17_holdout_report import exits                      # noqa: E402
from b22_league_thresholds import KINDS, TAGS            # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402

MENU = [5, 8, 10, 12]
RULES = {"every 60 s, one at a time": "_one", "every 5 s, one at a time": "_dec5_one"}
SCHEMES = {"shared 5c": None, "one for all": (), "per league": ("league",), "per contract": ("league", "kind")}


def trades() -> pl.DataFrame:
    parts = []
    for L, tag in TAGS.items():
        B.TAG = tag
        for rule, suf in RULES.items():
            for m in MENU:
                f = B.load(L, suf + (f"_min{m}" if m != 5 else ""))
                if f is None:
                    print("missing", L, rule, m)
                    continue
                for ex, g in exits(f.filter(pl.col("kind").is_in(KINDS))).items():
                    parts.append(g.select(pl.lit(L).alias("league"), "kind", pl.lit(rule).alias("rule"), pl.lit(ex).alias("exit"),
                                          pl.lit(m).alias("min"), "fold", "game_id", "size", "pnl"))
    return pl.concat(parts)


def stats(g: pl.DataFrame, games: pl.Series) -> dict:
    pg = (pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left")
          .fill_null(0)["pnl"].to_numpy())
    return dict(trades=g.height, net_c=round(g["pnl"].sum() / g["size"].sum(), 2) if g.height else None,
                t_game=round(pg.mean() / pg.std(ddof=1) * np.sqrt(len(pg)), 2), win_pct=round(100 * (g["pnl"] > 0).mean(), 1) if g.height else None)


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    t = trades()

    # 1. fixed minimums, all folds
    fixed = []
    for (rule, ex, L, kind), d in t.group_by("rule", "exit", "league", "kind", maintain_order=True):
        games = d.filter(pl.col("min") == 5)["game_id"].unique()
        for m in MENU:
            fixed.append(dict(rule=rule, exit=ex, contract=f"{L} {kind}", min=m) | stats(d.filter(pl.col("min") == m), games))
    for (rule, ex, L), d in t.group_by("rule", "exit", "league", maintain_order=True):
        games = d.filter(pl.col("min") == 5)["game_id"].unique()
        for m in MENU:
            fixed.append(dict(rule=rule, exit=ex, contract=f"{L} both", min=m) | stats(d.filter(pl.col("min") == m), games))
    ro = {r: i for i, r in enumerate(RULES)}
    fx = pl.DataFrame(fixed).with_columns(pl.col("rule").replace_strict(ro).alias("_o")).sort("_o", "exit", "contract", "min").drop("_o")

    # 2. walk-forward
    wf, chosen = [], []
    for (rule, ex), d in t.group_by("rule", "exit", maintain_order=True):
        for k in range(2, 6):
            prior, test = d.filter(pl.col("fold") < k), d.filter(pl.col("fold") == k)
            for name, keys in SCHEMES.items():
                if keys is None:
                    pick = {(L, kn): 5 for L in TAGS for kn in KINDS}
                else:
                    tot = prior.group_by(list(keys) + ["min"]).agg(pl.col("pnl").sum()).sort("pnl", "min", descending=[True, False])
                    best = {tuple(r[c] for c in keys): r["min"] for r in tot.unique(list(keys), keep="first", maintain_order=True).iter_rows(named=True)} \
                        if keys else {(): int(tot["min"][0])}
                    pick = {(L, kn): best.get(tuple({"league": L, "kind": kn}[c] for c in keys), 5) for L in TAGS for kn in KINDS}
                    chosen += [dict(rule=rule, exit=ex, scheme=name, fold=k, group=" ".join(g) or "all", min=m) for g, m in best.items()]
                for (L, kn), m in pick.items():
                    wf.append(test.filter(pl.col("league") == L, pl.col("kind") == kn, pl.col("min") == m).with_columns(pl.lit(name).alias("scheme")))
    w = pl.concat(wf)
    base = t.filter(pl.col("min") == 5, pl.col("fold") >= 2)
    rows = []
    for by in (["league"], ["league", "kind"]):
        for key, d in w.group_by(["rule", "exit"] + by, maintain_order=True):
            games = base.filter(*[pl.col(c) == v for c, v in zip(["rule", "exit"] + by, key)])["game_id"].unique()
            for name in SCHEMES:
                rows.append(dict(rule=key[0], exit=key[1], contract=" ".join(key[2:]) + ("" if len(by) == 2 else " both"), scheme=name)
                            | stats(d.filter(pl.col("scheme") == name), games))
    so = {n: i for i, n in enumerate(SCHEMES)}
    wt = (pl.DataFrame(rows).with_columns(pl.col("rule").replace_strict(ro).alias("_o"), pl.col("scheme").replace_strict(so).alias("_s"))
          .sort("_o", "exit", "contract", "_s").drop("_o", "_s"))
    ch = pl.DataFrame(chosen).pivot(on="fold", index=["rule", "exit", "scheme", "group"], values="min").sort("rule", "exit", "scheme", "group")
    lines = [__doc__.split("\n\n")[0], "", "Net in cents per contract after fees; t = game-level t.",
             "", "-- 1. each minimum fixed, all test folds", str(fx),
             "", "-- 2. walk-forward, scored folds 2-5", str(wt),
             "", "-- minimum chosen per scored fold", str(ch)]
    (OUT / "b23_min_signal_one.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
