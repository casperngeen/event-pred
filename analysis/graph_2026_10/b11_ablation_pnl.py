"""Model ablations through the frozen maker strategy: does removing the play memory or the message passing
between contracts change trading P&L, not just the forecast scores?

Versions (all feed delay 10 s, horizons 30/60/120/1800 s, traded-price panel with futures, 5 test folds):
    full        GRU play memory + 2 rounds of message passing, seed 0 (the model behind b9)
    full_seed1  the same, seed 1: how much P&L moves from the seed alone
    no_graph    no message passing (--rounds 0)
    no_memory   play memory replaced by zeros (--memory none)
    neither     both removed
Strategy as b9_cutoff_grid_all.py: fixed exit, offset 1c, strictly-through fills, frozen sizing
"quadratic signal x linear width (missing 1)" (and the hard >= 8c cut-off for reference). Net cents per
contract with the game-clustered t, per-fold net, and the per-game P&L difference against full (paired t).

Trade logs come from b7_maker_trades.py with MEMORY / SEED / RUN_TAG set per version, e.g.
    MEMORY=none RUN_TAG=_h30-60-120-1800_tpf_r0 venv/bin/python -W ignore analysis/graph_2026_10/b7_maker_trades.py \
        results/ingame NBA 10 data/sports/derived/ingame/NBA/panel_v3f

    venv/bin/python -W ignore analysis/graph_2026_10/b11_ablation_pnl.py [gap | seeds]
    -> out/b11_ablation_pnl.txt   (gap: out/b11_gap_pnl.txt, seeds: out/b11_seeds_pnl.txt)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

MODE = sys.argv[1] if len(sys.argv) > 1 else "ablation"   # read before the imports below, which rewrite sys.argv
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_sizing import RULES, game_t                     # noqa: E402

TAG = "_h30-60-120-1800_tpf"
VERSIONS = {"full": "", "full_seed1": "_seed1", "no_graph": "_r0", "no_memory": "_none", "neither": "_r0_none"}
# `gap`: the full model with the ladder gap added (b12_ladder_bounce.py), folds 2-5 only (fold 1 has no slope)
if MODE == "gap":
    VERSIONS = {"full": "", "full_seed1": "_seed1", "full + gap (trades)": "_gap", "full + gap (mids)": "_gapmid"}
# `seeds`: the five seeds of the full model and the average of their forecasts (b13_seed_ensemble.py)
if MODE == "seeds":
    VERSIONS = {"full": "", "full_seed1": "_seed1", "full_seed2": "_seed2", "full_seed3": "_seed3", "full_seed4": "_seed4",
                "average of 5": "_ens5"}
# `nofutures`: the simpler model without the futures contracts in the graph (panel_v3, b14_futures_ablation.py)
# against the model with them, per seed and for the average of the five seeds; diffs are against the with-futures average
BASE = "full"
if MODE == "nofutures":
    TAG = "_h30-60-120-1800_tp"
    VERSIONS = {"with futures, average of 5": "f_ens5"} | {f"with futures, seed{s}": "f" + (f"_seed{s}" if s else "") for s in range(5)} \
        | {"without, average of 5": "_ens5"} | {f"without, seed{s}": f"_seed{s}" if s else "" for s in range(5)}
    BASE = "with futures, average of 5"
STRATS = [("NBA", "spread", "120s"), ("MLB", "total", "120s"), ("NBA", "total", "120s"), ("NFL", "total", "120s"),
          ("MLB", "spread", "120s"), ("NFL", "spread", "120s")]     # last two: dropped candidates, for reference
SIZING = {"frozen sizing": RULES["quadratic signal x linear width (missing 1)"], "hard >= 8c": RULES["hard: signal >= 8, any width"]}


def per_game(g: pl.DataFrame, games: pl.Series) -> np.ndarray:
    return (pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left")
            .fill_null(0).sort("game_id")["pnl"].to_numpy())


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    lines = ["Ablations through the maker strategy, feed delay 10 s, all test folds. per_contract = net cents per contract "
             "(size-weighted), t = game-clustered; diff_vs_full = per-game P&L minus the base version's (full; nofutures: with futures, average of 5), summed over games (paired t)."]
    for L in sorted({s[0] for s in STRATS}):
        logs = {}
        for v, suf in VERSIONS.items():
            f = pl.read_parquet(OUT / f"b7_maker_trades_{L}_d10{TAG}{suf}.parquet").filter(pl.col("k") == 1, pl.col("through") == 0)
            if MODE == "gap":
                f = f.filter(pl.col("fold") >= 2)
            logs[v] = liquidity(with_net(f))
        for _, kind, hold in [s for s in STRATS if s[0] == L]:
            lines.append(f"\n================ {L} {kind}, hold {hold}")
            for sname, size in SIZING.items():
                xs = {v: x.filter(pl.col("kind") == kind, pl.col("hold") == hold) for v, x in logs.items()}
                games = xs[BASE]["game_id"].unique().sort()       # decision games are the same for every version
                gs = {v: x.with_columns(size.alias("size")).filter(pl.col("size") > 0).with_columns((pl.col("size") * pl.col("net")).alias("pnl"))
                      for v, x in xs.items()}
                base = per_game(gs[BASE], games)
                rows = []
                for v, g in gs.items():
                    c = g["size"].sum()
                    pg = per_game(g, games)
                    d = pg - base
                    r = dict(version=v, trades=g.height, total_size=round(c), gross_per_contract=round((g["size"] * g["gross"]).sum() / c, 2) if c else None, per_contract=round(g["pnl"].sum() / c, 2) if c else None,
                             t=round(game_t(g, games), 1) if c else None, total_pnl=round(g["pnl"].sum()),
                             diff_vs_full=round(d.sum()), diff_t=round(d.mean() / d.std(ddof=1) * np.sqrt(len(d)), 1) if v != BASE else None)
                    for k in sorted(g["fold"].unique().to_list()):
                        h = g.filter(pl.col("fold") == k)
                        r[f"fold{k}"] = round(h["pnl"].sum() / h["size"].sum(), 1) if h["size"].sum() else None
                    rows.append(r)
                lines += [f"-- {sname}", str(pl.DataFrame(rows))]
    (OUT / f"b11_{MODE}_pnl.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
