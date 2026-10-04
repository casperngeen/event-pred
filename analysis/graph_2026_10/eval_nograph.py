"""No-graph flexible check and the standard evaluation table (before the graph build).

Is there signal a flexible model finds that the per-league 4b leverage form misses? If gradient-
boosted trees on the same inputs do not beat 4b, the graph's gain has to come from structure
(per-pair weights, propagation), not from non-linearity in the inputs 4b already has.

Live single-leg futures (wins excluded), one model per league x kind (per league, as decided after
A1), the same purged walk-forward folds for all:

    own    4b without the rival terms: z_own, state, z_own x state (what the own result alone explains)
    4a     z_own + x_rival
    4b     4a + state + shocks x state (state: p(1-p) before the night, season left, games behind)
    gbm    boosted trees on 4b's raw inputs: z_own, x_rival, p(1-p), season left, games behind, price
    gbm+   gbm + every other input known at t: rival games played, own games, played, spread and
           quote age before the night, win %, streak, last 10, rest days

Trading models (rem_h) add the move already made (moved), and gbm+ the spread and price at t.
Trees: depth 3, learning rate 0.05; trees and leaf size chosen per fold on the training rows'
latest nights (stg.sports.walkforward.inner_split). Scores: stg.sports.scoring (r2 against no
move, r2 on teams that did not play, within-night rank IC, calibration slope, paired-by-night t
against 4b). Trading net: trade when the calibrated |pred| (slope from earlier folds only) beats
spread + both fees (7 p(1-p)); exit at the actual t+h spread; t clustered by night.

    venv/bin/python -W ignore analysis/graph_2026_10/eval_nograph.py > analysis/graph_2026_10/out/eval_nograph.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stg.sports.asof import SportsData                                      # noqa: E402
from stg.sports.scoring import KEY, score                                   # noqa: E402
from stg.sports.walkforward import inner_split, walk_forward, walk_forward_fn  # noqa: E402
from checks_4a import exits                                                 # noqa: E402
from linear_weights import load                                             # noqa: E402
from linear_weights_4b import GRID, specs, with_state                       # noqa: E402

HS = ("12h", "24h")
FEE = 7.0
GBM_GRID = [{"max_iter": m, "min_samples_leaf": leaf} for m in (50, 150, 400) for leaf in (20, 60)]
EXTRA = ["n_rival_games", "n_games", "played_f", "spread_pre", "one_sided_pre_f", "age_pre_h", "win_pct",
         "streak", "last10", "days_since_last_game"]
EXTRA_T = ["value_t", "spread_t", "age_t_h"]
OUT = Path(__file__).resolve().parent / "out"


def prep() -> pl.DataFrame:
    s = with_state(load()).filter(pl.col("kind") != "wins")
    return s.with_columns(pl.col("played").cast(pl.Float64).alias("played_f"),
                          pl.col("one_sided_pre").cast(pl.Float64).alias("one_sided_pre_f"),
                          (pl.col("quote_age_pre").dt.total_minutes() / 60).alias("age_pre_h"),
                          (pl.col("quote_age_t").dt.total_minutes() / 60).alias("age_t_h"),
                          pl.col("n_rival_games").cast(pl.Float64), pl.col("n_games").cast(pl.Float64))


def gbm(feats: list[str], label: str, label_end: str):
    def X(rows: pl.DataFrame) -> np.ndarray:
        return rows.select(feats).to_numpy().astype(float)

    def make(params: dict, rows: pl.DataFrame):
        return HistGradientBoostingRegressor(learning_rate=0.05, max_depth=3, l2_regularization=1.0,
                                             random_state=0, **params).fit(X(rows), rows[label].to_numpy())

    def fit(train: pl.DataFrame):
        f, v = inner_split(train, label_end)
        best = GBM_GRID[len(GBM_GRID) // 2]
        if f.height >= 100 and v.height:
            yv = v[label].to_numpy()
            best = min(GBM_GRID, key=lambda prm: float(((make(prm, f).predict(X(v)) - yv) ** 2).sum()))
        m = make(best, train)
        return lambda rows: m.predict(X(rows))
    return fit


def models(g: pl.DataFrame, kind: str, label: str, label_end: str, trading: bool) -> dict[str, pl.DataFrame]:
    sp = specs(g, kind, trading)
    rival = "x_rival" in sp["4a"]
    own = [f for f in sp["4b"] if "x_rival" not in f]
    raw = ["z_own"] + (["x_rival"] if rival else []) + ["pq", "frac_remaining", "games_behind", "value_pre"] \
        + (["moved"] if trading else [])
    plus = raw + EXTRA + (EXTRA_T if trading else [])
    out = {name: walk_forward(g, feats, label, label_end, lam=GRID)[0]
           for name, feats in (("own", own), ("4a", sp["4a"]), ("4b", sp["4b"]))}
    out["gbm"] = walk_forward_fn(g, label, label_end, gbm(raw, label, label_end))
    out["gbm+"] = walk_forward_fn(g, label, label_end, gbm(plus, label, label_end))
    return out


def calibrated(p: pl.DataFrame, label: str) -> pl.DataFrame:
    """Scale each fold's predictions by the slope of y on pred over the earlier folds only."""
    out = []
    for f in sorted(p["fold"].unique().to_list()):
        prev, cur = p.filter(pl.col("fold") < f), p.filter(pl.col("fold") == f)
        c = None
        if prev.height >= 30 and prev["pred"].std() > 0:
            x, y = prev["pred"].to_numpy(), prev[label].to_numpy()
            c = float(np.clip((x @ y) / (x @ x), 0, 2))
        out.append(cur.with_columns(pl.lit(c, pl.Float64).alias("calib_c")))
    return pl.concat(out)


def trade_net(p: pl.DataFrame, h: str, spreads: pl.DataFrame) -> dict:
    fee = lambda x: FEE * (x / 100) * (1 - x / 100)                          # noqa: E731
    mid_h = pl.col("value_t") + pl.col(f"rem_{h}")
    p = (calibrated(p, f"rem_{h}").join(spreads, on=KEY, how="left")
         .filter(pl.col("calib_c").is_not_null(), pl.col("spread_exit").is_not_null(), ~pl.col("one_sided_exit"),
                 pl.col("calib_c") * pl.col("pred").abs() > pl.col("spread_t") + 2 * fee(pl.col("value_t")))
         .with_columns(pl.when(pl.col("pred") > 0)
                       .then(mid_h - pl.col("spread_exit") / 2 - pl.col("value_t") - pl.col("spread_t") / 2)
                       .otherwise(pl.col("value_t") - pl.col("spread_t") / 2 - mid_h - pl.col("spread_exit") / 2)
                       .sub(fee(pl.col("value_t")) + fee(mid_h)).alias("net")))
    if p.height == 0:
        return {"trades": 0}
    n = p.group_by("night").agg(pl.col("net").sum())["net"].to_numpy()
    se = n.std(ddof=1) * np.sqrt(len(n)) / p.height if len(n) > 2 else np.nan
    return {"trades": p.height, "net": float(p["net"].mean()), "net_t": float(p["net"].mean() / se) if se > 0 else np.nan}


def fmt(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.col(c).round(3) for c, t in df.schema.items() if t == pl.Float64])


def main() -> None:
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    d = SportsData()
    s = prep()
    spreads = {h: exits(d, s.select("league", "season", *KEY, f"label_end_{h}"), h)
               .select(*KEY, "spread_exit", pl.col("one_sided_exit").fill_null(True)) for h in HS}
    rows = []
    for (lg, kind), g in sorted(s.group_by("league", "kind"), key=lambda x: x[0]):
        for h in HS:
            for task, label, trading in (("relation", f"total_{h}", False), ("relation_lo", f"lo_total_{h}", False),
                                         ("trading", f"rem_{h}", True)):
                m = models(g, kind, label, f"label_end_{h}", trading)
                for name, p in m.items():
                    r = {"league": lg, "kind": kind, "h": h, "task": task, "model": name,
                         **score(p, label, base=m["4b"] if name != "4b" else None)}
                    if task == "trading":
                        r |= trade_net(p, h, spreads[h])
                    rows.append(r)
            print(f"done {lg} {kind} {h}", file=sys.stderr)
    sc = pl.DataFrame(rows, infer_schema_length=None)
    sc = sc.join(sc.filter(pl.col("model") == "own").select("league", "kind", "h", "task", pl.col("r2").alias("r2_own")),
                 on=["league", "kind", "h", "task"]).with_columns((pl.col("r2") - pl.col("r2_own")).alias("d_own"))
    OUT.mkdir(exist_ok=True)
    sc.write_parquet(OUT / "eval_nograph.parquet")

    print(__doc__.split("\n\n")[1], "\n")
    order = {m: i for i, m in enumerate(("own", "4a", "4b", "gbm", "gbm+"))}
    srt = lambda x: x.with_columns(pl.col("model").replace_strict(order).alias("_m")).sort("league", "kind", "h", "_m").drop("_m")  # noqa: E731

    print("######## 1. relation (total_h, cents): r2 vs no move; d_own = gain over own-shock only; r2 on teams that")
    print("         did not play; within-night rank IC (t); calibration slope; R^2 gain vs 4b with its paired-by-night t")
    print(fmt(srt(sc.filter(pl.col("task") == "relation")).select(
        "league", "kind", "h", "model", "n", "r2", "d_own", "n_quiet", "r2_quiet", "ic", "ic_t", "calib",
        "gain_vs_base", "t_vs_base")))

    print("\n######## 2. relation in log-odds: r2 and gain vs 4b (t)")
    print(fmt(srt(sc.filter(pl.col("task") == "relation_lo")).select(
        "league", "kind", "h", "model", "r2", "d_own", "r2_quiet", "ic", "calib", "gain_vs_base", "t_vs_base")))

    print("\n######## 3. trading (rem_h, cents): fit, ranking, calibration; trades where calibrated edge > cost, net cents (t)")
    print(fmt(srt(sc.filter(pl.col("task") == "trading")).select(
        "league", "kind", "h", "model", "n", "r2", "ic", "ic_t", "calib", "gain_vs_base", "t_vs_base",
        "trades", "net", "net_t")))

    print("\n######## 4. tally of cells (league x kind x horizon) where each model beats / loses to 4b at |t| > 2")
    print(sc.filter(pl.col("model") != "4b").group_by("task", "model").agg(
        pl.len().alias("cells"), (pl.col("t_vs_base") > 2).sum().alias("beats_4b"),
        (pl.col("t_vs_base") < -2).sum().alias("loses"), pl.col("gain_vs_base").median().round(3).alias("median_gain"))
        .sort("task", "model"))


if __name__ == "__main__":
    main()
