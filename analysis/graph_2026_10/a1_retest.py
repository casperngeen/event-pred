"""A1: out-of-sample re-test of the proposal's changes (reports/graph_learning_proposal.md) before
the graph build: live single-leg futures, wins excluded; log-odds (main) and cents.

Training is pooled across leagues: each league's test fold (its own nights, 5 folds after the first
40%) trains on every league's rows whose labels ended before the fold starts; the ridge penalty is
chosen on those training rows only. Specs, relation model (total_h):

    P0  pooled 4b: league dummies, z x league, x_rival x league, shared shape terms
        (pq, frac, z x pq, z x frac, x_rival x pq, x_rival x frac)
    P1  rivals by proximity (A2): x_rival replaced by x_division x league (+ x pq, x frac) and the
        price-weighted wide rivals p x x_wide (conference / playoffs: rest of the conference;
        champ: everyone outside the division)
    P2  P1 + margin surprise (A4): ms x league, ms x pq (MLB: price-implied margin)
    P3  P2 + x_meet: rivals that meet the team within 7 days of the decision
    4b  the per-league 4b model (unpooled), for reference

Trading model (rem_12h): T0 = P2 + moved; T1 = T0 + the same team's champ and division moved
(the leader inputs, A6).

    venv/bin/python -W ignore analysis/graph_2026_10/a1_retest.py > analysis/graph_2026_10/out/a1_retest.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stg.sports.asof import SportsData                                      # noqa: E402
from stg.sports.graph import between_game_graph                             # noqa: E402
from stg.sports.nextgame import night_shocks                                # noqa: E402
from stg.sports.walkforward import choose_lam, night_folds, oos_scores, ridge_fit, walk_forward  # noqa: E402
from linear_weights import load as load_live                                # noqa: E402
from linear_weights_4b import GRID, specs as specs_4b, with_state           # noqa: E402

RUNS = [("MLB", 2025), ("NBA", 2025), ("NFL", 2025)]
LEAGUES = ["MLB", "NBA", "NFL"]
KINDS = ["division", "playoffs", "conference", "champ"]
OUT = Path(__file__).resolve().parent / "out"


def build(d: SportsData) -> pl.DataFrame:
    s = with_state(load_live()).filter(pl.col("kind") != "wins")
    edges = pl.read_parquet("data/sports/derived/edges.parquet")
    mem, sh, meet = [], [], []
    for lg, season in RUNS:
        g = between_game_graph(d, edges, lg, season)
        mem.append(g.nodes.group_by("team").agg(pl.col("conf").drop_nulls().first(), pl.col("division").drop_nulls().first())
                   .with_columns(pl.lit(lg).alias("league")))
        t, _ = night_shocks(d, lg, season)
        sh.append(t.with_columns(pl.lit(lg).alias("league")))
        gm = (d.contracts.filter(pl.col("league") == lg, pl.col("kind") == "winner", pl.col("game_id").is_not_null())
              .group_by("game_id").agg(pl.col("home").first(), pl.col("away").first(), pl.col("sched_start").first()))
        meet.append(pl.concat([gm.select(pl.col("home").alias("team"), pl.col("away").alias("rival"), "sched_start"),
                               gm.select(pl.col("away").alias("team"), pl.col("home").alias("rival"), "sched_start")])
                    .with_columns(pl.lit(lg).alias("league")))
    mem, sh, meet = pl.concat(mem), pl.concat(sh, how="diagonal_relaxed"), pl.concat(meet)

    # per row, every other team of the league that played tonight, with its proximity class
    zt = sh.join(mem, on=["league", "team"], how="left").rename({"team": "rival", "conf": "r_conf", "division": "r_div", "z": "rz"})
    rows = s.join(mem, on=["league", "team"], how="left")
    pr = (rows.select("league", "node", "night", "team", "kind", "conf", "division", "t")
          .join(zt.select("league", "night", "rival", "r_conf", "r_div", "rz"), on=["league", "night"])
          .filter(pl.col("rival") != pl.col("team"))
          .with_columns((pl.col("division").is_not_null() & (pl.col("r_div") == pl.col("division"))).alias("is_div"),
                        (pl.col("r_conf") == pl.col("conf")).alias("same_conf")))
    wide = (pl.when(pl.col("kind") == "champ").then(~pl.col("is_div"))
            .when(pl.col("kind").is_in(["conference", "playoffs"])).then(pl.col("same_conf") & ~pl.col("is_div"))
            .otherwise(False))
    soon = (pr.select("league", "node", "night", "team", "rival", "t").join(meet, on=["league", "team", "rival"])
            .filter(pl.col("sched_start") > pl.col("t"), pl.col("sched_start") <= pl.col("t") + pl.duration(days=7))
            .select("node", "night", "rival").unique().with_columns(pl.lit(True).alias("meets")))
    pr = pr.join(soon, on=["node", "night", "rival"], how="left").with_columns(pl.col("meets").fill_null(False))
    ex = pr.group_by("node", "night").agg(
        pl.col("rz").filter(pl.col("is_div")).sum().alias("x_division"),
        pl.col("rz").filter(wide).sum().alias("x_wide"),
        pl.col("rz").filter(pl.col("meets") & (pl.col("is_div") | pl.col("same_conf"))).sum().alias("x_meet"))
    rows = (rows.join(ex, on=["node", "night"], how="left")
            .join(sh.select("league", "night", "team", pl.col("ms").alias("ms_own")), on=["league", "night", "team"], how="left")
            .with_columns(pl.col("x_division", "x_wide", "x_meet").fill_null(0.0),
                          pl.col("ms_own").fill_null(0.0),
                          ((pl.col("value_pre") / 100) * pl.col("x_wide").fill_null(0.0)).alias("x_wide_pw")))
    # leader inputs: the same team's champ / division moved (cents and log-odds)
    lead = s.select("league", "team", "night", "kind", "moved", "lo_moved")
    for k in ("champ", "division"):
        lk = lead.filter(pl.col("kind") == k).select("league", "team", "night", pl.col("moved").alias(f"lead_{k}"),
                                                       pl.col("lo_moved").alias(f"lo_lead_{k}"))
        rows = rows.join(lk, on=["league", "team", "night"], how="left").with_columns(
            pl.when(pl.col("kind") == k).then(0.0).otherwise(pl.col(f"lead_{k}")).fill_null(0.0).alias(f"lead_{k}"),
            pl.when(pl.col("kind") == k).then(0.0).otherwise(pl.col(f"lo_lead_{k}")).fill_null(0.0).alias(f"lo_lead_{k}"))
    for lg in LEAGUES:
        L = (pl.col("league") == lg).cast(pl.Float64)
        rows = rows.with_columns(L.alias(f"L_{lg}"), (pl.col("z_own") * L).alias(f"z*{lg}"),
                                 (pl.col("x_rival") * L).alias(f"x_rival*{lg}"),
                                 (pl.col("x_division") * L).alias(f"x_division*{lg}"),
                                 (pl.col("ms_own") * L).alias(f"ms*{lg}"))
    for f in ("pq", "frac_remaining"):
        rows = rows.with_columns((pl.col("x_division") * pl.col(f)).alias(f"x_division*{f}"))
    return rows.with_columns((pl.col("ms_own") * pl.col("pq")).alias("ms*pq"))


def spec_cols(rows: pl.DataFrame, kind: str) -> dict[str, list[str]]:
    lgs = [lg for lg in LEAGUES if rows.filter(pl.col("kind") == kind, pl.col("league") == lg).height]
    dummies = [f"L_{lg}" for lg in lgs[1:]]
    shape = ["pq", "frac_remaining", "z_own*pq", "z_own*frac_remaining"]
    rival_old = ([f"x_rival*{lg}" for lg in lgs] + ["x_rival*pq", "x_rival*frac_remaining"]) if kind != "champ" else []
    div_lgs = [lg for lg in lgs if lg != "NBA"]
    rival_new = ([f"x_division*{lg}" for lg in div_lgs] + (["x_division*pq", "x_division*frac_remaining"] if div_lgs else [])
                 + (["x_wide_pw"] if kind != "division" else []))
    p0 = dummies + [f"z*{lg}" for lg in lgs] + shape + rival_old
    p1 = dummies + [f"z*{lg}" for lg in lgs] + shape + rival_new
    p2 = p1 + [f"ms*{lg}" for lg in lgs] + ["ms*pq"]
    return {"P0": p0, "P1": p1, "P2": p2, "P3": p2 + ["x_meet"]}


def pooled_wf(df: pl.DataFrame, feats: list[str], label: str, label_end: str, league: str) -> pl.DataFrame:
    d = df.filter(pl.col(label).is_not_null())
    tgt = d.filter(pl.col("league") == league)
    preds = []
    for nights in night_folds(tgt["night"].unique().to_list()):
        test = tgt.filter(pl.col("night").is_in(nights))
        train = d.filter(pl.col(label_end) < test["t"].min())
        if train.height < max(60, 5 * len(feats)):
            continue
        lam = choose_lam(train, feats, label, label_end, GRID)
        X, y = train.select(feats).to_numpy().astype(float), train[label].to_numpy().astype(float)
        beta, mu, sd, b0 = ridge_fit(X, y, lam)
        Xt = test.select(feats).to_numpy().astype(float)
        preds.append(test.with_columns(pl.Series("pred", (Xt - mu) / sd @ beta + b0)))
    return pl.concat(preds) if preds else d.head(0).with_columns(pl.lit(None, pl.Float64).alias("pred"))


def main() -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    rows = build(SportsData())
    out = []
    for kind in KINDS:
        k = rows.filter(pl.col("kind") == kind)
        sp = spec_cols(rows, kind)
        for units, pre in (("logodds", "lo_"), ("cents", "")):
            for h in ("12h", "24h"):
                lab = f"{pre}total_{h}"
                for lg in k["league"].unique().to_list():
                    r = {"kind": kind, "league": lg, "units": units, "h": h}
                    g4 = k.filter(pl.col("league") == lg)
                    p, _ = walk_forward(g4, specs_4b(g4, kind, False)["4b"], lab, f"label_end_{h}", lam=GRID)
                    r["4b_perleague"] = round(oos_scores(p, lab).get("r2_vs_zero", np.nan), 3)
                    for name, feats in sp.items():
                        r[name] = round(oos_scores(pooled_wf(k, feats, lab, f"label_end_{h}", lg), lab).get("r2_vs_zero", np.nan), 3)
                    out.append(r)
    res = pl.DataFrame(out)
    print(__doc__.split("\n\n")[1], "\n")
    for units in ("logodds", "cents"):
        print(f"######## relation model, {units}: OOS R^2 vs zero")
        print(res.filter(pl.col("units") == units).drop("units").sort("kind", "league", "h"))
        print()

    tr = []
    for kind in KINDS:
        k = rows.filter(pl.col("kind") == kind)
        p2 = spec_cols(rows, kind)["P2"]
        for units, pre in (("logodds", "lo_"), ("cents", "")):
            mv = f"{pre}moved"
            leads = [f"{pre}lead_{x}" for x in ("champ", "division") if x != kind]
            for lg in k["league"].unique().to_list():
                lab = f"{pre}rem_12h"
                a = oos_scores(pooled_wf(k, p2 + [mv], lab, "label_end_12h", lg), lab)
                b = oos_scores(pooled_wf(k, p2 + [mv] + leads, lab, "label_end_12h", lg), lab)
                tr.append({"kind": kind, "league": lg, "units": units, "T0": round(a.get("r2_vs_zero", np.nan), 3),
                           "T1_with_leaders": round(b.get("r2_vs_zero", np.nan), 3)})
    print("######## trading model (rem_12h): OOS R^2 vs zero, without / with the leader inputs")
    print(pl.DataFrame(tr).sort("units", "kind", "league"))
    res.write_parquet(OUT / "a1_retest.parquet")


if __name__ == "__main__":
    main()
