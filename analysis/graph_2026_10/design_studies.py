"""Studies that decide how the between-game graph is built (batch A: 1-4; batch B: 5-7).

    1. proportional reallocation   within an exclusive event (champ / conference / division), do the
                                   teams that did not play absorb the players' net move S in proportion
                                   to their prices, dp_j = -S p_j / sum(p), or equally, -S / n?
    2. cross-kind rival effects    a kind's move on its own z, its division rivals' z and its other
                                   conference teams' z (the other conference is the reference: all z
                                   sum to zero)
    3. lead-lag across kinds       (a) share of the 48 h response already in the price at the decision,
                                   1 h, 12 h, 24 h, by kind; (b) does another kind's move by the
                                   decision (same team) predict this kind's remaining move?
    4. margin surprise             the team's margin surprise next to z

In-sample OLS on the sample tables (stg.sports.samples), t clustered by night; descriptive, for design.

    venv/bin/python -W ignore analysis/graph_2026_10/design_studies.py A > analysis/graph_2026_10/out/design_studies_A.txt
    venv/bin/python -W ignore analysis/graph_2026_10/design_studies.py B > analysis/graph_2026_10/out/design_studies_B.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.graph import between_game_graph            # noqa: E402
from stg.sports.nextgame import night_shocks               # noqa: E402

RUNS = [("MLB", 2025), ("NBA", 2025), ("NFL", 2025)]
SAMPLES = Path("data/sports/derived/samples")
HS = ["1h", "12h", "24h", "48h"]


def ols(df: pl.DataFrame, y: str, xs: list[str], cluster: str = "night") -> dict | None:
    d = df.filter(pl.col(y).is_not_null(), *[pl.col(x).is_not_null() for x in xs])
    if d.height < max(40, 8 * len(xs)):
        return None
    X = np.column_stack([np.ones(d.height)] + [d[x].to_numpy().astype(float) for x in xs])
    Y = d[y].to_numpy().astype(float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    u = Y - X @ b
    g = d[cluster].cast(pl.Utf8).to_numpy()
    meat = sum(np.outer(X[g == k].T @ u[g == k], X[g == k].T @ u[g == k]) for k in np.unique(g))
    G = len(np.unique(g))
    se = np.sqrt(np.clip(np.diag(XtX_inv @ meat @ XtX_inv * G / (G - 1)), 0, None))
    r2 = 1 - (u ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    return {"n": d.height, "r2": round(float(r2), 3),
            **{x: (float(b[i + 1]), float(b[i + 1] / se[i + 1]) if se[i + 1] > 0 else np.nan) for i, x in enumerate(xs)}}


def fmt(v) -> str:
    return f"{v[0]:.3f} (t {v[1]:.1f})" if isinstance(v, tuple) else str(v)


def load(d: SportsData) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """samples, team memberships (conf, division), team-night shocks (z, ms)."""
    edges = pl.read_parquet("data/sports/derived/edges.parquet")
    samp, mem, sh = [], [], []
    for lg, season in RUNS:
        samp.append(pl.read_parquet(SAMPLES / f"{lg}_{season}.parquet"))
        g = between_game_graph(d, edges, lg, season)
        mem.append(g.nodes.group_by("team").agg(pl.col("conf").drop_nulls().first(),
                                                pl.col("division").drop_nulls().first())
                   .with_columns(pl.lit(lg).alias("league")))
        t, _ = night_shocks(d, lg, season)
        sh.append(t.with_columns(pl.lit(lg).alias("league")))
    return (pl.concat(samp, how="diagonal_relaxed"), pl.concat(mem, how="diagonal_relaxed"),
            pl.concat(sh, how="diagonal_relaxed"))


# --------------------------------------------------------------------------- 1. proportional reallocation

def study1(s: pl.DataFrame, mem: pl.DataFrame) -> None:
    print("1. proportional reallocation within exclusive events (12 h moves, all rows with values, cents)")
    rows = []
    x = s.join(mem, on=["league", "team"], how="left").filter(pl.col("value_pre").is_not_null(),
                                                                pl.col("total_12h").is_not_null())
    for kind, key in (("champ", "league"), ("conference", "conf"), ("division", "division")):
        k = x.filter(pl.col("kind") == kind, pl.col(key).is_not_null())
        ev_key = ["league"] if key == "league" else ["league", key]
        size = mem.filter(pl.col(key).is_not_null()).group_by(ev_key).agg(pl.len().alias("n_event"))
        grp = [*ev_key, "night"]
        k = (k.join(size, on=ev_key)
             .with_columns(pl.len().over(grp).alias("n_have"),
                           pl.col("total_12h").filter(pl.col("played")).sum().over(grp).alias("S"),
                           pl.col("value_pre").filter(~pl.col("played")).sum().over(grp).alias("P"),
                           (~pl.col("played")).sum().over(grp).alias("n_np"))
             .filter(pl.col("n_have") >= 0.8 * pl.col("n_event"), pl.col("n_np") > 0, pl.col("P") > 0))
        np_ = k.filter(~pl.col("played")).with_columns(
            (-pl.col("S") * pl.col("value_pre") / pl.col("P")).alias("pred_prop"),
            (-pl.col("S") / pl.col("n_np")).alias("pred_equal"))
        for (lg,), g in np_.group_by("league"):
            a, b = ols(g, "total_12h", ["pred_prop"]), ols(g, "total_12h", ["pred_equal"])
            c = ols(g, "total_12h", ["pred_prop", "pred_equal"])
            ev = k.filter(pl.col("league") == lg).group_by(grp).agg(
                pl.col("total_12h").filter(pl.col("played")).sum().alias("S"),
                pl.col("total_12h").filter(~pl.col("played")).sum().alias("NP"))
            cons = ols(ev.with_columns((-pl.col("S")).alias("minus_S")), "NP", ["minus_S"])
            if a and b:
                rows.append({"league": lg, "kind": kind, "nonplayer_rows": a["n"],
                             "slope_prop": fmt(a["pred_prop"]), "r2_prop": a["r2"],
                             "slope_equal": fmt(b["pred_equal"]), "r2_equal": b["r2"],
                             "race_prop": fmt(c["pred_prop"]) if c else None,
                             "race_equal": fmt(c["pred_equal"]) if c else None,
                             "nonplayers_sum_on_-S": fmt(cons["minus_S"]) if cons else None})
    print(pl.DataFrame(rows).sort("league", "kind"))
    print("   slope 1 = full reallocation of the players' move; race = both predictors together")


# --------------------------------------------------------------------------- 2. cross-kind rivals

def study2(s: pl.DataFrame, mem: pl.DataFrame, sh: pl.DataFrame) -> None:
    print("\n2. cross-kind rival effects (live rows, total_12h, cents per unit z; reference: other conference)")
    z = sh.join(mem, on=["league", "team"], how="left")
    sum_conf = z.group_by("league", "night", "conf").agg(pl.col("z").sum().alias("z_conf"))
    sum_div = z.filter(pl.col("division").is_not_null()).group_by("league", "night", "division").agg(
        pl.col("z").sum().alias("z_div"))
    x = (s.filter(pl.col("live")).join(mem, on=["league", "team"], how="left")
         .join(sum_conf, on=["league", "night", "conf"], how="left")
         .join(sum_div, on=["league", "night", "division"], how="left")
         .with_columns(pl.col("z_conf").fill_null(0.0), pl.col("z_div").fill_null(0.0))
         .with_columns(pl.when(pl.col("division").is_not_null()).then(pl.col("z_div") - pl.col("z_own"))
                       .otherwise(0.0).alias("x_division"),
                       pl.when(pl.col("division").is_not_null()).then(pl.col("z_conf") - pl.col("z_div"))
                       .otherwise(pl.col("z_conf") - pl.col("z_own")).alias("x_conf_other")))
    rows = []
    for (lg, kind), g in x.filter(pl.col("kind") != "wins").group_by("league", "kind"):
        xs = ["z_own", "x_conf_other"] + (["x_division"] if g["x_division"].abs().sum() > 0 else [])
        r = ols(g, "total_12h", xs)
        if r:
            rows.append({"league": lg, "kind": kind, "n": r["n"], "r2": r["r2"],
                         **{k: fmt(r[k]) if k in r else None for k in ("z_own", "x_division", "x_conf_other")}})
    print(pl.DataFrame(rows).sort("league", "kind"))


# --------------------------------------------------------------------------- 3. lead-lag

def study3(s: pl.DataFrame) -> None:
    print("\n3a. absorption by kind (live rows): b(h) / b(48h) for the own-z weight; spread at the decision")
    rows = []
    for (lg, kind), g in s.filter(pl.col("live")).group_by("league", "kind"):
        b48 = ols(g, "total_48h", ["z_own"])
        if not b48 or abs(b48["z_own"][0]) < 1e-9:
            continue
        r = {"league": lg, "kind": kind, "b_48h": round(b48["z_own"][0], 3),
             "spread_med": g["spread_t"].median()}
        for lab, col in [("at_decision", "moved")] + [(h, f"total_{h}") for h in HS[:-1]]:
            o = ols(g, col, ["z_own"])
            r[lab] = round(o["z_own"][0] / b48["z_own"][0], 2) if o else None
        rows.append(r)
    print(pl.DataFrame(rows).sort("league", "kind"))

    print("\n3b. does another kind's move by the decision (same team, same night) predict this kind's "
          "remaining 12 h move? coefficient on the leader's moved, controlling own z and own moved (live rows)")
    mv = s.filter(pl.col("moved").is_not_null()).select("league", "team", "night", "kind", "moved")
    wide = mv.pivot(on="kind", index=["league", "team", "night"], values="moved")
    rows = []
    kinds = [k for k in ("division", "playoffs", "conference", "champ", "wins") if k in wide.columns]
    for (lg, kind), g in s.filter(pl.col("live")).group_by("league", "kind"):
        g = g.join(wide.rename({k: f"lead_{k}" for k in kinds}), on=["league", "team", "night"], how="left")
        for lead in kinds:
            if lead == kind or g[f"lead_{lead}"].null_count() == g.height:
                continue
            r = ols(g, "rem_12h", ["z_own", "moved", f"lead_{lead}"])
            if r:
                rows.append({"league": lg, "follower": kind, "leader": lead, "n": r["n"],
                             "b_leader_moved": fmt(r[f"lead_{lead}"]), "b_own_moved": fmt(r["moved"])})
    print(pl.DataFrame(rows).sort("league", "follower", "leader"))


# --------------------------------------------------------------------------- 4. margin surprise

def study4(s: pl.DataFrame, sh: pl.DataFrame) -> None:
    print("\n4. margin surprise into futures (live rows with a margin surprise; total_12h; ms in points / runs)")
    x = s.filter(pl.col("live")).join(sh.select("league", "night", "team", "ms"), on=["league", "night", "team"],
                                      how="left").filter(pl.col("played"))
    rows = []
    for (lg, kind), g in x.group_by("league", "kind"):
        a = ols(g, "total_12h", ["z_own"])
        b = ols(g, "total_12h", ["z_own", "ms"])
        if a and b:
            sd = g["ms"].std()
            rows.append({"league": lg, "kind": kind, "n": b["n"], "ms_sd": round(sd, 1),
                         "b_z_only": fmt(a["z_own"]), "b_z": fmt(b["z_own"]), "b_ms_per_pt": fmt(b["ms"]),
                         "ms_1sd_in_c": round(b["ms"][0] * sd, 2), "r2_z": a["r2"], "r2_z_ms": b["r2"]})
    print(pl.DataFrame(rows).sort("league", "kind"))


# --------------------------------------------------------------------------- batch B

def team_night_ppre(d: SportsData) -> pl.DataFrame:
    from stg.sports.features import REGULAR
    from stg.sports.graph import game_shocks
    out = []
    for lg, season in RUNS:
        sh = game_shocks(d, lg, season).filter(pl.col("game_type").is_in(REGULAR[lg]))
        out.append(sh.with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("night"))
                   .group_by("night", "team").agg(pl.col("p_pre").mean()).with_columns(pl.lit(lg).alias("league")))
    return pl.concat(out)


def study5(s: pl.DataFrame, pp: pl.DataFrame) -> None:
    print("5. asymmetry and non-linearity (live rows, team played, total_12h, cents)")
    x = (s.filter(pl.col("live"), pl.col("played")).join(pp, on=["league", "night", "team"], how="left")
         .with_columns(pl.col("z_own").clip(lower_bound=0).alias("z_pos"), pl.col("z_own").clip(upper_bound=0).alias("z_neg"),
                       (pl.col("p_pre") > 0.5).cast(pl.Float64).alias("fav"))
         .with_columns((pl.col("z_own") * pl.col("fav")).alias("z_x_fav"),
                       (pl.col("z_own") * pl.col("x_rival")).alias("z_x_rival")))
    rows = []
    for (lg, kind), g in x.group_by("league", "kind"):
        a = ols(g, "total_12h", ["z_pos", "z_neg"])
        b = ols(g, "total_12h", ["z_own", "z_x_fav", "fav"])
        rv = ols(g, "total_12h", ["z_own", "x_rival", "z_x_rival"]) if g["x_rival"].abs().sum() > 0 and kind != "champ" else None
        if a and b:
            rows.append({"league": lg, "kind": kind, "n": a["n"], "win_part": fmt(a["z_pos"]), "loss_part": fmt(a["z_neg"]),
                         "z": fmt(b["z_own"]), "z_x_fav": fmt(b["z_x_fav"]),
                         "z_x_rival": fmt(rv["z_x_rival"]) if rv else None})
    print(pl.DataFrame(rows).sort("league", "kind"))
    print("   win_part / loss_part: weight on z when the team beat / fell short of expectations (equal = symmetric);"
          " z_x_fav: extra weight when the team was the pre-game favourite; z_x_rival: own x rival interaction")


def study6(s: pl.DataFrame) -> None:
    print("\n6a. moves without linked games: |total_24h| on rows where neither the team nor a rival played "
          "(division) or the team did not play (all kinds), vs rows where the team played (live rows, cents)")
    x = s.filter(pl.col("live"), pl.col("total_24h").is_not_null())
    rows = []
    for (lg, kind), g in x.group_by("league", "kind"):
        r = {"league": lg, "kind": kind}
        for name, f in (("team_played", pl.col("played")), ("team_idle", ~pl.col("played")),
                        ("nothing_linked", ~pl.col("played") & (pl.col("n_rival_games") == 0))):
            h = g.filter(f)
            r[f"{name}_n"] = h.height
            r[f"{name}_mean|d|"] = round(h["total_24h"].abs().mean(), 2) if h.height else None
            r[f"{name}_moved"] = round((h["total_24h"] != 0).mean(), 2) if h.height else None
        rows.append(r)
    print(pl.DataFrame(rows).sort("league", "kind"))

    print("\n6b. common factor: share of what the shocks leave unexplained (12 h) that is common to a league-kind"
          " on the night (R^2 of night means of the residual); and to the whole league")
    rows = []
    for (lg,), gl in x.group_by("league"):
        res = []
        for (kind,), g in gl.group_by("kind"):
            g = g.filter(pl.col("total_12h").is_not_null())
            if g.height < 100:
                continue
            X = np.column_stack([np.ones(g.height), g["z_own"].to_numpy(), g["x_rival"].to_numpy()])
            y = g["total_12h"].to_numpy()
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            g = g.with_columns(pl.Series("res", y - X @ b))
            fe = g.with_columns(pl.col("res").mean().over("night").alias("m"))
            r2 = 1 - ((fe["res"] - fe["m"]) ** 2).sum() / ((fe["res"] - fe["res"].mean()) ** 2).sum()
            k = g.select("night").n_unique()
            rows.append({"league": lg, "kind": kind, "rows": g.height, "nights": k, "night_R2": round(float(r2), 3),
                         "chance_R2": round(k / g.height, 3)})
            res.append(g.select("night", "kind", "res"))
        if res:
            a = pl.concat(res).with_columns(pl.col("res").mean().over("night").alias("m"))
            r2 = 1 - ((a["res"] - a["m"]) ** 2).sum() / ((a["res"] - a["res"].mean()) ** 2).sum()
            rows.append({"league": lg, "kind": "ALL", "rows": a.height, "nights": a.select("night").n_unique(),
                         "night_R2": round(float(r2), 3), "chance_R2": round(a.select("night").n_unique() / a.height, 3)})
    print(pl.DataFrame(rows).sort("league", "kind"))
    print("   night_R2 well above chance_R2 (nights / rows) = a common move the shock terms miss")


def study7(s: pl.DataFrame, d: SportsData, sh: pl.DataFrame) -> None:
    print("\n7. head-to-head coupling: rival exposure split by whether that rival meets the team within 7 days"
          " of the decision (live rows, total_12h)")
    edges = pl.read_parquet("data/sports/derived/edges.parquet")
    rows = []
    for lg, season in RUNS:
        g = between_game_graph(d, edges, lg, season)
        team_of = g.nodes.select(pl.col("node").alias("src"), pl.col("team").alias("rival"))
        riv = (g.relations.filter(pl.col("relation").str.starts_with("rival")).join(team_of, on="src")
               .select(pl.col("dst").alias("node"), "rival").unique())
        games = (d.contracts.filter(pl.col("league") == lg, pl.col("kind") == "winner", pl.col("game_id").is_not_null())
                 .group_by("game_id").agg(pl.col("home").first(), pl.col("away").first(), pl.col("sched_start").first()))
        meet = pl.concat([games.select(pl.col("home").alias("team"), pl.col("away").alias("rival"), "sched_start"),
                          games.select(pl.col("away").alias("team"), pl.col("home").alias("rival"), "sched_start")])
        x = s.filter(pl.col("league") == lg, pl.col("live"), pl.col("kind").is_in(["division", "playoffs", "conference"]))
        pr = (x.select("node", "night", "team", "t").join(riv, on="node")
              .join(sh.filter(pl.col("league") == lg).select("night", pl.col("team").alias("rival"), "z"),
                    on=["night", "rival"]))
        soon = (pr.select("node", "night", "team", "rival", "t").unique().join(meet, on=["team", "rival"])
                .filter(pl.col("sched_start") > pl.col("t"), pl.col("sched_start") <= pl.col("t") + pl.duration(days=7))
                .select("node", "night", "rival").unique().with_columns(pl.lit(True).alias("meets")))
        pr = pr.join(soon, on=["node", "night", "rival"], how="left").with_columns(pl.col("meets").fill_null(False))
        agg = pr.group_by("node", "night").agg(pl.col("z").filter(pl.col("meets")).sum().alias("x_rival_meet"),
                                               pl.col("z").filter(~pl.col("meets")).sum().alias("x_rival_other"))
        x = x.join(agg, on=["node", "night"], how="left").with_columns(pl.col("x_rival_meet", "x_rival_other").fill_null(0.0))
        for (kind,), gk in x.group_by("kind"):
            r = ols(gk, "total_12h", ["z_own", "x_rival_meet", "x_rival_other"])
            if r:
                rows.append({"league": lg, "kind": kind, "n": r["n"],
                             "share_rows_with_meeting_rival": round((gk["x_rival_meet"] != 0).mean(), 2),
                             "z_own": fmt(r["z_own"]), "x_rival_meet": fmt(r["x_rival_meet"]),
                             "x_rival_other": fmt(r["x_rival_other"])})
    print(pl.DataFrame(rows).sort("league", "kind"))


def main() -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(240)
    d = SportsData()
    s, mem, sh = load(d)
    batch = sys.argv[1] if len(sys.argv) > 1 else "A"
    if batch == "A":
        study1(s, mem)
        study2(s, mem, sh)
        study3(s)
        study4(s, sh)
    else:
        study5(s, team_night_ppre(d))
        study6(s)
        study7(s, d, sh)


if __name__ == "__main__":
    main()
