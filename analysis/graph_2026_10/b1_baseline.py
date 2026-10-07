"""Step 5: a structural baseline for in-game spread and total contracts, with no learning.

The final margin (home minus away) is the current margin plus a normal remainder whose spread
shrinks with the time left:  M = m + X,  X ~ N(mu, (S_M^2) * tau),  tau = share of regulation left.
mu is set so that P(M > 0) equals the market's home win price p, so with z = Phi^-1(p) and
s = S_M sqrt(tau):

    spread "home wins by > k"   P = Phi(z - k / s)        "away wins by > k"   P = Phi(-z - k / s)

(only p, k and the time left matter: the score enters through p). The final total is the current
total plus a normal remainder:  T = t + Y,  Y ~ N(L * tau, S_T^2 * tau),  L = the game's pregame
total, read from the total contracts' ladder just before kickoff (the strike where the over is
priced 50c):

    total "total > k"           P = Phi((t + L tau - k) / (S_T sqrt(tau)))

S_M and S_T are one number each per league, fitted on the first TRAIN_SHARE of games by date (mean
absolute error against market mids) and tested on the rest. Time left (tau):
    NFL  game clock, 60 regulation minutes (overtime: its clock / 60)
    NBA  game clock, 48 regulation minutes (overtime: its clock / 48)
    MLB  outs left of the regulation 54 (6 per inning), at least 1/54 in extra innings
Game state at each minute is the last play known by then (NFL: the play's end; NBA: its time; MLB:
the pitch's end, since its score includes the pitch's result). NBA and MLB use a sample of games.

Evaluated on a 1-min grid in play, on contracts with a usable two-sided quote (gap <= 10c):
    levels   market mid vs the baseline: mean abs error, bias (market - baseline), correlation, by
             contract type, distance to the threshold and time left
    changes  the next minute's change in the market mid vs the baseline's change (recomputed with
             the next minute's winner price, score and clock): R^2 against no change, and slope

    venv/bin/python -W ignore analysis/graph_2026_10/b1_baseline.py NFL > analysis/graph_2026_10/out/b1_baseline_NFL.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import load_pbp                    # noqa: E402
from stg.sports.prices import quotes_at, scan_quotes       # noqa: E402

LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "NFL"
CFG = {
    "NFL": dict(grid_m=np.arange(8.0, 22.1, 0.5), grid_t=np.arange(8.0, 22.1, 0.5), dist=[-14, -7, -3, 0, 3, 7, 14],
                n_games=10_000, state_t="t_event", fit_t0=13.0),
    "NBA": dict(grid_m=np.arange(6.0, 24.1, 0.5), grid_t=np.arange(8.0, 30.1, 0.5), dist=[-20, -10, -5, 0, 5, 10, 20],
                n_games=400, state_t="t_event", fit_t0=18.0),
    "MLB": dict(grid_m=np.arange(1.5, 8.01, 0.25), grid_t=np.arange(1.5, 8.01, 0.25), dist=[-4, -2, -1, 0, 1, 2, 4],
                n_games=600, state_t="known_at", fit_t0=4.5),
}[LEAGUE]
SEED = 0
TRAIN_SHARE = 0.4
GAP = 10.0
TAU_EDGES = [0.1, 0.25, 0.5, 0.75]
OUT = Path(__file__).resolve().parent / "out"


def tau_left() -> pl.Expr:
    if LEAGUE == "MLB":
        made = (pl.col("period") - 1) * 6 + pl.when(pl.col("half") == "bottom").then(3).otherwise(0) + pl.col("outs")
        return ((54 - made) / 54).clip(1 / 54, 1.0)
    reg, per = (60.0, 15.0) if LEAGUE == "NFL" else (48.0, 12.0)
    clk = pl.col("clock")
    mm = pl.when(clk.str.contains(":")).then(clk.str.split(":").list.get(0, null_on_oob=True).cast(pl.Float64, strict=False)).otherwise(0.0)
    ss = pl.when(clk.str.contains(":")).then(clk.str.split(":").list.get(1, null_on_oob=True).cast(pl.Float64, strict=False)) \
        .otherwise(clk.cast(pl.Float64, strict=False))
    clock_min = mm + ss / 60
    in_reg = ((4 - pl.col("period")) * per + clock_min) / reg
    return pl.when(pl.col("period") <= 4).then(in_reg).otherwise(clock_min / reg).clip(1e-3, 1.0)


def mids(con: pl.DataFrame, at: pl.DataFrame) -> pl.DataFrame:
    """at: (ticker, t) -> two-sided mid with gap <= GAP and sides <= 10 min old, else null."""
    q = quotes_at(scan_quotes(con["series_ticker"].unique().to_list()), at.select("ticker", "t").unique(), max_spread=GAP)
    return q.select("ticker", "t", "mid").unique(["ticker", "t"])


def pregame_total(con: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """game_id -> L, the strike where the total ladder prices the over at 50c, 1 min before kickoff."""
    tot = con.filter(pl.col("kind") == "total").join(games.select("game_id", "first_play"), on="game_id")
    at = tot.select("ticker", "game_id", "strike", (pl.col("first_play") - pl.duration(minutes=1)).alias("t"))
    m = at.join(mids(con, at), on=["ticker", "t"], how="left").drop_nulls("mid").sort("game_id", "strike")
    out = []
    for (gid,), g in m.partition_by("game_id", as_dict=True).items():
        k, p = g["strike"].to_numpy(), g["mid"].to_numpy() / 100
        if len(k) < 2 or not (p.max() >= 0.5 >= p.min()):
            continue
        order = np.argsort(-p)                      # P(over) falls as the strike rises
        out.append((gid, float(np.interp(-0.5, -p[order], k[order]))))
    return pl.DataFrame(out, schema={"game_id": pl.String, "L": pl.Float64}, orient="row")


def baseline(df: pl.DataFrame, s_m: float, s_t: float) -> np.ndarray:
    p = np.clip(df["p_home"].to_numpy() / 100, 0.01, 0.99)
    z = norm.ppf(p)
    tau = df["tau"].to_numpy()
    k = df["strike"].to_numpy()
    kind = df["kind"].to_numpy()
    home = df["side"].to_numpy() == "home"
    s = s_m * np.sqrt(tau)
    spread = np.where(home, norm.cdf(z - k / s), norm.cdf(-z - k / s))
    lt = df["L"].to_numpy() * tau
    total = norm.cdf((df["total_now"].to_numpy() + lt - k) / (s_t * np.sqrt(tau)))
    return 100 * np.where(kind == "spread", spread, total)


def fit(df: pl.DataFrame) -> tuple[float, float]:
    sp, to = df.filter(pl.col("kind") == "spread"), df.filter(pl.col("kind") == "total")
    t0 = CFG["fit_t0"]
    best_m = min(CFG["grid_m"], key=lambda s: np.nanmean(np.abs(sp["mid"].to_numpy() - baseline(sp, s, t0))))
    best_t = min(CFG["grid_t"], key=lambda s: np.nanmean(np.abs(to["mid"].to_numpy() - baseline(to, t0, s))))
    return float(best_m), float(best_t)


def level_table(df: pl.DataFrame, by: str) -> pl.DataFrame:
    return (df.group_by("kind", by).agg(pl.len().alias("n"), (pl.col("mid") - pl.col("base")).abs().mean().round(2).alias("mae"),
                                        (pl.col("mid") - pl.col("base")).mean().round(2).alias("bias"),
                                        pl.corr("mid", "base").round(3).alias("corr"))
            .sort("kind", by))


def main() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    d = SportsData()
    con = (d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "game",
                              pl.col("kind").is_in(["winner", "spread", "total"]), pl.col("game_id").is_not_null())
           .select("ticker", "series_ticker", "kind", "side", "strike", "game_id"))
    with_ladders = con.filter(pl.col("kind") == "spread")["game_id"].unique().implode()
    games = (d.results.filter(pl.col("league") == LEAGUE).select("game_id", "first_play", "feed_end")
             .filter(pl.col("game_id").is_in(with_ladders)))
    games = games.sample(n=min(CFG["n_games"], games.height), seed=SEED).sort("first_play")
    n_train = int(games.height * TRAIN_SHARE)
    games = games.with_columns((pl.int_range(pl.len()) < n_train).alias("train"))
    L = pregame_total(con, games)
    # game state on a 1-min grid (as of each minute: the last play that had happened)
    grid = (games.with_columns(pl.datetime_ranges(pl.col("first_play").dt.truncate("1m") + pl.duration(minutes=1),
                                                  pl.col("feed_end").dt.truncate("1m") - pl.duration(minutes=1), "1m").alias("t"))
            .explode("t").select("game_id", "train", "t").sort("t"))
    extra = ["half", "outs"] if LEAGUE == "MLB" else ["clock"]
    pbp = (load_pbp(LEAGUE).filter(~pl.col("t_outside_window"))
           .select("game_id", pl.col(CFG["state_t"]).alias("t"), "period", *extra, "home_score", "away_score")
           .drop_nulls("t").sort("t"))
    st = (grid.join_asof(pbp, on="t", by="game_id", strategy="backward", check_sortedness=False)
          .with_columns(tau_left().alias("tau"), (pl.col("home_score") + pl.col("away_score")).alias("total_now"),
                        (pl.col("home_score") - pl.col("away_score")).alias("margin"))
          .drop_nulls("tau"))
    st = st.with_columns((pl.col("t") + pl.duration(minutes=1)).alias("t1"))
    st1 = (st.select("game_id", "t1").rename({"t1": "t"}).sort("t")
           .join_asof(pbp, on="t", by="game_id", strategy="backward", check_sortedness=False)
           .with_columns(tau_left().alias("tau"), (pl.col("home_score") + pl.col("away_score")).alias("total_now"))
           .select("game_id", pl.col("t").alias("t1"), pl.col("tau").alias("tau_1"), pl.col("total_now").alias("total_now_1")))
    st = st.join(st1, on=["game_id", "t1"], how="left")
    # winner price (home), now and in a minute
    w = con.filter(pl.col("kind") == "winner", pl.col("side").is_in(["home", "away"]))
    wat = pl.concat([st.select("game_id", "t").join(w.select("game_id", "ticker", "side"), on="game_id"),
                     st.select("game_id", pl.col("t1").alias("t")).join(w.select("game_id", "ticker", "side"), on="game_id")]).unique()
    wm = wat.join(mids(con, wat), on=["ticker", "t"], how="left").with_columns(
        pl.when(pl.col("side") == "home").then(pl.col("mid")).otherwise(100 - pl.col("mid")).alias("ph"))
    wm = wm.group_by("game_id", "t").agg(pl.col("ph").mean().alias("p_home"))
    st = (st.join(wm, on=["game_id", "t"], how="left")
          .join(wm.rename({"t": "t1", "p_home": "p_home_1"}), on=["game_id", "t1"], how="left")
          .drop_nulls(["p_home"]))
    # spread / total contracts
    c = con.filter(pl.col("kind").is_in(["spread", "total"]), pl.col("side").is_in(["home", "away"]) | (pl.col("kind") == "total"))
    x = st.join(c.select("game_id", "ticker", "kind", "side", "strike"), on="game_id").join(L, on="game_id", how="left")
    cat = pl.concat([x.select("ticker", "t"), x.select("ticker", pl.col("t1").alias("t"))]).unique()
    cm = mids(con, cat)
    x = (x.join(cm, on=["ticker", "t"], how="left")
         .join(cm.rename({"t": "t1", "mid": "mid_1"}), on=["ticker", "t1"], how="left")
         .filter(pl.col("mid").is_not_null(), (pl.col("kind") == "spread") | pl.col("L").is_not_null()))
    hs = pl.when(pl.col("side") == "home").then(1).otherwise(-1)
    x = x.with_columns(pl.when(pl.col("kind") == "spread").then(pl.col("strike") - hs * pl.col("margin"))
                       .otherwise(pl.col("strike") - pl.col("total_now")).alias("dist"))
    print(f"{LEAGUE}: {games.height} games ({n_train} train by date, {games.height - n_train} test); "
          f"{st.height:,} game-minutes with a winner price; pregame total found for {L.height} games; "
          f"{x.height:,} contract-minutes with a usable mid")

    s_m, s_t = fit(x.filter(pl.col("train")))
    unit = "runs" if LEAGUE == "MLB" else "points"
    print(f"fitted on train games: S_M = {s_m} {unit} (final-margin sd over a full game), S_T = {s_t} {unit} (total)")
    x = x.with_columns(pl.Series("base", baseline(x, s_m, s_t)))
    x1 = x.drop_nulls(["mid_1", "p_home_1", "tau_1"]).with_columns(
        pl.col("p_home_1").alias("p_home"), pl.col("tau_1").alias("tau"), pl.col("total_now_1").alias("total_now"))
    x1 = x1.with_columns(pl.Series("base_1", baseline(x1, s_m, s_t)))
    test = x.filter(~pl.col("train"))
    de = CFG["dist"]
    lab_d = [f"<{de[0]}"] + [f"{a}..{b}" for a, b in zip(de[:-1], de[1:])] + [f">{de[-1]}"]
    lab_t = [f"<{TAU_EDGES[0]}"] + [f"{a}..{b}" for a, b in zip(TAU_EDGES[:-1], TAU_EDGES[1:])] + [f">{TAU_EDGES[-1]}"]
    test = test.with_columns(pl.col("dist").cut(de, labels=lab_d).cast(pl.String).alias("distance"),
                             pl.col("tau").cut(TAU_EDGES, labels=lab_t).cast(pl.String).alias("time_left"))
    print("\n######## levels, test games: market mid vs baseline (cents)")
    print(test.group_by("kind").agg(pl.len().alias("n"), (pl.col("mid") - pl.col("base")).abs().mean().round(2).alias("mae"),
                                    (pl.col("mid") - pl.col("base")).abs().median().round(2).alias("median_abs_err"),
                                    (pl.col("mid") - pl.col("base")).mean().round(2).alias("bias"),
                                    pl.corr("mid", "base").round(3).alias("corr")).sort("kind"))
    print(f"by distance to the threshold ({'runs' if LEAGUE == 'MLB' else 'points'}; negative = already past it):")
    print(level_table(test, "distance").with_columns(pl.col("distance").replace_strict(lab_d, list(range(len(lab_d))), default=99)
                                                      .alias("_o")).sort("kind", "_o").drop("_o"))
    print("by share of regulation left:")
    print(level_table(test, "time_left"))

    ch = x1.filter(~pl.col("train")).with_columns((pl.col("mid_1") - pl.col("mid")).alias("dm"),
                                                  (pl.col("base_1") - pl.col("base")).alias("db"))
    print("\n######## changes over the next minute, test games (cents)")
    rows = []
    for (kind,), g in ch.partition_by("kind", as_dict=True).items():
        for label, gg in (("all minutes", g), ("market moved >= 2c", g.filter(pl.col("dm").abs() >= 2))):
            dm, db = gg["dm"].to_numpy(), gg["db"].to_numpy()
            rows.append(dict(kind=kind, rows=label, n=len(dm), sd_market_change=round(float(dm.std()), 2),
                             r2_vs_no_change=round(1 - float(np.mean((dm - db) ** 2) / np.mean(dm ** 2)), 3),
                             slope=round(float(np.polyfit(db, dm, 1)[0]), 3), corr=round(float(np.corrcoef(db, dm)[0, 1]), 3)))
    print(pl.DataFrame(rows).sort("kind", "rows"))
    x.write_parquet(OUT / f"b1_baseline_{LEAGUE}.parquet")


if __name__ == "__main__":
    main()
