"""Does each relation of the between-game graph move futures the way its sign prior says, and how fast?

One row per (futures contract C, night). A night is one league's slate (games whose first play
falls on one US-Eastern date). C's mid is read just before the night's first play (t0) and at
t_end + h, t_end being when the night's last result is known, for each horizon h. Per league and
contract kind:

    mid(C, t_end + h) - mid(C, t0)  =  a + b_self * z_own(C) + sum_r b_r * x_r(C) + e

    z_own(C)   the surprise z = y - p_pre (stg.sports.graph.game_shocks) of C's own team that night
               (summed over a doubleheader; 0 if it did not play)
    x_r(C)     C's net exposure through relation r: the summed z of every team linked to C by r that
               played that night. A game between two of C's rivals adds z + (-z) = 0, as it should.

Relations, for a contract of kind k: rival_k (prior -), and a placebo (prior 0): z_next, the
team's surprise on the next night it plays. It cannot cause tonight's move (and a non-zero
coefficient would flag lookahead); it is counted only while that game is outside the window.
Standard errors are clustered two ways, by night and by contract: rows of one night share common
moves, and at long horizons one contract's windows on consecutive nights overlap. Later nights'
results are independent of tonight's, so they add noise, not bias.

Every game adds z and -z, so a contract's rival exposure sums to -z_own minus the z of its rivals'
games against outsiders. Rival effects are identified only through those games; for champ, where
every team is a rival, the exposure is exactly -z_own and the rival term is dropped (shown as
"not identified"): its b_self is then the net effect of the night's results on the contract. For
the same reason a cross-conference placebo is not identified (own + rivals + the other conference
sum to 0), hence the next-night placebo.

A second pass keeps only fresh quotes (both sides updated within FRESH of each read), to tell slow
absorption from stale overnight quotes.

Single-leg nodes only (division, playoffs, conference, champ); wins ladders are left out.

    venv/bin/python -W ignore analysis/graph_2026_10/sign_check.py > analysis/graph_2026_10/out/sign_check.txt
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                                    # noqa: E402
from stg.sports.graph import between_game_graph, game_shocks              # noqa: E402

RUNS = [("MLB", 2025, "R"), ("NBA", 2025, "2"), ("NFL", 2025, "2")]
KINDS = ("division", "playoffs", "conference", "champ")
M = dt.timedelta(minutes=1)
HORIZONS = {"end": 0 * M, "5m": 5 * M, "15m": 15 * M, "1h": 60 * M, "3h": 180 * M,
            "12h": 720 * M, "24h": 1440 * M, "48h": 2880 * M}
MAX_SPREAD, MAX_AGE = 10.0, dt.timedelta(days=1)     # futures books are thin and slow
FRESH = dt.timedelta(minutes=30)
OUT = Path(__file__).resolve().parent / "out"


def ols_twoway(y: np.ndarray, X: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Cameron-Gelbach-Miller two-way clustered standard errors."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b

    def meat(g):
        m = np.zeros((X.shape[1], X.shape[1]))
        for k in np.unique(g):
            s = X[g == k].T @ u[g == k]
            m += np.outer(s, s)
        return m
    both = np.char.add(np.char.add(g1.astype(str), "|"), g2.astype(str))
    V = XtX_inv @ (meat(g1) + meat(g2) - meat(both)) @ XtX_inv
    if np.any(np.diag(V) <= 0):                     # two-way can go negative: fall back to the first way
        V = XtX_inv @ meat(g1) @ XtX_inv
    G = min(len(np.unique(g1)), len(np.unique(g2)))
    V *= G / (G - 1)
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def panel(d: SportsData, edges: pl.DataFrame, league: str, season: int, regular: str,
          max_age: dt.timedelta = MAX_AGE) -> pl.DataFrame:
    """One row per (contract, night) with z_own, the exposures and the mid changes per horizon."""
    g = between_game_graph(d, edges, league, season)
    sh = (game_shocks(d, league, season).filter(pl.col("game_type") == regular)
          .with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("night")))
    nights = sh.group_by("night").agg(pl.col("first_play").min().alias("t0"), pl.col("known_at").max().alias("t_end"))
    zt = sh.group_by("night", "team").agg(pl.col("z").sum())          # a team's z that night

    nd = g.nodes.filter(pl.col("kind").is_in(KINDS)).select(
        "node", "team", "kind", "conf", "known_at", "valid_to", pl.col("tickers").list.first().alias("ticker"))
    # links into C: rival edges (src team -> C)
    links = (g.relations.filter(pl.col("relation").str.starts_with("rival"))
             .join(nd.select(pl.col("node").alias("src"), pl.col("team").alias("src_team")), on="src")
             .select(pl.col("dst").alias("node"), "src_team", pl.lit("rival").alias("term")))

    rows = nd.join(nights, how="cross").filter(pl.col("known_at") <= pl.col("t0"))
    own = zt.rename({"z": "z_own"})
    rows = rows.join(own, on=["night", "team"], how="left").with_columns(pl.col("z_own").fill_null(0.0))
    # the next night it plays, counted from tonight even if it did not play tonight
    nx = (rows.select("node", "team", "night")
          .join(zt.join(nights.select("night", "t0"), on="night").rename({"night": "nn", "z": "z_next",
                                                                          "t0": "next_t0"}), on="team")
          .filter(pl.col("nn") > pl.col("night")).sort("nn")
          .group_by("node", "night").agg(pl.col("z_next").first(), pl.col("next_t0").first()))
    rows = rows.join(nx, on=["node", "night"], how="left")
    ex = (links.join(rows.select("node", "night").unique(), on="node")
          .join(zt.rename({"team": "src_team"}), on=["night", "src_team"])
          .group_by("node", "night").agg(pl.col("z").sum().alias("x_rival")))
    rows = rows.join(ex, on=["node", "night"], how="left").with_columns(pl.col("x_rival").fill_null(0.0))

    at = [rows.select("ticker", pl.col("t0").alias("t"))]
    for h, dh in HORIZONS.items():
        rows = rows.with_columns((pl.col("t_end") + dh).alias(f"t_{h}"))
        rows = rows.with_columns(pl.when(pl.col("next_t0") > pl.col(f"t_{h}")).then(pl.col("z_next"))
                                 .otherwise(0.0).fill_null(0.0).alias(f"x_next_{h}"))
        at.append(rows.select("ticker", pl.col(f"t_{h}").alias("t")))
    q = (d.quotes_at(pl.concat(at).unique(), max_spread=MAX_SPREAD, max_age=max_age)
         .select("ticker", "t", "mid").unique(["ticker", "t"]))
    rows = rows.join(q.rename({"t": "t0", "mid": "m0"}), on=["ticker", "t0"], how="left")
    for h in HORIZONS:
        rows = (rows.join(q.rename({"t": f"t_{h}", "mid": f"m_{h}"}), on=["ticker", f"t_{h}"], how="left")
                .with_columns(pl.when(pl.col("valid_to").is_null() | (pl.col("valid_to") > pl.col(f"t_{h}")))
                              .then(pl.col(f"m_{h}") - pl.col("m0")).alias(f"d_{h}")))
    return rows.with_columns(pl.lit(league).alias("league"))


def fit(rows: pl.DataFrame) -> pl.DataFrame:
    out = []
    for (league, kind), s in rows.group_by("league", "kind"):
        for h in HORIZONS:
            s = s.with_columns(pl.col(f"x_next_{h}").alias("x_next"))
            terms = ["z_own", "x_rival", "x_next"]
            ok = s.filter(pl.col(f"d_{h}").is_not_null(),
                          pl.any_horizontal(pl.col(t) != 0 for t in terms[:2]))   # nights it was exposed
            if ok.height < 50:
                continue
            use = list(terms)
            X = np.column_stack([np.ones(ok.height)] + [ok[t].to_numpy() for t in use])
            while np.linalg.matrix_rank(X) < X.shape[1]:   # e.g. champ: rival exposure = -z_own exactly
                dropped = use.pop(1)
                out.append(dict(league=league, kind=kind,
                                term=dropped.replace("x_next", "placebo").replace("x_", ""), horizon=h, n=ok.height,
                                beta=np.nan, t=np.nan))
                X = np.column_stack([np.ones(ok.height)] + [ok[t].to_numpy() for t in use])
            b, se = ols_twoway(ok[f"d_{h}"].to_numpy(), X, ok["night"].cast(pl.Utf8).to_numpy(),
                               ok["node"].to_numpy())
            for i, t in enumerate(use, 1):
                out.append(dict(league=league, kind=kind,
                                term=t.replace("z_own", "self").replace("x_next", "placebo").replace("x_", ""),
                                horizon=h, n=ok.height, beta=b[i], t=b[i] / se[i] if se[i] > 0 else np.nan))
    return pl.DataFrame(out)


def table(res: pl.DataFrame) -> pl.DataFrame:
    cell = (res.with_columns(pl.when(pl.col("beta").is_nan()).then(pl.lit("not identified"))
                             .otherwise(pl.format("{} ({})", pl.col("beta").round(2), pl.col("t").round(1)))
                             .alias("cell"))
            .pivot(on="horizon", index=["league", "kind", "term"], values="cell"))
    n = res.filter(pl.col("horizon") == "1h").select("league", "kind", "term", pl.col("n").alias("n_1h"))
    cell = cell.join(n, on=["league", "kind", "term"], how="left")
    kind_order = {k: i for i, k in enumerate(KINDS)}
    term_order = {"self": 0, "rival": 1, "placebo": 2}
    return (cell.with_columns(pl.col("kind").replace_strict(kind_order).alias("_k"),
                              pl.col("term").replace_strict(term_order).alias("_t"))
            .sort("league", "_k", "_t").drop("_k", "_t")
            .select("league", "kind", "term", "n_1h", *[h for h in HORIZONS if h in cell.columns]))


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(240)
    d = SportsData()
    edges = pl.read_parquet("data/sports/derived/edges.parquet")
    OUT.mkdir(exist_ok=True)
    print(__doc__.split("\n\n")[1], "\n")
    for label, age in (("all quotes", MAX_AGE), ("fresh quotes only", FRESH)):
        rows = pl.concat([panel(d, edges, *r, max_age=age) for r in RUNS], how="diagonal_relaxed")
        res = fit(rows)
        res.write_parquet(OUT / f"sign_check_{'fresh' if age == FRESH else 'all'}.parquet")
        cell = table(res)
        print(f"######## {label}: spread <= {MAX_SPREAD}c, each side updated within {age} of the read. "
              f"Cells: beta in cents per unit z (t, two-way clustered by night and contract); "
              f"'end' = the night's last result known.\n")
        for lg in ("MLB", "NBA", "NFL"):
            print(f"== {lg} 2025 (regular season)")
            print(cell.filter(pl.col("league") == lg).drop("league"))
            print()


if __name__ == "__main__":
    main()
