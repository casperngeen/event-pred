"""Maker markouts on the graph's futures (division, playoffs, conference, champ), split by what just
happened to the contract: does liquidity provision pay, and is it picked off after linked results?

Maker markout per contract (cents): maker sold YES at p -> p - mid(t + tau); bought -> mid(t + tau) - p;
held to settlement, mid -> 100 / 0. Futures quotes are hourly candles plus trades, so tau >= 1 h.
Each fill is classed by the contract's linked results (its own team's games, and its rivals' as the
graph's rival relation for that kind defines them):

    own game live      the team's game in progress
    own result 0-3h    within 3 h after the team's result
    rival result 0-3h  within 3 h after a rival's result (and not the above)
    linked 3-24h       3-24 h after any linked result
    quiet              more than 24 h since any linked result

Maker fees per series from Kalshi's API (/series/{ticker}): taker-only series (quadratic) charge makers
nothing; maker-fee series use MAKER_RATE x multiplier x p (1 - p) (base rate unverified). t-stats
cluster by the trade's US-Eastern date. Average maker's fills, not one's own queue position.

    venv/bin/python -W ignore analysis/graph_2026_10/markouts_futures.py > analysis/graph_2026_10/out/markouts_futures.txt
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import httpx
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                                      # noqa: E402
from stg.sports.features import REGULAR                                     # noqa: E402
from stg.sports.graph import NODE_MAX_AGE, NODE_MAX_SPREAD, between_game_graph, game_shocks  # noqa: E402
from stg.sports.loaders import scan_trades                                  # noqa: E402

RUNS = [("MLB", 2025), ("NBA", 2025), ("NFL", 2025)]
KINDS = ("division", "playoffs", "conference", "champ")
TAUS = {"1h": dt.timedelta(hours=1), "12h": dt.timedelta(hours=12), "24h": dt.timedelta(hours=24)}
MAKER_RATE = 1.75
API = "https://api.elections.kalshi.com/trade-api/v2/series/"
OUT = Path(__file__).resolve().parent / "out"


def maker_fee_rates(series: list[str]) -> dict[str, float]:
    """cents x p(1-p) a maker pays per contract, per series (0 for taker-only series)."""
    out = {}
    for s in series:
        d = httpx.get(API + s, timeout=30).json().get("series", {})
        out[s] = MAKER_RATE * float(d.get("fee_multiplier") or 1) if "maker" in (d.get("fee_type") or "") else 0.0
    return out


def linked_events(d: SportsData, g, league: str, season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per node: its own team's games (start, result) and its rivals' result times."""
    sh = game_shocks(d, league, season).filter(pl.col("game_type").is_in(REGULAR[league]))
    own = g.nodes.select("node", "team").join(
        sh.select("team", pl.col("first_play").alias("start"), pl.col("known_at").alias("res")), on="team")
    team_of = g.nodes.select(pl.col("node").alias("src"), pl.col("team").alias("rival"))
    riv = (g.relations.filter(pl.col("relation").str.starts_with("rival"))
           .join(team_of, on="src").select(pl.col("dst").alias("node"), "rival").unique()
           .join(sh.select(pl.col("team").alias("rival"), pl.col("known_at").alias("res")), on="rival")
           .select("node", "res").unique())
    return own, riv


def classify(tr: pl.DataFrame, own: pl.DataFrame, riv: pl.DataFrame) -> pl.DataFrame:
    t = tr.sort("created_time")
    last_own = own.select("node", pl.col("res").alias("own_res")).sort("own_res")
    t = t.join_asof(last_own, left_on="created_time", right_on="own_res", by="node", strategy="backward")
    t = t.join_asof(riv.select("node", pl.col("res").alias("riv_res")).sort("riv_res"),
                    left_on="created_time", right_on="riv_res", by="node", strategy="backward")
    nxt = own.select("node", "start", "res").sort("start")       # the most recent own game started by t
    t = t.join_asof(nxt.rename({"start": "g_start", "res": "g_res"}), left_on="created_time",
                    right_on="g_start", by="node", strategy="backward")
    h = lambda c: (pl.col("created_time") - pl.col(c)).dt.total_minutes() / 60     # noqa: E731
    last_any = pl.min_horizontal(h("own_res"), h("riv_res"))
    return t.with_columns(
        pl.when(pl.col("g_start").is_not_null() & (pl.col("created_time") < pl.col("g_res"))).then(pl.lit("own game live"))
        .when(h("own_res") <= 3).then(pl.lit("own result 0-3h"))
        .when(h("riv_res") <= 3).then(pl.lit("rival result 0-3h"))
        .when(last_any <= 24).then(pl.lit("linked 3-24h"))
        .otherwise(pl.lit("quiet")).alias("state"))


def league_trades(d: SportsData, edges: pl.DataFrame, league: str, season: int) -> pl.DataFrame:
    g = between_game_graph(d, edges, league, season)
    nodes = g.nodes.filter(pl.col("kind").is_in(KINDS)).select("node", "kind", "team", "valid_to",
                                                               pl.col("tickers").list.first().alias("ticker"))
    series = d.contracts.filter(pl.col("ticker").is_in(nodes["ticker"].implode()))["series_ticker"].unique().to_list()
    fees = maker_fee_rates(series)
    tr = (scan_trades(series).filter(pl.col("ticker").is_in(nodes["ticker"].implode())).collect()
          .join(nodes, on="ticker")
          .join(d.contracts.select("ticker", "series_ticker"), on="ticker")
          .join(d.finals.select("ticker", "result"), on="ticker", how="left")
          .filter(pl.col("valid_to").is_null() | (pl.col("created_time") < pl.col("valid_to")))
          .with_columns(pl.when(pl.col("taker_side") == "yes").then(-1.0).otherwise(1.0).alias("maker_side")))
    at = pl.concat([tr.select("ticker", (pl.col("created_time") + x).alias("t")) for x in TAUS.values()]).unique()
    q = (d.quotes_at(at, max_spread=NODE_MAX_SPREAD, max_age=NODE_MAX_AGE)
         .select("ticker", "t", "mid").unique(["ticker", "t"]))
    for name, x in TAUS.items():
        tr = (tr.with_columns((pl.col("created_time") + x).alias("_t"))
              .join(q.rename({"t": "_t", "mid": f"mid_{name}"}), on=["ticker", "_t"], how="left").drop("_t")
              .with_columns((pl.col("maker_side") * (pl.col(f"mid_{name}") - pl.col("yes_price"))).alias(f"mo_{name}")))
    p = pl.col("yes_price") / 100
    tr = tr.with_columns(
        pl.when(pl.col("result").is_in(["yes", "no"]))
        .then(pl.col("maker_side") * (pl.when(pl.col("result") == "yes").then(100.0).otherwise(0.0) - pl.col("yes_price")))
        .alias("mo_settle"),
        (pl.col("series_ticker").replace_strict(fees, default=0.0) * p * (1 - p)).alias("maker_fee"),
        pl.col("created_time").dt.convert_time_zone("America/New_York").dt.date().alias("day"),
        pl.lit(league).alias("league"))
    own, riv = linked_events(d, g, league, season)
    return classify(tr, own, riv)


def wmean_t(df: pl.DataFrame, col: str) -> tuple[float, float]:
    x = df.filter(pl.col(col).is_not_null())
    if x.height < 30:
        return np.nan, np.nan
    w, y = x["count"].to_numpy(), x[col].to_numpy()
    m = float((w * y).sum() / w.sum())
    g = x.with_columns(pl.Series("_r", w * (y - m))).group_by("day").agg(pl.col("_r").sum())["_r"].to_numpy()
    se = float(np.sqrt((g ** 2).sum()) / w.sum())
    return m, (m / se if se > 0 else np.nan)


def table(tr: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    rows = []
    for key, g in tr.group_by(by):
        r = dict(zip(by, key)) | {"trades": g.height, "contracts": round(g["count"].sum())}
        for c in ("mo_1h", "mo_12h", "mo_24h", "mo_settle"):
            m, t = wmean_t(g, c)
            r[c] = f"{m:.2f} (t {t:.1f})" if np.isfinite(m) else None
        fee = float((g["maker_fee"] * g["count"]).sum() / g["count"].sum())
        m24, _ = wmean_t(g, "mo_24h")
        r["fee"] = round(fee, 2)
        r["net_24h"] = round(m24 - fee, 2) if np.isfinite(m24) else None
        rows.append(r)
    return pl.DataFrame(rows).sort(by)


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    d = SportsData()
    edges = pl.read_parquet("data/sports/derived/edges.parquet")
    tr = pl.concat([league_trades(d, edges, *r) for r in RUNS], how="diagonal_relaxed")
    OUT.mkdir(exist_ok=True)
    tr.select("league", "kind", "node", "created_time", "yes_price", "count", "maker_side", "state", "maker_fee",
              "mo_1h", "mo_12h", "mo_24h", "mo_settle").write_parquet(OUT / "markouts_futures.parquet")
    print(__doc__.split("\n\n")[1], "\n")
    print(f"{tr.height:,} trades, {tr['count'].sum():,.0f} contracts\n")
    print("1. by league and kind (cents per contract, contract-weighted; t clustered by day)")
    print(table(tr, ["league", "kind"]))
    print("\n2. by what just happened to the contract")
    print(table(tr, ["league", "state"]))
    print("\n3. by kind x state (all leagues)")
    print(table(tr, ["kind", "state"]))


if __name__ == "__main__":
    main()
