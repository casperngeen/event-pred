"""Export the in-game training table (b3_table) as model-ready arrays, one league, game by game.

Writes data/sports/derived/ingame/<league>/panel_v2/ (numpy arrays + manifest.json). This is the one
file set that may be copied to the GPU cluster (soc-gpu-cluster rule): it refuses to write if any
game, play or target reaches the holdout (2026-07-01).

Anything that comes from the play-by-play depends on how late a live feed delivers it, so it is NOT
baked in here: the panel stores play and step times and the raw play state, and the loader
(stg.models.ingame.make_batch, argument delay_s) works out what is known at each step for a given
feed delay: the plays the memory reads, the game state (time left, score, possession), the contract's
distance to its threshold and the structural baseline. Market data (quotes, winner price, activity)
are stored as they were.

    plays_x [P, Fp]     per play: group, win-prob change, score change, offense, time left, score,
                        three league-specific situation fields, stoppage / scoring, gap to the last play
    plays_t [P]         when the play happened (epoch s)       game_play_ptr [G+1]
    plays_state [P, 4]  raw state after the play: tau, home margin, total, offense (+1 home, -1 away, 0)
    steps_t [S]         step time (epoch s)                     game_step_ptr [G+1]
    steps_x [S, Fs]     market features: home winner mid, whether it exists, recent winner activity
    step_node_ptr [S+1] contract rows of step s
    nodes_x [N, Fc]     market features of each contract at the step (kind, side, price range, value,
                        live, quote age)        nodes_kind [N]  0 winner, 1 spread, 2 total
    node_strike [N], node_side [N] (+1 home, -1 away, 0 total), game_L [G] pregame total
    value [N]           value now (cents)
    lo, hi, value_h [N, H]   target range and value at each horizon (manifest "horizons"); NaN if none
    y [N]               the contract's result (0 / 1), NaN if none
    game_fold [G]

Rows are dropped when the contract has no value now (no anchor), and when it is neither live nor
priced between 2c and 98c (all but settled: nothing to forecast). Prices and features are float16
(cent and half-cent prices are exact in float16).

With the second argument "prints" it writes panel_v3 instead: the anchor (value now) is the last
trade print if at most PRINT_MAX_AGE_S old (else the quote value), the targets lo / hi / value_h are
the lowest / highest / volume-weighted traded price in (t + h, t + h + 15 s] (NaN if nothing traded),
and five print inputs are added to nodes_x (PRINT_FEATS). Quote inputs are kept.

With "prints+futures" it writes panel_v3f: as panel_v3, plus each team's season-long contracts
(champion, conference, division, playoffs: node kinds 3-6, targets over 120 s windows, no result).

With "mid" it writes panel_v4m: the inputs of panel_v3 (quotes and five print inputs, no futures), but
the anchor (value now) is the middle of the current bid and ask whenever both sides exist (any width;
else the quote value, else the last print), and the target at each horizon is a single point: the
middle of the bid and ask at t + h, kept only if both sides are at most FRESH_S old then (else NaN;
after the game's end, the result). lo = hi = value_h. Neither end contains the bounce between the
bid and the ask, which a trade price does.

    venv/bin/python -W ignore analysis/graph_2026_10/b4_export.py NFL [prints | prints+futures | mid]
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "NFL"

import b1_response as R                                    # noqa: E402  (reads the league from argv)
from stg.sports.loaders import derived_dir                 # noqa: E402
from stg.splits import SPORTS_HOLDOUT_START                # noqa: E402

MODE = sys.argv[2] if len(sys.argv) > 2 else "quotes"
assert MODE in ("quotes", "prints", "prints+futures", "mid"), MODE
PRINTS, FUTURES, MID = MODE != "quotes", MODE == "prints+futures", MODE == "mid"     # PRINTS: print inputs
FRESH_S = 30                 # mid mode: a target mid counts only if both sides are at most this old at t + h
KIND_CODE = {"winner": 0, "spread": 1, "total": 2, "champ": 3, "conference": 4, "division": 5, "playoffs": 6}
PRINT_MAX_AGE_S = 120        # prints mode: the anchor is the last print if at most this old, else the quote value

D = derived_dir() / "ingame" / LEAGUE
OUT = D / ("panel_v4m" if MID else "panel_v3f" if FUTURES else "panel_v3" if PRINTS else "panel_v2")
S = {"NFL": (13.0, 14.5), "NBA": (14.5, 19.5), "MLB": (4.0, 4.25)}[LEAGUE]
SCALE = {"NFL": 14.0, "NBA": 20.0, "MLB": 4.0}[LEAGUE]          # margins and distances
TOT_SCALE = {"NFL": 45.0, "NBA": 225.0, "MLB": 9.0}[LEAGUE]
PLAY_SCALE = {"NFL": 7.0, "NBA": 3.0, "MLB": 1.0}[LEAGUE]       # points on one play
S_SCALES = {"NFL": (4.0, 10.0, 100.0), "NBA": (1.0, 1.0, 1.0), "MLB": (3.0, 4.0, 3.0)}[LEAGUE]
GROUPS = {"NFL": [g for g, _ in R.NFL_GROUPS],
          "NBA": ["made_2", "made_3", "miss", "turnover", "ft_trip", "period_end"],
          "MLB": [g for g, _ in R.MLB_GROUPS] + ["other"]}[LEAGUE]
EPOCH = lambda c: pl.col(c).dt.epoch("us").cast(pl.Float64) / 1e6      # noqa: E731


def f32(df: pl.DataFrame) -> np.ndarray:
    return df.select(pl.all().cast(pl.Float32).fill_null(0)).to_numpy().astype(np.float32)


def play_arrays(p: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    gap = p.select((pl.col("t_event") - pl.col("t_event").shift(1).over("game_id")).dt.total_seconds()
                   .fill_null(0).clip(0, 3600).alias("x"))["x"].to_numpy()
    feats = p.select(
        *[(pl.col("play_group") == g).alias(f"g_{g}") for g in GROUPS],
        (pl.col("dwp").fill_null(0) * 10).alias("dwp"), (pl.col("dmargin") / PLAY_SCALE).alias("dmargin"),
        (pl.col("dpoints") / PLAY_SCALE).alias("dpoints"), (pl.col("offense") == "home").alias("off_home"),
        (pl.col("offense") == "away").alias("off_away"), pl.col("tau").fill_null(0).alias("tau"),
        (pl.col("margin") / SCALE).alias("margin"), (pl.col("total") / TOT_SCALE).alias("total"),
        pl.col("wp_post").fill_null(0.5).alias("wp"),
        *[(pl.col(c).cast(pl.Float64).clip(0, 4 * sc) / sc).alias(c) for c, sc in zip(("s1", "s2", "s3"), S_SCALES)],
        pl.col("is_stoppage").alias("stoppage"), pl.col("is_scoring").alias("scoring"),
        pl.lit(0.0).alias("log_gap")).with_columns(pl.Series("log_gap", np.log1p(gap) / 5))
    state = p.select(pl.col("tau").fill_null(1.0), pl.col("margin").cast(pl.Float64), pl.col("total").cast(pl.Float64),
                     pl.when(pl.col("offense") == "home").then(1.0).when(pl.col("offense") == "away").then(-1.0).otherwise(0.0))
    return f32(feats), p.select(EPOCH("t_event"))["t_event"].to_numpy(), f32(state), feats.columns


NODE_FEATS = ["k_winner", "k_spread", "k_total", "side_home", "side_away", "live", "has_bid", "has_ask", "bid", "ask",
              "value", "log_age"]
PRINT_FEATS = ["has_print", "print_px", "log_print_age", "flow", "log_vol"]
if PRINTS:
    NODE_FEATS = NODE_FEATS + PRINT_FEATS


def node_frame(cs: pl.DataFrame) -> pl.DataFrame:
    return cs.select(
        (pl.col("kind") == "winner").alias("k_winner"), (pl.col("kind") == "spread").alias("k_spread"),
        (pl.col("kind") == "total").alias("k_total"), (pl.col("side") == "home").fill_null(False).alias("side_home"),
        (pl.col("side") == "away").fill_null(False).alias("side_away"), pl.col("live").fill_null(False),
        (pl.col("bid").is_not_null() & (pl.col("bid") > 0)).alias("has_bid"),
        (pl.col("ask").is_not_null() & (pl.col("ask") < 100)).alias("has_ask"),
        (pl.col("bid").fill_null(0) / 100).alias("bid"), (pl.col("ask").fill_null(100) / 100).alias("ask"),
        (pl.col("value").fill_null(pl.col("anchor")) / 100).alias("value"),
        (pl.col("quote_age_s").fill_null(3600).clip(0, 3600).log1p() / 8).alias("log_age"),
        *([(pl.col("pr_age_s") <= PRINT_MAX_AGE_S).fill_null(False).alias("has_print"),
           (pl.col("pr_px").fill_null(pl.col("anchor")) / 100).alias("print_px"),
           (pl.col("pr_age_s").fill_null(3600).clip(0, 3600).log1p() / 8).alias("log_print_age"),
           (pl.col("flow_60s").sign() * pl.col("flow_60s").abs().log1p() / 8).alias("flow"),
           (pl.col("vol_60s").log1p() / 8).alias("log_vol")] if PRINTS else []))


def main() -> None:
    meta = json.loads((D / "meta.json").read_text())
    hz = list(meta["horizons_s"].keys())
    games = pl.read_parquet(D / "games.parquet").sort("first_play")
    plays = pl.read_parquet(D / "plays.parquet")
    steps = pl.read_parquet(D / "steps.parquet")
    con = pl.read_parquet(D / "contracts.parquet")
    H = SPORTS_HOLDOUT_START
    assert (games["feed_end"] + dt.timedelta(seconds=max(meta["horizons_s"].values()))).max() < H, "targets reach the holdout"
    assert plays["known_at"].max() < H and steps["t"].max() < H, "plays / steps reach the holdout"
    gids = games["game_id"].to_list()
    gidx = {g: i for i, g in enumerate(gids)}

    plays = plays.with_columns(pl.col("game_id").replace_strict(gidx).alias("g")).sort("g", "seq")
    px, pt, pstate, pcols = play_arrays(plays)
    game_play_ptr = np.r_[0, np.cumsum(np.bincount(plays["g"].to_numpy(), minlength=len(gids)))]
    k_end = dict(steps.filter(~pl.col("after_end")).group_by("game_id").agg(pl.col("k").max()).iter_rows())   # last in-game step
    steps = (steps.filter(~pl.col("after_end")).with_columns(pl.col("game_id").replace_strict(gidx).alias("g")).sort("g", "k")
             .with_row_index("sid"))
    sx = f32(steps.select(
        (pl.col("p_home").forward_fill().over("g").fill_null(50) / 100).alias("p_home"),
        pl.col("p_home").is_not_null().alias("has_p_home"),
        (pl.col("winner_trades_5m").fill_null(0).log1p() / 7).alias("log_activity")))
    st = steps.select(EPOCH("t"))["t"].to_numpy()
    game_step_ptr = np.r_[0, np.cumsum(np.bincount(steps["g"].to_numpy(), minlength=len(gids)))]
    sid_of = steps.select("g", "k", "sid")
    cons = con.select("ticker", "kind", "side", "strike", "game_id")    # a futures contract is listed once per game

    def game_rows(gid: str) -> pl.DataFrame:
        cs = pl.read_parquet(D / "contract_steps" / f"{gid}.parquet").with_columns(pl.lit(gidx[gid]).alias("g"))
        if MID:   # quote age at t + h (both sides at most this old), joined on the step grid
            for name, sec in meta["horizons_s"].items():
                cs = cs.join(cs.select("ticker", (pl.col("k") - sec // 5).alias("k"), pl.col("quote_age_s").alias(f"qa_{name}")),
                             on=["ticker", "k"], how="left")
            two = pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") >= pl.col("bid"))
            anchor = (pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).otherwise(pl.col("value"))
                      .fill_null(pl.when(pl.col("pr_age_s") <= PRINT_MAX_AGE_S).then(pl.col("pr_px"))))
        elif PRINTS:
            anchor = pl.when(pl.col("pr_age_s") <= PRINT_MAX_AGE_S).then(pl.col("pr_px")).otherwise(pl.col("value"))
        else:
            anchor = pl.col("value")
        cs = (cs.join(cons.filter(pl.col("game_id") == gid).drop("game_id"), on="ticker").join(sid_of, on=["g", "k"], how="inner")
              .with_columns(anchor.alias("anchor"))
              .filter(pl.col("anchor").is_not_null(), pl.col("live") | pl.col("anchor").is_between(2, 98),
                      pl.lit(FUTURES) | pl.col("kind").is_in(["winner", "spread", "total"])))
        return cs.sort("sid", "kind", "side", "strike", "ticker")

    counts = {gid: game_rows(gid).height for gid in gids}            # pass 1: sizes
    n = sum(counts.values())
    OUT.mkdir(parents=True, exist_ok=True)
    mm = lambda name, shape, dtype: np.lib.format.open_memmap(OUT / f"{name}.npy", "w+", dtype, shape)   # noqa: E731
    nh = len(hz)
    A = dict(nodes_x=mm("nodes_x", (n, len(NODE_FEATS)), np.float16), nodes_kind=mm("nodes_kind", (n,), np.int8),
             node_strike=mm("node_strike", (n,), np.float32), node_side=mm("node_side", (n,), np.int8),
             value=mm("value", (n,), np.float32), lo=mm("lo", (n, nh), np.float16), hi=mm("hi", (n, nh), np.float16),
             value_h=mm("value_h", (n, nh), np.float16), y=mm("y", (n,), np.float16), node_sid=mm("node_sid", (n,), np.int64))
    i = 0
    for gid in gids:                                                  # pass 2: fill
        cs = game_rows(gid)
        j = i + cs.height
        A["nodes_x"][i:j] = node_frame(cs).select(pl.all().cast(pl.Float32)).to_numpy()
        A["nodes_kind"][i:j] = cs["kind"].replace_strict(KIND_CODE).to_numpy()
        A["node_strike"][i:j] = cs["strike"].fill_null(0).to_numpy()
        A["node_side"][i:j] = cs["side"].replace_strict({"home": 1, "away": -1}, default=0).to_numpy()
        A["value"][i:j] = cs["anchor"].to_numpy()
        for h, name in enumerate(hz):
            if MID:               # the bid / ask mid at t + h if both sides are fresh; after the end, the result
                b, a = cs[f"bid_{name}"].to_numpy(), cs[f"ask_{name}"].to_numpy()
                fresh = (cs[f"qa_{name}"].fill_null(10**9).to_numpy() <= FRESH_S) & np.isfinite(b) & np.isfinite(a) & (a >= b)
                after = cs[f"value_{name}"].fill_null(np.nan).to_numpy()
                ended = (cs["k"].to_numpy() + meta["horizons_s"][name] // 5) > k_end[gid]
                v = np.where(fresh, (b + a) / 2, np.where(ended & np.isin(after, (0.0, 100.0)), after, np.nan))
                A["lo"][i:j, h] = A["hi"][i:j, h] = A["value_h"][i:j, h] = v
                continue
            if PRINTS:            # traded prices in (t + h, t + h + 15 s]; NaN if nothing traded
                A["lo"][i:j, h] = cs[f"plo_{name}"].fill_null(np.nan).to_numpy()
                A["hi"][i:j, h] = cs[f"phi_{name}"].fill_null(np.nan).to_numpy()
                A["value_h"][i:j, h] = cs[f"pv_{name}"].fill_null(np.nan).to_numpy()
                continue
            b, a, v = cs[f"bid_{name}"], cs[f"ask_{name}"], cs[f"value_{name}"]
            q = (b.is_not_null() | a.is_not_null()).to_numpy()
            A["lo"][i:j, h] = np.where(q, b.fill_null(0).to_numpy(), v.fill_null(np.nan).to_numpy())
            A["hi"][i:j, h] = np.where(q, a.fill_null(100).to_numpy(), v.fill_null(np.nan).to_numpy())
            A["value_h"][i:j, h] = v.fill_null(np.nan).to_numpy()
        A["y"][i:j] = cs["y"].fill_null(np.nan).to_numpy()
        A["node_sid"][i:j] = cs["sid"].to_numpy()
        i = j
    step_node_ptr = np.r_[0, np.cumsum(np.bincount(np.asarray(A["node_sid"]), minlength=steps.height))]
    for v in A.values():
        v.flush()
    small = dict(plays_x=px, plays_t=pt, plays_state=pstate, game_play_ptr=game_play_ptr, steps_t=st, steps_x=sx,
                 game_step_ptr=game_step_ptr, step_node_ptr=step_node_ptr,
                 game_L=games["L"].fill_null(games["L"].median()).to_numpy().astype(np.float32),
                 game_fold=games["fold"].to_numpy().astype(np.int16))
    for k, v in small.items():
        np.save(OUT / f"{k}.npy", v)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        league=LEAGUE, version=4 if MID else 3 if PRINTS else 2,
        targets="fresh mid" if MID else "trade prints" if PRINTS else "quotes", futures=FUTURES, fresh_s=FRESH_S if MID else None, games=gids, horizons=hz, play_features=pcols, step_market_features=["p_home", "has_p_home",
        "log_activity"], node_market_features=NODE_FEATS, kinds=list(KIND_CODE) if FUTURES else ["winner", "spread", "total"], baseline_S=S,
        scale=SCALE, tot_scale=TOT_SCALE, holdout_start=H.isoformat(), built=dt.datetime.now(dt.timezone.utc).isoformat(),
        rows=n), indent=1))
    mb = sum(f.stat().st_size for f in OUT.glob("*.npy")) / 2**20
    print(f"{LEAGUE}: {len(gids)} games, {len(pt):,} plays, {len(st):,} steps, {n:,} contract rows, horizons {hz} "
          f"({mb:.0f} MB) -> {OUT}")


if __name__ == "__main__":
    main()
