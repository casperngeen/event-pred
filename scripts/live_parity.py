"""Check live_model.py against the training panel on a past game: rebuild every step of the game from
the raw files (statsapi feed, Kalshi trades and candles) with the live code, and compare the model's
inputs and the 5 seeds' average forecast with the panel's (stg.models.ingame.make_batch), at the same
feed delay.

    venv/bin/python -W ignore scripts/live_parity.py MLB 823121 [--delay 10] [--weights results/ingame/deploy]
    venv/bin/python -W ignore scripts/live_parity.py NFL 401772987 [--espn-wp]

NFL plays come from the stored events (stg.sports.loaders.load_pbp), whose play end times and win
probability are nflverse's (published after the game): the check is of the code, on the training inputs.
--espn-wp uses ESPN's win probability instead, as live, to see how much that changes the forecast.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_model as LM  # noqa: E402

ROOT = LM.ROOT
SP = ROOT / "data" / "sports"
sys.path.insert(0, str(ROOT / "stg_infra"))


def raw_market(con: pl.DataFrame, first_play: dt.datetime) -> tuple[list[dict], list[dict]]:
    """The game's trades and candles from the stored history, as the live code takes them."""
    tick = con["ticker"].implode()
    tr, cd = [], []
    for s in con["series_ticker"].unique():
        f = list((SP / "trades" / s).glob("*.parquet"))
        t = (pl.scan_parquet(f, missing_columns="insert", extra_columns="ignore").filter(pl.col("ticker").is_in(tick))
             .select("trade_id", "ticker", "count", "yes_price", "taker_side", pl.col("created_time").cast(pl.Datetime("us", "UTC")))
             .unique("trade_id").collect())
        tr += [dict(ticker=r["ticker"], trade_id=r["trade_id"], t=r["created_time"].timestamp(), price=r["yes_price"],
                    count=r["count"], taker_side=r["taker_side"]) for r in t.iter_rows(named=True)]
        f = list((SP / "candles" / s).glob("*.parquet"))
        c = (pl.scan_parquet(f, missing_columns="insert", extra_columns="ignore").filter(pl.col("ticker").is_in(tick))
             .with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))).unique(["ticker", "ts"])
             .select("ticker", "ts", "yes_bid_close", "yes_ask_close").collect())
        cd += [dict(ticker=r["ticker"], ts=r["ts"].timestamp(), bid=r["yes_bid_close"], ask=r["yes_ask_close"])
               for r in c.iter_rows(named=True)]
    return tr, cd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("league", choices=["MLB", "NFL"])
    ap.add_argument("game_id")
    ap.add_argument("--espn-wp", action="store_true")
    ap.add_argument("--delay", type=float, default=10.0)
    ap.add_argument("--weights", default="results/ingame/deploy")
    a = ap.parse_args()
    gid, L_ = a.game_id, a.league
    PANEL = SP / "derived" / "ingame" / ("MLB_pitch" if L_ == "MLB" else L_)
    pn = LM.ingame.Panel(PANEL / "panel_v3", horizons=["30s", "60s", "120s", "1800s"])
    g = pn.manifest["games"].index(gid)
    gm = pl.read_parquet(PANEL / "games.parquet").filter(pl.col("game_id") == gid).row(0, named=True)
    start = pl.read_parquet(SP / "pbp" / "games.parquet").filter(pl.col("game_id") == gid)["start_utc"][0]
    con = pl.read_parquet(PANEL / "contracts.parquet").filter(pl.col("game_id") == gid,
                                                             pl.col("kind").is_in(["winner", "spread", "total"]))
    contracts = [dict(ticker=r["ticker"], kind=r["kind"], side=r["side"], strike=r["strike"]) for r in con.iter_rows(named=True)]
    fp = gm["first_play"].timestamp()

    # --- live side
    t0 = time.time()
    if L_ == "MLB":
        doc = json.loads(gzip.decompress((SP / "pbp" / "raw" / "MLB" / f"{gid}.json.gz").read_bytes()))
        plays = LM.mlb_plays(doc, start)
    else:
        from stg.sports.loaders import load_pbp
        ev = load_pbp("NFL").filter(pl.col("game_id") == gid, ~pl.col("t_outside_window")).sort("seq")
        # b3_table.plays_nfl: the home team's ESPN id is the team whose scoring raises the home score
        hid = ev.filter(pl.col("home_score").diff() > 0)["team_id"].drop_nulls().mode()[0]
        wp = pl.col("home_wp_post") if a.espn_wp else pl.coalesce("nflverse_home_wp_post", "home_wp_post")
        plays = LM.nfl_plays(ev.with_columns(wp.alias("wp")), hid)
    px, pt, pstate = LM.play_arrays(plays, L_)
    mk = LM.Market([c["ticker"] for c in contracts])
    tr, cd = raw_market(con, gm["first_play"])
    mk.add_trades(tr), mk.add_candles(cd)
    L = LM.pregame_total(contracts, mk, fp - 60)
    steps, ph = [], None
    for k in range(int(round((pn.steps_t[pn.game_step_ptr[g + 1] - 1] - fp) / LM.STEP)) + 1):
        s = LM.step_inputs(fp + LM.STEP * k, fp, contracts, mk, ph)
        ph = s["p_home"]
        steps.append(s)
    fc = LM.Forecaster(L_, a.weights)
    mine = fc.forecast(steps, px, pt, pstate, gm["L"], a.delay)
    print(f"live side: {len(plays)} pitches, {len(steps)} steps, {len(mine)} contract-steps, {time.time() - t0:.0f} s; "
          f"pregame total L: live {L:.3f}, panel {gm['L']:.3f}")

    # --- panel side
    ps = pn.plays_t[pn.game_play_ptr[g]:pn.game_play_ptr[g + 1]]
    print(f"plays: live {len(pt)}, panel {len(ps)}; times equal {len(pt) == len(ps) and np.allclose(pt, ps)}; "
          f"features max |diff| {np.abs(px - pn.plays_x[pn.game_play_ptr[g]:pn.game_play_ptr[g + 1]]).max() if len(pt) == len(ps) else 'n/a'}")
    b = LM.ingame.make_batch(pn, [g], None, None, torch.device("cpu"), stride=1, delay_s=a.delay)
    with torch.no_grad():
        d = torch.stack([m(b)[0] for m in fc.models]).mean(0).numpy()
    sid = pn.game_steps(g)
    st = pn.steps_t[sid]
    km = np.round((st - fp) / LM.STEP).astype(int)
    print(f"steps with contracts: live {len({r['k'] for r in mine})}, panel {len(sid)}")
    key = lambda kind, side, strike: (int(kind), int(side), round(float(strike), 2))   # noqa: E731
    pan = {}
    m = b.mask.numpy()
    nodes, kind, val = b.nodes.numpy(), b.kind.numpy(), b.value.numpy()
    side = np.sign(np.round(nodes[..., LM.NODE_FEATS.index("side_home")] - nodes[..., LM.NODE_FEATS.index("side_away")]))
    strike = pn.node_strike[pn.step_node_ptr[sid[0]]:pn.step_node_ptr[sid[-1] + 1]]
    i0 = pn.step_node_ptr[sid[0]]
    for i in range(len(sid)):
        a_, b_ = pn.step_node_ptr[sid[i]] - i0, pn.step_node_ptr[sid[i] + 1] - i0
        for j in range(b_ - a_):
            pan[(km[i],) + key(kind[i, j], side[i, j], strike[a_ + j])] = (nodes[i, j], val[i, j], d[i, j])
    live = {(r["k"],) + key(LM.KIND_CODE[r["kind"]], {"home": 1, "away": -1}.get(r["side"], 0), r["strike"] or 0): r for r in mine}
    both = sorted(set(pan) & set(live))
    print(f"contract-steps: live {len(live)}, panel {len(pan)}, both {len(both)}, live only {len(set(live) - set(pan))}, "
          f"panel only {len(set(pan) - set(live))}")
    if not both:
        return
    nd = np.array([pan[k][0] for k in both])
    # the live node features as the model saw them (rebuild through forecast's path for the matched rows)
    ld = []
    for k in both:
        r = live[k]
        ld.append(np.array(r["x"], np.float32).astype(np.float16).astype(np.float32))
    ld = np.array(ld)
    diff = np.abs(nd[:, :len(LM.NODE_FEATS)] - ld)
    print("node features, share of matched rows that differ (> 1e-3) and max |diff|:")
    for f, n in enumerate(LM.NODE_FEATS):
        bad = diff[:, f] > 1e-3
        if bad.any():
            print(f"  {n:14s} {bad.mean():.4f}  {diff[:, f].max():.4f}")
    dv = np.array([abs(pan[k][1] - live[k]["anchor"]) for k in both])
    print(f"anchor (value now): share differing {np.mean(dv > 1e-3):.4f}, max |diff| {dv.max():.3f}c")
    dd = np.array([np.abs(pan[k][2] - live[k]["delta"]) for k in both])
    h = fc.horizons
    print("forecast change, |live - panel| (cents): " + ", ".join(
        f"{h[i]} median {np.median(dd[:, i]):.4f} p99 {np.quantile(dd[:, i], 0.99):.4f} max {dd[:, i].max():.3f}" for i in range(len(h))))
    worst = np.argsort(-dd[:, h.index('120s')])[:5]
    for w in worst:
        k = both[w]
        print(f"  worst 120 s: step {k[0]} kind {k[1]} side {k[2]} strike {k[3]}: panel {pan[k][2][h.index('120s')]:.3f} "
              f"live {live[k]['delta'][h.index('120s')]:.3f}; features differing: "
              f"{[LM.NODE_FEATS[f] for f in np.nonzero(diff[w] > 1e-3)[0]]}")
        print("    panel", np.round(nd[w, :len(LM.NODE_FEATS)], 3).tolist(), "\n    live ", np.round(ld[w], 3).tolist(), live[k]["ticker"])


if __name__ == "__main__":
    main()
