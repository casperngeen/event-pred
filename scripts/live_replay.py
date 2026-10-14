"""Replay a past MLB game through live_trade.Engine as if it were live, to test the runner end to end
before a real game: the statsapi feed as it stood after each pitch (the finished game's document cut at
each pitch's end), arriving FEED_LAG_S later; Kalshi trades arriving TRADE_LAG_S after they print; 1-min
candles CANDLE_LAG_S after their minute ends. A simulated clock steps the engine every second. Nothing is
published (dry run). Logs go to <out>/replay/MLB_<game>/.

    venv/bin/python -u -W ignore scripts/live_replay.py 823121 [--feed-lag 3] [--out <scratch dir>]
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import heapq
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_model as LM          # noqa: E402
import live_trade as T           # noqa: E402

TRADE_LAG_S, CANDLE_LAG_S = 4.0, 20.0


def ts(s: str | None) -> float | None:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def cut(doc: dict, t: float) -> dict:
    """The feed as it stood at t: events that had ended (pitches) or started (other events) by t; a plate
    appearance not yet over has no result; win probabilities of finished plate appearances only."""
    d = copy.deepcopy(doc)
    plays, done, score = [], set(), {"awayScore": 0, "homeScore": 0}
    for p in d["feed"]["liveData"]["plays"]["allPlays"]:
        ev = [e for e in p["playEvents"] if ((ts(e.get("endTime")) if e.get("isPitch") else ts(e.get("startTime"))) or 9e18) <= t]
        if not ev:
            break
        # over once its last event is (as training times it: the last pitch's end), not at the play's own end time
        complete = len(ev) == len(p["playEvents"])
        p["playEvents"] = ev
        p["about"]["isComplete"] = complete
        if complete:
            done.add(p["about"]["atBatIndex"])
            score = {k: p["result"].get(k, score[k]) for k in score}
        else:                                   # no result yet: the score before the plate appearance
            p["result"] = dict(score)
        plays.append(p)
        if not complete:
            break
    d["feed"]["liveData"]["plays"]["allPlays"] = plays
    d["win_probability"] = [w for w in doc["win_probability"] if w["atBatIndex"] in done]
    d["feed"]["gameData"]["status"]["detailedState"] = "In Progress"
    return d


def replay(eng: T.Engine, doc: dict, tr: list[dict], cd: list[dict], feed_lag: float | dict, lag_t: float, lag_c: float,
           snaps: list[tuple[float, Path]] | None = None) -> None:
    """Run a finished MLB game through eng on a simulated 1-s clock. tr: trades {ticker, trade_id, t, price, count,
    taker_side}; cd: 1-min candles {ticker, ts (end of the minute), bid, ask} in cents. Each arrives lag_t / lag_c
    after it exists, or at its own "arr" if it has one; the feed as it stood after a pitch arrives feed_lag after
    the pitch's end, or (feed_lag a dict) at feed_lag[pitch end], else pitch end + feed_lag["default"].
    snaps: [(arrival time, file)] of feed versions recorded live; the engine then gets those exact versions when
    they arrive (provisional pitch times, a ball in play before its result) instead of cuts of the final feed."""
    ends = sorted({ts(e["endTime"]) for p in doc["feed"]["liveData"]["plays"]["allPlays"] for e in p["playEvents"]
                   if e.get("isPitch") and e.get("endTime")})
    q = []                                     # (arrival time, order, kind, payload)
    for e in ([] if snaps else ends):
        q.append((feed_lag.get(e, e + feed_lag["default"]) if isinstance(feed_lag, dict) else e + feed_lag, len(q), "feed", e))
    for at, f in snaps or []:
        q.append((at, len(q), "snap", f))
    for r in tr:
        q.append((r.get("arr", r["t"] + lag_t), len(q), "trade", r))
    for c in cd:
        q.append((c.get("arr", c["ts"] + lag_c), len(q), "candle", c))
    heapq.heapify(q)
    t0, t_end = ends[0] - 1800, ends[-1] + 600
    clock = t0
    while clock <= t_end:
        trades, candles, feed_at = [], [], None
        while q and q[0][0] <= clock:
            at, _, kind, x = heapq.heappop(q)
            if kind in ("feed", "snap"):
                feed_at = (kind, x)
            elif kind == "trade":
                trades.append(dict(ticker=x["ticker"], trade_id=x["trade_id"], taker_side=x["taker_side"], count=x["count"],
                                   yes_price=x["price"], created_time=dt.datetime.fromtimestamp(x["t"], dt.timezone.utc).isoformat()))
            else:
                candles.append(dict(ticker=x["ticker"], end_period_ts=int(x["ts"]),
                                    received_at=dt.datetime.fromtimestamp(at, dt.timezone.utc).isoformat(),
                                    yes_bid={"close": None if x["bid"] is None else x["bid"] / 100},
                                    yes_ask={"close": None if x["ask"] is None else x["ask"] / 100}))
        if trades:
            eng.on_trades(trades)
        if candles:
            eng.on_candles(candles)
        if feed_at is not None:
            d = cut(doc, feed_at[1]) if feed_at[0] == "feed" else json.loads(gzip.decompress(feed_at[1].read_bytes()))
            eng.on_feed(d, dt.datetime.fromtimestamp(clock, dt.timezone.utc))
        eng.step(clock)
        clock += 1.0
    eng.final = True
    eng.step(clock + LM.STEP)


def main() -> None:
    import live_parity as P
    ap = argparse.ArgumentParser()
    ap.add_argument("game_id")
    ap.add_argument("--feed-lag", type=float, default=3.0)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--no-lag", action="store_true", help="every input arrives the moment it exists (a check against live_parity)")
    ap.add_argument("--out", type=Path, default=Path("data/sports_live"))
    a = ap.parse_args()
    gid = a.game_id
    panel = P.SP / "derived" / "ingame" / "MLB_pitch"
    gm = pl.read_parquet(P.SP / "pbp" / "games.parquet").filter(pl.col("game_id") == gid).row(0, named=True)
    con = pl.read_parquet(panel / "contracts.parquet").filter(pl.col("game_id") == gid, pl.col("kind").is_in(["winner", "spread", "total"]))
    # the Kalshi market rows the live engine reads kind / side / strike from: rebuild their titles from the panel
    markets = [dict(ticker=r["ticker"], series_ticker=r["series_ticker"], event_ticker=r["ticker"].rsplit("-", 1)[0],
                    yes_sub_title=f"over {r['strike']}" if r["strike"] is not None else None) for r in con.iter_rows(named=True)]
    g = dict(league="MLB", game_id=gid, start=gm["start_utc"].isoformat(), home=gm["home"], away=gm["away"],
             home_k=gm["home"], away_k=gm["away"], contracts=markets)
    wall = time.time()
    fc = LM.Forecaster("MLB", LM.ROOT / "results" / "ingame" / "deploy")
    root = a.out / "replay" / f"MLB_{gid}"
    for f in [*root.glob("*.jsonl"), *root.glob("*.jsonl.gz")]:
        f.unlink()
    eng = T.Engine(g, fc, root, T.KINDS_DEFAULT, a.delay, pub_live=False)
    assert len(eng.contracts) == con.height, (len(eng.contracts), con.height)

    doc = json.loads(gzip.decompress((P.SP / "pbp" / "raw" / "MLB" / f"{gid}.json.gz").read_bytes()))
    tr, cd = P.raw_market(con, None)
    lag = (0.0, 0.0, 0.0) if a.no_lag else (a.feed_lag, TRADE_LAG_S, CANDLE_LAG_S)
    replay(eng, doc, tr, cd, *lag)
    s = eng.summary()
    (root / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1), f"\n{time.time() - wall:.0f} s wall -> {root}")


if __name__ == "__main__":
    main()
