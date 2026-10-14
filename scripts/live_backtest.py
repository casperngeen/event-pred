"""Backtest the live runner on finished MLB games fetched straight from the public APIs (no local data), so it can
run in a Lumid FlowMesh job: the statsapi feed and win probability, the game's Kalshi contracts, all their trades
and 1-min candles. Each game is replayed through live_trade.Engine as if live (live_replay.replay: the feed as it
stood after each pitch, arriving --feed-lag s later; trades 4 s, candles 20 s after they exist). Nothing is
published. Only games whose contracts settled after Kalshi's historical cutoff (they are on the live endpoints)
are fetched; the deployed models were trained on games up to 2026-06-30, so later games are out of sample.

    python scripts/live_backtest.py --from 2026-09-20 --to 2026-09-21 [--games 776543 ...] [--out DIR] [--budget-min 50]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_model as LM          # noqa: E402
import live_record as R          # noqa: E402
import live_replay as RP         # noqa: E402
import live_trade as T           # noqa: E402

UTC = dt.timezone.utc


class NoPublish:
    def send(self, signals: list[dict], t_ns: int) -> None:
        pass


def iso_ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def fetch_game(api: R.Api, g: dict, day: dt.date) -> tuple[dict, list[dict], list[dict]] | None:
    """The finished game's feed document, trades and candles (cents), or None if it is not usable."""
    st = dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00"))
    win = {"min_close_ts": int((st - dt.timedelta(days=1)).timestamp()), "max_close_ts": int((st + dt.timedelta(days=3)).timestamp())}
    g["contracts"] = R.contracts(api, g, day, win)
    if not g["contracts"]:
        return None
    pk = g["game_id"]
    feed = httpx.get(f"{R.MLB_API}/api/v1.1/game/{pk}/feed/live", timeout=60).json()
    if feed["gameData"]["status"]["detailedState"] not in R.FINAL:
        return None
    wp = httpx.get(f"{R.MLB_API}/api/v1/game/{pk}/winProbability", timeout=60).json() or []
    wp = [{k: w.get(k) for k in ("atBatIndex", "homeTeamWinProbability", "awayTeamWinProbability",
                                 "homeTeamWinProbabilityAdded")} for w in wp]
    doc = {"feed": feed, "win_probability": wp}
    tr = []
    for m in g["contracts"]:
        for r in api.paged("/markets/trades", "trades", {"ticker": m["ticker"]}, 1000):
            px = r.get("yes_price_dollars")
            tr.append(dict(ticker=r["ticker"], trade_id=r["trade_id"], taker_side=r["taker_side"], t=iso_ts(r["created_time"]),
                           price=float(px) * 100 if px is not None else float(r["yes_price"]),
                           count=float(r.get("count_fp", r.get("count")))))
    # 1-min candles from 3 h before the start to the last close, in batches of up to 100 tickers / 10,000 candles
    lo = int((st - dt.timedelta(hours=3)).timestamp())
    hi = min(int(max(iso_ts(m["close_time"]) for m in g["contracts"])), int(time.time()))
    per = max(1, 9000 // ((hi - lo) // 60 + 1))
    cd, tick = [], [m["ticker"] for m in g["contracts"]]
    for i in range(0, len(tick), per):
        d = api.get("/markets/candlesticks", {"market_tickers": ",".join(tick[i:i + per]), "start_ts": lo, "end_ts": hi,
                                              "period_interval": 1}) or {}
        for mk in d.get("markets") or []:
            for c in mk.get("candlesticks") or []:
                b, a = [(c.get(s) or {}).get("close_dollars", (c.get(s) or {}).get("close")) for s in ("yes_bid", "yes_ask")]
                cd.append(dict(ticker=mk["market_ticker"], ts=float(c["end_period_ts"]),
                               bid=None if b is None else float(b) * 100, ask=None if a is None else float(a) * 100))
    return doc, tr, cd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--to", dest="d1", type=dt.date.fromisoformat)
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--out", type=Path, default=R.OUT)
    ap.add_argument("--weights", default=str(LM.ROOT / "results" / "ingame" / "deploy"))
    ap.add_argument("--kinds", default=",".join(T.KINDS_DEFAULT))
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--feed-lag", type=float, default=3.0)
    ap.add_argument("--budget-min", type=float, default=50.0, help="start no new game after this many minutes")
    a = ap.parse_args()
    wall = time.time()
    api = R.Api(rate=4.0)
    cutoff = iso_ts((api.get("/historical/cutoff", {}) or {})["market_settled_ts"])
    fc = LM.Forecaster("MLB", a.weights)
    days = [a.d0 + dt.timedelta(days=i) for i in range(((a.d1 or a.d0) - a.d0).days + 1)]
    out = []
    for day in days:
        for g in R.schedule("MLB", day):
            if a.games and g["game_id"] not in a.games:
                continue
            if (time.time() - wall) / 60 > a.budget_min:
                print(f"budget reached, skipping {g['game_id']}", flush=True)
                out.append(dict(game=g["game_id"], skipped="budget"))
                continue
            t0 = time.time()
            if iso_ts(g["start"]) < cutoff:
                out.append(dict(game=g["game_id"], skipped="before Kalshi historical cutoff"))
                continue
            got = fetch_game(api, g, day)
            if got is None:
                out.append(dict(game=g["game_id"], skipped="no contracts or not final"))
                print(f"{g['game_id']} {g['away']}@{g['home']}: skipped", flush=True)
                continue
            doc, tr, cd = got
            t1 = time.time()
            root = a.out / "backtest" / f"MLB_{g['game_id']}"
            for f in [*root.glob("*.jsonl"), *root.glob("*.jsonl.gz")]:
                f.unlink()
            eng = T.Engine(g, fc, root, tuple(a.kinds.split(",")), a.delay, pub_live=False)
            eng.pub = NoPublish()
            RP.replay(eng, doc, tr, cd, a.feed_lag, RP.TRADE_LAG_S, RP.CANDLE_LAG_S)
            s = eng.summary() | dict(date=day.isoformat(), match=f"{g['away']}@{g['home']}", contracts=len(g["contracts"]),
                                     modelled=len(eng.contracts), trades=len(tr), candles=len(cd),
                                     fetch_s=round(t1 - t0), replay_s=round(time.time() - t1))
            (root / "summary.json").write_text(json.dumps(s, indent=1))
            out.append(s)
            print(json.dumps(s), flush=True)
    done = [s for s in out if "skipped" not in s]
    tot = dict(games=len(done), orders=sum(s["orders"] for s in done), filled=sum(s["filled"] for s in done),
               closed=sum(s["closed"] for s in done), pnl_closed_c=sum(s["pnl_closed_c"] or 0 for s in done),
               wall_s=round(time.time() - wall))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "backtest" / "summary.json").write_text(json.dumps(dict(total=tot, games=out), indent=1))
    print("TOTAL", json.dumps(tot), flush=True)


if __name__ == "__main__":
    main()
