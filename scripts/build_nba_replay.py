"""Build replay folders for finished NBA games from the public APIs, in the layout live_record writes, so
live_rerun_nba.py can re-run them (scenario "backtest": the finished feed, trades at their print time, candles at
the minute's end). One folder per game under --out/<ET date>/NBA_<espn id>: game.json, one ESPN summary in feed/,
feed_arrivals.jsonl (one row, the game's end), trades.jsonl, candles.jsonl. Only games whose contracts settled after
Kalshi's historical cutoff (they are on the live endpoints) are built.

    venv/bin/python -u -W ignore scripts/build_nba_replay.py --from 2026-10-21 --to 2026-10-23 [--out DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_record as R          # noqa: E402


def iso_ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def build(api: R.Api, g: dict, day: dt.date, root: Path) -> str:
    st = dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00"))
    win = {"min_close_ts": int((st - dt.timedelta(days=1)).timestamp()), "max_close_ts": int((st + dt.timedelta(days=10)).timestamp())}
    g["contracts"] = R.contracts(api, g, day, win)
    if not g["contracts"]:
        return "no contracts"
    doc = httpx.get(f"{R.ESPN_LEAGUE['NBA']}/summary", params={"event": g["game_id"]}, timeout=60).json()
    state = ((((doc.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type") or {}).get("state")
    if state != "post":
        return f"not final ({state})"
    root.mkdir(parents=True, exist_ok=True)
    (root / "feed").mkdir(exist_ok=True)
    (root / "game.json").write_text(json.dumps(g))
    end = st + dt.timedelta(hours=4)
    stamp = end.strftime("%Y%m%d_%H%M%S_000000")
    (root / "feed" / f"{stamp}.json.gz").write_bytes(gzip.compress(json.dumps(doc).encode()))
    (root / "feed_arrivals.jsonl").write_text(json.dumps(dict(stamp=stamp, received_at=R.iso(end), plays=len(doc.get("plays") or []))) + "\n")
    n = 0
    with open(root / "trades.jsonl", "w") as f:
        for m in g["contracts"]:
            for r in api.paged("/markets/trades", "trades", {"ticker": m["ticker"]}, 1000):
                f.write(json.dumps(r | {"received_at": r["created_time"], "via": "hist"}) + "\n")
                n += 1
    lo = int((st - dt.timedelta(hours=3)).timestamp())
    hi = min(int(max(iso_ts(m["close_time"]) for m in g["contracts"])), int(time.time()))
    hi = min(hi, int((st + dt.timedelta(hours=6)).timestamp()))
    per = max(1, 9000 // ((hi - lo) // 60 + 1))
    tick = [m["ticker"] for m in g["contracts"]]
    with open(root / "candles.jsonl", "w") as f:
        for i in range(0, len(tick), per):
            d = api.get("/markets/candlesticks", {"market_tickers": ",".join(tick[i:i + per]), "start_ts": lo, "end_ts": hi,
                                                  "period_interval": 1}) or {}
            for mk in d.get("markets") or []:
                for c in mk.get("candlesticks") or []:
                    f.write(json.dumps({"ticker": mk.get("market_ticker") or mk.get("ticker"),
                                        "received_at": R.iso(dt.datetime.fromtimestamp(c["end_period_ts"] + 1, dt.timezone.utc))} | c) + "\n")
    return f"ok: {len(g['contracts'])} contracts, {n} trades"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--to", dest="d1", type=dt.date.fromisoformat)
    ap.add_argument("--out", type=Path, default=R.OUT / "replay_hist")
    a = ap.parse_args()
    api = R.Api(rate=12.0)
    for k in range(((a.d1 or a.d0) - a.d0).days + 1):
        day = a.d0 + dt.timedelta(days=k)
        for g in R.schedule("NBA", day):
            root = a.out / day.isoformat() / f"NBA_{g['game_id']}"
            if (root / "candles.jsonl").exists():
                continue
            t0 = time.time()
            print(day, g["game_id"], g["away"], "@", g["home"], build(api, g, day, root), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
