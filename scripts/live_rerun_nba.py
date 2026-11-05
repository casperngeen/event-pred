"""Re-run a recorded NBA/NFL game through live_trade.Engine, from what the runner recorded that night.

    live      the runner's own inputs: each ESPN feed version when it arrived (feed_arrivals.jsonl), trades when it
              received them (trades.jsonl received_at), candles when it received them (candles.jsonl). Should
              reproduce the live orders (a check on the replay itself).
    backtest  the backtest's assumptions: every play known exactly at its time + --delay (the finished feed is
              handed over before the game, so no arrival lag), trades the moment they print, candles at the
              minute's end. Fills still need a later print strictly through the limit (the frozen rule).

    venv/bin/python -u -W ignore scripts/live_rerun_nba.py data/sports_live/vm/2026-10-24/NBA_401909853 \\
        --scenario live --scenario backtest --delay 60 --model gru_d60_h30-60-120-1800_tp
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import heapq
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_model as LM          # noqa: E402
import live_record as R          # noqa: E402
import live_rerun as RR          # noqa: E402
import live_trade as T           # noqa: E402

ts = RR.ts


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()


def load(root: Path) -> dict:
    g = json.loads((root / "game.json").read_text())
    arrivals = []
    for a in R.rows(root / "feed_arrivals.jsonl"):
        f = root / "feed" / f"{a['stamp']}.json.gz"
        if f.exists():
            arrivals.append((ts(a["received_at"]), f))
    trades, seen = [], set()
    for r in R.rows(root / "trades.jsonl"):
        if r["trade_id"] not in seen:
            seen.add(r["trade_id"])
            trades.append(r)
    return dict(g=g, arrivals=sorted(arrivals), final=sorted((root / "feed").glob("*.json.gz"))[-1],
                trades=trades, candles=list(R.rows(root / "candles.jsonl")))


def run(d: dict, scen: str, fc, delay: float, out: Path, gate_true: bool = False, queue: str | None = None, src: Path | None = None,
        baseline: str | None = None, lag_s: float = 45.0, late_block: bool = False) -> T.Engine:
    root = out / scen
    root.mkdir(parents=True, exist_ok=True)
    for f in [*root.glob("*.jsonl"), *root.glob("*.jsonl.gz")]:
        f.unlink()
    eng = T.Engine(d["g"], fc, root, T.KINDS_DEFAULT, delay, pub_live=False)
    eng.pub, eng.shadow = RR.NoPublish(), False
    eng.gate_true = gate_true
    eng.late_block = late_block
    if baseline == "fade":                                  # no model: the "model value" is the last print 60 s ago (fade the last minute's move)
        decide = eng._decide

        def fade(t: float, now: float, rows: list[dict]) -> None:
            new = []
            for r in rows:
                p = eng.mk.p[r["ticker"]]
                i = int(np.searchsorted(p["t"], t - 60.0, side="right")) - 1
                j = int(np.searchsorted(p["t"], t, side="right")) - 1       # no print 60 s ago: no signal
                new.append(r | {"level": {T.HORIZON: float(p["px"][i] if i >= 0 else p["px"][j]) if j >= 0 else 50.0}})
            decide(t, now, new)
        eng._decide = fade
        eng.gate_true = False
    if baseline in ("lagged", "both"):                      # model value against the print LAG s ago (the price when the model's plays were priced)
        decide = eng._decide                                # lagged: signal = value - old print; both: smaller of that and the usual signal, if they agree

        def lagged(t: float, now: float, rows: list[dict]) -> None:
            new = []
            for r in rows:
                p = eng.mk.p[r["ticker"]]
                j = int(np.searchsorted(p["t"], t, side="right")) - 1
                i = int(np.searchsorted(p["t"], t - lag_s, side="right")) - 1
                if j < 0 or i < 0:
                    new.append(r)
                    continue
                ref, cur = float(p["px"][j]), r["level"][T.HORIZON]
                sig = cur - float(p["px"][i])
                if baseline == "both":
                    sig = math.copysign(min(abs(sig), abs(cur - ref)), sig) if sig * (cur - ref) > 0 else 0.0
                new.append(r | {"level": {T.HORIZON: ref + sig}})
            decide(t, now, new)
        eng._decide = lagged
        eng.gate_true = False
    if queue:                                               # price-time priority from the recorded book (live_queue.py)
        import live_queue as Q
        eng.book = Q.QueueBook(eng.log, Q.BookEvents(src / "book_events.csv.gz", queue))
    live = scen == "live"
    q = []                                                  # (arrival, order, kind, payload)
    if live:
        for at, f in d["arrivals"]:
            q.append((at, len(q), "feed", f))
        t0 = d["arrivals"][0][0] - 1
    else:
        # the finished feed is there just before the first play: the engine reads the pregame total (L, a model input) from the market
        # when the feed first arrives, so handing it over earlier (as the first version of this scenario did) gave L = NaN, which the
        # forecaster silently replaces with a fixed fallback
        doc0 = json.loads(gzip.decompress(d["final"].read_bytes()))
        start = dt.datetime.fromisoformat(d["g"]["start"].replace("Z", "+00:00"))
        if d["g"]["league"] == "NFL":
            class _Early(dict):                             # every play already seen long before it ended, so each counts at its own end
                def get(self, k, default=None): return 0.0
            first = LM.epoch(LM.nfl_live_plays(doc0, start, _Early(), delay)[0]["t"].min())
        else:
            first = LM.nba_live_plays(doc0, start, None, delay)[1]
        t0 = first - 30
        q.append((t0, len(q), "feed", d["final"]))
    for r in d["trades"]:
        q.append((ts(r["received_at"]) if live else ts(r["created_time"]), len(q), "trade", r))
    for c in d["candles"]:
        got = ts(c["received_at"])
        if c["end_period_ts"] > got:                        # the minute in progress
            continue
        q.append((got if live else float(c["end_period_ts"]), len(q), "candle", c))
    heapq.heapify(q)
    t_end = d["arrivals"][-1][0] + 600
    clock = t0
    cache: dict[Path, dict] = {}
    while clock <= t_end:
        trades, candles, feed = [], [], None
        while q and q[0][0] <= clock:
            at, _, kind, x = heapq.heappop(q)
            if kind == "feed":
                feed = x
            elif kind == "trade":
                trades.append(x)
            else:
                candles.append(x | {"received_at": iso(min(at, clock))})
        if trades:
            eng.on_trades(trades)
        if candles:
            eng.on_candles(candles)
        if feed is not None:
            eng.on_feed(json.loads(gzip.decompress(feed.read_bytes())), dt.datetime.fromtimestamp(clock, dt.timezone.utc))
        eng.step(clock)
        clock += 1.0
    eng.final = True
    eng.step(clock + LM.STEP)
    return eng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--scenario", action="append", required=True, choices=["live", "backtest"])
    ap.add_argument("--delay", type=float, default=60.0)
    ap.add_argument("--model", default="gru_d60_h30-60-120-1800_tp")
    ap.add_argument("--gate-true", action="store_true", help="skip decisions whose signal differs on real-time stamps")
    ap.add_argument("--queue", choices=["prop", "ahead", "behind"], help="price-time-priority fills from book_events.csv.gz; where a cancellation sits in the queue")
    ap.add_argument("--baseline", choices=["fade", "lagged", "both"], help="replace the model by a rule (fade: the last print 60 s ago is the 'value'); "
                    "lagged / both: the model's value against the print --lag s ago (both also needs the usual signal to agree)")
    ap.add_argument("--lag", type=float, default=45.0)
    ap.add_argument("--late-block", action="store_true", help="no new entries in the late-game window (live_trade.LATE_RULE)")
    ap.add_argument("--scale", type=float, help="size = scale x weight to the nearest 0.01 (0 = no order), instead of whole contracts")
    ap.add_argument("--contracts", type=int, help="whole contracts per order at full weight, as the 24-26 Oct runners ran (K = 5); default: fractional weight")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    T.CONTRACTS_K = a.contracts
    if a.scale:
        T.contracts_of = lambda weight, k: math.floor(a.scale * weight * 100 + 0.5) / 100
    d = load(a.root)
    fc = LM.Forecaster(d["g"]["league"], LM.ROOT / "results" / "ingame" / "deploy", name=a.model)
    out = a.out or a.root / "rerun"
    for s in a.scenario:
        t0 = time.time()
        eng = run(d, s, fc, a.delay, out, a.gate_true, a.queue, a.root, a.baseline, a.lag, a.late_block)
        print(json.dumps(dict(scenario=s, wall_s=round(time.time() - t0)) | eng.summary()), flush=True)


if __name__ == "__main__":
    main()
