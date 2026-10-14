"""Re-run a game the live runner traded, from what was recorded that night, with each input switched between what
the runner had and a faster / fuller version, to see what each change does to the paper results.

    feed     live   each pitch arrives when the runner actually received it (plays_seen.jsonl)
             d10    pitch end + 10 s exactly (the backtest's assumption)
             fast   pitch end + --fast-lag s (statsapi without its CDN cache: ~7 s after the feed's version stamp,
                    which is ~1 s before the pitch's end, in a live test on 2026-10-13)
             (live / d10 / fast replay cuts of the finished game's feed, whose pitch times statsapi rewrote
             afterwards and which never shows a ball in play before its result; these replay the feed versions
             recorded that night instead, exactly as they read then:)
             snap   each recorded version when the runner first received it (feed_arrivals.jsonl)
             snapfast  each recorded version --snap-lag s after its version stamp (cache-busted statsapi; only the
                    versions the slower runner happened to fetch exist, so some in between are missing)
    trades   poll   the REST trades the runner got (trades.jsonl), when it got them (429s lost ~a quarter)
             ws     every trade on Kalshi's WebSocket (ws_trades.csv.gz), 0.14 s after it printed
             tape   the same trades the moment they print (the backtest's assumption)
    quotes   poll   the 1-min candles the runner got (candles.jsonl), when it got them (~20 s after the minute)
             book   the same minute-close quotes at the minute's end + 0.2 s, as the WebSocket order book gives them
                    (rebuilt from books_ws.jsonl, equal to the candles' bid and ask on every spread / total minute)
             tape   the same at the minute's end exactly (the backtest's assumption)
    fills    prints a resting limit fills only on a later print strictly through it (the frozen rule)
             queue  also on prints at the limit once the size resting ahead of it has traded; an order crossing
                    the book takes at once (live_trade.PaperBook with depth from book_levels.csv.gz / book_top.csv.gz)

    cancel   none   resting entries stay up to 60 s (the frozen rule)
             window a resting entry is cancelled once the last pitch we know of is 15-60 s old (the decision
                    skip window: the next pitch is likely), as the live runner could
             oracle a resting entry is cancelled at the next pitch's start (hindsight: the most this can do)
    sides    all    as the frozen rule; noshortover: never sell a total (the "over" side)
    inplay   inplay no decisions while the feed shows a ball in play without its result (the live engine since
                    2026-10-13); noinplay: without that rule. Only snap / snapfast feeds can show that state

Paper P&L per contract (net of fees); orders still open at the end are valued at the contract's result.

    venv/bin/python -u -W ignore scripts/live_rerun.py data/sports_live/sandbox/2026-10-12/MLB_849812 \\
        --scenario live,poll,poll,prints,none --scenario fast,ws,book,queue,window
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_model as LM          # noqa: E402
import live_record as R          # noqa: E402
import live_replay as RP         # noqa: E402
import live_trade as T           # noqa: E402

WS_LAG_S, BOOK_LAG_S = 0.14, 0.2


def ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class NoPublish:
    def send(self, signals: list[dict], t_ns: int) -> None:
        pass


class Depth:
    """The order book of the spread / total contracts as it stood at any time, from book_levels / book_top."""

    def __init__(self, root: Path) -> None:
        lv = pl.read_csv(root / "book_levels.csv.gz").sort("ms")
        self.lv = {k: (g["ms"].to_numpy(), g["qty"].to_numpy())
                   for k, g in lv.group_by(["ticker", "side", "price_c"])}
        tp = pl.read_csv(root / "book_top.csv.gz").sort("ms")
        self.tp = {k[0]: (g["ms"].to_numpy(), g["bid"].to_numpy(), g["ask"].to_numpy()) for k, g in tp.group_by(["ticker"])}

    def queue(self, ticker: str, side: str, price_c: int, t: float) -> float:
        x = self.lv.get((ticker, side, price_c))
        if x is None:
            return 0.0
        i = int(np.searchsorted(x[0], t * 1000, side="right")) - 1
        return float(x[1][i]) if i >= 0 else 0.0

    def top(self, ticker: str, t: float) -> tuple[float | None, float | None]:
        x = self.tp.get(ticker)
        if x is None:
            return None, None
        i = int(np.searchsorted(x[0], t * 1000, side="right")) - 1
        if i < 0:
            return None, None
        b, a = x[1][i], x[2][i]
        return (None if np.isnan(b) else float(b)), (None if np.isnan(a) else float(a))


def load(root: Path, doc_file: Path | None = None) -> dict:
    """doc_file: the finished game's feed for the live / d10 / fast cuts; default the last version recorded,
    which can still carry provisional times (CLE@TB 2026-10-12: re-pull the archive for those)."""
    g = json.loads((root / "game.json").read_text())
    doc = json.loads(gzip.decompress((doc_file or sorted((root / "feed").glob("*.json.gz"))[-1]).read_bytes()))
    seen = {round(r["happened"], 3): r["received"] for r in R.rows(root / "plays_seen.jsonl")}
    poll = []
    for r in R.rows(root / "trades.jsonl"):
        px = r.get("yes_price_dollars")
        poll.append(dict(ticker=r["ticker"], trade_id=r["trade_id"], taker_side=r["taker_side"], t=ts(r["created_time"]),
                         price=float(px) * 100 if px is not None else float(r["yes_price"]),
                         count=float(r.get("count_fp", r.get("count"))), arr=ts(r["received_at"])))
    ws = pl.read_csv(root / "ws_trades.csv.gz", has_header=False, new_columns=["ticker", "ms", "p", "n", "side"])
    ws = [dict(ticker=t, trade_id=f"ws{i}", taker_side=sd, t=ms / 1000, price=p * 100, count=n, arr=ms / 1000 + WS_LAG_S)
          for i, (t, ms, p, n, sd) in enumerate(ws.iter_rows())]
    cd = {}
    for c in R.rows(root / "candles.jsonl"):
        got = ts(c["received_at"])
        if c["end_period_ts"] > got:                 # the minute in progress
            continue
        v = lambda s: (c.get(s) or {}).get("close_dollars")          # noqa: E731
        k = (c["ticker"], c["end_period_ts"])
        if k not in cd:                               # the first time we had the finished minute
            cd[k] = dict(ticker=c["ticker"], ts=float(c["end_period_ts"]), arr=got,
                         bid=None if v("yes_bid") is None else float(v("yes_bid")) * 100,
                         ask=None if v("yes_ask") is None else float(v("yes_ask")) * 100)
    first = {}
    for a in R.rows(root / "feed_arrivals.jsonl"):
        f = root / "feed" / f"{a['stamp']}.json.gz"
        if a["stamp"] not in first and f.exists():
            first[a["stamp"]] = (ts(a["received_at"]), f)
    return dict(g=g, doc=doc, seen=seen, poll=poll, ws=ws, candles=list(cd.values()), snaps=first)


def results(tickers: list[str]) -> dict[str, float]:
    api = R.Api(rate=4.0)
    out = {}
    for t in tickers:
        m = (api.get(f"/markets/{t}", {}) or {}).get("market") or {}
        if m.get("result") in ("yes", "no"):
            out[t] = 100.0 if m["result"] == "yes" else 0.0
    return out


def run(d: dict, scen: tuple[str, ...], fast_lag: float, depth: Depth | None, out: Path, fc, snap_lag: float = 7.0,
        delay: float = 0.0) -> T.Engine:
    feed, trades, quotes, fills = scen[:4]
    cancel = scen[4] if len(scen) > 4 else "none"
    sides = scen[5] if len(scen) > 5 else "all"
    inplay = scen[6] if len(scen) > 6 else "inplay"
    doc = d["doc"]
    ends = sorted({ts(e["endTime"]) for p in doc["feed"]["liveData"]["plays"]["allPlays"] for e in p["playEvents"]
                   if e.get("isPitch") and e.get("endTime")})
    snaps = None
    if feed == "snap":
        snaps = sorted(d["snaps"].values())
    elif feed == "snapfast":
        stamp = lambda s: dt.datetime.strptime(s, "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()  # noqa: E731
        snaps = sorted((stamp(s) + snap_lag, f) for s, (_, f) in d["snaps"].items())
    if feed in ("live", "snap", "snapfast"):
        lag = {"default": float(np.median([d["seen"][round(e, 3)] - e for e in ends if round(e, 3) in d["seen"]]))}
        lag |= {e: d["seen"][round(e, 3)] for e in ends if round(e, 3) in d["seen"]}
    else:
        lag = 10.0 if feed == "d10" else fast_lag
    tr = d["poll"] if trades == "poll" else d["ws"] if trades == "ws" else [r | {"arr": r["t"]} for r in d["ws"]]
    cd = (d["candles"] if quotes == "poll" else
          [c | {"arr": c["ts"] + (BOOK_LAG_S if quotes == "book" else 0.0)} for c in d["candles"]])
    root = out / "_".join(scen)
    root.mkdir(parents=True, exist_ok=True)
    for f in [*root.glob("*.jsonl"), *root.glob("*.jsonl.gz")]:
        f.unlink()
    eng = T.Engine(d["g"], fc, root, T.KINDS_DEFAULT, 0.0, pub_live=False)
    eng.pub = NoPublish()
    eng.book.depth = depth if fills == "queue" else None
    eng.no_short_over = sides == "noshortover"
    eng.skip_in_play = inplay == "inplay"
    eng.live_clock = feed in ("snap", "snapfast")      # final-feed cuts already carry the archive's times
    eng.delay = delay
    eng.shadow = False                               # the 10 s-delay forecast is only logged, never traded on
    if cancel == "window":
        eng.cancel_rule = "window"
    elif cancel == "oracle":
        starts = np.array(sorted(ts(e["startTime"]) for p in doc["feed"]["liveData"]["plays"]["allPlays"]
                                 for e in p["playEvents"] if e.get("isPitch") and e.get("startTime")))
        eng.cancel_rule = lambda t0: float(starts[np.searchsorted(starts, t0, side="right")]) if t0 < starts[-1] else np.inf
    RP.replay(eng, doc, tr, cd, lag, RP.TRADE_LAG_S, RP.CANDLE_LAG_S, snaps)
    return eng


def score(eng: T.Engine, res: dict[str, float]) -> dict:
    o = eng.book.orders
    filled = [x for x in o if x["stage"] in ("hold", "exit", "closed", "settle")]
    net = []
    for x in filled:
        if x["stage"] == "closed":
            net.append(x["net"])
        elif x["ticker"] in res:                     # held to the result (or still open at the end)
            net.append(x["side"] * (res[x["ticker"]] - x.get("entry", x["limit"])) - x["entry_fee"])
    return dict(orders=len(o), filled=len(filled), closed=sum(x["stage"] == "closed" for x in o),
                to_result=len(filled) - sum(x["stage"] == "closed" for x in o),
                fill_rate=round(len(filled) / len(o), 3) if o else None,
                net_c_total=round(float(np.sum(net)), 2), net_c_per_fill=round(float(np.mean(net)), 2) if net else None,
                feed_delay_median_s=round(float(np.nanmedian(eng.lags)), 1) if eng.lags else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--scenario", action="append", required=True, help="feed,trades,quotes,fills[,cancel[,sides[,inplay]]]")
    ap.add_argument("--fast-lag", type=float, default=6.0)
    ap.add_argument("--snap-lag", type=float, default=7.0)
    ap.add_argument("--delay", type=float, default=0.0, help="the engine's --delay: a pitch counts once its end + delay has passed")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--doc", type=Path, help="the finished game's feed (.json.gz) to cut, e.g. re-pulled from statsapi")
    a = ap.parse_args()
    wall = time.time()
    d = load(a.root, a.doc)
    scens = [tuple(s.split(",")) for s in a.scenario]
    depth = Depth(a.root) if any(s[3] == "queue" for s in scens) else None
    res = results([c["ticker"] for c in d["g"]["contracts"]])
    fc = LM.Forecaster(d["g"]["league"], LM.ROOT / "results" / "ingame" / "deploy")
    out = a.out or a.root / ("rerun" if not a.delay else f"rerun_delay{a.delay:g}")
    for s in scens:
        t0 = time.time()
        eng = run(d, s, a.fast_lag, depth, out, fc, a.snap_lag, a.delay)
        r = dict(scenario=",".join(s)) | score(eng, res) | dict(wall_s=round(time.time() - t0))
        (out / "_".join(s) / "score.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r), flush=True)
    print(f"{time.time() - wall:.0f} s")


if __name__ == "__main__":
    main()
