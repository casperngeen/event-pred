"""Check live_model.Market (incremental) against a full rebuild after every batch, feeding a recorded game's
trades and candles in the order the replay delivers them (the polled trades, then the stream's, so late rows,
duplicates and replaced candles all occur).

    venv/bin/python -W ignore scripts/test_live_market.py data/sports_live/sandbox/2026-10-12/MLB_849820 [--every 50]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_model as LM          # noqa: E402
import live_rerun as RR          # noqa: E402


def reference(mk: LM.Market) -> tuple[dict, dict]:
    """The arrays a from-scratch build of the same rows gives (the pre-2026-10-13 Market.build)."""
    q, p = {t: [] for t in mk.tickers}, {t: [] for t in mk.tickers}
    for r in mk.candles.values():
        q[r["ticker"]].append(((r["ts"], 0), 0.0 if r["bid"] is None else r["bid"], 100.0 if r["ask"] is None else r["ask"]))
    for r in mk.trades.values():
        yes = r["taker_side"] == "yes"
        key = (r["t"], 1, yes, r["price"] if yes else -r["price"], r["trade_id"])
        q[r["ticker"]].append((key, math.nan if yes else r["price"], r["price"] if yes else math.nan))
        p[r["ticker"]].append((r["t"], yes, r["price"] if yes else -r["price"], r["trade_id"], r["price"], r["count"]))
    Q, P = {}, {}
    for t in mk.tickers:
        o = sorted(q[t], key=lambda x: x[0])
        tt = np.array([x[0][0] for x in o], float)
        bid, ask = np.array([x[1] for x in o], float), np.array([x[2] for x in o], float)
        ib = np.maximum.accumulate(np.where(np.isfinite(bid), np.arange(len(o)), -1)) if len(o) else np.array([], int)
        ia = np.maximum.accumulate(np.where(np.isfinite(ask), np.arange(len(o)), -1)) if len(o) else np.array([], int)
        pick = lambda a, i: np.where(i >= 0, a[np.maximum(i, 0)], np.nan)    # noqa: E731
        Q[t] = dict(t=tt, bid=pick(bid, ib), bid_at=pick(tt, ib), ask=pick(ask, ia), ask_at=pick(tt, ia))
        o = sorted(p[t], key=lambda x: (x[0], x[1], x[2], x[3]))
        P[t] = dict(t=np.array([x[0] for x in o], float), yes=np.array([x[1] for x in o], bool),
                    px=np.array([x[4] for x in o], float), n=np.array([x[5] for x in o], float))
    return Q, P


def same(a: dict, b: dict) -> bool:
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--every", type=int, default=50, help="compare after every n-th batch")
    a = ap.parse_args()
    d = RR.load(a.root)
    tickers = [c["ticker"] for c in d["g"]["contracts"]]
    ev = [(r["arr"], "t", r) for r in d["poll"] + d["ws"]] + [(c["arr"], "c", c) for c in d["candles"]]
    ev += [(c["ts"] + 0.2, "c", c | {"bid": None if c["bid"] is None else c["bid"] + 1}) for c in d["candles"][::97]]   # changed candles
    ev.sort(key=lambda x: x[0])
    mk, n, checks = LM.Market(tickers), 0, 0
    i = 0
    while i < len(ev):                                  # one batch per simulated second, as the replay delivers
        sec = math.floor(ev[i][0])
        tr, cd = [], []
        while i < len(ev) and math.floor(ev[i][0]) == sec:
            (tr if ev[i][1] == "t" else cd).append(ev[i][2])
            i += 1
        mk.add_trades(tr)
        mk.add_candles(cd)
        n += 1
        if n % a.every == 0 or i == len(ev):
            mk.build()
            Q, P = reference(mk)
            bad = [t for t in tickers if not (same(Q[t], mk.q[t]) and same(P[t], mk.p[t]))]
            assert not bad, f"batch {n}: differs on {bad}"
            checks += 1
    print(f"ok: {n} batches, {len(mk.trades)} trades, {len(mk.candles)} candles, {checks} full comparisons identical")


if __name__ == "__main__":
    main()
