"""Price-time-priority fills for the paper book, from the recorded WebSocket order book (book_events.csv.gz, written by
extract_book_events.py): our resting limit joins the back of the queue at its price and moves forward as the orders ahead
of it trade or cancel.

    Events at a price level, in time order (a level = one side's bid price; a buy of yes joins the yes bids, a sale of
    yes joins the no bids at 100 - price):
      add      size joins behind us (a book increase)
      execute  a trade consumed the level: the front of the queue goes first, then us. A book decrease within TIE_MS of a
               trade at that level (up to the trade's size) is the trade's execution; every trade in the 24 Oct NBA data
               had one in the same millisecond
      cancel   any other book decrease. Where in the queue it came from is not recorded, so CANCEL says how to place it:
                 prop    uniformly over the queue (ahead shrinks by the cancelled share of the level)
                 ahead   all of it from in front of us (the optimistic bound)
                 behind  from behind us first (the pessimistic bound)
    A trade strictly through our price also fills us (price priority), whatever the accounting says.
    A fill needs the executions that reach us to cover our whole size (whole contracts, 1-5).

    book = QueueBook(log, BookEvents(path), cancel="prop")   # in place of T.PaperBook(log)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_trade as T           # noqa: E402

TIE_MS = 50
ADD, EXEC, CANCEL, THRU, SNAP = 0, 1, 2, 3, 4


def key(c: float) -> int:
    return int(round(float(c) * 100))


class BookEvents:
    """One game's classified level events, replayed in time order against a growing set of watched orders."""

    def __init__(self, path: Path, cancel: str = "prop") -> None:
        E = pd.read_csv(path)
        E["i"] = np.arange(len(E))
        E["pk"] = (E.price_c * 100).round().astype(np.int64)
        T_ = E[E.kind == "T"]
        # the level a trade consumed: a taker buying yes hits a no bid at 100 - yes price; a taker selling yes hits a yes bid
        t_side = np.where(T_.side == "yes", "no", "yes")
        t_pk = np.where(T_.side == "yes", 10000 - T_.pk, T_.pk)
        D = E[E.kind == "D"]
        neg = D[D.qty < 0]
        left = {k: [v.ts_ms.values, (-v.qty.values).copy(), v.i.values] for k, v in neg.groupby(["ticker", "side", "pk"])}
        used: dict[int, float] = {}                                  # row index of a decrease -> volume matched to trades
        for tk, sd, pk, ts, n in zip(T_.ticker, t_side, t_pk, T_.ts_ms, T_.qty):
            x = left.get((tk, sd, int(pk)))
            if x is None:
                continue
            lo, hi = np.searchsorted(x[0], ts - TIE_MS), np.searchsorted(x[0], ts + TIE_MS, side="right")
            for j in range(lo, hi):
                if n <= 0:
                    break
                take = min(x[1][j], n)
                if take > 0:
                    x[1][j] -= take
                    n -= take
                    used[int(x[2][j])] = used.get(int(x[2][j]), 0.0) + take
        ev = []                                                      # (ts, order, code, ticker, side, pk, qty)
        for r in E[E.kind == "S"].itertuples():
            ev.append((r.ts_ms, r.i, SNAP, r.ticker, r.side, r.pk, r.qty))
        for r in D.itertuples():
            if r.qty > 0:
                ev.append((r.ts_ms, r.i, ADD, r.ticker, r.side, r.pk, r.qty))
            else:
                ex = used.get(r.i, 0.0)
                if ex > 0:
                    ev.append((r.ts_ms, r.i, EXEC, r.ticker, r.side, r.pk, ex))
                if -r.qty - ex > 1e-9:
                    ev.append((r.ts_ms, r.i, CANCEL, r.ticker, r.side, r.pk, -r.qty - ex))
        for tk, sd, pk, ts, i in zip(T_.ticker, t_side, t_pk, T_.ts_ms, T_.i):
            ev.append((ts, i, THRU, tk, sd, int(pk), 0.0))
        ev.sort(key=lambda e: (e[0], e[1], e[2]))
        self.ev = ev
        self.reset(cancel)

    def reset(self, cancel: str) -> None:
        """Start the replay over (the classified events are kept) with another cancellation assumption."""
        self.ptr = 0
        self.size: dict[tuple, float] = {}
        self.watch: dict[tuple, list] = {}                           # (ticker, side, pk) -> queue records
        self.by_ticker: dict[tuple, list] = {}                       # (ticker, side) -> watched levels' records
        self.cancel = cancel
        self.stats = dict(add=0.0, exec=0.0, cancel=0.0)

    def top(self, ticker: str, side: str) -> int | None:
        """The highest price (key) bid on that side, or None."""
        ps = [k[2] for k, v in self.size.items() if k[0] == ticker and k[1] == side and v > 1e-9]
        return max(ps) if ps else None

    def join(self, ticker: str, side: str, pk: int, need: float) -> dict:
        rec = dict(A=self.size.get((ticker, side, pk), 0.0), need=need, got=0.0, t_fill=None, k=(ticker, side, pk))
        self.watch.setdefault(rec["k"], []).append(rec)
        self.by_ticker.setdefault((ticker, side), []).append(rec)
        return rec

    def leave(self, rec: dict) -> None:
        self.watch[rec["k"]].remove(rec)
        self.by_ticker[rec["k"][:2]].remove(rec)

    def advance(self, now: float) -> None:
        ev, i, lim = self.ev, self.ptr, now * 1000
        n = len(ev)
        while i < n and ev[i][0] <= lim:
            ts, _, code, tk, sd, pk, q = ev[i]
            k = (tk, sd, pk)
            i += 1
            if code == SNAP:
                self.size[k] = q
            elif code == ADD:
                self.size[k] = self.size.get(k, 0.0) + q
                self.stats["add"] += q
            elif code == EXEC:
                s0 = self.size.get(k, 0.0)
                self.size[k] = max(s0 - q, 0.0)
                self.stats["exec"] += q
                for r in self.watch.get(k, ()):
                    if r["t_fill"] is not None:
                        continue
                    if q <= r["A"]:
                        r["A"] -= q
                    else:
                        r["got"] += q - r["A"]
                        r["A"] = 0.0
                        if r["got"] >= r["need"] - 1e-9:
                            r["t_fill"] = ts / 1000
            elif code == CANCEL:
                s0 = self.size.get(k, 0.0)
                self.size[k] = max(s0 - q, 0.0)
                self.stats["cancel"] += q
                for r in self.watch.get(k, ()):
                    if r["t_fill"] is not None or s0 <= 1e-9:
                        continue
                    if self.cancel == "prop":
                        r["A"] *= max(1.0 - q / s0, 0.0)
                    elif self.cancel == "ahead":
                        r["A"] = max(r["A"] - q, 0.0)
                    else:                                            # behind: only what exceeds the queue behind us hits the front
                        r["A"] = max(r["A"] - max(q - max(s0 - r["A"], 0.0), 0.0), 0.0)
            else:                                                    # a trade through our price fills us
                for r in self.by_ticker.get((tk, sd), ()):
                    if r["t_fill"] is None and r["k"][2] > pk:
                        r["t_fill"] = ts / 1000
        self.ptr = i


class QueueBook(T.PaperBook):
    """PaperBook whose entry and exit limits fill by queue position in the recorded book."""

    def __init__(self, log, events: BookEvents) -> None:
        super().__init__(log, depth=None)
        self.ev = events

    @staticmethod
    def _level(rest: int, price: float) -> tuple[str, int]:
        return ("yes", key(price)) if rest == 1 else ("no", key(100 - price))

    def _post(self, ticker: str, rest: int, price: float, t: float):
        """(the queue record or None, the price taken at once if the limit crosses the book else None)."""
        side, pk = self._level(rest, price)
        other = self.ev.top(ticker, "no" if rest == 1 else "yes")        # best opposite bid
        if other is not None:
            ask = (10000 - other) / 100 if rest == 1 else other / 100     # yes ask / yes bid in cents
            if (rest == 1 and ask <= price) or (rest == -1 and ask >= price):
                return None, float(ask)
        return self.ev.join(ticker, side, pk, 1.0), None

    def place(self, d: dict) -> None:
        o = dict(id=len(self.orders), ticker=d["ticker"], kind=d["kind"], t0=d["t"], side=d["side"], limit=d["limit"],
                 size=d["size"], signal=d["signal"], stage="entry")
        o["q"], took = self._post(o["ticker"], o["side"], o["limit"], o["t0"])
        if o["q"] is not None:
            o["q"]["need"] = float(o["size"])
            o["ahead"] = o["q"]["A"]
        else:
            o["ahead"] = 0.0
        self.orders.append(o)
        self.busy[o["ticker"]] = o
        self.log("orders.jsonl", dict(event="placed", **{k: v for k, v in o.items() if k != "q"}))
        if took is not None:
            o.update(stage="hold", t_fill=o["t0"], entry=took, entry_fee=T.fee(T.TAKER, took))
            self.log("orders.jsonl", dict(event="filled", id=o["id"], t=o["t_fill"], price=took, how="crossed"))

    def update(self, mk, now: float) -> None:
        mk.build()
        self.ev.advance(now)
        for o in self.orders:
            if o["stage"] in ("closed", "expired", "settle"):
                continue
            p, s = mk.p[o["ticker"]], o["side"]
            if o["stage"] == "entry":
                end = min(o["t0"] + T.ENTRY_S, o.get("cancel_at", math.inf))
                q = o.get("q")
                if q is not None and q["t_fill"] is not None and q["t_fill"] <= end:
                    o.update(stage="hold", t_fill=q["t_fill"], entry_fee=T.fee(T.MAKER, o["limit"]))
                    self.ev.leave(q)
                    self.log("orders.jsonl", dict(event="filled", id=o["id"], t=o["t_fill"], price=o["limit"]))
                elif now > end + T.GRACE_S:
                    o["stage"] = "expired"
                    if q is not None:
                        self.ev.leave(q)
                    self.busy[o["ticker"]] = None
                    self.log("orders.jsonl", dict(event="expired" if end == o["t0"] + T.ENTRY_S else "cancelled", id=o["id"], t=end))
            if o["stage"] == "hold" and now > o["t_fill"] + T.HOLD_S + T.GRACE_S:
                te = o["t_fill"] + T.HOLD_S
                n = int(np.searchsorted(p["t"], te, side="right"))
                ref_e = float(p["px"][n - 1]) if n else o["limit"]
                o.update(stage="exit", te=te, exit_limit=ref_e + T.OFFSET * s)
                self.ev.advance(te)
                o["xq"], o["exit_took"] = (self._post(o["ticker"], -s, o["exit_limit"], te) if 0 < o["exit_limit"] < 100 else (None, None))
                if o["xq"] is not None:
                    o["xq"]["need"] = float(o["size"])
                self.log("orders.jsonl", dict(event="exit posted", id=o["id"], t=te, price=o["exit_limit"]))
            if o["stage"] == "exit":
                lx, te, xq = o["exit_limit"], o["te"], o.get("xq")
                if o.get("exit_took") is not None:
                    self._close(o, o["exit_took"], "crossed", te, T.fee(T.TAKER, o["exit_took"]))
                elif xq is not None and xq["t_fill"] is not None and xq["t_fill"] <= te + T.EXIT_S:
                    self.ev.leave(xq)
                    self._close(o, lx, "limit filled", xq["t_fill"], T.fee(T.MAKER, lx))
                elif now > te + T.EXIT_S + T.GRACE_S:
                    if xq is not None and xq in self.ev.watch.get(xq["k"], []):
                        self.ev.leave(xq)
                    x = self._first(p, te + T.EXIT_S, te + 2 * T.EXIT_S, taker_yes=(s == -1), price=None, above=False)
                    if x >= 0:
                        self._close(o, float(p["px"][x]), "taken", float(p["t"][x]), T.fee(T.TAKER, float(p["px"][x])))
                    elif now > te + 2 * T.EXIT_S + T.GRACE_S:
                        o["stage"] = "settle"
                        self.log("orders.jsonl", dict(event="to settlement", id=o["id"], t=te + 2 * T.EXIT_S))
