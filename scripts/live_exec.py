"""The exchange simulator (event_execution.Execution) driven by the live order-book stream instead of a recording.

The recorder (live_books_ws.py) already writes every WebSocket message of the game's contracts to books_ws.jsonl; this
module tails that file, so decisions and recordings share one stream and a night can be replayed exactly.

Clock: the book is on the exchange clock (event_execution.book_ts). A message becomes known to us about LAG + flush
seconds after its exchange time, so the book is only advanced to wall time minus SAFETY; an order sent at wall time T
reaches the exchange at T + latency and is applied once the book has caught up to that moment (messages that still
arrive later than SAFETY are applied late and counted in stats['late_messages']).

    eng.book = LiveExecution(eng.log, root / "books_ws.jsonl", tickers, latency=.09)
    eng.book.ref = ...                       # last print price before t, as in b49_event_replay_v2
"""
from __future__ import annotations

import heapq
import json
import threading
import time
from pathlib import Path

import numpy as np

import event_execution as X

SAFETY = 1.0                 # s behind wall time the book is advanced to (receive lag + recorder flush + polling)
BATCH, INBOX_MAX = 100_000, 150_000       # messages applied per update; the tail waits while this many are unapplied (catching up after a restart)
LATENCY = {"vm": .015, "sandbox": .09}       # decision -> exchange acceptance, from the measured API round trips


class BookTail(threading.Thread):
    """Follow a growing books_ws.jsonl and hand each complete line of the wanted tickers to `sink(ts, message)`."""

    def __init__(self, path: Path, tickers: set[str], sink, poll: float = .1) -> None:
        super().__init__(daemon=True)
        self.path, self.tickers, self.sink, self.poll = Path(path), set(tickers), sink, poll
        self.stop_event = threading.Event()
        self.bad = 0

    def run(self) -> None:
        while not self.path.exists() and not self.stop_event.is_set():
            time.sleep(1)
        with self.path.open("rb") as f:
            rest = b""
            while not self.stop_event.is_set():
                chunk = f.read(1 << 20)
                if not chunk:
                    time.sleep(self.poll)
                    continue
                *lines, rest = (rest + chunk).split(b"\n")
                for line in lines:
                    self.take(line)

    def take(self, line: bytes) -> None:
        if b"market_ticker" not in line:
            return
        try:
            m = json.loads(line)
        except ValueError:
            self.bad += 1
            return
        if m.get("type") in ("orderbook_snapshot", "orderbook_delta", "trade") and (m.get("msg") or {}).get("market_ticker") in self.tickers:
            self.sink(X.book_ts(m), m)


class LiveExecution(X.Execution):
    """Execution fed incrementally. The engine calls update(mk, now) every second and place()/cancel_entries() as before."""

    def __init__(self, log, books_path: Path | None, tickers, latency: float, safety: float = SAFETY, shadow=None, id0: int = 0, **kw) -> None:
        """shadow: the engine's older paper book (live_trade.PaperBook), given every order as well so both fill models see the same orders."""
        super().__init__(log, latency, **kw)
        self.safety, self.inbox, self.wall, self.serial_in = safety, [], 0., 0
        self.shadow, self.id0 = shadow, id0
        self.lock = threading.Lock()
        self.tail = None
        if books_path is not None:
            self.tail = BookTail(books_path, set(tickers), self.feed)
            self.tail.start()

    def feed(self, ts: float, m: dict) -> None:
        while len(self.inbox) > INBOX_MAX and self.tail is not None and not self.tail.stop_event.is_set():
            time.sleep(.05)                              # backlog (a restart replays the file): do not hold the whole game in memory
        with self.lock:
            self.serial_in += 1
            heapq.heappush(self.inbox, (ts, self.serial_in, m))
            self.stats["arrival_delay_sum_ms"] += max(0., (time.time() - ts) * 1000)
            self.stats["arrival_delay_n"] += 1

    def update(self, mk, now: float) -> None:
        self.wall = max(self.wall, now)
        horizon = max(self.now, now - self.safety)
        with self.lock:
            batch = []
            while self.inbox and self.inbox[0][0] <= horizon and len(batch) < BATCH:
                ts, _, m = heapq.heappop(self.inbox)
                if ts < self.now - 1e-6:
                    self.stats["late_messages"] += 1       # known only after the clock passed it
                batch.append((ts, m))
            if self.inbox and self.inbox[0][0] <= horizon:   # more to apply: do not move the clock past it
                horizon = max(self.now, batch[-1][0])
        self.attach(X.EventStream.from_messages(batch))
        self.advance(horizon)
        if self.shadow is not None and mk is not None:
            self.shadow.update(mk, now)

    def place(self, d: dict) -> None:
        """The order leaves at the later of the decision time and now; the exchange applies it latency later."""
        assert not self.is_busy(d["ticker"])
        sent = max(d["t"], self.wall)
        limit = X.snap(d["limit"], d["side"])
        p = dict(id=self.id0 + len(self.orders), ticker=d["ticker"], kind=d["kind"], side=d["side"], limit=limit, size=d["size"],
                 signal=d["signal"], decision_t=d["t"], t0=sent, pending=d["size"], lots=[], entry_done=False)
        self.orders.append(p)
        self.busy[p["ticker"]] = p
        self.emit("submitted", id=p["id"], ticker=p["ticker"], size=p["size"], limit=limit, wanted_limit=d["limit"],
                  side=p["side"], decision_t=d["t"], sent=sent)
        self.schedule(sent + self.latency, "entry", p)
        if self.shadow is not None:                      # the older print-fill rule gets the same order (it may still be busy with its own)
            if self.shadow.is_busy(d["ticker"]):
                self.stats["shadow_skipped_busy"] += 1
            else:
                self.shadow.place(d)

    def cancel_entries(self, t: float) -> None:
        if self.shadow is not None:
            self.shadow.cancel_entries(t)
        for p in self.orders:
            if not p["entry_done"]:
                self.schedule(max(self.wall, t) + self.latency, "cancel_entry", p)

    @staticmethod
    def resume_point(events: list[dict]) -> int:
        """Next order id after an earlier run of this game (ids continue; execution.jsonl is appended to)."""
        return 1 + max((e["id"] for e in events if e.get("event") == "submitted"), default=-1)

    def resume(self, events: list[dict]) -> None:
        """A restarted runner cannot rebuild the queue it was in. Entries still resting and lots still held in the earlier
        run are abandoned (logged); their contracts are free again."""
        done = {e["id"] for e in events if e.get("event") in ("entry_cancel_ack", "entry_rejected")}
        resting = sorted({e["id"] for e in events if e.get("event") == "submitted"} - done)
        if resting or any(e.get("event") == "fill" for e in events):
            self.emit("abandoned_on_restart", resting_entries=resting, earlier_fills=sum(e.get("event") == "fill" for e in events))

    def summary(self) -> dict:
        """Same keys as Engine.summary for the status line and summary.json; P&L of lots closed before the end only."""
        closed = [lot for lot in self.lots if lot["remaining"] <= X.EPS and lot["exits"]]
        pnl = [sum(x["size"] * lot["side"] * (x["price"] - lot["entry"]) - x["fee_c"] for x in lot["exits"]) - lot["entry_fee_c"] for lot in closed]
        held = [lot for lot in self.lots if lot["remaining"] > X.EPS]
        n = self.stats["arrival_delay_n"]
        return dict(orders=len(self.orders), filled=len({lot["parent"] for lot in self.lots}), closed=len(closed),
                    to_settlement=len(held), pnl_closed_c=float(sum(pnl)),
                    net_per_contract_c=float(np.mean([p / lot["size"] for p, lot in zip(pnl, closed)])) if closed else None,
                    book_arrival_delay_ms=self.stats["arrival_delay_sum_ms"] / n if n else None,
                    late_messages=self.stats["late_messages"], exchange_diagnostics=dict(self.stats))

    def close(self) -> None:
        if self.tail is not None:
            self.tail.stop_event.set()
