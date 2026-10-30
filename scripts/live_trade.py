"""Paper trading, one league's games on one day: record (live_record.py), forecast every 5 s
(live_model.py), decide with the frozen rule, keep our own paper book, and publish the decisions to Lumid.

Per game, every 5-s step of the game (first play + 5k s):

    forecast    the 5 seeds' average change of each winner / spread / total contract; plays count as
                known once our feed delivered them and --delay s (default 0) have passed since their end. The
                frozen setting (10 s) is logged alongside as delta_d10, for comparison only
    decide      holdout_spec section 3 on the candidate kinds (--kinds, default spread,total): a print
                in the last 60 s is the reference; signal = forecast level at +120 s - reference;
                |signal| >= 5c; one order or position per contract at a time; MLB: no decision while
                15-60 s have passed since the last pitch the model knows, nor while the feed shows a ball
                in play whose result it doesn't have yet (live only: the live feed lists the pitch "In play,
                run(s)" seconds after contact, with the old score, and its result ~20 s later; finished-game
                feeds, which the models learn from, never show that state); weight ((|signal| - 5) / 10)^2
                x width / 10 (each capped at 1; width 1 when a side is missing), placed as round(K x weight)
                whole contracts (--contracts K, default 5; no order if that is 0: b34_integer_sizing, journal
                2026-10-24)
    paper book  buy limit at reference - 1c / sell at + 1c, resting 60 s, filled by a later print strictly
                through it; exit 120 s after the fill: rest 1c past the last print for 120 s, else take
                the next opposite-side print within 120 s, else hold to settlement. Fees: maker 1.75 p(1-p)
                per limit fill, taker 7 p(1-p). Stages are closed only GRACE_S after their deadline, so
                prints still on their way (polled every 5 s) are counted
    publish     to Lumid (lqt_inbox, signal.publish), every step, per candidate contract:
                    ingame_edge     the signal of an order placed at this step, else 0 (cents x 100)
                    ingame_target   our paper position incl. a resting entry: side x contracts (x 10,000)
                    ingame_limit    the resting order's limit price (cents x 100), 0 if none
                zeros for every contract once the game is final. --publish sends; without it, the
                payloads are only logged (dry run). The PAT is read from LUMID_PAT.

Logs, in <out>/<date>/<league>_<game>/ (next to the recorder's raw files):

    plays_seen.jsonl    each play: when it happened (MLB pitch end / NFL snap), when we received it, the delay
    forecasts.jsonl     every step: per contract anchor, forecast change at each horizon (and at 10 s delay)
    decisions.jsonl     every step: per candidate contract, reference, signal, width, size, what was done
    orders.jsonl        paper orders: placed, filled, expired, exit, closed (with net P&L)
    published.jsonl     what was (or in a dry run would have been) sent to Lumid
    summary.json        at the end: feed delay quantiles, decisions, orders, fills, P&L

    venv/bin/python -u -W ignore scripts/live_trade.py --league MLB                  # today, dry run
    venv/bin/python -u -W ignore scripts/live_trade.py --league MLB --games 849812 --publish
    venv/bin/python -u -W ignore scripts/live_trade.py --league NFL --date 2026-10-12 --out $FLOWMESH_OUTPUT
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

import live_model as LM
import live_record as R

KINDS_DEFAULT = ("spread", "total")
HORIZON = "120s"
MIN_SIGNAL, REF_MAX_S, WIDTH_S = 5.0, 60.0, 120.0
CONTRACTS_K = None             # contracts per order at full weight: None = the backtest weight (0-1) to the nearest 0.01; K = whole contracts, round(K x weight)
ENTRY_S, HOLD_S, EXIT_S, OFFSET = 60.0, 120.0, 120.0, 1.0
SKIP_AGE = {"MLB": (15.0, 60.0), "NFL": None, "NBA": None}
# NBA calls not final yet. The model learnt from finished games' feeds, which show only the final version of a play:
#   placeholder rows ESPN fills in later ("No Shot (Default Shot)", "No Turnover": 945 / 129 rows in 25 recorded
#   games, replaced within ~20 s at p90): no decisions while one is old enough for the model to count (time + delay),
#   for at most REVIEW_MAX_S (finished games' feeds keep a few)
#   reviews ("Challenge", "Coach's Challenge (replaycenter)": the call may be overturned, ~3 a game, ~90 s): no
#   decisions while one is pending (at most REVIEW_MAX_S), nor until its outcome is delay + REVIEW_SETTLE_S old, so
#   the corrected possessions have reached the model
NBA_PLACEHOLDER = re.compile(r"^No |Default")
NBA_REVIEW = re.compile(r"Challenge|Review|replaycenter")
NBA_REVIEW_PENDING = re.compile(r"^Challenge$|replaycenter")
REVIEW_SETTLE_S, REVIEW_MAX_S = 30.0, 600.0
STALE_S, STALE_N = 120.0, 5    # NFL / NBA: no decisions while any of the last STALE_N new plays reached us > STALE_S late
# NBA, 4th quarter / overtime: a position needs ~5 min of real time to close (60 s to fill, 120 s hold, 120 s exit), and the
# last game minutes run long or short with the lead (fouls and timeouts when it is close; a running clock in a blowout). Over
# 43 recorded games, the share of Q4 plays with < 6.7 real min left (5 min + the ~100 s the model's clock is stale) was 24-83%
# at 1-2 min of game clock for leads <= 10, 85-100% for leads > 10, and under 12% at 3-5 min for leads <= 20. Cut-offs, in game
# seconds left, by the lead (points) the model's latest known play shows: (max lead, seconds)
LATE_RULE = ((5, 120.0), (10, 180.0), (20, 240.0), (math.inf, 300.0))
LATE_KINDS = ("total",)       # the contract kinds the window blocks: spreads made +1.6c inside it and -5.5c outside (61 / 25 fills, 40 games), so they are only tagged
NBA_REG_S = 48 * 60.0          # live_model's NBA tau = game clock left / 48 min (overtime: the period's clock / 48 min)
MAKER, TAKER = 1.75, 7.0
GRACE_S = 15.0                 # a stage closes this long after its deadline (prints arrive up to ~10 s late)
SHADOW_DELAY = 10.0            # the frozen feed delay, logged alongside
INBOX = "https://lum.id/lqt/submit/lqt_inbox"
TICKS = 100                    # Lumid fixed point: 1.0 = 10,000, so 1c = 100
MARK_BPS = 9999                # the confidence our rows carry: the .lqts strategy trades only rows with this mark


def fee(rate: float, price: float) -> float:
    p = price / 100
    return rate * p * (1 - p)


def size_of(signal: float, width: float | None) -> float:
    s = min(1.0, ((abs(signal) - MIN_SIGNAL) / 10) ** 2) if abs(signal) >= MIN_SIGNAL else 0.0
    w = 1.0 if width is None else min(1.0, max(0.0, width / 10))
    return s * w


def contracts_of(weight: float, k: int | None) -> float:
    """Contracts for a 0-1 weight, half up (0 = no order): k None: the weight to the nearest 0.01; else round(k x weight) whole contracts."""
    return float(np.floor(weight * 100 + 0.5)) / 100 if k is None else float(np.floor(k * weight + 0.5))


class Log:
    def __init__(self, root: Path) -> None:
        self.root, self.lock = root, threading.Lock()

    def __call__(self, name: str, rows: list[dict] | dict) -> None:
        rows = [rows] if isinstance(rows, dict) else rows
        if rows:
            with self.lock, (self.root / name).open("a") as f:
                for r in rows:
                    f.write(json.dumps(r, separators=(",", ":"), default=float) + "\n")


class Publisher:
    def __init__(self, live: bool, log: Log) -> None:
        self.live, self.log = live, log
        self.pat = os.environ.get("LUMID_PAT", "") if live else ""
        if live and not self.pat:
            raise SystemExit("--publish needs LUMID_PAT in the environment")
        self.client = None
        if live:
            import httpx
            self.client = httpx.Client(timeout=10, headers={"Authorization": f"Bearer {self.pat}"})

    def send(self, signals: list[dict], t_ns: int) -> None:
        if not signals:
            return
        recs = [dict(s, ts_event_ns=t_ns) for s in signals]
        status = "dry-run"
        if self.live:
            body = {"source_endpoint": "fyp-ingame", "records": [{
                "msg_id": str(uuid.uuid4()), "topic": "signal.publish", "sender": "fyp-ingame", "status": "pending",
                "payload": {"topic": "signal.publish", "auth": {"pat": self.pat}, "signals": recs}}]}
            try:
                r = self.client.post(INBOX, json=body)
                status = str(r.status_code)
            except Exception as e:           # a failed publish must not stop the game
                status = f"error {e!r}"[:200]
        self.log("published.jsonl", {"t_ns": t_ns, "status": status, "signals": recs})


def in_play_pending(doc: dict) -> str | None:
    """MLB: the description of the last pitch if it was put in play and its plate appearance has no result yet."""
    plays = doc["feed"]["liveData"]["plays"]["allPlays"]
    if not plays or plays[-1]["about"].get("isComplete", True):
        return None
    ev = [e for e in plays[-1]["playEvents"] if e.get("isPitch")]
    if not ev or not ev[-1]["details"].get("isInPlay"):
        return None
    return ev[-1]["details"].get("description") or "in play"


class PaperBook:
    """The frozen maker rule's orders, driven by the prints as they arrive (Market.p, sweep order).

    Fills: a resting limit is filled by a later print strictly through it (the frozen rule). With depth (the
    order book, e.g. live_rerun.Depth: queue(ticker, side, price_c, t) -> contracts resting at that level,
    top(ticker, t) -> (bid, ask)), also by prints exactly at the limit once more than the size resting ahead of
    us when we posted has traded there (cancellations ahead of us are ignored, so this is conservative); and an
    order that crosses the book when posted takes the best opposite price at once, at the taker fee."""

    def __init__(self, log: Log, depth=None) -> None:
        self.log, self.orders, self.busy, self.depth = log, [], {}, depth

    def _post(self, ticker: str, rest: int, price: float, t: float) -> tuple[float, float | None]:
        """(the size resting ahead of a new limit, the price it takes at once if it crosses the book, else None).
        rest: +1 a resting buy (on the yes bids), -1 a resting sell (a no bid at 100 - price)."""
        if self.depth is None:
            return 0.0, None
        bid, ask = self.depth.top(ticker, t)
        if rest == 1 and ask is not None and ask <= price:
            return 0.0, float(ask)
        if rest == -1 and bid is not None and bid >= price:
            return 0.0, float(bid)
        return (self.depth.queue(ticker, "yes", round(price), t) if rest == 1
                else self.depth.queue(ticker, "no", round(100 - price), t)), None

    def _fill(self, p: dict, a: float, b: float, rest: int, price: float, ahead: float) -> int:
        """Index of the print in (a, b] that fills a resting limit (rest +1 buy / -1 sell), else -1."""
        if self.depth is None:
            return self._first(p, a, b, taker_yes=(rest == -1), price=price, above=(rest == -1))
        i, j = np.searchsorted(p["t"], a, side="right"), np.searchsorted(p["t"], b, side="right")
        side = p["yes"][i:j] if rest == -1 else ~p["yes"][i:j]
        px = p["px"][i:j]
        through = side & ((px > price) if rest == -1 else (px < price))
        at = side & (np.abs(px - price) < 1e-6)
        hit = through | (at & (np.cumsum(np.where(at, p["n"][i:j], 0.0)) > ahead))
        return i + int(np.argmax(hit)) if hit.any() else -1

    def resume(self, events: list[dict]) -> None:
        """Rebuild the orders from orders.jsonl after a restart, so the runner carries on from where it was: an order
        in the book keeps its stage, its contract stays busy, and nothing is placed twice. What the log lacks (the
        queue position ahead of a posted exit) is taken as 0, as when there is no order book."""
        by_id: dict[int, dict] = {}
        for e in events:
            o = by_id.get(e.get("id"))
            kind = e["event"]
            if kind == "placed":
                o = {k: v for k, v in e.items() if k != "event"}
                o.setdefault("ahead", 0.0)
                by_id[o["id"]] = o
            elif o is None:
                continue
            elif kind == "filled":
                crossed = e.get("how") == "crossed"
                o.update(stage="hold", t_fill=e["t"], entry_fee=fee(TAKER if crossed else MAKER, e["price"] if crossed else o["limit"]))
                if crossed:
                    o["entry"] = e["price"]
            elif kind in ("expired", "cancelled"):
                o["stage"] = "expired"
            elif kind == "exit posted":
                o.update(stage="exit", te=e["t"], exit_limit=e["price"], exit_ahead=0.0, exit_took=None)
            elif kind == "closed":
                o.update(stage="closed", exit=e["price"], how=e["how"], t_exit=e["t"], net=e["net"])
            elif kind == "to settlement":
                o["stage"] = "settle"
        self.orders = [by_id[i] for i in sorted(by_id)]
        self.busy = {}
        for o in self.orders:
            self.busy[o["ticker"]] = None if o["stage"] in ("closed", "expired") else o

    def is_busy(self, ticker: str) -> bool:
        return self.busy.get(ticker) is not None

    def place(self, d: dict) -> None:
        o = dict(id=len(self.orders), ticker=d["ticker"], kind=d["kind"], t0=d["t"], side=d["side"], limit=d["limit"],
                 size=d["size"], signal=d["signal"], stage="entry")
        o["ahead"], took = self._post(o["ticker"], o["side"], o["limit"], o["t0"])
        self.orders.append(o)
        self.busy[o["ticker"]] = o
        self.log("orders.jsonl", dict(event="placed", **o))
        if took is not None:                        # crosses the book: a taker fill at the opposite best price
            o.update(stage="hold", t_fill=o["t0"], entry=took, entry_fee=fee(TAKER, took))
            self.log("orders.jsonl", dict(event="filled", id=o["id"], t=o["t_fill"], price=took, how="crossed"))

    @staticmethod
    def _first(p: dict, a: float, b: float, taker_yes: bool, price: float | None, above: bool) -> int:
        """Index of the first print in (a, b] by that taker side (beyond price, strictly, if given), else -1."""
        i, j = np.searchsorted(p["t"], a, side="right"), np.searchsorted(p["t"], b, side="right")
        side = p["yes"][i:j] if taker_yes else ~p["yes"][i:j]
        hit = side if price is None else side & ((p["px"][i:j] > price) if above else (p["px"][i:j] < price))
        return i + int(np.argmax(hit)) if hit.any() else -1

    def update(self, mk: LM.Market, now: float) -> None:
        mk.build()
        for o in self.orders:
            if o["stage"] in ("closed", "expired", "settle"):
                continue
            p, s = mk.p[o["ticker"]], o["side"]
            if o["stage"] == "entry":
                end = min(o["t0"] + ENTRY_S, o.get("cancel_at", math.inf))
                f = self._fill(p, o["t0"], end, s, o["limit"], o["ahead"])
                if f >= 0:
                    o.update(stage="hold", t_fill=float(p["t"][f]), entry_fee=fee(MAKER, o["limit"]))
                    self.log("orders.jsonl", dict(event="filled", id=o["id"], t=o["t_fill"], price=o["limit"]))
                elif now > end + GRACE_S:
                    o["stage"] = "expired"
                    self.busy[o["ticker"]] = None
                    self.log("orders.jsonl", dict(event="expired" if end == o["t0"] + ENTRY_S else "cancelled", id=o["id"], t=end))
            if o["stage"] == "hold" and now > o["t_fill"] + HOLD_S + GRACE_S:
                te = o["t_fill"] + HOLD_S
                n = int(np.searchsorted(p["t"], te, side="right"))
                ref_e = float(p["px"][n - 1]) if n else o["limit"]
                o.update(stage="exit", te=te, exit_limit=min(99.0, max(1.0, ref_e + OFFSET * s)))   # a 0c or 100c limit cannot be posted
                o["exit_ahead"], o["exit_took"] = self._post(o["ticker"], -s, o["exit_limit"], te) if 0 < o["exit_limit"] < 100 else (0.0, None)
                self.log("orders.jsonl", dict(event="exit posted", id=o["id"], t=te, price=o["exit_limit"]))
            if o["stage"] == "exit":
                lx, te = o["exit_limit"], o["te"]
                f = self._fill(p, te, te + EXIT_S, -s, lx, o["exit_ahead"]) if 0 < lx < 100 else -1
                if o.get("exit_took") is not None:
                    self._close(o, o["exit_took"], "crossed", te, fee(TAKER, o["exit_took"]))
                elif f >= 0:
                    self._close(o, lx, "limit filled", float(p["t"][f]), fee(MAKER, lx))
                elif now > te + EXIT_S + GRACE_S:
                    x = self._first(p, te + EXIT_S, te + 2 * EXIT_S, taker_yes=(s == -1), price=None, above=False)
                    if x >= 0:
                        self._close(o, float(p["px"][x]), "taken", float(p["t"][x]), fee(TAKER, float(p["px"][x])))
                    elif now > te + 2 * EXIT_S + GRACE_S:
                        o["stage"] = "settle"           # held to the result: the contract stays busy to the end
                        self.log("orders.jsonl", dict(event="to settlement", id=o["id"], t=te + 2 * EXIT_S))

    def cancel_entries(self, t: float) -> None:
        """Take every resting entry off the book at t (one not yet filled by a print up to t)."""
        for o in self.busy.values():
            if o is not None and o["stage"] == "entry" and "cancel_at" not in o and t < o["t0"] + ENTRY_S:
                o["cancel_at"] = t

    def _close(self, o: dict, price: float, how: str, t: float, exit_fee: float) -> None:
        net = o["side"] * (price - o.get("entry", o["limit"])) - o["entry_fee"] - exit_fee
        o.update(stage="closed", exit=price, how=how, t_exit=t, net=net)
        self.busy[o["ticker"]] = None
        self.log("orders.jsonl", dict(event="closed", id=o["id"], t=t, price=price, how=how, net=net, pnl=net * o["size"]))

    def position(self, ticker: str) -> tuple[float, float]:
        """(side x size including a resting entry, the resting order's limit or 0)."""
        o = self.busy.get(ticker)
        if o is None:
            return 0.0, 0.0
        return o["side"] * o["size"], (o["limit"] if o["stage"] == "entry" else 0.0)


class Engine:
    """One game: takes data as it arrives (the recorder's hooks, or a replay), and at each 5-s step
    forecasts, decides, updates the paper book and publishes. step(now) is driven by the caller's clock."""

    def __init__(self, g: dict, fc: LM.Forecaster, root: Path, kinds: tuple[str, ...], delay: float, pub_live: bool,
                 pub_kinds: tuple[str, ...] | None = None, resume: bool = False) -> None:
        self.g, self.fc, self.L_ = g, fc, g["league"]
        self.kinds, self.delay = kinds, delay
        self.pub_kinds = pub_kinds or kinds
        self.start = dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00"))
        self.contracts = LM.contract_meta(self.L_, g["contracts"], g["home_k"], g["away_k"])
        self.mk = LM.Market([c["ticker"] for c in self.contracts])
        root.mkdir(parents=True, exist_ok=True)
        self.log = Log(root)
        self.book, self.pub = PaperBook(self.log), Publisher(pub_live, self.log)
        self.lock = threading.Lock()
        self.doc, self.doc_new = None, False
        self.first_seen: dict[str, float] = {}
        self.plays = None
        self.px = self.pt = self.pstate = None
        self.px_t = self.pt_t = self.pstate_t = None   # NBA: the same plays stamped at their real end times (see shadow_true)
        self.first_play, self.L, self.p_home = None, math.nan, None
        self.first_play_logged = False
        self.in_play: str | None = None                # MLB: the pitch description of a ball in play awaiting its result
        self.skip_in_play = True                       # no decisions while it awaits its result (reruns: switchable)
        self.live_clock = True                         # MLB: pitch times as the archive gives them (LM.mlb_plays)
        self.last_k, self.final = -1, False
        self.lags: list[float] = []
        self.review_type: dict[str, str] = {}          # NBA: ESPN play id -> its review type as last seen
        self.review_pending: dict[str, float] = {}     # NBA: reviews without an outcome yet -> when first seen
        self.review_hold = 0.0                         # NBA: no decisions before this time (a review's outcome settling)
        self.placeholders: list[float] = []            # NBA: times (ESPN) of rows still showing a placeholder
        self.n_dec, self.placed = 0, {}
        # resting entries off the book while a pitch is likely: "window" = once the last pitch we know of is
        # SKIP_AGE old (as for decisions); a callable(t0) -> cancel time = given (e.g. the next pitch's start, hindsight)
        self.cancel_rule: str | None = None
        self.no_short_over = False                     # True: never sell a total ("over") contract
        self.late_block = False                        # NBA: True = no new entries in the late-game window (LATE_RULE); False = only logged
        self.k = CONTRACTS_K                           # contracts per order at full weight (None: fractional)
        self.shadow = True                             # also forecast at the frozen 10 s delay, for the log (reruns: off)
        # NBA: a play counts at max(end + delay, arrival); a late play is stamped with its arrival, so a clump of late plays
        # is bunched in the model's history (training never sees that). shadow_true also forecasts with every arrived play
        # stamped at its real end time, which is the state the backtest would have; gate_true skips a decision unless that
        # forecast gives the same side and at least MIN_SIGNAL
        self.shadow_true = self.L_ == "NBA"
        self.gate_true = self.shadow_true               # on for NBA; reruns set it from --gate-true
        self.rows_true: dict[str, dict] = {}
        self.resumed: set[str] = set()                 # plays seen before a restart
        if resume:
            self._resume(root)

    def _resume(self, root: Path) -> None:
        """Carry on from the files an earlier run of this game left in root: each play keeps the time it first reached
        us (so the plays a restart re-reads do not look hours late), and the paper orders are rebuilt. The files are
        appended to, never rewritten."""
        if (root / "plays_seen.jsonl").exists():
            for r in R.rows(root / "plays_seen.jsonl"):
                if r["key"] not in self.first_seen:
                    self.first_seen[r["key"]] = r["received"]
                    self.lags.append(r["delay_s"])
            self.resumed = set(self.first_seen)
        if (root / "orders.jsonl").exists():
            self.book.resume(list(R.rows(root / "orders.jsonl")))
        if (root / "decisions.jsonl").exists():
            self.first_play_logged = any(r.get("event") == "first play" for r in R.rows(root / "decisions.jsonl"))
        if self.first_seen or self.book.orders:
            print(f"{self.g['game_id']}: resumed {len(self.first_seen)} plays seen, {len(self.book.orders)} orders", flush=True)

    def use_exchange_sim(self, root: Path, latency: float) -> None:
        """Replace the print-fill paper book by the exchange simulator (live_exec.LiveExecution) fed from the recorder's
        books_ws.jsonl. The older book stays as `shadow` and gets the same orders (orders.jsonl as before; the simulator
        writes execution.jsonl)."""
        import live_exec as LE
        prior = list(R.rows(root / "execution.jsonl")) if (root / "execution.jsonl").exists() else []
        ex = LE.LiveExecution(self.log, root / "books_ws.jsonl", [c["ticker"] for c in self.contracts], latency,
                              shadow=self.book, id0=LE.LiveExecution.resume_point(prior))
        ex.ref = self._last_print
        ex.resume(prior)
        self.book = ex

    def _last_print(self, ticker: str, t: float) -> float | None:
        self.mk.build()
        p = self.mk.p[ticker]
        i = int(np.searchsorted(p["t"], t, side="right"))
        return float(p["px"][i - 1]) if i else None

    # ------------------------------------------------------------------ data in
    def on_feed(self, doc: dict, got: dt.datetime) -> None:
        with self.lock:
            self.doc, self.doc_new = doc, True
            self._seen(doc, got.timestamp())

    def _seen(self, doc: dict, got: float) -> None:
        """First time each play reached us; the delay against when it happened (MLB: the pitch's end, as
        the model times it; NFL: the snap, ESPN's only time; NBA: ESPN's time of the row)."""
        new = []
        if self.L_ == "MLB":
            for p in doc["feed"]["liveData"]["plays"]["allPlays"]:
                for i, e in enumerate(p["playEvents"]):
                    if not e.get("isPitch"):
                        continue
                    key = f"{p['about']['atBatIndex']}.{i}"
                    if key not in self.first_seen and e.get("endTime"):
                        self.first_seen[key] = got
                        happened = LM.epoch(dt.datetime.fromisoformat(e["endTime"].replace("Z", "+00:00")))
                        new.append(dict(key=key, happened=happened, received=got, delay_s=got - happened))
            final = doc["feed"]["gameData"]["status"]["detailedState"] in R.FINAL
        else:
            rows = LM.espn_rows(doc, self.L_)
            if self.L_ == "NBA":
                self._calls(rows, got)
            for r in rows:
                if r["espn_play_id"] not in self.first_seen and r["t"] is not None:
                    self.first_seen[r["espn_play_id"]] = got
                    new.append(dict(key=r["espn_play_id"], happened=LM.epoch(r["t"]), received=got,
                                    delay_s=got - LM.epoch(r["t"]), event_type=r["event_type"]))
            st = (((doc.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type") or {}
            final = st.get("state") == "post"
        self.final = self.final or final
        self.lags += [n["delay_s"] for n in new]
        self.log("plays_seen.jsonl", new)

    def _calls(self, rows: list[dict], got: float) -> None:
        """NBA: track placeholder rows and reviews (see NBA_PLACEHOLDER)."""
        self.placeholders = [LM.epoch(r["t"]) for r in rows if r["t"] is not None and NBA_PLACEHOLDER.search(r["event_type"] or "")]
        for r in rows:
            e, k = r["event_type"] or "", r["espn_play_id"]
            if not NBA_REVIEW.search(e) or self.review_type.get(k) == e:
                continue
            self.review_type[k] = e
            if NBA_REVIEW_PENDING.search(e):
                self.review_pending.setdefault(k, got)
            else:                                      # an outcome: overturned / stands / supported
                self.review_pending.pop(k, None)
                if k not in self.resumed:              # one already settled before a restart holds nothing now
                    self.review_hold = max(self.review_hold, got + self.delay + REVIEW_SETTLE_S)
        if self.review_pending or self.placeholders:
            self.log("calls.jsonl", dict(t=got, pending=sorted(self.review_pending), hold_until=self.review_hold,
                                         placeholders=len(self.placeholders)))

    def _unsettled(self, t: float) -> str | None:
        """NBA: why the plays the model would see at t may not be final yet, else None."""
        if self.L_ != "NBA":
            return None
        if any(t - t0 < REVIEW_MAX_S for t0 in self.review_pending.values()):
            return "review pending"
        if t < self.review_hold:
            return "review outcome settling"
        if any(p + self.delay <= t < p + REVIEW_MAX_S for p in self.placeholders):      # a few are never filled in
            return "placeholder play"
        return None

    def on_trades(self, rows: list[dict]) -> None:
        out = []
        for r in rows:
            px = r.get("yes_price_dollars")
            out.append(dict(ticker=r["ticker"], trade_id=r["trade_id"], taker_side=r["taker_side"],
                            t=LM.epoch(dt.datetime.fromisoformat(r["created_time"].replace("Z", "+00:00"))),
                            price=float(px) * 100 if px is not None else float(r["yes_price"]),
                            count=float(r.get("count_fp", r.get("count")))))
        with self.lock:
            self.mk.add_trades(out)

    def on_candles(self, rows: list[dict]) -> None:
        """Only finished candles (the batch call also returns the minute in progress)."""
        def close(c: dict, side: str) -> float | None:
            v = (c.get(side) or {}).get("close_dollars", (c.get(side) or {}).get("close"))
            return float(v) * 100 if v is not None else None
        out = []
        for c in rows:
            got = LM.epoch(dt.datetime.fromisoformat(c["received_at"]))
            if c["end_period_ts"] <= got:
                out.append(dict(ticker=c["ticker"], ts=float(c["end_period_ts"]), bid=close(c, "yes_bid"), ask=close(c, "yes_ask")))
        with self.lock:
            self.mk.add_candles(out)

    # ------------------------------------------------------------------ each step
    def _rebuild_plays(self) -> None:
        if self.L_ == "MLB":
            self.plays = LM.mlb_plays(self.doc, self.start, self.live_clock)
            self.in_play = in_play_pending(self.doc)
        elif self.L_ == "NBA":
            self.plays, first_row = LM.nba_live_plays(self.doc, self.start, self.first_seen, self.delay)
            if self.shadow_true:
                pt_ = LM.nba_live_plays(self.doc, self.start, None, self.delay)[0]
                if pt_.height:
                    self.px_t, self.pt_t, self.pstate_t = LM.play_arrays(pt_, self.L_)
        else:
            self.plays, _ = LM.nfl_live_plays(self.doc, self.start, self.first_seen, self.delay)
        if self.plays.height:
            self.px, self.pt, self.pstate = LM.play_arrays(self.plays, self.L_)
            if self.first_play is None:
                self.first_play = first_row if self.L_ == "NBA" else LM.epoch(self.plays["t"].min())
                self.L = LM.pregame_total(self.contracts, self.mk, self.first_play - 60)
                if not self.first_play_logged:
                    self.log("decisions.jsonl", dict(event="first play", t=self.first_play, L=self.L, late_block=self.late_block,
                                                     late_kinds=list(LATE_KINDS) if self.late_block else []))
        self.doc_new = False

    def step(self, now: float) -> None:
        with self.lock:
            if self.doc_new:
                self._rebuild_plays()
            if self.first_play is None:
                return
            if self.cancel_rule == "window" and self._skip(now) is not None:
                self.book.cancel_entries(now)
            k = math.floor((now - self.first_play) / LM.STEP)
            if k <= self.last_k:
                self.book.update(self.mk, now)
                return
            self.last_k = k
            t = self.first_play + LM.STEP * k
            st = LM.step_inputs(t, self.first_play, self.contracts, self.mk, self.p_home)
            self.p_home = st["p_home"]
            rows = self.fc.forecast([st], self.px, self.pt, self.pstate, self.L, self.delay)
            shadow = ({r["ticker"]: r["delta"] for r in self.fc.forecast([st], self.px, self.pt, self.pstate, self.L, SHADOW_DELAY)}
                      if self.shadow else None)
            self.rows_true = {}
            if self.shadow_true and self.px_t is not None:
                self.rows_true = {r["ticker"]: r for r in self.fc.forecast([st], self.px_t, self.pt_t, self.pstate_t, self.L, self.delay)}
            h = self.fc.horizons
            self.log("forecasts.jsonl", [dict(t=t, ticker=r["ticker"], anchor=r["anchor"], plays_known=r["plays_known"],
                                              delta=dict(zip(h, np.round(r["delta"], 3).tolist())),
                                              delta_d10=dict(zip(h, np.round(shadow[r["ticker"]], 3).tolist())) if shadow else None,
                                              delta_true=dict(zip(h, np.round(self.rows_true[r["ticker"]]["delta"], 3).tolist())) if r["ticker"] in self.rows_true else None)
                                         for r in rows])
            self.book.update(self.mk, now)
            self._decide(t, now, rows)
            self._publish(now)

    def _skip(self, t: float) -> float | None:
        """MLB: seconds since the last pitch the model knows, if in the skip window, else None."""
        win = SKIP_AGE[self.L_]
        if win is None or self.pt is None:
            return None
        n = int(np.searchsorted(self.pt, t - self.delay, side="right"))
        age = t - self.pt[n - 1] if n else math.inf
        return age if win[0] <= age < win[1] else None

    def _stale(self) -> float | None:
        """ESPN feeds: the longest delay among the last STALE_N new plays, if above STALE_S (a stuck feed:
        5 of 26 recorded NBA preseason games ran 10-50 min behind; a stall's plays arrive in a clump whose median can
        be back under STALE_S while the clump is still among the last STALE_N: 24 Oct, BKN @ CLE), else None. MLB: never."""
        if self.L_ == "MLB" or len(self.lags) < STALE_N:
            return None
        m = float(np.max(self.lags[-STALE_N:]))
        return m if m > STALE_S else None

    def _late(self, t: float) -> str | None:
        """NBA: why a new entry at t falls in the late-game window (LATE_RULE), else None. Read from the latest play the model
        knows (its end + delay has passed), so the clock and the lead are as stale as the model's own."""
        if self.L_ != "NBA" or self.pt is None:
            return None
        n = int(np.searchsorted(self.pt, t - self.delay, side="right"))
        if not n:
            return None
        tau, margin = float(self.pstate[n - 1][0]), abs(float(self.pstate[n - 1][1]))
        left = tau * NBA_REG_S
        cut = next(s for lead, s in LATE_RULE if margin <= lead)
        return f"{int(left // 60)}:{int(left % 60):02d} left, lead {margin:.0f}, cut-off {int(cut // 60)}:{int(cut % 60):02d}" if left < cut else None

    def _decide(self, t: float, now: float, rows: list[dict]) -> None:
        skip, stale, unsettled = self._skip(t), self._stale(), self._unsettled(t)
        late = self._late(t)
        out = []
        self.placed = {}
        for r in rows:
            if r["kind"] not in self.kinds:
                continue
            p = self.mk.p[r["ticker"]]
            n = int(np.searchsorted(p["t"], t, side="right"))
            d = dict(t=t, ticker=r["ticker"], kind=r["kind"], level=r["level"][HORIZON])
            if not n or t - p["t"][n - 1] > REF_MAX_S:
                out.append(d | {"action": "no print in 60 s"})
                continue
            ref = float(p["px"][n - 1])
            sig = d["level"] - ref
            yes_i = [i for i in range(n - 1, -1, -1) if p["yes"][i] and t - p["t"][i] <= WIDTH_S][:1]
            no_i = [i for i in range(n - 1, -1, -1) if not p["yes"][i] and t - p["t"][i] <= WIDTH_S][:1]
            width = float(p["px"][yes_i[0]] - p["px"][no_i[0]]) if yes_i and no_i else None
            d |= dict(ref=ref, signal=round(sig, 3), width=width)
            if late is not None:
                d["late"] = late                           # logged whether or not it blocks, so a night can be split by the rule
            rt = self.rows_true.get(r["ticker"])
            sig_t = rt["level"][HORIZON] - ref if rt is not None else None
            if sig_t is not None:
                d["signal_true"] = round(sig_t, 3)
            side = 1 if sig > 0 else -1
            lim = ref - OFFSET if side == 1 else ref + OFFSET
            if stale is not None:
                d["action"] = f"skip: feed {stale:.0f} s behind"
            elif unsettled is not None:
                d["action"] = f"skip: {unsettled}"
            elif self.late_block and late is not None and r["kind"] in LATE_KINDS and abs(sig) >= MIN_SIGNAL:
                d["action"] = f"skip: late game ({late})"
            elif self.skip_in_play and self.in_play is not None:
                d["action"] = f"skip: ball in play, result pending ({self.in_play})"
            elif skip is not None:
                d["action"] = f"skip: {skip:.0f} s since the last known pitch"
            elif abs(sig) < MIN_SIGNAL:
                d["action"] = "signal < 5c"
            elif self.gate_true and sig_t is not None and not (sig_t * sig > 0 and abs(sig_t) >= MIN_SIGNAL):
                d["action"] = "skip: signal differs on real-time stamps"
            elif self.book.is_busy(r["ticker"]):
                d["action"] = "busy"
            elif not 0 < lim < 100:
                d["action"] = "limit outside 1-99c"
            elif self.no_short_over and r["kind"] == "total" and side == -1:
                d["action"] = "no short over"
            elif (n := contracts_of(size_of(sig, width), self.k)) <= 0:
                d |= dict(action="size rounds to 0", weight=round(size_of(sig, width), 4))
            else:
                d |= dict(action="order", side=side, limit=lim, size=round(n, 4), weight=round(size_of(sig, width), 4))
                self.book.place(d)
                if callable(self.cancel_rule):
                    self.book.orders[-1]["cancel_at"] = self.cancel_rule(t)
                self.placed[r["ticker"]] = sig
                self.n_dec += 1
            out.append(d)
        self.log("decisions.jsonl", out)

    def _publish(self, now: float) -> None:
        sig = []
        for c in self.contracts:
            if c["kind"] not in self.pub_kinds:
                continue
            pos, lim = (0.0, 0.0) if self.final else self.book.position(c["ticker"])
            edge = 0.0 if self.final else self.placed.get(c["ticker"], 0.0)
            for name, v in (("ingame_edge", edge * TICKS), ("ingame_target", pos * 10_000), ("ingame_limit", lim * TICKS)):
                sig.append(dict(signal_name=name, instrument_id=c["ticker"], score_ticks=int(round(v)),
                                confidence_bps=MARK_BPS, side_hint=int(np.sign(v))))
        self.pub.send(sig, int(now * 1e9))

    def summary(self) -> dict:
        if hasattr(self.book, "summary"):                 # exchange simulator; the older book's numbers are kept as `print_fill_rule`
            lag = np.array(self.lags) if self.lags else np.array([np.nan])
            return dict(game=self.g["game_id"], league=self.L_, plays_seen=len(self.lags), late_block=self.late_block,
                        feed_delay_s={q: float(np.nanquantile(lag, q)) for q in (0.1, 0.5, 0.9)},
                        **self.book.summary(), print_fill_rule=self._paper_summary(self.book.shadow.orders))
        return self._paper_summary(self.book.orders, full=True)

    def _paper_summary(self, o: list, full: bool = False) -> dict:
        closed = [x for x in o if x["stage"] == "closed"]
        lag = np.array(self.lags) if self.lags else np.array([np.nan])
        head = (dict(game=self.g["game_id"], league=self.L_, plays_seen=len(self.lags), late_block=self.late_block,
                     feed_delay_s={q: float(np.nanquantile(lag, q)) for q in (0.1, 0.5, 0.9)}) if full else {})
        return dict(**head,
                    orders=len(o), filled=sum(x["stage"] in ("hold", "exit", "closed", "settle") for x in o),
                    closed=len(closed), to_settlement=sum(x["stage"] == "settle" for x in o),
                    pnl_closed_c=float(sum(x["net"] * x["size"] for x in closed)),
                    net_per_contract_c=float(np.mean([x["net"] for x in closed])) if closed else None)


def run_live(g: dict, eng: Engine, rec: R.Recorder) -> None:
    stop = threading.Event()
    t = threading.Thread(target=R.run_game, args=(rec, False, stop), daemon=True)
    t.start()
    last_status = 0.0
    while t.is_alive():
        try:
            eng.step(time.time())
        except Exception as e:                        # one bad step must not end the game
            eng.log("decisions.jsonl", dict(event="error", t=time.time(), error=repr(e)[:500]))
            print(f"{g['game_id']} step error: {e!r}", flush=True)
        if time.time() - last_status > 60:
            last_status = time.time()
            s = eng.summary()
            print(f"{g['league']} {g['game_id']} {dt.datetime.now(dt.timezone.utc):%H:%M:%S}: plays {s['plays_seen']}, "
                  f"feed delay median {s['feed_delay_s'][0.5]:.1f} s, orders {s['orders']}, filled {s['filled']}, "
                  f"closed {s['closed']} ({s['pnl_closed_c']:+.1f}c)", flush=True)
        time.sleep(1)
    stop.set()
    eng.final = True
    eng.step(time.time() + LM.STEP)                    # publish the zeros
    if hasattr(eng.book, "close"):
        eng.book.close()
    (eng.log.root / "summary.json").write_text(json.dumps(eng.summary(), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["MLB", "NFL", "NBA"], default="MLB")
    ap.add_argument("--date", type=dt.date.fromisoformat, default=R.now().astimezone(R.ET).date())
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--out", type=Path, default=R.OUT)
    ap.add_argument("--weights", default=str(LM.ROOT / "results" / "ingame" / "deploy"))
    ap.add_argument("--kinds", default=",".join(KINDS_DEFAULT))
    ap.add_argument("--delay", type=float, default=0.0, help="a play counts once it has arrived and its end + delay has passed, whichever is later (frozen: 10)")
    ap.add_argument("--model", help="run name of the weights, e.g. gru_d60_h30-60-120-1800_tp (default: the league's 10-s model)")
    ap.add_argument("--no-books", action="store_true", help="no 15-s order-book polling (live_books_ws.py streams them)")
    ap.add_argument("--publish-kinds", help="kinds whose signals are sent (default: --kinds)")
    ap.add_argument("--publish", action="store_true", help="send to Lumid (default: dry run, payloads only logged)")
    ap.add_argument("--contracts", type=int, default=CONTRACTS_K, help="whole contracts per order at full weight (K): round(K x weight); default: the backtest weight to the nearest 0.01")
    ap.add_argument("--rate", type=float, default=8.0, help="Kalshi requests/s for this runner (all runners together: <= 20)")
    ap.add_argument("--exchange-sim", action="store_true", help="paper fills from the exchange simulator (live_exec.py) fed by the recorder's books_ws.jsonl; the print-fill book keeps running on the same orders (default: print-fill rule only)")
    ap.add_argument("--sim-latency", type=float, default=float(os.environ.get("SIM_LATENCY", .09)), help="decision -> exchange acceptance, s: 0.015 on the VM, 0.09 on the Lumid sandbox")
    ap.add_argument("--late-block", action="store_true", help="NBA: no new entries in the late-game window (LATE_RULE); without it the window is only logged ('late' on decision rows)")
    a = ap.parse_args()
    fc = LM.Forecaster(a.league, a.weights, name=a.model)
    api = R.Api(rate=a.rate)                         # the runners share one Kalshi key (signed: 20 req/s in all)
    games = [g for g in R.schedule(a.league, a.date) if not a.games or g["game_id"] in a.games]
    threads = []
    for g in games:
        g["contracts"] = R.contracts(api, g, a.date)
        root = R.game_root(a.out, a.date, g)
        eng = Engine(g, fc, root, tuple(a.kinds.split(",")), a.delay, a.publish,
                     tuple(a.publish_kinds.split(",")) if a.publish_kinds else None, resume=True)
        eng.k = a.contracts
        eng.late_block = a.late_block
        if a.exchange_sim:
            eng.use_exchange_sim(root, a.sim_latency)
        hooks = R.Hooks(on_feed=eng.on_feed, on_trades=eng.on_trades, on_candles=eng.on_candles)
        rec = R.Recorder(api, g, g["contracts"], root, hooks)
        if a.no_books:
            rec.books_every = None
        print(f"{g['league']} {g['game_id']} {g['away']}@{g['home']} {g['start']}: {len(g['contracts'])} contracts, "
              f"{len(eng.contracts)} modelled, {'PUBLISHING' if a.publish else 'dry run'}", flush=True)

        def go(g=g, eng=eng, rec=rec) -> None:
            R.wait_until_start(g)
            run_live(g, eng, rec)
            print(f"{g['game_id']}: done {json.dumps(eng.summary())}", flush=True)

        threads.append(threading.Thread(target=go))
        threads[-1].start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
