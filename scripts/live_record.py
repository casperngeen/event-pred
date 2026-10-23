"""Live recorder for MLB and NFL games (paper trading): play-by-play and Kalshi quotes / trades as they
arrive, each stamped with the time we received it, so the live forecaster can be fed exactly what was
known when, and the feed's real delay can be measured against the model's assumed 10 s.

Per game, from 20 min before the scheduled start until 10 min after the game is final:

    MLB        statsapi /feed/live/timestamps every POLL_FEED_S; the full /feed/live (+ /winProbability) only
               when a new timecode appears (one per pitch / event, so ~2-3 a minute). Every statsapi request
               carries a random query parameter: its CDN otherwise serves copies up to 10 s old (max-age=10,
               plus stale-while-revalidate), which put the median feed delay at ~15 s after the pitch on
               2026-10-12 against ~7 s uncached
    NFL, NBA   ESPN /summary every POLL_FEED_S, saved when its plays or win probabilities change (NBA: recorded
               only so far, to measure the feed's delay; there is no NBA trading runner)
    Kalshi     the game's winner, spread and total contracts (matched on the ET date + both teams' Kalshi
               codes): trades as they print from Kalshi's WebSocket trade channel when a Kalshi key is set
               (KALSHI_KEY_ID), with the REST trades poll as a backup every POLL_TRADES_BACKUP_S (it fills any
               gap while the stream is down; trades are kept once, by trade_id); without a key, the REST poll
               every POLL_TRADES_S. Polling every ticker every 5 s drew HTTP 429s all night on 2026-10-12 and
               missed a quarter of the trades. Order books every POLL_BOOK_S, 1-min candles for all tickers in
               one batch call every POLL_CANDLE_S

Raw responses are kept (not parsed) so the forecaster can reuse fetch_sports / fetch_pbp's parsers:

    <out>/<date>/<league>_<game>/game.json                 the game and its contracts (Kalshi market rows)
    <out>/<date>/<league>_<game>/feed/<stamp>.json.gz       MLB {"feed", "win_probability"} as fetch_pbp; NFL / NBA the ESPN document
    <out>/<date>/<league>_<game>/feed_arrivals.jsonl        stamp, received_at
    <out>/<date>/<league>_<game>/trades.jsonl               raw Kalshi trades + received_at, via ("ws" / "rest")
    <out>/<date>/<league>_<game>/books.jsonl                raw order books + ticker, received_at
    <out>/<date>/<league>_<game>/candles.jsonl              raw 1-min candles + ticker, received_at

A caller (live_trade.py) can pass hooks to get each new feed document, trade and candle as it arrives.

    venv/bin/python scripts/live_record.py                         # today's MLB games (US Eastern date)
    venv/bin/python scripts/live_record.py --league NFL --date 2026-10-12
    venv/bin/python scripts/live_record.py --games 849812 --once   # one round of every poll now, then stop
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import gzip
import hashlib
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sports import Api  # noqa: E402

MLB_API = "https://statsapi.mlb.com"
ESPN_ROOT = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_LEAGUE = {"NFL": f"{ESPN_ROOT}/football/nfl", "NBA": f"{ESPN_ROOT}/basketball/nba"}
ESPN = ESPN_LEAGUE["NFL"]                      # live_model / older callers
OUT = Path("data/sports_live")
SERIES = {"MLB": ("KXMLBGAME", "KXMLBSPREAD", "KXMLBTOTAL"), "NFL": ("KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL"),
          "NBA": ("KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL")}
# spellings of one team across the sources and over time (fetch_pbp.TO_SRC, stg.sports.contracts.ALIAS; Kalshi
# wrote the Rams LA in 2025 and LAR in 2026): a game's event is matched on any of them
SPELLINGS = {"MLB": [{"AZ", "ARI"}, {"ATH", "A", "OAK"}, {"MIA", "FLA"}, {"KC", "KAN"}, {"WSH", "WAS"}],
             "NFL": [{"JAX", "JAC"}, {"LAR", "LA"}, {"WSH", "WAS"}],
             "NBA": [{"GS", "GSW"}, {"NO", "NOP"}, {"NY", "NYK"}, {"SA", "SAS"}, {"UTAH", "UTA"}, {"WSH", "WAS"}]}
MARKET_FIELDS = ("ticker", "series_ticker", "event_ticker", "title", "subtitle", "yes_sub_title", "no_sub_title",
                 "floor_strike", "cap_strike", "strike_type", "open_time", "close_time")
ET = ZoneInfo("America/New_York")
POLL_FEED_S, POLL_TRADES_S, POLL_BOOK_S, POLL_CANDLE_S = 2, 5, 15, 60
POLL_TRADES_BACKUP_S = 60                      # the REST trades poll while the WebSocket stream carries them
WS_URL, WS_PATH = "wss://api.elections.kalshi.com/trade-api/ws/v2", "/trade-api/ws/v2"
BEFORE, AFTER = dt.timedelta(minutes=20), dt.timedelta(minutes=10)
FINAL = {"Final", "Game Over", "Completed Early"}


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="milliseconds")


def append(p: Path, rows: list[dict], lock: threading.Lock) -> None:
    if not rows:
        return
    with lock, p.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def rows(p: Path):
    """The records of a .jsonl file, or of its gzipped copy (p + ".gz") once it has been compressed for storage."""
    gz = p.with_name(p.name + ".gz")
    with (gzip.open(gz, "rt") if not p.exists() and gz.exists() else p.open()) as f:
        yield from map(json.loads, f)


def bust() -> dict:
    """A query parameter no cache has seen, so the CDN in front of statsapi passes the request to the origin."""
    return {"_": str(random.getrandbits(48))}


def ws_headers(api: Api) -> dict:
    """Kalshi signing for the WebSocket: RSA-PSS(SHA-256) over timestamp_ms + "GET" + the WebSocket path."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    t = str(int(time.time() * 1000))
    sig = api.key.sign((t + "GET" + WS_PATH).encode(),
                       padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": api.key_id, "KALSHI-ACCESS-TIMESTAMP": t, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


def schedule(league: str, day: dt.date) -> list[dict]:
    """The day's games: id (statsapi gamePk / ESPN event id), scheduled start, and both teams' codes in
    the source's spelling (home / away) and Kalshi's (home_k / away_k)."""
    if league == "MLB":
        d = httpx.get(f"{MLB_API}/api/v1/schedule", params={"sportId": 1, "date": day.isoformat(), "hydrate": "team"},
                      timeout=30).json()
        games = [{"game_id": str(g["gamePk"]), "start": g["gameDate"], "game_type": g["gameType"],
                  "away": g["teams"]["away"]["team"]["abbreviation"], "home": g["teams"]["home"]["team"]["abbreviation"]}
                 | ({"series_game": g["seriesGameNumber"]} if g["gameType"] in ("F", "D", "L", "W") and g.get("seriesGameNumber") else {})
                 for dd in d["dates"] for g in dd["games"]]
    else:
        d = httpx.get(f"{ESPN_LEAGUE[league]}/scoreboard", params={"dates": day.strftime("%Y%m%d"), "limit": 100}, timeout=30).json()
        games = []
        for e in d.get("events", []):
            c = {x["homeAway"]: x["team"] for x in e["competitions"][0]["competitors"]}
            games.append({"game_id": str(e["id"]), "start": e["date"].replace("Z", ":00Z") if len(e["date"]) == 17 else e["date"],
                          "game_type": str((e.get("season") or {}).get("type")), "away": c["away"]["abbreviation"],
                          "home": c["home"]["abbreviation"], "home_espn_id": c["home"]["id"]})
    for g in games:
        g["league"] = league
    return games


def spellings(league: str, code: str) -> set[str]:
    return next((s for s in SPELLINGS[league] if code in s), {code})


def contracts(api: Api, g: dict, day: dt.date, params: dict | None = None) -> list[dict]:
    """The game's open Kalshi contracts (or those params select, e.g. a close-time window for a past game). Event tickers carry the ET date and away + home Kalshi codes, but
    not always the same start time across series. Sets g["away_k"] / g["home_k"], the codes Kalshi used.
    Postseason (statsapi seriesGameNumber): Kalshi titles each game-winner event "Game N: ..." and after a postponement
    re-points its events by game number, not by MLB's game id (16 Oct 2026: TB @ CLE Game 4 rained out; Kalshi's
    26OCT16 contracts settled on Saturday's Game 4, its 26OCT17 ones on Sunday's Game 5), so the open event titled
    with the game's number decides the tag; otherwise the day's date does."""
    tag = day.strftime("%y%b%d").upper()
    pairs = [(a, h) for a in spellings(g["league"], g["away"]) for h in spellings(g["league"], g["home"])]
    if g.get("series_game") and params is None:
        for e in api.paged("/events", "events", {"series_ticker": SERIES[g["league"]][0], "status": "open"}, 200):
            ev = e["event_ticker"].split("-", 1)[1]
            if (e.get("title") or "").startswith(f"Game {g['series_game']}:") and any(ev.endswith(a + h) for a, h in pairs):
                tag = ev[:7]
                break
    out = []
    for s in SERIES[g["league"]]:
        for m in api.paged("/markets", "markets", {"series_ticker": s} | (params or {"status": "open"}), 200):
            ev = m["event_ticker"].split("-", 1)[1]
            hit = next(((a, h) for a, h in pairs if ev.startswith(tag) and ev.endswith(a + h)), None)
            if hit:
                g["away_k"], g["home_k"] = hit
                out.append({k: m.get(k) for k in MARKET_FIELDS} | {"series_ticker": s, "open_time": m.get("open_time"),
                                                                    "close_time": m.get("close_time")})
    g.setdefault("away_k", g["away"]), g.setdefault("home_k", g["home"])
    return out


class Hooks:
    """What a caller wants to hear about as data arrive (each called from a poll thread)."""

    def __init__(self, on_feed: Callable[[dict, dt.datetime], None] | None = None,
                 on_trades: Callable[[list[dict]], None] | None = None,
                 on_candles: Callable[[list[dict]], None] | None = None) -> None:
        self.on_feed, self.on_trades, self.on_candles = on_feed, on_trades, on_candles


class Recorder:
    def __init__(self, api: Api, g: dict, cons: list[dict], root: Path, hooks: Hooks | None = None) -> None:
        self.api, self.g, self.cons, self.root = api, g, cons, root
        self.hooks = hooks or Hooks()
        self.lock = threading.Lock()
        self.trade_seen: set[str] = set()
        self.trade_from: dict[str, int] = {}          # per ticker, the REST poll's own watermark (not the stream's)
        self.stream = api.key is not None             # trades from the WebSocket (needs a Kalshi key)
        self.final_at: dt.datetime | None = None
        self.last_digest: str | None = None
        self.books_every: float | None = POLL_BOOK_S     # None: no book polling (live_books_ws.py streams every change)
        (root / "feed").mkdir(parents=True, exist_ok=True)
        (root / "game.json").write_text(json.dumps({**g, "contracts": cons}, indent=1))
        self.feed_seen = {p.name.split(".")[0] for p in (root / "feed").glob("*.json.gz")}

    def _save_feed(self, stamp: str, doc: dict, got: dt.datetime, extra: dict) -> None:
        (self.root / "feed" / f"{stamp}.json.gz").write_bytes(gzip.compress(json.dumps(doc, separators=(",", ":")).encode()))
        self.feed_seen.add(stamp)
        append(self.root / "feed_arrivals.jsonl", [{"stamp": stamp, "received_at": iso(got)} | extra], self.lock)
        if self.hooks.on_feed:
            self.hooks.on_feed(doc, got)

    def poll_feed(self, hc: httpx.Client) -> None:
        if self.g["league"] == "MLB":
            self._poll_mlb(hc)
        else:
            self._poll_espn(hc)

    def _poll_mlb(self, hc: httpx.Client) -> None:
        pk = self.g["game_id"]
        codes = hc.get(f"{MLB_API}/api/v1.1/game/{pk}/feed/live/timestamps", params=bust()).json()
        if not codes or codes[-1] in self.feed_seen:
            return
        got = now()
        feed = hc.get(f"{MLB_API}/api/v1.1/game/{pk}/feed/live", params=bust()).json()
        wp = hc.get(f"{MLB_API}/api/v1/game/{pk}/winProbability", params=bust()).json() or []
        wp = [{k: w.get(k) for k in ("atBatIndex", "homeTeamWinProbability", "awayTeamWinProbability",
                                     "homeTeamWinProbabilityAdded")} for w in wp]
        self._save_feed(feed["metaData"]["timeStamp"], {"feed": feed, "win_probability": wp}, got,
                        {"latest_listed": codes[-1]})
        state = feed["gameData"]["status"]["detailedState"]
        if state in FINAL and self.final_at is None:
            self.final_at = got
            print(f"{pk}: {state} at {iso(got)}", flush=True)

    def _poll_espn(self, hc: httpx.Client) -> None:
        got = now()
        doc = hc.get(f"{ESPN_LEAGUE[self.g['league']]}/summary", params={"event": self.g["game_id"]}).json()
        drives = doc.get("drives") or {}
        plays = ([p.get("id") for p in doc.get("plays") or []] if self.g["league"] == "NBA" else
                 [p.get("id") for d in drives.get("previous", []) + ([drives["current"]] if drives.get("current") else [])
                  for p in d.get("plays", [])])
        wp = [(w.get("playId"), w.get("homeWinPercentage")) for w in doc.get("winprobability") or []]
        status = (((doc.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type") or {}
        digest = hashlib.sha1(json.dumps([plays, wp, status.get("state")]).encode()).hexdigest()
        if digest == self.last_digest:
            return
        self.last_digest = digest
        self._save_feed(got.strftime("%Y%m%d_%H%M%S_%f"), doc, got, {"plays": len(plays)})
        if status.get("state") == "post" and self.final_at is None:
            self.final_at = got
            print(f"{self.g['game_id']}: final at {iso(got)}", flush=True)

    def _take_trades(self, rows: list[dict], got: dt.datetime, via: str) -> None:
        """Keep the trades not seen before (the stream and the backup poll overlap) and pass them on."""
        with self.lock:
            new = [r for r in rows if r["trade_id"] not in self.trade_seen]
            self.trade_seen.update(r["trade_id"] for r in new)
        new = [{**r, "received_at": iso(got), "via": via} for r in new]
        append(self.root / "trades.jsonl", new, self.lock)
        if new and self.hooks.on_trades:
            self.hooks.on_trades(new)

    def poll_trades(self) -> None:
        for c in self.cons:
            t = c["ticker"]
            params = {"ticker": t}
            if t in self.trade_from:
                params["min_ts"] = self.trade_from[t]
            got = now()
            rows = self.api.paged("/markets/trades", "trades", params, 1000)
            for r in rows:                             # the watermark moves on every trade returned, seen or not
                ct = int(dt.datetime.fromisoformat(r["created_time"].replace("Z", "+00:00")).timestamp())
                self.trade_from[t] = max(self.trade_from.get(t, 0), ct - 1)   # min_ts is whole seconds
            self._take_trades(rows, got, "rest")

    def stream_trades(self, stop: threading.Event) -> None:
        """The game's trades from Kalshi's WebSocket trade channel, in the REST trade's shape, until stop is set.
        On a dropped connection: reconnect after 5 s (the backup poll fills the gap)."""
        import websockets
        tickers = [c["ticker"] for c in self.cons]

        def row(m: dict) -> dict:
            return {"trade_id": m["trade_id"], "ticker": m["market_ticker"], "taker_side": m["taker_side"],
                    "yes_price_dollars": m.get("yes_price_dollars"), "no_price_dollars": m.get("no_price_dollars"),
                    "count_fp": m.get("count_fp"), "is_block_trade": m.get("is_block_trade"),
                    "created_time": dt.datetime.fromtimestamp(m["ts_ms"] / 1000, dt.timezone.utc).isoformat(timespec="milliseconds")}

        async def run() -> None:
            while not stop.is_set():
                try:
                    async with websockets.connect(WS_URL, additional_headers=ws_headers(self.api), max_size=None,
                                                  ping_interval=20) as ws:
                        await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                                  "params": {"channels": ["trade"], "market_tickers": tickers}}))
                        print(f"{self.g['game_id']}: trade stream connected ({len(tickers)} contracts)", flush=True)
                        while not stop.is_set():
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                            except asyncio.TimeoutError:
                                continue
                            m = json.loads(raw)
                            if m.get("type") == "trade":
                                self._take_trades([row(m["msg"])], now(), "ws")
                            elif m.get("type") == "error":
                                print(f"{self.g['game_id']}: trade stream error message {m}", flush=True)
                except Exception as e:
                    print(f"{self.g['game_id']}: trade stream dropped {e!r}; reconnecting in 5 s", flush=True)
                    await asyncio.sleep(5)

        asyncio.run(run())

    def poll_books(self) -> None:
        rows = []
        for c in self.cons:
            got = now()
            d = self.api.get(f"/markets/{c['ticker']}/orderbook", {})
            if d:
                rows.append({"ticker": c["ticker"], "received_at": iso(got), **d})
        append(self.root / "books.jsonl", rows, self.lock)

    def poll_candles(self) -> None:
        if not self.cons:
            return
        got = now()
        end = int(got.timestamp())
        d = self.api.get("/markets/candlesticks", {"market_tickers": ",".join(c["ticker"] for c in self.cons),
                                                   "start_ts": end - 3 * 60, "end_ts": end,
                                                   "period_interval": 1}) or {}
        rows = [{"ticker": mk.get("market_ticker") or mk.get("ticker"), "received_at": iso(got), **c}
                for mk in d.get("markets") or [] for c in mk.get("candlesticks") or []]
        append(self.root / "candles.jsonl", rows, self.lock)
        if rows and self.hooks.on_candles:
            self.hooks.on_candles(rows)


def loop(fn, every: float, stop: threading.Event, name: str) -> None:
    while not stop.is_set():
        t0 = time.monotonic()
        try:
            fn()
        except Exception as e:                       # keep recording through one bad poll
            print(f"{name}: {e!r}", flush=True)
        stop.wait(max(0.0, every - (time.monotonic() - t0)))


def run_game(rec: Recorder, once: bool, stop: threading.Event | None = None) -> None:
    """Poll until the game is final + AFTER (or stop is set)."""
    hc = httpx.Client(timeout=30)
    polls = [(lambda: rec.poll_feed(hc), POLL_FEED_S, "feed"),
             (rec.poll_trades, POLL_TRADES_BACKUP_S if rec.stream and not once else POLL_TRADES_S, "trades"),
             (rec.poll_books, rec.books_every, "books"), (rec.poll_candles, POLL_CANDLE_S, "candles")]
    polls = [x for x in polls if x[1]]
    if once:
        for fn, _, name in polls:
            fn()
            print(f"  {name}: ok", flush=True)
        return
    stop = stop or threading.Event()
    threads = [threading.Thread(target=loop, args=(fn, every, stop, name), daemon=True) for fn, every, name in polls]
    if rec.stream:
        threads.append(threading.Thread(target=rec.stream_trades, args=(stop,), daemon=True))
    for t in threads:
        t.start()
    while not stop.is_set() and not (rec.final_at and now() > rec.final_at + AFTER):
        time.sleep(5)
    stop.set()
    for t in threads:
        t.join()


def game_root(out: Path, day: dt.date, g: dict) -> Path:
    return out / day.isoformat() / f"{g['league']}_{g['game_id']}"


def wait_until_start(g: dict) -> None:
    start = dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00"))
    wait = (start - BEFORE - now()).total_seconds()
    if wait > 0:
        print(f"{g['league']} {g['game_id']}: waiting {wait / 3600:.1f} h", flush=True)
        time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["MLB", "NFL", "NBA"], default="MLB")
    ap.add_argument("--date", type=dt.date.fromisoformat, default=now().astimezone(ET).date())
    ap.add_argument("--games", nargs="*", help="game ids (default: all that day)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--once", action="store_true", help="one round of every poll, then stop")
    ap.add_argument("--no-books", action="store_true", help="no order-book polling (live_books_ws.py streams them)")
    ap.add_argument("--rate", type=float, default=15.0, help="Kalshi requests per second (a key's runners share 20)")
    a = ap.parse_args()
    api = Api(rate=a.rate)
    games = [g for g in schedule(a.league, a.date) if not a.games or g["game_id"] in a.games]
    threads = []
    for g in games:
        cons = contracts(api, g, a.date)
        print(f"{g['league']} {g['game_id']} {g['away']}@{g['home']} {g['start']}: {len(cons)} contracts", flush=True)
        rec = Recorder(api, g, cons, game_root(a.out, a.date, g))
        if a.no_books:
            rec.books_every = None
        if a.once:
            run_game(rec, once=True)
            continue

        def wait_then_run(g=g, rec=rec) -> None:
            wait_until_start(g)
            print(f"{g['game_id']}: recording from {iso(now())}", flush=True)
            run_game(rec, once=False)
            print(f"{g['game_id']}: done", flush=True)

        threads.append(threading.Thread(target=wait_then_run))
        threads[-1].start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
