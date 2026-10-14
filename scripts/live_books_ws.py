"""Record every order-book change (and every trade) of the day's game contracts from Kalshi's WebSocket feed, for
analysis; the trading runner does not read it. Channels orderbook_delta (a full snapshot on subscribing, then every
change, each with its sequence number) and trade. Every message is written as received, with the arrival time, to
<out>/<date>/<league>_<game>/books_ws.jsonl (gzipped to books_ws.jsonl.gz when the stream ends); on a dropped connection it reconnects and resubscribes (Kalshi then
sends fresh snapshots). Starts 30 min before the day's first game. Stops once nothing has arrived for IDLE (the
contracts stop changing soon after a game ends; 2026-10-12's last message came 5 h before the old fixed stop), but not
before the last game's start + MIN_RUN, and in any case 1 h after the last game's contracts close.

The stream needs a signed login: KALSHI_KEY_ID (env) and the private key file KALSHI_KEY_FILE (env).

    python scripts/live_books_ws.py --leagues MLB NFL [--date 2026-10-12] [--out /home/casperngeen/sports_live]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import gzip
import json
import os
import sys
import time
from pathlib import Path

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_record as R  # noqa: E402

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"
LEAD, TAIL = dt.timedelta(minutes=30), dt.timedelta(hours=1)
IDLE, MIN_RUN = dt.timedelta(minutes=30), dt.timedelta(hours=2)


def login_headers(key_id: str, key) -> dict:
    """Kalshi signing: RSA-PSS(SHA-256) over timestamp_ms + "GET" + the WebSocket path."""
    t = str(int(time.time() * 1000))
    sig = key.sign((t + "GET" + WS_PATH).encode(),
                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": key_id, "KALSHI-ACCESS-TIMESTAMP": t, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


class Sink:
    """ticker -> that game's books_ws.jsonl, flushed every few seconds."""

    def __init__(self, files: dict[str, Path]) -> None:
        self.f = {p: p.open("a") for p in set(files.values())}
        self.by_ticker = {t: self.f[p] for t, p in files.items()}
        self.other = next(iter(self.f.values()))
        self.last_flush, self.n, self.last_msg = time.monotonic(), 0, R.now()

    def write(self, raw: str, got: str) -> None:
        m = json.loads(raw)
        t = (m.get("msg") or {}).get("market_ticker")
        (self.by_ticker.get(t) or self.other).write(json.dumps({"received_at": got, **m}, separators=(",", ":")) + "\n")
        self.n += 1
        self.last_msg = R.now()
        if time.monotonic() - self.last_flush > 5:
            for f in self.f.values():
                f.flush()
            self.last_flush = time.monotonic()


async def stream(tickers: list[str], sink: Sink, key_id: str, key, until: dt.datetime, quiet_from: dt.datetime) -> None:
    def over() -> bool:
        t = R.now()
        return t >= until or (t >= quiet_from and t - sink.last_msg >= IDLE)

    while not over():
        try:
            async with websockets.connect(WS_URL, additional_headers=login_headers(key_id, key), max_size=None,
                                          ping_interval=20) as ws:
                await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                          "params": {"channels": ["orderbook_delta", "trade"], "market_tickers": tickers}}))
                print(f"{R.iso(R.now())} connected, {len(tickers)} contracts", flush=True)
                while not over():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    except asyncio.TimeoutError:
                        continue
                    sink.write(raw, R.iso(R.now()))
        except Exception as e:                        # dropped or refused: wait, reconnect, resubscribe (fresh snapshots)
            print(f"{R.iso(R.now())} stream error {e!r}; reconnecting in 5 s ({sink.n} messages so far)", flush=True)
            await asyncio.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=["MLB", "NFL"])
    ap.add_argument("--date", type=dt.date.fromisoformat, default=R.now().astimezone(R.ET).date())
    ap.add_argument("--out", type=Path, default=R.OUT)
    a = ap.parse_args()
    key_id = os.environ["KALSHI_KEY_ID"]
    key = serialization.load_pem_private_key(Path(os.environ["KALSHI_KEY_FILE"]).read_bytes(), password=None)
    api = R.Api(rate=4.0)
    files, starts, closes = {}, [], []
    for league in a.leagues:
        for g in R.schedule(league, a.date):
            cons = R.contracts(api, g, a.date)
            root = R.game_root(a.out, a.date, g)
            root.mkdir(parents=True, exist_ok=True)
            files |= {c["ticker"]: root / "books_ws.jsonl" for c in cons}
            if cons:
                starts.append(dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00")))
                closes += [dt.datetime.fromisoformat(c["close_time"].replace("Z", "+00:00")) for c in cons if c.get("close_time")]
            print(f"{league} {g['game_id']} {g['away']}@{g['home']} {g['start']}: {len(cons)} contracts", flush=True)
    if not files:
        print("no contracts today")
        return
    # a game's contracts close up to days after it ends (settlement), so stop at the last start + 8 h if that is sooner
    until = min(max(closes, default=max(starts) + dt.timedelta(hours=8)), max(starts) + dt.timedelta(hours=8)) + TAIL
    wait = (min(starts) - LEAD - R.now()).total_seconds()
    print(f"streaming {len(files)} contracts from {R.iso(min(starts) - LEAD)} to {R.iso(until)}", flush=True)
    if wait > 0:
        time.sleep(wait)
    sink = Sink(files)
    asyncio.run(stream(sorted(files), sink, key_id, key, until, max(starts) + MIN_RUN))
    print(f"{R.iso(R.now())} stopping; last message {R.iso(sink.last_msg)}", flush=True)
    for p in set(files.values()):                   # ~10x smaller; the raw file goes once its copy is complete
        sink.f[p].close()
        with p.open("rb") as src, gzip.open(p.with_suffix(".jsonl.gz"), "wb") as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
        p.unlink()
    print(f"done, {sink.n} messages", flush=True)


if __name__ == "__main__":
    main()
