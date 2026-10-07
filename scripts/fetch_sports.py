"""Pull Kalshi MLB / NBA / NFL markets, 1-minute candles and trades, 2024 onward.

Stages, run in order; each one resumes where a previous run stopped:

    venv/bin/python scripts/fetch_sports.py list      # every market on the chosen series
    venv/bin/python scripts/fetch_sports.py estimate  # trades / disk / time, before pulling
    venv/bin/python scripts/fetch_sports.py seed      # reuse legs data/trades/ already holds in full
    venv/bin/python scripts/fetch_sports.py candles   # bid/ask candles (see "candles" below)
    venv/bin/python scripts/fetch_sports.py trades    # every trade, all listed markets
    venv/bin/python scripts/fetch_sports.py futures1m # 1-min candles on season-long legs (optional)

``--sports MLB NBA NFL`` picks the leagues (series whose ticker starts with
``KX<league>``; parlay/MVE series are not in the Sports category and are never
pulled). ``--since 2025-01-01`` (default) bounds market close time in every stage.

Candles: game-level legs (see ``game_level``) get 1-min candles, winner
markets (…GAME) over their whole life and every other game-level leg over the
last ``--hours`` (12) before it closes; season-long legs get hourly candles
over their whole life (``futures1m`` pulls them again at 1 min, over their whole
life, into ``candles_futures_1m/``; 1-min candles exist only in active minutes,
~3% of a future's life). Kalshi's historical and live candle endpoints name
their fields differently; ``flatten_candle`` reads both.

Holdout: markets closing on or after ``HOLDOUT_START`` are written under
``data/sports/_HOLDOUT_DO_NOT_USE/`` and never into the working tree, the same
convention as the macro 2026 block. Only counts and sizes are printed for them.

Layout (``data/`` is git-ignored):
    data/sports/markets.parquet                     listing, working block
    data/sports/candles/<series>/<n>.parquet        candles (1-min game-level, hourly season-long)
    data/sports/candles_futures_1m/<series>/<n>.parquet  1-min candles, season-long legs only
    data/sports/trades/<series>/<n>.parquet         trades
    data/sports/_HOLDOUT_DO_NOT_USE/{markets.parquet,candles/,trades/}
    data/sports/state/                              resume checkpoints

Schema differs from ``data/trades/`` in one way: Kalshi now allows fractional
contracts, so ``count`` and prices are Float64 (prices in cents).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import faulthandler
import os
import random
import shutil
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import polars as pl
import psutil

B = "https://api.elections.kalshi.com/trade-api/v2"
API_PREFIX = "/trade-api/v2"                 # signed paths include it
# Optional API key: signed reads get the account's tier budget (Basic: 200 tokens/s,
# 10 per request = 20 req/s) instead of the unauthenticated limit, which throttles
# candle endpoints to ~4 req/s. KALSHI_KEY_ID is the key's id; the RSA private key
# is read from KALSHI_KEY_FILE (default: kalshi_api_key.txt at the repo root, git-ignored).
KEY_ID_ENV, KEY_FILE_ENV = "KALSHI_KEY_ID", "KALSHI_KEY_FILE"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sports"
HOLD = OUT / "_HOLDOUT_DO_NOT_USE"
STATE = OUT / "state"
HOLDOUT_START = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
UTC = dt.timezone.utc

# A leg is "game-level" (1-min candles over the last --hours before close) when
# it is open at most GAME_LEVEL_MAX_OPEN, or when it resolves on the game day
# its event ticker names and is open at most GAME_DAY_MAX_OPEN (winner markets:
# any length) -- NFL game legs list ~12 days ahead, some months ahead. Season
# futures, and dated season-long legs (stat leaders, trade deadlines), get
# hourly candles over their whole life instead.
GAME_LEVEL_MAX_OPEN = dt.timedelta(days=10)
GAME_DAY_MAX_OPEN = dt.timedelta(days=45)
# Winner markets (…GAME) trade and reprice all through the days before a game
# (NFL winner 3-7 days out: a trade in 77% of hours, 1c spread); spreads, totals
# and props are mostly stale then (NFL spread: 9% of hours, 7+ days out 65c
# wide). So winners get 1-min candles over their whole life, the rest --hours.
FULL_LIFE_SERIES = re.compile(r"GAME$")
GAME_DAY = re.compile(r"^(\d{2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2})")
CANDLE_CHUNK = dt.timedelta(days=3)          # the API caps a request at ~5,000 periods
BYTES_PER_TRADE = 49.2                       # measured on data/trades/ (parquet, zstd)
BYTES_PER_CANDLE = 40.0                      # rough; flattened float columns
MIN_FREE_GB = 5.0                            # the pull stops (resumably) below this
FLUSH_LEGS = 10_000                          # also save every N legs, so an interruption loses little
RATE_CUT, RATE_CUT_EVERY_S = 0.75, 2.0         # on a 429: rate *= 0.75, at most once per 2 s
RATE_STEP_UP, RATE_MIN = 0.02, 1.0           # each success: rate += 0.02 req/s (~+1/s per 50 ok)
HEARTBEAT_S = 60                             # log memory / in-flight legs / retries this often
RSS_FLUSH_GB = 6.0                           # flush the buffer early when the process grows past this
BIG_LEG_ROWS = 500_000                       # log any single leg this large when it lands

# ticker -> [start monotonic, rows fetched so far] for legs being pulled right now,
# so the heartbeat can say which legs are in flight if the process balloons
INFLIGHT: dict[str, list] = {}


# --------------------------------------------------------------------------- http

class Api:
    """Shared client with a global, adaptive request rate and retry on 429/5xx.

    The rate starts at ``rate`` and adapts (AIMD): a 429 cuts it by a quarter (at
    most once per ``RATE_CUT_EVERY_S``, since concurrent workers all hit the same
    wall at once), and every success adds a little back, up to ``rate``. Kalshi's
    historical candle endpoint throttles well below the trades endpoints, so a
    fixed rate either wastes a third of its requests on 429s or is too slow."""

    def __init__(self, rate: float = 12.0) -> None:
        self.c = httpx.Client(timeout=60, limits=httpx.Limits(max_connections=32))
        self.key_id, self.key = os.environ.get(KEY_ID_ENV), None
        if self.key_id:
            from cryptography.hazmat.primitives import serialization
            key_file = Path(os.environ.get(KEY_FILE_ENV, ROOT / "kalshi_api_key.txt"))
            self.key = serialization.load_pem_private_key(key_file.read_bytes(), password=None)
        self.max_rate = rate
        self.rate = rate
        self.lock = threading.Lock()
        self.next_t = 0.0
        self.last_cut = 0.0
        self.n = 0
        self.retries = {"transport": 0, "429": 0, "5xx": 0}   # for the heartbeat

    def _headers(self, path: str) -> dict:
        """Kalshi request signing: RSA-PSS(SHA-256) over timestamp_ms + method + path (no query)."""
        if not self.key:
            return {}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        t = str(int(time.time() * 1000))
        sig = self.key.sign((t + "GET" + API_PREFIX + path).encode(),
                            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                            hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": t,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    def _ok(self) -> None:
        with self.lock:
            self.rate = min(self.max_rate, self.rate + RATE_STEP_UP)

    def _throttled(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now - self.last_cut >= RATE_CUT_EVERY_S:
                self.rate = max(RATE_MIN, self.rate * RATE_CUT)
                self.last_cut = now

    def _wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + 1.0 / self.rate
            self.n += 1
        if t > now:
            time.sleep(t - now)

    def get(self, path: str, params: dict) -> dict | None:
        for attempt in range(8):
            self._wait()
            try:
                r = self.c.get(B + path, params=params, headers=self._headers(path))
            except httpx.TransportError as e:
                self.retries["transport"] += 1
                if attempt >= 3:
                    print(f"  retry {attempt + 1}/8 {path} {params.get('ticker', '')}: {e!r}", flush=True)
                time.sleep(2 ** min(attempt, 5))
                continue
            if r.status_code == 429:
                # the shared rate does the pacing; this worker only steps back briefly
                self.retries["429"] += 1
                self._throttled()
                if attempt >= 3:
                    print(f"  retry {attempt + 1}/8 {path} {params.get('ticker', '')}: HTTP 429 "
                          f"(rate now {self.rate:.1f}/s)", flush=True)
                time.sleep(float(r.headers.get("retry-after", 1 + random.random())))
                continue
            if r.status_code >= 500:
                self.retries["5xx"] += 1
                if attempt >= 3:
                    print(f"  retry {attempt + 1}/8 {path} {params.get('ticker', '')}: HTTP {r.status_code}", flush=True)
                time.sleep(float(r.headers.get("retry-after", 2 ** min(attempt, 5))))
                continue
            self._ok()
            if r.status_code == 404:
                return None
            if r.is_error:
                raise RuntimeError(f"HTTP {r.status_code} on {path} {params}: {r.text[:200]}")
            return r.json()
        raise RuntimeError(f"gave up after 8 tries on {path} {params}")

    def paged(self, path: str, key: str, params: dict, limit: int, page_fn=None) -> list:
        # page_fn turns each page into something compact as it arrives, so a
        # leg with millions of rows never holds them all as raw JSON dicts
        out, cursor = [], None
        while True:
            p = dict(params, limit=limit)
            if cursor:
                p["cursor"] = cursor
            d = self.get(path, p)
            if not d:
                return out
            page = d.get(key) or []
            if params.get("ticker") in INFLIGHT:
                INFLIGHT[params["ticker"]][1] += len(page)
            if page_fn and page:
                out.append(page_fn(page))
            else:
                out.extend(page)
            cursor = d.get("cursor")
            if not cursor or not d.get(key):
                return out


def ts(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def fnum(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- list

MARKET_COLS = ["ticker", "event_ticker", "series_ticker", "market_type", "title",
               "yes_sub_title", "no_sub_title", "status", "result", "open_time",
               "close_time", "expiration_time", "settlement_ts", "volume", "open_interest",
               "last_price", "expiration_value", "strike_type", "rules_primary"]


# Pro-league season markets whose tickers lack the KX<league> prefix: the
# championships, conference titles and pennants that close the regular-season ->
# postseason chain, plus awards, stat leaders and record markets. Coaching,
# next-team, entertainment and college series are left out on purpose.
EXTRA = {
    "NFL": ["KXSB", "KXAFC", "KXNFC", "KXNFCAFCSB", "KXTEAMSINSB"],
    "NBA": ["KXTEAMSINNBAF"],
    "MLB": ["KXWSAL", "KXWSNL", "KXTEAMSINWS", "KXALMVP", "KXNLMVP", "KXALALLSTAR", "KXOHTANIMVPCY"],
}
EXTRA_PAT = {
    "NFL": r"^KX(LEADERNFL|LEADERPINT|RECORDNFL)",
    "NBA": r"^KX(LEADERNBA|RECORDNBA|WINSTREAK)",
    "MLB": r"^KXLEADERMLB",
}


def series_list(api: Api, sports: list[str]) -> list[str]:
    d = api.get("/series", {"category": "Sports"}) or {}
    pat = re.compile(rf"^KX({'|'.join(sports)})")
    extra_pat = re.compile("|".join(EXTRA_PAT[s] for s in sports))
    extra = {t for s in sports for t in EXTRA[s]}
    return sorted(s["ticker"] for s in d.get("series", [])
                  if pat.match(s["ticker"]) or s["ticker"] in extra or extra_pat.match(s["ticker"]))


def market_row(m: dict, series: str) -> dict:
    return {
        "ticker": m["ticker"], "event_ticker": m.get("event_ticker"), "series_ticker": series,
        "market_type": m.get("market_type"), "title": m.get("title"),
        "yes_sub_title": m.get("yes_sub_title"), "no_sub_title": m.get("no_sub_title"),
        "status": m.get("status"), "result": m.get("result"),
        "open_time": ts(m.get("open_time")), "close_time": ts(m.get("close_time")),
        "expiration_time": ts(m.get("expiration_time")), "settlement_ts": ts(m.get("settlement_ts")),
        "volume": fnum(m.get("volume_fp")), "open_interest": fnum(m.get("open_interest_fp")),
        "last_price": (fnum(m.get("last_price_dollars")) or 0) * 100,
        "expiration_value": m.get("expiration_value"), "strike_type": m.get("strike_type"),
        "rules_primary": m.get("rules_primary"),
    }


def stage_list(api: Api, sports: list[str], since: dt.datetime, workers: int) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    series = series_list(api, sports)
    print(f"series: {len(series)} ({', '.join(sports)})", flush=True)
    part_dir = STATE / "listing"
    part_dir.mkdir(exist_ok=True)
    todo = [s for s in series if not (part_dir / f"{s}.parquet").exists()]

    def one(s: str) -> int:
        rows: dict[str, dict] = {}
        # historical first, live second: a leg on both keeps the live (newer) record
        for path in ("/historical/markets", "/markets"):
            for m in api.paged(path, "markets", {"series_ticker": s}, 1000):
                ct = ts(m.get("close_time"))
                if ct and ct >= since:
                    rows[m["ticker"]] = market_row(m, s)
        df = pl.DataFrame(list(rows.values()), schema_overrides={"volume": pl.Float64,
                          "open_interest": pl.Float64, "last_price": pl.Float64},
                          infer_schema_length=None) if rows else pl.DataFrame()
        df.write_parquet(part_dir / f"{s}.parquet")
        return len(rows)

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, s): s for s in todo}
        for i, f in enumerate(as_completed(futs), 1):
            n = f.result()
            if i % 25 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] series listed ({futs[f]}: {n} legs)", flush=True)

    parts = [pl.read_parquet(p) for p in part_dir.glob("*.parquet")]
    mk = pl.concat([p for p in parts if p.height], how="diagonal_relaxed").unique("ticker")
    work = mk.filter(pl.col("close_time") < HOLDOUT_START)
    hold = mk.filter(pl.col("close_time") >= HOLDOUT_START)
    OUT.mkdir(parents=True, exist_ok=True)
    HOLD.mkdir(parents=True, exist_ok=True)
    work.write_parquet(OUT / "markets.parquet")
    hold.write_parquet(HOLD / "markets.parquet")
    print(f"\nworking block: {work.height:,} legs, close {work['close_time'].min()} -> "
          f"{work['close_time'].max()}")
    print(f"holdout (close >= {HOLDOUT_START.date()}): {hold.height:,} legs  [counts only]")


# --------------------------------------------------------------------------- estimate

def league(series: str) -> str:
    for lg in EXTRA:
        if series in EXTRA[lg] or re.match(EXTRA_PAT[lg], series):
            return lg
    return series[2:5]


def family(series: str) -> str:
    """Coarse contract family, for the per-family contracts-per-trade ratio."""
    s = series[5:]
    for k in ("GAME", "SPREAD", "TOTAL", "WINS", "PLAYOFF", "MVP"):
        if s.startswith(k):
            return k
    return "OTHER"


def load_all_markets(since: dt.datetime) -> pl.DataFrame:
    frames = [pl.read_parquet(p) for p in (OUT / "markets.parquet", HOLD / "markets.parquet") if p.exists()]
    return pl.concat(frames, how="diagonal_relaxed").filter(pl.col("close_time") >= since)


def stage_estimate(api: Api, since: dt.datetime, workers: int, hours: float | None, per_series: int = 6) -> None:
    mk = load_all_markets(since).with_columns(
        pl.col("series_ticker").map_elements(league, return_dtype=pl.String).alias("league"),
        pl.col("series_ticker").map_elements(family, return_dtype=pl.String).alias("family"),
        (pl.col("close_time") >= HOLDOUT_START).alias("holdout"),
        pl.col("close_time").dt.year().alias("year"),
    )
    traded = mk.filter(pl.col("volume") > 0)
    # contracts per trade: first page (<= 1000 trades) of a random sample per series
    random.seed(0)
    sample = []
    for s, g in traded.group_by("series_ticker"):
        tick = g["ticker"].to_list()
        sample += random.sample(tick, min(per_series, len(tick)))

    def cpt(t: str) -> tuple[str, float, int]:
        rows = []
        for path in ("/historical/trades", "/markets/trades"):
            d = api.get(path, {"ticker": t, "limit": 1000}) or {}
            rows += d.get("trades") or []
        n = len(rows)
        return t, (sum(float(r["count_fp"]) for r in rows) / n) if n else float("nan"), n

    res = []
    with ThreadPoolExecutor(workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(cpt, t) for t in sample]), 1):
            res.append(f.result())
            if i % 200 == 0:
                print(f"  sampled {i}/{len(sample)}", flush=True)
    smp = (pl.DataFrame(res, schema=["ticker", "cpt", "n"], orient="row")
           .filter(pl.col("n") > 0).join(mk.select("ticker", "league", "family"), on="ticker"))
    # median contracts/trade per (league, family); fall back to the league median
    fam = smp.group_by("league", "family").agg(pl.col("cpt").median().alias("cpt_f"))
    lea = smp.group_by("league").agg(pl.col("cpt").median().alias("cpt_l"))
    est = (traded.join(fam, on=["league", "family"], how="left").join(lea, on="league", how="left")
           .with_columns((pl.col("volume") / pl.coalesce("cpt_f", "cpt_l")).alias("trades_est")))

    by = (est.group_by("league", "year", "holdout")
          .agg(pl.len().alias("legs"), pl.col("volume").sum().alias("contracts"),
               pl.col("trades_est").sum().alias("trades"))
          .with_columns((pl.col("trades") * BYTES_PER_TRADE / 1e9).alias("GB"),
                        # ~1 request per 1,000 trades, 2 endpoints per leg
                        ((pl.col("trades") / 1000 + 2 * pl.col("legs")) / api_rate / 3600).alias("hours"))
          .sort("league", "year", "holdout"))
    fam_tbl = (est.group_by("league", "family").agg(pl.len().alias("legs"),
               pl.col("trades_est").sum().alias("trades"), pl.col("cpt_f").first().alias("contracts_per_trade"))
               .with_columns((pl.col("trades") * BYTES_PER_TRADE / 1e9).alias("GB")).sort("GB", descending=True))

    is_game = [game_level(m) for m in mk.select("event_ticker", "series_ticker", "open_time", "close_time")
                                        .iter_rows(named=True)]
    gl = mk.filter(pl.Series(is_game))
    life_min = (gl["close_time"] - gl["open_time"]).dt.total_minutes().clip(0, None)
    full = gl["series_ticker"].map_elements(lambda s: bool(FULL_LIFE_SERIES.search(s)), return_dtype=pl.Boolean)
    if hours:
        life_min = pl.select(pl.when(full).then(life_min).otherwise(life_min.clip(0, hours * 60))).to_series()
    cand = float(life_min.sum()) * 0.6          # ~60% of minutes carry a candle (probe: 1,020/2,880 idle-inclusive)
    fut = mk.join(gl.select("ticker"), on="ticker", how="anti")
    fut_hours = float((fut["close_time"] - fut["open_time"]).dt.total_hours().clip(0, None).sum())

    with pl.Config(tbl_rows=60, tbl_width_chars=160, float_precision=2, thousands_separator=True):
        print(f"\nlegs listed: {mk.height:,}   with volume: {traded.height:,}   sampled for contracts/trade: {smp.height}")
        print("\n=== trades: by league x year (holdout = close >= %s) ===" % HOLDOUT_START.date())
        print(by)
        print("\n=== trades: by league x contract family ===")
        print(fam_tbl)
        tot = by.select(pl.col("trades").sum(), pl.col("GB").sum(), pl.col("hours").sum()).row(0)
        print(f"\nTRADES total: ~{tot[0]/1e6:,.0f}M trades, ~{tot[1]:,.1f} GB on disk, ~{tot[2]:,.1f} h at {api_rate:.0f} req/s")
        print(f"CANDLES: game-level legs {gl.height:,} -> ~{cand/1e6:,.1f}M 1-min candles "
              f"(~{cand*BYTES_PER_CANDLE/1e9:,.1f} GB, ~{gl.height * 1.3 / api_rate / 3600:,.1f} h); "
              f"futures {fut.height:,} legs hourly -> ~{fut_hours/1e6:,.2f}M candles")


# --------------------------------------------------------------------------- candles

CANDLE_FIELDS = ("open", "high", "low", "close", "mean", "previous")


def flatten_candle(c: dict, ticker: str) -> dict:
    """The live endpoint names fields volume_fp / close_dollars, the historical
    one volume / close (both in dollars): read either. Prices in cents."""
    row = {"ticker": ticker, "ts": dt.datetime.fromtimestamp(c["end_period_ts"], UTC),
           "volume": fnum(c.get("volume_fp", c.get("volume"))),
           "open_interest": fnum(c.get("open_interest_fp", c.get("open_interest")))}
    for side in ("yes_bid", "yes_ask", "price"):
        q = c.get(side) or {}
        for f in CANDLE_FIELDS:
            if f in ("mean", "previous") and side != "price":
                continue
            v = fnum(q.get(f"{f}_dollars", q.get(f)))
            row[f"{side}_{f}"] = v * 100 if v is not None else None
    return row


def game_level(m: dict) -> bool:
    if not m["open_time"] or not m["close_time"]:
        return False
    open_for = m["close_time"] - m["open_time"]
    if open_for <= GAME_LEVEL_MAX_OPEN:
        return True
    d = GAME_DAY.match((m["event_ticker"] or "").partition("-")[2])
    if not d or abs((m["close_time"].date() - dt.datetime.strptime(d.group(1), "%y%b%d").date()).days) > 2:
        return False
    return open_for <= GAME_DAY_MAX_OPEN or m["series_ticker"].endswith("GAME")


def candle_window(m: dict, hours: float | None) -> tuple[dt.datetime, dt.datetime, int] | None:
    """(start, end, minutes per candle) to pull for a leg, or None if there is nothing to pull.
    Game-level legs: 1-min candles, over the last ``hours`` before close unless a
    winner (…GAME) leg; season-long legs: hourly over their life. The end is
    clamped to now, so futures closing years out are not requested into the future."""
    game = game_level(m)
    interval = 1 if game else FUTURES_INTERVAL
    h = None if not game or FULL_LIFE_SERIES.search(m["series_ticker"]) else hours   # futures: whole life
    lo, hi = m["open_time"], m["close_time"]
    if lo and hi and interval == 1 and h is not None:
        lo = max(lo, hi - dt.timedelta(hours=h))
    if hi:
        hi = min(hi, dt.datetime.now(UTC))
    if not lo or not hi or hi <= lo:
        return None
    return lo, hi, interval


def candles_for(api: Api, m: dict, hours: float | None, cutoff: dt.datetime) -> list[dict]:
    w = candle_window(m, hours)
    if not w:
        return []
    lo, hi, interval = w
    step = CANDLE_CHUNK if interval == 1 else dt.timedelta(days=200)
    rows, t = [], lo
    while t < hi:
        u = min(t + step, hi)
        p = {"start_ts": int(t.timestamp()), "end_ts": int(u.timestamp()), "period_interval": interval}
        hist = f"/historical/markets/{m['ticker']}/candlesticks"
        live = f"/series/{m['series_ticker']}/markets/{m['ticker']}/candlesticks"
        # settled before the cutoff -> historical first; otherwise live first
        first, second = (hist, live) if m["close_time"] < cutoff else (live, hist)
        d = api.get(first, p) or api.get(second, p) or {}
        rows += [flatten_candle(c, m["ticker"]) for c in d.get("candlesticks") or []]
        if m["ticker"] in INFLIGHT:
            INFLIGHT[m["ticker"]][1] = len(rows)
        t = u
    return rows


# The live batch endpoint (GET /markets/candlesticks, up to 100 tickers) returns at
# most 10,000 candles per request across all its markets, and only serves legs
# settled after the historical cutoff. Kalshi throttles every candle endpoint to
# ~4 req/s, so for live legs one batch request replaces ~10-40 single ones.
BATCH_MAX_CANDLES = 9_000                    # headroom under the 10,000 cap
BATCH_MAX_TICKERS = 100


def candle_batches(legs: list[dict], hours: float | None) -> tuple[list[list[dict]], list[dict]]:
    """Pack live legs into batch requests that share one window; legs too long for
    any batch come back separately, for the single-leg path."""
    wins = [(m, candle_window(m, hours)) for m in legs]
    solo = [m for m, w in wins if w is None or (w[1] - w[0]).total_seconds() / 60 / w[2] > BATCH_MAX_CANDLES]
    solo_t = {m["ticker"] for m in solo}
    packable = sorted(((m, w) for m, w in wins if m["ticker"] not in solo_t), key=lambda mw: (mw[1][2], mw[1][0]))
    batches, cur, lo, hi, iv = [], [], None, None, None
    for m, (a, b, interval) in packable:
        if cur:
            nlo, nhi = min(lo, a), max(hi, b)
            # upper bound: every leg could have a candle every period of the shared window
            n = (nhi - nlo).total_seconds() / 60 / interval * (len(cur) + 1)
            if interval == iv and len(cur) < BATCH_MAX_TICKERS and n <= BATCH_MAX_CANDLES:
                cur.append((m, (a, b))); lo, hi = nlo, nhi
                continue
            batches.append(cur)
        cur, lo, hi, iv = [(m, (a, b))], a, b, interval
    if cur:
        batches.append(cur)
    return [[m for m, _ in bt] for bt in batches], solo


def candles_batch(api: Api, legs: list[dict], hours: float | None, cutoff: dt.datetime) -> dict[str, list[dict]]:
    """Candles for a packed batch, each leg's rows trimmed back to its own window.
    A leg the batch response leaves out is fetched on its own, so a quirk of the
    batch endpoint can only cost requests, never data."""
    wins = {m["ticker"]: candle_window(m, hours) for m in legs}
    lo = min(w[0] for w in wins.values()); hi = max(w[1] for w in wins.values())
    interval = next(iter(wins.values()))[2]
    d = api.get("/markets/candlesticks", {"market_tickers": ",".join(wins), "start_ts": int(lo.timestamp()),
                                          "end_ts": int(hi.timestamp()), "period_interval": interval}) or {}
    out: dict[str, list[dict]] = {}
    for mk in d.get("markets") or []:
        t = mk.get("market_ticker")
        if t not in wins:
            continue
        a, b = int(wins[t][0].timestamp()), int(wins[t][1].timestamp())
        out[t] = [flatten_candle(c, t) for c in mk.get("candlesticks") or [] if a <= c["end_period_ts"] <= b]
    for m in legs:
        if m["ticker"] not in out:
            out[m["ticker"]] = candles_for(api, m, hours, cutoff)
    return out


# --------------------------------------------------------------------------- trades

TRADE_SCHEMA = {"trade_id": pl.String, "ticker": pl.String, "count": pl.Float64, "yes_price": pl.Float64,
                "no_price": pl.Float64, "taker_side": pl.String,
                "created_time": pl.Datetime("us", "UTC"), "is_block_trade": pl.Boolean}


def trade_page(page: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{"trade_id": r["trade_id"], "ticker": r["ticker"], "count": fnum(r.get("count_fp")),
                          "yes_price": (fnum(r.get("yes_price_dollars")) or 0) * 100,
                          "no_price": (fnum(r.get("no_price_dollars")) or 0) * 100,
                          "taker_side": r.get("taker_side"), "created_time": ts(r["created_time"]),
                          "is_block_trade": r.get("is_block_trade")} for r in page], schema=TRADE_SCHEMA)


def trades_for(api: Api, m: dict, cutoff: dt.datetime) -> pl.DataFrame:
    """Trades created before Kalshi's ``/historical/cutoff`` live only on the
    historical endpoint, later ones only on the live one: ask only where a leg's
    [open, close] can have trades (+1 h slack either side)."""
    ticker, slack = m["ticker"], dt.timedelta(hours=1)
    pages = []
    if not m["open_time"] or m["open_time"] < cutoff + slack:
        pages += api.paged("/historical/trades", "trades", {"ticker": ticker}, 1000, trade_page)
    if not m["close_time"] or m["close_time"] > cutoff - slack:
        pages += api.paged("/markets/trades", "trades", {"ticker": ticker}, 1000, trade_page)
    return pl.concat(pages) if pages else pl.DataFrame(schema=TRADE_SCHEMA)


# --------------------------------------------------------------------------- scope

# Game-level contract kinds that make up "core": winner, spread, total and their
# half / first-5 / team variants, and series winners. Everything else that is
# game-level is a player or event prop; season-long legs (futures, awards,
# leaders, win totals) always count as core.
CORE_KIND = re.compile(r"^KX(MLB|NBA|NFL)(GAME|SPREAD|TOTAL|1H|2H|F5|TEAMTOTAL|SERIES|STGAME|SUMMERGAME)")


def in_scope(m: dict, scope: str) -> bool:
    if scope == "all":
        return True
    core = not game_level(m) or bool(CORE_KIND.match(m["series_ticker"]))   # season-long legs are core
    return core if scope == "core" else not core


# --------------------------------------------------------------------------- shared pull loop

def pull(api: Api, kind: str, since: dt.datetime, workers: int, flush_every: int, only: list[str] | None,
         scope: str, hours: float | None) -> int:
    mk = load_all_markets(since)
    mk = mk.filter(pl.col("volume") > 0)      # never-traded legs: no trades, near-empty quotes
    if only:
        mk = mk.filter(pl.col("series_ticker").is_in(only))
    done_f = STATE / f"{kind}_done.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    todo = [m for m in mk.sort("volume").iter_rows(named=True)
            if m["ticker"] not in done and in_scope(m, scope)]
    cutoff = ts((api.get("/historical/cutoff", {}) or {}).get("trades_created_ts")) or dt.datetime.now(UTC)
    print(f"{kind} [{scope}]: {len(todo):,} legs to pull ({len(done):,} already done); "
          f"historical cutoff {cutoff}", flush=True)

    lock = threading.Lock()
    buf: dict[tuple[bool, str], list[pl.DataFrame]] = {}
    newly: list[str] = []
    counter = {"legs": 0, "rows": 0, "part": int(time.time())}

    stop = threading.Event()

    def flush() -> None:
        for (hold, s), frames in list(buf.items()):
            if not frames:
                continue
            d = (HOLD if hold else OUT) / kind / s
            d.mkdir(parents=True, exist_ok=True)
            df = pl.concat(frames, how="diagonal_relaxed")
            if kind == "trades":
                df = df.unique("trade_id")
            else:   # consecutive CANDLE_CHUNK requests share their boundary candle
                df = df.unique(["ticker", "ts"], keep="first")
            counter["part"] += 1
            df.write_parquet(d / f"{counter['part']}.parquet")
        buf.clear()
        with open(done_f, "a") as f:
            f.write("".join(t + "\n" for t in newly))
        newly.clear()
        free = shutil.disk_usage(OUT).free / 1e9
        if free < MIN_FREE_GB and not stop.is_set():
            stop.set()
            print(f"STOPPING: {free:.1f} GB free < {MIN_FREE_GB} GB. Free space and rerun to resume.", flush=True)

    def label(m: dict) -> str:
        # holdout legs are never named in logs, only their series
        return f"holdout:{m['series_ticker']}" if m["close_time"] >= HOLDOUT_START else m["ticker"]

    # a unit is one leg, or (live candles) a batch of legs fetched in one request
    if kind == "candles_futures_1m":    # season-long legs only; game-level legs are already 1-min
        todo = [m for m in todo if not game_level(m)]
        print(f"{kind}: {len(todo):,} season-long legs to pull", flush=True)
    if kind.startswith("candles"):
        live = [m for m in todo if m["close_time"] >= cutoff]
        batches, solo = candle_batches(live, hours)
        units = batches + [[m] for m in solo] + [[m] for m in todo if m["close_time"] < cutoff]
        print(f"candles: {len(live):,} live legs -> {len(batches):,} batch requests + {len(solo):,} single; "
              f"{len(todo) - len(live):,} historical legs, one at a time", flush=True)
    else:
        units = [[m] for m in todo]

    def one(unit: list[dict]) -> list[tuple[dict, pl.DataFrame | Exception | None]]:
        if stop.is_set():
            return [(m, None) for m in unit]
        key = unit[0]["ticker"] if len(unit) == 1 else f"batch of {len(unit)}: {label(unit[0])}"
        INFLIGHT[key] = [time.monotonic(), 0]
        try:
            if kind == "trades":
                return [(unit[0], trades_for(api, unit[0], cutoff))]
            if len(unit) == 1:
                return [(unit[0], pl.DataFrame(candles_for(api, unit[0], hours, cutoff), infer_schema_length=None))]
            got = candles_batch(api, unit, hours, cutoff)
            return [(m, pl.DataFrame(got[m["ticker"]], infer_schema_length=None)) for m in unit]
        except Exception as e:        # one bad leg must not take the whole pull down
            return [(m, e) for m in unit]
        finally:
            INFLIGHT.pop(key, None)

    proc = psutil.Process()
    by_ticker = {m["ticker"]: m for m in todo}
    failed: list[tuple[str, str]] = []
    hb_stop = threading.Event()

    def heartbeat() -> None:
        while not hb_stop.wait(HEARTBEAT_S):
            now = time.monotonic()
            vm = psutil.virtual_memory()
            fl = sorted(INFLIGHT.items(), key=lambda kv: -kv[1][1])
            legs = ", ".join(f"{label(by_ticker[t]) if t in by_ticker else t} {r:,} rows/{(now - t0_)/60:.0f}m"
                             for t, (t0_, r) in fl[:6])
            print(f"HB {dt.datetime.now():%H:%M:%S} rss {proc.memory_info().rss/2**30:.2f} GB  "
                  f"sys avail {vm.available/2**30:.1f}/{vm.total/2**30:.0f} GB  swap {psutil.swap_memory().used/2**30:.1f} GB  "
                  f"buffered {sum(x.height for fs in buf.values() for x in fs):,} rows  "
                  f"done {counter['legs']:,}/{len(todo):,}  failed {len(failed)}  req {api.n:,}  rate {api.rate:.1f}/s  "
                  f"retries {api.retries}  in flight {len(INFLIGHT)}: {legs}", flush=True)

    t0 = time.time()
    threading.Thread(target=heartbeat, daemon=True).start()
    print(f"pid {os.getpid()}  {'signed with API key' if api.key else 'unsigned'}  workers {workers}  flush every {flush_every:,} rows / {FLUSH_LEGS:,} legs  "
          f"early flush above {RSS_FLUSH_GB} GB rss", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, u) for u in units]
        for f, (m, rows) in ((f, r) for f in as_completed(futs) for r in f.result()):
            if rows is None:          # skipped after a stop; not marked done
                continue
            with lock:
                if isinstance(rows, Exception):   # not marked done, so a rerun retries it
                    failed.append((m["ticker"], repr(rows)))
                    print(f"LEG FAILED {label(m)}: {rows!r}"[:400], flush=True)
                    continue
                hold = m["close_time"] >= HOLDOUT_START
                if rows.height:
                    buf.setdefault((hold, m["series_ticker"]), []).append(rows)
                if rows.height >= BIG_LEG_ROWS:
                    print(f"  big leg {label(m)}: {rows.height:,} {kind}, "
                          f"rss {proc.memory_info().rss/2**30:.2f} GB", flush=True)
                newly.append(m["ticker"])
                counter["legs"] += 1
                counter["rows"] += rows.height
                rss = proc.memory_info().rss / 2**30
                if (sum(x.height for fs in buf.values() for x in fs) >= flush_every or len(newly) >= FLUSH_LEGS
                        or counter["legs"] == len(todo) or rss > RSS_FLUSH_GB):
                    if rss > RSS_FLUSH_GB:
                        print(f"  early flush: rss {rss:.2f} GB > {RSS_FLUSH_GB} GB", flush=True)
                    flush()
                if counter["legs"] // 500 > counter.get("shown", 0) or counter["legs"] == len(todo):
                    counter["shown"] = counter["legs"] // 500
                    el = time.time() - t0
                    print(f"[{counter['legs']:,}/{len(todo):,}] legs  {counter['rows']:,} {kind}  "
                          f"{el/60:.0f} min  ({api.n:,} requests)", flush=True)
    flush()
    hb_stop.set()
    fail_f = STATE / f"{kind}_failed.txt"
    if failed:
        fail_f.write_text("".join(f"{t}\t{e}\n" for t, e in failed))
        print(f"{kind}: {len(failed):,} legs FAILED (listed in {fail_f}); "
              f"rerun the stage to retry them", flush=True)
    else:
        fail_f.unlink(missing_ok=True)
    print(f"{kind}: finished, {counter['legs']:,} legs, rate at end {api.rate:.1f}/s, {counter['rows']:,} rows, {len(failed)} failed, "
          f"retries {api.retries}", flush=True)
    return len(failed)


# --------------------------------------------------------------------------- seed

def stage_seed(since: dt.datetime) -> None:
    """Copy legs the local archive already holds in full, and mark them done.

    A leg is complete when ``data/trades/`` holds at least Kalshi's recorded
    volume in contracts. Partial legs are left for ``trades`` to re-pull whole.
    """
    import glob
    mk = load_all_markets(since).filter(pl.col("volume") > 0).select("ticker", "series_ticker", "close_time", "volume")
    arch = (pl.scan_parquet(glob.glob(str(ROOT / "data/trades/*.parquet")))
            .filter(pl.col("ticker").is_in(mk["ticker"].implode()))
            .select("trade_id", "ticker", "count", "yes_price", "no_price", "taker_side", "created_time")
            .unique("trade_id").collect())
    have = arch.group_by("ticker").agg(pl.col("count").sum().alias("local"))
    complete = (mk.join(have, on="ticker").filter(pl.col("local") >= 0.999 * pl.col("volume")))
    done_f = STATE / "trades_done.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    complete = complete.filter(~pl.col("ticker").is_in(list(done)))
    tr = (arch.join(complete.select("ticker", "series_ticker", "close_time"), on="ticker")
          .with_columns(pl.col("count").cast(pl.Float64), pl.col("yes_price").cast(pl.Float64),
                        pl.col("no_price").cast(pl.Float64), pl.lit(None, pl.Boolean).alias("is_block_trade"),
                        pl.col("created_time").cast(pl.Datetime("us", "UTC"))))
    tr = tr.with_columns((pl.col("close_time") >= HOLDOUT_START).alias("_hold"))
    for (s, hold), g in tr.group_by("series_ticker", "_hold"):
        d = (HOLD if hold else OUT) / "trades" / s
        d.mkdir(parents=True, exist_ok=True)
        g.drop("series_ticker", "close_time", "_hold").write_parquet(d / "seed_local_archive.parquet")
    STATE.mkdir(parents=True, exist_ok=True)
    with open(done_f, "a") as f:
        f.write("".join(t + "\n" for t in complete["ticker"]))
    print(f"seeded {complete.height:,} complete legs, {tr.height:,} trades, from data/trades/")


# --------------------------------------------------------------------------- main

api_rate = 12.0
FUTURES_INTERVAL = 60                        # minutes per candle on season-long legs; futures1m sets 1


def main() -> None:
    global api_rate, FUTURES_INTERVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "estimate", "seed", "candles", "trades", "futures1m"])
    ap.add_argument("--sports", nargs="+", default=["MLB", "NBA", "NFL"])
    ap.add_argument("--since", default="2025-01-01")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rate", type=float, default=18.0,
                    help="max requests/s across all workers (Kalshi throttles above ~20)")
    ap.add_argument("--scope", default="all", choices=["core", "props", "all"],
                    help="core = winner/spread/total variants + season markets; props = the rest")
    # 12 h: full-life 1-min candles on every game-level leg (~31 GB) would not fit
    # beside the trades; the last 12 h before close covers the game and its run-up
    ap.add_argument("--hours", type=float, default=12.0,
                    help="candles: only the last N hours before a game-level leg closes (0 = its whole life); "
                         "winner (…GAME) legs always get their whole life, season-long legs hourly candles")
    ap.add_argument("--flush-every", type=int, default=2_000_000)
    ap.add_argument("--only", nargs="+", help="restrict candles/trades to these series tickers")
    a = ap.parse_args()
    faulthandler.enable()                # dump every thread's stack on a hard crash (not on SIGKILL)
    api_rate = a.rate
    a.hours = a.hours or None          # 0 -> whole life
    api = Api(a.rate)
    since = dt.datetime.fromisoformat(a.since).replace(tzinfo=UTC)
    if a.stage == "list":
        stage_list(api, a.sports, since, a.workers)
    elif a.stage == "seed":
        stage_seed(since)
    elif a.stage == "estimate":
        stage_estimate(api, since, a.workers, a.hours)
    else:
        kind = a.stage
        if kind == "futures1m":
            FUTURES_INTERVAL, kind = 1, "candles_futures_1m"
        n_failed = pull(api, kind, since, a.workers, a.flush_every, a.only, a.scope, a.hours)
        print(f"requests made: {api.n:,}")
        return 3 if n_failed else 0      # 3 = finished, but some legs need a rerun
    print(f"requests made: {api.n:,}")


if __name__ == "__main__":
    sys.exit(main())
