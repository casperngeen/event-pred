"""Extend the quarantined 2026 OOS block past 2026-06-18.

Pulls every settled market (close_time in [2026-06-01, now)) on the core macro
series, plus the full trade history of each of those tickers, into
``data/_OOS_DO_NOT_USE_trades_2026_ext/``. Full history rather than
``min_ts=06-18`` so a July leg listed in June is complete; the overlap with
``trades_2026_oos.parquet`` is removed on ``trade_id`` at merge time.

Kalshi splits its archive at ``/historical/cutoff``: markets settled and trades
created before it are only on the ``/historical/*`` endpoints, later ones only
on the live ones. Both are queried and deduped.

Prints counts and date ranges only -- never prices or results. This is OOS
data; see ``stg/splits.py`` and ``data/MANIFEST_new_pulls.md``.

    venv/bin/python scripts/fetch_oos_extension.py
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import httpx
import polars as pl

B = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path(__file__).resolve().parents[1] / "data" / "_OOS_DO_NOT_USE_trades_2026_ext"
LO = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)

# Every series with a 2026 event in markets_api_pull_raw.jsonl.
SERIES = [
    "KXADP", "KXAIRFARECPI", "KXCPI", "KXCPICORE", "KXCPICOREYOY", "KXCPIYOY",
    "KXFED", "KXFEDDECISION", "KXGDP", "KXISMPMI", "KXJOBLESSCLAIMS",
    "KXPAYROLLS", "KXPCECORE", "KXSHELTERCPI", "KXU3", "KXUSDURABLE",
    "KXUSEDCARCPI", "KXUSGASCPI", "KXUSISMSERV", "KXUSMICHCSP", "KXUSNFP",
    "KXUSPPI", "KXUSPPIYOY", "KXUSRETAIL", "KXWTI",
]


def get(c: httpx.Client, path: str, params: dict) -> dict:
    for attempt in range(8):
        try:
            r = c.get(B + path, params=params)
        except httpx.TransportError:
            time.sleep(2 ** min(attempt, 5))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(int(r.headers.get("retry-after", 2 ** min(attempt, 5))))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {path} {params}")


def paged(c: httpx.Client, path: str, key: str, params: dict) -> list[dict]:
    out, cursor = [], None
    while True:
        p = dict(params, limit=1000 if key == "trades" else 200)
        if cursor:
            p["cursor"] = cursor
        d = get(c, path, p)
        out.extend(d.get(key, []))
        cursor = d.get("cursor")
        if not cursor or not d.get(key):
            return out
        time.sleep(0.05)


def parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    markets: dict[str, dict] = {}
    listing = OUT / "markets_2026_ext_raw.jsonl"
    with httpx.Client(timeout=60) as c:
        if listing.exists():  # resume: the listing finished on a previous run
            markets = {m["ticker"]: m for m in map(json.loads, open(listing))}
            print(f"reusing listing: {len(markets)} legs", flush=True)
        for s in ([] if listing.exists() else SERIES):
            events = paged(c, "/events", "events", {"series_ticker": s})
            n0 = len(markets)
            for e in events:
                et = e["event_ticker"]
                for path in ("/historical/markets", "/markets"):
                    for m in paged(c, path, "markets", {"event_ticker": et}):
                        ct = parse(m["close_time"])
                        if LO <= ct < now:
                            markets[m["ticker"]] = m
            print(f"{s:16s} events={len(events):4d}  settled legs since 06-01: {len(markets) - n0}", flush=True)

        if not listing.exists():
            with open(listing, "w") as f:
                for m in markets.values():
                    f.write(json.dumps(m) + "\n")

        ckpt = OUT / "trades_checkpoint.parquet"
        done = pl.read_parquet(ckpt) if ckpt.exists() else None
        seen = set(done["ticker"].unique()) if done is not None else set()
        frames = [done] if done is not None else []
        todo = [t for t in markets if t not in seen]
        buf: list[dict] = []
        for i, t in enumerate(todo, 1):
            rows = paged(c, "/historical/trades", "trades", {"ticker": t})
            rows += paged(c, "/markets/trades", "trades", {"ticker": t})
            buf.extend(rows)
            if i % 50 == 0 or i == len(todo):
                if buf:
                    frames.append(normalize(buf))
                    buf = []
                if frames:
                    pl.concat(frames).write_parquet(ckpt)
                print(f"[{i}/{len(todo)}] tickers fetched", flush=True)

    tr = pl.concat(frames).unique("trade_id", keep="first").sort("ticker", "created_time")
    tr.write_parquet(OUT / "trades_2026_ext.parquet")
    ckpt.unlink(missing_ok=True)
    print(f"\nlegs: {len(markets)}   events: {len({m['event_ticker'] for m in markets.values()})}")
    print(f"trades: {tr.height}   tickers with trades: {tr['ticker'].n_unique()}")
    print(f"created_time: {tr['created_time'].min()} -> {tr['created_time'].max()}")


def normalize(rows: list[dict]) -> pl.DataFrame:
    # Same schema as scripts/fetch_kalshi_data.py:normalize_trades.
    return pl.DataFrame(rows).with_columns(
        pl.col("count_fp").cast(pl.Float64).cast(pl.Int64).alias("count"),
        (pl.col("yes_price_dollars").cast(pl.Float64) * 100).round(0).cast(pl.Int64).alias("yes_price"),
        (pl.col("no_price_dollars").cast(pl.Float64) * 100).round(0).cast(pl.Int64).alias("no_price"),
        pl.col("created_time").str.to_datetime(time_unit="ns", time_zone="UTC"),
    ).select("trade_id", "ticker", "count", "yes_price", "no_price", "taker_side", "created_time")


if __name__ == "__main__":
    main()
