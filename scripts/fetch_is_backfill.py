"""Backfill in-sample trades the archive never fetched, on the macro series.

``data/trades/`` was assembled per ticker from a list that was never complete
(``research_log.md`` §12), so some legs that traded on Kalshi have no local
trades, or fewer contracts locally than Kalshi's recorded volume. This pulls the
full trade history of every such leg on the registered macro series, except WTI
and WTIW (not used as triggers; excluded on request 2026-09-28), and the equity
index targets (INXU / INXD / NASDAQ100U: ~16k legs, used by no current study).

Candidates come from two listings, so an event missing from ``data/markets/``
is still found:
  1. ``data/markets/`` (latest fetch per ticker), series canonicalised through
     the registry's alias table;
  2. Kalshi's own event list per series (``/events``), with each event's legs
     from ``/historical/markets`` + ``/markets``.
A leg is fetched when it closed before ``OOS_START``, Kalshi records volume > 0,
and the archive holds fewer contracts than that volume.

Writes
  data/backfill_2026_09/markets_listing.jsonl     the API leg records (provenance)
  data/trades/trades_backfill_2026_09.parquet    new trade_ids only, archive schema
  data/markets/markets_backfill_2026_09.parquet  metadata for listed legs absent from
                                                 data/markets/ (their events are
                                                 otherwise invisible to load_markets)

``--markets-only`` writes the last file from an existing listing, without refetching.

In-sample only: every leg closes before ``OOS_START`` and every trade is
asserted to be created before it. Prints counts only.

    venv/bin/python scripts/fetch_is_backfill.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import httpx
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stg_infra"))
from stg.panel.registry import SPECS, canonical  # noqa: E402
from stg.splits import OOS_START, assert_no_oos  # noqa: E402

B = "https://api.elections.kalshi.com/trade-api/v2"
OUT_DIR = ROOT / "data" / "backfill_2026_09"
OUT_TRADES = ROOT / "data" / "trades" / "trades_backfill_2026_09.parquet"
OUT_MARKETS = ROOT / "data" / "markets" / "markets_backfill_2026_09.parquet"
SKIP = {"WTI", "WTIW", "INXU", "INXD", "NASDAQ100U"}
CANON = [c for c in SPECS if c not in SKIP]
WALL = OOS_START.replace(tzinfo=dt.timezone.utc) if OOS_START.tzinfo is None else OOS_START
# series tickers to ask /events for: every raw prefix of each canon, with and
# without the KX prefix (legacy events are listed under either)
RAW = {c: {c, *SPECS[c].aliases} for c in CANON}
RAW["CPISHELTER"] |= {"SHELTERCPI"}           # relaunched as KXSHELTERCPI
API_SERIES = sorted({p + r for rs in RAW.values() for r in rs for p in ("", "KX")})


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
        if r.status_code == 404:
            return {}
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


def canon_of(ticker: str) -> str:
    return canonical(ticker.removeprefix("KX").split("-")[0])


def vol(m: dict) -> float:
    return float(m.get("volume_fp") or m.get("volume") or 0)


def normalize(rows: list[dict], fetched_at: dt.datetime) -> pl.DataFrame:
    # Same schema as data/trades/*.parquet (scripts/fetch_kalshi_data.py:normalize_trades + _fetched_at).
    return pl.DataFrame(rows).with_columns(
        pl.col("count_fp").cast(pl.Float64).cast(pl.Int64).alias("count"),
        (pl.col("yes_price_dollars").cast(pl.Float64) * 100).round(0).cast(pl.Int64).alias("yes_price"),
        (pl.col("no_price_dollars").cast(pl.Float64) * 100).round(0).cast(pl.Int64).alias("no_price"),
        pl.col("created_time").str.to_datetime(time_unit="ns", time_zone="UTC"),
        pl.lit(fetched_at).cast(pl.Datetime("ns")).alias("_fetched_at"),
    ).select("trade_id", "ticker", "count", "yes_price", "no_price", "taker_side", "created_time", "_fetched_at")


def cents(x) -> int | None:
    return None if x in (None, "") else int(round(float(x) * 100))


def write_markets(api: dict[str, dict], fetched_at: dt.datetime) -> None:
    """Metadata rows, in the data/markets/ schema, for API legs data/markets/ lacks."""
    have = set(pl.scan_parquet(str(ROOT / "data/markets/*.parquet")).select("ticker")
               .filter(pl.col("ticker").is_in(list(api))).collect()["ticker"].to_list())
    miss = [m for t, m in api.items() if t not in have]
    if not miss:
        print("markets: every listed leg is already in data/markets/")
        return
    ts = pl.Datetime("ns", "UTC")
    df = pl.DataFrame({
        "ticker": [m["ticker"] for m in miss], "event_ticker": [m["event_ticker"] for m in miss],
        "market_type": [m.get("market_type") for m in miss], "title": [m.get("title") for m in miss],
        "yes_sub_title": [m.get("yes_sub_title") for m in miss], "no_sub_title": [m.get("no_sub_title") for m in miss],
        "status": [m.get("status") for m in miss],
        **{k: [cents(m.get(f"{k}_dollars")) for m in miss] for k in ("yes_bid", "yes_ask", "no_bid", "no_ask", "last_price")},
        "volume": [int(vol(m)) for m in miss],
        "volume_24h": [int(float(m.get("volume_24h_fp") or 0)) for m in miss],
        "open_interest": [int(float(m.get("open_interest_fp") or 0)) for m in miss],
        "result": [m.get("result") or "" for m in miss],
        **{k: [m.get(k) for m in miss] for k in ("created_time", "open_time", "close_time")},
    }, schema_overrides={k: pl.Int64 for k in ("yes_bid", "yes_ask", "no_bid", "no_ask", "last_price")}).with_columns(
        *(pl.col(k).str.to_datetime(time_zone="UTC").cast(ts) for k in ("created_time", "open_time", "close_time")),
        pl.lit(fetched_at).cast(pl.Datetime("ns")).alias("_fetched_at"),
    )
    assert_no_oos(df, time_col="close_time")
    df.write_parquet(OUT_MARKETS)
    print(f"markets: wrote {df.height} legs on {df['event_ticker'].n_unique()} events to "
          f"{OUT_MARKETS.relative_to(ROOT)}: {sorted(df['event_ticker'].unique().to_list())}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--markets-only" in sys.argv:
        api = {m["ticker"]: m for m in map(json.loads, open(OUT_DIR / "markets_listing.jsonl"))}
        write_markets(api, dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
        return
    if OUT_TRADES.exists():
        raise SystemExit(f"{OUT_TRADES} exists; move it aside to refetch (the local-contract "
                         "comparison would otherwise count it as already present)")
    fetched_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    # 1. local listing and local contracts per ticker
    local_mk = (pl.scan_parquet(str(ROOT / "data/markets/*.parquet"))
                .select("ticker", "event_ticker", "volume", "close_time", "_fetched_at")
                .sort("_fetched_at", descending=True).unique("ticker", keep="first")
                .filter(pl.col("close_time") < WALL).collect())
    local_mk = local_mk.filter(pl.col("ticker").map_elements(canon_of, return_dtype=pl.String).is_in(CANON))
    local_n = dict(pl.scan_parquet(str(ROOT / "data/trades/*.parquet"))
                   .group_by("ticker").agg(pl.col("count").sum()).collect().iter_rows())
    print(f"series: {len(CANON)} ({', '.join(CANON)})")
    print(f"local listing: {local_mk.height} in-sample legs on these series")

    with httpx.Client(timeout=60) as c:
        # 2. Kalshi's own listing
        api: dict[str, dict] = {}
        for s in API_SERIES:
            events = paged(c, "/events", "events", {"series_ticker": s})
            for e in events:
                for path in ("/historical/markets", "/markets"):
                    for m in paged(c, path, "markets", {"event_ticker": e["event_ticker"]}):
                        if m["close_time"] < WALL.strftime("%Y-%m-%dT%H:%M:%SZ") and canon_of(m["ticker"]) in CANON:
                            api[m["ticker"]] = m
        with open(OUT_DIR / "markets_listing.jsonl", "w") as f:
            for m in api.values():
                f.write(json.dumps(m) + "\n")
        print(f"API listing: {len(api)} in-sample legs "
              f"({len(set(api) - set(local_mk['ticker']))} not in the local listing)")

        # 3. candidates: volume on Kalshi, fewer contracts locally
        volume = {t: float(v or 0) for t, v in local_mk.select("ticker", "volume").iter_rows()}
        volume |= {t: vol(m) for t, m in api.items()}          # API volume wins where both exist
        todo = sorted(t for t, v in volume.items() if v > 0 and local_n.get(t, 0) < v)
        print(f"legs to fetch: {len(todo)} "
              f"({sum(local_n.get(t, 0) == 0 for t in todo)} with no local trades, "
              f"{sum(local_n.get(t, 0) > 0 for t in todo)} partial)")

        # 4. fetch
        rows: list[dict] = []
        for i, t in enumerate(todo, 1):
            rows += paged(c, "/historical/trades", "trades", {"ticker": t})
            rows += paged(c, "/markets/trades", "trades", {"ticker": t})
            if i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] legs fetched", flush=True)

    if not rows:
        print("nothing to add")
        return
    tr = normalize(rows, fetched_at).unique("trade_id", keep="first")
    have = (pl.scan_parquet(str(ROOT / "data/trades/*.parquet")).select("trade_id")
            .filter(pl.col("trade_id").is_in(tr["trade_id"].implode())).collect()["trade_id"])
    new = tr.filter(~pl.col("trade_id").is_in(have.implode())).sort("ticker", "created_time")
    assert_no_oos(new, time_col="created_time")
    new.write_parquet(OUT_TRADES)
    write_markets(api, fetched_at)

    # 5. check: local + new contracts now match Kalshi's volume on every fetched leg
    got = dict(tr.group_by("ticker").agg(pl.col("count").sum()).iter_rows())
    short = [t for t in todo if got.get(t, 0) < volume[t]]
    by = (new.with_columns(pl.col("ticker").map_elements(canon_of, return_dtype=pl.String).alias("canon"))
          .group_by("canon").agg(pl.col("ticker").n_unique().alias("legs"), pl.len().alias("trades"),
                                 pl.col("count").sum().alias("contracts"),
                                 pl.col("ticker").str.split("-").list.slice(0, 2).list.join("-")
                                 .n_unique().alias("events"))
          .sort("trades", descending=True))
    print(f"\nfetched {tr.height} trades; {tr.height - new.height} already in the archive; "
          f"wrote {new.height} new to {OUT_TRADES.relative_to(ROOT)}")
    print(f"created_time: {new['created_time'].min()} -> {new['created_time'].max()}")
    print(f"legs where the API returned fewer contracts than Kalshi's volume: {len(short)}")
    with pl.Config(tbl_rows=40):
        print(by)


if __name__ == "__main__":
    main()
