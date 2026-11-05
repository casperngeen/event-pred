"""Compact a night's WebSocket order-book file (books_ws.jsonl[.gz], GBs) into the events a queue simulation needs, for
the spread / total contracts only: one csv.gz row per snapshot level (S), level-size change (D) or trade (T).

    ts_ms,kind,ticker,side,price_c,qty
    S/D: side = yes|no (the bid side the level is on), price_c = that side's bid price in cents, qty = the level's size
         (S) or the change in it (D).   T: side = the taker's side (yes|no), price_c = the yes price in cents, qty = count.

    python3 scripts/extract_book_events.py <books_ws.jsonl.gz> <out.csv.gz>
"""
import csv
import gzip
import json
import sys
from datetime import datetime


def ms(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def c(x):
    return round(float(x) * 100, 2)


def main(src, dst):
    op = gzip.open if src.endswith(".gz") else open
    n = 0
    with op(src, "rt") as f, gzip.open(dst, "wt", newline="") as g:
        w = csv.writer(g)
        w.writerow(["ts_ms", "kind", "ticker", "side", "price_c", "qty"])
        for line in f:
            if '"KXNBAGAME' in line and '"type":"orderbook' in line:
                continue
            try:
                m = json.loads(line)
            except ValueError:
                continue
            t, d = m.get("type"), m.get("msg") or {}
            tk = d.get("market_ticker", "")
            if "SPREAD" not in tk and "TOTAL" not in tk:
                continue
            if t == "orderbook_snapshot":
                t0 = ms(m["received_at"])
                for side in ("yes", "no"):
                    for p, q in d.get(f"{side}_dollars_fp") or []:
                        w.writerow([t0, "S", tk, side, c(p), float(q)])
            elif t == "orderbook_delta":
                w.writerow([d["ts_ms"], "D", tk, d["side"], c(d["price_dollars"]), float(d["delta_fp"])])
            elif t == "trade":
                w.writerow([d["ts_ms"], "T", tk, d["taker_side"], c(d["yes_price_dollars"]), float(d["count_fp"])])
            n += 1
    print(src, n, "rows", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
