"""Rebuild replay inputs for recorded games whose live runner had no contract list (so no candles, no per-ticker trades).

Hybrid input = recorded feed, feed arrivals and WebSocket book  +  contracts and 1-min candles from replay_hist (built from
public APIs by build_nba_replay.py; candle received_at = minute end + 1 s, the live recorder's measured first-seen lag is ~0 s)
+ trades taken from the recorded book's own trade messages with their real arrival stamps (live trades were WS too).

  b59_backfill_inputs.py control GAME_ID...   rebuild games that DO have real inputs, replay both, compare (fidelity check)
  b59_backfill_inputs.py apply   GAME_ID...   rebuild in place (originals kept as *.orig) for games without contracts
"""
import datetime as dt
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
SL = ROOT / 'data/sports_live'
CONTROL = Path(__file__).with_name('out') / 'b59_control'


def find(day, game, sources, name):
    for s in sources:
        p = SL / s / day / ('NBA_' + game) / name
        if p.exists():
            return p
    return None


def ws_trades(book, tickers):
    seen = {}
    op = gzip.open if str(book).endswith('.gz') else open
    with op(book, 'rt') as f:
        for line in f:
            if '"type":"trade"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get('msg') or {}
            if r.get('type') != 'trade' or m.get('market_ticker') not in tickers:
                continue
            tid = m['trade_id']
            if tid in seen and seen[tid]['received_at'] <= r['received_at']:
                continue
            seen[tid] = {'trade_id': tid, 'ticker': m['market_ticker'], 'taker_side': m['taker_side'],
                         'yes_price_dollars': m.get('yes_price_dollars'), 'no_price_dollars': m.get('no_price_dollars'),
                         'count_fp': m.get('count_fp'), 'is_block_trade': m.get('is_block_trade'),
                         'created_time': dt.datetime.fromtimestamp(m['ts_ms'] / 1000, dt.timezone.utc).isoformat(timespec='milliseconds'),
                         'received_at': r['received_at'], 'via': 'ws'}
    return sorted(seen.values(), key=lambda x: x['received_at'])


def build(day, game, dst, feed_src, book, link):
    """feed_src: directory with game.json, feed/, feed_arrivals.jsonl. dst: where the hybrid is written."""
    hist = SL / 'replay_hist' / day / ('NBA_' + game)
    hg = json.loads((hist / 'game.json').read_text())
    g = json.loads((feed_src / 'game.json').read_text())
    g['contracts'], g['away_k'], g['home_k'] = hg['contracts'], hg['away_k'], hg['home_k']
    tickers = {c['ticker'] for c in g['contracts']}
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('game.json', 'candles.jsonl', 'trades.jsonl'):
        p = dst / name
        if link is False and p.exists() and not (dst / (name + '.orig')).exists():
            shutil.copy2(p, dst / (name + '.orig'))
    (dst / 'game.json').write_text(json.dumps(g, indent=1) + '\n')
    with (dst / 'candles.jsonl').open('w') as out:
        for line in (hist / 'candles.jsonl').open():
            if json.loads(line)['ticker'] in tickers:
                out.write(line)
    rows = ws_trades(book, tickers)
    with (dst / 'trades.jsonl').open('w') as out:
        for r in rows:
            out.write(json.dumps(r, separators=(',', ':')) + '\n')
    if link:
        for name in ('feed', 'feed_arrivals.jsonl'):
            t = dst / name
            if not t.exists():
                t.symlink_to(feed_src / name)
        if not (dst / 'books_ws.jsonl.gz').exists():
            (dst / 'books_ws.jsonl.gz').symlink_to(book)
    (dst / 'backfilled_inputs.json').write_text(json.dumps(dict(
        note='contracts+candles from replay_hist (candle received_at = end+1 s); trades from the recorded WS book; feed, arrivals and book as recorded',
        contracts=len(tickers), trades=len(rows)), indent=1) + '\n')
    return len(tickers), len(rows)


def control(games):
    import b49_event_replay_v2 as BB
    inv = {e['game']: e for e in json.loads((Path(__file__).with_name('out') / 'b55_main_20x/inventory.json').read_text())['entries']}
    BB.OUT = CONTROL / 'results'
    BB.OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for gid in games:
        e = dict(inv[gid])
        day = e['date']
        feed_src = ROOT / e['root']          # the real recorded input: feed, arrivals (its contracts, candles and trades are ignored)
        dst = CONTROL / 'inputs' / day / ('NBA_' + gid)
        book = ROOT / e['book']
        n = build(day, gid, dst, feed_src, book, link=True)
        e['root'] = str(dst.relative_to(ROOT))
        res = BB.run(e, 'control')
        out[gid] = res
        print(gid, 'built', n, flush=True)
    return out


def apply(games):
    for gid in games:
        for day_dir in sorted((SL / 'event_replay_inputs/sandbox').glob(f'2026-10-*/NBA_{gid}')):
            day = day_dir.parent.name
            book = day_dir / 'books_ws.jsonl.gz'
            print(gid, day, build(day, gid, day_dir, day_dir, book, link=False), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent))
    {'control': control, 'apply': apply}[sys.argv[1]](sys.argv[2:])
