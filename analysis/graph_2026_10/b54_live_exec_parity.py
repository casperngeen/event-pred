"""Parity of the live path (scripts/live_exec.LiveExecution, fed message by message with a receive + flush delay and stepped
once a second) against the batch replay (b49) for the same orders: do the entries fill the same way?

    python b54_live_exec_parity.py <game dir of a b49 run>/NBA_xxx  <books_ws.jsonl[.gz]>  [flush_s]
Orders are the 'submitted' events of the batch run's execution.jsonl.gz (placed at the same decision times)."""
import collections
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import event_execution as X
import live_exec as LE

game = Path(sys.argv[1]);books = Path(sys.argv[2]);flush = float(sys.argv[3]) if len(sys.argv) > 3 else .25
arm = game / 'host'
batch_lots = json.loads((arm / 'lots.json').read_text())
sub = [json.loads(l) for l in gzip.open(arm / 'execution.jsonl.gz', 'rt') if '"submitted"' in l]
orders = sorted(sub, key=lambda e: e['decision_t']);print('orders to place', len(orders))
tickers = {e['ticker'] for e in orders}
meta = json.loads((books.parent / 'game.json').read_text());all_tk = {c['ticker'] for c in meta['contracts']}
events = []
ex = LE.LiveExecution(lambda n, r: events.append(r), None, all_tk, latency=.09)
ex.ref = lambda tk, t: None
op = gzip.open if str(books).endswith('.gz') else open
f = op(books, 'rt');pending = None;wanted = ('orderbook_snapshot', 'orderbook_delta', 'trade')


def nxt():
    for line in f:
        if '"market_ticker"' not in line:continue
        m = json.loads(line)
        if m.get('type') in wanted and (m.get('msg') or {}).get('market_ticker') in all_tk:
            return dt.datetime.fromisoformat(m['received_at'].replace('Z', '+00:00')).timestamp(), m
    return None


pending = nxt();t0 = orders[0]['decision_t'] - 120;wall = int(t0);end = orders[-1]['decision_t'] + 900
oi = 0;skipped = collections.Counter()
wall = int(pending[0] + flush)
while wall <= end:
    while pending and pending[0] + flush <= wall:
        ts, m = pending;ex.feed(X.book_ts(m), m);pending = nxt()
    ex.update(None, float(wall))
    while oi < len(orders) and orders[oi]['decision_t'] <= wall:
        e = orders[oi];oi += 1
        d = dict(t=e['decision_t'], ticker=e['ticker'], kind='spread' if 'SPREAD' in e['ticker'] else 'total', side=e['side'], limit=e['wanted_limit'], size=e['size'], signal=0.)
        if ex.is_busy(e['ticker']):skipped['busy_in_live_path'] += 1;continue
        ex.place(d)
    wall += 1
ex.update(None, float(end + 5))


def key(l):return (l['ticker'], l['side'], round(l['entry']), l['kind'])
b = collections.Counter();v = collections.Counter()
for l in batch_lots:b[key(l)] += l['size']
for l in ex.lots:v[key(l)] += l['size']
print('batch lots', len(batch_lots), 'live-path lots', len(ex.lots), 'skipped', dict(skipped))
bad = 0
for k in sorted(set(b) | set(v)):
    if abs(b[k] - v[k]) > 1e-6:bad += 1;print('DIFF', k, 'batch', round(b[k], 2), 'live', round(v[k], 2))
print('entry (ticker,side,price) groups', len(set(b) | set(v)), 'differing', bad)
tb = {(l['ticker'], round(l['entry']), l['side']): l['t_fill'] for l in batch_lots};dts = []
for l in ex.lots:
    k = (l['ticker'], round(l['entry']), l['side'])
    if k in tb:dts.append(l['t_fill'] - tb[k])
print('fill time live-batch (s): n', len(dts), 'median', sorted(dts)[len(dts) // 2] if dts else None, 'max abs', max(map(abs, dts)) if dts else None)
print('stats', {k: v for k, v in ex.stats.items() if k in ('late_messages', 'negative_level', 'trades_without_healthy_book', 'snapshots')})
