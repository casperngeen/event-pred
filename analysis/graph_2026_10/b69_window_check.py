"""Does the baseball 15-60 s skip window block decisions where a newer pitch had really been thrown? Same classifier, two samples.

For each fill of the no-window configuration: last known pitch = last archived pitch end at or before t - 10 s (the model's delay);
age = t - that end; in window = 15 <= age < 60 s; behind = a pitch ended after the last known one and by t (thrown, not yet known).
  backtest  development folds, every-pitch fills without the window (out/thesis_sizing20x/development_MLB.parquet), net c per unit at 20x
  replay    the 10 recorded games, guard-off run (out/b57_noguard), settlement included, net c per contract
"""
import collections
import glob
import gzip
import json
import re
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DELAY = 10.0


def classify(pe, t):
    """pe: sorted pitch end times (epoch s) of the game; t: decision time. -> (in_window, behind)."""
    k = int(np.searchsorted(pe, t - DELAY, side='right'))
    age = t - pe[k - 1] if k else np.inf
    behind = bool(np.searchsorted(pe, t, side='right') > k)
    return (15.0 <= age < 60.0), behind


def table(rows, unit):
    out = collections.OrderedDict()
    for (w, b), (n, q, pnl) in sorted(rows.items(), key=lambda kv: (not kv[0][0], not kv[0][1])):
        out[f"{'in window ' if w else 'outside   '}{'pitch missed ' if b else 'model current'}"] = (n, q, 100 * pnl / q if q else float('nan'))
    tot_n = sum(v[0] for v in rows.values())
    for k, (n, q, c) in out.items():
        print(f"  {k:26s} fills {n:6d} ({100*n/tot_n:4.0f}%)  {unit} {q:9.1f}  net c/unit {c:6.2f}")


# ---- backtest
plays = pl.read_parquet(ROOT / 'data/sports/derived/ingame/MLB_pitch/plays.parquet', columns=['game_id', 't_event']).with_columns(
    (pl.col('t_event').dt.epoch('us') / 1e6).alias('te'))
pe_by = {(g[0] if isinstance(g, tuple) else g): np.sort(d['te'].to_numpy()) for g, d in plays.group_by('game_id')}
x = pl.read_parquet(HERE / 'out/thesis_sizing20x/development_MLB.parquet').filter(
    pl.col('source') == 'every pitch, no skip window', pl.col('hold') == '120s')
rows = collections.defaultdict(lambda: [0, 0., 0.])
for g, t, size, pnl in zip(x['game_id'], x['t'], x['size'], x['pnl']):
    pe = pe_by.get(g)
    if pe is None:
        continue
    r = rows[classify(pe, float(t))]
    r[0] += 1; r[1] += size; r[2] += pnl / 100
print('BACKTEST, development folds (762 games), every-pitch fills without the window, 20x sizing')
table({k: tuple(v) for k, v in rows.items()}, 'quantity')
btot = {k: v for k, v in rows.items()}

# ---- replay
sys.path.insert(0, str(HERE))
import b49_event_replay_v2 as BB
B = BB.B
OUT = HERE / 'out'


def drows(path):
    out = []
    for line in gzip.open(path, 'rt'):
        v = json.loads(line)
        out += v if isinstance(v, list) else [v]
    return [d for d in out if isinstance(d, dict)]


def epoch(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


rrows = collections.defaultdict(lambda: [0, 0., 0.])
for f in sorted(glob.glob(str(OUT / 'b57_noguard/games/*/MLB_*/result.json'))):
    ng = Path(f).parent
    gid = ng.name.split('_')[1]
    inv = next(e for e in json.load(open(OUT / 'b55_main_20x/inventory.json'))['entries'] if e['game'] == gid)
    g, arrivals, doc, final, hs, aws, trades, candles, missing = B.load(inv)
    meta = {c['ticker']: c for c in g['contracts']}
    pe = np.array(sorted(epoch(e['endTime']) for pa in doc['feed']['liveData']['plays']['allPlays'] for e in pa['playEvents'] if e.get('isPitch') and e.get('endTime')))
    orders = [r for r in drows(ng / 'host/decisions.jsonl.gz') if r.get('action') == 'order']
    byp = collections.defaultdict(list)
    for lot in json.load(open(ng / 'host/lots.json')):
        byp[lot['parent']].append(lot)
    for i, o in enumerate(orders):
        key = classify(pe, o['t'])
        for lot in byp.get(i, []):
            gross = sum(x_['size'] * lot['side'] * (x_['price'] - lot['entry']) for x_ in lot['exits'])
            fee = lot['entry_fee_c'] + sum(x_['fee_c'] for x_ in lot['exits'])
            st = lot['remaining'] * lot['side'] * (B.A.payoff(meta[lot['ticker']], g['league'], g['home_k'], g['away_k'], hs, aws) - lot['entry']) if lot['remaining'] > 1e-9 else 0.
            r = rrows[key]
            r[0] += 1; r[1] += lot['size']; r[2] += (gross + st - fee) / 100
print('\nREPLAY, 10 recorded games, guard-off run (window and in-play wait off), settlement included')
table({k: tuple(v) for k, v in rrows.items()}, 'contracts')
