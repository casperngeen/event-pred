"""Why did the baseball guard (skip 15-60 s after the last known pitch; wait while a ball is in play) cost money in replay?

For every order of the guard-off run: which main-run skip reason covered that decision (window / ball in play / none), whether a
pitch had really been thrown since the last pitch the model knew (from the final feed's pitch end times), and the settled-
inclusive net of the order's fills. Reads out/b55_main_20x and out/b57_noguard; writes out/b65_mlb_guard.json.
"""
import collections
import datetime as dt
import glob
import gzip
import json
import re
import sys

import numpy as np

import b49_event_replay_v2 as BB

B = BB.B
OUT = BB.Path(__file__).with_name('out')


def rows(path):
    out = []
    for line in gzip.open(path, 'rt'):
        x = json.loads(line)
        out += x if isinstance(x, list) else [x]
    return [d for d in out if isinstance(d, dict)]


def epoch(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def pitch_ends(final):
    ends = []
    for pa in final['feed']['liveData']['plays']['allPlays']:
        for ev in pa['playEvents']:
            if ev.get('isPitch') and ev.get('endTime'):
                ends.append(epoch(ev['endTime']))
    return np.array(sorted(ends))


res = collections.defaultdict(lambda: dict(orders=0, lots=0, contracts=0., net=0., fees=0., settled=0., settle_net=0.))
split = collections.defaultdict(lambda: dict(orders=0, contracts=0., net=0.))
for f in sorted(glob.glob(str(OUT / 'b57_noguard/games/*/MLB_*/result.json'))):
    ng = BB.Path(f).parent
    gid = ng.name.split('_')[1]
    mainp = next(OUT.glob(f'b55_main_20x/games/*/MLB_{gid}/host/decisions.jsonl.gz'))
    skip = {}
    for r in rows(mainp):
        a = r.get('action', '')
        if a.startswith('skip') and 'ticker' in r:
            m = re.search(r'(\d+) s since the last known pitch', a)
            skip[(r['ticker'], round(r['t'], 1))] = ('in play', None) if 'ball in play' in a else ('window', float(m.group(1))) if m else ('other skip', None)
    inv = next(e for e in json.load(open(OUT / 'b55_main_20x/inventory.json'))['entries'] if e['game'] == gid)
    g, arrivals, doc, final, hs, aws, trades, candles, missing = B.load(inv)
    meta = {c['ticker']: c for c in g['contracts']}
    ends = pitch_ends(doc)
    orders = [r for r in rows(ng / 'host/decisions.jsonl.gz') if r.get('action') == 'order']
    byp = collections.defaultdict(list)
    for lot in json.load(open(ng / 'host/lots.json')):
        byp[lot['parent']].append(lot)
    for i, o in enumerate(orders):
        reason, age = skip.get((o['ticker'], round(o['t'], 1)), ('not skipped in main', None))
        klass = reason
        if reason == 'window':
            last_known = o['t'] - age
            behind = int(((ends > last_known + 1) & (ends <= o['t'] - 10)).sum() > 0)   # a pitch ended after the last known one and the model's 10 s delay has passed
            klass = 'window, pitch missed' if behind else 'window, model up to date'
        for lot in byp.get(i, []):
            gross = sum(x['size'] * lot['side'] * (x['price'] - lot['entry']) for x in lot['exits'])
            fee = lot['entry_fee_c'] + sum(x['fee_c'] for x in lot['exits'])
            st = 0.
            if lot['remaining'] > 1e-9:
                st = lot['remaining'] * lot['side'] * (B.A.payoff(meta[lot['ticker']], g['league'], g['home_k'], g['away_k'], hs, aws) - lot['entry'])
            d = res[klass]
            d['lots'] += 1; d['contracts'] += lot['size']; d['net'] += gross + st - fee; d['fees'] += fee
            d['settled'] += lot['remaining']; d['settle_net'] += st
res = {k: {a: round(b, 2) for a, b in v.items()} for k, v in res.items()}
for k, v in sorted(res.items()):
    v['net_c_per_contract'] = round(v['net'] / v['contracts'], 2) if v['contracts'] else None
    print(f"{k:28s} lots {v['lots']:4.0f}  contracts {v['contracts']:7.1f}  net ${v['net']/100:7.2f}  c/contract {v['net_c_per_contract']}  settled {v['settled']:.1f} contracts (${v['settle_net']/100:.2f})")
(OUT / 'b65_mlb_guard.json').write_text(json.dumps(res, indent=1))
